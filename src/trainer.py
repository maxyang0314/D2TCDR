import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import random
import pandas as pd

from collections import Counter

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel

def build_item_popularity_groups(
    train_data,
    popular_ratio=0.2
):
    """
    使用 target-domain training set 建立三種 item 集合。

    Popular：
        training data 中出現次數最高的前 popular_ratio。

    Unpopular-seen：
        training data 中出現過，但不屬於 Popular。

    Unseen：
        不在 observed_train_items 中。
        Unseen 集合本身會在測試時依 target 判斷，
        因為無法只靠 training data 預先列出所有測試 unseen item。

    Popularity 統計包含：
        1. training seq 中的 item
        2. training next 中的 item

    item 0 是 padding，不納入統計。
    """
    if not 0 < popular_ratio < 1:
        raise ValueError(
            "popular_ratio must be between 0 and 1, "
            f"got {popular_ratio}"
        )

    item_counter = Counter()

    # 1. 統計 training sequence 中出現的 item
    for seq in train_data['seq']:
        item_counter.update(
            int(item)
            for item in seq
            if int(item) != 0
        )

    # 2. 統計 training target 中出現的 item
    item_counter.update(
        int(item)
        for item in train_data['next']
        if int(item) != 0
    )

    if len(item_counter) == 0:
        raise ValueError(
            "No valid items found in training data."
        )

    # 出現次數由高至低；
    # 次數相同時用 item ID 排序，確保可重現
    sorted_items = sorted(
        item_counter.keys(),
        key=lambda item: (
            -item_counter[item],
            item
        )
    )

    num_popular = max(
        1,
        int(
            np.ceil(
                len(sorted_items) * popular_ratio
            )
        )
    )

    # training data 中出現過的全部 item
    observed_train_items = set(sorted_items)

    # observed items 中最熱門的前 20%
    popular_items = set(
        sorted_items[:num_popular]
    )

    # training 中出現過，但不屬於 popular
    unpopular_seen_items = (
        observed_train_items - popular_items
    )

    return (
        popular_items,
        unpopular_seen_items,
        observed_train_items,
        item_counter
    )


def per_sample_metrics(scores, labels, ks):
    """
    回傳每一筆測試資料各自的 HR@K 與 NDCG@K，
    讓後續可以依 ground-truth item 分組。

    scores: [batch_size, item_num]
    labels: [batch_size, 1]
    """
    labels = labels.view(-1)

    topk_indices = torch.topk(
        scores,
        k=max(ks),
        dim=-1
    ).indices

    # [batch_size, max_k]
    hit = topk_indices.eq(labels.unsqueeze(1))

    metrics = {}

    for k in ks:
        hit_k = hit[:, :k]

        # 一筆資料只要在 Top-K 命中，HR 就是 1
        metrics[f'HR@{k}'] = (
            hit_k.any(dim=1)
            .float()
            .cpu()
        )

        # rank 1 的 discount = 1/log2(2) = 1
        discounts = 1.0 / torch.log2(
            torch.arange(
                2,
                k + 2,
                device=scores.device,
                dtype=torch.float32
            )
        )

        metrics[f'NDCG@{k}'] = (
            hit_k.float() * discounts.unsqueeze(0)
        ).sum(dim=1).cpu()

    return metrics, topk_indices


def init_metric_sums(ks):
    """初始化指標加總值。"""
    metric_sums = {}

    for k in ks:
        metric_sums[f'HR@{k}'] = 0.0
        metric_sums[f'NDCG@{k}'] = 0.0

    return metric_sums


def finalize_metric_sums(metric_sums, sample_count):
    """
    將逐筆指標加總除以樣本數，並轉成百分比。
    """
    if sample_count == 0:
        return {
            key: None
            for key in metric_sums
        }

    return {
        key: round(value / sample_count * 100, 4)
        for key, value in metric_sums.items()
    }

def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  

def pad_or_truncate_sequence(seq, max_len):
    """
    移除 padding 0，保留最近 max_len 個 item，
    最後從左側重新補 0。
    """
    real_items = [
        int(item)
        for item in seq
        if int(item) != 0
    ]

    real_items = real_items[-max_len:]

    return (
        [0] * (max_len - len(real_items))
        + real_items
    )

def augment_tail_sequence(
    seq,
    max_len,
    drop_probability,
    preserve_recent,
    rng
):
    """
    對 Unpopular-seen target 的歷史 sequence
    做 context dropout。

    規則：
    1. 只刪除較早的歷史 item
    2. 最近 preserve_recent 個 item 固定保留
    3. 不打亂 item 順序
    4. 不修改 ground-truth next
    """
    if not 0 <= drop_probability <= 1:
        raise ValueError(
            'tail_drop_probability must be between 0 and 1.'
        )

    if preserve_recent < 1:
        raise ValueError(
            'tail_preserve_recent must be at least 1.'
        )

    real_items = [
        int(item)
        for item in seq
        if int(item) != 0
    ]

    # 序列太短，沒有較早 item 可以刪除
    if len(real_items) <= preserve_recent:
        return pad_or_truncate_sequence(
            real_items,
            max_len
        )

    older_items = real_items[:-preserve_recent]
    recent_items = real_items[-preserve_recent:]

    # 只對較舊的 item 做 dropout
    kept_older_items = []
    dropped_any_item = False

    for item in older_items:
        if rng.random() < drop_probability:
            dropped_any_item = True
        else:
            kept_older_items.append(item)

    # 若這次剛好一個都沒刪，強制隨機刪除一個較舊 item
    if not dropped_any_item and len(older_items) > 0:
        drop_index = rng.randrange(
            len(older_items)
        )

        kept_older_items = (
            older_items[:drop_index]
            + older_items[drop_index + 1:]
        )

    augmented_items = (
        kept_older_items
        + recent_items
    )

    return pad_or_truncate_sequence(
        augmented_items,
        max_len
    )

def build_doubled_unpopular_seen_train_data(
    train_data,
    unpopular_seen_items,
    max_len,
    drop_probability,
    preserve_recent,
    random_seed
):
    """
    保留所有原始 training rows。

    對每一筆 next 屬於 Unpopular-seen 的資料，
    額外產生一筆 context-dropout copy。

    Popular：
        只保留原始資料。

    Unpopular-seen：
        一筆原始資料 + 一筆 augmented 資料。
    """
    original_data = (
        train_data
        .copy()
        .reset_index(drop=True)
    )

    # 用來辨認原始資料與 augmented copy
    original_data['is_augmented'] = 0

    rng = random.Random(random_seed)

    augmented_rows = []
    unchanged_augmented_count = 0

    for _, row in original_data.iterrows():
        target_item = int(row['next'])

        # 只對 Unpopular-seen target 新增 copy
        if target_item not in unpopular_seen_items:
            continue

        augmented_row = row.copy()

        original_seq = list(row['seq'])

        normalized_original_seq = (
            pad_or_truncate_sequence(
                original_seq,
                max_len
            )
        )

        augmented_seq = augment_tail_sequence(
            seq=original_seq,
            max_len=max_len,
            drop_probability=drop_probability,
            preserve_recent=preserve_recent,
            rng=rng
        )

        if augmented_seq == normalized_original_seq:
            unchanged_augmented_count += 1

        # 只修改 sequence
        augmented_row['seq'] = augmented_seq

        # ground-truth next 維持原樣
        augmented_row['next'] = target_item

        if 'len_seq' in original_data.columns:
            augmented_row['len_seq'] = sum(
                int(item) != 0
                for item in augmented_seq
            )

        augmented_row['is_augmented'] = 1

        augmented_rows.append(
            augmented_row
        )

    augmented_data = pd.DataFrame(
        augmented_rows,
        columns=original_data.columns
    )

    doubled_train_data = pd.concat(
        [
            original_data,
            augmented_data
        ],
        ignore_index=True
    )

    # 打亂原始與 augmented rows
    doubled_train_data = (
        doubled_train_data
        .sample(
            frac=1,
            random_state=random_seed
        )
        .reset_index(drop=True)
    )

    original_tail_count = (
        original_data['next']
        .astype(int)
        .isin(unpopular_seen_items)
        .sum()
    )

    final_tail_count = (
        doubled_train_data['next']
        .astype(int)
        .isin(unpopular_seen_items)
        .sum()
    )

    stats = {
        'original_training_samples': (
            len(original_data)
        ),
        'original_tail_samples': int(
            original_tail_count
        ),
        'augmented_tail_copies': (
            len(augmented_data)
        ),
        'final_training_samples': (
            len(doubled_train_data)
        ),
        'final_tail_samples': int(
            final_tail_count
        ),
        'original_tail_ratio': round(
            original_tail_count
            / len(original_data),
            4
        ),
        'final_tail_ratio': round(
            final_tail_count
            / len(doubled_train_data),
            4
        ),
        'unchanged_augmented_sequences': (
            unchanged_augmented_count
        )
    }

    return doubled_train_data, stats

def model_train(train_data, val_data, test_data, con_data, model_joint, args, logger, pretrain_flag):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0

    # =========================================================
    # Source/Target recommendation training data
    #
    # 第一階段：
    #   train_data 是 source-domain training data
    #
    # 第二階段：
    #   train_data 是 target-domain training data
    # =========================================================
    training_data_for_sampling = train_data

    # =========================================================
    # 第一階段使用的 Target condition data
    #
    # pretrain_flag=True 時：
    #   con_data 是原始 target-domain training data
    # =========================================================
    condition_data_for_sampling = con_data

    # 第二階段使用的變數
    unpopular_seen_items_for_augmentation = None
    augmentation_stats = None

    # 第一階段 Target condition augmentation 使用的變數
    condition_popular_items = None
    condition_unpopular_seen_items = None
    condition_augmentation_stats = None
    # Stage 1 Target-Tail auxiliary recommendation data
    target_tail_aux_data = None

    # =========================================================
    # 第一階段：
    # 建立 augmented Target condition dataset
    # =========================================================
    if pretrain_flag:

        if con_data is None:
            raise ValueError(
                'con_data cannot be None during source pretraining.'
            )

        # Popularity 必須由原始 Target training data 定義
        (
            condition_popular_items,
            condition_unpopular_seen_items,
            condition_observed_items,
            condition_item_counter
        ) = build_item_popularity_groups(
            train_data=con_data,
            popular_ratio=args.popular_ratio
        )

        # 保留全部原始 Target rows，
        # 並為每筆 Unpopular-seen-target row
        # 額外建立一筆 context-dropout copy
        (
            condition_data_for_sampling,
            condition_augmentation_stats
        ) = build_doubled_unpopular_seen_train_data(
            train_data=con_data,
            unpopular_seen_items=(
                condition_unpopular_seen_items
            ),
            max_len=args.max_len,
            drop_probability=(
                args.tail_drop_probability
            ),
            preserve_recent=(
                args.tail_preserve_recent
            ),
            random_seed=(
                args.random_seed + 200000
            )
        )

        # =========================================================
        # Stage 1 Target-Tail auxiliary recommendation pool
        #
        # condition_data_for_sampling 已包含：
        # 1. 原始 Popular rows
        # 2. 原始 Tail rows
        # 3. augmented Tail rows
        #
        # 這裡只保留 next 為 Unpopular-seen 的 rows。
        # =========================================================
        target_tail_aux_data = (
            condition_data_for_sampling[
                condition_data_for_sampling[
                    'next'
                ]
                .astype(int)
                .isin(
                    condition_unpopular_seen_items
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        target_tail_original_count = int(
            (
                target_tail_aux_data[
                    'is_augmented'
                ] == 0
            ).sum()
        )

        target_tail_augmented_count = int(
            (
                target_tail_aux_data[
                    'is_augmented'
                ] == 1
            ).sum()
        )

        target_tail_aux_setup = {
            'training_stage': (
                'source-domain pretraining'
            ),
            'auxiliary_task': (
                'Target Unpopular-seen next-item prediction'
            ),
            'target_tail_aux_samples': len(
                target_tail_aux_data
            ),
            'original_tail_samples': (
                target_tail_original_count
            ),
            'augmented_tail_samples': (
                target_tail_augmented_count
            ),
            'target_tail_aux_batch_size': (
                args.target_tail_aux_batch_size
            ),
            'target_tail_aux_weight': (
                args.target_tail_aux_weight
            )
        }

        print(
            'First-Stage Target-Tail Auxiliary Setup'
            '------------------------------------------'
        )
        print(target_tail_aux_setup)

        logger.info(
            'First-Stage Target-Tail Auxiliary Setup'
            '------------------------------------------'
        )
        logger.info(target_tail_aux_setup)

        condition_setup = {
            'training_stage': (
                'source-domain pretraining'
            ),
            'augmentation_target': (
                'target-domain condition data'
            ),
            'augmentation_method': (
                'one original plus one augmented copy'
            ),
            'popular_items': len(
                condition_popular_items
            ),
            'unpopular_seen_items': len(
                condition_unpopular_seen_items
            ),
            'tail_drop_probability': (
                args.tail_drop_probability
            ),
            'tail_preserve_recent': (
                args.tail_preserve_recent
            ),
            **condition_augmentation_stats
        }

        print(
            'First-Stage Target Condition Augmentation Setup'
            '------------------------------------------'
        )
        print(condition_setup)

        logger.info(
            'First-Stage Target Condition Augmentation Setup'
            '------------------------------------------'
        )
        logger.info(condition_setup)


    # =========================================================
    # 第二階段：
    # 保留你目前 experiment-2.1 的 Target dataset augmentation
    # =========================================================
    else:

        (
            popular_items_for_augmentation,
            unpopular_seen_items_for_augmentation,
            observed_train_items_for_augmentation,
            item_counter_for_augmentation
        ) = build_item_popularity_groups(
            train_data=train_data,
            popular_ratio=args.popular_ratio
        )

        (
            training_data_for_sampling,
            augmentation_stats
        ) = build_doubled_unpopular_seen_train_data(
            train_data=train_data,
            unpopular_seen_items=(
                unpopular_seen_items_for_augmentation
            ),
            max_len=args.max_len,
            drop_probability=(
                args.tail_drop_probability
            ),
            preserve_recent=(
                args.tail_preserve_recent
            ),
            random_seed=args.random_seed
        )

        augmentation_setup = {
            'training_stage': (
                'target-domain fine-tuning'
            ),
            'augmentation_method': (
                'one original plus one augmented copy'
            ),
            'popular_items': len(
                popular_items_for_augmentation
            ),
            'unpopular_seen_items': len(
                unpopular_seen_items_for_augmentation
            ),
            'tail_drop_probability': (
                args.tail_drop_probability
            ),
            'tail_preserve_recent': (
                args.tail_preserve_recent
            ),
            **augmentation_stats
        }

        print(
            'Second-Stage Tail Dataset Augmentation Setup'
            '------------------------------------------'
        )
        print(augmentation_setup)

        logger.info(
            'Second-Stage Tail Dataset Augmentation Setup'
            '------------------------------------------'
        )
        logger.info(augmentation_setup)


    # Recommendation training 的 batch 數
    #
    # 第一階段依舊由 Source train_data 決定；
    # 第二階段則由 augmented Target train_data 決定。
    num_rows = training_data_for_sampling.shape[0]

    num_batches = int(
        num_rows / args.batch_size
    )

    for epoch_temp in range(epochs):

        model_joint.train()
        flag_update = 0
        for j in range(num_batches):
            batch_seed = (
                args.random_seed
                + epoch_temp * num_batches
                + j
            )

            batch_df = (
                training_data_for_sampling
                .sample(
                    n=args.batch_size,
                    random_state=batch_seed
                )
                .copy()
            )

            if (
                not pretrain_flag
                and epoch_temp == 0
                and j < 3
            ):
                batch_tail_count = (
                    batch_df['next']
                    .astype(int)
                    .isin(
                        unpopular_seen_items_for_augmentation
                    )
                    .sum()
                )

                batch_augmented_count = (
                    batch_df['is_augmented']
                    .sum()
                )

                batch_check = {
                    'batch_size': len(batch_df),
                    'tail_samples': int(
                        batch_tail_count
                    ),
                    'augmented_samples': int(
                        batch_augmented_count
                    ),
                    'tail_ratio': round(
                        batch_tail_count
                        / len(batch_df),
                        4
                    ),
                    'augmented_ratio': round(
                        batch_augmented_count
                        / len(batch_df),
                        4
                    )
                }

                print(
                    'Tail Dataset Batch Check',
                    batch_check
                )
            batch = batch_df.to_dict()

            seq = list(
                batch['seq'].values()
            )

            target = list(
                batch['next'].values()
            )

            # 暫時印出第一個 epoch 的前三個 batch
            if (
                not pretrain_flag
                and epoch_temp == 0
                and j < 3
            ):
                print(
                    'Tail Augmentation Batch Check',
                    augmentation_stats
                )

            batch = batch_df.to_dict()

            seq = list(
                batch['seq'].values()
            )

            target = list(
                batch['next'].values()
            )
            optimizer.zero_grad()
            seq = torch.LongTensor(seq)
            target = (torch.LongTensor(target)).unsqueeze(1)
            seq = seq.to(device)
            target = target.to(device)
            if pretrain_flag:

                # 與 Source batch seed 分開，
                # 避免兩個 DataFrame 使用相同抽樣序列
                con_batch_seed = (
                    args.random_seed
                    + 1000000
                    + epoch_temp * num_batches
                    + j
                )

                con_batch_df = (
                    condition_data_for_sampling
                    .sample(
                        n=args.batch_size,
                        random_state=con_batch_seed
                    )
                    .copy()
                )

                # 第一個 epoch 的前三個 condition (
                if (
                    epoch_temp == 0
                    and j < 3
                ):
                    condition_tail_count = (
                        con_batch_df['next']
                        .astype(int)
                        .isin(
                            condition_unpopular_seen_items
                        )
                        .sum()
                    )

                    condition_augmented_count = int(
                        con_batch_df['is_augmented']
                        .sum()
                    )

                    condition_batch_check = {
                        'batch_size': len(
                            con_batch_df
                        ),
                        'tail_condition_samples': int(
                            condition_tail_count
                        ),
                        'augmented_condition_samples': (
                            condition_augmented_count
                        ),
                        'tail_condition_ratio': round(
                            condition_tail_count
                            / len(con_batch_df),
                            4
                        ),
                        'augmented_condition_ratio': round(
                            condition_augmented_count
                            / len(con_batch_df),
                            4
                        )
                    }

                    print(
                        'First-Stage Target Condition Batch Check',
                        condition_batch_check
                    )

                    logger.info(
                        'First-Stage Target Condition Batch Check %s',
                        condition_batch_check
                    )

                con_batch = con_batch_df.to_dict()

                con_seq = list(
                    con_batch['seq'].values()
                )

                con_seq = torch.LongTensor(
                    con_seq
                )

                con_seq = con_seq.to(device)
                # =========================================================
                # 額外抽取 Target Tail auxiliary recommendation batch
                # =========================================================
                target_tail_aux_seed = (
                    args.random_seed
                    + 2000000
                    + epoch_temp * num_batches
                    + j
                )

                target_tail_aux_batch_df = (
                    target_tail_aux_data
                    .sample(
                        n=args.target_tail_aux_batch_size,
                        replace=(
                            len(target_tail_aux_data)
                            < args.target_tail_aux_batch_size
                        ),
                        random_state=target_tail_aux_seed
                    )
                    .copy()
                )

                target_tail_aux_seq = torch.LongTensor(
                    target_tail_aux_batch_df[
                        'seq'
                    ].tolist()
                ).to(device)

                target_tail_aux_target = (
                    torch.LongTensor(
                        target_tail_aux_batch_df[
                            'next'
                        ].tolist()
                    )
                    .unsqueeze(1)
                    .to(device)
                )

                # 第一個 epoch 的前三個 batch 檢查
                if (
                    epoch_temp == 0
                    and j < 3
                ):
                    auxiliary_augmented_count = int(
                        target_tail_aux_batch_df[
                            'is_augmented'
                        ].sum()
                    )

                    auxiliary_batch_check = {
                        'batch_size': len(
                            target_tail_aux_batch_df
                        ),
                        'original_tail_samples': (
                            len(target_tail_aux_batch_df)
                            - auxiliary_augmented_count
                        ),
                        'augmented_tail_samples': (
                            auxiliary_augmented_count
                        ),
                        'augmented_ratio': round(
                            auxiliary_augmented_count
                            / len(target_tail_aux_batch_df),
                            4
                        )
                    }

                    print(
                        'First-Stage Target-Tail Auxiliary Batch Check',
                        auxiliary_batch_check
                    )
            else:
                con_seq = None
                target_tail_aux_seq = None
                target_tail_aux_target = None
            # =========================================================
            # 1. 原本的 Source recommendation forward
            # =========================================================
            (
                scores,
                diffu_rep,
                weights,
                t,
                item_rep_dis,
                seq_rep_dis,
                em_loss
            ) = model_joint(
                seq,
                target,
                con_seq,
                pretrain_flag,
                args,
                epoch_temp,
                train_flag=True
            )

            main_rec_loss = model_joint.loss_diffu_ce(
                diffu_rep,
                target,
                pretrain_flag
            )


            # =========================================================
            # 2. 預設第二階段沒有 auxiliary loss
            # =========================================================
            target_tail_aux_loss = torch.zeros(
                (),
                device=device
            )


            # =========================================================
            # 3. 第一階段額外計算 Target-Tail auxiliary loss
            # =========================================================
            if pretrain_flag:
                (
                    _,
                    target_tail_diffu_rep,
                    _,
                    _,
                    _,
                    _,
                    _
                ) = model_joint(
                    target_tail_aux_seq,
                    target_tail_aux_target,
                    None,
                    False,
                    args,
                    epoch_temp,
                    train_flag=True
                )

                target_tail_aux_loss = (
                    model_joint.loss_diffu_ce(
                        target_tail_diffu_rep,
                        target_tail_aux_target,
                        False
                    )
                )


            # =========================================================
            # 4. 合併總 loss
            # =========================================================
            loss_all = (
                main_rec_loss
                + em_loss * args.loss_lambda
                + args.target_tail_aux_weight
                * target_tail_aux_loss
            )


            # =========================================================
            # 5. Loss 檢查放在這裡
            #    必須在 loss_all.backward() 前面
            # =========================================================
            if (
                pretrain_flag
                and epoch_temp == 0
                and j < 3
            ):
                loss_check = {
                    'source_rec_loss': round(
                        main_rec_loss.detach().item(),
                        6
                    ),
                    'alignment_loss': round(
                        em_loss.detach().item(),
                        6
                    ),
                    'weighted_alignment_loss': round(
                        (
                            args.loss_lambda * em_loss
                        ).detach().item(),
                        6
                    ),
                    'target_tail_aux_loss': round(
                        target_tail_aux_loss.detach().item(),
                        6
                    ),
                    'weighted_target_tail_aux_loss': round(
                        (
                            args.target_tail_aux_weight
                            * target_tail_aux_loss
                        ).detach().item(),
                        6
                    ),
                    'total_loss': round(
                        loss_all.detach().item(),
                        6
                    )
                }

                print(
                    'First-Stage Loss Check',
                    loss_check
                )

                logger.info(
                    'First-Stage Loss Check %s',
                    loss_check
                )


            # =========================================================
            # 6. 最後才做 backward 和更新參數
            # =========================================================
            loss_all.backward()
            optimizer.step()
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        lr_scheduler.step()
        
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                for j in range(int(val_data.shape[0]/args.batch_size)):
                    batch = val_data[j * args.batch_size: (j + 1)* args.batch_size].to_dict()
                    seq = list(batch['seq'].values())
                    target=list(batch['next'].values())
                    seq = torch.LongTensor(seq)
                    target = (torch.LongTensor(target)).unsqueeze(1)
                    seq = seq.to(device)
                    target = target.to(device)
                    scores_rec, rep_diffu, _, _, _, _ , _ = model_joint(seq, target, None, pretrain_flag, args, epoch_temp, train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu, pretrain_flag)    ### inner_production
                    output = torch.full_like(scores_rec_diffu, -100.0)
                    row_indices = torch.arange(args.batch_size).unsqueeze(1)
                    negative_samples = list(batch['negative_samples'].values())
                    output[row_indices, negative_samples] = scores_rec_diffu[row_indices, negative_samples]
                    scores_rec_diffu = output
                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, target, metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
                        
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update += 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                if flag_update >= 3:
                    best_model = copy.deepcopy(model_joint)
                    if pretrain_flag:
                        torch.save(model_joint.state_dict(), "./saved_model/"+args.s_dataset + "_" + args.t_dataset+"/model.pth")
            if bad_count >= args.patience:
                break
          
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
    
    top_100_item = []

    with torch.no_grad():
        # ============================================
        # 1. 初始化 Overall / Popular / Unpopular 指標
        # ============================================
        overall_metric_sums = init_metric_sums(metric_ks)
        popular_metric_sums = init_metric_sums(metric_ks)
        unpopular_seen_metric_sums = init_metric_sums(metric_ks)
        unseen_metric_sums = init_metric_sums(metric_ks)

        overall_count = 0
        popular_count = 0
        unpopular_seen_count = 0
        unseen_count = 0

        # Top-K 推薦清單中 popular items 的數量
        popular_recommend_count = {
            k: 0
            for k in metric_ks
        }

        total_recommend_count = {
            k: 0
            for k in metric_ks
        }

        popular_items = None
        unpopular_seen_items = None
        observed_train_items = None
        item_counter = None

        # 只在 target-domain 階段執行 popularity 分析
        if not pretrain_flag:
            popular_ratio = getattr(args, 'popular_ratio', 0.2)

            (
            popular_items,
            unpopular_seen_items,
            observed_train_items,
            item_counter
            ) = build_item_popularity_groups(
                train_data,
                popular_ratio=popular_ratio
            )

            popularity_definition = {
                'popular_ratio': popular_ratio,

                # 訓練集 seq + next 中出現過的商品數
                'observed_train_items': len(
                    observed_train_items
                ),

                # observed items 中的前 20%
                'popular_items': len(
                    popular_items
                ),

                # observed items 中其餘 80%
                'unpopular_seen_items': len(
                    unpopular_seen_items
                ),

                # Unseen 數量要等測試 target 跑完才知道
                'unseen_items_in_test': (
                    'calculated from test ground truth'
                )
            }

            print(
                'Popularity Definition'
                '------------------------------------------------'
            )
            print(popularity_definition)

            logger.info(
                'Popularity Definition'
                '------------------------------------------------'
            )
            logger.info(popularity_definition)

        popular_lookup = None

        # 使用 range(0, ..., batch_size)，讓最後不足一個 batch 的資料也被評估
        for start_idx in range(
            0,
            test_data.shape[0],
            args.batch_size
        ):
            end_idx = min(
                start_idx + args.batch_size,
                test_data.shape[0]
            )

            batch_df = test_data.iloc[start_idx:end_idx]

            seq = batch_df['seq'].tolist()
            target_list = batch_df['next'].tolist()
            negative_samples_list = (
                batch_df['negative_samples'].tolist()
            )

            current_batch_size = len(batch_df)

            seq = torch.LongTensor(seq).to(device)

            target = (
                torch.LongTensor(target_list)
                .unsqueeze(1)
                .to(device)
            )

            scores_rec, rep_diffu, _, _, _, _, _ = best_model(
                seq,
                target,
                None,
                pretrain_flag,
                args,
                0.2,
                train_flag=False
            )

            scores_rec_diffu = best_model.diffu_rep_pre(
                rep_diffu,
                pretrain_flag
            )

            # ============================================
            # 2. 保留原本 1 positive + 100 negatives 的設定
            # ============================================
            output = torch.full_like(
                scores_rec_diffu,
                -100.0
            )

            row_indices = torch.arange(
                current_batch_size,
                device=device
            ).unsqueeze(1)

            negative_samples = torch.as_tensor(
                negative_samples_list,
                dtype=torch.long,
                device=device
            )

            output[
                row_indices,
                negative_samples
            ] = scores_rec_diffu[
                row_indices,
                negative_samples
            ]

            scores_rec_diffu = output

            # 儲存 Top-100，保留原本功能
            top_100_k = min(
                100,
                scores_rec_diffu.shape[1]
            )

            top_100_indices = torch.topk(
                scores_rec_diffu,
                k=top_100_k,
                dim=-1
            ).indices

            top_100_item.append(
                top_100_indices.detach().cpu()
            )

            # ============================================
            # 3. 計算每一筆資料的 HR / NDCG
            # ============================================
            sample_metrics, topk_indices = per_sample_metrics(
                scores_rec_diffu,
                target,
                metric_ks
            )

            overall_count += current_batch_size

            for metric_name, values in sample_metrics.items():
                overall_metric_sums[metric_name] += (
                    values.sum().item()
                )

            # ==================================================
            # 4. Target-domain Popular / Unpopular-seen /
            #    Unseen 分組
            # ==================================================
            if not pretrain_flag:
                target_cpu = (
                    target.view(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )

                # 是否屬於 training 中的 popular items
                popular_mask = torch.tensor(
                    [
                        int(item) in popular_items
                        for item in target_cpu
                    ],
                    dtype=torch.bool
                )

                # 是否曾在 training seq 或 next 出現
                observed_mask = torch.tensor(
                    [
                        int(item) in observed_train_items
                        for item in target_cpu
                    ],
                    dtype=torch.bool
                )

                # Training 中出現過，但不是 popular
                unpopular_seen_mask = (
                    observed_mask & ~popular_mask
                )

                # Training seq 與 next 都完全沒出現過
                unseen_mask = ~observed_mask

                batch_popular_count = (
                    popular_mask.sum().item()
                )

                batch_unpopular_seen_count = (
                    unpopular_seen_mask.sum().item()
                )

                batch_unseen_count = (
                    unseen_mask.sum().item()
                )

                # 確認三組互斥且涵蓋整個 batch
                batch_group_total = (
                    batch_popular_count
                    + batch_unpopular_seen_count
                    + batch_unseen_count
                )

                if batch_group_total != current_batch_size:
                    raise RuntimeError(
                        "Popularity grouping error: "
                        f"group total={batch_group_total}, "
                        f"batch size={current_batch_size}"
                    )

                popular_count += batch_popular_count

                unpopular_seen_count += (
                    batch_unpopular_seen_count
                )

                unseen_count += batch_unseen_count

                for metric_name, values in sample_metrics.items():
                    if batch_popular_count > 0:
                        popular_metric_sums[
                            metric_name
                        ] += (
                            values[popular_mask]
                            .sum()
                            .item()
                        )

                    if batch_unpopular_seen_count > 0:
                        unpopular_seen_metric_sums[
                            metric_name
                        ] += (
                            values[unpopular_seen_mask]
                            .sum()
                            .item()
                        )

                    if batch_unseen_count > 0:
                        unseen_metric_sums[
                            metric_name
                        ] += (
                            values[unseen_mask]
                            .sum()
                            .item()
                        )

                # ========================================
                # 5. Top-K 推薦清單中的 PopularRatio
                # ========================================
                if popular_lookup is None:
                    popular_lookup = torch.zeros(
                        scores_rec_diffu.shape[1],
                        dtype=torch.bool,
                        device=device
                    )

                    valid_popular_ids = [
                        int(item)
                        for item in popular_items
                        if 0 <= int(item) < popular_lookup.shape[0]
                    ]

                    if len(valid_popular_ids) > 0:
                        popular_lookup[
                            torch.LongTensor(
                                valid_popular_ids
                            ).to(device)
                        ] = True

                for k in metric_ks:
                    recommended_items = topk_indices[:, :k]

                    popular_recommend_count[k] += (
                        popular_lookup[recommended_items]
                        .sum()
                        .item()
                    )

                    total_recommend_count[k] += (
                        current_batch_size * k
                    )

        # ============================================
        # 6. 計算最終結果
        # ============================================
        test_metrics_dict_mean = finalize_metric_sums(
            overall_metric_sums,
            overall_count
        )

        print(
            'Test Overall'
            '------------------------------------------------------'
        )
        print(test_metrics_dict_mean)

        logger.info(
            'Test Overall'
            '------------------------------------------------------'
        )
        logger.info(test_metrics_dict_mean)

        if not pretrain_flag:
            popular_metrics_dict = finalize_metric_sums(
                popular_metric_sums,
                popular_count
            )

            unpopular_seen_metrics_dict = (
                finalize_metric_sums(
                    unpopular_seen_metric_sums,
                    unpopular_seen_count
                )
            )

            unseen_metrics_dict = finalize_metric_sums(
                unseen_metric_sums,
                unseen_count
            )

            popular_ratio_dict = {
                f'PopularRatio@{k}': round(
                    popular_recommend_count[k]
                    / total_recommend_count[k]
                    * 100,
                    4
                )
                if total_recommend_count[k] > 0
                else None
                for k in metric_ks
            }

            group_size_dict = {
                'Overall_test_samples': overall_count,

                'Popular_test_samples': (
                    popular_count
                ),

                'Unpopular_seen_test_samples': (
                    unpopular_seen_count
                ),

                'Unseen_test_samples': (
                    unseen_count
                ),

                # 用來檢查三組加總
                'Grouped_test_samples': (
                    popular_count
                    + unpopular_seen_count
                    + unseen_count
                )
            }

            print(
                'Test Group Size'
                '---------------------------------------------------'
            )
            print(group_size_dict)

            print(
                'Test Popular Ground Truth'
                '------------------------------------------'
            )
            print(popular_metrics_dict)

            print(
                'Test Unpopular-Seen Ground Truth'
                '-----------------------------------'
            )
            print(unpopular_seen_metrics_dict)

            print(
                'Test Unseen Ground Truth'
                '-------------------------------------------'
            )
            print(unseen_metrics_dict)

            print(
                'Recommendation Popularity'
                '------------------------------------------'
            )
            print(popular_ratio_dict)

            logger.info(
                'Test Group Size'
                '---------------------------------------------------'
            )
            logger.info(group_size_dict)

            logger.info(
                'Test Popular Ground Truth'
                '------------------------------------------'
            )
            logger.info(popular_metrics_dict)

            logger.info(
                'Test Unpopular-Seen Ground Truth'
                '-----------------------------------'
            )
            logger.info(unpopular_seen_metrics_dict)

            logger.info(
                'Test Unseen Ground Truth'
                '-------------------------------------------'
            )
            logger.info(unseen_metrics_dict)

            logger.info(
                'Recommendation Popularity'
                '------------------------------------------'
            )
            logger.info(popular_ratio_dict)
        
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)

    return best_model, test_metrics_dict_mean
    
