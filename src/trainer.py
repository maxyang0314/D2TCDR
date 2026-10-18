import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import pandas as pd
import copy

from collections import Counter

from agent.state import (
    build_target_frequency,
    build_sample_states
)

from agent.policy import (
    DebiasAgent,
    sample_actions
)

from agent.reward import (
    RewardBaseline,
    calculate_reward,
    evaluate_reward
)

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

def model_train(train_data, val_data, test_data, con_data, model_joint, args, logger, pretrain_flag, reward_data=None):
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
    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    
    # ==============================
    # Agent Initialization
    # ==============================

    use_agent = (
        not pretrain_flag
        and args.agent_enabled == 1
    )

    best_model = copy.deepcopy(model_joint)
    best_val_ndcg10 = float("-inf")

    agent = None
    best_agent_state = None

    budget_carry = 0.0

    if not pretrain_flag:

        (
            popular_items,
            tail_items,
            observed_items,
            item_counter
        ) = build_item_popularity_groups(
            train_data,
            popular_ratio=args.popular_ratio
        )

        target_counter, max_frequency = (
            build_target_frequency(train_data)
        )

        if reward_data is None or len(reward_data) == 0:
            raise ValueError(
                "Target training requires Reward Set."
            )

        if args.reward_eval_samples > 0:
            reward_subset = reward_data.iloc[
                :args.reward_eval_samples
            ].copy()
        else:
            reward_subset = reward_data.copy()

    if use_agent:

        agent = DebiasAgent(
            hidden_dim=args.agent_hidden
        ).to(device)

        agent_optimizer = optim.Adam(
            agent.parameters(),
            lr=args.agent_lr
        )

        reward_baseline = RewardBaseline(
            window=args.reward_baseline_window
        )

        best_agent_state = copy.deepcopy(
            agent.state_dict()
        )

        print("RL Agent Enabled")
        print("Popular Items:", len(popular_items))
        print("Tail Items:", len(tail_items))
        print("Reward Samples:", len(reward_subset))

    for epoch_temp in range(epochs):

        model_joint.train()
        flag_update = 0
        for j in range(num_batches):
            # ==============================
            # Step 1: Load Original Batch
            # ==============================

            batch_df = train_data.sample(
                n=args.batch_size
            ).reset_index(drop=True)

            batch_weights = None
            joint_log_prob = None

            # ==============================
            # Step 2: Agent Joint Action
            # ==============================

            if use_agent:

                states, eligible_mask = (
                    build_sample_states(
                        batch_df,
                        popular_items,
                        tail_items,
                        target_counter,
                        max_frequency,
                        device
                    )
                )

                (
                    duplicate_idx,
                    sample_weights,
                    joint_log_prob,
                    budget_carry
                ) = sample_actions(
                    agent,
                    states,
                    eligible_mask,
                    budget_carry=budget_carry,
                    budget_ratio=args.agent_budget_ratio
                )

                # Action B: Duplicate Samples
                if duplicate_idx.numel() > 0:

                    duplicate_positions = (
                        duplicate_idx.detach()
                        .cpu()
                        .tolist()
                    )

                    duplicate_rows = batch_df.iloc[
                        duplicate_positions
                    ].copy()

                    batch_df = pd.concat(
                        [batch_df, duplicate_rows],
                        ignore_index=True
                    )

                # Action C: Weights
                # Duplicates inherit original weights
                batch_weights = torch.cat(
                    [
                        sample_weights,
                        sample_weights[duplicate_idx]
                    ]
                ).detach()

            # ==============================
            # Step 3: Build Tensors
            # ==============================

            seq = torch.as_tensor(
                batch_df["seq"].tolist(),
                dtype=torch.long,
                device=device
            )

            target = torch.as_tensor(
                batch_df["next"].tolist(),
                dtype=torch.long,
                device=device
            ).unsqueeze(1)

            optimizer.zero_grad()

            # ==============================
            # Step 4: Source Condition
            # ==============================

            if pretrain_flag:

                con_batch_df = con_data.sample(
                    n=args.batch_size
                )

                con_seq = torch.as_tensor(
                    con_batch_df["seq"].tolist(),
                    dtype=torch.long,
                    device=device
                )

            else:
                con_seq = None

            # ==============================
            # Step 5: D2TCDR Forward
            # ==============================

            (
                scores,
                diffu_rep,
                _,
                _,
                _,
                _,
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
            
            # ==============================
            # Step 6: Recommendation Loss
            # ==============================

            if use_agent:

                ce_per_sample = (
                    model_joint.loss_diffu_ce_per_sample(
                        diffu_rep,
                        target,
                        pretrain_flag=False
                    )
                )

                assert (
                    ce_per_sample.numel()
                    == batch_weights.numel()
                )

                weighted_ce = (
                    ce_per_sample * batch_weights
                )

                loss_diffu_value = (
                    weighted_ce.sum()
                    / ce_per_sample.numel()
                )

            else:

                loss_diffu_value = (
                    model_joint.loss_diffu_ce(
                        diffu_rep,
                        target,
                        pretrain_flag
                    )
                )

            # Preserve Original Auxiliary Loss
            loss_all = (
                loss_diffu_value
                + em_loss * args.loss_lambda
            )

            # ==============================
            # Step 7: Update D2TCDR
            # ==============================

            loss_all.backward()
            optimizer.step()

            # ==============================
            # Step 8: Reward Evaluation
            # ==============================

            if use_agent and joint_log_prob is not None:

                metrics = evaluate_reward(
                    model_joint,
                    reward_subset,
                    args,
                    popular_items,
                    tail_items,
                    epoch=epoch_temp,
                    random_seed=args.random_seed + 707
                )

                reward = calculate_reward(
                    metrics,
                    args.reward_ref_overall,
                    args.reward_ref_popular,
                    args.reward_lambda_overall,
                    args.reward_lambda_popular,
                    args.reward_epsilon_overall,
                    args.reward_epsilon_popular
                )

                # ==============================
                # Step 9: Advantage
                # ==============================

                previous_average = (
                    reward_baseline.previous_average(
                        reward
                    )
                )

                advantage = (
                    reward - previous_average
                )

                # ==============================
                # Step 10: REINFORCE
                # ==============================

                agent_optimizer.zero_grad()

                policy_loss = (
                    -advantage * joint_log_prob
                )

                policy_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    agent.parameters(),
                    max_norm=5.0
                )

                # Save parameters before update
                selection_before = (
                    agent.selection_head.weight
                    .detach().clone()
                )

                weight_before = (
                    agent.weight_head.weight
                    .detach().clone()
                )

                # Update Agent
                agent_optimizer.step()

                # Measure actual parameter changes
                selection_delta = (
                    agent.selection_head.weight.detach()
                    - selection_before
                ).abs().max().item()

                weight_delta = (
                    agent.weight_head.weight.detach()
                    - weight_before
                ).abs().max().item()

                reward_baseline.update(reward)

                # ==============================
                # Logging
                # ==============================

                if j < 5 or (j + 1) % 5 == 0:

                    tail_count = eligible_mask.sum().item()
                    duplicate_count = duplicate_idx.numel()

                    message = (
                        f"Epoch={epoch_temp}, "
                        f"Batch={j + 1}, "
                        f"TailSamples={tail_count}, "
                        f"Oversampled={duplicate_count}, "
                        f"WeightMax={batch_weights.max().item():.4f}, "
                        f"Reward={reward:.6f}, "
                        f"Advantage={advantage:.6f}, "
                        f"PolicyLoss={policy_loss.item():.6f}, "
                        f"GradNorm={grad_norm.item():.6f}, "
                        f"Tail_HR@10={metrics['Tail_HR@10']:.4f}, "
                        f"Overall_HR@10={metrics['Overall_HR@10']:.4f}, "
                        f"Popular_HR@10={metrics['Popular_HR@10']:.4f}"
                    )

                    print(message)
                    logger.info(message)

        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        lr_scheduler.step()
        
        if (
            (epoch_temp + 1) % args.eval_interval == 0
            or (epoch_temp + 1) == epochs
        ):
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
                        
            
            # ==============================
            # Update Validation Metrics
            # ==============================

            for key_temp, values_temp in metrics_dict.items():

                values_mean = round(
                    np.mean(values_temp) * 100,
                    4
                )

                if (
                    values_mean
                    > best_metrics_dict["Best_" + key_temp]
                ):

                    best_metrics_dict[
                        "Best_" + key_temp
                    ] = values_mean

                    best_epoch[
                        "Best_epoch_" + key_temp
                    ] = epoch_temp

            # ==============================
            # Model Selection: NDCG@10
            # ==============================

            current_ndcg10 = (
                np.mean(metrics_dict["NDCG@10"])
                * 100
            )

            if current_ndcg10 > best_val_ndcg10:

                best_val_ndcg10 = current_ndcg10
                bad_count = 0

                best_model = copy.deepcopy(
                    model_joint
                )

                if use_agent:
                    best_agent_state = copy.deepcopy(
                        agent.state_dict()
                    )

                if pretrain_flag:
                    torch.save(
                        best_model.state_dict(),
                        "./saved_model/"
                        + args.s_dataset
                        + "_"
                        + args.t_dataset
                        + "/model.pth"
                    )

                print(
                    "Best Validation NDCG@10:",
                    best_val_ndcg10
                )

            else:
                bad_count += 1

            if bad_count >= args.patience:
                break
          
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    
    # ==============================
    # Fallback Checkpoint
    # ==============================

    if best_val_ndcg10 == float("-inf"):

        best_model = copy.deepcopy(model_joint)

        if use_agent:
            best_agent_state = copy.deepcopy(
                agent.state_dict()
            )

    best_model.eval()

    # ==============================
    # Save Agent Policy
    # ==============================

    if use_agent:

        import os

        model_dir = (
            "./saved_model/"
            + args.s_dataset
            + "_"
            + args.t_dataset
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )

        torch.save(
            best_agent_state,
            model_dir + "/agent_policy.pth"
        )

        print("Agent Policy Saved")

    # ==============================
    # Evaluate Reward Reference
    # ==============================

    if not pretrain_flag:

        reward_metrics = evaluate_reward(
            best_model,
            reward_subset,
            args,
            popular_items,
            tail_items,
            random_seed=args.random_seed + 707
        )

        print(
            "REWARD_SET_RESULTS:",
            reward_metrics
        )

        logger.info(
            "REWARD_SET_RESULTS: %s",
            reward_metrics
        )

    
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

        # ========================================================
        # 建立目前 Domain 的 Popular / Unpopular-seen 集合
        # ========================================================

        # pretrain_flag=True：目前評估的是 Source domain
        # pretrain_flag=False：目前評估的是 Target domain
        domain_name = (
            'Source'
            if pretrain_flag
            else 'Target'
        )

        # main 分支目前只有 popular_ratio。
        # 未來套用到 Source-next 分支時，如果有
        # source_popular_ratio，就優先使用該參數。
        if pretrain_flag:
            popular_ratio = getattr(
                args,
                'source_popular_ratio',
                getattr(
                    args,
                    'popular_ratio',
                    0.2
                )
            )
        else:
            popular_ratio = getattr(
                args,
                'popular_ratio',
                0.2
            )

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
            'domain': domain_name,
            'popular_ratio': popular_ratio,

            # 目前 domain 的 training seq + next
            # 中出現過的商品種類數
            'observed_train_items': len(
                observed_train_items
            ),

            # observed items 中最熱門的前 20%
            'popular_items': len(
                popular_items
            ),

            # observed items 中其餘 80%
            'unpopular_seen_items': len(
                unpopular_seen_items
            ),

            # 測試時才會知道有哪些 ground truth unseen
            'unseen_items_in_test': (
                'calculated from test ground truth'
            )
        }

        print(
            f'{domain_name} Popularity Definition'
            '---------------------------------------------'
        )
        print(popularity_definition)

        logger.info(
            '%s Popularity Definition'
            '---------------------------------------------',
            domain_name
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
            # 4. 依目前 Domain 的 test ground truth 分組
            # ==================================================

            target_cpu = (
                target.view(-1)
                .detach()
                .cpu()
                .tolist()
            )

            # Ground truth 是否為 Popular item
            popular_mask = torch.tensor(
                [
                    int(item) in popular_items
                    for item in target_cpu
                ],
                dtype=torch.bool
            )

            # Ground truth 是否曾出現在目前 domain 的
            # training seq 或 training next
            observed_mask = torch.tensor(
                [
                    int(item) in observed_train_items
                    for item in target_cpu
                ],
                dtype=torch.bool
            )

            # Training 中出現過，但不是 Popular
            unpopular_seen_mask = (
                observed_mask
                & ~popular_mask
            )

            # Training seq 和 next 都沒出現過
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

            # 確認 Popular、Unpopular-seen、Unseen
            # 三組互斥且涵蓋整個 batch
            batch_group_total = (
                batch_popular_count
                + batch_unpopular_seen_count
                + batch_unseen_count
            )

            if batch_group_total != current_batch_size:
                raise RuntimeError(
                    f"{domain_name} popularity grouping error: "
                    f"group total={batch_group_total}, "
                    f"batch size={current_batch_size}"
                )

            popular_count += batch_popular_count

            unpopular_seen_count += (
                batch_unpopular_seen_count
            )

            unseen_count += batch_unseen_count

            # 分別累積三組 HR / NDCG
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
                        values[
                            unpopular_seen_mask
                        ]
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

            # ==================================================
            # 5. Top-K 推薦清單中的 PopularRatio
            # ==================================================

            # 建立 item ID → 是否為 Popular 的查詢表
            if popular_lookup is None:
                popular_lookup = torch.zeros(
                    scores_rec_diffu.shape[1],
                    dtype=torch.bool,
                    device=device
                )

                valid_popular_ids = [
                    int(item)
                    for item in popular_items
                    if (
                        0
                        <= int(item)
                        < popular_lookup.shape[0]
                    )
                ]

                if len(valid_popular_ids) > 0:
                    popular_lookup[
                        torch.LongTensor(
                            valid_popular_ids
                        ).to(device)
                    ] = True

            # 統計 Top-K 中有多少推薦商品屬於 Popular
            for k in metric_ks:
                recommended_items = (
                    topk_indices[:, :k]
                )

                popular_recommend_count[k] += (
                    popular_lookup[
                        recommended_items
                    ]
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
            f'{domain_name} Test Overall'
            '------------------------------------------------------'
        )
        print(test_metrics_dict_mean)

        logger.info(
            '%s Test Overall'
            '------------------------------------------------------'
        )
        logger.info(test_metrics_dict_mean)

        # ========================================================
        # 計算目前 Domain 的分組結果
        # ========================================================

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
            'domain': domain_name,

            'Overall_test_samples': (
                overall_count
            ),

            'Popular_test_samples': (
                popular_count
            ),

            'Unpopular_seen_test_samples': (
                unpopular_seen_count
            ),

            'Unseen_test_samples': (
                unseen_count
            ),

            'Grouped_test_samples': (
                popular_count
                + unpopular_seen_count
                + unseen_count
            )
        }

        print(
            f'{domain_name} Test Group Size'
            '----------------------------------------------'
        )
        print(group_size_dict)

        print(
            f'{domain_name} Test Popular Ground Truth'
            '-------------------------------------'
        )
        print(popular_metrics_dict)

        print(
            f'{domain_name} Test Unpopular-Seen Ground Truth'
            '------------------------------'
        )
        print(unpopular_seen_metrics_dict)

        print(
            f'{domain_name} Test Unseen Ground Truth'
            '--------------------------------------'
        )
        print(unseen_metrics_dict)

        print(
            f'{domain_name} Recommendation Popularity'
            '-------------------------------------'
        )
        print(popular_ratio_dict)

        logger.info(
            '%s Test Group Size'
            '----------------------------------------------',
            domain_name
        )
        logger.info(group_size_dict)

        logger.info(
            '%s Test Popular Ground Truth'
            '-------------------------------------',
            domain_name
        )
        logger.info(popular_metrics_dict)

        logger.info(
            '%s Test Unpopular-Seen Ground Truth'
            '------------------------------',
            domain_name
        )
        logger.info(
            unpopular_seen_metrics_dict
        )

        logger.info(
            '%s Test Unseen Ground Truth'
            '--------------------------------------',
            domain_name
        )
        logger.info(unseen_metrics_dict)

        logger.info(
            '%s Recommendation Popularity'
            '-------------------------------------',
            domain_name
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
    
