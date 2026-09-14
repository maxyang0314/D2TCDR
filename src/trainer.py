import os
import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import time
import random

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

def calculate_sequence_popular_ratio(
    seq,
    popular_items
):
    """
    計算一條 sequence 中 Popular interaction 的比例。
    Padding 0 不計算。
    """

    valid_items = [
        int(item)
        for item in seq
        if int(item) != 0
    ]

    if len(valid_items) == 0:
        return 0.0

    popular_count = sum(
        1
        for item in valid_items
        if item in popular_items
    )

    return popular_count / len(valid_items)

# ============================================================
# Experiment F：Target Inner-Train / Meta split
# ============================================================

def split_target_train_meta_data(
    train_data,
    popular_ratio=0.2,
    meta_ratio=0.1,
    random_seed=1997
):
    """
    將 Target training data 分成：

        Inner Train
        Meta Pool

    Meta Pool 不參與 recommender optimizer update，
    只用來建立 Meta Objective / g_meta。

    split 會依 Popular-Next / Tail-Next 分層，
    避免 Meta Pool 的 composition 偏掉。
    """

    if not 0.0 < meta_ratio < 1.0:
        raise ValueError(
            'f_meta_ratio must be between 0 and 1.'
        )

    (
        popular_items,
        unpopular_seen_items,
        _,
        _
    ) = build_item_popularity_groups(
        train_data,
        popular_ratio=popular_ratio
    )

    next_is_popular = train_data['next'].apply(
        lambda x: int(x) in popular_items
    )

    popular_next_df = train_data[
        next_is_popular
    ]

    tail_next_df = train_data[
        ~next_is_popular
    ]

    popular_meta_size = max(
        1,
        int(
            round(
                len(popular_next_df)
                * meta_ratio
            )
        )
    )

    tail_meta_size = max(
        1,
        int(
            round(
                len(tail_next_df)
                * meta_ratio
            )
        )
    )

    popular_meta_indices = (
        popular_next_df
        .sample(
            n=popular_meta_size,
            random_state=random_seed
        )
        .index
        .tolist()
    )

    tail_meta_indices = (
        tail_next_df
        .sample(
            n=tail_meta_size,
            random_state=random_seed + 1
        )
        .index
        .tolist()
    )

    meta_indices = (
        popular_meta_indices
        + tail_meta_indices
    )

    meta_data = (
        train_data
        .loc[meta_indices]
        .sample(
            frac=1.0,
            random_state=random_seed + 2
        )
        .reset_index(drop=True)
    )

    inner_train_data = (
        train_data
        .drop(index=meta_indices)
        .reset_index(drop=True)
    )

    return (
        inner_train_data,
        meta_data,
        popular_items,
        unpopular_seen_items
    )

# ============================================================
# Experiment F：Balanced Meta Batch
# ============================================================

def sample_balanced_meta_batch(
    meta_data,
    popular_items,
    batch_size=64,
    random_seed=1997
):
    """
    每個 Meta batch 固定 Popular-Next / Tail-Next 各一半。
    """

    if batch_size < 2:
        raise ValueError(
            'f_meta_batch_size must be >= 2.'
        )

    next_is_popular = meta_data['next'].apply(
        lambda x: int(x) in popular_items
    )

    popular_df = meta_data[
        next_is_popular
    ]

    tail_df = meta_data[
        ~next_is_popular
    ]

    num_popular = batch_size // 2

    num_tail = (
        batch_size
        - num_popular
    )

    popular_indices = (
        popular_df
        .sample(
            n=num_popular,
            replace=(
                len(popular_df)
                < num_popular
            ),
            random_state=random_seed
        )
        .index
        .tolist()
    )

    tail_indices = (
        tail_df
        .sample(
            n=num_tail,
            replace=(
                len(tail_df)
                < num_tail
            ),
            random_state=random_seed + 1
        )
        .index
        .tolist()
    )

    batch_indices = (
        popular_indices
        + tail_indices
    )

    meta_batch = (
        meta_data
        .loc[batch_indices]
        .sample(
            frac=1.0,
            random_state=random_seed + 2
        )
        .reset_index(drop=True)
    )

    return meta_batch

# ============================================================
# Experiment F：M0 / M1 / M2 Meta Objectives
# ============================================================

def compute_f_meta_objectives(
    scores,
    targets,
    popular_items,
    margin_beta=0.25,
    popmass_gamma=0.25
):
    """
    M0:
        Balanced Popular/Tail CE

    M1:
        M0 + Tail-vs-Popular Margin

    M2:
        M0 + Popular Probability Mass
    """

    targets = targets.view(-1)

    device = scores.device

    # --------------------------------------------------------
    # Popular-item lookup
    # --------------------------------------------------------

    popular_ids = torch.tensor(
        sorted(popular_items),
        dtype=torch.long,
        device=device
    )

    popular_lookup = torch.zeros(
        scores.size(1),
        dtype=torch.bool,
        device=device
    )

    popular_lookup[
        popular_ids
    ] = True

    popular_target_mask = (
        popular_lookup[
            targets
        ]
    )

    tail_target_mask = (
        ~popular_target_mask
    )

    if not popular_target_mask.any():
        raise ValueError(
            'Meta batch contains no Popular targets.'
        )

    if not tail_target_mask.any():
        raise ValueError(
            'Meta batch contains no Tail targets.'
        )

    # ========================================================
    # Base CE
    # ========================================================

    per_sample_ce = F.cross_entropy(
        scores,
        targets,
        reduction='none'
    )

    popular_ce = (
        per_sample_ce[
            popular_target_mask
        ]
        .mean()
    )

    tail_ce = (
        per_sample_ce[
            tail_target_mask
        ]
        .mean()
    )

    # ========================================================
    # M0：Balanced Recommendation
    # ========================================================

    balanced_loss = (
        0.5 * popular_ce
        + 0.5 * tail_ce
    )

    # ========================================================
    # M1：Tail Margin
    # ========================================================

    target_scores = (
        scores
        .gather(
            1,
            targets.unsqueeze(1)
        )
        .squeeze(1)
    )

    popular_candidate_scores = (
        scores[
            :,
            popular_ids
        ]
    )

    max_popular_scores = (
        popular_candidate_scores
        .max(dim=1)
        .values
    )

    tail_margin = (
        target_scores[
            tail_target_mask
        ]
        -
        max_popular_scores[
            tail_target_mask
        ]
    )

    margin_loss = (
        F.softplus(
            -tail_margin
        )
        .mean()
    )

    m1_loss = (
        balanced_loss
        +
        margin_beta
        * margin_loss
    )

    # ========================================================
    # M2：Popular Probability Mass
    # ========================================================

    probabilities = F.softmax(
        scores,
        dim=1
    )

    popular_probability_mass = (
        probabilities[
            :,
            popular_ids
        ]
        .sum(dim=1)
    )

    popmass_loss = (
        popular_probability_mass[
            tail_target_mask
        ]
        .mean()
    )

    m2_loss = (
        balanced_loss
        +
        popmass_gamma
        * popmass_loss
    )

    return {
        'popular_ce':
            popular_ce,

        'tail_ce':
            tail_ce,

        'balanced_loss':
            balanced_loss,

        'margin_loss':
            margin_loss,

        'popmass_loss':
            popmass_loss,

        'M0_balanced':
            balanced_loss,

        'M1_margin':
            m1_loss,

        'M2_popmass':
            m2_loss
    }

# ============================================================
# Experiment F-E v4：
# Evaluate Meta Objectives under current model parameters
# ============================================================

def evaluate_f_meta_losses(
    model_joint,
    meta_batch,
    popular_items,
    args,
    random_seed
):
    """
    在目前 model parameters 下，
    用固定 Meta batch 計算：

        M0
        M1
        M2

    此 function 不更新 model。
    """

    device = args.device

    model_joint = model_joint.to(device)
    model_joint.eval()

    core_model = (
        model_joint.module
        if isinstance(
            model_joint,
            nn.DataParallel
        )
        else model_joint
    )

    seq = torch.LongTensor(
        meta_batch['seq'].tolist()
    ).to(device)

    target = (
        torch.LongTensor(
            meta_batch['next'].tolist()
        )
        .unsqueeze(1)
        .to(device)
    )

    # ----------------------------------------
    # 固定 stochastic process
    # ----------------------------------------

    set_f_probe_seed(
        random_seed
    )

    model_joint.zero_grad(
        set_to_none=True
    )

    with torch.no_grad():

        (
            _,
            diffu_rep,
            _,
            _,
            _,
            _,
            _
        ) = model_joint(
            seq,
            target,
            None,
            False,
            args,
            0,
            train_flag=True
        )

        scores = (
            core_model
            .diffu_rep_pre(
                diffu_rep,
                False
            )
        )

        objective_dict = (
            compute_f_meta_objectives(
                scores=scores,
                targets=target,
                popular_items=popular_items,
                margin_beta=
                    args.f_margin_beta,
                popmass_gamma=
                    args.f_popmass_gamma
            )
        )

    return {
        'M0_balanced':
            objective_dict[
                'M0_balanced'
            ].item(),

        'M1_margin':
            objective_dict[
                'M1_margin'
            ].item(),

        'M2_popmass':
            objective_dict[
                'M2_popmass'
            ].item()
    }

def calculate_gradient_l2_norm(
    loss,
    model
):
    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True
    )

    total_squared_norm = None

    for gradient in gradients:

        if gradient is None:
            continue

        squared_norm = (
            gradient
            .detach()
            .pow(2)
            .sum()
        )

        if total_squared_norm is None:
            total_squared_norm = (
                squared_norm
            )
        else:
            total_squared_norm = (
                total_squared_norm
                + squared_norm
            )

    if total_squared_norm is None:
        return 0.0

    return (
        torch.sqrt(
            total_squared_norm
        )
        .item()
    )

# ============================================================
# Experiment F-E v4：
# Compute per-sample parameter gradients
# ============================================================

def compute_f_sample_parameter_gradients(
    model_joint,
    regulated_df,
    args,
    random_seed
):
    """
    對一筆 regulated sample 計算：

        L_i^a
        g_i^a = gradient(L_i^a)

    不 flatten gradient，
    因為等一下要真的對 parameters 做 virtual update。
    """

    device = args.device

    model_joint = model_joint.to(device)
    model_joint.eval()

    core_model = (
        model_joint.module
        if isinstance(
            model_joint,
            nn.DataParallel
        )
        else model_joint
    )

    # ========================================================
    # 沿用 v3 的 probe parameter space
    # ========================================================

    (
        probe_parameters,
        probe_parameter_names,
        probe_parameter_count
    ) = get_f_probe_parameters(
        model_joint=model_joint,
        probe_space=args.f_probe_space
    )

    seq = torch.LongTensor(
        regulated_df[
            'seq'
        ].tolist()
    ).to(device)

    target = (
        torch.LongTensor(
            regulated_df[
                'next'
            ].tolist()
        )
        .unsqueeze(1)
        .to(device)
    )

    set_f_probe_seed(
        random_seed
    )

    model_joint.zero_grad(
        set_to_none=True
    )

    (
        _,
        diffu_rep,
        _,
        _,
        _,
        _,
        _
    ) = model_joint(
        seq,
        target,
        None,
        False,
        args,
        0,
        train_flag=True
    )

    scores = (
        core_model
        .diffu_rep_pre(
            diffu_rep,
            False
        )
    )

    sample_loss = (
        F.cross_entropy(
            scores,
            target.view(-1)
        )
    )

    gradients = torch.autograd.grad(
        sample_loss,
        probe_parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True
    )

    cleaned_gradients = []

    total_squared_norm = 0.0

    for (
        parameter,
        gradient
    ) in zip(
        probe_parameters,
        gradients
    ):

        if gradient is None:

            gradient = (
                torch.zeros_like(
                    parameter
                )
            )

        gradient = (
            gradient
            .detach()
            .clone()
        )

        cleaned_gradients.append(
            gradient
        )

        total_squared_norm += (
            gradient
            .pow(2)
            .sum()
            .item()
        )

    gradient_norm = (
        total_squared_norm
        ** 0.5
    )

    return (
        probe_parameters,
        cleaned_gradients,
        sample_loss.item(),
        gradient_norm
    )

# ============================================================
# Experiment F：Meta Objective Sanity Check
# ============================================================

def run_experiment_f_meta_check(
    model_joint,
    meta_data,
    popularity_reference_data,
    args,
    logger
):
    """
    使用同一個 Reference Model、
    同一個 balanced Meta batch，

    分別計算：
        M0 Balanced
        M1 Balanced + Margin
        M2 Balanced + PopMass

    並輸出每個 objective 的 gradient norm。

    注意：
        這裡不做 optimizer.step()
        不更新 recommender。
    """

    device = args.device

    (
        popular_items,
        unpopular_seen_items,
        _,
        _
    ) = build_item_popularity_groups(
        popularity_reference_data,
        popular_ratio=args.popular_ratio
    )

    meta_batch = (
        sample_balanced_meta_batch(
            meta_data=meta_data,
            popular_items=popular_items,
            batch_size=
                args.f_meta_batch_size,
            random_seed=
                args.random_seed
                + 700000
        )
    )

    popular_count = sum(
        int(item) in popular_items
        for item
        in meta_batch['next']
    )

    tail_count = (
        len(meta_batch)
        - popular_count
    )

    print(
        'Experiment F Meta Batch'
        '---------------------------------------------'
    )

    meta_batch_info = {
        'Meta Pool Size':
            len(meta_data),

        'Meta Batch Size':
            len(meta_batch),

        'Popular Next':
            popular_count,

        'Tail Next':
            tail_count,

        'Popular Items':
            len(popular_items),

        'Tail Items':
            len(unpopular_seen_items)
    }

    print(meta_batch_info)

    logger.info(
        'Experiment F Meta Batch'
    )

    logger.info(
        meta_batch_info
    )

    seq = torch.LongTensor(
        meta_batch['seq'].tolist()
    ).to(device)

    target = (
        torch.LongTensor(
            meta_batch['next'].tolist()
        )
        .unsqueeze(1)
        .to(device)
    )

    results = {}

    objective_names = [
        'M0_balanced',
        'M1_margin',
        'M2_popmass'
    ]

    model_joint = (
        model_joint
        .to(device)
    )

    model_joint.eval()

    core_model = (
        model_joint.module
        if isinstance(
            model_joint,
            nn.DataParallel
        )
        else model_joint
    )

    # ========================================================
    # 每個 Objective 重新 forward 一次，
    # 避免 retain_graph 造成 GPU memory 壓力。
    #
    # 每次固定相同 random seed，
    # 確保 M0/M1/M2 比較公平。
    # ========================================================

    for objective_name in objective_names:

        random.seed(
            args.random_seed
            + 800000
        )

        np.random.seed(
            args.random_seed
            + 800000
        )

        torch.manual_seed(
            args.random_seed
            + 800000
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                args.random_seed
                + 800000
            )

        model_joint.zero_grad(
            set_to_none=True
        )

        (
            _,
            diffu_rep,
            _,
            _,
            _,
            _,
            _
        ) = model_joint(
            seq,
            target,
            None,
            False,
            args,
            0,
            train_flag=True
        )

        scores = (
            core_model
            .diffu_rep_pre(
                diffu_rep,
                False
            )
        )

        objective_dict = (
            compute_f_meta_objectives(
                scores=scores,
                targets=target,
                popular_items=
                    popular_items,
                margin_beta=
                    args.f_margin_beta,
                popmass_gamma=
                    args.f_popmass_gamma
            )
        )

        selected_loss = (
            objective_dict[
                objective_name
            ]
        )

        grad_norm = (
            calculate_gradient_l2_norm(
                selected_loss,
                model_joint
            )
        )

        result = {
            'Objective':
                objective_name,

            'Total Loss':
                round(
                    selected_loss.item(),
                    6
                ),

            'Balanced Loss':
                round(
                    objective_dict[
                        'balanced_loss'
                    ].item(),
                    6
                ),

            'Popular CE':
                round(
                    objective_dict[
                        'popular_ce'
                    ].item(),
                    6
                ),

            'Tail CE':
                round(
                    objective_dict[
                        'tail_ce'
                    ].item(),
                    6
                ),

            'Margin Loss':
                round(
                    objective_dict[
                        'margin_loss'
                    ].item(),
                    6
                ),

            'PopMass Loss':
                round(
                    objective_dict[
                        'popmass_loss'
                    ].item(),
                    6
                ),

            'Meta Grad Norm':
                round(
                    grad_norm,
                    6
                )
        }

        results[
            objective_name
        ] = result

        print(
            'Experiment F Meta Objective'
            '---------------------------------------------'
        )

        print(result)

        logger.info(
            'Experiment F Meta Objective'
        )

        logger.info(
            result
        )

    return results

# ============================================================
# Experiment F-E v3：
# Gradient Probe Parameter Space
# ============================================================

def get_f_probe_parameters(
    model_joint,
    probe_space='representation'
):
    """
    Experiment F-E gradient observation space.

    output:
        只看 Target item embedding
        = 舊 F-E v2

    representation:
        只看與 sequence representation
        直接相關的 parameters

    combined:
        representation + target item embedding
    """

    core_model = (
        model_joint.module
        if isinstance(
            model_joint,
            nn.DataParallel
        )
        else model_joint
    )

    parameters = []
    parameter_names = []

    # ========================================================
    # Helper
    # ========================================================

    def add_parameter(
        name,
        parameter
    ):
        if parameter.requires_grad:

            parameters.append(
                parameter
            )

            parameter_names.append(
                name
            )

    def add_module(
        prefix,
        module
    ):
        for name, parameter in (
            module.named_parameters()
        ):

            add_parameter(
                prefix
                + '.'
                + name,
                parameter
            )

    # ========================================================
    # Output-space
    # ========================================================

    if probe_space in [
        'output',
        'combined'
    ]:

        add_parameter(
            'target_embeddings.weight',
            core_model
            .target_embeddings
            .weight
        )

    # ========================================================
    # Representation-space
    # ========================================================

    if probe_space in [
        'representation',
        'combined'
    ]:

        # ----------------------------------------------------
        # Position information
        # ----------------------------------------------------

        add_parameter(
            'position_embeddings.weight',
            core_model
            .position_embeddings
            .weight
        )

        # ----------------------------------------------------
        # Input representation normalization
        # ----------------------------------------------------

        add_module(
            'LayerNorm',
            core_model.LayerNorm
        )

        # ----------------------------------------------------
        # Core sequence representation:
        #
        # Diffu_xstart.att
        # = Transformer_rep
        # ----------------------------------------------------

        add_module(
            'diffu.xstart_model.att',
            core_model
            .diffu
            .xstart_model
            .att
        )

        # ----------------------------------------------------
        # Final diffusion representation norm
        # ----------------------------------------------------

        add_module(
            'diffu.xstart_model.norm_diffu_rep',
            core_model
            .diffu
            .xstart_model
            .norm_diffu_rep
        )

    if len(parameters) == 0:

        raise ValueError(
            f'No probe parameters found '
            f'for probe_space={probe_space}'
        )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in parameters
    )

    return (
        parameters,
        parameter_names,
        total_parameter_count
    )

# ============================================================
# Experiment F-E v4：
# Virtual One-Step Meta Improvement
# ============================================================

def compute_f_virtual_meta_reward(
    model_joint,
    probe_parameters,
    sample_gradients,
    baseline_meta_losses,
    meta_batch,
    popular_items,
    args,
    meta_random_seed
):
    """
    1. 暫存原始 parameters
    2. virtual update:
           theta' = theta - eta * gradient
    3. 計算更新後 Meta Loss
    4. Reward:
           L_meta(theta) - L_meta(theta')
    5. 完整恢復原始 parameters

    注意：
        不做 optimizer.step()
        不永久修改 Reference Model。
    """

    virtual_lr = (
        args.f_virtual_lr
    )

    # ========================================================
    # Backup
    # ========================================================

    parameter_backups = [
        parameter
        .detach()
        .clone()

        for parameter
        in probe_parameters
    ]

    # ========================================================
    # Diagnostics
    # ========================================================

    parameter_squared_norm = 0.0
    gradient_squared_norm = 0.0

    for (
        parameter,
        gradient
    ) in zip(
        probe_parameters,
        sample_gradients
    ):

        parameter_squared_norm += (
            parameter
            .detach()
            .pow(2)
            .sum()
            .item()
        )

        gradient_squared_norm += (
            gradient
            .pow(2)
            .sum()
            .item()
        )

    parameter_norm = (
        parameter_squared_norm
        ** 0.5
    )

    gradient_norm = (
        gradient_squared_norm
        ** 0.5
    )

    virtual_step_norm = (
        virtual_lr
        * gradient_norm
    )

    relative_step = (
        virtual_step_norm
        /
        (
            parameter_norm
            + 1e-12
        )
    )

    try:

        # ====================================================
        # Virtual Update
        # ====================================================

        with torch.no_grad():

            for (
                parameter,
                gradient
            ) in zip(
                probe_parameters,
                sample_gradients
            ):

                parameter.add_(
                    gradient,
                    alpha=-virtual_lr
                )

        # ====================================================
        # Evaluate Meta Loss after virtual update
        # ====================================================

        updated_meta_losses = (
            evaluate_f_meta_losses(
                model_joint=
                    model_joint,

                meta_batch=
                    meta_batch,

                popular_items=
                    popular_items,

                args=args,

                random_seed=
                    meta_random_seed
            )
        )

    finally:

        # ====================================================
        # Restore original Reference Model
        # ====================================================

        with torch.no_grad():

            for (
                parameter,
                backup
            ) in zip(
                probe_parameters,
                parameter_backups
            ):

                parameter.copy_(
                    backup
                )

    # ========================================================
    # Reward
    #
    # Positive:
    # virtual update improves Meta objective
    #
    # Negative:
    # virtual update harms Meta objective
    # ========================================================

    reward_result = {}

    for objective_name in [
        'M0_balanced',
        'M1_margin',
        'M2_popmass'
    ]:

        before_loss = (
            baseline_meta_losses[
                objective_name
            ]
        )

        after_loss = (
            updated_meta_losses[
                objective_name
            ]
        )

        reward = (
            before_loss
            - after_loss
        )

        reward_result[
            objective_name
        ] = {

            'reward':
                reward,

            'meta_before':
                before_loss,

            'meta_after':
                after_loss,

            'meta_delta':
                after_loss
                - before_loss,

            'virtual_step_norm':
                virtual_step_norm,

            'relative_step':
                relative_step
        }

    return reward_result

# ============================================================
# Experiment F-E：Gradient Utilities
# ============================================================

def get_flat_gradients(
    loss,
    parameters
):
    """
    對指定 parameters 求 gradient，
    並攤平成一條 vector。
    """

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True
    )

    flat_gradients = []

    for parameter, gradient in zip(
        parameters,
        gradients
    ):

        if gradient is None:

            flat_gradients.append(
                torch.zeros_like(
                    parameter
                ).reshape(-1)
            )

        else:

            flat_gradients.append(
                gradient
                .detach()
                .reshape(-1)
            )

    return torch.cat(
        flat_gradients
    )

def set_f_probe_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Experiment F-E：Build Probe Samples
# ============================================================

def build_f_e_probe_samples(
    train_data,
    popular_items,
    unpopular_seen_items,
    low_threshold,
    high_threshold,
    samples_per_state,
    random_seed
):
    """
    從 Inner Train 中：

    1. 只保留 Unpopular-Next
    2. 依 Sequence Popular Ratio
       分 Low / Medium / High
    3. 每組固定抽 samples_per_state 筆
    """

    state_indices = {
        'low': [],
        'medium': [],
        'high': []
    }

    for index, row in train_data.iterrows():

        target_item = int(
            row['next']
        )

        # ----------------------------------------
        # Experiment E controlled condition:
        # Next = Unpopular
        # ----------------------------------------

        if (
            target_item
            not in unpopular_seen_items
        ):
            continue

        sequence_ratio = (
            calculate_sequence_popular_ratio(
                row['seq'],
                popular_items
            )
        )

        if sequence_ratio < low_threshold:

            state = 'low'

        elif sequence_ratio < high_threshold:

            state = 'medium'

        else:

            state = 'high'

        state_indices[
            state
        ].append(index)

    probe_data = {}

    states = [
        'low',
        'medium',
        'high'
    ]

    for state_offset, state in enumerate(states):

        candidate_indices = (
            state_indices[state]
        )

        if (
            len(candidate_indices)
            < samples_per_state
        ):
            raise ValueError(
                f'Not enough {state} samples. '
                f'Available='
                f'{len(candidate_indices)}, '
                f'Requested='
                f'{samples_per_state}'
            )

        rng = np.random.RandomState(
            random_seed
            + state_offset
        )

        selected_indices = (
            rng.choice(
                candidate_indices,
                size=samples_per_state,
                replace=False
            )
        )

        probe_data[state] = (
            train_data
            .loc[selected_indices]
            .copy()
            .reset_index(drop=True)
        )

    return probe_data

# ============================================================
# Experiment F-E：Build Meta Gradients
# ============================================================

def build_f_meta_gradients(
    model_joint,
    meta_data,
    popular_items,
    args
):
    """
    產生：

        g_meta(M0)
        g_meta(M1)
        g_meta(M2)

    第一版只看 Target item embedding gradient。
    """

    device = args.device

    model_joint = (
        model_joint.to(device)
    )

    model_joint.eval()

    core_model = (
        model_joint.module
        if isinstance(
            model_joint,
            nn.DataParallel
        )
        else model_joint
    )

    # ========================================================
    # 第一版 gradient probe parameter
    # ========================================================

    (
        probe_parameters,
        probe_parameter_names,
        probe_parameter_count
    ) = get_f_probe_parameters(
        model_joint=model_joint,
        probe_space=args.f_probe_space
    )

    print(
        'Experiment F-E Probe Parameter Space'
        '---------------------------------------------'
    )

    print({
        'Probe Space':
            args.f_probe_space,

        'Number of Parameter Tensors':
            len(
                probe_parameters
            ),

        'Total Parameters':
            probe_parameter_count
    })

    # ========================================================
    # 固定一個 balanced Meta batch
    # ========================================================

    meta_batch = (
        sample_balanced_meta_batch(
            meta_data=meta_data,

            popular_items=
            popular_items,

            batch_size=
            args.f_meta_batch_size,

            random_seed=
            args.random_seed
            + 700000
        )
    )

    seq = torch.LongTensor(
        meta_batch[
            'seq'
        ].tolist()
    ).to(device)

    target = (
        torch.LongTensor(
            meta_batch[
                'next'
            ].tolist()
        )
        .unsqueeze(1)
        .to(device)
    )

    objective_names = [
        'M0_balanced',
        'M1_margin',
        'M2_popmass'
    ]

    meta_gradient_info = {}

    for objective_name in objective_names:

        # ----------------------------------------
        # 每個 objective 用同樣 stochastic seed
        # ----------------------------------------

        meta_seed = (
            args.random_seed
            + 800000
        )

        set_f_probe_seed(
            meta_seed
        )

        model_joint.zero_grad(
            set_to_none=True
        )

        (
            _,
            diffu_rep,
            _,
            _,
            _,
            _,
            _
        ) = model_joint(
            seq,
            target,
            None,

            False,      # pretrain_flag
            args,
            0,

            train_flag=True
        )

        scores = (
            core_model
            .diffu_rep_pre(
                diffu_rep,
                False
            )
        )

        objective_dict = (
            compute_f_meta_objectives(
                scores=scores,
                targets=target,

                popular_items=
                popular_items,

                margin_beta=
                args.f_margin_beta,

                popmass_gamma=
                args.f_popmass_gamma
            )
        )

        selected_loss = (
            objective_dict[
                objective_name
            ]
        )

        gradient = (
            get_flat_gradients(
                selected_loss,
                probe_parameters
            )
        )

        gradient_norm = (
            torch.norm(
                gradient,
                p=2
            )
        )

        meta_gradient_info[
            objective_name
        ] = {

            'gradient':
                gradient,

            # ------------------------------------
            # 這個是後面主要使用的
            # ------------------------------------
            'normalized_gradient':
                (
                    gradient
                    /
                    (
                        gradient_norm
                        + 1e-12
                    )
                ),

            'norm':
                gradient_norm.item(),

            'loss':
                selected_loss.item()
        }

    return (
        meta_gradient_info,
        meta_batch
    )

# ============================================================
# Experiment F-E：Sample Action Gradient
# ============================================================

def compute_f_sample_gradient(
    model_joint,
    regulated_df,
    args,
    random_seed
):
    """
    對一筆 regulated Target sample
    計算 CE gradient。

    g_i^a = gradient of sample loss
    """

    device = args.device

    model_joint = (
        model_joint.to(device)
    )

    model_joint.eval()

    core_model = (
        model_joint.module
        if isinstance(
            model_joint,
            nn.DataParallel
        )
        else model_joint
    )

    (
        probe_parameters,
        _,
        _
    ) = get_f_probe_parameters(
        model_joint=model_joint,
        probe_space=args.f_probe_space
    )
    

    seq = torch.LongTensor(
        regulated_df[
            'seq'
        ].tolist()
    ).to(device)

    target = (
        torch.LongTensor(
            regulated_df[
                'next'
            ].tolist()
        )
        .unsqueeze(1)
        .to(device)
    )

    set_f_probe_seed(
        random_seed
    )

    model_joint.zero_grad(
        set_to_none=True
    )

    (
        _,
        diffu_rep,
        _,
        _,
        _,
        _,
        _
    ) = model_joint(
        seq,
        target,
        None,

        False,
        args,
        0,

        train_flag=True
    )

    scores = (
        core_model
        .diffu_rep_pre(
            diffu_rep,
            False
        )
    )

    # ========================================================
    # sample gradient 不使用 M0/M1/M2
    #
    # 它只回答：
    # 這個 regulated sample
    # 自己會產生什麼 training direction？
    # ========================================================

    sample_loss = (
        F.cross_entropy(
            scores,
            target.view(-1)
        )
    )

    sample_gradient = (
        get_flat_gradients(
            sample_loss,
            probe_parameters
        )
    )

    return (
        sample_gradient,
        sample_loss.item()
    )

# ============================================================
# Experiment F-E：Reward Calculation
# ============================================================

def compare_f_gradients(
    meta_gradient_info,
    sample_gradient
):
    """
    比較：

        g_meta
        vs
        g_i^a

    同時輸出：
        raw dot
        cosine
        normalized-meta utility
    """

    sample_norm = (
        torch.norm(
            sample_gradient,
            p=2
        )
    )

    results = {}

    for (
        objective_name,
        meta_info
    ) in meta_gradient_info.items():

        meta_gradient = (
            meta_info[
                'gradient'
            ]
        )

        normalized_meta_gradient = (
            meta_info[
                'normalized_gradient'
            ]
        )

        # ====================================================
        # Raw dot product
        # 只做 diagnostic
        # ====================================================

        dot_product = (
            torch.dot(
                meta_gradient,
                sample_gradient
            )
        )

        # ====================================================
        # Cosine
        # 只看方向
        # ====================================================

        cosine_similarity = (
            dot_product
            /
            (
                torch.norm(
                    meta_gradient,
                    p=2
                )
                *
                sample_norm
                +
                1e-12
            )
        )

        # ====================================================
        # Experiment F primary Reward
        #
        # normalize g_meta，
        # 但保留 action gradient magnitude
        # ====================================================

        utility = (
            torch.dot(
                normalized_meta_gradient,
                sample_gradient
            )
        )

        results[
            objective_name
        ] = {

            'utility':
                utility.item(),

            'dot':
                dot_product.item(),

            'cosine':
                cosine_similarity.item(),

            'sample_grad_norm':
                sample_norm.item(),

            'meta_grad_norm':
                meta_info['norm']
        }

    return results

# ============================================================
# Experiment F-E：Offline Reward Validation
# ============================================================

def run_experiment_f_e_probe(
    model_joint,
    inner_train_data,
    meta_data,
    popularity_reference_data,
    args,
    logger
):
    """
    Experiment F-E：

    State:
        Low / Medium / High Sequence Popularity

    Action:
        D2

    Strength:
        p = 0.1 / 0.2 / 0.4

    Meta Objective:
        M0 / M1 / M2

    Output:
        per-state × per-strength reward table
    """

    # ========================================================
    # Popularity definition
    # 保持跟 Experiment E 完全一致
    # ========================================================

    (
        popular_items,
        unpopular_seen_items,
        _,
        _
    ) = build_item_popularity_groups(
        popularity_reference_data,
        popular_ratio=
            args.popular_ratio
    )

    # ========================================================
    # Build Probe Samples
    # ========================================================

    probe_data = (
        build_f_e_probe_samples(
            train_data=
                inner_train_data,

            popular_items=
                popular_items,

            unpopular_seen_items=
                unpopular_seen_items,

            low_threshold=
                args.e_context_low_threshold,

            high_threshold=
                args.e_context_high_threshold,

            samples_per_state=
                args.f_probe_samples,

            random_seed=
                args.random_seed
                + 1000000
        )
    )

    probe_info = {
        state:
        len(data)
        for state, data
        in probe_data.items()
    }

    print(
        'Experiment F-E Probe Samples'
        '---------------------------------------------'
    )

    print(probe_info)

    logger.info(
        'Experiment F-E Probe Samples'
    )

    logger.info(
        probe_info
    )

    # ========================================================
    # Build Meta Gradients
    # ========================================================

    (
        meta_gradient_info,
        _
    ) = build_f_meta_gradients(
        model_joint=
            model_joint,

        meta_data=
            meta_data,

        popular_items=
            popular_items,

        args=args
    )

    print(
        'Experiment F-E Meta Gradient'
        '---------------------------------------------'
    )

    for objective_name, info in (
        meta_gradient_info.items()
    ):

        print({
            'Objective':
                objective_name,

            'Meta Loss':
                round(
                    info['loss'],
                    6
                ),

            'Meta Grad Norm':
                round(
                    info['norm'],
                    6
                )
        })

    # ========================================================
    # Probe
    # ========================================================

    strengths = [
        0.1,
        0.2,
        0.4
    ]

    states = [
        'low',
        'medium',
        'high'
    ]

    records = []

    for state_index, state in enumerate(states):

        state_df = (
            probe_data[
                state
            ]
        )

        for sample_index in range(
            len(state_df)
        ):

            original_df = (
                state_df
                .iloc[
                    [sample_index]
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            original_seq = (
                original_df
                .iloc[0]['seq']
            )

            seq_pop_ratio = (
                calculate_sequence_popular_ratio(
                    original_seq,
                    popular_items
                )
            )

            # ================================================
            # 每個 repeat 共用 seed
            #
            # p=.1/.2/.4 使用同一 random stream，
            # 比較才公平。
            # ================================================

            for repeat_index in range(
                args.f_probe_repeats
            ):

                probe_seed = (
                    args.random_seed
                    + 1100000
                    + state_index
                      * 100000
                    + sample_index
                      * 100
                    + repeat_index
                )

                for strength in strengths:

                    (
                        regulated_df,
                        regulation_stats
                    ) = (
                        rule_based_tail_context_regulation(
                            batch_df=
                                original_df,

                            popular_items=
                                popular_items,

                            unpopular_seen_items=
                                unpopular_seen_items,

                            context_state=
                                state,

                            low_threshold=
                                args.e_context_low_threshold,

                            high_threshold=
                                args.e_context_high_threshold,

                            max_len=
                                args.max_len,

                            # 保持 Experiment E 設定
                            augmentation_probability=
                                args.tail_aug_probability,

                            drop_probability=
                                strength,

                            preserve_recent=
                                args.tail_preserve_recent,

                            random_seed=
                                probe_seed
                        )
                    )

                    # ----------------------------------------
                    # Model forward 的 seed
                    # 也固定，不依 strength 改變
                    # ----------------------------------------

                    model_seed = (
                        probe_seed
                        + 500000
                    )

                    (
                        sample_gradient,
                        sample_loss
                    ) = (
                        compute_f_sample_gradient(
                            model_joint=
                                model_joint,

                            regulated_df=
                                regulated_df,

                            args=args,

                            random_seed=
                                model_seed
                        )
                    )

                    reward_result = (
                        compare_f_gradients(
                            meta_gradient_info=
                                meta_gradient_info,

                            sample_gradient=
                                sample_gradient
                        )
                    )

                    for (
                        objective_name,
                        reward_info
                    ) in reward_result.items():

                        records.append({

                            'meta_objective':
                                objective_name,

                            'state':
                                state,

                            'sample_index':
                                sample_index,

                            'seq_pop_ratio':
                                seq_pop_ratio,

                            'strength':
                                strength,

                            'repeat':
                                repeat_index,

                            'sample_loss':
                                sample_loss,

                            'utility':
                                reward_info[
                                    'utility'
                                ],

                            'dot':
                                reward_info[
                                    'dot'
                                ],

                            'cosine':
                                reward_info[
                                    'cosine'
                                ],

                            'sample_grad_norm':
                                reward_info[
                                    'sample_grad_norm'
                                ],

                            'meta_grad_norm':
                                reward_info[
                                    'meta_grad_norm'
                                ],

                            'augmented':
                                regulation_stats[
                                    'augmented_samples'
                                ],

                            'dropped_items':
                                regulation_stats[
                                    'dropped_items'
                                ]
                        })

    # ========================================================
    # Aggregate
    # ========================================================

    result_df = pd.DataFrame(
        records
    )

    summary_df = (
        result_df
        .groupby(
            [
                'meta_objective',
                'state',
                'strength'
            ],
            as_index=False
        )
        .agg(
            utility_mean=(
                'utility',
                'mean'
            ),

            utility_std=(
                'utility',
                'std'
            ),

            cosine_mean=(
                'cosine',
                'mean'
            ),

            sample_grad_norm_mean=(
                'sample_grad_norm',
                'mean'
            ),

            sample_loss_mean=(
                'sample_loss',
                'mean'
            ),

            augmentation_rate=(
                'augmented',
                'mean'
            ),

            dropped_items_mean=(
                'dropped_items',
                'mean'
            )
        )
    )

    # ============================================================
    # Experiment F-E：
    # Paired Strength Comparison
    # ============================================================

    paired_df = (
        result_df
        .pivot_table(
            index=[
                'meta_objective',
                'state',
                'sample_index',
                'repeat'
            ],

            columns='strength',

            values='utility',

            aggfunc='mean'
        )
        .reset_index()
    )

    # 確保三個 strength 都存在
    required_strengths = [
        0.1,
        0.2,
        0.4
    ]

    if all(
        strength in paired_df.columns
        for strength
        in required_strengths
    ):

        paired_df[
            'delta_02_vs_01'
        ] = (
            paired_df[0.2]
            - paired_df[0.1]
        )

        paired_df[
            'delta_04_vs_01'
        ] = (
            paired_df[0.4]
            - paired_df[0.1]
        )

        paired_df[
            'delta_02_vs_04'
        ] = (
            paired_df[0.2]
            - paired_df[0.4]
        )

        paired_summary_df = (
            paired_df
            .groupby(
                [
                    'meta_objective',
                    'state'
                ],
                as_index=False
            )
            .agg(
                delta_02_vs_01_mean=(
                    'delta_02_vs_01',
                    'mean'
                ),

                delta_02_vs_01_std=(
                    'delta_02_vs_01',
                    'std'
                ),

                delta_02_vs_04_mean=(
                    'delta_02_vs_04',
                    'mean'
                ),

                delta_02_vs_04_std=(
                    'delta_02_vs_04',
                    'std'
                ),

                delta_04_vs_01_mean=(
                    'delta_04_vs_01',
                    'mean'
                )
            )
        )

        # --------------------------------------------------------
        # Win Rate
        # --------------------------------------------------------

        win_rate_rows = []

        for (
            objective_name,
            state
        ), group in paired_df.groupby(
            [
                'meta_objective',
                'state'
            ]
        ):

            win_rate_rows.append({

                'meta_objective':
                    objective_name,

                'state':
                    state,

                'p02_beats_p01':
                    (
                        group[
                            'delta_02_vs_01'
                        ]
                        > 0
                    ).mean(),

                'p02_beats_p04':
                    (
                        group[
                            'delta_02_vs_04'
                        ]
                        > 0
                    ).mean(),

                'p04_beats_p01':
                    (
                        group[
                            'delta_04_vs_01'
                        ]
                        > 0
                    ).mean()
            })

        win_rate_df = pd.DataFrame(
            win_rate_rows
        )

        paired_summary_df = (
            paired_summary_df
            .merge(
                win_rate_df,
                on=[
                    'meta_objective',
                    'state'
                ],
                how='left'
            )
        )
    print(
        '\nExperiment F-E Paired Reward Comparison'
        '---------------------------------------------'
    )

    print(
        paired_summary_df.to_string(
            index=False
        )
    )
    # ========================================================
    # Print
    # ========================================================

    for objective_name in [
        'M0_balanced',
        'M1_margin',
        'M2_popmass'
    ]:

        objective_summary = (
            summary_df[
                summary_df[
                    'meta_objective'
                ]
                ==
                objective_name
            ]
        )

        print(
            '\nExperiment F-E Reward Summary'
            '---------------------------------------------'
        )

        print(
            'Meta Objective:',
            objective_name
        )

        print(
            objective_summary[
                [
                    'state',
                    'strength',
                    'utility_mean',
                    'utility_std',
                    'cosine_mean',
                    'augmentation_rate',
                    'dropped_items_mean'
                ]
            ].to_string(
                index=False
            )
        )

        logger.info(
            'Experiment F-E Reward Summary'
        )

        logger.info(
            objective_name
        )

        logger.info(
            objective_summary.to_dict(
                orient='records'
            )
        )

    # ========================================================
    # Save raw + summary
    # ========================================================

    output_dir = os.path.join(
        args.log_file,
        args.s_dataset
        + '_'
        + args.t_dataset
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = time.strftime(
        "%Y-%m-%d_%H-%M-%S",
        time.localtime()
    )

    raw_path = os.path.join(
        output_dir,
        'experiment_f_e_raw_'
        + timestamp
        + '.csv'
    )

    summary_path = os.path.join(
        output_dir,
        'experiment_f_e_summary_'
        + timestamp
        + '.csv'
    )

    paired_summary_path = os.path.join(
        output_dir,
        'experiment_f_e_paired_summary_'
        + timestamp
        + '.csv'
    )

    paired_summary_df.to_csv(
        paired_summary_path,
        index=False
    )

    print(
        'Experiment F-E paired summary saved at:',
        paired_summary_path
    )
    
    result_df.to_csv(
        raw_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        '\nExperiment F-E raw result saved at:',
        raw_path
    )

    print(
        'Experiment F-E summary saved at:',
        summary_path
    )

    return summary_df

# ============================================================
# Experiment F-E v4：
# Virtual One-Step Reward Validation
# ============================================================

def run_experiment_f_e_virtual_probe(
    model_joint,
    inner_train_data,
    meta_data,
    popularity_reference_data,
    args,
    logger
):
    """
    State:
        Low / Medium / High

    Action:
        D2

    Strength:
        p = 0.1 / 0.2 / 0.4

    Reward:
        MetaLoss(theta)
        -
        MetaLoss(theta - eta * grad(L_i^a))
    """

    # ========================================================
    # Popularity definition
    # ========================================================

    (
        popular_items,
        unpopular_seen_items,
        _,
        _
    ) = build_item_popularity_groups(
        popularity_reference_data,
        popular_ratio=
            args.popular_ratio
    )

    # ========================================================
    # Probe Samples
    # ========================================================

    probe_data = (
        build_f_e_probe_samples(
            train_data=
                inner_train_data,

            popular_items=
                popular_items,

            unpopular_seen_items=
                unpopular_seen_items,

            low_threshold=
                args.e_context_low_threshold,

            high_threshold=
                args.e_context_high_threshold,

            samples_per_state=
                args.f_probe_samples,

            random_seed=
                args.random_seed
                + 1000000
        )
    )

    print(
        'Experiment F-E v4 Probe Samples'
        '---------------------------------------------'
    )

    print({
        state:
            len(data)
        for state, data
        in probe_data.items()
    })

    # ========================================================
    # Fixed balanced Meta Batch
    # ========================================================

    meta_batch = (
        sample_balanced_meta_batch(
            meta_data=
                meta_data,

            popular_items=
                popular_items,

            batch_size=
                args.f_meta_batch_size,

            random_seed=
                args.random_seed
                + 700000
        )
    )

    # ========================================================
    # Fixed Meta evaluation randomness
    # ========================================================

    meta_random_seed = (
        args.random_seed
        + 800000
    )

    # ========================================================
    # Baseline Meta Loss:
    # L_meta(theta)
    # ========================================================

    baseline_meta_losses = (
        evaluate_f_meta_losses(
            model_joint=
                model_joint,

            meta_batch=
                meta_batch,

            popular_items=
                popular_items,

            args=args,

            random_seed=
                meta_random_seed
        )
    )

    print(
        'Experiment F-E v4 Baseline Meta Loss'
        '---------------------------------------------'
    )

    print(
        baseline_meta_losses
    )

    logger.info(
        'Experiment F-E v4 Baseline Meta Loss'
    )

    logger.info(
        baseline_meta_losses
    )

    # ========================================================
    # Probe parameter info
    # ========================================================

    (
        probe_parameters_check,
        probe_names,
        probe_count
    ) = get_f_probe_parameters(
        model_joint=
            model_joint,

        probe_space=
            args.f_probe_space
    )

    print(
        'Experiment F-E v4 Probe Space'
        '---------------------------------------------'
    )

    print({
        'Probe Space':
            args.f_probe_space,

        'Parameter Tensors':
            len(
                probe_parameters_check
            ),

        'Total Parameters':
            probe_count,

        'Virtual LR':
            args.f_virtual_lr
    })

    # ========================================================
    # Offline Action Probe
    # ========================================================

    strengths = [
        0.1,
        0.2,
        0.4
    ]

    states = [
        'low',
        'medium',
        'high'
    ]

    records = []

    for (
        state_index,
        state
    ) in enumerate(states):

        state_df = (
            probe_data[state]
        )

        for sample_index in range(
            len(state_df)
        ):

            original_df = (
                state_df
                .iloc[
                    [sample_index]
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            original_seq = (
                original_df
                .iloc[0]['seq']
            )

            seq_pop_ratio = (
                calculate_sequence_popular_ratio(
                    original_seq,
                    popular_items
                )
            )

            for repeat_index in range(
                args.f_probe_repeats
            ):

                probe_seed = (
                    args.random_seed
                    + 1100000
                    + state_index
                      * 100000
                    + sample_index
                      * 100
                    + repeat_index
                )

                for strength in strengths:

                    # ========================================
                    # D2 Action
                    # ========================================

                    (
                        regulated_df,
                        regulation_stats
                    ) = (
                        rule_based_tail_context_regulation(
                            batch_df=
                                original_df,

                            popular_items=
                                popular_items,

                            unpopular_seen_items=
                                unpopular_seen_items,

                            context_state=
                                state,

                            low_threshold=
                                args.e_context_low_threshold,

                            high_threshold=
                                args.e_context_high_threshold,

                            max_len=
                                args.max_len,

                            # v4:
                            # action 已經被選擇，
                            # 所以必定執行 D2
                            augmentation_probability=
                                1.0,

                            drop_probability=
                                strength,

                            preserve_recent=
                                args.tail_preserve_recent,

                            random_seed=
                                probe_seed
                        )
                    )

                    model_seed = (
                        probe_seed
                        + 500000
                    )

                    # ========================================
                    # g_i^a
                    # ========================================

                    (
                        probe_parameters,
                        sample_gradients,
                        sample_loss,
                        sample_grad_norm
                    ) = (
                        compute_f_sample_parameter_gradients(
                            model_joint=
                                model_joint,

                            regulated_df=
                                regulated_df,

                            args=args,

                            random_seed=
                                model_seed
                        )
                    )

                    # ========================================
                    # Virtual update + Meta Reward
                    # ========================================

                    reward_result = (
                        compute_f_virtual_meta_reward(
                            model_joint=
                                model_joint,

                            probe_parameters=
                                probe_parameters,

                            sample_gradients=
                                sample_gradients,

                            baseline_meta_losses=
                                baseline_meta_losses,

                            meta_batch=
                                meta_batch,

                            popular_items=
                                popular_items,

                            args=args,

                            meta_random_seed=
                                meta_random_seed
                        )
                    )

                    for (
                        objective_name,
                        reward_info
                    ) in (
                        reward_result.items()
                    ):

                        records.append({

                            'meta_objective':
                                objective_name,

                            'state':
                                state,

                            'sample_index':
                                sample_index,

                            'seq_pop_ratio':
                                seq_pop_ratio,

                            'strength':
                                strength,

                            'repeat':
                                repeat_index,

                            'sample_loss':
                                sample_loss,

                            'sample_grad_norm':
                                sample_grad_norm,

                            'reward':
                                reward_info[
                                    'reward'
                                ],

                            'meta_before':
                                reward_info[
                                    'meta_before'
                                ],

                            'meta_after':
                                reward_info[
                                    'meta_after'
                                ],

                            'meta_delta':
                                reward_info[
                                    'meta_delta'
                                ],

                            'virtual_step_norm':
                                reward_info[
                                    'virtual_step_norm'
                                ],

                            'relative_step':
                                reward_info[
                                    'relative_step'
                                ],

                            'augmented':
                                regulation_stats[
                                    'augmented_samples'
                                ],

                            'dropped_items':
                                regulation_stats[
                                    'dropped_items'
                                ]
                        })

    # ========================================================
    # DataFrame
    # ========================================================

    result_df = pd.DataFrame(
        records
    )

    # ========================================================
    # Aggregate
    # ========================================================

    summary_df = (
        result_df
        .groupby(
            [
                'meta_objective',
                'state',
                'strength'
            ],
            as_index=False
        )
        .agg(

            reward_mean=(
                'reward',
                'mean'
            ),

            reward_std=(
                'reward',
                'std'
            ),

            meta_after_mean=(
                'meta_after',
                'mean'
            ),

            sample_grad_norm_mean=(
                'sample_grad_norm',
                'mean'
            ),

            virtual_step_norm_mean=(
                'virtual_step_norm',
                'mean'
            ),

            relative_step_mean=(
                'relative_step',
                'mean'
            ),

            dropped_items_mean=(
                'dropped_items',
                'mean'
            )
        )
    )

    # ========================================================
    # Paired comparison
    # ========================================================

    paired_df = (
        result_df
        .pivot_table(
            index=[
                'meta_objective',
                'state',
                'sample_index',
                'repeat'
            ],

            columns='strength',

            values='reward',

            aggfunc='mean'
        )
        .reset_index()
    )

    paired_df[
        'delta_02_vs_01'
    ] = (
        paired_df[0.2]
        - paired_df[0.1]
    )

    paired_df[
        'delta_02_vs_04'
    ] = (
        paired_df[0.2]
        - paired_df[0.4]
    )

    paired_df[
        'delta_04_vs_01'
    ] = (
        paired_df[0.4]
        - paired_df[0.1]
    )

    paired_rows = []

    for (
        objective_name,
        state
    ), group in paired_df.groupby(
        [
            'meta_objective',
            'state'
        ]
    ):

        paired_rows.append({

            'meta_objective':
                objective_name,

            'state':
                state,

            'delta_02_vs_01_mean':
                group[
                    'delta_02_vs_01'
                ].mean(),

            'delta_02_vs_04_mean':
                group[
                    'delta_02_vs_04'
                ].mean(),

            'delta_04_vs_01_mean':
                group[
                    'delta_04_vs_01'
                ].mean(),

            'p02_beats_p01':
                (
                    group[
                        'delta_02_vs_01'
                    ]
                    > 0
                ).mean(),

            'p02_beats_p04':
                (
                    group[
                        'delta_02_vs_04'
                    ]
                    > 0
                ).mean(),

            'p04_beats_p01':
                (
                    group[
                        'delta_04_vs_01'
                    ]
                    > 0
                ).mean()
        })

    paired_summary_df = (
        pd.DataFrame(
            paired_rows
        )
    )

    # ========================================================
    # Print results
    # ========================================================

    for objective_name in [
        'M0_balanced',
        'M1_margin',
        'M2_popmass'
    ]:

        print(
            '\nExperiment F-E v4 Reward Summary'
            '---------------------------------------------'
        )

        print(
            'Meta Objective:',
            objective_name
        )

        print(
            summary_df[
                summary_df[
                    'meta_objective'
                ]
                ==
                objective_name
            ][
                [
                    'state',
                    'strength',
                    'reward_mean',
                    'reward_std',
                    'meta_after_mean',
                    'virtual_step_norm_mean',
                    'relative_step_mean',
                    'dropped_items_mean'
                ]
            ].to_string(
                index=False
            )
        )

    print(
        '\nExperiment F-E v4 Paired Reward Comparison'
        '---------------------------------------------'
    )

    print(
        paired_summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    output_dir = os.path.join(
        args.log_file,
        args.s_dataset
        + '_'
        + args.t_dataset
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = time.strftime(
        "%Y-%m-%d_%H-%M-%S",
        time.localtime()
    )

    raw_path = os.path.join(
        output_dir,
        'experiment_f_e_v4_raw_'
        + timestamp
        + '.csv'
    )

    summary_path = os.path.join(
        output_dir,
        'experiment_f_e_v4_summary_'
        + timestamp
        + '.csv'
    )

    paired_path = os.path.join(
        output_dir,
        'experiment_f_e_v4_paired_'
        + timestamp
        + '.csv'
    )

    result_df.to_csv(
        raw_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    paired_summary_df.to_csv(
        paired_path,
        index=False
    )

    print(
        '\nExperiment F-E v4 raw saved at:',
        raw_path
    )

    print(
        'Experiment F-E v4 summary saved at:',
        summary_path
    )

    print(
        'Experiment F-E v4 paired saved at:',
        paired_path
    )

    return (
        summary_df,
        paired_summary_df
    )

# ============================================================
# Experiment F-E V5：
# One training-aligned virtual batch update
# ============================================================

def compute_f_v5_batch_update(
    model_joint,
    batch_df,
    meta_batch,
    popular_items,
    args,
    model_random_seed,
    meta_random_seed
):
    """
    從同一個 Reference Model 複製一份 temporary model。

    對 batch 做一次真正的 Target Stage-2 style update：

        forward
        -> loss_diffu_ce
        -> Adam
        -> optimizer.step()

    然後評估 update 後的 M0/M1/M2 Meta Loss。

    Reference Model 本身完全不修改。
    """

    device = args.device

    # --------------------------------------------------------
    # 每個 Action 都從相同 Reference Model 開始
    # --------------------------------------------------------

    virtual_model = (
        copy.deepcopy(
            model_joint
        )
        .to(device)
    )

    core_model = (
        virtual_model.module
        if isinstance(
            virtual_model,
            nn.DataParallel
        )
        else virtual_model
    )

    # --------------------------------------------------------
    # 與正式 training 相同：Adam + all parameters
    # --------------------------------------------------------

    virtual_optimizer = optim.Adam(
        virtual_model.parameters(),
        lr=args.f_v5_lr,
        weight_decay=args.weight_decay
    )

    seq = torch.LongTensor(
        batch_df['seq'].tolist()
    ).to(device)

    target = (
        torch.LongTensor(
            batch_df['next'].tolist()
        )
        .unsqueeze(1)
        .to(device)
    )

    set_f_probe_seed(
        model_random_seed
    )

    virtual_model.train()

    virtual_optimizer.zero_grad()

    (
        _,
        diffu_rep,
        _,
        _,
        _,
        _,
        em_loss
    ) = virtual_model(
        seq,
        target,
        None,

        False,  # Target Stage-2
        args,
        0,

        train_flag=True
    )

    # ========================================================
    # 完全照 model_train() Stage-2 loss 寫法
    # ========================================================

    loss_diffu_value = (
        core_model.loss_diffu_ce(
            diffu_rep,
            target,
            False
        )
    )

    loss_all = (
        loss_diffu_value
        + em_loss
        * args.loss_lambda
    )

    loss_all.backward()

    # --------------------------------------------------------
    # gradient norm：只當 diagnostic
    # --------------------------------------------------------

    grad_squared_norm = 0.0

    for parameter in (
        virtual_model.parameters()
    ):

        if parameter.grad is not None:

            grad_squared_norm += (
                parameter.grad
                .detach()
                .pow(2)
                .sum()
                .item()
            )

    grad_norm = (
        grad_squared_norm
        ** 0.5
    )

    virtual_optimizer.step()

    # ========================================================
    # Update 後 Meta Loss
    # ========================================================

    updated_meta_losses = (
        evaluate_f_meta_losses(
            model_joint=
                virtual_model,

            meta_batch=
                meta_batch,

            popular_items=
                popular_items,

            args=args,

            random_seed=
                meta_random_seed
        )
    )

    training_loss = (
        loss_all
        .detach()
        .item()
    )

    del virtual_optimizer
    del virtual_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'training_loss':
            training_loss,

        'grad_norm':
            grad_norm,

        'meta_losses':
            updated_meta_losses
    }

# ============================================================
# Experiment F-E V5：
# Training-aligned Mixed-Batch Virtual Reward
# ============================================================

def run_experiment_f_e_v5_batch_probe(
    model_joint,
    inner_train_data,
    meta_data,
    popularity_reference_data,
    args,
    logger
):
    """
    V5:

    同一個 mixed Target batch：

        No-Reg
        vs
        Low/Medium/High × p=.1/.2/.4

    每個 Action：
        full-model Adam update

    Reward：
        Regulated Meta improvement
        -
        No-Reg Meta improvement
    """

    (
        popular_items,
        unpopular_seen_items,
        _,
        _
    ) = build_item_popularity_groups(
        popularity_reference_data,
        popular_ratio=
            args.popular_ratio
    )

    # ========================================================
    # Fixed Meta Batch
    # ========================================================

    meta_batch = (
        sample_balanced_meta_batch(
            meta_data=
                meta_data,

            popular_items=
                popular_items,

            batch_size=
                args.f_meta_batch_size,

            random_seed=
                args.random_seed
                + 700000
        )
    )

    meta_random_seed = (
        args.random_seed
        + 800000
    )

    baseline_meta_losses = (
        evaluate_f_meta_losses(
            model_joint=
                model_joint,

            meta_batch=
                meta_batch,

            popular_items=
                popular_items,

            args=args,

            random_seed=
                meta_random_seed
        )
    )

    print(
        'Experiment F-E V5 Baseline Meta Loss'
        '---------------------------------------------'
    )

    print(
        baseline_meta_losses
    )

    states = [
        'low',
        'medium',
        'high'
    ]

    strengths = [
        0.1,
        0.2,
        0.4
    ]

    records = []

    # ========================================================
    # 每個 repeat 使用一個相同 mixed batch
    # ========================================================

    for repeat_index in range(
        args.f_v5_repeats
    ):

        batch_seed = (
            args.random_seed
            + 2000000
            + repeat_index
        )

        base_batch = (
            inner_train_data
            .sample(
                n=args.f_v5_batch_size,
                replace=False,
                random_state=batch_seed
            )
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # 所有 Action 共用完全相同 model stochastic seed
        # ----------------------------------------------------

        model_seed = (
            args.random_seed
            + 2100000
            + repeat_index
        )

        # ====================================================
        # 1. No-Reg baseline update
        # ====================================================

        no_reg_result = (
            compute_f_v5_batch_update(
                model_joint=
                    model_joint,

                batch_df=
                    base_batch,

                meta_batch=
                    meta_batch,

                popular_items=
                    popular_items,

                args=args,

                model_random_seed=
                    model_seed,

                meta_random_seed=
                    meta_random_seed
            )
        )

        no_reg_rewards = {}

        for objective_name in [
            'M0_balanced',
            'M1_margin',
            'M2_popmass'
        ]:

            no_reg_rewards[
                objective_name
            ] = (
                baseline_meta_losses[
                    objective_name
                ]
                -
                no_reg_result[
                    'meta_losses'
                ][
                    objective_name
                ]
            )

        # ====================================================
        # 2. State × Strength
        # ====================================================

        for (
            state_index,
            state
        ) in enumerate(states):

            regulation_seed = (
                args.random_seed
                + 2200000
                + repeat_index * 100
                + state_index
            )

            for strength in strengths:

                (
                    regulated_batch,
                    d2_stats
                ) = (
                    rule_based_tail_context_regulation(
                        batch_df=
                            base_batch,

                        popular_items=
                            popular_items,

                        unpopular_seen_items=
                            unpopular_seen_items,

                        context_state=
                            state,

                        low_threshold=
                            args.e_context_low_threshold,

                        high_threshold=
                            args.e_context_high_threshold,

                        max_len=
                            args.max_len,

                        # 完全回到 Experiment E training setting
                        augmentation_probability=
                            args.f_v5_aug_probability,

                        drop_probability=
                            strength,

                        preserve_recent=
                            args.tail_preserve_recent,

                        # 同 state 不同 strength
                        # 使用同一 random stream
                        random_seed=
                            regulation_seed
                    )
                )

                action_result = (
                    compute_f_v5_batch_update(
                        model_joint=
                            model_joint,

                        batch_df=
                            regulated_batch,

                        meta_batch=
                            meta_batch,

                        popular_items=
                            popular_items,

                        args=args,

                        # 跟 No-Reg 完全相同
                        model_random_seed=
                            model_seed,

                        meta_random_seed=
                            meta_random_seed
                    )
                )

                for objective_name in [
                    'M0_balanced',
                    'M1_margin',
                    'M2_popmass'
                ]:

                    raw_reward = (
                        baseline_meta_losses[
                            objective_name
                        ]
                        -
                        action_result[
                            'meta_losses'
                        ][
                            objective_name
                        ]
                    )

                    incremental_reward = (
                        raw_reward
                        -
                        no_reg_rewards[
                            objective_name
                        ]
                    )

                    records.append({
                        'repeat':
                            repeat_index,

                        'meta_objective':
                            objective_name,

                        'state':
                            state,

                        'strength':
                            strength,

                        'raw_reward':
                            raw_reward,

                        'no_reg_reward':
                            no_reg_rewards[
                                objective_name
                            ],

                        'incremental_reward':
                            incremental_reward,

                        'training_loss':
                            action_result[
                                'training_loss'
                            ],

                        'grad_norm':
                            action_result[
                                'grad_norm'
                            ],

                        'eligible_samples':
                            d2_stats[
                                'eligible_state_unpopnext'
                            ],

                        'augmented_samples':
                            d2_stats[
                                'augmented_samples'
                            ],

                        'dropped_items':
                            d2_stats[
                                'dropped_items'
                            ],

                        'augmentation_ratio':
                            d2_stats[
                                'augmentation_ratio_among_eligible'
                            ]
                    })
                result_df = pd.DataFrame(
                    records
                )

                summary_df = (
                    result_df
                    .groupby(
                        [
                            'meta_objective',
                            'state',
                            'strength'
                        ],
                        as_index=False
                    )
                    .agg(
                        incremental_reward_mean=(
                            'incremental_reward',
                            'mean'
                        ),

                        incremental_reward_std=(
                            'incremental_reward',
                            'std'
                        ),

                        raw_reward_mean=(
                            'raw_reward',
                            'mean'
                        ),

                        no_reg_reward_mean=(
                            'no_reg_reward',
                            'mean'
                        ),

                        eligible_samples_mean=(
                            'eligible_samples',
                            'mean'
                        ),

                        augmented_samples_mean=(
                            'augmented_samples',
                            'mean'
                        ),

                        dropped_items_mean=(
                            'dropped_items',
                            'mean'
                        )
                    )
                )

                print(
                    '\nExperiment F-E V5 Summary'
                    '---------------------------------------------'
                )

                for objective_name in [
                    'M0_balanced',
                    'M1_margin',
                    'M2_popmass'
                ]:

                    print(
                        '\nMeta Objective:',
                        objective_name
                    )

                    print(
                        summary_df[
                            summary_df[
                                'meta_objective'
                            ]
                            ==
                            objective_name
                        ].to_string(
                            index=False
                        )
                    )
                    paired_df = (
                        result_df
                        .pivot_table(
                            index=[
                                'meta_objective',
                                'state',
                                'repeat'
                            ],

                            columns='strength',

                            values='incremental_reward',

                            aggfunc='mean'
                        )
                        .reset_index()
                    )

                    paired_df[
                        'delta_02_vs_01'
                    ] = (
                        paired_df[0.2]
                        - paired_df[0.1]
                    )

                    paired_df[
                        'delta_02_vs_04'
                    ] = (
                        paired_df[0.2]
                        - paired_df[0.4]
                    )

                    paired_df[
                        'delta_04_vs_01'
                    ] = (
                        paired_df[0.4]
                        - paired_df[0.1]
                    )

                    paired_summary = (
                        paired_df
                        .groupby(
                            [
                                'meta_objective',
                                'state'
                            ],
                            as_index=False
                        )
                        .agg(
                            delta_02_vs_01_mean=(
                                'delta_02_vs_01',
                                'mean'
                            ),

                            delta_02_vs_04_mean=(
                                'delta_02_vs_04',
                                'mean'
                            ),

                            delta_04_vs_01_mean=(
                                'delta_04_vs_01',
                                'mean'
                            )
                        )
                    )

                    print(
                        '\nExperiment F-E V5 Paired Comparison'
                        '---------------------------------------------'
                    )

                    print(
                        paired_summary.to_string(
                            index=False
                        )
                    )

                    return (
                        summary_df,
                        paired_summary
                    )
            
# ============================================================
# Experiment E：State Distribution Inspection
# ============================================================

def analyze_e_sequence_popularity_distribution(
    train_data,
    popular_ratio=0.2,
    logger=None
):
    """
    Experiment E 前置檢查：

    只觀察 Target training data 中
    Next = Unpopular 的 samples，

    統計它們的 Sequence Popular Ratio 分布。

    這一步只做 State diagnosis，
    不做 dropout、不修改 training sample。
    """

    (
        popular_items,
        unpopular_seen_items,
        observed_train_items,
        _
    ) = build_item_popularity_groups(
        train_data,
        popular_ratio=popular_ratio
    )

    ratio_counter = Counter()
    valid_length_counter = Counter()

    unpopular_next_samples = 0

    for _, row in train_data.iterrows():

        target_item = int(row['next'])

        # ----------------------------------------------------
        # Experiment E controlled condition:
        # 只觀察 Next = Unpopular
        # ----------------------------------------------------
        if target_item not in unpopular_seen_items:
            continue

        unpopular_next_samples += 1

        seq = list(row['seq'])

        # Sequence Popular Ratio
        seq_pop_ratio = calculate_sequence_popular_ratio(
            seq,
            popular_items
        )

        # 避免 floating point 造成相同 ratio 被拆成不同 key
        rounded_ratio = round(seq_pop_ratio, 6)

        ratio_counter[rounded_ratio] += 1

        # 同時記錄 valid sequence length
        # 用來解釋為什麼 ratio 不一定只有 0.125 的倍數
        valid_length = sum(
            1
            for item in seq
            if int(item) != 0
        )

        valid_length_counter[valid_length] += 1

    # ========================================================
    # Output
    # ========================================================

    lines = []

    lines.append(
        "Experiment E State Distribution"
        "---------------------------------------------"
    )

    lines.append(
        f"Target training samples: {len(train_data)}"
    )

    lines.append(
        f"Observed train items: {len(observed_train_items)}"
    )

    lines.append(
        f"Popular items: {len(popular_items)}"
    )

    lines.append(
        f"Unpopular-seen items: {len(unpopular_seen_items)}"
    )

    lines.append(
        f"Unpopular-Next samples: {unpopular_next_samples}"
    )

    lines.append("")
    lines.append(
        "Sequence Popular Ratio Distribution "
        "(Next = Unpopular)"
    )
    lines.append(
        "---------------------------------------------"
    )

    for ratio in sorted(ratio_counter.keys()):

        count = ratio_counter[ratio]

        percentage = (
            count / unpopular_next_samples * 100
            if unpopular_next_samples > 0
            else 0.0
        )

        lines.append(
            f"ratio = {ratio:.6f} "
            f"-> {count} samples "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # 額外印 valid sequence length
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "Valid Sequence Length Distribution"
    )
    lines.append(
        "---------------------------------------------"
    )

    for length in sorted(valid_length_counter.keys()):

        count = valid_length_counter[length]

        percentage = (
            count / unpopular_next_samples * 100
            if unpopular_next_samples > 0
            else 0.0
        )

        lines.append(
            f"length = {length} "
            f"-> {count} samples "
            f"({percentage:.2f}%)"
        )

    # print + log
    for line in lines:
        print(line)

        if logger is not None:
            logger.info(line)

    return {
        'ratio_distribution': dict(
            sorted(ratio_counter.items())
        ),
        'valid_length_distribution': dict(
            sorted(valid_length_counter.items())
        ),
        'unpopular_next_samples': unpopular_next_samples,
        'popular_items': len(popular_items),
        'unpopular_seen_items': len(unpopular_seen_items)
    }

def pad_or_truncate_sequence(seq, max_len):
    """
    移除 padding 0，
    保留最近 max_len 個 interaction，
    最後重新 left padding。
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

def dropout_older_context(
    seq,
    max_len,
    drop_probability,
    preserve_recent,
    rng
):
    """
    只對較早的歷史 context 做 dropout。

    - 不修改 next
    - 不改 interaction order
    - 最近 preserve_recent 個 interaction 一定保留
    """

    if not 0.0 <= drop_probability <= 1.0:
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

    # 沒有足夠的較早 context 可以刪
    if len(real_items) <= preserve_recent:
        return pad_or_truncate_sequence(
            real_items,
            max_len
        )

    older_items = (
        real_items[:-preserve_recent]
    )

    recent_items = (
        real_items[-preserve_recent:]
    )

    # 只 dropout 較早 interaction
    kept_older_items = [
        item
        for item in older_items
        if rng.random() >= drop_probability
    ]

    augmented_items = (
        kept_older_items
        + recent_items
    )

    return pad_or_truncate_sequence(
        augmented_items,
        max_len
    )

def rule_based_tail_context_regulation(
    batch_df,
    popular_items,
    unpopular_seen_items,
    context_state,
    low_threshold,
    high_threshold,
    max_len,
    augmentation_probability,
    drop_probability,
    preserve_recent,
    random_seed
):
    """
    Experiment E / D2:
    State-dependent Sequence Regulation

    Controlled samples:
        Next = Unpopular

    State:
        Low:
            Sequence Popular Ratio < low_threshold

        Medium:
            low_threshold <= Sequence Popular Ratio < high_threshold

        High:
            Sequence Popular Ratio >= high_threshold

    Strength:
        drop_probability

    Legacy:
        Sequence Popular Ratio > low_threshold
        保留原本 D2 行為。
    """

    if not 0.0 <= augmentation_probability <= 1.0:
        raise ValueError(
            'tail_aug_probability must be between 0 and 1.'
        )
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError(
            'Experiment E thresholds must satisfy '
            '0 <= low_threshold < high_threshold <= 1.'
        )
    
    batch_df = (
        batch_df
        .copy()
        .reset_index(drop=True)
    )

    rng = random.Random(
        random_seed
    )

    # -------------------------------
    # Statistics
    # -------------------------------

    unpopular_next_samples = 0
    eligible_samples = 0
    augmented_samples = 0

    dropped_item_count = 0

    eligible_pop_ratio_sum = 0.0

    for row_index in batch_df.index:

        target_item = int(
            batch_df.at[
                row_index,
                'next'
            ]
        )

        # ========================================
        # S2: Next-item Popularity
        #
        # Popular Next -> Normal
        # ========================================

        if target_item not in unpopular_seen_items:
            continue

        unpopular_next_samples += 1

        original_seq = list(
            batch_df.at[
                row_index,
                'seq'
            ]
        )

        # ========================================
        # S1: Sequence Popularity
        # ========================================

        seq_pop_ratio = (
            calculate_sequence_popular_ratio(
                original_seq,
                popular_items
            )
        )

        # ========================================
        # Experiment E：State Selection
        # ========================================

        if context_state == 'low':

            state_match = (
                seq_pop_ratio < low_threshold
            )

        elif context_state == 'medium':

            state_match = (
                low_threshold
                <= seq_pop_ratio
                < high_threshold
            )

        elif context_state == 'high':

            state_match = (
                seq_pop_ratio >= high_threshold
            )

        elif context_state == 'legacy':

            # 原本 D2：
            # Popular-context + Unpopular Next
            state_match = (
                seq_pop_ratio > low_threshold
            )

        else:

            raise ValueError(
                f'Unknown Experiment E context state: '
                f'{context_state}'
            )


        if not state_match:
            continue


        eligible_samples += 1

        eligible_pop_ratio_sum += (
            seq_pop_ratio
        )

        # 並非所有 eligible sample 都做 augmentation
        if rng.random() >= augmentation_probability:
            continue

        original_real_length = sum(
            int(item) != 0
            for item in original_seq
        )

        augmented_seq = (
            dropout_older_context(
                seq=original_seq,
                max_len=max_len,
                drop_probability=drop_probability,
                preserve_recent=preserve_recent,
                rng=rng
            )
        )

        augmented_real_length = sum(
            int(item) != 0
            for item in augmented_seq
        )

        batch_df.at[
            row_index,
            'seq'
        ] = augmented_seq

        if 'len_seq' in batch_df.columns:

            batch_df.at[
                row_index,
                'len_seq'
            ] = augmented_real_length

        augmented_samples += 1

        dropped_item_count += max(
            0,
            original_real_length
            - augmented_real_length
        )

    stats = {
        'batch_size':
        len(batch_df),

        'experiment_e_state':
        context_state,

        'low_threshold':
        low_threshold,

        'high_threshold':
        high_threshold,

        'drop_probability':
        drop_probability,

        'unpopular_next_samples':
        unpopular_next_samples,

        'eligible_state_unpopnext':
        eligible_samples,

        'augmented_samples':
        augmented_samples,

        'eligible_ratio':
        round(
            eligible_samples
            / max(len(batch_df), 1),
            4
        ),

        'augmentation_ratio_among_eligible':
        round(
            augmented_samples
            / max(eligible_samples, 1),
            4
        ),

        'avg_eligible_sequence_pop_ratio':
        round(
            eligible_pop_ratio_sum
            / max(eligible_samples, 1),
            4
        ),

        'dropped_items':
        dropped_item_count
    }

    return (
        batch_df,
        stats
    )

def build_context_group_masks(
    seq_batch,
    popular_items,
    threshold,
    device
):
    """
    Popular Ratio > threshold  → Popular-context
    Popular Ratio < threshold  → Tail-context
    Popular Ratio == threshold → Balanced，不放入 group MMD
    """

    ratios = []

    for seq in seq_batch:
        ratio = calculate_sequence_popular_ratio(
            seq,
            popular_items
        )
        ratios.append(ratio)

    ratio_tensor = torch.tensor(
        ratios,
        dtype=torch.float32,
        device=device
    )

    popular_mask = (
        ratio_tensor > threshold
    )

    tail_mask = (
        ratio_tensor < threshold
    )

    return (
        popular_mask,
        tail_mask,
        ratio_tensor
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

def model_train(train_data, val_data, test_data, con_data, model_joint, args, logger, pretrain_flag, popularity_reference_data=None):
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
    # ============================================================
    # D3：Group-aware Alignment
    # ============================================================

    source_popular_items = None
    target_popular_items = None

    if (
        pretrain_flag
        and args.group_alignment == 1
    ):

        (
            source_popular_items,
            _,
            _,
            _
        ) = build_item_popularity_groups(
            train_data,
            popular_ratio=args.popular_ratio
        )

        target_popularity_data = (
            popularity_reference_data
            if popularity_reference_data
            is not None
            else con_data
        )

        (
            target_popular_items,
            _,
            _,
            _
        ) = build_item_popularity_groups(
            target_popularity_data,
            popular_ratio=args.popular_ratio
        )

        group_alignment_info = {
            'group_alignment': True,
            'alignment_mode': args.group_alignment_mode,
            'context_threshold':
                args.group_context_threshold,
            'popular_weight':
                args.group_pop_weight,
            'tail_weight':
                args.group_tail_weight,
            'source_popular_items':
                len(source_popular_items),
            'target_popular_items':
                len(target_popular_items)
        }

        print(
            'Group-aware Alignment Definition'
            '------------------------------------'
        )
        print(group_alignment_info)

        logger.info(
            'Group-aware Alignment Definition'
            '------------------------------------'
        )
        logger.info(group_alignment_info)

    # ============================================================
    # D2: Tail-target Context Regulation
    # Only used in Stage-2 Target training
    # ============================================================

    d2_target_popular_items = None
    d2_target_unpopular_items = None

    if (
        not pretrain_flag
        and args.tail_context_regulation == 1
    ):

        d2_popularity_data = (
            popularity_reference_data
            if popularity_reference_data
            is not None
            else train_data
        )

        (
            d2_target_popular_items,
            d2_target_unpopular_items,
            _,
            _
        ) = build_item_popularity_groups(
            d2_popularity_data,
            popular_ratio=args.popular_ratio
        )

        d2_setup = {
            'Experiment':
            'E',

            'Mechanism':
            'M2 Representation Dominance',

            'Action':
            'D2 Sequence Regulation',

            'Next_condition':
            'Unpopular Next',

            'Context_state':
            args.e_context_state,

            'Low':
            (
                f'SeqPopRatio < '
                f'{args.e_context_low_threshold}'
            ),

            'Medium':
            (
                f'{args.e_context_low_threshold} '
                f'<= SeqPopRatio < '
                f'{args.e_context_high_threshold}'
            ),

            'High':
            (
                f'SeqPopRatio >= '
                f'{args.e_context_high_threshold}'
            ),

            'augmentation_probability':
            args.tail_aug_probability,

            'drop_probability':
            args.tail_drop_probability,

            'preserve_recent':
            args.tail_preserve_recent,

            'popular_items':
            len(d2_target_popular_items),

            'unpopular_seen_items':
            len(d2_target_unpopular_items)
        }

        print(
            'D2 Tail-target Context Regulation'
            '---------------------------------------------'
        )

        print(d2_setup)

        logger.info(
            'D2 Tail-target Context Regulation'
        )

        logger.info(
            d2_setup
        )

    for epoch_temp in range(epochs):

        model_joint.train()
        flag_update = 0
        for j in range(num_batches):
            batch_df = train_data.sample(n=args.batch_size).copy()
            d2_batch_stats = None

            # ========================================================
            # D2 only applies to Stage-2 Target training
            # ========================================================

            if (
                not pretrain_flag
                and args.tail_context_regulation == 1
            ):

                augmentation_seed = (
                    args.random_seed
                    + epoch_temp * num_batches
                    + j
                    + 100000
                )

                (
                    batch_df,
                    d2_batch_stats
                ) = rule_based_tail_context_regulation(
                    batch_df=batch_df,

                    popular_items=
                    d2_target_popular_items,

                    unpopular_seen_items=
                    d2_target_unpopular_items,

                    context_state=
                    args.e_context_state,

                    low_threshold=
                    args.e_context_low_threshold,

                    high_threshold=
                    args.e_context_high_threshold,

                    max_len=
                    args.max_len,

                    augmentation_probability=
                    args.tail_aug_probability,

                    drop_probability=
                    args.tail_drop_probability,

                    preserve_recent=
                    args.tail_preserve_recent,

                    random_seed=
                    augmentation_seed
                )

                # ========================================================
                # D2 Batch Check
                # 只顯示第一個 epoch 的前三個 batch
                # ========================================================

                if (
                    not pretrain_flag
                    and args.tail_context_regulation == 1
                    and epoch_temp == 0
                    and j < 3
                ):

                    print(
                        'D2 Context Regulation Batch Check'
                        '---------------------------------------------'
                    )

                    print(d2_batch_stats)

                    logger.info(
                        'D2 Context Regulation Batch Check'
                    )

                    logger.info(d2_batch_stats)
            seq_list = (
                batch_df['seq'].tolist()
            )

            target_list = (
                batch_df['next'].tolist()
            )
            optimizer.zero_grad()
            seq = torch.LongTensor(seq_list).to(device)
            target = (torch.LongTensor(target_list).unsqueeze(1).to(device))

            # ========================================================
            # Target condition：
            # 維持原本 uniform random sampling
            # ========================================================

            if pretrain_flag:

                con_batch_df = con_data.sample(n=args.batch_size)
                con_seq_list = con_batch_df['seq'].tolist()
                con_seq = torch.LongTensor(con_seq_list).to(device)

                # ====================================================
                # D3：Group-aware Alignment
                # 建立 Source / Target Popular/Tail context masks
                # ====================================================

                if args.group_alignment == 1:

                    (
                        source_pop_mask,
                        source_tail_mask,
                        source_context_ratios
                    ) = build_context_group_masks(
                        seq_list,
                        source_popular_items,
                        args.group_context_threshold,
                        device
                    )

                    (
                        target_pop_mask,
                        target_tail_mask,
                        target_context_ratios
                    ) = build_context_group_masks(
                        con_seq_list,
                        target_popular_items,
                        args.group_context_threshold,
                        device
                    )
                else:
                    source_pop_mask = None
                    source_tail_mask = None
                    target_pop_mask = None
                    target_tail_mask = None
            else:
                con_seq = None
                source_pop_mask = None
                source_tail_mask = None
                target_pop_mask = None
                target_tail_mask = None
            if (
                pretrain_flag
                and args.group_alignment == 1
                and j == 0
            ):

                group_batch_stats = {
                    'Epoch':
                        epoch_temp,

                    'Source Popular-context':
                        int(
                            source_pop_mask
                            .sum()
                            .item()
                        ),

                    'Source Tail-context':
                        int(
                            source_tail_mask
                            .sum()
                            .item()
                        ),

                    'Target Popular-context':
                        int(
                            target_pop_mask
                            .sum()
                            .item()
                        ),

                    'Target Tail-context':
                        int(
                            target_tail_mask
                            .sum()
                            .item()
                        )
                }

                print(
                    'Group-aware Alignment Batch'
                    '------------------------------------'
                )

                print(group_batch_stats)

                logger.info(
                    'Group-aware Alignment Batch'
                    '------------------------------------'
                )

                logger.info(
                    group_batch_stats
                )
            scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis, em_loss = model_joint(seq, target, con_seq, pretrain_flag, args, epoch_temp, train_flag=True,
            source_pop_mask=source_pop_mask, source_tail_mask=source_tail_mask, target_pop_mask=target_pop_mask, target_tail_mask=target_tail_mask)  
            loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, target, pretrain_flag)  ## use this not above        
            loss_all = loss_diffu_value + em_loss*args.loss_lambda
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

        evaluation_popularity_data = (
            popularity_reference_data
            if (
                not pretrain_flag
                and popularity_reference_data
                is not None
            )
            else train_data
        )

        (
            popular_items,
            unpopular_seen_items,
            observed_train_items,
            item_counter
        ) = build_item_popularity_groups(
            evaluation_popularity_data,
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
    
