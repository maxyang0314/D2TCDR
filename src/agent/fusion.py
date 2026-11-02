import torch


def make_popular_lookup(popular_items, item_num, device):
    lookup = torch.zeros(
        item_num + 1,
        dtype=torch.bool,
        device=device
    )
    indices = torch.tensor(
        sorted(popular_items),
        dtype=torch.long,
        device=device
    )
    if indices.numel() > 0:
        lookup[indices] = True
    return lookup


def build_fusion_states(seq, popular_lookup):
    """
    只使用 History。
    State = [HPR, normalized history length]
    """
    valid = seq.ne(0)
    lengths = valid.sum(dim=1)

    popular_count = (
        popular_lookup[seq] & valid
    ).sum(dim=1)

    hpr = (
        popular_count.float()
        / lengths.clamp(min=1).float()
    )

    length_ratio = lengths.float() / seq.size(1)

    return torch.stack(
        [hpr, length_ratio],
        dim=1
    )


@torch.no_grad()
def candidate_scores(model, seq, candidates, args, epoch=0):
    """回傳 [N, C]，只對同一組候選商品評分。"""
    _, rep, _, _, _, _, _ = model(
        seq,
        None,
        None,
        False,
        args,
        epoch,
        train_flag=False
    )
    all_scores = model.diffu_rep_pre(
        rep,
        pretrain_flag=False
    )
    return all_scores.gather(1, candidates)


def fuse_candidate_scores(scores_a, scores_b, betas):
    """每筆 History 使用不同 beta。"""
    pa = torch.softmax(scores_a, dim=1)
    pb = torch.softmax(scores_b, dim=1)

    beta = betas.view(-1, 1)

    return (1.0 - beta) * pa + beta * pb


@torch.no_grad()
def fused_ranking_scores(
    model_a,
    model_b,
    agent,
    seq,
    candidates,
    popular_lookup,
    args,
    epoch=0
):
    """
    Validation/Test:
    每筆 History 選擇 Greedy Beta。
    非候選 Item 分數設為 -100。
    """
    scores_a = candidate_scores(
        model_a, seq, candidates, args, epoch
    )
    scores_b = candidate_scores(
        model_b, seq, candidates, args, epoch
    )

    states = build_fusion_states(seq, popular_lookup)
    betas, _, beta_indices = agent.choose_beta(
        states,
        training=False
    )

    fused = fuse_candidate_scores(
        scores_a, scores_b, betas
    )

    output = torch.full(
        (seq.size(0), args.item_num + 1),
        -100.0,
        device=seq.device,
        dtype=fused.dtype
    )
    output.scatter_(1, candidates, fused)

    return output, beta_indices


def evaluate_fusion_reward(
    model_a,
    model_b,
    agent,
    reward_data,
    args,
    popular_items,
    tail_items,
    popular_lookup,
    epoch=0,
    random_seed=42,
    training=True
):
    """
    Reward:
      - 每筆 Reward Sample 抽一個 beta
      - 所有樣本共用一個 Batch-level Reward
      - 回傳 beta log probabilities 給 REINFORCE

    training=False 時使用 Greedy Beta。
    """
    device = torch.device(args.device)

    was_training_b = model_b.training
    model_a.eval()
    model_b.eval()

    counts = {
        "Overall": 0,
        "Popular": 0,
        "Tail": 0
    }
    hits = {
        "Overall": 0.0,
        "Popular": 0.0,
        "Tail": 0.0
    }

    beta_counts = torch.zeros(
        5, dtype=torch.long
    )
    beta_log_prob_terms = []

    cuda_devices = (
        [torch.cuda.current_device()]
        if device.type == "cuda" else []
    )

    try:
        for start in range(
            0, len(reward_data), args.batch_size
        ):
            df = reward_data.iloc[
                start:start + args.batch_size
            ]

            seq = torch.as_tensor(
                df["seq"].tolist(),
                dtype=torch.long,
                device=device
            )
            target = torch.as_tensor(
                df["next"].tolist(),
                dtype=torch.long,
                device=device
            )
            candidates = torch.as_tensor(
                df["negative_samples"].tolist(),
                dtype=torch.long,
                device=device
            )

            if not torch.all(candidates[:, 0] == target):
                raise ValueError(
                    "Reward positive candidate mismatch"
                )

            # 對 A、B 使用相同且固定的 Diffusion 評估亂數
            # Beta 抽樣在此區塊外，保留正常探索
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(random_seed + start)
                scores_a = candidate_scores(
                    model_a, seq, candidates, args, epoch
                )

            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(random_seed + start)
                scores_b = candidate_scores(
                    model_b, seq, candidates, args, epoch
                )

            # Per-sample Beta
            fusion_states = build_fusion_states(
                seq, popular_lookup
            )

            if training:
                betas, log_probs, indices = (
                    agent.choose_beta(
                        fusion_states,
                        training=True
                    )
                )
                beta_log_prob_terms.append(
                    log_probs.sum()
                )
            else:
                with torch.no_grad():
                    betas, _, indices = agent.choose_beta(
                        fusion_states,
                        training=False
                    )

            beta_counts += torch.bincount(
                indices.detach().cpu(),
                minlength=5
            )

            with torch.no_grad():
                fused = fuse_candidate_scores(
                    scores_a, scores_b, betas
                )

                top10 = fused.topk(
                    k=10, dim=1
                ).indices

                # Reward candidates column 0 = positive
                batch_hits = (
                    top10 == 0
                ).any(dim=1).float()

                popular_mask = torch.tensor(
                    [
                        int(t) in popular_items
                        for t in target.tolist()
                    ],
                    dtype=torch.bool,
                    device=device
                )
                tail_mask = torch.tensor(
                    [
                        int(t) in tail_items
                        for t in target.tolist()
                    ],
                    dtype=torch.bool,
                    device=device
                )

                counts["Overall"] += target.numel()
                hits["Overall"] += batch_hits.sum().item()

                counts["Popular"] += (
                    popular_mask.sum().item()
                )
                hits["Popular"] += (
                    batch_hits[popular_mask].sum().item()
                )

                counts["Tail"] += tail_mask.sum().item()
                hits["Tail"] += (
                    batch_hits[tail_mask].sum().item()
                )

    finally:
        model_b.train(was_training_b)

    if counts["Popular"] == 0 or counts["Tail"] == 0:
        raise ValueError(
            "Reward set needs Popular and Tail samples"
        )

    metrics = {
        "Overall_HR@10": (
            hits["Overall"] / counts["Overall"]
        ),
        "Popular_HR@10": (
            hits["Popular"] / counts["Popular"]
        ),
        "Tail_HR@10": (
            hits["Tail"] / counts["Tail"]
        ),
        "counts": counts
    }

    if training:
        beta_joint_log_prob = torch.stack(
            beta_log_prob_terms
        ).sum()
    else:
        beta_joint_log_prob = None

    return (
        metrics,
        beta_joint_log_prob,
        beta_counts.tolist()
    )