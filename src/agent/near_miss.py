
import hashlib
import numpy as np
import torch


def make_row_hash(sequence, target):
    """
    Verify that Rank CSV matches the training row.
    Must use the same hash as analyze_tail_rank.py.
    """
    sequence = [int(x) for x in sequence]

    text = (
        f"{int(target)}|"
        + ",".join(map(str, sequence))
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def matched_uniform_alpha(
    train_df,
    tail_items,
    target_counter,
    max_frequency,
    near_alpha=0.6
):
    """
    Match the total additional loss-weight mass
    between Near-miss and Uniform rules.
    """

    targets = train_df["next"].astype(
        int
    ).to_numpy()

    tail_mask = np.array([
        target in tail_items
        for target in targets
    ], dtype=bool)

    ranks = train_df[
        "teacher_rank"
    ].to_numpy()

    near_mask = (
        tail_mask
        & (ranks >= 11)
        & (ranks <= 20)
    )

    # Same F definition as agent/state.py
    denominator = np.log1p(max_frequency)

    frequencies = np.array([
        target_counter.get(int(target), 0)
        for target in targets
    ])

    F = (
        np.log1p(frequencies)
        / denominator
    )

    frequency_factor = np.clip(
        1.0 - F, 0.0, 1.0
    )

    # Sum of (1-F) over Near-miss samples
    near_mass = frequency_factor[
        near_mask
    ].sum()

    # Sum of (1-F) over all Tail-next samples
    tail_mass = frequency_factor[
        tail_mask
    ].sum()

    if tail_mass <= 0:
        return 0.0

    uniform_alpha = (
        near_alpha * near_mass / tail_mass
    )

    return float(uniform_alpha)


def rank_rule_weights(
    states,
    eligible_mask,
    ranks,
    mode,
    near_alpha=0.6,
    uniform_alpha=0.0
):
    """
    states[:, 0] = HPR
    states[:, 1] = F

    eligible_mask = Tail-next samples
    ranks = Frozen Baseline Rankings
    """

    weights = torch.ones(
        states.size(0),
        dtype=states.dtype,
        device=states.device
    )

    frequency_factor = (
        1.0 - states[:, 1]
    ).clamp(0.0, 1.0)

    if mode == "rank_rule":

        near_mask = (
            eligible_mask
            & (ranks >= 11)
            & (ranks <= 20)
        )

        weights[near_mask] += (
            near_alpha
            * frequency_factor[near_mask]
        )

    elif mode == "uniform_match":

        weights[eligible_mask] += (
            uniform_alpha
            * frequency_factor[eligible_mask]
        )

    else:
        raise ValueError(
            f"Unknown Reweighting Mode: {mode}"
        )

    return weights
