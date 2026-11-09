
from collections import Counter
import math
import torch


def build_target_frequency(train_data):
    """
    Count target frequency from training next items.
    """
    counter = Counter(
        int(item)
        for item in train_data["next"]
        if int(item) != 0
    )

    if not counter:
        raise ValueError("Training targets are empty.")

    return counter, max(counter.values())


def build_sample_states(
    batch_df,
    popular_items,
    tail_items,
    target_counter,
    max_frequency,
    neighbor_quality,
    device
):
    """
    State:
        [HPR, Normalized Target Frequency, NQ]

    Eligible:
        Tail-next with valid Tail neighbors.
    """
    hpr_values = []
    frequency_values = []
    quality_values = []
    eligible_values = []

    denominator = math.log1p(max_frequency)

    for history, target in zip(
        batch_df["seq"],
        batch_df["next"]
    ):
        target = int(target)

        valid_history = [
            int(item)
            for item in history
            if int(item) != 0
        ]

        # State 1: Historical Popular Ratio
        if valid_history:
            hpr = sum(
                item in popular_items
                for item in valid_history
            ) / len(valid_history)
        else:
            hpr = 0.0

        # State 2: Normalized Target Frequency
        frequency = (
            math.log1p(
                target_counter.get(target, 0)
            ) / denominator
        )

        # State 3: Neighbor Quality
        quality = neighbor_quality.get(
            target, 0.0
        )

        # Soft Label eligibility
        eligible = (
            target in tail_items
            and target in neighbor_quality
        )

        hpr_values.append(hpr)
        frequency_values.append(frequency)
        quality_values.append(quality)
        eligible_values.append(eligible)

    states = torch.tensor(
        list(zip(
            hpr_values,
            frequency_values,
            quality_values
        )),
        dtype=torch.float32,
        device=device
    )

    eligible_mask = torch.tensor(
        eligible_values,
        dtype=torch.bool,
        device=device
    )

    return states, eligible_mask
