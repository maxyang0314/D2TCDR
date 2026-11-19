
from collections import Counter
import math
import torch


def build_target_frequency(train_data):
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
    device,
    include_recent_tail=False
):
    """
    Agent State:
        [HPR, F]
        or [HPR, F, RTR]

    No Embedding Features.
    """

    hpr_values = []
    frequency_values = []
    recent_tail_values = []
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
        hpr = (
            sum(
                item in popular_items
                for item in valid_history
            ) / len(valid_history)
            if valid_history else 0.0
        )

        # State 2: Normalized Target Frequency
        normalized_frequency = (
            math.log1p(target_counter.get(target, 0))
            / denominator
        )

        hpr_values.append(hpr)
        frequency_values.append(normalized_frequency)

        # Optional State 3: Recent Tail Ratio
        if include_recent_tail:
            recent_history = valid_history[-2:]

            rtr = (
                sum(
                    item in tail_items
                    for item in recent_history
                ) / len(recent_history)
                if recent_history else 0.0
            )

            recent_tail_values.append(rtr)

        # Only Tail-next samples are eligible
        eligible_values.append(target in tail_items)

    columns = [hpr_values, frequency_values]

    if include_recent_tail:
        columns.append(recent_tail_values)

    states = torch.tensor(
        list(zip(*columns)),
        dtype=torch.float32,
        device=device
    )

    eligible_mask = torch.tensor(
        eligible_values,
        dtype=torch.bool,
        device=device
    )

    return states, eligible_mask
