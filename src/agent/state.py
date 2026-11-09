
from collections import Counter
import math
import torch


def build_target_frequency(train_data):
    """
    Count target frequency from training next items only.
    """

    counter = Counter(
        int(item)
        for item in train_data["next"]
        if int(item) != 0
    )

    if not counter:
        raise ValueError("Training targets are empty.")

    max_frequency = max(counter.values())

    return counter, max_frequency


def build_sample_states(
    batch_df,
    popular_items,
    tail_items,
    target_counter,
    max_frequency,
    device,
    neighbor_quality=None
):
    """
    State: [HPR, F, NQ]
    Eligibility: all Tail-next samples
    """
    if neighbor_quality is None:
        neighbor_quality = {}

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

        hpr = (
            sum(item in popular_items
                for item in valid_history)
            / len(valid_history)
            if valid_history else 0.0
        )

        frequency = (
            math.log1p(target_counter.get(target, 0))
            / denominator
        )

        nq = neighbor_quality.get(target, 0.0)

        hpr_values.append(hpr)
        frequency_values.append(frequency)
        quality_values.append(nq)

        # All Tail-next samples enter the Agent
        eligible_values.append(
            target in tail_items
        )

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

