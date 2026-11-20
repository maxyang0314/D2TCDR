
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
    include_trd=False
):
    """
    State:
        E3: [HPR, F]
        E7: [HPR, F, TRD]

    TRD:
        Frozen Baseline Rank normalized to [0, 1].

        TRD = (Rank - 1) / 100

    Only Tail-next samples are eligible for reweighting.
    """

    if include_trd and "teacher_rank" not in batch_df.columns:
        raise ValueError(
            "TRD requires teacher_rank column. "
            "Please load the Rank CSV first."
        )

    state_values = []
    eligible_values = []

    denominator = math.log1p(max_frequency)

    # Get frozen baseline ranks
    if include_trd:
        teacher_ranks = batch_df[
            "teacher_rank"
        ].tolist()

    for i, (history, target) in enumerate(
        zip(
            batch_df["seq"],
            batch_df["next"]
        )
    ):

        target = int(target)

        valid_history = [
            int(item)
            for item in history
            if int(item) != 0
        ]

        # ==============================
        # State 1: HPR
        # ==============================

        if valid_history:
            hpr = sum(
                item in popular_items
                for item in valid_history
            ) / len(valid_history)
        else:
            hpr = 0.0

        # ==============================
        # State 2: F
        # ==============================

        target_frequency = target_counter.get(
            target, 0
        )

        normalized_frequency = (
            math.log1p(target_frequency)
            / denominator
        )

        # Tail-next Eligibility
        eligible = target in tail_items

        features = [
            hpr,
            normalized_frequency
        ]

        # ==============================
        # State 3: TRD (Optional)
        # ==============================

        if include_trd:

            rank = int(teacher_ranks[i])

            if eligible:

                if rank < 1 or rank > 101:
                    raise ValueError(
                        f"Invalid Tail-next rank: {rank}"
                    )

                trd = (rank - 1) / 100.0

            else:
                # Non-Tail samples do not need actions
                trd = 0.0

            features.append(trd)

        state_values.append(features)
        eligible_values.append(eligible)

    # ==============================
    # Convert to Tensors
    # ==============================

    states = torch.tensor(
        state_values,
        dtype=torch.float32,
        device=device
    )

    eligible_mask = torch.tensor(
        eligible_values,
        dtype=torch.bool,
        device=device
    )

    return states, eligible_mask
