
import math
from collections import Counter, defaultdict
from itertools import combinations

import torch


def build_tail_neighbors(
    train_data,
    tail_items,
    top_k=5,
    min_cooccurrence=2
):
    """
    Build Tail-to-Tail neighbors using
    Target Training histories only.
    """
    if top_k < 1 or min_cooccurrence < 1:
        raise ValueError(
            "top_k and min_cooccurrence must be positive"
        )

    item_count = Counter()
    pair_count = Counter()

    # 1. Count item and pair co-occurrence
    for history in train_data["seq"]:
        items = sorted({
            int(item)
            for item in history
            if int(item) != 0
        })

        item_count.update(items)
        pair_count.update(combinations(items, 2))

    # 2. Collect related Tail items
    candidates = defaultdict(list)

    for (item_a, item_b), count in pair_count.items():
        if count < min_cooccurrence:
            continue

        if (
            item_a not in tail_items
            or item_b not in tail_items
        ):
            continue

        similarity = count / math.sqrt(
            item_count[item_a] * item_count[item_b]
        )

        candidates[item_a].append(
            (item_b, similarity)
        )
        candidates[item_b].append(
            (item_a, similarity)
        )

    # 3. Select Top-K and normalize similarities
    neighbors = {}
    neighbor_quality = {}

    for target, values in candidates.items():
        values.sort(
            key=lambda entry: (-entry[1], entry[0])
        )

        selected = values[:top_k]

        total_similarity = sum(
            sim for _, sim in selected
        )

        if total_similarity <= 0:
            continue

        neighbors[target] = [
            (item, sim / total_similarity)
            for item, sim in selected
        ]

        neighbor_quality[target] = selected[0][1]

    return neighbors, neighbor_quality


def build_neighbor_batch(
    targets,
    neighbors,
    top_k,
    device
):
    """
    Convert neighbors into batch tensors.

    neighbor_ids:   [batch_size, top_k]
    neighbor_probs: [batch_size, top_k]
    """
    neighbor_ids = []
    neighbor_probs = []

    for target in targets:
        values = neighbors.get(int(target), ())

        ids = [item for item, _ in values]
        probs = [prob for _, prob in values]

        pad = top_k - len(ids)

        neighbor_ids.append(
            ids + [0] * pad
        )
        neighbor_probs.append(
            probs + [0.0] * pad
        )

    return (
        torch.tensor(
            neighbor_ids,
            dtype=torch.long,
            device=device
        ),
        torch.tensor(
            neighbor_probs,
            dtype=torch.float32,
            device=device
        )
    )
