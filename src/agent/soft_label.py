
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
    item_count = Counter()
    pair_count = Counter()

    # Count co-occurrence in Target Training histories
    for history in train_data["seq"]:
        items = sorted({
            int(item)
            for item in history
            if int(item) != 0
        })

        item_count.update(items)
        pair_count.update(combinations(items, 2))

    candidates = defaultdict(list)

    for (a, b), count in pair_count.items():
        if count < min_cooccurrence:
            continue

        if a not in tail_items or b not in tail_items:
            continue

        sim = count / math.sqrt(
            item_count[a] * item_count[b]
        )

        candidates[a].append((b, sim))
        candidates[b].append((a, sim))

    neighbors = {}
    neighbor_quality = {}

    for target, values in candidates.items():
        values.sort(key=lambda x: (-x[1], x[0]))

        selected = values[:top_k]
        total = sum(sim for _, sim in selected)

        if total <= 0:
            continue

        neighbors[target] = [
            (item, sim / total)
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
    ids = []
    probs = []

    for target in targets:
        values = neighbors.get(int(target), [])

        item_ids = [item for item, _ in values]
        item_probs = [p for _, p in values]

        padding = top_k - len(item_ids)

        ids.append(item_ids + [0] * padding)
        probs.append(item_probs + [0.0] * padding)

    return (
        torch.tensor(
            ids, dtype=torch.long, device=device
        ),
        torch.tensor(
            probs, dtype=torch.float32, device=device
        )
    )
