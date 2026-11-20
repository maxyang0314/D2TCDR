"""Diagnose Tail-next ranks using a frozen D2TCDR baseline checkpoint.

Run from D2TCDR/src, e.g.:
 python analyze_tail_rank.py --checkpoint saved_model/movie_video/target_model.pth

This script is read-only with respect to model checkpoints. It uses the
same 1-positive + 100-negative evaluation design as the repository.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import Att_Diffuse_model, create_model_diffu
from trainer import build_item_popularity_groups


def make_row_hash(sequence, target):
    sequence = [int(x) for x in sequence]
    text = f"{int(target)}|" + ",".join(map(str, sequence))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_candidates(target, n_items, row_id, seed):
    """One positive + 100 unique negatives in [1, n_items]."""
    rng = random.Random(seed + row_id * 10007)
    candidates = [int(target)]
    visited = {int(target)}
    while len(candidates) < 101:
        v = rng.randint(1, n_items)
        if v not in visited:
            visited.add(v)
            candidates.append(v)
    return candidates


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-baseline Tail rank analysis")
    p.add_argument("--checkpoint", default="saved_model/movie_video/target_model.pth")
    p.add_argument("--train-file", default="../data/video/train.df")
    p.add_argument("--output", default="analysis/movie_video_teacher_ranks.csv")
    p.add_argument("--reward-holdout-ratio", type=float, default=0.10)
    p.add_argument("--random-seed", type=int, default=1997)
    p.add_argument("--popular-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-tail-samples", type=int, default=0,
                   help="0=all Tail-next training samples. For a quick diagnostic use 3000.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Must match training configuration used by main.py.
    p.add_argument("--source-item-num", type=int, default=98507)
    p.add_argument("--target-item-num", type=int, default=23978)
    p.add_argument("--max-len", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--emb-dropout", type=float, default=0.3)
    p.add_argument("--p", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--schedule-sampler-name", default="lossaware")
    p.add_argument("--diffusion-steps", type=int, default=32)
    p.add_argument("--lambda-uncertainty", type=float, default=0.001)
    p.add_argument("--noise-schedule", default="trunc_lin")
    p.add_argument("--rescale-timesteps", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    device = torch.device(args.device)
    torch.manual_seed(args.random_seed + 707)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    original_train = pd.read_pickle(args.train_file)
    # EXACTLY the target train/reward holdout split in main.py.
    reward_idx = original_train.sample(
        frac=args.reward_holdout_ratio, random_state=args.random_seed
    ).index
    train_data = original_train.drop(index=reward_idx).reset_index(drop=True)
    popular, tail, observed, _ = build_item_popularity_groups(
        train_data, popular_ratio=args.popular_ratio
    )

    args.item_num = args.target_item_num
    model = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    targets = train_data["next"].astype(int).to_numpy()
    tail_positions = np.flatnonzero(np.isin(targets, list(tail)))
    if args.max_tail_samples > 0:
        tail_positions = tail_positions[:args.max_tail_samples]
    ranks = np.full(len(train_data), -1, dtype=np.int16)

    with torch.inference_mode():
        for start in range(0, len(tail_positions), args.batch_size):
            row_ids = tail_positions[start:start + args.batch_size]
            batch_df = train_data.iloc[row_ids]
            seq = torch.as_tensor(batch_df["seq"].tolist(), dtype=torch.long, device=device)
            nxt = torch.as_tensor(batch_df["next"].tolist(), dtype=torch.long, device=device).view(-1, 1)
            _, diffu_rep, _, _, _, _, _ = model(
                seq, nxt, None, False, args, 0.2, train_flag=False
            )
            scores = model.diffu_rep_pre(diffu_rep, pretrain_flag=False)
            candidate_list = [
                make_candidates(int(y), args.target_item_num, int(row_id), args.random_seed + 303)
                for row_id, y in zip(row_ids, batch_df["next"])
            ]
            candidates = torch.as_tensor(candidate_list, dtype=torch.long, device=device)
            sampled_scores = scores.gather(1, candidates)
            # True next item is always candidates[:, 0]. Ties are counted conservatively.
            these_ranks = 1 + (sampled_scores[:, 1:] >= sampled_scores[:, :1]).sum(dim=1)
            ranks[row_ids] = these_ranks.cpu().numpy().astype(np.int16)
            if start == 0 or (start // args.batch_size) % 20 == 0:
                print(f"Ranked Tail-next samples: {min(start + len(row_ids), len(tail_positions))}/{len(tail_positions)}", flush=True)

    output = pd.DataFrame({
            "row_id": np.arange(len(train_data)),
            "target": targets,
            "row_hash": [make_row_hash(seq, target) for seq, target in zip(train_data["seq"], targets)],
            "is_tail": np.isin(targets, list(tail)),
            "teacher_rank": ranks,
        })
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    valid = ranks[ranks > 0]
    bins = [
        ("01-10", 1, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101", 101, 101),
    ]
    print("\n===== Tail-next sampled rank distribution =====")
    print(f"Train rows: {len(train_data)} | Total Tail-next: {np.isin(targets, list(tail)).sum()} | Ranked: {len(valid)}")
    counts = {}
    for name, low, high in bins:
        count = int(((valid >= low) & (valid <= high)).sum())
        counts[name] = count
        print(f"Rank {name:>6}: {count:>7} ({100 * count / len(valid):6.2f}%)" if len(valid) else f"Rank {name:>6}: 0")
    print(f"Saved: {out_path}")
    summary = {
        "training_rows": int(len(train_data)),
        "total_tail_next": int(np.isin(targets, list(tail)).sum()),
        "ranked_tail_next": int(len(valid)),
        "checkpoint": str(args.checkpoint),
        "rank_definition": "1 true + 100 fixed random negative candidates",
        "seed": args.random_seed,
        "bins": counts,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    if args.max_tail_samples:
        print("WARNING: partial diagnostic cache; do not use for full rule-based training.")


if __name__ == "__main__":
    main()
