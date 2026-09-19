import math
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical


ACTION_NONE = 0
ACTION_D1 = 1
ACTION_D2 = 2
ACTION_NAMES = ("none", "d1", "d2")


class RegulationPolicy(nn.Module):
    """Small contextual-bandit policy for per-sample regulation."""

    def __init__(self, state_dim=6, hidden_dim=32, num_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, states):
        return self.net(states)


class RewardBaseline:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.value = None

    def advantage(self, rewards):
        mean_reward = rewards.detach().mean().item()
        if self.value is None:
            self.value = mean_reward
        baseline = self.value
        self.value = self.momentum * self.value + (1.0 - self.momentum) * mean_reward
        return rewards - baseline


def _context_state(ratio, low_threshold, high_threshold):
    if ratio < low_threshold:
        return "low"
    if ratio < high_threshold:
        return "medium"
    return "high"


def build_agent_states(
    seq_list,
    target_list,
    popular_items,
    next_counter,
    low_threshold,
    high_threshold,
    device,
):
    """Encode S1 exposure, S2 popularity ratio and S3 PP/PT/TT/TP."""
    max_frequency = max(next_counter.values()) if next_counter else 1
    log_denominator = max(math.log1p(max_frequency), 1.0)
    rows = []
    metadata = []

    for seq, target in zip(seq_list, target_list):
        valid = [int(item) for item in seq if int(item) != 0]
        ratio = (
            sum(item in popular_items for item in valid) / len(valid)
            if valid else 0.0
        )
        frequency = int(next_counter.get(int(target), 0))
        normalized_exposure = math.log1p(frequency) / log_denominator
        context_popular = ratio >= low_threshold
        target_popular = int(target) in popular_items

        # S3 order: PP, PT, TT, TP.
        relation = [0.0] * 4
        relation_index = {
            (True, True): 0,
            (True, False): 1,
            (False, False): 2,
            (False, True): 3,
        }[(context_popular, target_popular)]
        relation[relation_index] = 1.0
        rows.append([normalized_exposure, ratio] + relation)
        metadata.append({
            "frequency": frequency,
            "sequence_popularity": ratio,
            "context_state": _context_state(ratio, low_threshold, high_threshold),
            "target_is_tail": not target_popular,
        })

    return torch.tensor(rows, dtype=torch.float32, device=device), metadata


def sample_actions(policy, states, target_is_tail, deterministic=False):
    logits = policy(states)
    tail_mask = torch.tensor(target_is_tail, dtype=torch.bool, device=states.device)

    # D1/D2 are defined only for observed Tail-next samples. Popular-next uses A0.
    logits = logits.clone()
    logits[~tail_mask, ACTION_D1:] = -1e9
    distribution = Categorical(logits=logits)
    actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
    return actions, distribution.log_prob(actions), distribution.entropy(), logits.softmax(-1)


def d1_sample_weights(actions, metadata, device):
    """Experiment-D fixed strengths: f=1:.25, f=2:.40, f>=3:.10."""
    values = []
    for action, info in zip(actions.tolist(), metadata):
        alpha = 0.0
        if action == ACTION_D1 and info["target_is_tail"]:
            frequency = info["frequency"]
            alpha = 0.25 if frequency <= 1 else (0.40 if frequency == 2 else 0.10)
        values.append(1.0 + alpha)
    return torch.tensor(values, dtype=torch.float32, device=device)


def apply_d2_actions(seq_list, actions, metadata, max_len, preserve_recent, rng):
    """Apply Experiment-E strength only to samples for which the agent chose D2."""
    drop_by_state = {"low": 0.10, "medium": 0.20, "high": 0.20}
    output = []
    applied = 0
    for seq, action, info in zip(seq_list, actions.tolist(), metadata):
        if action != ACTION_D2 or not info["target_is_tail"]:
            output.append(list(seq))
            continue

        real = [int(item) for item in seq if int(item) != 0]
        if len(real) > preserve_recent:
            older, recent = real[:-preserve_recent], real[-preserve_recent:]
            probability = drop_by_state[info["context_state"]]
            real = [item for item in older if rng.random() >= probability] + recent
        output.append([0] * (max_len - len(real[-max_len:])) + real[-max_len:])
        applied += 1
    return output, applied


def per_sample_learning_and_margin(model, rep_diffu, targets, popular_item_tensor):
    scores = torch.matmul(rep_diffu, model.target_embeddings.weight.t())
    labels = targets.squeeze(-1)
    learning_loss = nn.functional.cross_entropy(scores, labels, reduction="none")
    ground_truth_scores = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    popular_scores = scores.index_select(1, popular_item_tensor)
    tail_margin = ground_truth_scores - popular_scores.max(dim=1).values
    return learning_loss, tail_margin
