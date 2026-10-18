
import torch
import torch.nn as nn

from torch.distributions import Categorical


class DebiasAgent(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Shared Policy Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action B: Oversampling Selection
        self.selection_head = nn.Linear(
            hidden_dim, 1
        )

        # Action C: Loss Reweighting
        self.weight_head = nn.Linear(
            hidden_dim, 4
        )

        self.register_buffer(
            "alpha_options",
            torch.tensor(
                [0.0, 0.15, 0.25, 0.35]
            )
        )

    def forward(self, states):

        features = self.encoder(states)

        selection_logits = (
            self.selection_head(features)
            .squeeze(-1)
        )

        weight_logits = (
            self.weight_head(features)
        )

        return selection_logits, weight_logits


def sample_actions(
    agent,
    states,
    eligible_mask,
    budget_carry=0.0,
    budget_ratio=0.20
):
    """
    Returns:
        duplicate_idx:
            original sample indices to duplicate

        sample_weights:
            CE weight for each original sample

        joint_log_prob:
            log probability of joint actions

        next_carry:
            fractional oversampling budget
    """

    device = states.device

    tail_indices = torch.where(
        eligible_mask
    )[0]

    n_tail = tail_indices.numel()

    sample_weights = torch.ones(
        states.size(0),
        device=device
    )

    if n_tail == 0:
        return (
            tail_indices,
            sample_weights,
            None,
            budget_carry
        )

    tail_states = states[tail_indices]

    selection_logits, weight_logits = agent(
        tail_states
    )

    # ==============================
    # Action C: Reweighting
    # ==============================

    weight_distribution = Categorical(
        logits=weight_logits
    )

    alpha_indices = weight_distribution.sample()

    alphas = agent.alpha_options[alpha_indices]

    normalized_frequency = tail_states[:, 1]

    tail_weights = (
        1.0
        + alphas * (1.0 - normalized_frequency)
    )

    sample_weights[tail_indices] = tail_weights

    log_prob_c = (
        weight_distribution
        .log_prob(alpha_indices)
        .sum()
    )

    # ==============================
    # Action B: Oversampling
    # ==============================

    budget = (
        budget_ratio * n_tail
        + budget_carry
    )

    k = min(int(budget + 1e-9), n_tail)

    next_carry = budget - k

    remaining_logits = selection_logits

    selected_indices = []

    log_prob_b = (
        selection_logits.sum() * 0.0
    )

    # Stochastic sampling without replacement
    for _ in range(k):

        distribution = Categorical(
            logits=remaining_logits
        )

        chosen = distribution.sample()

        selected_indices.append(chosen)

        log_prob_b = (
            log_prob_b
            + distribution.log_prob(chosen)
        )

        selection_mask = (
            torch.arange(
                n_tail,
                device=device
            ) == chosen
        )

        remaining_logits = (
            remaining_logits.masked_fill(
                selection_mask,
                float("-inf")
            )
        )

    if selected_indices:

        selected_local = torch.stack(
            selected_indices
        )

        duplicate_idx = tail_indices[
            selected_local
        ]

    else:
        duplicate_idx = tail_indices[:0]

    # Joint Action Log Probability
    joint_log_prob = (
        log_prob_b + log_prob_c
    )

    return (
        duplicate_idx,
        sample_weights,
        joint_log_prob,
        next_carry
    )
