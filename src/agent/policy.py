
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class DebiasAgent(nn.Module):
    """
    Per-sample continuous reweighting.

    State:
        [HPR, F]
        or [HPR, F, RTR]

    Architecture:
        State -> MLP -> Beta Head

    No Embedding or Attention.
    """

    def __init__(self, hidden_dim=32, state_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.weight_head = nn.Linear(hidden_dim, 2)

    def forward(self, states):

        hidden = self.encoder(states)
        params = self.weight_head(hidden)

        concentration = F.softplus(params) + 1.0
        a, b = concentration.unbind(dim=-1)

        return Beta(a, b)


def sample_actions(
    agent,
    states,
    eligible_mask,
    training=True
):
    sample_weights = torch.ones(
        states.size(0),
        dtype=states.dtype,
        device=states.device
    )

    tail_indices = torch.where(eligible_mask)[0]

    if tail_indices.numel() == 0:
        return (
            sample_weights,
            None,
            states.new_empty((0,))
        )

    tail_states = states[tail_indices]
    distribution = agent(tail_states)

    if training:
        alphas = distribution.sample().clamp(
            1e-6, 1.0 - 1e-6
        )

        log_prob = distribution.log_prob(
            alphas
        ).mean()

    else:
        alphas = distribution.mean
        log_prob = None

    # Model C Reweighting Formula
    frequency = tail_states[:, 1]

    sample_weights[tail_indices] = (
        1.0 + alphas * (1.0 - frequency)
    )

    return (
        sample_weights,
        log_prob,
        alphas.detach()
    )


def rule_based_weights(
    states,
    eligible_mask,
    use_recent_tail=False
):
    """
    Fixed-rule reweighting without RL.
    """

    weights = torch.ones(
        states.size(0),
        dtype=states.dtype,
        device=states.device
    )

    hpr = states[:, 0].clamp(0.0, 1.0)
    frequency = states[:, 1].clamp(0.0, 1.0)

    # Original rule hypothesis
    alpha = hpr

    # Optional recent-tail hypothesis
    if use_recent_tail:
        rtr = states[:, 2].clamp(0.0, 1.0)
        alpha = alpha * (0.5 + 0.5 * rtr)

    weights[eligible_mask] = (
        1.0
        + alpha[eligible_mask]
        * (1.0 - frequency[eligible_mask])
    )

    return weights
