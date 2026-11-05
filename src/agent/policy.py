import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class DebiasAgent(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Continuous Model C
        self.weight_head = nn.Linear(hidden_dim, 2)

    def forward(self, states):

        features = self.encoder(states)
        params = self.weight_head(features)

        concentration = F.softplus(params) + 1.0
        a, b = concentration.unbind(dim=-1)

        return Beta(a, b)


def sample_actions(agent, states, eligible_mask, training=True):

    sample_weights = torch.ones(
        states.size(0),
        dtype=states.dtype,
        device=states.device
    )

    tail_indices = torch.where(eligible_mask)[0]

    if tail_indices.numel() == 0:
        return sample_weights, None, states.new_empty((0,))

    tail_states = states[tail_indices]
    distribution = agent(tail_states)

    if training:
        alphas = distribution.sample()
        log_prob = distribution.log_prob(alphas).mean()
    else:
        alphas = distribution.mean
        log_prob = None

    normalized_frequency = tail_states[:, 1]

    tail_weights = (
        1.0 + alphas * (1.0 - normalized_frequency)
    )

    sample_weights[tail_indices] = tail_weights

    return sample_weights, log_prob, alphas.detach()