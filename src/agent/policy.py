
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class DebiasAgent(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Shared state encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Soft Label Strength Head
        self.strength_head = nn.Linear(
            hidden_dim, 2
        )

    def forward(self, states):
        features = self.encoder(states)

        params = self.strength_head(features)

        concentration = (
            F.softplus(params) + 1.0
        )

        a, b = concentration.unbind(dim=-1)

        return Beta(a, b)


def sample_actions(
    agent,
    states,
    eligible_mask,
    max_eta=0.3,
    training=True
):
    if not 0 < max_eta < 1:
        raise ValueError(
            "max_eta must be between 0 and 1"
        )

    etas = torch.zeros(
        states.size(0),
        dtype=states.dtype,
        device=states.device
    )

    indices = torch.where(eligible_mask)[0]

    if indices.numel() == 0:
        return etas, None, states.new_empty((0,))

    distribution = agent(states[indices])

    if training:
        strength = distribution.sample()
        log_prob = (
            distribution.log_prob(strength).mean()
        )
    else:
        strength = distribution.mean
        log_prob = None

    # Restrict actual eta to (0, max_eta)
    selected_etas = strength * max_eta

    etas[indices] = selected_etas

    return (
        etas,
        log_prob,
        selected_etas.detach()
    )
