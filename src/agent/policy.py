
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Beta, Categorical


class DebiasAgent(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Shared State Encoder: HPR, F, NQ
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Method Selection: 0 or 1
        self.method_head = nn.Linear(hidden_dim, 2)

        # Original Model C: Reweighting
        self.weight_head = nn.Linear(hidden_dim, 2)

        # New Method 1: Soft Label
        self.soft_label_head = nn.Linear(hidden_dim, 2)

    def forward(self, states):

        features = self.encoder(states)

        method_logits = self.method_head(features)

        # Method 0 strength: alpha
        rw_params = (
            F.softplus(self.weight_head(features)) + 1.0
        )
        rw_a, rw_b = rw_params.unbind(dim=-1)
        rw_dist = Beta(rw_a, rw_b)

        # Method 1 strength: eta
        soft_params = (
            F.softplus(self.soft_label_head(features)) + 1.0
        )
        soft_a, soft_b = soft_params.unbind(dim=-1)
        soft_dist = Beta(soft_a, soft_b)

        return method_logits, rw_dist, soft_dist


def sample_actions(
    agent,
    states,
    eligible_mask,
    soft_available_mask,
    max_eta=0.3,
    training=True
):

    if not 0 < max_eta < 1:
        raise ValueError(
            "max_eta must be between 0 and 1"
        )

    batch_size = states.size(0)

    # -1 means no Agent action
    method_ids = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=states.device
    )

    alphas = states.new_zeros(batch_size)
    etas = states.new_zeros(batch_size)

    tail_indices = torch.where(eligible_mask)[0]

    if tail_indices.numel() == 0:
        return method_ids, alphas, etas, None

    # Shared features, three heads
    method_logits, rw_dist, soft_dist = (
        agent(states[tail_indices])
    )

    # Cannot choose Soft Label without neighbors
    method_logits = method_logits.clone()
    method_logits[
        ~soft_available_mask[tail_indices], 1
    ] = -1e9

    method_dist = Categorical(logits=method_logits)

    if training:

        # Select Method
        decisions = method_dist.sample()

        # Sample possible strengths
        rw_u = rw_dist.sample()
        soft_u = soft_dist.sample()

        # Only selected strength enters REINFORCE
        selected_log_prob = torch.where(
            decisions == 0,
            rw_dist.log_prob(rw_u),
            soft_dist.log_prob(soft_u)
        )

        policy_log_prob = (
            method_dist.log_prob(decisions)
            + selected_log_prob
        ).mean()

    else:

        decisions = method_logits.argmax(dim=-1)

        rw_u = rw_dist.mean
        soft_u = soft_dist.mean

        policy_log_prob = None

    method_ids[tail_indices] = decisions

    # Method 0: Reweighting alpha
    alphas[tail_indices] = torch.where(
        decisions == 0,
        rw_u,
        torch.zeros_like(rw_u)
    ).detach()

    # Method 1: Soft Label eta
    etas[tail_indices] = torch.where(
        decisions == 1,
        soft_u * max_eta,
        torch.zeros_like(soft_u)
    ).detach()

    return (
        method_ids,
        alphas,
        etas,
        policy_log_prob
    )
