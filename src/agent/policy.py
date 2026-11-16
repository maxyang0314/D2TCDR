
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class DebiasAgent(nn.Module):
    """NN-based per-sample Tail-next reweighting policy."""

    def __init__(self, hidden_dim=32, embedding_dim=128, max_len=8):
        super().__init__()

        self.max_len = max_len

        # Learn position information in the history sequence.
        self.position_embeddings = nn.Embedding(
            max_len, embedding_dim
        )

        # Learn how important each history item is
        # with respect to the target item.
        self.attention_score = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Input:
        # HPR + Frequency + History Embedding
        # + Target Embedding + Element-wise Interaction
        self.encoder = nn.Sequential(
            nn.Linear(2 + 3 * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Continuous reweighting head
        self.weight_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        states,
        histories,
        next_items,
        embedding_table
    ):
        if (
            histories.ndim != 2
            or histories.size(1) > self.max_len
        ):
            raise ValueError(
                "histories must have shape "
                "[batch, length <= max_len]"
            )

        next_items = next_items.view(-1)

        # Use target model embeddings as features.
        # Prevent policy gradients from directly changing
        # the recommendation model's embedding table.
        embedding_table = embedding_table.detach()

        history_emb = F.embedding(
            histories, embedding_table
        )
        next_emb = F.embedding(
            next_items, embedding_table
        )

        # Position-aware attention
        positions = torch.arange(
            histories.size(1),
            device=histories.device
        )
        position_emb = self.position_embeddings(
            positions
        ).unsqueeze(0)

        target_context = next_emb.unsqueeze(1).expand_as(
            history_emb
        )

        attention_input = torch.cat(
            [history_emb + position_emb, target_context],
            dim=-1
        )

        attention_logits = self.attention_score(
            attention_input
        ).squeeze(-1)

        # Ignore padding items (ID = 0).
        valid_mask = histories.ne(0)

        attention_logits = attention_logits.masked_fill(
            ~valid_mask, -1e4
        )

        attention = torch.softmax(
            attention_logits, dim=-1
        )

        attention = attention * valid_mask.to(
            attention.dtype
        )

        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        # Attention-weighted history representation
        history_context = (
            attention.unsqueeze(-1) * history_emb
        ).sum(dim=1)

        # Combine numerical and embedding features.
        features = torch.cat(
            [
                states,
                history_context,
                next_emb,
                history_context * next_emb
            ],
            dim=-1
        )

        hidden = self.encoder(features)
        params = self.weight_head(hidden)

        concentration = F.softplus(params) + 1.0
        a, b = concentration.unbind(dim=-1)

        return Beta(a, b)


def sample_actions(
    agent,
    states,
    eligible_mask,
    histories,
    next_items,
    embedding_table,
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

    # Only Tail-next samples receive agent actions.
    distribution = agent(
        states[tail_indices],
        histories[tail_indices],
        next_items.view(-1)[tail_indices],
        embedding_table
    )

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

    # NN determines the reweighting strength directly.
    # Tail weight: 1 to 2
    # Other samples: weight 1
    sample_weights[tail_indices] = 1.0 + alphas

    return (
        sample_weights,
        log_prob,
        alphas.detach()
    )
