"""AKT forgetting attention adapted from official DiffuQKT model.py.

Separate inclusive question/interaction encoders and strict historical retrieval
implement paper Eqs. 9-11. All masks include padding and query validity.
"""
import math
import torch
from torch import nn


def masked_softmax(scores, forbidden):
    weights = scores.masked_fill(forbidden, torch.finfo(scores.dtype).min).softmax(dim=-1)
    return weights.masked_fill(forbidden, 0.0)


class ForgettingAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim < 2 or heads < 1 or dim % heads:
            raise ValueError("embedding_dim must be >=2 and divisible by num_heads.")
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(dim, dim) for _ in range(3)]
        self.out_proj = nn.Linear(dim, dim)
        self.gammas = nn.Parameter(torch.zeros(heads, 1, 1))
        nn.init.xavier_uniform_(self.gammas)

    def forward(self, query, key, value, valid, strict=False):
        batch, length, dim = query.shape
        def split(projected):
            return projected.reshape(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = split(self.q_proj(query)), split(self.k_proj(key)), split(self.v_proj(value))
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads)
        position = torch.arange(length, device=query.device)
        distance = (position[:, None] - position[None, :]).abs()
        causal = position[None, :] >= position[:, None] if strict else position[None, :] > position[:, None]
        forbidden = causal[None, None] | (~valid[:, None, None, :]) | (~valid[:, None, :, None])
        with torch.no_grad():
            initial = masked_softmax(scores, forbidden)
            remaining = initial.sum(dim=-1, keepdim=True) - initial.cumsum(dim=-1)
            effective_distance = (remaining * distance).clamp(min=0).sqrt()
        effect = torch.exp(-self.gammas.abs()[None] * effective_distance).clamp(1e-5, 1e5)
        weights = masked_softmax(scores * effect, forbidden)
        output = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        output = self.out_proj(output)
        # No-history rows receive zero attention, including the output projection bias.
        has_history = (~forbidden).any(dim=-1).any(dim=1)
        return output * has_history.unsqueeze(-1)


class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attention = ForgettingAttention(dim, heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, query, key, value, valid, strict=False):
        state = self.norm1(query + self.dropout(self.attention(query, key, value, valid, strict)))
        state = self.norm2(state + self.dropout(self.feedforward(state)))
        return state * valid.unsqueeze(-1)


class KnowledgeRetriever(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.question_encoder = TransformerLayer(dim, heads, dropout)
        self.interaction_encoder = TransformerLayer(dim, heads, dropout)
        self.retrieval = TransformerLayer(dim, heads, dropout)

    def forward(self, question, answer, valid):
        interaction = question + answer
        questions = self.question_encoder(question, question, question, valid)
        interactions = self.interaction_encoder(interaction, interaction, interaction, valid)
        # The value at position t depends on a_t, so the retrieval mask must be strict.
        return self.retrieval(questions, questions, interactions, valid, strict=True)
