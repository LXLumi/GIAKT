"""Stable finite mixtures, answer-conditioned responsibilities and shared candidate scoring."""
import torch
from torch import nn
from torch.nn import functional as F


def mixture_logit(logits, log_prior):
    correct = torch.logsumexp(log_prior+F.logsigmoid(logits), dim=-1)
    incorrect = torch.logsumexp(log_prior+F.logsigmoid(-logits), dim=-1)
    return correct-incorrect


def responsibilities(logits, answers, log_prior, valid, stop_gradient=True, temperature=1.):
    likelihood = torch.where(answers.unsqueeze(-1).bool(), F.logsigmoid(logits), F.logsigmoid(-logits))
    weights = ((log_prior+likelihood)/temperature).softmax(-1)
    if stop_gradient:
        weights = weights.detach()
    return weights*valid.unsqueeze(-1)


def candidate_moments(candidates, weights):
    mean = (weights.unsqueeze(-1)*candidates).sum(-2)
    variance = (weights.unsqueeze(-1)*(candidates-mean.unsqueeze(-2)).square()).sum(-2)
    # Exact identity fallback, independent of floating-point normalization roundoff.
    identical = (candidates == candidates[..., :1, :]).all(-1).all(-1, keepdim=True)
    mean = torch.where(identical, candidates[..., 0, :], mean)
    variance = torch.where(identical, torch.zeros_like(variance), variance)
    return mean, variance


class CandidatePredictor(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.hidden, self.output = nn.Linear(2*dim, dim), nn.Linear(dim, 1)
        self.dropout = dropout

    def forward(self, state, candidates):
        context = state.unsqueeze(-2).expand_as(candidates)
        hidden = F.relu(self.hidden(torch.cat([context, candidates], dim=-1)))
        if self.training and self.dropout:
            mask = (torch.rand_like(hidden[..., :1, :]) >= self.dropout).to(hidden.dtype)/(1-self.dropout)
            hidden = hidden*mask
        return self.output(hidden).squeeze(-1)
