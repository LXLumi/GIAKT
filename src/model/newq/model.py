"""Change initial Q while inheriting DiffuQKT's complete enhancement and KT path."""
import math

import torch
from torch import nn

from ..diffuqkt import DiffuQKT


class NewQ(DiffuQKT):
    """Q=sum_k alpha(q,k)*(C_k+D_q*V_k)+R_q, with R_q=W_R*E_q.

    The unchanged diffusion receives C_bar+R as its d-dimensional condition.
    This carries the same question residual into both training and sampling.
    """

    MAX_PARAMETER_RATIO = 4.0

    def __init__(self, num_problems, num_skills, embedding_dim=128, num_heads=2, dropout=0.4,
                 diffusion_steps=100, beta_start=1e-4, beta_end=0.02, reverse_variance="posterior",
                 contrastive_negatives="enhanced", contrastive_weight=0.001,
                 diffusion_weight=0.01, temperature=0.3, question_embedding_dim=16,
                 attention_temperature=1.0):
        if (isinstance(question_embedding_dim, bool)
                or not isinstance(question_embedding_dim, int) or question_embedding_dim < 1):
            raise ValueError("question_embedding_dim must be a positive integer.")
        if not math.isfinite(attention_temperature) or attention_temperature <= 0:
            raise ValueError("attention_temperature must be finite and positive.")
        super().__init__(num_problems, num_skills, embedding_dim, num_heads, dropout,
                         diffusion_steps, beta_start, beta_end, reverse_variance,
                         contrastive_negatives, contrastive_weight, diffusion_weight, temperature)
        self.baseline_parameter_count = sum(parameter.numel() for parameter in self.parameters())
        added = (num_problems + 1 + 2 * embedding_dim) * question_embedding_dim
        ratio = (self.baseline_parameter_count + added) / self.baseline_parameter_count
        if ratio > self.MAX_PARAMETER_RATIO:
            raise ValueError(f"NewQ parameter ratio {ratio:.3f} exceeds the 4x baseline limit; "
                             "reduce question_embedding_dim.")
        self.question_embedding = nn.Embedding(num_problems + 1, question_embedding_dim, padding_idx=0)
        self.key_projection = nn.Linear(embedding_dim, question_embedding_dim, bias=False)
        self.residual_projection = nn.Linear(question_embedding_dim, embedding_dim, bias=False)
        nn.init.normal_(self.question_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.question_embedding.weight[0].zero_()
        self.attention_temperature = attention_temperature
        self.question_embedding_dim = question_embedding_dim
        self.parameter_ratio = ratio

    def question_components(self, batch):
        """Fuse each observed concept set; padding and repeated labels get no extra weight."""
        skill_ids = batch["skill_ids"]
        if skill_ids.ndim == 2:
            skill_ids = skill_ids.unsqueeze(-1)
        if skill_ids.ndim != 3 or skill_ids.shape[-1] == 0:
            raise ValueError("skill_ids must have shape [batch, time] or [batch, time, skills].")
        skill_ids = skill_ids.sort(dim=-1).values
        unique = torch.ones_like(skill_ids, dtype=torch.bool)
        unique[..., 1:] = skill_ids[..., 1:] != skill_ids[..., :-1]
        valid = batch["valid_mask"].unsqueeze(-1)
        skill_mask = skill_ids.ne(0) & unique & valid
        concept_vectors = self.concept(skill_ids)
        variation_vectors = self.variation(skill_ids)
        question = self.question_embedding(batch["problem_ids"]) * valid
        keys = self.key_projection(concept_vectors)
        scores = (question.unsqueeze(-2) * keys).sum(dim=-1)
        scores = scores / (math.sqrt(self.question_embedding_dim) * self.attention_temperature)
        scores = scores.masked_fill(~skill_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * skill_mask
        fused_concept = (weights.unsqueeze(-1) * concept_vectors).sum(dim=-2)
        fused_variation = (weights.unsqueeze(-1) * variation_vectors).sum(dim=-2)
        scale = self.difficulty(batch["problem_ids"]).sigmoid() * valid
        residual = self.residual_projection(question) * valid
        # Keep the inherited diffusion's width, training target and sampling API.
        concept_condition = (fused_concept + residual) * valid
        original = (concept_condition + scale * fused_variation) * valid
        return {"concept": concept_condition, "fused_concept": fused_concept,
                "variation": fused_variation, "scale": scale, "conditioning": scale,
                "original": original, "residual": residual, "question_embedding": question,
                "concept_weights": weights, "concept_ids": skill_ids}
