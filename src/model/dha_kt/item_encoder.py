"""A bounded no-diffusion adapter of NewQV2's concept-set and residual representation."""
import torch
from torch import nn
from torch.nn import functional as F


class ItemEncoder(nn.Module):
    def __init__(self, num_problems, num_skills, dim, attention_dim, residual_dim, scale, temperature,
                 learned_residual=True, fusion="add", keep_2d=False):
        super().__init__()
        self.dim, self.output_dim = dim, 2*dim if fusion == "concat" and keep_2d else dim
        self.fusion, self.keep_2d, self.learned_residual = fusion, keep_2d, learned_residual
        self.scale, self.temperature = scale, temperature
        self.concept, self.variation = [nn.Embedding(num_skills+1, dim, padding_idx=0) for _ in range(2)]
        self.difficulty = nn.Embedding(num_problems+1, 1, padding_idx=0)
        for embedding in [self.concept, self.variation, self.difficulty]:
            nn.init.uniform_(embedding.weight, 0, 1)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.attention_code = nn.Embedding(num_problems+1, attention_dim, padding_idx=0)
        self.key_projection = nn.Linear(dim, attention_dim, bias=False)
        embeddings = [self.attention_code]
        if learned_residual:
            self.residual_code = nn.Embedding(num_problems+1, residual_dim, padding_idx=0)
            self.residual_projection = nn.Linear(residual_dim, dim, bias=False)
            embeddings.append(self.residual_code)
        for embedding in embeddings:
            nn.init.normal_(embedding.weight, std=.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        if fusion == "concat" and not keep_2d:
            self.fusion_projection = nn.Linear(2*dim, dim, bias=False)
            with torch.no_grad():
                self.fusion_projection.weight.copy_(torch.cat([torch.eye(dim), torch.eye(dim)], dim=-1))

    def fuse(self, semantic, residual):
        if self.fusion == "add":
            return semantic+residual
        joined = torch.cat([semantic, residual], dim=-1)
        return joined if self.keep_2d else self.fusion_projection(joined)

    def forward(self, batch):
        skills = batch["skill_ids"]
        if skills.ndim == 2:
            skills = skills.unsqueeze(-1)
        if skills.ndim != 3 or not skills.shape[-1]:
            raise ValueError("skill_ids must have a nonempty concept axis.")
        skills = skills.sort(dim=-1).values
        unique = torch.ones_like(skills, dtype=torch.bool)
        unique[..., 1:] = skills[..., 1:] != skills[..., :-1]
        valid = batch["valid_mask"].unsqueeze(-1)
        mask = unique & skills.ne(0) & valid
        concepts, variations = self.concept(skills), self.variation(skills)
        query = F.normalize(self.attention_code(batch["problem_ids"]), dim=-1, eps=1e-8)
        keys = F.normalize(self.key_projection(concepts), dim=-1, eps=1e-8)
        scores = (query.unsqueeze(-2)*keys).sum(-1)*(self.scale/self.temperature)
        weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1)*mask
        concept = (weights.unsqueeze(-1)*concepts).sum(-2)*valid
        variation = (weights.unsqueeze(-1)*variations).sum(-2)*valid
        difficulty = self.difficulty(batch["problem_ids"]).sigmoid()*valid
        semantic = (concept+difficulty*variation)*valid
        residual = self.residual_projection(self.residual_code(batch["problem_ids"]))*valid if self.learned_residual else torch.zeros_like(semantic)
        # Keys use the nonzero entries of this sorted, deduplicated event concept view.
        canonical = skills.masked_fill(~mask, 0)
        return dict(concept=concept, variation=variation, difficulty=difficulty, semantic=semantic,
                    residual=residual, original=self.fuse(semantic, residual)*valid,
                    canonical_skill_ids=canonical, concept_weights=weights, concept_mask=mask)
