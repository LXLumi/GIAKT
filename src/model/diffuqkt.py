"""DiffuQKT: Rasch question embeddings, conditional diffusion and contrastive KT."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import KnowledgeRetriever
from .diffusion import ConditionalQuestionDiffusion


def stable_state_contrastive(enhanced, original, valid, student_ids, temperature=0.3,
                             negatives="enhanced"):
    """Eq. 14: same-position other-student states; omit padding and same-student windows."""
    if temperature <= 0 or negatives not in {"enhanced", "original"}:
        raise ValueError("Invalid contrastive temperature or negative mode.")
    batch, length, _ = enhanced.shape
    if not valid.any():
        return enhanced.sum() * 0
    e, o = F.normalize(enhanced, dim=-1), F.normalize(original, dim=-1)
    candidates = e if negatives == "enhanced" else o
    logits = torch.einsum("btd,ntd->tbn", e, candidates) / temperature
    diagonal = torch.eye(batch, dtype=torch.bool, device=enhanced.device)[None]
    positive = (e * o).sum(dim=-1).transpose(0, 1) / temperature
    logits = torch.where(diagonal, positive.unsqueeze(-1), logits)
    different_student = student_ids[:, None] != student_ids[None, :]
    allowed = (different_student[None] | diagonal) & valid.transpose(0, 1)[:, None, :]
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    losses = logits.logsumexp(dim=-1) - positive
    return losses[valid.transpose(0, 1)].mean()


class DiffuQKT(nn.Module):
    def __init__(self, num_problems, num_skills, embedding_dim=128, num_heads=2, dropout=0.4,
                 diffusion_steps=100, beta_start=1e-4, beta_end=0.02, reverse_variance="posterior",
                 contrastive_negatives="enhanced", contrastive_weight=0.001,
                 diffusion_weight=0.01, temperature=0.3):
        super().__init__()
        if not 0 <= dropout < 1 or num_problems < 1 or num_skills < 1:
            raise ValueError("Invalid dropout or vocabulary sizes.")
        if min(contrastive_weight, diffusion_weight) < 0 or temperature <= 0:
            raise ValueError("Loss weights must be nonnegative and temperature positive.")
        if contrastive_negatives not in {"enhanced", "original"}:
            raise ValueError("Unknown contrastive negative mode.")
        self.concept = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=0)
        self.variation = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=0)
        self.difficulty = nn.Embedding(num_problems + 1, 1, padding_idx=0)
        self.answer = nn.Embedding(2, embedding_dim)
        for embedding in [self.concept, self.variation, self.difficulty, self.answer]:
            nn.init.uniform_(embedding.weight, 0, 1)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()
        self.diffusion = ConditionalQuestionDiffusion(embedding_dim, diffusion_steps, beta_start,
                                                      beta_end, reverse_variance)
        self.retriever = KnowledgeRetriever(embedding_dim, num_heads, dropout)
        self.predictor = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(embedding_dim, 1))
        self.contrastive_weight, self.diffusion_weight = contrastive_weight, diffusion_weight
        self.temperature, self.contrastive_negatives = temperature, contrastive_negatives

    def embed_skills(self, embedding, skill_ids):
        """Mean all atomic skills of one event; preserve scalar ASSIST17 behavior."""
        if skill_ids.ndim == 2:
            return embedding(skill_ids)
        if skill_ids.ndim != 3:
            raise ValueError("skill_ids must have shape [batch, time] or [batch, time, skills].")
        mask = skill_ids.ne(0).unsqueeze(-1)
        vectors = embedding(skill_ids) * mask
        return vectors.sum(dim=-2) / mask.sum(dim=-2).clamp_min(1)

    def question_components(self, batch):
        """One representation path shared by reconstruction, contrastive KT and sampling."""
        valid = batch["valid_mask"].unsqueeze(-1)
        concept = self.embed_skills(self.concept, batch["skill_ids"]) * valid
        scale = self.difficulty(batch["problem_ids"]).sigmoid() * valid
        variation = self.embed_skills(self.variation, batch["skill_ids"]) * valid
        return {"concept": concept, "variation": variation, "scale": scale, "conditioning": scale,
                "original": (concept + scale * variation) * valid}

    def forward(self, batch, generator=None):
        valid = batch["valid_mask"]
        targets = batch["prediction_mask"] & valid
        if not valid.any() or not targets.any():
            raise ValueError("Batch must contain valid events and next-answer targets.")
        components = self.question_components(batch)
        concept, condition, original = [components[key] for key in ["concept", "conditioning", "original"]]
        if self.training:
            enhanced, diffusion_loss = self.diffusion.training_representation(original, concept, condition, valid)
        else:
            enhanced = self.diffusion.sample(concept, condition, valid, generator)
            # Evaluation uses reverse sampling, not the training reconstruction objective.
            diffusion_loss = None
        answer = self.answer(batch["answers"])
        state = self.retriever(enhanced, answer, valid)
        logits = self.predictor(torch.cat([state, enhanced], dim=-1)).squeeze(-1)
        kt_loss = F.binary_cross_entropy_with_logits(logits[targets], batch["answers"][targets].float())
        contrastive_loss = original.sum() * 0 if self.training else None
        if self.training and self.contrastive_weight > 0:
            original_state = self.retriever(original, answer, valid)
            contrastive_loss = stable_state_contrastive(state, original_state, targets,
                batch["student_ids"], self.temperature, self.contrastive_negatives)
        loss = kt_loss
        if self.training:
            loss = loss + self.contrastive_weight * contrastive_loss + self.diffusion_weight * diffusion_loss
        return {"logits": logits, "probabilities": logits.sigmoid(), "states": state,
                "loss": loss, "kt_loss": kt_loss, "contrastive_loss": contrastive_loss,
                "diffusion_loss": diffusion_loss}
