"""Cosine concept fusion with configurable residual diffusion paths."""
import math
from numbers import Real
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from ..diffuqkt import DiffuQKT
from .diffusion import ConcatenatedTargetDiffusion, ResidualConditionDiffusion


class NewQV2(DiffuQKT):
    """Independent item codes with optional residual conditioning/reconstruction."""

    NORMALIZATION_EPS = 1e-8
    MAX_PARAMETER_RATIO = 4.0

    def __init__(self, num_problems, num_skills, embedding_dim=128, num_heads=2, dropout=0.4,
                 diffusion_steps=100, beta_start=1e-4, beta_end=0.02, reverse_variance="posterior",
                 contrastive_negatives="enhanced", contrastive_weight=0.001,
                 diffusion_weight=0.01, temperature=0.3, attention_embedding_dim=8,
                 residual_embedding_dim=8, attention_scale=2.0, attention_temperature=1.0,
                 residual_diffusion_enabled=False, residual_diffusion_role="condition",
                 residual_condition_detach=False, residual_fusion="add",
                 residual_concat_keep_2d=False, diffusion_difficulty_condition=True):
        if not isinstance(diffusion_difficulty_condition, bool):
            raise ValueError("diffusion_difficulty_condition must be a bool.")
        if not isinstance(residual_diffusion_enabled, bool):
            raise ValueError("residual_diffusion_enabled must be a bool.")
        if not isinstance(residual_diffusion_role, str) or residual_diffusion_role not in {"condition", "target", "both"}:
            raise ValueError("residual_diffusion_role must be condition, target or both.")
        if not isinstance(residual_condition_detach, bool):
            raise ValueError("residual_condition_detach must be a bool.")
        if not isinstance(residual_fusion, str) or residual_fusion not in {"add", "concat"}:
            raise ValueError("residual_fusion must be add or concat.")
        if not isinstance(residual_concat_keep_2d, (bool, int)) or residual_concat_keep_2d not in (0, 1):
            raise ValueError("residual_concat_keep_2d must be a bool or integer 0/1.")
        residual_concat_keep_2d = bool(residual_concat_keep_2d)
        use_residual_condition = residual_diffusion_enabled and residual_diffusion_role in {"condition", "both"}
        concatenate_2d = residual_fusion == "concat" and residual_concat_keep_2d
        full_target = residual_diffusion_enabled and residual_diffusion_role in {"target", "both"}
        for name, value in [("attention_embedding_dim", attention_embedding_dim),
                            ("residual_embedding_dim", residual_embedding_dim)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in [("attention_scale", attention_scale),
                            ("attention_temperature", attention_temperature)]:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        super().__init__(num_problems, num_skills, embedding_dim, num_heads, dropout,
                         diffusion_steps, beta_start, beta_end, reverse_variance,
                         contrastive_negatives, contrastive_weight, diffusion_weight, temperature)
        self.baseline_parameter_count = sum(parameter.numel() for parameter in self.parameters())
        added = (num_problems + 1 + embedding_dim) * (attention_embedding_dim + residual_embedding_dim)
        if use_residual_condition:
            added += 2 * embedding_dim**2
        if residual_fusion == "concat" and not concatenate_2d:
            added += 2 * embedding_dim**2
        ratio = (self.baseline_parameter_count + added) / self.baseline_parameter_count
        if ratio > self.MAX_PARAMETER_RATIO:
            raise ValueError(f"NewQV2 parameter ratio {ratio:.3f} exceeds the 4x baseline limit; "
                             "reduce attention_embedding_dim or residual_embedding_dim.")
        self.attention_question_embedding = nn.Embedding(num_problems + 1, attention_embedding_dim, padding_idx=0)
        self.residual_question_embedding = nn.Embedding(num_problems + 1, residual_embedding_dim, padding_idx=0)
        self.key_projection = nn.Linear(embedding_dim, attention_embedding_dim, bias=False)
        self.residual_projection = nn.Linear(residual_embedding_dim, embedding_dim, bias=False)
        for embedding in [self.attention_question_embedding, self.residual_question_embedding]:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.attention_embedding_dim = attention_embedding_dim
        self.residual_embedding_dim = residual_embedding_dim
        self.attention_scale = float(attention_scale)
        self.attention_temperature = float(attention_temperature)
        self.residual_diffusion_enabled = residual_diffusion_enabled
        self.residual_diffusion_role = residual_diffusion_role
        self.residual_condition_detach = residual_condition_detach
        self.diffusion_difficulty_condition = diffusion_difficulty_condition
        self.residual_fusion = residual_fusion
        self.residual_concat_keep_2d = residual_concat_keep_2d
        self.question_representation_dim = 2 * embedding_dim if concatenate_2d else embedding_dim
        self.effective_residual_diffusion_mode = residual_diffusion_role if residual_diffusion_enabled else "off"
        if concatenate_2d and full_target:
            with torch.random.fork_rng(devices=[]):
                self.diffusion = ConcatenatedTargetDiffusion(self.diffusion, use_residual_condition)
        elif use_residual_condition:
            # Expansion happens after all historical parameters are initialized.
            # Its temporary random initializers do not shift subsequent training RNG.
            with torch.random.fork_rng(devices=[]):
                self.diffusion = ResidualConditionDiffusion(self.diffusion)
        if residual_fusion == "concat" and not concatenate_2d:
            with torch.random.fork_rng(devices=[]):
                self.residual_fusion_projection = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
            # The shared fusion starts as addition, then learns independent mixing weights.
            with torch.no_grad():
                identity = torch.eye(embedding_dim)
                self.residual_fusion_projection.weight.copy_(torch.cat([identity, identity], dim=-1))
        if concatenate_2d:
            self.answer = nn.Embedding(2, self.question_representation_dim)
            nn.init.uniform_(self.answer.weight, 0, 1)
            self.initialize_downstream(self.question_representation_dim, num_heads, dropout)
        self.parameter_ratio = sum(parameter.numel() for parameter in self.parameters()) / self.baseline_parameter_count
        if self.parameter_ratio > self.MAX_PARAMETER_RATIO:
            raise ValueError(f"NewQV2 parameter ratio {self.parameter_ratio:.3f} exceeds the 4x baseline limit; "
                             "reduce attention_embedding_dim or residual_embedding_dim.")
        if self.parameter_ratio > 3:
            warnings.warn(f"NewQV2 parameter ratio {self.parameter_ratio:.3f} exceeds the 3x target "
                          "but is within the 4x hard limit.", RuntimeWarning, stacklevel=2)

    def fuse_residual(self, question, residual):
        """Use the same fusion for original Q and any post-diffusion residual."""
        if self.residual_fusion == "add":
            return question + residual
        concatenated = torch.cat([question, residual], dim=-1)
        return concatenated if self.residual_concat_keep_2d else self.residual_fusion_projection(concatenated)

    def question_components(self, batch):
        """Cosine-score the observed concept set; keep original and semantic Q distinct."""
        skill_ids = batch["skill_ids"]
        if skill_ids.ndim == 2:
            skill_ids = skill_ids.unsqueeze(-1)
        if skill_ids.ndim != 3 or skill_ids.shape[-1] == 0:
            raise ValueError("skill_ids must have shape [batch, time] or [batch, time, skills].")
        skill_ids = skill_ids.sort(dim=-1).values
        unique = torch.ones_like(skill_ids, dtype=torch.bool)
        unique[..., 1:] = skill_ids[..., 1:] != skill_ids[..., :-1]
        valid = batch["valid_mask"].unsqueeze(-1)
        concept_mask = skill_ids.ne(0) & unique & valid
        concepts, variations = self.concept(skill_ids), self.variation(skill_ids)
        attention_code = self.attention_question_embedding(batch["problem_ids"]) * valid
        residual_code = self.residual_question_embedding(batch["problem_ids"]) * valid
        query = F.normalize(attention_code, p=2, dim=-1, eps=self.NORMALIZATION_EPS)
        keys = F.normalize(self.key_projection(concepts), p=2, dim=-1, eps=self.NORMALIZATION_EPS)
        scores = (query.unsqueeze(-2) * keys).sum(dim=-1)
        scores = scores * (self.attention_scale / self.attention_temperature)
        scores = scores.masked_fill(~concept_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * concept_mask.to(scores.dtype)
        concept = (weights.unsqueeze(-1) * concepts).sum(dim=-2) * valid
        variation = (weights.unsqueeze(-1) * variations).sum(dim=-2) * valid
        scale = self.difficulty(batch["problem_ids"]).sigmoid() * valid
        semantic = (concept + scale * variation) * valid
        residual = self.residual_projection(residual_code) * valid
        return {"concept": concept, "fused_concept": concept, "variation": variation,
                "scale": scale, "conditioning": scale, "semantic_original": semantic,
                "residual": residual, "original": self.fuse_residual(semantic, residual) * valid,
                "attention_question_embedding": attention_code,
                "residual_question_embedding": residual_code,
                "concept_weights": weights, "concept_ids": skill_ids, "concept_mask": concept_mask}

    def diffusion_route(self, components):
        """One target/condition/output decision for training, evaluation and diagnostics."""
        is_condition = self.residual_diffusion_enabled and self.residual_diffusion_role in {"condition", "both"}
        is_target = self.residual_diffusion_enabled and self.residual_diffusion_role in {"target", "both"}
        residual_condition = components["residual"] if is_condition else None
        if residual_condition is not None and self.residual_condition_detach:
            # Stop only condition-input gradients; original/target/output R stays live.
            residual_condition = residual_condition.detach()
        return {"diffusion_target": components["original"] if is_target else components["semantic_original"],
                "residual_condition": residual_condition,
                "fuse_residual_after_generation": not is_target,
                "add_residual_after_generation": not is_target and self.residual_fusion == "add"}

    def experiment_metadata(self):
        """Persist requested switches and semantic mode; layer shape alone is insufficient."""
        is_target = self.residual_diffusion_enabled and self.residual_diffusion_role in {"target", "both"}
        is_condition = self.residual_diffusion_enabled and self.residual_diffusion_role in {"condition", "both"}
        return {"residual_diffusion_enabled": self.residual_diffusion_enabled,
                "diffusion_difficulty_condition": self.diffusion_difficulty_condition,
                "residual_diffusion_role": self.residual_diffusion_role,
                "residual_condition_detach": self.residual_condition_detach,
                "residual_fusion": self.residual_fusion,
                "residual_concat_keep_2d": self.residual_concat_keep_2d,
                "effective_residual_concat_keep_2d": self.residual_fusion == "concat" and self.residual_concat_keep_2d,
                "question_representation_dim": self.question_representation_dim,
                "diffusion_output_dim": getattr(self.diffusion, "output_dim", self.diffusion.dim),
                "effective_residual_condition_detach": is_condition and self.residual_condition_detach,
                "effective_residual_diffusion_mode": self.effective_residual_diffusion_mode,
                "diffusion_target_name": "original" if is_target else "semantic_original",
                "has_residual_condition": is_condition,
                "fuse_residual_after_generation": not is_target,
                "add_residual_after_generation": not is_target and self.residual_fusion == "add",
                "generated_representation_kind": "original" if is_target else "semantic",
                "parameter_count": sum(parameter.numel() for parameter in self.parameters())}

    def prepare_training(self, dataset):
        """Make constructor defaults explicit in new artifacts, including legacy configs."""
        return {"residual_diffusion_enabled": self.residual_diffusion_enabled,
                "diffusion_difficulty_condition": self.diffusion_difficulty_condition,
                "residual_diffusion_role": self.residual_diffusion_role,
                "residual_condition_detach": self.residual_condition_detach,
                "residual_fusion": self.residual_fusion,
                "residual_concat_keep_2d": self.residual_concat_keep_2d}

    def generate_questions(self, components, batch, generator=None):
        """Return the actual diffusion output and the completed KT representation."""
        route = self.diffusion_route(components)
        concept = components["concept"]
        condition = components["conditioning"]
        if not self.diffusion_difficulty_condition:
            # Remove only the explicit D condition, preserving D in Q and the target.
            # A zero input keeps parameter shapes and historical checkpoints compatible.
            condition = torch.zeros_like(condition)
        valid = batch["valid_mask"]
        extra = {} if route["residual_condition"] is None else {"residual_condition": route["residual_condition"]}
        if self.training:
            generated, diffusion_loss = self.diffusion.training_representation(
                route["diffusion_target"], concept, condition, valid, **extra)
        else:
            generated = self.diffusion.sample(concept, condition, valid, generator, **extra)
            diffusion_loss = None
        enhanced = self.fuse_residual(generated, components["residual"]) if route["fuse_residual_after_generation"] else generated
        enhanced = enhanced * valid.unsqueeze(-1)
        return generated, enhanced, {"diffusion_loss": diffusion_loss}

    def enhance_questions(self, components, batch, generator=None):
        _, enhanced, losses = self.generate_questions(components, batch, generator)
        return enhanced, losses
