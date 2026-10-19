"""Diffusion Hypothesis Assimilation: cross-fitted candidates and causal two-pass evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from ..diffuqkt.diffusion import ConditionalQuestionDiffusion
from ..newq_v2.diffusion import ConcatenatedTargetDiffusion, ResidualConditionDiffusion
from src.training.dha_kt_stages import apply_stage, stage_for_epoch
from src.utils.config import project_path
from src.utils.files import sha256
from .candidates import build_candidates, stable_seed, unique_conditions
from .diffusion import ProfileDiffusion
from .evidence import CandidatePredictor, candidate_moments, mixture_logit, responsibilities
from .item_encoder import ItemEncoder
from .reference import ReferenceBank
from .retriever import EvidenceRetriever


def _settings(defaults, values, name):
    if values is not None and not isinstance(values, dict):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(values or {})-set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
    return {**defaults, **copy.deepcopy(values or {})}


def _positive(value, name, integer=False, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int,) if integer else (int, float)) or not math.isfinite(value) or value <= minimum:
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} > {minimum}.")


class DHAKT(nn.Module):
    extra_metric_names = ("pilot_loss", "candidate_probability_variance", "responsibility_entropy", "responsibility_prior_l1",
                          "raw_candidate_bce", "candidate_norm", "num_supervised_profile_keys", "sampling_seconds")
    supports_resume = True
    STAGES = ("base_warmup", "diffusion_pretrain", "joint")

    def __init__(self, num_problems, num_skills, embedding_dim=128, num_heads=2, dropout=.4,
                 attention_embedding_dim=8, residual_embedding_dim=8, attention_scale=2., attention_temperature=1.,
                 learned_residual_enabled=True, residual_fusion="add", residual_concat_keep_2d=False,
                 generated_correction_enabled=True, candidate_backend="evidence_profile", profile_injection="residual_update",
                 profile_residual_condition="none", legacy_residual_mode=None, legacy_residual_condition_detach=False,
                 generated_candidates=3, raw_prior_mass=.5, responsibility_stop_gradient=True,
                 responsibility_temperature=1., responsibility_mode="posterior", surprise_input=False, profile=None, diffusion=None, candidate_sampling=None,
                 stages=None, bank_schema=None, max_parameter_ratio=4., diffusion_supervision_keys_per_step=128,
                 contrastive_weight=0., pilot_weight=.5, diffusion_weight=.1, temperature=.3, contrastive_negatives="enhanced"):
        super().__init__()
        for name, value in [("num_problems", num_problems), ("num_skills", num_skills), ("embedding_dim", embedding_dim),
                            ("num_heads", num_heads), ("attention_embedding_dim", attention_embedding_dim),
                            ("residual_embedding_dim", residual_embedding_dim), ("generated_candidates", generated_candidates),
                            ("diffusion_supervision_keys_per_step", diffusion_supervision_keys_per_step)]:
            _positive(value, name, integer=True)
        for name, value in [("attention_scale", attention_scale), ("attention_temperature", attention_temperature),
                            ("responsibility_temperature", responsibility_temperature), ("max_parameter_ratio", max_parameter_ratio), ("temperature", temperature)]:
            _positive(value, name)
        for name, value in [("learned_residual_enabled", learned_residual_enabled), ("generated_correction_enabled", generated_correction_enabled),
                            ("legacy_residual_condition_detach", legacy_residual_condition_detach),
                            ("responsibility_stop_gradient", responsibility_stop_gradient), ("surprise_input", surprise_input)]:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool.")
        if not isinstance(residual_concat_keep_2d, (bool, int)) or residual_concat_keep_2d not in (0, 1):
            raise ValueError("residual_concat_keep_2d must be bool or integer 0/1.")
        if residual_fusion not in {"add", "concat"} or not learned_residual_enabled and residual_fusion != "add":
            raise ValueError("Fusion must be add/concat; disabling learned R is restricted to add.")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1 or embedding_dim < 2 or embedding_dim % num_heads:
            raise ValueError("Invalid dropout, embedding dimension or head count.")
        if not 0 < raw_prior_mass < 1 or contrastive_weight != 0 or contrastive_negatives not in {"enhanced", "original"}:
            raise ValueError("DHA requires 0<raw_prior_mass<1 and contrastive_weight=0 (no identity InfoNCE).")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in [pilot_weight, diffusion_weight]):
            raise ValueError("DHA loss weights must be finite and nonnegative.")
        if candidate_backend not in {"evidence_profile", "legacy_question"}:
            raise ValueError("candidate_backend must be evidence_profile or legacy_question.")
        if responsibility_mode not in {"posterior", "prior"}:
            raise ValueError("responsibility_mode must be posterior or prior (r -> pi ablation).")
        if profile_residual_condition not in {"none", "detach", "trainable"}:
            raise ValueError("profile_residual_condition must be none/detach/trainable.")
        if candidate_backend == "evidence_profile":
            if profile_injection not in {"residual_update", "semantic_update"} or legacy_residual_mode is not None or legacy_residual_condition_detach:
                raise ValueError("Profile backend requires profile injection and neutral legacy settings.")
            if not learned_residual_enabled and profile_residual_condition != "none":
                raise ValueError("No learned residual: profile residual condition must be none.")
        elif profile_injection is not None or profile_residual_condition != "none" or legacy_residual_mode not in {"off", "condition", "target", "both"}:
            raise ValueError("Legacy backend requires explicit legacy route, profile_injection=null and profile_residual_condition=none.")
        self.profile = _settings(dict(artifact_path=None, manifest_sha256=None, bins=8, fixed_scale=4., shrinkage_kappa=5., no_evidence_policy="raw_only"), profile, "profile")
        _positive(self.profile["bins"], "profile.bins", integer=True, minimum=1)
        for field in ["fixed_scale", "shrinkage_kappa"]:
            _positive(self.profile[field], "profile."+field)
        if self.profile["no_evidence_policy"] not in {"raw_only", "prior_only"}:
            raise ValueError("Unknown no-evidence policy.")
        supplied_diffusion = diffusion or {}
        legacy = candidate_backend == "legacy_question"
        steps = supplied_diffusion.get("steps", 100)
        self.diffusion_config = _settings(dict(prediction_type="x0", schedule="linear" if legacy else "cosine", steps=100,
            sampler="ddpm" if legacy else "ddim", sampling_steps=steps if legacy else min(8, steps), ddim_eta=0.,
            time_embedding_dim=embedding_dim if legacy else 64, hidden_dim=2*embedding_dim if legacy else 128,
            beta_start=1e-4, beta_end=.02, reverse_variance="posterior"), diffusion, "diffusion")
        dc = self.diffusion_config
        for field, minimum in [("steps", 1), ("sampling_steps", 0), ("time_embedding_dim", 1), ("hidden_dim", 0)]:
            _positive(dc[field], "diffusion."+field, integer=True, minimum=minimum)
        if dc["sampling_steps"] > dc["steps"] or dc["prediction_type"] != "x0" or not isinstance(dc["ddim_eta"], (int, float)) or not 0 <= dc["ddim_eta"] <= 1:
            raise ValueError("Invalid x0 sampler settings.")
        if legacy and (dc["schedule"] != "linear" or dc["sampler"] != "ddpm" or dc["sampling_steps"] != dc["steps"] or dc["ddim_eta"] != 0
                       or dc["time_embedding_dim"] != embedding_dim or dc["hidden_dim"] != 2*embedding_dim):
            raise ValueError("Legacy uses full linear DDPM, d time embeddings and 2d hidden width; override profile sampler settings explicitly.")
        if not legacy and dc["sampler"] != "ddim":
            raise ValueError("Profile backend implements the explicitly declared DDIM sampler.")
        self.candidate_sampling = _settings(dict(seed=2026, train_draw_policy="epoch_fixed", eval_condition_draw=0,
            train_cache="forward_only", eval_cache="checkpoint_keyed"), candidate_sampling, "candidate_sampling")
        cs = self.candidate_sampling
        if cs["train_draw_policy"] != "epoch_fixed" or cs["train_cache"] != "forward_only" or cs["eval_cache"] not in {"checkpoint_keyed", "none"}:
            raise ValueError("Unsupported candidate draw/cache protocol.")
        for field in ["seed", "eval_condition_draw"]:
            if isinstance(cs[field], bool) or not isinstance(cs[field], int) or cs[field] < 0:
                raise ValueError(f"candidate_sampling.{field} must be a nonnegative integer.")
        self.stages = _settings(dict(base_warmup_epochs=0, diffusion_pretrain_epochs=0), stages, "stages")
        for field, value in self.stages.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"stages.{field} must be a nonnegative integer.")
        self.candidate_backend, self.profile_injection = candidate_backend, profile_injection
        self.profile_residual_condition, self.legacy_residual_mode = profile_residual_condition, legacy_residual_mode
        self.legacy_residual_condition_detach = legacy_residual_condition_detach
        self.generated_candidates, self.generated_correction_enabled = generated_candidates, generated_correction_enabled
        self.learned_residual_enabled = learned_residual_enabled
        self.responsibility_stop_gradient, self.responsibility_temperature = responsibility_stop_gradient, responsibility_temperature
        self.responsibility_mode = responsibility_mode
        self.surprise_input, self.raw_prior_mass = surprise_input, float(raw_prior_mass)
        self.pilot_weight, self.diffusion_weight = float(pilot_weight), float(diffusion_weight)
        self.max_parameter_ratio, self.supervision_keys_per_step = float(max_parameter_ratio), diffusion_supervision_keys_per_step
        self.item_encoder = ItemEncoder(num_problems, num_skills, embedding_dim, attention_embedding_dim, residual_embedding_dim,
                                        attention_scale, attention_temperature, learned_residual_enabled, residual_fusion, bool(residual_concat_keep_2d))
        self.question_dim = self.item_encoder.output_dim
        if legacy:
            self.diffusion = ConditionalQuestionDiffusion(embedding_dim, dc["steps"], dc["beta_start"], dc["beta_end"], dc["reverse_variance"])
            has_condition = legacy_residual_mode in {"condition", "both"}
            if self.question_dim == 2*embedding_dim and legacy_residual_mode in {"target", "both"}:
                self.diffusion = ConcatenatedTargetDiffusion(self.diffusion, has_condition)
            elif has_condition:
                self.diffusion = ResidualConditionDiffusion(self.diffusion)
        else:
            self.diffusion = ProfileDiffusion(self.profile["bins"], 2*embedding_dim+1,
                embedding_dim if profile_residual_condition != "none" else 0, **{k:dc[k] for k in
                ["steps", "schedule", "sampling_steps", "ddim_eta", "time_embedding_dim", "hidden_dim", "beta_start", "beta_end"]})
            self.profile_to_residual = nn.Linear(self.profile["bins"], embedding_dim, bias=False)
            nn.init.normal_(self.profile_to_residual.weight, std=.01)
        self.reference_bank = ReferenceBank(bank_schema)
        self.answer = nn.Embedding(2, self.question_dim)
        nn.init.uniform_(self.answer.weight, 0, 1)
        self.retriever = EvidenceRetriever(self.question_dim, num_heads, dropout)
        self.predictor = CandidatePredictor(self.question_dim, dropout)
        self.adapter = nn.Linear(2*self.question_dim+int(surprise_input), self.question_dim, bias=False)
        nn.init.zeros_(self.adapter.weight)
        prior = torch.tensor([raw_prior_mass]+[(1-raw_prior_mass)/generated_candidates]*generated_candidates)
        self.register_buffer("log_prior", prior.log())
        self.register_buffer("stage_code", torch.tensor(2, dtype=torch.long))
        self.register_buffer("epoch_index", torch.tensor(0, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        initial_stage = stage_for_epoch(1, **self.stages)
        self.stage_code.fill_(self.STAGES.index(initial_stage))
        apply_stage(self, initial_stage)
        self.data_context = None
        self._candidate_cache, self._snapshot_versions, self._snapshot_digest = {}, None, None
        coordinate_settings = {key:value for key,value in self.profile.items() if key not in {"artifact_path", "manifest_sha256"}}
        self.configuration_fingerprint = hashlib.sha256(json.dumps(dict(backend=candidate_backend, profile=coordinate_settings,
            diffusion=dc, candidate_sampling=cs, fusion=residual_fusion, keep_2d=bool(residual_concat_keep_2d),
            injection=profile_injection, residual_condition=profile_residual_condition, legacy_route=legacy_residual_mode,
            legacy_detach=legacy_residual_condition_detach, generated=generated_correction_enabled, learned=learned_residual_enabled), sort_keys=True).encode()).hexdigest()
        self.baseline_parameter_count = 30*embedding_dim**2+(2*num_skills+43)*embedding_dim+num_problems+2+3*num_heads
        self.parameter_count = sum(p.numel() for p in self.parameters())
        self.parameter_ratio = self.parameter_count/self.baseline_parameter_count
        if self.parameter_ratio > self.max_parameter_ratio:
            raise ValueError(f"DHA parameter ratio {self.parameter_ratio:.3f} exceeds explicit maximum {self.max_parameter_ratio}.")
        if self.parameter_ratio > 3:
            warnings.warn(f"DHA full parameter ratio {self.parameter_ratio:.3f} exceeds 3x baseline.", RuntimeWarning, stacklevel=2)
        self.loss_metric_weights = {}

    @property
    def training_stage(self):
        return self.STAGES[int(self.stage_code)]

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        if bool(self.reference_bank.ready):
            self.reference_bank.validate()
            if self.profile["manifest_sha256"] is not None and self.profile["manifest_sha256"] != self.reference_bank.manifest_sha256:
                raise ValueError("Saved profile manifest SHA256 mismatch.")
        apply_stage(self, self.training_stage)
        self._candidate_cache.clear()
        self._snapshot_versions = None
        return result

    def set_data_context(self, split):
        if split not in {"train", "valid", "test"}:
            raise ValueError("DHA needs an explicit train/valid/test data context.")
        self.data_context = split

    def on_train_epoch_start(self, epoch):
        self.epoch_index.fill_(epoch)
        stage = stage_for_epoch(epoch, **self.stages)
        self.stage_code.fill_(self.STAGES.index(stage))
        apply_stage(self, stage)
        self._candidate_cache.clear()

    def on_optimizer_step(self):
        self.global_step.add_(1)
        self._candidate_cache.clear()

    def selection_group(self):
        return self.training_stage

    def condition_draw(self):
        if self.candidate_backend != "evidence_profile" or not bool(self.reference_bank.ready):
            return 0
        return (max(0, int(self.epoch_index)-1) % self.reference_bank.partial_profile.shape[1]
                if self.training else self.candidate_sampling["eval_condition_draw"])

    def profile_condition(self, residual):
        if self.profile_residual_condition == "none":
            return None
        return residual.detach() if self.profile_residual_condition == "detach" else residual

    def evaluation_snapshot(self, generator):
        versions = tuple(parameter._version for parameter in self.parameters()) + tuple(buffer._version for buffer in self.reference_bank.buffers())
        if versions != self._snapshot_versions:
            digest = hashlib.sha256()
            for name, parameter in self.named_parameters():
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            self._snapshot_digest, self._snapshot_versions = digest.hexdigest(), versions
            self._candidate_cache.clear()
        return (self._snapshot_digest, self.reference_bank.manifest_sha256, self.configuration_fingerprint,
                generator.initial_seed() if generator is not None else self.candidate_sampling["seed"])

    def prepare_training(self, dataset):
        if self.candidate_backend == "evidence_profile":
            if not bool(self.reference_bank.ready):
                artifact_path = self.profile["artifact_path"]
                if not artifact_path:
                    training_path = getattr(dataset, "path", None)
                    if training_path is None:
                        raise ValueError("DHA profile.artifact_path cannot be inferred without dataset.path; pass --reference-artifact <directory>.")
                    # <dataset>/processed/train.npz -> <dataset>/dha_kt/reference.
                    artifact_path = Path(training_path).resolve().parent.parent/"dha_kt"/"reference"
                artifact_path = project_path(artifact_path).resolve()
                self.reference_bank.load_artifact(artifact_path, self.profile["manifest_sha256"])
                self.profile["artifact_path"] = str(artifact_path)
            bank, manifest = self.reference_bank, self.reference_bank.manifest
            if self.profile["manifest_sha256"] is not None and self.profile["manifest_sha256"] != bank.manifest_sha256:
                raise ValueError("Saved reference manifest differs from model coordinate settings.")
            # The hash utility is static preprocessing code; it is called once before training.
            from scripts.dha_kt.build_reference import runtime_fingerprint
            if manifest["train_runtime_sha256"] != runtime_fingerprint(dataset.arrays):
                raise ValueError("Reference training scope differs from the runtime training data; rebuild matching references.")
            if manifest["dataset_metadata_sha256"] != sha256(dataset.path.parent/"metadata.json"):
                raise ValueError("Reference metadata differs from this dataset/split.")
            if bank.partial_profile.shape[2] != self.item_encoder.difficulty.num_embeddings or bank.partial_profile.shape[-1] != self.profile["bins"]:
                raise ValueError("Reference question vocabulary/profile dimension differs from model.")
            if manifest["fixed_profile_scale"] != self.profile["fixed_scale"] or manifest["shrinkage_kappa"] != self.profile["shrinkage_kappa"]:
                raise ValueError("Reference coordinate scale/shrinkage differs from model configuration.")
            if self.candidate_sampling["eval_condition_draw"] >= bank.partial_profile.shape[1]:
                raise ValueError("Evaluation reference draw outside artifact.")
            self.profile["manifest_sha256"] = bank.manifest_sha256
        return dict(profile=copy.deepcopy(self.profile), bank_schema=self.reference_bank.schema(),
                    stages=copy.deepcopy(self.stages))

    def _profile_loss(self):
        bank, draw = self.reference_bank, self.condition_draw()
        supported = bank.dense_observed[bank.supervision_views, draw, bank.supervision_problems].any(-1)
        choices = supported.nonzero().flatten()
        if not choices.numel():
            return sum(p.sum()*0 for p in self.diffusion.parameters()), 0
        choices = choices[torch.randperm(choices.numel(), device=choices.device)[:self.supervision_keys_per_step]]
        problems, skills, views = bank.supervision_problems[choices], bank.supervision_skills[choices], bank.supervision_views[choices]
        valid = torch.ones(problems.shape[0], 1, dtype=torch.bool, device=problems.device)
        parts = self.item_encoder(dict(problem_ids=problems[:, None], skill_ids=skills[:, None, :], valid_mask=valid))
        structure = torch.cat([parts["concept"], parts["semantic"], parts["difficulty"]], -1).squeeze(1)
        partial, counts, observed = bank.partial(views, draw, problems)
        residual = self.profile_condition(parts["residual"].squeeze(1))
        loss = self.diffusion.denoising_loss(bank.dense_profile[views, draw, problems], bank.dense_counts[views, draw, problems],
            bank.dense_observed[views, draw, problems], structure, partial, counts, observed, residual, self.profile["shrinkage_kappa"])
        return loss, choices.numel()

    def _legacy_loss(self, parts, batch):
        _, selected, _, _ = unique_conditions(self, parts, batch)
        target = selected["original"] if self.legacy_residual_mode in {"target", "both"} else selected["semantic"]
        condition = selected["residual"] if self.legacy_residual_mode in {"condition", "both"} else None
        if condition is not None and self.legacy_residual_condition_detach:
            condition = condition.detach()
        extra = {} if condition is None else dict(residual_condition=condition[:, None, :])
        valid = torch.ones(target.shape[0], 1, device=target.device, dtype=torch.bool)
        _, loss = self.diffusion.training_representation(target[:, None, :], selected["concept"][:, None, :], selected["difficulty"][:, None, :], valid, **extra)
        return loss, target.shape[0]

    def forward(self, batch, generator=None, return_details=False):
        import time
        valid, targets = batch["valid_mask"], batch["valid_mask"] & batch["prediction_mask"]
        if not valid.any() or not targets.any():
            raise ValueError("Batch must contain valid events and next-answer targets.")
        parts = self.item_encoder(batch)
        q0 = parts["original"]
        x0 = (q0+self.answer(batch["answers"]))*valid.unsqueeze(-1)
        question_context = self.retriever.encode_questions(q0, valid)
        h0 = self.retriever.forward_with_interactions(q0, x0, valid, question_context)
        stage = self.training_stage
        if q0.is_cuda:
            torch.cuda.synchronize(q0.device)
        sampling_started = time.perf_counter()
        if stage in {"base_warmup", "diffusion_pretrain"}:
            candidates = q0.unsqueeze(-2).expand(q0.shape[:-1]+(self.generated_candidates+1, self.question_dim))
        else:
            candidates = build_candidates(self, parts, batch, generator)
        if q0.is_cuda:
            torch.cuda.synchronize(q0.device)
        sampling_seconds = time.perf_counter()-sampling_started
        pilot_logits = self.predictor(h0, candidates)
        pilot_logit = mixture_logit(pilot_logits, self.log_prior)
        r = responsibilities(pilot_logits, batch["answers"], self.log_prior, valid,
                             self.responsibility_stop_gradient, self.responsibility_temperature)
        if self.responsibility_mode == "prior":
            r = self.log_prior.exp().expand_as(r)*valid.unsqueeze(-1)
        mean, variance = candidate_moments(candidates, r)
        evidence = [mean-q0, variance]
        if self.surprise_input:
            evidence.append((batch["answers"]-pilot_logit.sigmoid()).unsqueeze(-1))
        correction = self.adapter(torch.cat(evidence, dim=-1)) if stage == "joint" else torch.zeros_like(x0)
        x1 = (x0+correction)*valid.unsqueeze(-1)
        h1 = self.retriever.forward_with_interactions(q0, x1, valid, question_context) if stage == "joint" else h0
        final_logits = self.predictor(h1, candidates) if stage == "joint" else pilot_logits
        final_logit = mixture_logit(final_logits, self.log_prior)
        kt = F.binary_cross_entropy_with_logits(final_logit[targets], batch["answers"][targets].float())
        pilot = F.binary_cross_entropy_with_logits(pilot_logit[targets], batch["answers"][targets].float())
        den, supervised = None, 0
        if self.training and stage != "base_warmup":
            den, supervised = self._profile_loss() if self.candidate_backend == "evidence_profile" else self._legacy_loss(parts, batch)
        if not self.training or stage == "base_warmup":
            total = kt
        elif stage == "diffusion_pretrain":
            total = den
        else:
            total = kt+self.pilot_weight*pilot+self.diffusion_weight*den
        self.loss_metric_weights = {"diffusion_loss": max(1, supervised)}
        raw_bce = F.binary_cross_entropy_with_logits(final_logits[..., 0][targets], batch["answers"][targets].float())
        result = dict(logits=final_logit, probabilities=final_logit.sigmoid(), states=h1, loss=total, kt_loss=kt,
            contrastive_loss=None, diffusion_loss=den, pilot_loss=pilot if self.training and stage == "joint" else None,
            candidate_probability_variance=final_logits.sigmoid().var(-1, unbiased=False)[targets].mean().detach(),
            responsibility_entropy=(-(r*r.clamp_min(1e-30).log()).sum(-1))[valid].mean().detach(),
            responsibility_prior_l1=(r-self.log_prior.exp()).abs().sum(-1)[valid].mean().detach(),
            raw_candidate_bce=raw_bce.detach(), candidate_norm=candidates.norm(dim=-1)[valid].mean().detach(),
            num_supervised_profile_keys=kt.new_tensor(supervised), sampling_seconds=kt.new_tensor(sampling_seconds))
        if return_details:
            result["details"] = dict(candidates=candidates, responsibilities=r, mean=mean, variance=variance,
                                     pilot_logits=pilot_logits, final_logits=final_logits, x0=x0, x1=x1, q0=q0, h0=h0)
        return result

    def experiment_metadata(self):
        return dict(candidate_backend=self.candidate_backend, learned_residual_enabled=self.learned_residual_enabled,
            generated_correction_enabled=self.generated_correction_enabled, residual_fusion=self.item_encoder.fusion,
            residual_concat_keep_2d=self.item_encoder.keep_2d, question_representation_dim=self.question_dim,
            profile_injection=self.profile_injection, profile_residual_condition=self.profile_residual_condition,
            legacy_residual_mode=self.legacy_residual_mode, legacy_residual_condition_detach=self.legacy_residual_condition_detach,
            generated_candidates=self.generated_candidates, raw_prior_mass=self.raw_prior_mass,
            responsibility_stop_gradient=self.responsibility_stop_gradient, responsibility_temperature=self.responsibility_temperature,
            responsibility_mode=self.responsibility_mode,
            surprise_input=self.surprise_input, training_stage=self.training_stage, stage_epoch=int(self.epoch_index), global_step=int(self.global_step),
            profile_manifest_sha256=self.reference_bank.manifest_sha256, profile_scope=self.reference_bank.manifest.get("training_scope"),
            diffusion_target="fixed_reference_profile" if self.candidate_backend == "evidence_profile" else "live_question",
            sampling_steps=self.diffusion_config["sampling_steps"], diffusion_schedule=self.diffusion_config["schedule"],
            terminal_alpha_bar=float(self.diffusion.alpha_bar[-1]), terminal_snr=float(self.diffusion.alpha_bar[-1]/(1-self.diffusion.alpha_bar[-1])),
            generator_target_width=self.profile["bins"] if self.candidate_backend == "evidence_profile" else getattr(self.diffusion,"output_dim",self.diffusion.dim),
            diffusion_settings=copy.deepcopy(self.diffusion_config), candidate_sampling_settings=copy.deepcopy(self.candidate_sampling),
            stage_schedule=copy.deepcopy(self.stages), profile_settings=copy.deepcopy(self.profile),
            generator_frozen=not any(p.requires_grad for p in self.diffusion.parameters()), parameter_count=self.parameter_count,
            baseline_parameter_count=self.baseline_parameter_count, parameter_ratio=self.parameter_ratio, max_parameter_ratio=self.max_parameter_ratio,
            configuration_fingerprint=self.configuration_fingerprint)
