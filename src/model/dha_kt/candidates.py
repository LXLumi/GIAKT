"""One graph per unique legal condition key, with stable per-key candidate noise."""
import hashlib
import json

import torch

from .diffusion import LegacySampleCall


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()[:8], "little") % (2**63-1)


class KeyedNoise:
    def __init__(self, keys, width, reference, seed):
        self.generators = [torch.Generator().manual_seed(stable_seed((seed, key))) for key in keys]
        self.width, self.reference = width, reference

    def __call__(self, reference=None):
        noise = torch.stack([torch.randn(self.width, generator=g) for g in self.generators]).to(self.reference)
        return noise.reshape(reference.shape[:-1]+(self.width,)) if reference is not None else noise


def unique_conditions(model, components, batch):
    valid = batch["valid_mask"]
    if model.candidate_backend == "evidence_profile":
        student_views = model.reference_bank.views_for_students(batch["student_ids"], model.data_context)
    else:
        student_views = torch.zeros_like(batch["student_ids"])
    draw = model.condition_draw()
    problems = batch["problem_ids"].detach().cpu().reshape(-1).tolist()
    skills = components["canonical_skill_ids"].detach().cpu().reshape(-1, components["canonical_skill_ids"].shape[-1]).tolist()
    views = student_views[:, None].expand_as(valid).reshape(-1).cpu().tolist()
    keys, first, inverse, seen = [], [], [], {}
    positions = valid.reshape(-1).nonzero().flatten()
    for position in positions.cpu().tolist():
        key = (views[position], draw, problems[position], tuple(sorted({x for x in skills[position] if x})))
        if key not in seen:
            seen[key] = len(keys)
            keys.append(key)
            first.append(position)
        inverse.append(seen[key])
    indices = torch.tensor(first, device=valid.device)
    selected = {name: components[name].reshape(-1, components[name].shape[-1])[indices]
                for name in ["concept", "difficulty", "semantic", "residual", "original"]}
    return keys, selected, positions, torch.tensor(inverse, device=valid.device)


def generate_unique(model, keys, parts, generator, return_correction=False):
    count, candidates = len(keys), model.generated_candidates
    q0 = parts["original"]
    if not model.generated_correction_enabled:
        result = q0[:, None, :].expand(count, candidates+1, -1)
        return (result, torch.zeros_like(parts["residual"])[:, None, :].expand(-1, candidates, -1)) if return_correction else result
    seed = generator.initial_seed() if generator is not None else model.candidate_sampling["seed"]
    salt = int(model.epoch_index) if model.training else 0
    manifest_key = model.reference_bank.manifest_sha256
    noise_keys = [(key, candidate, model.candidate_backend, model.legacy_residual_mode,
                   model.item_encoder.fusion, model.configuration_fingerprint, manifest_key, salt)
                  for key in keys for candidate in range(1, candidates+1)]
    def repeat(value):
        return value.repeat_interleave(candidates, dim=0)
    if model.candidate_backend == "evidence_profile":
        views = torch.tensor([key[0] for key in keys], device=q0.device)
        problems = torch.tensor([key[2] for key in keys], device=q0.device)
        partial, counts, observed = model.reference_bank.partial(views, keys[0][1], problems)
        structure = torch.cat([parts["concept"], parts["semantic"], parts["difficulty"]], dim=-1)
        residual = model.profile_condition(parts["residual"])
        noise = KeyedNoise(noise_keys, model.profile["bins"], q0, seed)
        sampled = model.diffusion.sample(repeat(structure), repeat(partial), repeat(counts), repeat(observed), noise(),
                                         None if residual is None else repeat(residual), step_noise=lambda _: noise())
        sampled = sampled/model.profile["fixed_scale"]
        correction = model.profile_to_residual(sampled).reshape(count, candidates, -1)
        if model.profile["no_evidence_policy"] == "raw_only":
            correction = correction*observed.any(-1)[:, None, None]
        semantic, residual = parts["semantic"][:, None, :], parts["residual"][:, None, :]
        if model.profile_injection == "residual_update":
            generated = model.item_encoder.fuse(semantic.expand_as(correction), residual+correction)
        else:
            generated = model.item_encoder.fuse(semantic+correction, residual.expand_as(correction))
    else:
        correction = None
        condition = parts["residual"] if model.legacy_residual_mode in {"condition", "both"} else None
        if condition is not None and model.legacy_residual_condition_detach:
            condition = condition.detach()
        width = getattr(model.diffusion, "output_dim", model.diffusion.dim)
        noise = KeyedNoise(noise_keys, width, q0, seed)
        call = LegacySampleCall(model.diffusion, None if condition is None else repeat(condition)[:, None, :], noise)
        concept, difficulty = repeat(parts["concept"])[:, None, :], repeat(parts["difficulty"])[:, None, :]
        valid = torch.ones(concept.shape[:2], device=concept.device, dtype=torch.bool)
        sampled = call.sample(concept, difficulty, valid).squeeze(1).reshape(count, candidates, -1)
        if model.legacy_residual_mode in {"target", "both"}:
            generated = sampled
        else:
            generated = model.item_encoder.fuse(sampled, parts["residual"][:, None, :].expand_as(sampled))
    result = torch.cat([q0[:, None, :], generated], dim=1)
    return (result, correction) if return_correction else result


def build_candidates(model, components, batch, generator=None):
    keys, parts, positions, inverse = unique_conditions(model, components, batch)
    cache_allowed = not model.training and not torch.is_grad_enabled() and model.candidate_sampling["eval_cache"] == "checkpoint_keyed"
    if cache_allowed:
        snapshot = model.evaluation_snapshot(generator)
        cache_keys = [(snapshot, key) for key in keys]
        missing = [index for index, key in enumerate(cache_keys) if key not in model._candidate_cache]
        if missing:
            selected = torch.tensor(missing, device=positions.device)
            generated = generate_unique(model, [keys[i] for i in missing], {k:v[selected] for k,v in parts.items()}, generator)
            for index, values in zip(missing, generated):
                model._candidate_cache[cache_keys[index]] = values.detach()
        unique = torch.stack([model._candidate_cache[key] for key in cache_keys])
    else:
        unique = generate_unique(model, keys, parts, generator)
    flat = components["original"].new_zeros((batch["valid_mask"].numel(), model.generated_candidates+1, model.question_dim))
    flat = flat.index_copy(0, positions, unique[inverse])
    return flat.reshape(batch["valid_mask"].shape+(model.generated_candidates+1, model.question_dim))
