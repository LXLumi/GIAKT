"""Read-only saved-weight analysis of NewQ V2; no optimizer or artifact rewrites."""
import argparse
import copy
import json
import math

import torch
from torch.nn import functional as F

from src.model.registry import build_model, checkpoint_model_name
from src.training.trainer import load_checkpoint, make_loader, move_batch
from src.utils.config import load_config, project_path
from src.utils.files import sha256
from src.utils.metrics import binary_metrics
from src.utils.reproducibility import seed_everything, select_device


def distribution(values):
    """Report finite-value quantiles and explicit counts for any nonfinite values."""
    values = values.detach().double().flatten().cpu()
    finite = values[torch.isfinite(values)]
    result = {"count": values.numel(), "nonfinite_count": int((~torch.isfinite(values)).sum())}
    result.update({name: None for name in ["mean", "p50", "p95", "p99", "max"]})
    if finite.numel():
        result.update(mean=float(finite.mean()), max=float(finite.max()))
        for name, probability in [("p50", .5), ("p95", .95), ("p99", .99)]:
            result[name] = float(torch.quantile(finite, probability))
    return result


def prediction_metrics(logits, answers):
    """Keep the existing probability BCE and the stable logit BCE distinct."""
    logits, answers = logits.detach().cpu(), answers.detach().cpu()
    finite = bool(torch.isfinite(logits).all())
    stable = float(F.binary_cross_entropy_with_logits(logits.double(), answers.double()))
    result = {"stable_bce_from_logits": stable if math.isfinite(stable) else None,
              "stable_bce_is_finite": math.isfinite(stable),
              "nonfinite_logit_count": int((~torch.isfinite(logits)).sum()),
              "max_abs_logit": float(logits.abs().max()) if finite else None}
    if finite:
        metrics = binary_metrics(answers.tolist(), logits.sigmoid().tolist())
        result.update(auc=metrics["auc"], accuracy=metrics["accuracy"],
                      probability_bce=metrics["bce"], num_predictions=metrics["num_predictions"])
    else:
        result.update(auc=None, accuracy=None, probability_bce=None, num_predictions=answers.numel())
    return result


def gradient_probe(model, batch, seed):
    """Probe three losses on one training batch without stepping any parameter."""
    groups = ["attention_question_embedding", "key_projection", "residual_question_embedding",
              "residual_projection", "concept", "variation", "difficulty", "diffusion", "retriever", "predictor"]
    if hasattr(model, "residual_fusion_projection"):
        groups.append("residual_fusion_projection")
    report = {}
    model.train()
    for loss_name in ["kt_loss", "contrastive_loss", "diffusion_loss"]:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        model(batch)[loss_name].backward()
        stats = {}
        for name in groups:
            gradients = [parameter.grad.detach().double() for parameter in getattr(model, name).parameters()
                         if parameter.grad is not None]
            norm = math.sqrt(sum(float(value.square().sum()) for value in gradients)) if gradients else None
            stats[name] = {"has_gradient": bool(gradients),
                           "norm": norm if norm is None or math.isfinite(norm) else None,
                           "nonfinite_count": sum(int((~torch.isfinite(value)).sum()) for value in gradients)}
        report[loss_name] = stats
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"split": "train", "batches": 1, "windows": batch["problem_ids"].shape[0],
            "valid_interactions": int(batch["valid_mask"].sum()),
            "prediction_targets": int(batch["prediction_mask"].sum()), "objectives": report}


def diagnose_checkpoint(config, checkpoint_path, split="valid", max_batches=2):
    """Full reverse sampling in a bounded data scope, followed by a single gradient probe."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive.")
    config = copy.deepcopy(config)
    control = config["train"]
    seed_everything(control["seed"], control["num_threads"])
    device = select_device(control["device"])
    control["device"] = str(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint_model_name(checkpoint) != "newq_v2":
        raise ValueError("This diagnostic requires a newq_v2 checkpoint.")
    metadata_path = project_path(config["dataset"]["paths"]["processed"]) / "metadata.json"
    if sha256(metadata_path) != checkpoint["metadata_sha256"]:
        raise ValueError("Checkpoint and processed data metadata differ.")
    for key in ["max_windows", "max_seq_len"]:
        if key not in control and key in checkpoint["config"]["train"]:
            control[key] = checkpoint["config"]["train"][key]
    model = build_model("newq_v2", checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    metadata = model.experiment_metadata()
    generated_norm_name = f"{metadata['generated_representation_kind']}_generated_norm"
    loader = make_loader(config, split)
    generator = torch.Generator(device=device).manual_seed(control["eval_seed"])
    collected = {key: [] for key in ["entropy", "uniform_deviation", "residual_ratio", generated_norm_name, "enhanced_norm"]}
    labels, logits, examples = [], [], []
    multi_items, items = set(), set()
    batches, interactions = 0, 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            batch = move_batch(batch, device)
            valid, targets = batch["valid_mask"], batch["prediction_mask"] & batch["valid_mask"]
            parts = model.question_components(batch)
            generated, enhanced, _ = model.generate_questions(parts, batch, generator)
            states = model.retriever(enhanced, model.answer(batch["answers"]), valid)
            predicted = model.predictor(torch.cat([states, enhanced], dim=-1)).squeeze(-1)
            logits.append(predicted[targets].cpu())
            labels.append(batch["answers"][targets].cpu())
            counts = parts["concept_mask"].sum(-1)
            multi = valid & counts.gt(1)
            weights = parts["concept_weights"]
            if multi.any():
                entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1)[multi] / counts[multi].float().log()
                uniform = parts["concept_mask"].to(weights.dtype) / counts.unsqueeze(-1).clamp_min(1)
                deviation = (weights-uniform).abs().max(-1).values[multi]
                collected["entropy"].append(entropy.cpu())
                collected["uniform_deviation"].append(deviation.cpu())
                multi_items.update(batch["problem_ids"][multi].tolist())
                for row, position in multi.nonzero().tolist():
                    if len(examples) >= 8:
                        break
                    mask = parts["concept_mask"][row, position]
                    examples.append({"problem_id": int(batch["problem_ids"][row, position]),
                                     "concept_ids": parts["concept_ids"][row, position][mask].tolist(),
                                     "weights": weights[row, position][mask].tolist()})
            ratio = parts["residual"].norm(dim=-1) / parts["semantic_original"].norm(dim=-1).clamp_min(1e-8)
            collected["residual_ratio"].append(ratio[valid].cpu())
            collected[generated_norm_name].append(generated.norm(dim=-1)[valid].cpu())
            collected["enhanced_norm"].append(enhanced.norm(dim=-1)[valid].cpu())
            items.update(batch["problem_ids"][valid].tolist())
            interactions += int(valid.sum())
            batches += 1
    if not logits:
        raise ValueError("No prediction batches in diagnostic scope.")
    stats = {key: distribution(torch.cat(values) if values else torch.empty(0)) for key, values in collected.items()}
    training_batch = move_batch(next(iter(make_loader(config, "train"))), device)
    gradients = gradient_probe(model, training_batch, control["seed"])
    return {**metadata, "model_name": "newq_v2", "checkpoint": str(project_path(checkpoint_path)),
            "checkpoint_epoch": checkpoint["epoch"], "device": str(device), "split": split,
            "sampling_steps": model.diffusion.steps, "parameter_count": sum(p.numel() for p in model.parameters()),
            "baseline_parameter_count": model.baseline_parameter_count, "parameter_ratio": model.parameter_ratio,
            "weighting": "interaction; valid positions include the initial context; attention excludes K<=1",
            "residual_ratio_denominator_epsilon": 1e-8,
            "scope": {"batches": batches, "loaded_windows": len(loader.dataset), "valid_interactions": interactions,
                      "distinct_items": len(items), "multi_concept_distinct_items": len(multi_items),
                      "max_windows": control.get("max_windows"), "max_seq_len": control.get("max_seq_len")},
            "statistics": stats, "attention_examples": examples,
            "predictions": prediction_metrics(torch.cat(logits), torch.cat(labels)), "gradient_probe": gradients}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/assistment2009.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Optional report path; default is stdout only.")
    args = parser.parse_args(argv)
    config = load_config(args.config, model_name="newq_v2")
    config["train"]["device"] = args.device
    for name in ["max_windows", "max_seq_len"]:
        value = getattr(args, name)
        if value is not None:
            config["train"][name] = value
    report = diagnose_checkpoint(config, args.checkpoint, args.split, args.max_batches)
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        output = project_path(args.output)
        protected = project_path(args.checkpoint).resolve()
        if output.resolve() == protected:
            raise ValueError("Diagnostic output must not overwrite the checkpoint.")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)
    return report


if __name__ == "__main__":
    main()
