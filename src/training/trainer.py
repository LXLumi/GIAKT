"""Adam training, validation selection, checkpoints, and held-out evaluation."""
from __future__ import annotations
import copy
import csv
from datetime import datetime
import json
import re
import random
import time
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from src.data.interactions import InteractionDataset, collate_sequences
from src.model.registry import build_model, checkpoint_model_name
from src.training.results import CSV_DECIMAL_PLACES, rebuild_results_csv
from src.utils.config import project_path, resolve_model_name
from src.utils.files import sha256
from src.utils.metrics import binary_metrics
from src.utils.reproducibility import seed_everything, select_device


def _display_metrics(metrics):
    """Format a console-only copy; selection and saved metrics keep full precision."""
    fields = {"auc", "accuracy", "bce", "loss", "kt_loss", "contrastive_loss", "diffusion_loss"}
    return {key: f"{value:.4f}" if key in fields and value is not None else value
            for key, value in metrics.items()}


def model_kwargs(config, metadata):
    return dict(num_problems=metadata["num_problems"], num_skills=metadata["num_skills"],
                **config["model"], **config["loss"])


def make_loader(config, split, shuffle=False):
    control = config["train"]
    path = project_path(config["dataset"]["paths"]["processed"]) / f"{split}.npz"
    dataset = InteractionDataset(path, control.get("max_windows"), control.get("max_seq_len"))
    dataset.data_split = split
    generator = torch.Generator().manual_seed(control["seed"])
    return DataLoader(dataset, batch_size=control["batch_size"], shuffle=shuffle,
                      num_workers=control["num_workers"], collate_fn=collate_sequences,
                      generator=generator, pin_memory=control["device"].startswith("cuda"))


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(model, loader, device, optimizer=None, grad_clip=15.0, eval_seed=2026,
              max_batches=None, return_predictions=False):
    training = optimizer is not None
    model.train(training)
    context_hook = getattr(model, "set_data_context", None)
    if callable(context_hook):
        split = getattr(getattr(loader, "dataset", None), "data_split", None)
        if split is None:
            path = getattr(getattr(loader, "dataset", None), "path", None)
            split = path.stem if path is not None else getattr(model, "data_context", None)
        context_hook(split)
    generator = None if training else torch.Generator(device=device).manual_seed(eval_seed)
    labels, probabilities, students, questions = [], [], [], []
    totals = dict(loss=0.0, kt_loss=0.0, contrastive_loss=0.0, diffusion_loss=0.0)
    totals.update({name: 0.0 for name in getattr(model, "extra_metric_names", ())})
    weights = {key: 0 for key in totals}
    started, batches = time.perf_counter(), 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            result = model(batch, generator=generator)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"Non-finite loss in batch {index}.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Non-finite gradient in batch {index}.")
                optimizer.step()
                step_hook = getattr(model, "on_optimizer_step", None)
                if callable(step_hook):
                    step_hook()
        mask = batch["prediction_mask"]
        number = int(mask.sum())
        for key in totals:
            # None denotes a training-only objective that was not evaluated.
            if result.get(key) is None:
                continue
            metric_weights = getattr(model, "loss_metric_weights", {})
            weight = metric_weights.get(key)
            if weight is None:
                # Preserve DiffuQKT's original event/target-weighted reporting.
                weight = int(batch["valid_mask"].sum()) if key == "diffusion_loss" else number
            totals[key] += float(result[key].detach()) * weight
            weights[key] += weight
        labels.extend(batch["answers"][mask].detach().cpu().tolist())
        probabilities.extend(result["probabilities"][mask].detach().cpu().tolist())
        if return_predictions:
            students.extend(batch["student_ids"][:, None].expand_as(mask)[mask].cpu().tolist())
            questions.extend(batch["problem_ids"][mask].cpu().tolist())
        batches += 1
    metrics = binary_metrics(labels, probabilities)
    metrics.update({key: totals[key] / weights[key] if weights[key] else None for key in totals})
    metrics.update(seconds=time.perf_counter() - started, batches=batches,
                   loss_mode="joint_training" if training else "prediction_only")
    # Training total is target-weighted batch objective; each component uses its own mask.
    if return_predictions:
        return metrics, dict(labels=np.asarray(labels), probabilities=np.asarray(probabilities),
                             studentId=np.asarray(students), problemId=np.asarray(questions))
    return metrics


def load_checkpoint(path, device):
    # Only load trusted checkpoints produced by this project (torch.load uses pickle).
    return torch.load(project_path(path), map_location=device, weights_only=False)


def allocate_run_directory(output_root, dataset_name, epochs, model_name="diffuqkt"):
    """Reserve a run under its model/dataset group, retaining legacy run names."""
    if not isinstance(dataset_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset_name):
        raise ValueError("Dataset name must contain only letters, digits, underscores or hyphens.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    model_name = resolve_model_name(model_name)
    root = project_path(output_root)
    model_directory = "diffuqk" if model_name == "diffuqkt" else model_name
    group = root / model_directory / dataset_name
    group.mkdir(parents=True, exist_ok=True)
    model_prefix = "" if model_name == "diffuqkt" else f"{model_name}_"
    prefix = f"{model_prefix}{dataset_name}_{epochs}epochs_"
    pattern = re.compile(re.escape(prefix) + r"([1-9][0-9]*)")
    indices = [0]
    # Flat legacy runs also reserve their numbers until they are moved into a group.
    for parent in (root, group):
        for entry in parent.iterdir():
            match = pattern.fullmatch(entry.name)
            if match:
                indices.append(int(match.group(1)))
    index = max(indices) + 1
    while True:
        directory = group / f"{prefix}{index}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            # Another run may have reserved the same number since the scan.
            index += 1


def evaluate_checkpoint(config, checkpoint_path, split="test", output_dir=None, verbose=True,
                        expected_model=None):
    control = config["train"]
    seed_everything(control["seed"], control["num_threads"])
    device = select_device(control["device"])
    config["train"]["device"] = str(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    saved_model = checkpoint_model_name(checkpoint)
    if expected_model is not None and resolve_model_name(expected_model) != saved_model:
        raise ValueError(f"Checkpoint model is {saved_model}, but --model selected {expected_model}.")
    data_dir = project_path(config["dataset"]["paths"]["processed"])
    if sha256(data_dir / "metadata.json") != checkpoint["metadata_sha256"]:
        raise ValueError("Checkpoint and processed data metadata differ; use the original ID mappings/splits.")
    # Preserve smoke/truncated-data scope unless the caller explicitly overrides it.
    for key in ["max_windows", "max_seq_len", "max_batches"]:
        if key not in control and key in checkpoint["config"]["train"]:
            control[key] = checkpoint["config"]["train"][key]
    model = build_model(saved_model, checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions = run_epoch(model, make_loader(config, split), device,
        eval_seed=control["eval_seed"], max_batches=control.get("max_batches"), return_predictions=True)
    if output_dir is not None:
        output_dir = project_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(output_dir / f"{split}_predictions.npz", **predictions)
    if verbose:
        print(json.dumps({"model_name": saved_model, "split": split,
                          "checkpoint_epoch": checkpoint["epoch"], **_display_metrics(metrics)},
                         indent=2), flush=True)
    return metrics


def train(config, overwrite=False):
    # Keep resolved output paths local so repeated calls can allocate new runs.
    config = copy.deepcopy(config)
    config["model_name"] = resolve_model_name(config.get("model_name", "diffuqkt"))
    control = config["train"]
    if control["epochs"] < 1 or control["patience"] < 1 or control["batch_size"] < 1:
        raise ValueError("epochs, patience and batch_size must be positive.")
    seed_everything(control["seed"], control["num_threads"])
    device = select_device(control["device"])
    control["device"] = str(device)
    if control.get("output_dir"):
        output = project_path(control["output_dir"])
    else:
        output = allocate_run_directory(control.get("output_root", "experiments"),
                                         config["dataset"]["name"], control["epochs"], config["model_name"])
        try:
            control["output_dir"] = output.relative_to(project_path(".").resolve()).as_posix()
        except ValueError:
            control["output_dir"] = str(output)
    if (output / "best.pt").exists() and not overwrite:
        raise FileExistsError(f"{output} already contains a run; choose --output-dir or --overwrite.")
    configured_root = project_path(control.get("output_root", "experiments")).resolve()
    results_root = configured_root if output.resolve().is_relative_to(configured_root) else output.parent
    print(f"Run directory: {output}", flush=True)
    data_dir = project_path(config["dataset"]["paths"]["processed"])
    if control.get("preprocess_each_run", False) or not (data_dir / "metadata.json").exists():
        from scripts.preprocessing import preprocess_dataset
        print("Preparing canonical data and student splits...", flush=True)
        preprocess_dataset(config)
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Processed data missing. Run main.py --config <dataset.yaml> --preprocess.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    metadata_fingerprint = sha256(metadata_path)
    # Exclude static preprocessing; include data loading, CUDA setup and final test.
    training_started = time.perf_counter()
    resumed = None
    if control.get("resume_checkpoint"):
        resumed = load_checkpoint(control["resume_checkpoint"], device)
        if checkpoint_model_name(resumed) != config["model_name"] or resumed["metadata_sha256"] != metadata_fingerprint:
            raise ValueError("Resume checkpoint model/data metadata mismatch.")
        for key in ["max_windows", "max_seq_len", "max_batches"]:
            if key not in control and key in resumed["config"]["train"]:
                control[key] = resumed["config"]["train"][key]
        config["model"] = copy.deepcopy(resumed["config"]["model"])
        config["loss"] = copy.deepcopy(resumed["config"]["loss"])
    train_loader = make_loader(config, "train", True)
    kwargs = copy.deepcopy(resumed["model_kwargs"]) if resumed is not None else model_kwargs(config, metadata)
    model = build_model(config["model_name"], kwargs).to(device)
    if resumed is not None:
        if not getattr(model, "supports_resume", False) or "rng_state" not in resumed:
            raise ValueError("This checkpoint does not support full training-state resume.")
        if control["epochs"] <= resumed["epoch"]:
            raise ValueError("--epochs must exceed the saved epoch when resuming.")
        model.load_state_dict(resumed["model"], strict=True)
    prepare_training = getattr(model, "prepare_training", None)
    prepared_kwargs = {}
    if callable(prepare_training):
        # Each model owns its preparation; the shared trainer only persists its arguments.
        prepared_kwargs = prepare_training(train_loader.dataset)
        kwargs.update(prepared_kwargs)
        if "train_item_ids" in kwargs:
            print(f"Model regularization training items: {len(kwargs['train_item_ids']):,}", flush=True)
    metadata_hook = getattr(model, "experiment_metadata", None)
    model_metadata = metadata_hook() if callable(metadata_hook) else {}
    if model_metadata:
        # Models opting into run metadata also persist normalized constructor options.
        config["model"].update(prepared_kwargs)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Model experiment settings: {json.dumps(model_metadata, sort_keys=True)}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=control["learning_rate"],
                                 weight_decay=control["weight_decay"])
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
    valid_loader = make_loader(config, "valid")
    print(f"Model: {config['model_name']}; device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"train windows: {len(train_loader.dataset)}; validation windows: {len(valid_loader.dataset)}", flush=True)
    history, best, best_epoch, wait = [], -float("inf"), 0, 0
    start_epoch, selection_group = 1, None
    if resumed is not None:
        history = copy.deepcopy(resumed["history"])
        best, best_epoch, wait = resumed["best_score"], resumed["best_epoch"], resumed["selection_wait"]
        start_epoch, selection_group = resumed["epoch"]+1, resumed["training_stage"]
        old_best_path = project_path(control["resume_checkpoint"]).parent/"best.pt"
        if not old_best_path.exists():
            raise FileNotFoundError("Resume requires best.pt beside the checkpoint to preserve validation selection.")
        torch.save(load_checkpoint(old_best_path, device), output/"best.pt")
        rng = resumed["rng_state"]
        torch.set_rng_state(rng["torch"].cpu())
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        train_loader.generator.set_state(rng["loader"].cpu())
        if device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    for epoch in range(start_epoch, control["epochs"] + 1):
        epoch_hook = getattr(model, "on_train_epoch_start", None)
        if callable(epoch_hook):
            epoch_hook(epoch)
        group_hook = getattr(model, "selection_group", None)
        if callable(group_hook) and group_hook() != selection_group:
            selection_group = group_hook()
            best, wait = -float("inf"), 0
            print(f"Training stage: {selection_group}; validation selection starts within this stage.", flush=True)
        train_metrics = run_epoch(model, train_loader, device, optimizer, control["grad_clip"],
                                  max_batches=control.get("max_batches"))
        valid_metrics = run_epoch(model, valid_loader, device, eval_seed=control["eval_seed"],
                                  max_batches=control.get("max_batches"))
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        if callable(group_hook):
            record["training_stage"] = selection_group
        history.append(record)
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        score = valid_metrics["auc"] if valid_metrics["auc"] is not None else -valid_metrics["bce"]
        improved = score > best
        if improved:
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        checkpoint = {"epoch": epoch, "model_name": config["model_name"],
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": config, "model_kwargs": kwargs, "validation": valid_metrics,
                      "metadata_sha256": metadata_fingerprint, "best_epoch": best_epoch}
        if callable(metadata_hook):
            model_metadata = metadata_hook()
        checkpoint.update(model_metadata)
        if getattr(model, "supports_resume", False):
            checkpoint.update(history=copy.deepcopy(history), best_score=best, selection_wait=wait,
                rng_state=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                               loader=train_loader.generator.get_state(),
                               cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        loss_parts = [f"KT {train_metrics['kt_loss']:.4f}"]
        for label, key in [("CL", "contrastive_loss"), ("Diff", "diffusion_loss")]:
            if train_metrics.get(key) is not None:
                loss_parts.append(f"{label} {train_metrics[key]:.4f}")
        if train_metrics.get("pilot_loss") is not None:
            loss_parts.append(f"Pilot {train_metrics['pilot_loss']:.4f}")
        train_display, valid_display = _display_metrics(train_metrics), _display_metrics(valid_metrics)
        prefix = f"Epoch {epoch:03d} | "
        print(f"{prefix}train | loss {train_display['loss']:>9} "
              f"| AUC {train_display['auc'] or 'NA':>6} | ACC {train_display['accuracy'] or 'NA':>6} "
              f"| ({', '.join(loss_parts)})\n"
              f"{' ' * len(prefix)}valid | loss {valid_display['loss']:>9} "
              f"| AUC {valid_display['auc'] or 'NA':>6} | ACC {valid_display['accuracy'] or 'NA':>6} "
              f"| {train_metrics['seconds'] + valid_metrics['seconds']:.1f}s", flush=True)
        if wait >= control["patience"] and (not callable(group_hook) or selection_group == "joint"):
            print(f"Early stopping after {wait} epochs without improvement.", flush=True)
            break
    # The test split is used once, only after choosing the best validation checkpoint.
    best_checkpoint = load_checkpoint(output / "best.pt", device)
    validation = best_checkpoint["validation"]
    (output / "valid_metrics.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    test = evaluate_checkpoint(config, output / "best.pt", "test", output, verbose=False)
    duration_seconds = time.perf_counter() - training_started
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    best_train = {key[len("train_"):]: value for key, value in history[best_epoch-1].items()
                  if key.startswith("train_")}
    summary = {"model_name": config["model_name"], "best_epoch": best_epoch, "epochs_completed": len(history),
               "selection_metric": "validation_auc" if validation["auc"] is not None else "validation_bce",
               "best_validation": history[best_epoch-1], "train": best_train,
               "validation": validation, "test": test, "duration_seconds": duration_seconds,
               "finished_at": finished_at,
               "results_decimal_places": CSV_DECIMAL_PLACES,
               "device": str(device), "seed": control["seed"],
               "limited_run": any(control.get(k) is not None for k in ["max_windows", "max_seq_len", "max_batches"]),
               "metadata_sha256": metadata_fingerprint, "model_kwargs": kwargs,
               "output_dir": str(output)}
    summary.update(model_metadata)
    if callable(getattr(model, "selection_group", None)):
        summary["stage_seconds"] = {stage: sum(row["train_seconds"]+row["valid_seconds"] for row in history if row["training_stage"] == stage)
                                    for stage in {row["training_stage"] for row in history}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        results_path = rebuild_results_csv(results_root)
        print(f"Experiment results: {results_path}", flush=True)
    except (OSError, UnicodeError, csv.Error) as error:
        # Excel can hold the CSV open on Windows; all per-run results remain saved.
        warnings.warn(f"Could not update results.csv: {error}. Close the CSV and run "
                      f'python -m src.training.results --root "{results_root}".', stacklevel=2)
    display_summary = {key: summary[key] for key in
                       ["model_name", "epochs_completed", "best_epoch", "selection_metric"]}
    display_summary.update(validation=_display_metrics(validation), test=_display_metrics(test))
    print(json.dumps(display_summary, indent=2), flush=True)
    print(f"Saved run: {output}", flush=True)
    return summary
