"""Adam training, validation selection, checkpoints, and held-out evaluation."""
from __future__ import annotations
import copy
import csv
import json
import re
import time
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from src.data.interactions import InteractionDataset, collate_sequences
from src.model.diffuqkt import DiffuQKT
from src.training.results import rebuild_results_csv
from src.utils.config import project_path
from src.utils.files import sha256
from src.utils.metrics import binary_metrics
from src.utils.reproducibility import seed_everything, select_device


def model_kwargs(config, metadata):
    return dict(num_problems=metadata["num_problems"], num_skills=metadata["num_skills"],
                **config["model"], **config["loss"])


def make_loader(config, split, shuffle=False):
    control = config["train"]
    path = project_path(config["dataset"]["paths"]["processed"]) / f"{split}.npz"
    dataset = InteractionDataset(path, control.get("max_windows"), control.get("max_seq_len"))
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
    generator = None if training else torch.Generator(device=device).manual_seed(eval_seed)
    labels, probabilities, students, questions = [], [], [], []
    totals = dict(loss=0.0, kt_loss=0.0, contrastive_loss=0.0, diffusion_loss=0.0)
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
        mask = batch["prediction_mask"]
        number = int(mask.sum())
        for key in totals:
            # None denotes a training-only objective that was not evaluated.
            if result[key] is None:
                continue
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


def allocate_run_directory(output_root, dataset_name, epochs):
    """Reserve a new dataset/epoch directory without reusing an existing run."""
    if not isinstance(dataset_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset_name):
        raise ValueError("Dataset name must contain only letters, digits, underscores or hyphens.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    root = project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{epochs}epochs_"
    pattern = re.compile(re.escape(prefix) + r"([1-9][0-9]*)")
    indices = [0]
    for entry in root.iterdir():
        match = pattern.fullmatch(entry.name)
        if match:
            indices.append(int(match.group(1)))
    index = max(indices) + 1
    while True:
        directory = root / f"{prefix}{index}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            # Another run may have reserved the same number since the scan.
            index += 1


def evaluate_checkpoint(config, checkpoint_path, split="test", output_dir=None, verbose=True):
    control = config["train"]
    seed_everything(control["seed"], control["num_threads"])
    device = select_device(control["device"])
    config["train"]["device"] = str(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    data_dir = project_path(config["dataset"]["paths"]["processed"])
    if sha256(data_dir / "metadata.json") != checkpoint["metadata_sha256"]:
        raise ValueError("Checkpoint and processed data metadata differ; use the original ID mappings/splits.")
    # Preserve smoke/truncated-data scope unless the caller explicitly overrides it.
    for key in ["max_windows", "max_seq_len", "max_batches"]:
        if key not in control and key in checkpoint["config"]["train"]:
            control[key] = checkpoint["config"]["train"][key]
    model = DiffuQKT(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions = run_epoch(model, make_loader(config, split), device,
        eval_seed=control["eval_seed"], max_batches=control.get("max_batches"), return_predictions=True)
    if output_dir is not None:
        output_dir = project_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(output_dir / f"{split}_predictions.npz", **predictions)
    if verbose:
        print(json.dumps({"split": split, "checkpoint_epoch": checkpoint["epoch"], **metrics}, indent=2), flush=True)
    return metrics


def train(config, overwrite=False):
    # Keep resolved output paths local so repeated calls can allocate new runs.
    config = copy.deepcopy(config)
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
                                         config["dataset"]["name"], control["epochs"])
        try:
            control["output_dir"] = output.relative_to(project_path(".").resolve()).as_posix()
        except ValueError:
            control["output_dir"] = str(output)
    if (output / "best.pt").exists() and not overwrite:
        raise FileExistsError(f"{output} already contains a run; choose --output-dir or --overwrite.")
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
    train_loader = make_loader(config, "train", True)
    kwargs = model_kwargs(config, metadata)
    model = DiffuQKT(**kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=control["learning_rate"],
                                 weight_decay=control["weight_decay"])
    valid_loader = make_loader(config, "valid")
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"train windows: {len(train_loader.dataset)}; validation windows: {len(valid_loader.dataset)}", flush=True)
    history, best, best_epoch, wait = [], -float("inf"), 0, 0
    for epoch in range(1, control["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, control["grad_clip"],
                                  max_batches=control.get("max_batches"))
        valid_metrics = run_epoch(model, valid_loader, device, eval_seed=control["eval_seed"],
                                  max_batches=control.get("max_batches"))
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                  **{f"valid_{k}": v for k, v in valid_metrics.items()}}
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
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": config, "model_kwargs": kwargs, "validation": valid_metrics,
                      "metadata_sha256": metadata_fingerprint, "best_epoch": best_epoch}
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        cl = train_metrics['contrastive_loss']
        diff = train_metrics['diffusion_loss']
        print(f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
              f"(KT {train_metrics['kt_loss']:.4f}, "
              f"CL {cl:.4f}, Diff {diff:.4f}) "
              f"AUC {train_metrics['auc']} | valid BCE {valid_metrics['bce']:.4f} "
              f"AUC {valid_metrics['auc']} ACC {valid_metrics['accuracy']:.4f} "
              f"| {train_metrics['seconds'] + valid_metrics['seconds']:.1f}s", flush=True)
        if wait >= control["patience"]:
            print(f"Early stopping after {wait} epochs without improvement.", flush=True)
            break
    # The test split is used once, only after choosing the best validation checkpoint.
    best_checkpoint = load_checkpoint(output / "best.pt", device)
    validation = best_checkpoint["validation"]
    (output / "valid_metrics.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    test = evaluate_checkpoint(config, output / "best.pt", "test", output, verbose=False)
    duration_seconds = time.perf_counter() - training_started
    best_train = {key[len("train_"):]: value for key, value in history[best_epoch-1].items()
                  if key.startswith("train_")}
    summary = {"best_epoch": best_epoch, "epochs_completed": len(history),
               "selection_metric": "validation_auc" if validation["auc"] is not None else "validation_bce",
               "best_validation": history[best_epoch-1], "train": best_train,
               "validation": validation, "test": test, "duration_seconds": duration_seconds,
               "device": str(device), "seed": control["seed"],
               "limited_run": any(control.get(k) is not None for k in ["max_windows", "max_seq_len", "max_batches"]),
               "metadata_sha256": metadata_fingerprint, "model_kwargs": kwargs,
               "output_dir": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        results_path = rebuild_results_csv(output.parent)
        print(f"Experiment results: {results_path}", flush=True)
    except OSError as error:
        # Excel can hold the CSV open on Windows; all per-run results remain saved.
        warnings.warn(f"Could not update results.csv: {error}. Close the CSV and run "
                      f'python -m src.training.results --root "{output.parent}".', stacklevel=2)
    print(json.dumps({key: summary[key] for key in ["epochs_completed", "best_epoch", "selection_metric",
                                                   "validation", "test"]}, indent=2), flush=True)
    print(f"Saved run: {output}", flush=True)
    return summary
