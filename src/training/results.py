"""Summarize completed experiments in a three-row-per-run CSV."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

from src.utils.config import project_path


CSV_NAME = "results.csv"
CSV_COLUMNS = ("experiment", "split", "AUC", "ACC", "loss", "time")


def format_duration(seconds):
    """Display elapsed time as hours and remaining seconds, e.g. 1h65s."""
    if seconds is None:
        return ""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration_seconds must be finite and non-negative.")
    hours, remaining = divmod(int(seconds + 0.5), 3600)
    return f"{hours}h{remaining}s"


def _number(value):
    if value is None or value == "":
        return ""
    value = float(value)
    return f"{value:.8f}" if math.isfinite(value) else ""


def _best_record(summary, directory):
    """Support older summaries and fall back to the selected history row."""
    record = summary.get("best_validation") or {}
    best_epoch = summary["best_epoch"]
    if (str(record.get("epoch", best_epoch)) == str(best_epoch)
            and any(key.startswith("train_") for key in record)):
        return record
    history_path = directory / "history.csv"
    if history_path.exists():
        with history_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("epoch") == str(best_epoch):
                    return row
    return {}


def _split_metrics(record, split):
    prefix = f"{split}_"
    return {key[len(prefix):]: value for key, value in record.items()
            if key.startswith(prefix)}


def experiment_rows(directory):
    """Read best-epoch metrics without reevaluating or editing the experiment."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
    record = _best_record(summary, directory)
    train = {**_split_metrics(record, "train"), **(summary.get("train") or {})}
    valid = {**_split_metrics(record, "valid"), **(summary.get("validation") or {})}
    test = summary["test"]
    if not isinstance(test, dict):
        raise ValueError("A completed summary must contain test metrics.")
    duration = format_duration(summary.get("duration_seconds"))
    # This tuple contains raw component losses, not their weighted sum.
    components = [_number(train.get(key)) or "NA" for key in
                  ("kt_loss", "contrastive_loss", "diffusion_loss")]
    rows = []
    for split, metrics in (("train", train), ("valid", valid), ("test", test)):
        loss = (f"({', '.join(components)})" if split == "train"
                else _number(metrics.get("loss")))
        rows.append({"experiment": directory.name if split == "train" else "",
                     "split": split, "AUC": _number(metrics.get("auc")),
                     "ACC": _number(metrics.get("accuracy")), "loss": loss,
                     "time": duration if split == "train" else ""})
    return rows


def rebuild_results_csv(output_root="experiments"):
    """Rebuild the report from immediate run directories; repeated calls deduplicate."""
    root = project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not directory.is_dir() or not (directory / "summary.json").is_file():
            continue
        try:
            rows.extend(experiment_rows(directory))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            # An unfinished/invalid summary must not prevent reporting other runs.
            warnings.warn(f"Skipping {directory.name}/summary.json: {error}", stacklevel=2)
    destination = root / CSV_NAME
    temporary = None
    try:
        # A temporary file prevents readers from observing a half-written report.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=root, prefix=".results_", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments", help="Parent directory of completed runs.")
    args = parser.parse_args(argv)
    destination = rebuild_results_csv(args.root)
    print(f"Saved experiment results: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    main()
