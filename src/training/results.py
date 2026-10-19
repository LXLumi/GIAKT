"""Summarize completed experiments in a three-row-per-run CSV."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

from src.utils.config import project_path


CSV_NAME = "results.csv"
CSV_COLUMNS = ("experiment", "split", "AUC", "ACC", "loss", "备注")
CSV_DECIMAL_PLACES = 4


def format_duration(seconds):
    """Display elapsed time as total minutes and remaining seconds, e.g. 61m5s."""
    if seconds is None:
        return ""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("duration_seconds must be finite and non-negative.")
    minutes, remaining = divmod(int(seconds + 0.5), 60)
    return f"{minutes}m{remaining}s"


def _number(value, decimal_places):
    if value is None or value == "":
        return ""
    value = float(value)
    return f"{value:.{decimal_places}f}" if math.isfinite(value) else ""


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


def _finished_at(summary, summary_path):
    """Use the recorded local finish time, or the summary mtime for legacy runs."""
    value = summary.get("finished_at")
    finished = (datetime.fromisoformat(value) if value else
                datetime.fromtimestamp(summary_path.stat().st_mtime).astimezone())
    return finished.strftime("%Y-%m-%d %H:%M:%S")


def _read_existing_runs(path):
    """Read complete cached groups, including history whose run folder is absent."""
    runs, experiment = {}, None
    try:
        stream = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        return runs
    with stream:
        reader = csv.DictReader(stream)
        # Older CSVs have a time column and are regenerated from their summaries.
        if "备注" not in (reader.fieldnames or []):
            return runs
        for row in reader:
            if row.get("split") == "train":
                experiment = row.get("experiment")
                if experiment:
                    runs[experiment] = []
            if experiment and row.get("split") in ("train", "valid", "test"):
                runs[experiment].append({column: row.get(column) or "" for column in CSV_COLUMNS})
    return {name: rows for name, rows in runs.items()
            if [row["split"] for row in rows] == ["train", "valid", "test"]}


def experiment_rows(directory):
    """Read best-epoch metrics without reevaluating or editing the experiment."""
    directory = Path(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    # Runs created before display precision was recorded keep their eight decimals.
    decimal_places = summary.get("results_decimal_places", 8)
    record = _best_record(summary, directory)
    train = {**_split_metrics(record, "train"), **(summary.get("train") or {})}
    valid = {**_split_metrics(record, "valid"), **(summary.get("validation") or {})}
    test = summary["test"]
    if not isinstance(test, dict):
        raise ValueError("A completed summary must contain test metrics.")
    duration = format_duration(summary.get("duration_seconds"))
    labels = {"train": directory.name, "valid": _finished_at(summary, summary_path),
              "test": duration}
    # This tuple contains raw component losses, not their weighted sum.
    components = [_number(train.get(key), decimal_places) or "NA" for key in
                  ("kt_loss", "contrastive_loss", "diffusion_loss")]
    rows = []
    for split, metrics in (("train", train), ("valid", valid), ("test", test)):
        loss = (f"({', '.join(components)})" if split == "train"
                else _number(metrics.get("loss"), decimal_places))
        rows.append({"experiment": labels[split],
                     "split": split, "AUC": _number(metrics.get("auc"), decimal_places),
                     "ACC": _number(metrics.get("accuracy"), decimal_places), "loss": loss,
                     "备注": ""})
    return rows


def rebuild_results_csv(output_root="experiments"):
    """Scan grouped/legacy runs, preserving notes and cached history without folders."""
    root = project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / CSV_NAME
    cached = _read_existing_runs(destination)
    notes = {(name, row["split"]): row["备注"] for name, run_rows in cached.items() for row in run_rows}
    groups, found = [], set()
    directories = sorted((path.parent for path in root.rglob("summary.json")),
                         key=lambda entry: (entry.name, entry.as_posix()))
    for directory in directories:
        found.add(directory.name)
        try:
            run_rows = experiment_rows(directory)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            # An unfinished/invalid summary must not prevent reporting other runs.
            warnings.warn(f"Skipping {directory.relative_to(root)}/summary.json: {error}", stacklevel=2)
            continue
        for row in run_rows:
            row["备注"] = notes.get((directory.name, row["split"]), "")
        groups.append((directory.name, run_rows))
    # The CSV remains useful for comparison even after a run's artifacts are removed.
    groups.extend((name, run_rows) for name, run_rows in cached.items() if name not in found)
    rows = [row for _, run_rows in sorted(groups, key=lambda group: group[0]) for row in run_rows]
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
