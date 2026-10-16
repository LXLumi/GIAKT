"""Index ASSIST17, recover event order, and split by student (source CSVs read only)."""
from __future__ import annotations
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import load_config, project_path
from scripts.preprocessing import FIELDS, build_index, split_students, pack_sequences
from src.utils.files import sha256
MISSING = {"", "na", "nan", "none", "null", "noskill"}


def read_clean(path):
    frame = pd.read_csv(path, dtype={k: str for k in FIELDS}, keep_default_na=False)
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["sourceRow"] = np.arange(len(frame), dtype=np.int64)
    for name in FIELDS:
        frame[name] = frame[name].str.strip()
    good = np.ones(len(frame), dtype=bool)
    for name in FIELDS:
        good &= ~frame[name].str.lower().isin(MISSING).to_numpy()
    times = pd.to_numeric(frame.timeTaken, errors="coerce")
    answers = pd.to_numeric(frame.correct, errors="coerce")
    good &= np.isfinite(times) & (times > 0) & answers.isin([0, 1])
    rejected = int((~good).sum())
    frame = frame.loc[good].copy()
    frame["timeTaken"] = times.loc[good].to_numpy()
    frame["correct"] = answers.loc[good].astype(np.int64).to_numpy()
    return frame, rejected


def recover_chronology(frame, raw_path):
    """Match five-field tuples by occurrence using the cleaner's Decimal rounding."""
    columns = FIELDS + ["startTime", "action_num"]
    raw = pd.read_csv(raw_path, usecols=columns, dtype={k: str for k in columns},
                      keep_default_na=False)
    for name in columns:
        raw[name] = raw[name].str.strip()
    good = np.ones(len(raw), dtype=bool)
    for name in FIELDS:
        good &= ~raw[name].str.lower().isin(MISSING).to_numpy()
    numeric_time = pd.to_numeric(raw.timeTaken, errors="coerce")
    answers = pd.to_numeric(raw.correct, errors="coerce")
    good &= np.isfinite(numeric_time) & (numeric_time > 0) & answers.isin([0, 1])
    raw = raw.loc[good].copy()
    raw["correct"] = answers.loc[good].astype(np.int64).to_numpy()
    raw["_timeKey"] = raw.timeTaken.map(
        lambda x: str(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)))
    clean = frame.copy()
    clean["_timeKey"] = clean.timeTaken.map(lambda x: f"{x:.2f}")
    keys = ["studentId", "problemId", "skill", "correct", "_timeKey"]
    raw_count = raw.groupby(keys, sort=False).size()
    clean_count = clean.groupby(keys, sort=False).size()
    available = raw_count.reindex(clean_count.index).fillna(0)
    # A deduplicated tuple may correspond to several different real attempts.
    ambiguous = ((clean_count > 1) & (available != clean_count)) | ((clean_count == 1) & (available > 1))
    if ambiguous.any():
        raise ValueError("Clean rows are ambiguous among repeated raw events. Preserve "
                         "startTime/action_num when cleaning, or use --no-recover-chronology "
                         "to explicitly keep clean-file order.")
    for table in (raw, clean):
        table["_occurrence"] = table.groupby(keys, sort=False).cumcount()
    matched = clean.merge(raw[keys + ["_occurrence", "startTime", "action_num"]],
                          on=keys + ["_occurrence"], how="left", validate="one_to_one")
    for name in ["startTime", "action_num"]:
        matched[name] = pd.to_numeric(matched[name], errors="coerce")
        if not np.isfinite(matched[name]).all():
            raise ValueError(f"Cannot recover {name} for every clean row.")
    return matched.drop(columns=["_timeKey", "_occurrence"])


def preprocess(config):
    settings, paths = config["preprocess"], config["dataset"]["paths"]
    clean_path, out = project_path(paths["clean"]), project_path(paths["processed"])
    minimum, maximum = settings["min_time_seconds"], settings["max_time_seconds"]
    if not 0 < minimum <= maximum or settings["short_time_policy"] not in {"drop", "clip"}:
        raise ValueError("Invalid time bounds or short_time_policy.")
    if settings["min_student_interactions"] < 3:
        raise ValueError("min_student_interactions must be at least 3.")
    frame, invalid = read_clean(clean_path)
    report = {"input_rows": len(frame) + invalid, "invalid_rows_removed": invalid}
    if settings["recover_chronology"]:
        if {"startTime", "action_num"}.issubset(frame.columns):
            for name in ["startTime", "action_num"]:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
                if not np.isfinite(frame[name]).all():
                    raise ValueError(f"Invalid {name} values in cleaned CSV.")
            report["chronology"] = "cleaned CSV startTime, action_num, sourceRow"
        else:
            raw_path = project_path(paths["raw"])
            print("Recovering timestamps and action IDs from raw CSV...", flush=True)
            frame = recover_chronology(frame, raw_path)
            report["chronology"] = "startTime, action_num, sourceRow"
            report["raw_sha256"] = sha256(raw_path)
        duplicates = frame.duplicated(["studentId", "action_num", "startTime"])
        report["duplicate_events_removed"] = int(duplicates.sum())
        frame = frame.loc[~duplicates].copy()
        sort_keys = ["studentId", "startTime", "action_num", "sourceRow"]
    else:
        report["chronology"] = "clean file order (explicit fallback; timestamps unavailable)"
        report["duplicate_events_removed"] = 0
        sort_keys = ["studentId", "sourceRow"]
    report["below_min_time"] = int((frame.timeTaken < minimum).sum())
    report["above_max_time"] = int((frame.timeTaken > maximum).sum())
    if settings["short_time_policy"] == "drop":
        frame = frame.loc[frame.timeTaken >= minimum].copy()
    frame["timeTaken"] = frame.timeTaken.clip(lower=minimum, upper=maximum)
    frame = frame.sort_values(sort_keys, kind="stable")
    # ':' and '-' in this source are parts of a concept name, not delimiters.
    mappings = {name: build_index(frame[name]) for name in ["studentId", "problemId", "skill"]}
    for name, mapping in mappings.items():
        frame[name] = frame[name].map(mapping).astype(np.int64)
    lengths = frame.groupby("studentId").size()
    eligible = lengths[lengths >= settings["min_student_interactions"]].index
    usable = frame.loc[frame.studentId.isin(eligible)]
    report["short_students_removed"] = int(len(lengths) - len(eligible))
    report["short_student_rows_removed"] = int(len(frame) - len(usable))
    splits = split_students(eligible, settings["split_ratios"], settings["seed"])
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "indexed.csv", index=False, float_format="%.2f")
    for name, mapping in mappings.items():
        pd.DataFrame({"original": list(mapping), "index": list(mapping.values())}).to_csv(
            out / f"{name}_index.csv", index=False)
    (out / "mappings.json").write_text(json.dumps({name: {
        "raw_to_index": mapping, "index_to_raw": {str(i): raw for raw, i in mapping.items()}}
        for name, mapping in mappings.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    frame[["problemId", "skill"]].drop_duplicates().sort_values(["problemId", "skill"]).to_csv(
        out / "question_concepts.csv", index=False)
    report["splits"] = {}
    for name, students in splits.items():
        subset = usable.loc[usable.studentId.isin(students)]
        arrays = pack_sequences(subset, settings["max_seq_len"], maximum)
        np.savez_compressed(out / f"{name}.npz", **arrays)
        targets = int((arrays["lengths"] - 1).sum())
        if targets != len(subset) - len(students):
            raise AssertionError("Windowing lost or duplicated prediction targets.")
        report["splits"][name] = {"students": len(students), "interactions": len(subset),
                                  "windows": len(arrays["lengths"]), "targets": targets}
    report.update({"schema_version": 1, "padding_id": 0,
                   "num_students": len(mappings["studentId"]),
                   "num_problems": len(mappings["problemId"]),
                   "num_skills": len(mappings["skill"]),
                   "indexed_rows": len(frame), "usable_rows": len(usable),
                   "max_seq_len": settings["max_seq_len"], "preprocess": settings,
                   "input_path": str(clean_path), "input_sha256": sha256(clean_path),
                   "split_students": {name: students.tolist() for name, students in splits.items()}})
    (out / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "split_students"}, indent=2), flush=True)
    print(f"Prepared data: {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/assistment2017.yaml")
    parser.add_argument("--input", help="Override the cleaned CSV path")
    parser.add_argument("--output", help="Override the processed directory")
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--short-time-policy", choices=["drop", "clip"])
    parser.add_argument("--no-recover-chronology", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for argument, key in [(args.input, "clean"), (args.output, "processed")]:
        if argument:
            config["dataset"]["paths"][key] = argument
    if args.max_time is not None:
        config["preprocess"]["max_time_seconds"] = args.max_time
    if args.short_time_policy:
        config["preprocess"]["short_time_policy"] = args.short_time_policy
    if args.no_recover_chronology:
        config["preprocess"]["recover_chronology"] = False
    preprocess(config)


if __name__ == "__main__":
    main()
