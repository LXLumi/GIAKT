"""Convert ASSIST09 raw events to ASSIST17 field names, units and NPZ sequences."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.preprocessing import FIELDS, build_index, pack_sequences, split_students
from src.utils.files import sha256
from src.utils.config import load_config, project_path

RAW_TO_CANONICAL = {
    "user_id": "studentId", "problem_id": "problemId", "skill_id": "skill",
    "ms_first_response": "timeTaken", "correct": "correct", "order_id": "orderId",
}
SOURCE_COLUMNS = list(RAW_TO_CANONICAL)


def inspect_raw(config, rows=10000):
    """Read a bounded sample, returning only the required source columns."""
    if rows < 1:
        raise ValueError("Inspection rows must be positive.")
    path = project_path(config["dataset"]["paths"]["raw"])
    encoding = config["preprocess"]["source_encoding"]
    columns = pd.read_csv(path, encoding=encoding, nrows=0).columns.tolist()
    sample = pd.read_csv(path, encoding=encoding, usecols=SOURCE_COLUMNS, nrows=rows,
                         dtype=str, keep_default_na=False)
    times = pd.to_numeric(sample.ms_first_response, errors="coerce")
    skills = sample.skill_id.str.strip()
    report = {
        "source_path": str(path), "encoding": encoding, "source_columns": len(columns),
        "sample_rows": len(sample), "field_mapping": RAW_TO_CANONICAL,
        "unused_columns": [name for name in columns if name not in SOURCE_COLUMNS],
        "samples": sample.head(5).to_dict(orient="records"),
        "sample_missing_skill_rows": int(skills.str.lower().isin(["", "nan", "none", "null"]).sum()),
        "sample_multi_skill_rows": int(skills.str.contains("_", regex=False).sum()),
        "sample_response_ms_quantiles": {str(q): float(v) for q, v in
                                         times.quantile([0, .5, .99, 1]).items()},
        "time_conversion": "ms_first_response / 1000 -> timeTaken (seconds)",
        "chronology": "numeric order_id within each student",
        "skill_policy": "underscore separates atomic skill IDs; retain one event per answer",
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def read_raw(path, encoding="latin-1"):
    """Validate six necessary source columns and retain row IDs for event audits."""
    frame = pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype=str, encoding=encoding,
                        keep_default_na=False).rename(columns=RAW_TO_CANONICAL)
    frame["sourceRow"] = np.arange(len(frame), dtype=np.int64)
    for name in FIELDS + ["orderId"]:
        frame[name] = frame[name].str.strip()
    good = pd.Series(True, index=frame.index)
    invalid_ids = pd.Series(False, index=frame.index)
    for name in ["studentId", "problemId", "orderId"]:
        valid = frame[name].str.fullmatch(r"[0-9]+(?:\.0+)?")
        invalid_ids |= ~valid
        frame[name] = frame[name].str.replace(r"\.0+$", "", regex=True).str.lstrip("0").replace("", "0")
    skill_valid = frame.skill.str.fullmatch(r"[0-9]+(?:_[0-9]+)*")
    times_ms = pd.to_numeric(frame.timeTaken, errors="coerce")
    answers = pd.to_numeric(frame.correct, errors="coerce")
    invalid_time = ~np.isfinite(times_ms) | (times_ms <= 0)
    invalid_correct = ~answers.isin([0, 1])
    good &= ~invalid_ids & skill_valid & ~invalid_time & ~invalid_correct
    report = {"input_rows": len(frame), "invalid_rows_removed": int((~good).sum()),
              "invalid_id_rows": int(invalid_ids.sum()),
              "missing_or_invalid_skill_rows": int((~skill_valid).sum()),
              "nonpositive_or_invalid_time_rows": int(invalid_time.sum()),
              "invalid_correct_rows": int(invalid_correct.sum())}
    frame = frame.loc[good].copy()
    # Canonicalize each complete set once; do not split one answer into several events.
    skill_sets = {value: tuple(str(i) for i in sorted({int(x) for x in value.split("_")}))
                  for value in frame.skill.unique()}
    frame["rawSkills"] = frame.skill.map(skill_sets)
    frame["skill"] = frame.rawSkills.map("_".join)
    frame["orderId"] = frame.orderId.astype(np.int64)
    frame["correct"] = answers.loc[good].astype(np.int64)
    frame["timeTaken"] = times_ms.loc[good].astype(float) / 1000.0
    return frame, report


def clean_events(frame, settings, report):
    minimum, maximum = settings["min_time_seconds"], settings["max_time_seconds"]
    if not 0 < minimum <= maximum or settings["short_time_policy"] not in {"drop", "clip"}:
        raise ValueError("Invalid time bounds or short_time_policy.")
    if settings["min_student_interactions"] < 3:
        raise ValueError("min_student_interactions must be at least 3.")
    if not settings["recover_chronology"]:
        raise ValueError("ASSIST09 requires order_id chronology; disablement is unsupported.")
    # Duplicate logs are safe to drop only when all five event values agree.
    event_keys = ["studentId", "orderId"]
    unique_values = frame.drop_duplicates(FIELDS + ["orderId"])
    if unique_values.duplicated(event_keys).any():
        raise ValueError("Conflicting five-tuples for the same student/order_id event.")
    duplicates = frame.duplicated(event_keys)
    report["duplicate_events_removed"] = int(duplicates.sum())
    frame = frame.loc[~duplicates].copy()
    report["below_min_time"] = int((frame.timeTaken < minimum).sum())
    report["above_max_time"] = int((frame.timeTaken > maximum).sum())
    if settings["short_time_policy"] == "drop":
        frame = frame.loc[frame.timeTaken >= minimum].copy()
    frame["timeTaken"] = frame.timeTaken.clip(lower=minimum, upper=maximum)
    # ASSIST17 cleaned CSV stores centiseconds with ROUND_HALF_UP.
    frame["timeTaken"] = np.floor(frame.timeTaken * 100 + 0.5 + 1e-9) / 100
    frame["timeTaken"] = frame.timeTaken.clip(lower=minimum, upper=maximum)
    frame = frame.sort_values(["studentId", "orderId", "sourceRow"], kind="stable")
    report["chronology"] = "numeric order_id, sourceRow within each student"
    report["multi_skill_rows"] = int((frame.rawSkills.map(len) > 1).sum())
    report["num_skill_sets"] = int(frame.skill.nunique())
    report["max_skills_per_event"] = int(frame.rawSkills.map(len).max()) if len(frame) else 0
    return frame


def preprocess(config):
    settings, paths = config["preprocess"], config["dataset"]["paths"]
    raw_path, clean_path, out = [project_path(paths[key]) for key in ["raw", "clean", "processed"]]
    resolved = [path.resolve() for path in [raw_path, clean_path, out]]
    if resolved[0] == resolved[1] or resolved[0].is_relative_to(resolved[2]) or resolved[1].is_relative_to(resolved[2]):
        raise ValueError("Raw source, clean CSV and processed directory must have separate paths.")
    fingerprint = sha256(raw_path)
    frame, report = read_raw(raw_path, settings["source_encoding"])
    frame = clean_events(frame, settings, report)
    if frame.empty:
        raise ValueError("No valid events after cleaning.")
    # Build atomic vocabulary, sharing embeddings across every combination containing a skill.
    mappings = {name: build_index(frame[name]) for name in ["studentId", "problemId"]}
    mappings["skill"] = build_index(skill for skills in frame.rawSkills for skill in skills)
    for name in ["studentId", "problemId"]:
        frame[name] = frame[name].map(mappings[name]).astype(np.int64)
    frame = frame.sort_values(["studentId", "orderId", "sourceRow"], kind="stable")
    frame["skillIds"] = frame.rawSkills.map(lambda skills: tuple(mappings["skill"][s] for s in skills))
    # Deindex only the two scalar columns for the five-field, unindexed cleaned CSV.
    clean = frame[FIELDS].copy()
    for name in ["studentId", "problemId"]:
        inverse = {index: raw for raw, index in mappings[name].items()}
        clean[name] = clean[name].map(inverse)
    frame["skill"] = frame.skillIds.map(lambda skills: "_".join(map(str, skills)))
    lengths = frame.groupby("studentId").size()
    eligible = lengths[lengths >= settings["min_student_interactions"]].index
    usable = frame.loc[frame.studentId.isin(eligible)]
    splits = split_students(eligible, settings["split_ratios"], settings["seed"])
    report["short_students_removed"] = int(len(lengths) - len(eligible))
    report["short_student_rows_removed"] = int(len(frame) - len(usable))
    out.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, columns=FIELDS, index=False, float_format="%.2f")
    frame.to_csv(out / "indexed.csv", columns=FIELDS, index=False, float_format="%.2f")
    # Chronology and provenance stay outside the five-field interaction table.
    frame[["studentId", "orderId", "sourceRow"]].to_csv(out / "event_metadata.csv", index=False)
    for name, mapping in mappings.items():
        pd.DataFrame({"original": list(mapping), "index": list(mapping.values())}).to_csv(
            out / f"{name}_index.csv", index=False)
    (out / "mappings.json").write_text(json.dumps({name: {
        "raw_to_index": mapping, "index_to_raw": {str(i): raw for raw, i in mapping.items()}}
        for name, mapping in mappings.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    question_skills = frame[["problemId", "skillIds"]].explode("skillIds").rename(columns={"skillIds": "skill"})
    question_skills.drop_duplicates().sort_values(["problemId", "skill"]).to_csv(
        out / "question_concepts.csv", index=False)
    report["splits"] = {}
    for name, students in splits.items():
        subset = usable.loc[usable.studentId.isin(students)]
        arrays = pack_sequences(subset, settings["max_seq_len"], settings["max_time_seconds"],
                                max_skills=report["max_skills_per_event"])
        targets = int((arrays["lengths"] - 1).sum())
        if targets != len(subset) - len(students):
            raise AssertionError("Windowing lost or duplicated prediction targets.")
        np.savez_compressed(out / f"{name}.npz", **arrays)
        report["splits"][name] = {"students": len(students), "interactions": len(subset),
                                  "windows": len(arrays["lengths"]), "targets": targets}
    if sha256(raw_path) != fingerprint:
        raise RuntimeError("Raw source changed during preprocessing.")
    report.update({"schema_version": 2, "padding_id": 0, "fields": FIELDS,
                   "num_students": len(mappings["studentId"]),
                   "num_problems": len(mappings["problemId"]), "num_skills": len(mappings["skill"]),
                   "indexed_rows": len(frame), "usable_rows": len(usable),
                   "max_seq_len": settings["max_seq_len"], "preprocess": settings,
                   "input_path": str(raw_path), "input_sha256": fingerprint,
                   "clean_path": str(clean_path), "clean_sha256": sha256(clean_path),
                   "field_mapping": RAW_TO_CANONICAL, "time_unit": "seconds",
                   "skill_storage": "CSV: underscore-joined atomic IDs; NPZ: [windows, time, skills]",
                   "skill_aggregation": "mean of non-padding atomic concept/variation embeddings",
                   "split_students": {name: students.tolist() for name, students in splits.items()}})
    (out / "metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "split_students"}, indent=2), flush=True)
    print(f"Clean five-tuples: {clean_path}\nPrepared data: {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/assistment2009.yaml")
    parser.add_argument("--input", help="Override the raw CSV path")
    parser.add_argument("--output", help="Override the processed directory")
    parser.add_argument("--clean-output", help="Override the five-field cleaned CSV path")
    parser.add_argument("--inspect", action="store_true", help="Inspect a bounded sample, without writing data")
    parser.add_argument("--rows", type=int, default=10000, help="Maximum rows for --inspect")
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--short-time-policy", choices=["drop", "clip"])
    args = parser.parse_args()
    config = load_config(args.config)
    for value, key in [(args.input, "raw"), (args.output, "processed"), (args.clean_output, "clean")]:
        if value:
            config["dataset"]["paths"][key] = value
    if args.max_time is not None:
        config["preprocess"]["max_time_seconds"] = args.max_time
    if args.short_time_policy:
        config["preprocess"]["short_time_policy"] = args.short_time_policy
    if args.inspect:
        inspect_raw(config, args.rows)
    else:
        preprocess(config)


if __name__ == "__main__":
    main()
