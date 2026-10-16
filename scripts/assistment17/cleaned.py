"""Clean the ASSISTments 2017 dataset."""

from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "assistment17" / "assistment2017.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assistment17" / "assistment2017_cleaned.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "assistment2017.yaml"

SOURCE_FIELDS = (
    "studentId",
    "problemId",
    "skill",
    "timeTaken",
    "correct",
)
OUTPUT_FIELDS = (
    "studentId",
    "problemId",
    "skill",
    "timeTaken",
    "correct",
)
MISSING_VALUES = {"", "na", "nan", "null", "none"}
MISSING_SKILL_VALUES = MISSING_VALUES | {"noskill"}
TWO_DECIMAL_PLACES = Decimal("0.01")


def is_missing(value: str) -> bool:
    """Return whether a source value represents missing data."""
    return value.strip().lower() in MISSING_VALUES


def format_time_taken(value: str) -> Optional[str]:
    """Validate seconds and format them with two decimal places."""
    try:
        time_taken = Decimal(value.strip())
    except InvalidOperation:
        return None

    if not time_taken.is_finite() or time_taken <= 0:
        return None

    time_taken = time_taken.quantize(
        TWO_DECIMAL_PLACES,
        rounding=ROUND_HALF_UP,
    )
    if time_taken <= 0:
        return None

    return f"{time_taken:.2f}"


def clean_row(row: dict[str, str]) -> Optional[dict[str, str]]:
    """Validate and transform one source row, omitting attemptCount."""
    selected = {field: (row.get(field) or "").strip() for field in SOURCE_FIELDS}

    if any(is_missing(value) for value in selected.values()):
        return None
    if selected["skill"].lower() in MISSING_SKILL_VALUES:
        return None

    time_taken = format_time_taken(selected["timeTaken"])
    if time_taken is None:
        return None

    return {
        "studentId": selected["studentId"],
        "problemId": selected["problemId"],
        "skill": selected["skill"],
        "timeTaken": time_taken,
        "correct": selected["correct"],
    }


def validate_columns(reader: csv.DictReader) -> None:
    """Ensure that the source CSV contains every required field."""
    missing_columns = [field for field in SOURCE_FIELDS if field not in (reader.fieldnames or [])]
    if missing_columns:
        raise ValueError(f"Missing required source fields: {', '.join(missing_columns)}")


def update_dataset_config(
    config_path: Path,
    *,
    num_skills: int,
    num_students: int,
    num_problems: int,
    num_multi_skill_records: int,
) -> None:
    """Update final dataset statistics while preserving the YAML structure."""
    config_text = config_path.read_text(encoding="utf-8")
    values = {
        "num_skills": num_skills,
        "num_students": num_students,
        "num_problems": num_problems,
        "num_multi_skill_records": num_multi_skill_records,
    }

    for key, value in values.items():
        pattern = re.compile(
            rf"^([ \t]*{re.escape(key)}[ \t]*:)[^#\r\n]*(#.*)?$",
            re.MULTILINE,
        )

        def replace_value(match: re.Match[str]) -> str:
            comment = f"  {match.group(2)}" if match.group(2) else ""
            return f"{match.group(1)} {value}{comment}"

        config_text, replacements = pattern.subn(replace_value, config_text)
        if replacements != 1:
            raise ValueError(
                f"Expected exactly one '{key}' entry in {config_path}, found {replacements}"
            )

    config_path.write_text(config_text, encoding="utf-8")


def clean_dataset(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    config_path: Path = CONFIG_PATH,
) -> None:
    """Clean the dataset, save five fields, print counts, and update YAML."""
    total_records = 0
    cleaned_records = 0
    student_ids: set[str] = set()
    problem_ids: set[str] = set()
    skill_ids: set[str] = set()
    multi_skill_records = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        validate_columns(reader)
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for row in reader:
            total_records += 1
            cleaned = clean_row(row)
            if cleaned is None:
                continue

            writer.writerow(cleaned)
            cleaned_records += 1
            student_ids.add(cleaned["studentId"])
            problem_ids.add(cleaned["problemId"])
            skill_ids.add(cleaned["skill"])

            # This source stores one complete skill name per row. Hyphens,
            # spaces, and colons are parts of the name, not skill delimiters.
            # Therefore no row in the inspected dataset is a multi-skill row.

    update_dataset_config(
        config_path,
        num_skills=len(skill_ids),
        num_students=len(student_ids),
        num_problems=len(problem_ids),
        num_multi_skill_records=multi_skill_records,
    )

    print(f"Source records: {total_records}")
    print(f"Cleaned records: {cleaned_records}")
    print(f"Removed records: {total_records - cleaned_records}")
    print(f"Students: {len(student_ids)}")
    print(f"Problems: {len(problem_ids)}")
    print(f"Skills: {len(skill_ids)}")
    print(f"Multi-skill records: {multi_skill_records}")
    print(f"Output: {output_path}")
    print(f"Config updated: {config_path}")


if __name__ == "__main__":
    clean_dataset()
