"""Resolve configuration paths relative to the project."""
from pathlib import Path
import copy
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path="configs/assistment2017.yaml"):
    base = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text(encoding="utf-8")) or {}
    override = yaml.safe_load(project_path(path).read_text(encoding="utf-8")) or {}
    return merge_dict(base, override)
