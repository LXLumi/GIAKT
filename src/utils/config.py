"""Resolve configuration paths relative to the project."""
from pathlib import Path
import copy
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIGS = {"diffuqkt": "configs/models/diffuqkt.yaml", "linearq": "configs/models/linearq.yaml",
                 "newq": "configs/models/newq.yaml", "newq_v2": "configs/models/newq_v2.yaml",
                 "dha_kt": "configs/models/dha_kt.yaml"}


def resolve_model_name(name="diffuqkt"):
    """Canonical names shared by configuration and the model registry."""
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    canonical = name.lower()
    if canonical == "diffqkt":
        canonical = "diffuqkt"
    if canonical not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_CONFIGS)}.")
    return canonical


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


def load_config(path="configs/assistment2017.yaml", model_name=None):
    base = yaml.safe_load((PROJECT_ROOT / "configs/base.yaml").read_text(encoding="utf-8")) or {}
    override = yaml.safe_load(project_path(path).read_text(encoding="utf-8")) or {}
    name = resolve_model_name(model_name if model_name is not None else
                              override.get("model_name", base.get("model_name", "diffuqkt")))
    model_defaults = yaml.safe_load(project_path(MODEL_CONFIGS[name]).read_text(encoding="utf-8")) or {}
    config = merge_dict(merge_dict(base, model_defaults), override)
    config["model_name"] = name
    return config
