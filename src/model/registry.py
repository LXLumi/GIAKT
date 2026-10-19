"""Construct the selected model and recover model identity from checkpoints."""
from .diffuqkt import DiffuQKT
from .linearq import LinearQ
from .newq import NewQ
from .newq_v2 import NewQV2
from .dha_kt import DHAKT
from src.utils.config import resolve_model_name


MODEL_REGISTRY = {"diffuqkt": DiffuQKT, "linearq": LinearQ, "newq": NewQ, "newq_v2": NewQV2, "dha_kt": DHAKT}


def build_model(model_name, kwargs):
    return MODEL_REGISTRY[resolve_model_name(model_name)](**kwargs)


def checkpoint_model_name(checkpoint):
    """Historical checkpoints without a model tag belong to the original DiffuQKT."""
    return resolve_model_name(checkpoint.get("model_name")
                              or checkpoint.get("config", {}).get("model_name", "diffuqkt"))
