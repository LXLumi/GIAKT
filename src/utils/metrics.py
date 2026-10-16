"""Binary metrics; undefined single-class AUC is reported as null."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def binary_metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not labels.size:
        raise ValueError("No valid next-answer targets were evaluated.")
    return {
        "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "bce": float(log_loss(labels, probabilities, labels=[0, 1])),
        "num_predictions": int(labels.size),
    }
