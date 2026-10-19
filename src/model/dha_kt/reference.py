"""Verified, checkpoint-contained reference buffers; split context is independent of eval mode."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.utils.files import sha256


FLOAT_FIELDS = ["partial_profile", "partial_counts", "dense_profile", "dense_counts"]
BOOL_FIELDS = ["partial_observed", "dense_observed"]
INT_FIELDS = ["student_fold_ids", "student_split_ids", "supervision_views", "supervision_problems", "supervision_skills"]


class ReferenceBank(nn.Module):
    def __init__(self, schema=None):
        super().__init__()
        schema = schema or {}
        for names, dtype in [(FLOAT_FIELDS, torch.float32), (BOOL_FIELDS, torch.bool), (INT_FIELDS, torch.long)]:
            for name in names:
                self.register_buffer(name, torch.zeros(schema.get(name, [0]), dtype=dtype))
        self.register_buffer("ready", torch.tensor(False))
        self.manifest_text = ""
        self._manifest_cache, self._manifest_hash = None, None

    @property
    def manifest(self):
        if self._manifest_cache is None:
            self._manifest_cache = json.loads(self.manifest_text) if self.manifest_text else {}
        return self._manifest_cache

    @property
    def manifest_sha256(self):
        if self._manifest_hash is None and self.manifest_text:
            self._manifest_hash = hashlib.sha256(self.manifest_text.encode()).hexdigest()
        return self._manifest_hash

    def schema(self):
        return {name: list(getattr(self, name).shape) for name in FLOAT_FIELDS+BOOL_FIELDS+INT_FIELDS}

    def get_extra_state(self):
        return {"manifest_text": self.manifest_text}

    def set_extra_state(self, state):
        self.manifest_text = state["manifest_text"]
        self._manifest_cache, self._manifest_hash = None, None

    def load_artifact(self, path, expected_sha256=None):
        path = Path(path)
        path = path/"manifest.json" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(f"DHA reference manifest missing: {path}; run scripts.dha_kt.build_reference first.")
        # Preserve CRLF bytes on Windows: CLI/file SHA and checkpoint SHA must agree.
        text = path.read_bytes().decode("utf-8")
        actual = hashlib.sha256(text.encode()).hexdigest()
        if expected_sha256 is not None and expected_sha256 != actual:
            raise ValueError("DHA reference manifest SHA256 mismatch.")
        manifest = json.loads(text)
        if manifest.get("schema_version") != 1 or manifest.get("predictor_type") != "prefix_beta_skill":
            raise ValueError("Unsupported reference schema or reference predictor.")
        tensor_name = manifest["tensor_file"]
        if Path(tensor_name).name != tensor_name:
            raise ValueError("Reference tensor file must be local to the artifact directory.")
        if sha256(path.parent/tensor_name) != manifest["tensor_file_sha256"]:
            raise ValueError("DHA reference tensor SHA256 mismatch.")
        if sha256(path.parent/"prefix_predictions.npz") != manifest["prefix_predictions_sha256"]:
            raise ValueError("DHA pre-response prediction SHA256 mismatch.")
        with np.load(path.parent/tensor_name, allow_pickle=False) as arrays:
            for names, dtype in [(FLOAT_FIELDS, torch.float32), (BOOL_FIELDS, torch.bool), (INT_FIELDS, torch.long)]:
                for name in names:
                    if list(arrays[name].shape) != manifest["profile_table_shapes"][name]:
                        raise ValueError("Reference tensor shape differs from manifest.")
                    setattr(self, name, torch.as_tensor(arrays[name].copy(), dtype=dtype, device=self.ready.device))
        self.manifest_text = text
        self._manifest_cache, self._manifest_hash = manifest, actual
        self.validate()
        self.ready.fill_(True)

    def validate(self):
        manifest = self.manifest
        if not manifest:
            raise ValueError("Reference bank has no manifest.")
        shape = self.partial_profile.shape
        if len(shape) != 4 or shape[:2] != (manifest["fold_count"]+1, manifest["draw_count"]):
            raise ValueError("Invalid reference view/draw dimensions.")
        for name in FLOAT_FIELDS+BOOL_FIELDS:
            if getattr(self, name).shape != shape:
                raise ValueError("Reference profiles, counts and masks must have matching shapes.")
        for prefix in ["partial", "dense"]:
            profile, counts, observed = [getattr(self, prefix+suffix) for suffix in ["_profile", "_counts", "_observed"]]
            if not torch.isfinite(profile).all() or not torch.isfinite(counts).all() or (counts < 0).any():
                raise ValueError("Reference profiles/counts must be finite with nonnegative counts.")
            if not torch.equal(observed, counts > 0) or (profile[~observed] != 0).any():
                raise ValueError("Unobserved reference dimensions must be explicitly masked and zero-filled.")
        folds = self.student_fold_ids.cpu().numpy()
        if hashlib.sha256(folds.tobytes()).hexdigest() != manifest["student_fold_assignment_sha256"]:
            raise ValueError("Reference student-fold assignment SHA256 mismatch.")
        if (self.partial_counts > self.dense_counts).any():
            raise ValueError("Partial evidence counts exceed the nested dense evidence.")
        training_students = set(np.where(folds >= 0)[0].tolist())
        for view in manifest["views"]:
            pool = set(view["reference_students"])
            expected = training_students if view["excluded_fold"] is None else {s for s in training_students if folds[s] != view["excluded_fold"]}
            if pool != expected:
                raise ValueError("Reference student pool differs from the recorded fold exclusions.")
            if view["excluded_fold"] is not None and any(folds[s] == view["excluded_fold"] for s in pool):
                raise ValueError("Reference view contains an excluded student fold.")
            for draw in view["draws"]:
                if not set(draw["partial_students"]) <= set(draw["dense_students"]) <= pool:
                    raise ValueError("Partial/dense subsets violate the reference student scope.")

    def views_for_students(self, students, data_split):
        if not bool(self.ready):
            raise ValueError("Reference bank has not been prepared or restored.")
        if data_split not in {"train", "valid", "test"}:
            raise ValueError("Set an explicit train/valid/test data context before using reference evidence.")
        if (students < 1).any() or (students >= self.student_fold_ids.numel()).any():
            raise ValueError("Unknown student in DHA reference scope.")
        expected = {"train": 1, "valid": 2, "test": 3}[data_split]
        if (self.student_split_ids[students] != expected).any():
            raise ValueError("Student membership disagrees with the explicit data split.")
        folds = self.student_fold_ids[students]
        if data_split == "train":
            if (folds < 0).any():
                raise ValueError("Training student was absent from the artifact runtime scope.")
            return folds
        if (folds >= 0).any():
            raise ValueError("Validation/test student belongs to a training fold.")
        return torch.full_like(students, self.manifest["full_train_view"])

    def partial(self, views, draw, problems):
        known = (problems > 0) & (problems < self.partial_profile.shape[2])
        safe = torch.where(known, problems, torch.zeros_like(problems))
        mask = known.unsqueeze(-1)
        return (self.partial_profile[views, draw, safe]*mask,
                self.partial_counts[views, draw, safe]*mask,
                self.partial_observed[views, draw, safe] & mask)
