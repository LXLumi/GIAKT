"""Canonical prepared interaction arrays; CPU loading and trainer-owned transfer."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, path, max_windows=None, max_seq_len=None):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}
        if self.arrays["skill"].ndim not in {2, 3}:
            raise ValueError("skill must have shape [windows, time] or [windows, time, skills].")
        if self.arrays["skill"].shape[:2] != self.arrays["problemId"].shape:
            raise ValueError("Skill and problem sequence dimensions differ.")
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows must be positive.")
            self.arrays = {key: value[:max_windows] for key, value in self.arrays.items()}
        if max_seq_len is not None:
            if max_seq_len < 3:
                raise ValueError("max_seq_len must be at least 3.")
            self.arrays = {key: value[:, :max_seq_len] if value.ndim >= 2 else value
                           for key, value in self.arrays.items()}
            self.arrays["lengths"] = np.minimum(self.arrays["lengths"], max_seq_len)

    def __len__(self):
        return len(self.arrays["lengths"])

    def unique_problem_ids(self):
        """Question vocabulary in valid positions of this runtime split/scope."""
        problems = self.arrays["problemId"]
        valid = np.arange(problems.shape[1])[None, :] < self.arrays["lengths"][:, None]
        return np.unique(problems[valid & (problems > 0)])

    def __getitem__(self, index):
        length = int(self.arrays["lengths"][index])
        mask = torch.arange(self.arrays["problemId"].shape[1]) < length
        prediction_mask = mask.clone()
        prediction_mask[0] = False
        return {
            "problem_ids": torch.from_numpy(self.arrays["problemId"][index]).long(),
            "skill_ids": torch.from_numpy(self.arrays["skill"][index]).long(),
            "answers": torch.from_numpy(self.arrays["correct"][index]).long(),
            "student_ids": torch.tensor(int(self.arrays["studentId"][index]), dtype=torch.long),
            "valid_mask": mask,
            "prediction_mask": prediction_mask,
        }


def collate_sequences(items):
    """Trim the time axis of both scalar and multi-skill sequences."""
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    length = int(batch["valid_mask"].sum(dim=1).max())
    return {key: value[:, :length] if value.ndim >= 2 else value for key, value in batch.items()}
