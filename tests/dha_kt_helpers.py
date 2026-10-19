"""Small provenanced student splits shared by DHA integration checks."""
import json
from pathlib import Path

import numpy as np
from src.data.interactions import InteractionDataset, collate_sequences
from src.utils.config import load_config
from scripts.dha_kt.build_reference import build_reference
from src.model import DHAKT


def make_data(root, bins=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    splits = dict(train=list(range(1, 7)), valid=[7, 8], test=[9, 10])
    (root/"metadata.json").write_text(json.dumps(dict(num_problems=8, num_skills=4, split_students=splits)), encoding="utf-8")
    for split, students in splits.items():
        rows = []
        for student in students:
            for start in [0, 3]:
                length = min(4, 6-start)
                problems, answers, sources = np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
                skills = np.zeros((4, 3), dtype=np.int64)
                for j in range(length):
                    position = start+j
                    problems[j], answers[j], sources[j] = position%3+1, (position+student)%2, student*100+position
                    skills[j] = [position%3+1, (position+1)%3+1, 0]
                rows.append((student, length, problems, answers, sources, skills))
        np.savez_compressed(root/f"{split}.npz", studentId=np.array([r[0] for r in rows]), lengths=np.array([r[1] for r in rows]),
            problemId=np.stack([r[2] for r in rows]), correct=np.stack([r[3] for r in rows]), sourceRow=np.stack([r[4] for r in rows]), skill=np.stack([r[5] for r in rows]))
    config = load_config(model_name="dha_kt")
    config["dataset"]["paths"]["processed"] = str(root)
    config["model"].update(embedding_dim=8, dropout=0., attention_embedding_dim=2, residual_embedding_dim=2,
        profile=dict(artifact_path=str(root/"reference"), bins=bins, fixed_scale=4., shrinkage_kappa=5., no_evidence_policy="raw_only", manifest_sha256=None),
        diffusion=dict(prediction_type="x0", steps=4, schedule="cosine", sampling_steps=2, sampler="ddim", ddim_eta=0.,
                       time_embedding_dim=8, hidden_dim=16, beta_start=1e-4, beta_end=.02, reverse_variance="posterior"),
        stages=dict(base_warmup_epochs=0, diffusion_pretrain_epochs=0))
    config["train"].update(device="cpu", batch_size=4, num_workers=0, num_threads=2, epochs=2, patience=5,
        preprocess_each_run=False, output_root=str(root/"experiments"))
    build_reference(config, root/"reference", bins=bins, draw_count=3)
    return config


def prepared_model(root, **overrides):
    config = make_data(root)
    kwargs = dict(num_problems=8, num_skills=4, embedding_dim=8, num_heads=2, dropout=0., attention_embedding_dim=2,
        residual_embedding_dim=2, profile=config["model"]["profile"], diffusion=config["model"]["diffusion"])
    kwargs.update(overrides)
    model = DHAKT(**kwargs)
    dataset = InteractionDataset(Path(root)/"train.npz")
    model.prepare_training(dataset)
    model.set_data_context("train")
    batch = collate_sequences([dataset[0], dataset[2]])
    return model, batch, config
