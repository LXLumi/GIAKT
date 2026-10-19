"""Runtime training-only item collection and checkpoint recovery for LinearQ."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.data.interactions import InteractionDataset
from src.model.registry import build_model
from src.training.trainer import evaluate_checkpoint, load_checkpoint, train
from src.utils.config import load_config


class TrainingItemTests(unittest.TestCase):
    def test_runtime_item_collection_excludes_padding_invalid_positions_and_truncated_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.npz"
            np.savez_compressed(path,
                problemId=np.array([[1, 1, 2, 8, 7], [3, 4, 0, 8, 8]]),
                skill=np.ones((2, 5), dtype=np.int64), correct=np.zeros((2, 5), dtype=np.int64),
                studentId=np.array([1, 2]), lengths=np.array([3, 2]))
            self.assertEqual(InteractionDataset(path).unique_problem_ids().tolist(), [1, 2, 3, 4])
            limited = InteractionDataset(path, max_windows=1, max_seq_len=3)
            self.assertEqual(limited.unique_problem_ids().tolist(), [1, 2])

    def test_real_training_saves_only_train_items_and_recovers_them_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "metadata.json").write_text(json.dumps({"num_problems": 8, "num_skills": 4}), encoding="utf-8")
            split_questions = {"train": [[1, 1, 2], [2, 1, 1]],
                               "valid": [[3, 3, 3], [3, 3, 3]], "test": [[4, 4, 4], [4, 4, 4]]}
            for index, (split, questions) in enumerate(split_questions.items()):
                np.savez_compressed(data / f"{split}.npz", problemId=np.array(questions),
                    skill=np.ones((2, 3), dtype=np.int64), correct=np.array([[0, 1, 0], [1, 0, 1]]),
                    studentId=np.array([2*index+1, 2*index+2]), lengths=np.array([3, 3]))
            config = load_config(model_name="linearq")
            config["dataset"]["paths"]["processed"] = str(data)
            config["model"].update(embedding_dim=8, dropout=0)
            config["train"].update(epochs=2, patience=2, batch_size=2, device="cpu", num_threads=2,
                                    preprocess_each_run=False, output_root=str(root / "experiments"))
            with redirect_stdout(io.StringIO()):
                summary = train(config)
            self.assertEqual(summary["model_kwargs"]["train_item_ids"], [1, 2])
            self.assertGreater(summary["train"]["diffusion_loss"], 0)
            output = Path(summary["output_dir"])
            self.assertEqual(output.parent, root / "experiments" / "linearq" / "assistment2017")
            report = root / "experiments" / "results.csv"
            self.assertEqual(list((root / "experiments").rglob("results.csv")), [report])
            with report.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["experiment"], output.name)
            checkpoint = load_checkpoint(output / "best.pt", torch.device("cpu"))
            restored = build_model("linearq", checkpoint["model_kwargs"])
            restored.load_state_dict(checkpoint["model"], strict=True)
            self.assertEqual(restored.train_item_ids.tolist(), [1, 2])
            restored.item_regularization_loss().backward()
            grads = restored.question_embedding.weight.grad
            self.assertGreater(float(grads[1:3].abs().sum()), 0)
            self.assertTrue((grads[3:] == 0).all())
            self.assertTrue((grads[0] == 0).all())
            # Evaluation uses the checkpoint's actual path after moving its whole run.
            relocated = root / "relocated" / "linearq" / "assistment2017" / output.name
            relocated.parent.mkdir(parents=True)
            output.rename(relocated)
            evaluated = evaluate_checkpoint(config, relocated / "best.pt", verbose=False)
            for key in ("auc", "accuracy", "bce", "loss"):
                self.assertEqual(evaluated[key], summary["test"][key])


if __name__ == "__main__":
    unittest.main()
