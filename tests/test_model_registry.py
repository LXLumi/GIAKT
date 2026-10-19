"""Model-specific configuration and automatic checkpoint model recovery."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.model import DiffuQKT, LinearQ, NewQ, NewQV2
from src.model.registry import build_model, checkpoint_model_name
from src.training.trainer import evaluate_checkpoint
from src.utils.config import load_config
from src.utils.files import sha256
from test_diffuqkt import example_batch


class ModelRegistryTests(unittest.TestCase):
    def test_model_defaults_and_factories_do_not_mix_diffusion_with_linearq(self):
        diffusion = load_config()
        linear = load_config(model_name="LinearQ")
        self.assertEqual(diffusion["model_name"], "diffuqkt")
        self.assertEqual(diffusion["model"]["diffusion_steps"], 100)
        self.assertEqual(diffusion["loss"]["diffusion_weight"], .01)
        self.assertEqual(linear["model_name"], "linearq")
        self.assertNotIn("diffusion_steps", linear["model"])
        self.assertGreaterEqual(linear["loss"]["diffusion_weight"], 0)
        kwargs = dict(num_problems=8, num_skills=4, embedding_dim=8, dropout=0)
        self.assertIsInstance(build_model("LinearQ", kwargs), LinearQ)
        self.assertIsInstance(build_model("DiffQKT", {**kwargs, "diffusion_steps": 4}), DiffuQKT)
        self.assertIsInstance(build_model("newq", {**kwargs, "diffusion_steps": 4}), NewQ)
        newq = load_config(model_name="NewQ")
        self.assertEqual(newq["model"]["question_embedding_dim"], 16)
        self.assertEqual(newq["model"]["diffusion_steps"], 100)
        self.assertEqual(newq["loss"], diffusion["loss"])
        v2 = load_config(model_name="newq_v2")
        self.assertEqual(v2["model"]["attention_embedding_dim"], 8)
        self.assertEqual(v2["model"]["residual_embedding_dim"], 8)
        self.assertEqual(v2["model"]["attention_scale"], 2)
        self.assertEqual(v2["loss"], diffusion["loss"])
        self.assertNotIn("question_embedding_dim", v2["model"])
        self.assertIsInstance(build_model("newq_v2", {**kwargs, "diffusion_steps":4}), NewQV2)
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("unknown", kwargs)

    def test_saved_config_model_selection_and_cli_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text("model_name: linearq\nmodel:\n  embedding_dim: 16\n", encoding="utf-8")
            self.assertEqual(load_config(path)["model_name"], "linearq")
            overridden = load_config(path, model_name="diffuqkt")
            self.assertEqual(overridden["model_name"], "diffuqkt")
            self.assertEqual(overridden["model"]["embedding_dim"], 16)
            self.assertIn("diffusion_steps", overridden["model"])

    def test_evaluation_recovers_linearq_and_legacy_diffuqkt_without_model_argument(self):
        torch.set_num_threads(2)
        batch = example_batch()
        items = [{key: value[row] for key, value in batch.items()} for row in range(2)]
        loader = DataLoader(items, batch_size=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            for name in ("linearq", "diffuqkt", "newq", "newq_v2"):
                with self.subTest(model=name):
                    kwargs = dict(num_problems=8, num_skills=4, embedding_dim=8, dropout=0)
                    if name in {"diffuqkt", "newq", "newq_v2"}:
                        kwargs["diffusion_steps"] = 4
                    model = build_model(name, kwargs)
                    checkpoint = {"epoch": 2, "model": model.state_dict(), "model_kwargs": kwargs,
                                  "config": {"train": {}}, "metadata_sha256": sha256(root / "metadata.json")}
                    if name != "diffuqkt":
                        checkpoint["model_name"] = name
                    self.assertEqual(checkpoint_model_name(checkpoint), name)
                    checkpoint_path = root / f"{name}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    config = load_config()
                    config["train"].update(device="cpu", num_threads=2)
                    config["dataset"]["paths"]["processed"] = str(root)
                    with patch("src.training.trainer.make_loader", return_value=loader):
                        result = evaluate_checkpoint(config, checkpoint_path, verbose=False)
                    self.assertEqual(result["num_predictions"], 6)
                    self.assertIsNone(result["diffusion_loss"])
                    wrong = "diffuqkt" if name == "linearq" else "linearq"
                    with self.assertRaisesRegex(ValueError, "Checkpoint model"):
                        evaluate_checkpoint(config, checkpoint_path, verbose=False, expected_model=wrong)


if __name__ == "__main__":
    unittest.main()
