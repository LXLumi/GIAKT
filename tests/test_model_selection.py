"""Exercise epoch selection, saved parameters and final validation/test reporting."""
import ast
import csv
from datetime import datetime
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from src.training.trainer import train, load_checkpoint
from src.utils.config import load_config


def metrics(auc):
    return {"auc": auc, "accuracy": .7123456789, "bce": .5123456789, "num_predictions": 2,
            "loss": .5123456789, "kt_loss": .5123456789, "contrastive_loss": None, "diffusion_loss": None,
            "seconds": 0., "batches": 1, "loss_mode": "prediction_only"}


class ModelSelectionTests(unittest.TestCase):
    def test_validation_selects_best_epoch_before_single_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output = root / "data", root / "run"
            data.mkdir()
            (data / "metadata.json").write_text(json.dumps({"num_problems": 8, "num_skills": 4}),
                                                 encoding="utf-8")
            config = load_config()
            config["dataset"]["paths"]["processed"] = str(data)
            config["model"].update(embedding_dim=8, dropout=0, diffusion_steps=4)
            config["train"].update(epochs=3, device="cpu", preprocess_each_run=True, output_dir=str(output))
            calls, epoch = [], 0

            def loader(config, split, shuffle=False):
                # The timer starts before NPZ loading and model/device initialization.
                clock.perf_counter.assert_called_once()
                class FakeDataset(list):
                    arrays = {}
                return SimpleNamespace(split=split, dataset=FakeDataset([0, 1]))

            def epoch_loop(model, loader, device, optimizer=None, *args, **kwargs):
                nonlocal epoch
                calls.append(loader.split)
                if optimizer is not None:
                    epoch += 1
                    # Give each epoch distinct weights so parameter selection is testable.
                    with torch.no_grad():
                        model.predictor[0].bias.fill_(epoch)
                    result = metrics(.7 + .05 * epoch)
                    result.update(kt_loss=.4 + .01 * epoch, contrastive_loss=float(epoch),
                                  diffusion_loss=.1 * epoch, loss_mode="joint_training")
                    return result
                # All three scores display as 0.8000; full precision must still select epoch 2.
                return metrics([.800011, .800019, .800010][epoch-1])

            def test_evaluation(config, checkpoint_path, split="test", output_dir=None, **kwargs):
                calls.append(split)
                saved = load_checkpoint(checkpoint_path, torch.device("cpu"))
                self.assertEqual(saved["epoch"], 2)
                self.assertTrue((saved["model"]["predictor.0.bias"] == 2).all())
                return metrics(.75123456789)

            console = io.StringIO()
            with patch("src.training.trainer.time") as clock, \
                 patch("scripts.preprocessing.preprocess_dataset",
                       side_effect=lambda config: clock.perf_counter.assert_not_called()), \
                 patch("src.training.trainer.make_loader", side_effect=loader), \
                 patch("src.training.trainer.run_epoch", side_effect=epoch_loop), \
                 patch("src.training.trainer.evaluate_checkpoint", side_effect=test_evaluation), \
                 redirect_stdout(console):
                clock.perf_counter.side_effect = [100., 3765.]
                summary = train(config)
            self.assertEqual(calls, ["train", "valid", "train", "valid", "train", "valid", "test"])
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["validation"]["auc"], .800019)
            self.assertEqual(summary["test"]["auc"], .75123456789)
            self.assertEqual(summary["validation"]["accuracy"], .7123456789)
            self.assertEqual(summary["validation"]["loss"], .5123456789)
            self.assertEqual(summary["selection_metric"], "validation_auc")
            self.assertAlmostEqual(summary["train"]["auc"], .8)
            self.assertEqual(summary["duration_seconds"], 3665.)
            finished = datetime.fromisoformat(summary["finished_at"])
            self.assertIsNotNone(finished.utcoffset())
            saved_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["finished_at"], summary["finished_at"])
            self.assertEqual(saved_summary["validation"], summary["validation"])
            self.assertEqual(saved_summary["test"], summary["test"])
            self.assertEqual(saved_summary["results_decimal_places"], 4)
            with (output / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(float(history[1]["valid_auc"]), .800019)
            self.assertEqual(float(history[1]["train_accuracy"]), .7123456789)
            with (root / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["experiment"] for row in rows],
                             [output.name, finished.strftime("%Y-%m-%d %H:%M:%S"), "61m5s"])
            self.assertTrue(all(row["备注"] == "" for row in rows))
            self.assertEqual(ast.literal_eval(rows[0]["loss"]), (.42, 2., .2))
            self.assertEqual(rows[1]["AUC"], "0.8000")
            self.assertEqual(rows[2]["AUC"], "0.7512")
            self.assertEqual(rows[0]["ACC"], "0.7123")
            self.assertEqual(rows[1]["loss"], "0.5123")
            self.assertEqual(json.loads((output/"valid_metrics.json").read_text())["auc"], .800019)
            log = console.getvalue()
            lines = log.splitlines()
            for index, line in enumerate(lines):
                if not line.startswith("Epoch "):
                    continue
                valid_line = lines[index + 1]
                self.assertIn("valid |", valid_line)
                for label in ("loss", "AUC", "ACC"):
                    self.assertEqual(line.index(label), valid_line.index(label))
                    self.assertRegex(line, rf"{label}\s+\d+\.\d{{4}}(?:\s|$)")
                    self.assertRegex(valid_line, rf"{label}\s+\d+\.\d{{4}}(?:\s|$)")
            self.assertEqual(len(re.findall(r"^Epoch ", log, re.MULTILINE)), 3)
            self.assertIn('"auc": "0.8000"', log)
            self.assertIn('"auc": "0.7512"', log)
            self.assertNotIn("0.800019", log)
            self.assertNotIn("0.7123456789", log)
            self.assertIn('"validation"', log)
            self.assertIn('"test"', log)
            last = load_checkpoint(output / "last.pt", torch.device("cpu"))
            self.assertEqual(last["epoch"], 3)
            self.assertTrue((last["model"]["predictor.0.bias"] == 3).all())
            self.assertIn("optimizer", last)
            self.assertEqual(last["validation"]["auc"], .800010)
            best = load_checkpoint(output / "best.pt", torch.device("cpu"))
            self.assertEqual(best["validation"]["auc"], .800019)


if __name__ == "__main__":
    unittest.main()
