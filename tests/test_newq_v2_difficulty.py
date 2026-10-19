"""Optional explicit D conditioning, independent of D in the item representation."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.model import NewQV2
from src.model.newq_v2.diagnostics import diagnose_checkpoint
from src.model.registry import build_model
from src.training.trainer import evaluate_checkpoint, model_kwargs, train
from src.utils.config import load_config
from test_newq import multi_concept_batch


MODES = ["off", "condition", "target", "both"]
FUSIONS = [("add", False), ("concat", False), ("concat", True)]


class DifficultyConditionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.batch = multi_concept_batch()
        self.kwargs = dict(num_problems=8, num_skills=4, embedding_dim=8, num_heads=2,
                           dropout=0, diffusion_steps=4, attention_embedding_dim=2, residual_embedding_dim=2)

    def model(self, mode="both", fusion="add", keep=False, **extra):
        torch.manual_seed(173)
        return NewQV2(**self.kwargs, residual_diffusion_enabled=mode != "off",
                     residual_diffusion_role=mode if mode != "off" else "condition",
                     residual_fusion=fusion, residual_concat_keep_2d=keep, **extra)

    def test_boolean_validation_and_configuration_construction(self):
        self.assertIs(NewQV2(**self.kwargs).diffusion_difficulty_condition, True)
        for value in [0, 1, "true", "false", None, [], np.bool_(True)]:
            with self.assertRaisesRegex(ValueError, "diffusion_difficulty_condition must be a bool"):
                NewQV2(**self.kwargs, diffusion_difficulty_condition=value)
        config = load_config(model_name="newq_v2")
        self.assertIs(config["model"]["diffusion_difficulty_condition"], True)
        config["model"]["diffusion_difficulty_condition"] = False
        model = build_model("newq_v2", model_kwargs(config, {"num_problems": 8, "num_skills": 4}))
        self.assertIs(model.experiment_metadata()["diffusion_difficulty_condition"], False)
        self.assertIs(model.prepare_training(None)["diffusion_difficulty_condition"], False)
        for name in ["diffuqkt", "linearq", "newq", "dha_kt"]:
            self.assertNotIn("diffusion_difficulty_condition", load_config(model_name=name)["model"])

    def test_flag_preserves_parameters_rng_and_legacy_default_outputs(self):
        for mode in MODES:
            for fusion, keep in FUSIONS:
                default = self.model(mode, fusion, keep)
                expected_rng = torch.get_rng_state()
                for flag in [True, False]:
                    actual = self.model(mode, fusion, keep, diffusion_difficulty_condition=flag)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                    self.assertEqual(default.state_dict().keys(), actual.state_dict().keys())
                    for key, value in default.state_dict().items():
                        torch.testing.assert_close(actual.state_dict()[key], value, atol=0, rtol=0)
                    self.assertEqual(default.experiment_metadata()["parameter_count"],
                                     actual.experiment_metadata()["parameter_count"])
                    if flag:
                        for training in [True, False]:
                            outputs = []
                            for model in [default, actual]:
                                model.train(training)
                                torch.manual_seed(19)
                                outputs.append(model(self.batch, generator=torch.Generator().manual_seed(20)))
                            for key, value in outputs[0].items():
                                if value is None:
                                    self.assertIsNone(outputs[1][key])
                                else:
                                    torch.testing.assert_close(outputs[1][key], value, atol=0, rtol=0)

    def test_every_actual_denoising_step_obeys_switch_across_routes_and_widths(self):
        for mode in MODES:
            for fusion, keep in FUSIONS:
                for flag in [True, False]:
                    model = self.model(mode, fusion, keep, diffusion_difficulty_condition=flag,
                                       residual_condition_detach=True)
                    parts = model.question_components(self.batch)
                    d_column = model.diffusion.dim + getattr(model.diffusion, "output_dim", model.diffusion.dim)
                    for training in [True, False]:
                        seen = []
                        def capture(module, args):
                            seen.append(args[0].detach().clone())
                        model.train(training)
                        handle = model.diffusion.denoiser[0].register_forward_pre_hook(capture)
                        try:
                            _, enhanced, losses = model.generate_questions(parts, self.batch,
                                                                          torch.Generator().manual_seed(20))
                        finally:
                            handle.remove()
                        self.assertEqual(len(seen), 1 if training else model.diffusion.steps)
                        for inputs in seen:
                            torch.testing.assert_close(inputs[..., :8], parts["concept"])
                            expected = parts["scale"] if flag else torch.zeros_like(parts["scale"])
                            torch.testing.assert_close(inputs[..., d_column:d_column+1], expected)
                            if mode in {"condition", "both"}:
                                torch.testing.assert_close(inputs[..., -8:], parts["residual"])
                        self.assertTrue(torch.isfinite(enhanced).all())
                        self.assertTrue((enhanced[~self.batch["valid_mask"]] == 0).all())
                        if training:
                            self.assertTrue(torch.isfinite(losses["diffusion_loss"]))

    def test_denoiser_ignores_only_explicit_d_values_when_disabled(self):
        for mode in MODES:
            for fusion, keep in FUSIONS:
                for flag in [True, False]:
                    model = self.model(mode, fusion, keep, diffusion_difficulty_condition=flag)
                    parts = model.question_components(self.batch)
                    # Hold C, noisy target and R fixed: vary only the explicit condition.
                    changed = {**parts, "conditioning": parts["conditioning"] + .7}
                    for training in [True, False]:
                        model.train(training)
                        outputs = []
                        for components in [parts, changed]:
                            torch.manual_seed(19)
                            outputs.append(model.generate_questions(components, self.batch,
                                           torch.Generator().manual_seed(20))[0])
                        if flag:
                            self.assertFalse(torch.equal(*outputs))
                        else:
                            torch.testing.assert_close(*outputs, atol=0, rtol=0)

    def test_disabled_input_has_no_gradient_but_d_in_targets_still_learns(self):
        for mode in MODES:
            for fusion, keep in FUSIONS:
                for flag in [True, False]:
                    model = self.model(mode, fusion, keep, diffusion_difficulty_condition=flag)
                    parts = model.question_components(self.batch)
                    torch.testing.assert_close(parts["semantic_original"],
                                               parts["concept"] + parts["scale"] * parts["variation"])
                    generated, _, losses = model.generate_questions(parts, self.batch)
                    losses["diffusion_loss"].backward()
                    d_column = model.diffusion.dim + generated.shape[-1]
                    explicit_gradient = model.diffusion.denoiser[0].weight.grad[:, d_column]
                    if flag:
                        self.assertGreater(float(explicit_gradient.abs().sum()), 0)
                    else:
                        self.assertTrue((explicit_gradient == 0).all())
                    for module in [model.difficulty, model.concept, model.variation, model.diffusion.denoiser[2]]:
                        self.assertGreater(float(module.weight.grad.abs().sum()), 0)

    def test_training_checkpoint_restore_and_diagnostics_use_saved_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.json").write_text('{"num_problems":8,"num_skills":4}', encoding="utf-8")
            for index, split in enumerate(["train", "valid", "test"]):
                np.savez_compressed(root / f"{split}.npz", problemId=self.batch["problem_ids"].numpy(),
                    skill=self.batch["skill_ids"].numpy(), correct=self.batch["answers"].numpy(),
                    studentId=self.batch["student_ids"].numpy()+index*10,
                    lengths=self.batch["valid_mask"].sum(-1).numpy())
            config = load_config(model_name="newq_v2")
            config["dataset"]["paths"]["processed"] = str(root)
            config["model"].update({key: value for key, value in self.kwargs.items()
                                    if key not in {"num_problems", "num_skills"}})
            config["model"].update(diffusion_difficulty_condition=False, residual_diffusion_enabled=True,
                residual_diffusion_role="both", residual_fusion="concat", residual_concat_keep_2d=True)
            config["train"].update(device="cpu", num_workers=0, num_threads=2, batch_size=2,
                epochs=1, preprocess_each_run=False, output_root=str(root / "experiments"))
            with redirect_stdout(io.StringIO()):
                summary = train(config)
            directory = Path(summary["output_dir"])
            self.assertIs(summary["diffusion_difficulty_condition"], False)
            saved = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            self.assertIs(saved["model"]["diffusion_difficulty_condition"], False)
            checkpoint_path = directory / "best.pt"
            fingerprint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertIs(checkpoint["model_kwargs"]["diffusion_difficulty_condition"], False)
            self.assertIs(checkpoint["diffusion_difficulty_condition"], False)
            config["model"]["diffusion_difficulty_condition"] = True
            built = []
            def factory(name, kwargs):
                loaded = build_model(name, kwargs)
                built.append(loaded)
                return loaded
            with patch("src.training.trainer.build_model", side_effect=factory):
                evaluate_checkpoint(config, checkpoint_path, verbose=False)
            self.assertIs(built[0].diffusion_difficulty_condition, False)
            report = diagnose_checkpoint(config, checkpoint_path, max_batches=1)
            self.assertIs(report["diffusion_difficulty_condition"], False)
            self.assertEqual(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), fingerprint)
            checkpoint["model_kwargs"].pop("diffusion_difficulty_condition")
            checkpoint.pop("diffusion_difficulty_condition")
            legacy_path = root / "legacy.pt"
            torch.save(checkpoint, legacy_path)
            legacy = build_model("newq_v2", checkpoint["model_kwargs"])
            legacy.load_state_dict(checkpoint["model"], strict=True)
            self.assertIs(legacy.diffusion_difficulty_condition, True)
            config["model"]["diffusion_difficulty_condition"] = False
            report = diagnose_checkpoint(config, legacy_path, max_batches=1)
            self.assertIs(report["diffusion_difficulty_condition"], True)


if __name__ == "__main__":
    unittest.main()
