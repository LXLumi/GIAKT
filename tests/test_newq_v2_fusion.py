"""Residual fusion widths, shared projection, diffusion routes and saved-model restoration."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
import yaml

from src.model import NewQV2
from src.model.registry import build_model
from src.model.newq_v2.diagnostics import diagnose_checkpoint
from src.training.trainer import evaluate_checkpoint, model_kwargs, train
from src.utils.config import load_config
from test_newq import multi_concept_batch


MODES = ["off", "condition", "target", "both"]


class ResidualFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.batch = multi_concept_batch()
        self.kwargs = dict(num_problems=8, num_skills=4, embedding_dim=8, num_heads=2,
                           dropout=0, diffusion_steps=4, attention_embedding_dim=2, residual_embedding_dim=2)

    def model(self, mode="off", keep=False, fusion="concat", detach=False):
        torch.manual_seed(173)
        return NewQV2(**self.kwargs, residual_diffusion_enabled=mode != "off",
                     residual_diffusion_role=mode if mode != "off" else "condition",
                     residual_condition_detach=detach, residual_fusion=fusion, residual_concat_keep_2d=keep)

    def test_validation_config_constructor_and_boolean_normalization(self):
        legacy = NewQV2(**self.kwargs)
        self.assertEqual(legacy.residual_fusion, "add")
        self.assertIs(legacy.residual_concat_keep_2d, False)
        for value in [True, False, None, "cat", "ADD", 0, []]:
            with self.assertRaisesRegex(ValueError, "residual_fusion"):
                NewQV2(**self.kwargs, residual_fusion=value)
        for value in [2, -1, .0, 1.0, "true", "0", None, [], np.bool_(True)]:
            with self.assertRaisesRegex(ValueError, "residual_concat_keep_2d"):
                NewQV2(**self.kwargs, residual_concat_keep_2d=value)
        config = load_config(model_name="newq_v2")
        self.assertIn(config["model"]["residual_fusion"], {"add", "concat"})
        self.assertIn(config["model"]["residual_concat_keep_2d"], (False, True))
        config["model"].update({key: value for key, value in self.kwargs.items()
                                if key not in {"num_problems", "num_skills"}})
        for value in [False, True, 0, 1]:
            config["model"].update(residual_fusion="concat", residual_concat_keep_2d=value)
            model = build_model("newq_v2", model_kwargs(config, {"num_problems": 8, "num_skills": 4}))
            self.assertIs(model.residual_concat_keep_2d, bool(value))
            self.assertEqual(model.question_representation_dim, 16 if value else 8)

    def test_add_ignores_concat_dimension_without_parameters_or_rng_changes(self):
        control = self.model(fusion="add")
        actual = self.model(fusion="add", keep=True)
        for key, value in control.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value, atol=0, rtol=0)
        self.assertEqual(control.state_dict().keys(), actual.state_dict().keys())
        for model in [control, actual]:
            torch.manual_seed(19)
            result = model(self.batch)
            if model is control:
                expected = result
            else:
                for key, value in result.items():
                    torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        self.assertFalse(actual.experiment_metadata()["effective_residual_concat_keep_2d"])

    def test_original_and_enhanced_use_identical_order_and_shared_projection(self):
        for keep in [False, True]:
            for mode in MODES:
                model = self.model(mode, keep)
                if not keep:
                    with torch.no_grad():
                        model.residual_fusion_projection.weight.copy_(torch.arange(128).reshape(8, 16) / 128)
                parts = model.question_components(self.batch)
                joined = torch.cat([parts["semantic_original"], parts["residual"]], dim=-1)
                expected = joined if keep else torch.nn.functional.linear(joined, model.residual_fusion_projection.weight)
                torch.testing.assert_close(parts["original"], expected)
                route = model.diffusion_route(parts)
                is_target = mode in {"target", "both"}
                self.assertIs(route["diffusion_target"], parts["original"] if is_target else parts["semantic_original"])
                generated = torch.full_like(route["diffusion_target"], .3)
                expected = generated
                if not is_target:
                    expected = torch.cat([generated, parts["residual"]], dim=-1)
                    if not keep:
                        expected = torch.nn.functional.linear(expected, model.residual_fusion_projection.weight)
                expected = expected * self.batch["valid_mask"].unsqueeze(-1)
                with patch.object(model.diffusion, "training_representation", return_value=(generated, torch.tensor(.4))):
                    _, enhanced, _ = model.generate_questions(parts, self.batch)
                torch.testing.assert_close(enhanced, expected)
                self.assertFalse(route["add_residual_after_generation"])
                self.assertEqual(route["fuse_residual_after_generation"], not is_target)

    def test_projection_starts_as_addition_and_both_blocks_remain_trainable(self):
        model = self.model()
        identity = torch.eye(8)
        torch.testing.assert_close(model.residual_fusion_projection.weight, torch.cat([identity, identity], dim=-1))
        parts = model.question_components(self.batch)
        torch.testing.assert_close(parts["original"], parts["semantic_original"] + parts["residual"])
        model(self.batch)["loss"].backward()
        gradient = model.residual_fusion_projection.weight.grad
        self.assertGreater(float(gradient[:, :8].abs().sum()), 0)
        self.assertGreater(float(gradient[:, 8:].abs().sum()), 0)

    def test_train_eval_shapes_and_existing_three_losses_in_all_routes(self):
        for keep in [False, True]:
            width = 16 if keep else 8
            for mode in MODES:
                model = self.model(mode, keep)
                parts = model.question_components(self.batch)
                self.assertEqual(parts["concept"].shape[-1], 8)
                self.assertEqual(parts["residual"].shape[-1], 8)
                self.assertEqual(parts["original"].shape[-1], width)
                self.assertEqual(model.answer.weight.shape[-1], width)
                output_width = 16 if keep and mode in {"target", "both"} else 8
                self.assertEqual(model.experiment_metadata()["diffusion_output_dim"], output_width)
                result = model(self.batch)
                self.assertEqual(result["states"].shape, (2, 5, width))
                self.assertEqual(result["logits"].shape, (2, 5))
                self.assertTrue(torch.isfinite(result["loss"]))
                torch.testing.assert_close(result["loss"], result["kt_loss"] + .001*result["contrastive_loss"] + .01*result["diffusion_loss"])
                result["loss"].backward()
                model.eval()
                first = model(self.batch, generator=torch.Generator().manual_seed(21))
                second = model(self.batch, generator=torch.Generator().manual_seed(21))
                torch.testing.assert_close(first["logits"], second["logits"], atol=0, rtol=0)
                self.assertIsNone(first["diffusion_loss"])
                self.assertIsNone(first["contrastive_loss"])

    def test_full_2d_target_sampling_keeps_inherited_equations_and_d_conditions(self):
        for variance in ["posterior", "fixed_large"]:
            for mode in ["target", "both"]:
                model = self.model(mode, True).eval()
                diffusion = model.diffusion
                diffusion.variance = variance
                self.assertEqual(diffusion.denoiser[0].in_features, (5 if mode == "both" else 4)*8+1)
                self.assertEqual(diffusion.denoiser[2].out_features, 16)
                parts = model.question_components(self.batch)
                condition = parts["residual"] if mode == "both" else None
                valid = self.batch["valid_mask"].unsqueeze(-1)
                generator = torch.Generator().manual_seed(19)
                state = torch.randn(2, 5, 16, generator=generator) * valid
                variances = diffusion.posterior_variance if variance == "posterior" else diffusion.fixed_large_variance
                with torch.no_grad():
                    for step in range(diffusion.steps - 1, -1, -1):
                        time = torch.full((2, 5), step, dtype=torch.long)
                        original = diffusion.predict_x0(state, time, parts["concept"], parts["conditioning"], condition)
                        state = diffusion.mean_coef_x0[step]*original + diffusion.mean_coef_xt[step]*state
                        if step:
                            state = state + variances[step].sqrt()*torch.randn(state.shape, generator=generator)
                        state = state * valid
                    actual = diffusion.sample(parts["concept"], parts["conditioning"], self.batch["valid_mask"],
                                              torch.Generator().manual_seed(19), residual_condition=condition)
                torch.testing.assert_close(actual, state, atol=0, rtol=0)

    def test_detach_blocks_only_condition_path_for_both_concat_widths(self):
        for keep in [False, True]:
            for mode in MODES:
                for detach in [False, True]:
                    model = self.model(mode, keep, detach=detach)
                    with torch.no_grad():
                        if mode in {"condition", "both"}:
                            model.diffusion.denoiser[0].weight[:, -8:].fill_(.1)
                        if keep and mode in {"target", "both"}:
                            model.diffusion.denoiser[0].weight[:, 16:24].fill_(.02)
                            model.diffusion.denoiser[2].weight[8:].fill_(.03)
                    for objective in ["kt_loss", "contrastive_loss", "diffusion_loss"]:
                        model.zero_grad(set_to_none=True)
                        torch.manual_seed(29)
                        model(self.batch)[objective].backward()
                        for module in [model.residual_question_embedding, model.residual_projection]:
                            gradient = module.weight.grad
                            disconnected = objective == "diffusion_loss" and (mode == "off" or mode == "condition" and detach)
                            if disconnected:
                                self.assertIsNone(gradient)
                            else:
                                self.assertIsNotNone(gradient)
                                self.assertGreater(float(gradient.abs().sum()), 0)
                        if not keep:
                            gradient = model.residual_fusion_projection.weight.grad
                            if objective == "diffusion_loss" and mode not in {"target", "both"}:
                                self.assertIsNone(gradient)
                            else:
                                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_causality_and_padding_preserved_for_both_concat_widths(self):
        for keep in [False, True]:
            for mode in MODES:
                model = self.model(mode, keep).eval()
                expected = model(self.batch, generator=torch.Generator().manual_seed(21))
                changed = {key: value.clone() for key, value in self.batch.items()}
                changed["answers"][:, 2:] = 1-changed["answers"][:, 2:]
                actual = model(changed, generator=torch.Generator().manual_seed(21))
                torch.testing.assert_close(actual["logits"][:, :3], expected["logits"][:, :3])
                changed = {key: value.clone() for key, value in self.batch.items()}
                changed["problem_ids"][:, 3:] = 8
                changed["skill_ids"][:, 3:] = 4
                actual = model(changed, generator=torch.Generator().manual_seed(21))
                torch.testing.assert_close(actual["logits"][:, :3], expected["logits"][:, :3])
                changed = {key: value.clone() for key, value in self.batch.items()}
                changed["problem_ids"][~changed["valid_mask"]] = 8
                changed["skill_ids"][~changed["valid_mask"]] = 4
                actual = model(changed, generator=torch.Generator().manual_seed(21))
                torch.testing.assert_close(actual["logits"][changed["valid_mask"]], expected["logits"][changed["valid_mask"]])
                parts = model.question_components(changed)
                self.assertTrue((parts["original"][~changed["valid_mask"]] == 0).all())

    def test_full_parameter_budget_counts_wider_downstream_and_diffusion(self):
        expected = {False: {"off": 595290, "condition": 628058, "target": 595290, "both": 628058},
                    True: {"off": 1549914, "condition": 1582682, "target": 1615578, "both": 1648346}}
        for keep in [False, True]:
            for mode in MODES:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = NewQV2(2210, 101, residual_fusion="concat", residual_concat_keep_2d=keep,
                                   residual_diffusion_enabled=mode != "off", residual_diffusion_role=mode if mode != "off" else "condition")
                count = sum(p.numel() for p in model.parameters())
                self.assertEqual(count, expected[keep][mode])
                self.assertEqual(model.parameter_ratio, count/model.baseline_parameter_count)
        with self.assertRaisesRegex(ValueError, "4x baseline"):
            NewQV2(17645, 123, attention_embedding_dim=16, residual_embedding_dim=16,
                   residual_fusion="concat", residual_concat_keep_2d=True,
                   residual_diffusion_enabled=True, residual_diffusion_role="both")

    def test_training_saved_widths_evaluation_and_readonly_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"metadata.json").write_text('{"num_problems":8,"num_skills":4}', encoding="utf-8")
            for index, split in enumerate(["train", "valid", "test"]):
                np.savez_compressed(root/f"{split}.npz", problemId=self.batch["problem_ids"].numpy(),
                    skill=self.batch["skill_ids"].numpy(), correct=self.batch["answers"].numpy(),
                    studentId=self.batch["student_ids"].numpy()+index*10, lengths=self.batch["valid_mask"].sum(-1).numpy())
            config = load_config(model_name="newq_v2")
            config["dataset"]["paths"]["processed"] = str(root)
            config["model"].update(embedding_dim=8, dropout=0, diffusion_steps=4, attention_embedding_dim=2, residual_embedding_dim=2)
            config["train"].update(device="cpu", num_workers=0, num_threads=2, batch_size=2, epochs=1,
                                   preprocess_each_run=False, output_root=str(root/"experiments"))
            for keep in [0, 1]:
                for mode in MODES:
                    config["model"].update(residual_fusion="concat", residual_concat_keep_2d=keep,
                        residual_diffusion_enabled=mode != "off", residual_diffusion_role=mode if mode != "off" else "condition",
                        residual_condition_detach=True)
                    with redirect_stdout(io.StringIO()):
                        summary = train(config)
                    directory = Path(summary["output_dir"])
                    saved = yaml.safe_load((directory/"config.yaml").read_text(encoding="utf-8"))
                    self.assertIs(saved["model"]["residual_concat_keep_2d"], bool(keep))
                    self.assertEqual(summary["residual_fusion"], "concat")
                    self.assertEqual(summary["question_representation_dim"], 16 if keep else 8)
                    for filename in ["best.pt", "last.pt"]:
                        checkpoint = torch.load(directory/filename, map_location="cpu", weights_only=False)
                        self.assertEqual(checkpoint["model_kwargs"]["residual_fusion"], "concat")
                        self.assertIs(checkpoint["model_kwargs"]["residual_concat_keep_2d"], bool(keep))
                        loaded = build_model("newq_v2", checkpoint["model_kwargs"])
                        loaded.load_state_dict(checkpoint["model"], strict=True)
                    path = directory/"best.pt"
                    before = hashlib.sha256(path.read_bytes()).hexdigest()
                    config["model"].update(residual_fusion="add", residual_concat_keep_2d=not keep)
                    metrics = evaluate_checkpoint(config, path, verbose=False)
                    self.assertEqual(metrics["auc"], summary["test"]["auc"])
                    report = diagnose_checkpoint(config, path, max_batches=1)
                    self.assertEqual(report["residual_fusion"], "concat")
                    self.assertIs(report["residual_concat_keep_2d"], bool(keep))
                    self.assertEqual(report["question_representation_dim"], 16 if keep else 8)
                    self.assertEqual("residual_fusion_projection" in report["gradient_probe"]["objectives"]["kt_loss"], not bool(keep))
                    json.dumps(report, allow_nan=False)
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
