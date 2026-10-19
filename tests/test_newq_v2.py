"""Acceptance checks for semantic diffusion, independent codes and cosine fusion."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

from src.model import DiffuQKT, NewQ, NewQV2
from src.model.registry import build_model
from src.model.newq_v2.diagnostics import diagnose_checkpoint, prediction_metrics
from src.utils.config import load_config
from src.utils.files import sha256
from test_diffuqkt import example_batch
from test_newq import multi_concept_batch


class NewQV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(113)
        self.model = NewQV2(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4,
                             attention_embedding_dim=2, residual_embedding_dim=2)

    def test_manual_cosine_weights_semantic_and_original(self):
        batch = multi_concept_batch()
        self.assertFalse(torch.equal(self.model.attention_question_embedding.weight[1:], self.model.residual_question_embedding.weight[1:]))
        batch["skill_ids"][0, 0] = torch.tensor([2, 1, 1])
        with torch.no_grad():
            for embedding in [self.model.concept, self.model.variation, self.model.difficulty,
                              self.model.attention_question_embedding, self.model.residual_question_embedding]:
                embedding.weight.zero_()
            self.model.concept.weight[1, 0] = 1
            self.model.concept.weight[2, 1] = 1
            self.model.variation.weight[1, 0] = 2
            self.model.variation.weight[2, 1] = 4
            self.model.attention_question_embedding.weight[1:, 0] = 1
            self.model.residual_question_embedding.weight[1:, 0] = 1
            self.model.key_projection.weight.zero_()
            self.model.key_projection.weight[0, 0] = 1
            self.model.key_projection.weight[1, 1] = 1
            self.model.residual_projection.weight.zero_()
            self.model.residual_projection.weight[0, 0] = .2
        parts = self.model.question_components(batch)
        probability = torch.sigmoid(torch.tensor(2.))
        expected = torch.zeros(8)
        expected[0], expected[1] = 2*probability, 3*(1-probability)
        torch.testing.assert_close(parts["semantic_original"][0, 0], expected)
        expected[0] += .2
        torch.testing.assert_close(parts["original"][0, 0], expected)
        torch.testing.assert_close(parts["concept_weights"][0, 0][parts["concept_ids"][0, 0] == 1].sum(), probability)
        torch.testing.assert_close(parts["concept"], parts["fused_concept"])
        self.assertEqual(parts["attention_question_embedding"].shape, (2, 5, 2))
        self.assertEqual(parts["residual_question_embedding"].shape, (2, 5, 2))
        self.assertIsNot(self.model.attention_question_embedding.weight, self.model.residual_question_embedding.weight)
        self.assertNotIn("question_embedding.weight", self.model.state_dict())
        self.assertNotIn("attention_scale", dict(self.model.named_parameters()))

    def test_sets_empty_concepts_zero_queries_keys_and_padding_are_safe(self):
        batch = multi_concept_batch()
        before = self.model.question_components(batch)
        changed = {key:value.clone() for key,value in batch.items()}
        changed["skill_ids"] = torch.cat([batch["skill_ids"].flip(-1), batch["skill_ids"]], -1)
        after = self.model.question_components(changed)
        torch.testing.assert_close(before["original"], after["original"])
        mask = before["concept_mask"]
        self.assertTrue((before["concept_weights"][~mask] == 0).all())
        torch.testing.assert_close(before["concept_weights"].sum(-1)[batch["valid_mask"]],
                                   torch.ones(int(batch["valid_mask"].sum())))
        changed["skill_ids"][0, 0] = 0
        empty = self.model.question_components(changed)
        for key in ["concept", "variation", "semantic_original", "concept_weights"]:
            self.assertTrue((empty[key][0, 0] == 0).all(), key)
        torch.testing.assert_close(empty["original"][0, 0], empty["residual"][0, 0])
        changed["problem_ids"][~batch["valid_mask"]] = 8
        changed["skill_ids"][~batch["valid_mask"]] = 4
        padded = self.model.question_components(changed)
        for key in ["concept", "variation", "semantic_original", "residual", "original",
                    "attention_question_embedding", "residual_question_embedding", "concept_weights"]:
            self.assertTrue((padded[key][~batch["valid_mask"]] == 0).all(), key)
        for module in [self.model.attention_question_embedding, self.model.key_projection]:
            with torch.no_grad():
                module.weight.zero_()
            zero = self.model.question_components(changed)
            for key in ["concept_weights", "original"]:
                self.assertTrue(torch.isfinite(zero[key]).all())
        single = self.model.question_components(example_batch())
        torch.testing.assert_close(single["concept_weights"].squeeze(-1)[batch["valid_mask"]],
                                   torch.ones(int(batch["valid_mask"].sum())))

    def test_training_and_sampling_calls_exclude_residual_and_add_it_once(self):
        batch = multi_concept_batch()
        parts = self.model.question_components(batch)
        semantic_star = torch.full_like(parts["semantic_original"], 3.)
        loss = torch.tensor(.37)
        with patch.object(self.model.diffusion, "training_representation", return_value=(semantic_star, loss)) as training:
            enhanced, losses = self.model.enhance_questions(parts, batch)
        args = training.call_args.args
        for actual, name in zip(args[:3], ["semantic_original", "concept", "conditioning"]):
            self.assertIs(actual, parts[name])
        self.assertIs(args[3], batch["valid_mask"])
        self.assertIs(losses["diffusion_loss"], loss)
        torch.testing.assert_close(enhanced, (semantic_star+parts["residual"])*batch["valid_mask"].unsqueeze(-1))
        self.model.eval()
        generator = torch.Generator().manual_seed(41)
        with patch.object(self.model.diffusion, "sample", return_value=semantic_star) as sampling:
            enhanced, losses = self.model.enhance_questions(parts, batch, generator)
        args = sampling.call_args.args
        self.assertIs(args[0], parts["concept"])
        self.assertIs(args[1], parts["conditioning"])
        self.assertIs(args[2], batch["valid_mask"])
        self.assertIs(args[3], generator)
        torch.testing.assert_close(enhanced, (semantic_star+parts["residual"])*batch["valid_mask"].unsqueeze(-1))
        self.assertIsNone(losses["diffusion_loss"])

    def test_changing_residual_changes_only_final_outputs_in_both_modes(self):
        batch = multi_concept_batch()
        for training in [True, False]:
            with self.subTest(training=training):
                self.model.train(training)
                before = self.model.question_components(batch)
                torch.manual_seed(47)
                q_before, losses_before = self.model.enhance_questions(before, batch, torch.Generator().manual_seed(42))
                with torch.no_grad():
                    self.model.residual_question_embedding.weight[1:].add_(.07)
                after = self.model.question_components(batch)
                torch.manual_seed(47)
                q_after, losses_after = self.model.enhance_questions(after, batch, torch.Generator().manual_seed(42))
                for key in ["semantic_original", "concept", "conditioning", "concept_weights"]:
                    torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
                if training:
                    torch.testing.assert_close(losses_before["diffusion_loss"], losses_after["diffusion_loss"], atol=0, rtol=0)
                torch.testing.assert_close(q_after-q_before, after["residual"]-before["residual"], atol=5e-7, rtol=1e-5)

    def test_objective_gradients_respect_independent_parameter_paths(self):
        batch = multi_concept_batch()
        attention = [self.model.attention_question_embedding, self.model.key_projection]
        residual = [self.model.residual_question_embedding, self.model.residual_projection]
        for objective in ["kt_loss", "contrastive_loss", "diffusion_loss"]:
            with self.subTest(objective=objective):
                self.model.zero_grad(set_to_none=True)
                torch.manual_seed(49)
                result = self.model(batch)
                result[objective].backward()
                for module in attention:
                    self.assertIsNotNone(module.weight.grad)
                    self.assertTrue(torch.isfinite(module.weight.grad).all())
                    self.assertGreater(float(module.weight.grad.abs().sum()), 0)
                for module in residual:
                    gradient = module.weight.grad
                    if objective == "diffusion_loss":
                        self.assertTrue(gradient is None or (gradient == 0).all())
                    else:
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(float(gradient.abs().sum()), 0)
                for module in [self.model.concept, self.model.variation, self.model.difficulty,
                               self.model.attention_question_embedding, self.model.residual_question_embedding]:
                    if module.weight.grad is not None:
                        self.assertTrue((module.weight.grad[0] == 0).all())
                if objective == "diffusion_loss":
                    saved_residual = [module.weight.detach().clone() for module in residual]
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=.002, weight_decay=1e-5)
                    optimizer.step()
                    for module, saved in zip(residual, saved_residual):
                        torch.testing.assert_close(module.weight, saved, atol=0, rtol=0)
        self.model.zero_grad(set_to_none=True)
        self.model.question_components(batch)["residual"].square().sum().backward()
        for module in attention:
            self.assertIsNone(module.weight.grad)
        self.model.zero_grad(set_to_none=True)
        self.model.question_components(batch)["semantic_original"].square().sum().backward()
        for module in residual:
            self.assertIsNone(module.weight.grad)
        self.model.zero_grad(set_to_none=True)
        self.model.question_components(example_batch())["original"].square().sum().backward()
        for module in attention:
            self.assertTrue((module.weight.grad == 0).all())
        torch.testing.assert_close(result["loss"], result["kt_loss"]+.001*result["contrastive_loss"]+.01*result["diffusion_loss"])

    def test_contrastive_original_and_prediction_branches_both_keep_residual(self):
        batch = multi_concept_batch()
        parts = self.model.question_components(batch)
        semantic_star = parts["semantic_original"]*.3
        captured = []
        handle = self.model.retriever.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        try:
            with patch.object(self.model.diffusion, "training_representation", return_value=(semantic_star, torch.tensor(.2))):
                self.model(batch)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(captured[0], semantic_star+parts["residual"])
        torch.testing.assert_close(captured[1], parts["semantic_original"]+parts["residual"])

    def test_uniform_weights_zero_residual_recover_baseline_training_and_sampling(self):
        baseline = DiffuQKT(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4)
        baseline.load_state_dict({key:self.model.state_dict()[key] for key in baseline.state_dict()}, strict=True)
        with torch.no_grad():
            self.model.key_projection.weight.zero_()
            self.model.residual_projection.weight.zero_()
        batch = multi_concept_batch()
        for training in [True, False]:
            self.model.train(training)
            baseline.train(training)
            torch.manual_seed(53)
            expected = baseline(batch, generator=torch.Generator().manual_seed(51))
            torch.manual_seed(53)
            actual = self.model(batch, generator=torch.Generator().manual_seed(51))
            for key, value in expected.items():
                if value is None:
                    self.assertIsNone(actual[key])
                else:
                    torch.testing.assert_close(actual[key], value, atol=4e-6, rtol=1e-5, msg=key)
        self.assertIs(NewQV2.forward, DiffuQKT.forward)

    def test_causality_padding_and_observed_label_views(self):
        self.model.eval()
        batch = multi_concept_batch()
        with torch.no_grad():
            expected = self.model(batch, generator=torch.Generator().manual_seed(61))
            changed = {key:value.clone() for key,value in batch.items()}
            changed["answers"][:, 2:] = 1-changed["answers"][:, 2:]
            actual = self.model(changed, generator=torch.Generator().manual_seed(61))
            torch.testing.assert_close(actual["probabilities"][:, :3], expected["probabilities"][:, :3])
            changed = {key:value.clone() for key,value in batch.items()}
            changed["problem_ids"][:, 3:] = 8
            changed["skill_ids"][:, 3:] = 4
            actual = self.model(changed, generator=torch.Generator().manual_seed(61))
            torch.testing.assert_close(actual["probabilities"][:, :3], expected["probabilities"][:, :3])
            changed = {key:value.clone() for key,value in batch.items()}
            changed["problem_ids"][~batch["valid_mask"]] = 8
            changed["skill_ids"][~batch["valid_mask"]] = 4
            changed["answers"][~batch["valid_mask"]] = 1
            actual = self.model(changed, generator=torch.Generator().manual_seed(61))
            torch.testing.assert_close(actual["probabilities"][batch["valid_mask"]], expected["probabilities"][batch["valid_mask"]])
        changed["problem_ids"][0, 1] = changed["problem_ids"][0, 0]
        parts = self.model.question_components(changed)
        torch.testing.assert_close(parts["residual"][0, 0], parts["residual"][0, 1])
        self.assertFalse(torch.equal(parts["concept_weights"][0, 0], parts["concept_weights"][0, 1]))
        changed["answers"] = 1-changed["answers"]
        changed["student_ids"] += 100
        torch.testing.assert_close(parts["original"], self.model.question_components(changed)["original"])

    def test_actual_parameters_default_budget_warning_hard_limit_and_validation(self):
        for problems, skills, baseline_count, total in [(17645,123,546165,830549),(2210,101,525098,562522)]:
            model = NewQV2(problems, skills)
            self.assertEqual(model.baseline_parameter_count, baseline_count)
            self.assertEqual(sum(p.numel() for p in model.parameters()), total)
            self.assertEqual(total-baseline_count, (problems+129)*16)
            self.assertLess(model.parameter_ratio, 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = NewQV2(10000,4,attention_embedding_dim=64,residual_embedding_dim=64)
        self.assertGreater(model.parameter_ratio, 3)
        self.assertLessEqual(model.parameter_ratio, 4)
        self.assertTrue(any("3x target" in str(item.message) for item in caught))
        with self.assertRaisesRegex(ValueError,"4x baseline"):
            NewQV2(17645,123,attention_embedding_dim=128,residual_embedding_dim=128)
        for field in ["attention_embedding_dim", "residual_embedding_dim"]:
            for value in [0,-1,True,1.5,None]:
                with self.assertRaisesRegex(ValueError,field):
                    NewQV2(8,4,**{field:value})
        for field in ["attention_scale", "attention_temperature"]:
            for value in [0,-1,True,float("nan"),float("inf"),"2"]:
                with self.assertRaisesRegex(ValueError,field):
                    NewQV2(8,4,**{field:value})
        invalid = example_batch()
        invalid["skill_ids"] = torch.zeros(2,5,0,dtype=torch.long)
        with self.assertRaisesRegex(ValueError,"skill_ids"):
            self.model.question_components(invalid)
        old = NewQ(8,4,embedding_dim=8,dropout=0,diffusion_steps=4,question_embedding_dim=2)
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict(old.state_dict(),strict=True)
        self.assertIsInstance(build_model("newq_v2",dict(num_problems=8,num_skills=4,embedding_dim=8,diffusion_steps=4)),NewQV2)

    def test_stable_logit_bce_does_not_hide_extreme_predictions_with_probability_clipping(self):
        metrics = prediction_metrics(torch.tensor([1000.,-1000.]), torch.tensor([0,1]))
        self.assertEqual(metrics["stable_bce_from_logits"], 1000)
        self.assertLess(metrics["probability_bce"], 100)
        self.assertEqual(metrics["max_abs_logit"], 1000)

    def test_saved_weight_diagnostic_records_aligned_attention_and_gradient_scope_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = multi_concept_batch()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"num_problems":8,"num_skills":4}), encoding="utf-8")
            for split in ["train", "valid", "test"]:
                np.savez_compressed(root / f"{split}.npz", problemId=batch["problem_ids"].numpy(),
                    skill=batch["skill_ids"].numpy(), correct=batch["answers"].numpy(),
                    studentId=batch["student_ids"].numpy(), lengths=batch["valid_mask"].sum(-1).numpy())
            config = load_config(model_name="newq_v2")
            config["dataset"]["paths"]["processed"] = str(root)
            config["train"].update(device="cpu", num_threads=2, batch_size=2, num_workers=0)
            kwargs = dict(num_problems=8,num_skills=4,embedding_dim=8,dropout=0,diffusion_steps=4,
                          attention_embedding_dim=2,residual_embedding_dim=2)
            path = root / "checkpoint.pt"
            torch.save({"model_name":"newq_v2","model_kwargs":kwargs,"model":self.model.state_dict(),
                        "epoch":1,"config":config,"metadata_sha256":sha256(metadata)},path)
            before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            report = diagnose_checkpoint(config,path,max_batches=1)
            self.assertEqual(report["scope"]["valid_interactions"], 8)
            self.assertEqual(report["statistics"]["entropy"]["count"], 8)
            self.assertEqual(report["sampling_steps"], 4)
            self.assertEqual(report["predictions"]["num_predictions"], 6)
            self.assertEqual(report["gradient_probe"]["batches"], 1)
            diff = report["gradient_probe"]["objectives"]["diffusion_loss"]
            self.assertFalse(diff["residual_question_embedding"]["has_gradient"])
            self.assertFalse(diff["residual_projection"]["has_gradient"])
            for example in report["attention_examples"]:
                self.assertEqual(len(example["concept_ids"]),len(example["weights"]))
                self.assertEqual(example["concept_ids"],sorted(set(example["concept_ids"])))
                self.assertAlmostEqual(sum(example["weights"]),1,places=6)
            json.dumps(report,allow_nan=False)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
