"""NewQ's concept sets, task gradients, diffusion conditions and baseline equivalence."""
import math
import unittest

import torch

from src.model import DiffuQKT, NewQ
from src.model.diffuqkt.diffusion import ConditionalQuestionDiffusion
from test_diffuqkt import example_batch


def multi_concept_batch():
    batch = example_batch()
    batch["skill_ids"] = torch.tensor([
        [[1, 2, 0], [2, 3, 0], [1, 3, 4], [1, 2, 4], [2, 3, 0]],
        [[2, 3, 0], [1, 4, 0], [1, 2, 3], [0, 0, 0], [0, 0, 0]]])
    return batch


class NewQTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.model = NewQ(8, 4, embedding_dim=8, num_heads=2, dropout=0,
                          diffusion_steps=4, question_embedding_dim=2)

    def test_hand_calculated_weights_residual_and_padding(self):
        batch = example_batch()
        batch["skill_ids"] = torch.tensor([
            [[2, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [2, 1, 0, 0], [2, 1, 0, 0]],
            [[1, 2, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0], [2, 1, 1, 0], [2, 1, 1, 0]]])
        with torch.no_grad():
            self.model.concept.weight.zero_()
            self.model.concept.weight[1, 0] = 1
            self.model.concept.weight[2, 1] = 1
            self.model.variation.weight.zero_()
            self.model.variation.weight[1, 0] = 2
            self.model.variation.weight[2, 1] = 4
            self.model.difficulty.weight.zero_()
            self.model.question_embedding.weight.fill_(0)
            self.model.question_embedding.weight[1:, 0] = 2
            self.model.question_embedding.weight[2, 0] = -2
            self.model.key_projection.weight.zero_()
            self.model.key_projection.weight[0, 0] = 1
            self.model.residual_projection.weight.zero_()
            self.model.residual_projection.weight[0, 0] = .1
        components = self.model.question_components(batch)
        probability = torch.sigmoid(torch.tensor(math.sqrt(2)))
        expected = torch.zeros(8)
        expected[0], expected[1] = 2 * probability + .2, 3 * (1-probability)
        torch.testing.assert_close(components["original"][0, 0], expected)
        weights, ids = components["concept_weights"], components["concept_ids"]
        self.assertEqual(int((weights[0, 0] > 0).sum()), 2)
        torch.testing.assert_close(weights.sum(-1)[0, :2], torch.ones(2))
        torch.testing.assert_close(weights[0, 0][ids[0, 0] == 1].sum(), probability)
        torch.testing.assert_close(weights[1, 1][ids[1, 1] == 1].sum(), 1-probability)
        torch.testing.assert_close(weights[0, 1].max(), torch.tensor(1.))
        self.assertTrue((weights[ids == 0] == 0).all())
        self.assertTrue((weights[~batch["valid_mask"]] == 0).all())
        self.assertTrue(torch.isfinite(components["original"]).all())
        for key in ["original", "concept", "fused_concept", "variation", "residual"]:
            self.assertTrue((components[key][~batch["valid_mask"]] == 0).all(), key)
        torch.testing.assert_close(components["concept"], components["fused_concept"] + components["residual"])

    def test_permutation_repeated_labels_and_observed_views(self):
        batch = multi_concept_batch()
        before = self.model.question_components(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["skill_ids"] = torch.cat([batch["skill_ids"].flip(-1), batch["skill_ids"]], -1)
        after = self.model.question_components(changed)
        torch.testing.assert_close(before["original"], after["original"])
        changed["problem_ids"][0, 1] = changed["problem_ids"][0, 0]
        views = self.model.question_components(changed)
        torch.testing.assert_close(views["residual"][0, 0], views["residual"][0, 1])
        self.assertFalse(torch.equal(views["fused_concept"][0, 0], views["fused_concept"][0, 1]))
        changed["answers"] = 1 - changed["answers"]
        changed["student_ids"] += 100
        torch.testing.assert_close(views["original"], self.model.question_components(changed)["original"])

    def test_uniform_weights_and_zero_residual_recover_complete_baseline(self):
        baseline = DiffuQKT(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4)
        shared = {key: self.model.state_dict()[key] for key in baseline.state_dict()}
        baseline.load_state_dict(shared, strict=True)
        with torch.no_grad():
            self.model.key_projection.weight.zero_()
            self.model.residual_projection.weight.zero_()
        batch = multi_concept_batch()
        torch.testing.assert_close(self.model.question_components(batch)["original"],
                                   baseline.question_components(batch)["original"], atol=1e-6, rtol=0)
        for training in [True, False]:
            baseline.train(training)
            self.model.train(training)
            torch.manual_seed(10)
            expected = baseline(batch, generator=torch.Generator().manual_seed(20))
            torch.manual_seed(10)
            actual = self.model(batch, generator=torch.Generator().manual_seed(20))
            for key, value in expected.items():
                if value is None:
                    self.assertIsNone(actual[key])
                else:
                    # Weighted sums and sum/count differ slightly in float32 rounding.
                    torch.testing.assert_close(actual[key], value, atol=3e-6, rtol=0, msg=key)
        self.assertIs(NewQ.forward, DiffuQKT.forward)
        self.assertIs(NewQ.enhance_questions, DiffuQKT.enhance_questions)
        self.assertIs(type(self.model.diffusion), ConditionalQuestionDiffusion)
        self.assertEqual(self.model.diffusion.denoiser[0].in_features, 3*8+1)

    def test_each_baseline_objective_trains_the_new_representation(self):
        batch = multi_concept_batch()
        for objective in ["kt_loss", "contrastive_loss", "diffusion_loss"]:
            with self.subTest(objective=objective):
                self.model.zero_grad(set_to_none=True)
                torch.manual_seed(82)
                result = self.model(batch)
                result[objective].backward()
                for name in ["question_embedding", "key_projection", "residual_projection",
                             "concept", "variation", "difficulty"]:
                    gradient = getattr(self.model, name).weight.grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(float(gradient.abs().sum()), 0, name)
                for name in ["question_embedding", "concept", "variation", "difficulty"]:
                    self.assertTrue((getattr(self.model, name).weight.grad[0] == 0).all(), name)
        torch.testing.assert_close(result["loss"], result["kt_loss"]
            + .001 * result["contrastive_loss"] + .01 * result["diffusion_loss"])

    def test_attention_is_differentiable_and_single_concept_has_zero_score_gradient(self):
        components = self.model.question_components(multi_concept_batch())
        weights = components["concept_weights"]
        weights.retain_grad()
        components["original"].square().mean().backward()
        self.assertGreater(float(weights.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.key_projection.weight.grad.abs().sum()), 0)
        self.model.zero_grad(set_to_none=True)
        single = self.model.question_components(example_batch())
        single["original"].square().mean().backward()
        self.assertTrue((self.model.key_projection.weight.grad == 0).all())
        self.assertGreater(float(self.model.residual_projection.weight.grad.abs().sum()), 0)

    def test_residual_reaches_evaluation_sampling_and_predictions_remain_causal(self):
        self.model.eval()
        batch = example_batch()
        components = self.model.question_components(batch)
        before, losses = self.model.enhance_questions(components, batch, torch.Generator().manual_seed(21))
        self.assertIsNone(losses["diffusion_loss"])
        with torch.no_grad():
            self.model.residual_projection.weight.add_(1)
        after, _ = self.model.enhance_questions(self.model.question_components(batch), batch,
                                               torch.Generator().manual_seed(21))
        self.assertFalse(torch.equal(before[batch["valid_mask"]], after[batch["valid_mask"]]))
        with torch.no_grad():
            predicted = self.model(batch, generator=torch.Generator().manual_seed(22))
            changed = {key: value.clone() for key, value in batch.items()}
            changed["answers"][:, 2:] = 1-changed["answers"][:, 2:]
            result = self.model(changed, generator=torch.Generator().manual_seed(22))
            torch.testing.assert_close(predicted["probabilities"][:, :3], result["probabilities"][:, :3])
            changed = {key: value.clone() for key, value in batch.items()}
            changed["problem_ids"][:, 3:] = 8
            changed["skill_ids"][:, 3:] = 4
            result = self.model(changed, generator=torch.Generator().manual_seed(22))
            torch.testing.assert_close(predicted["probabilities"][:, :3], result["probabilities"][:, :3])
            self.assertIsNone(predicted["contrastive_loss"])
            self.assertIsNone(predicted["diffusion_loss"])

    def test_budget_counts_full_model_and_rejects_more_than_four_times_baseline(self):
        baseline = sum(parameter.numel() for parameter in DiffuQKT(
            8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4).parameters())
        total = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertEqual(total-baseline, (8+1+2*8)*2)
        self.assertEqual(self.model.baseline_parameter_count, baseline)
        self.assertEqual(self.model.parameter_ratio, total/baseline)
        with self.assertRaisesRegex(ValueError, "4x baseline"):
            NewQ(100000, 4, embedding_dim=8, question_embedding_dim=128)
        for dimension in [0, -1, True, 1.5]:
            with self.assertRaisesRegex(ValueError, "question_embedding_dim"):
                NewQ(8, 4, question_embedding_dim=dimension)
        for temperature in [0, -1, float("nan"), float("inf")]:
            with self.assertRaisesRegex(ValueError, "attention_temperature"):
                NewQ(8, 4, attention_temperature=temperature)


if __name__ == "__main__":
    unittest.main()
