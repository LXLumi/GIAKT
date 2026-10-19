"""Linear residuals, unchanged downstream behavior and end-to-end learning."""
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.model.linearq import LinearQ
from src.model.diffuqkt import DiffuQKT
from src.training.trainer import run_epoch
from test_diffuqkt import example_batch


class LinearQTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = LinearQ(8, 4, embedding_dim=8, num_heads=2, dropout=0)
        self.model.set_training_items([1, 2, 3, 4, 5])

    def test_question_specific_offsets_formula_and_zero_offset_baseline(self):
        batch = example_batch()
        # Equal skill and difficulty must still allow different question residuals.
        batch["skill_ids"][0, :2] = 1
        with torch.no_grad():
            self.model.difficulty.weight.fill_(0)
            self.model.question_embedding.weight[1].fill_(1.)
            self.model.question_embedding.weight[2].fill_(2.)
            self.model.linear_a.weight.copy_(torch.eye(8))
            self.model.linear_b.weight.copy_(torch.eye(8) * 3)
        parts = self.model.question_components(batch)
        torch.testing.assert_close(parts["original"][0, 0], parts["original"][0, 1])
        torch.testing.assert_close(parts["A"][0, 1], parts["A"][0, 0] * 2)
        torch.testing.assert_close(parts["B"][0, 1], parts["B"][0, 0] * 2)
        torch.testing.assert_close(parts["enhanced"] - parts["original"],
                                   parts["A"] + parts["scale"] * parts["B"])
        self.assertFalse(torch.equal(parts["enhanced"][0, 0], parts["enhanced"][0, 1]))
        self.assertTrue((parts["enhanced"][~batch["valid_mask"]] == 0).all())
        with torch.no_grad():
            self.model.linear_a.weight.zero_()
            self.model.linear_b.weight.zero_()
        identity = self.model.question_components(batch)
        torch.testing.assert_close(identity["enhanced"], identity["original"])

    def test_kt_and_contrastive_each_train_both_linear_heads_and_question_embeddings(self):
        for objective in ("kt_loss", "contrastive_loss"):
            with self.subTest(objective=objective):
                self.model.zero_grad(set_to_none=True)
                result = self.model(example_batch())
                result[objective].backward()
                for name in ("linear_a.weight", "linear_b.weight", "question_embedding.weight",
                             "concept.weight", "variation.weight", "difficulty.weight"):
                    gradient = dict(self.model.named_parameters())[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(float(gradient.abs().sum()), 0, name)
                self.assertTrue((self.model.question_embedding.weight.grad[0] == 0).all())

    def test_loss_includes_residual_without_diffusion_network_and_parameters_update(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.002)
        before = self.model.linear_a.weight.detach().clone()
        result = self.model(example_batch())
        self.assertGreater(float(result["diffusion_loss"].detach()), 0)
        self.assertFalse(hasattr(self.model, "diffusion"))
        self.assertFalse(any(name.startswith("diffusion.") for name in self.model.state_dict()))
        torch.testing.assert_close(result["loss"], result["kt_loss"] + .001 * result["contrastive_loss"]
                                   + .01 * result["diffusion_loss"])
        result["loss"].backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.linear_a.weight.detach()))

    def test_item_penalty_deduplicates_training_ids_and_normalizes_over_items_and_dimensions(self):
        self.model.set_training_items([0, 1, 1, 1, 2])
        with torch.no_grad():
            self.model.question_embedding.weight[1].fill_(1.)
            self.model.question_embedding.weight[2].fill_(3.)
            self.model.question_embedding.weight[3:].fill_(100.)
            self.model.linear_a.weight.copy_(torch.eye(8))
            self.model.linear_b.weight.zero_()
        self.assertEqual(self.model.train_item_ids.tolist(), [1, 2])
        # Equal item weights: (8*1^2 + 8*3^2) / (8*2), regardless of repetitions.
        torch.testing.assert_close(self.model.item_regularization_loss(), torch.tensor(5.))

    def test_interaction_frequency_changes_neither_item_penalty_nor_its_gradients(self):
        original = example_batch()
        repeated = {key: value.repeat((3,) + (1,) * (value.ndim - 1))
                    for key, value in original.items()}
        repeated["problem_ids"][repeated["valid_mask"]] = 1
        params = [self.model.linear_a.weight, self.model.linear_b.weight,
                  self.model.question_embedding.weight, self.model.difficulty.weight]
        first = self.model(original)["diffusion_loss"]
        first_grads = torch.autograd.grad(first, params)
        second = self.model(repeated)["diffusion_loss"]
        second_grads = torch.autograd.grad(second, params)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        for before, after in zip(first_grads, second_grads):
            torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_training_item_set_is_required_and_evaluation_needs_no_training_ids(self):
        uninitialized = LinearQ(8, 4, embedding_dim=8, dropout=0)
        with self.assertRaisesRegex(RuntimeError, "train_item_ids"):
            uninitialized(example_batch())
        uninitialized.eval()
        self.assertIsNone(uninitialized(example_batch())["diffusion_loss"])
        self.assertNotIn("train_item_ids", uninitialized.state_dict())
        for ids in ([], [0], [-1, 1], [9]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                uninitialized.set_training_items(ids)

    def test_item_penalty_epoch_report_averages_steps_instead_of_interactions(self):
        batch = example_batch()
        items = [{key: value[row] for key, value in batch.items()} for row in range(2)]
        values = iter([1., 3.])
        def regularizer():
            return self.model.linear_a.weight.sum() * 0 + next(values)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.)
        with patch.object(self.model, "item_regularization_loss", side_effect=regularizer):
            metrics = run_epoch(self.model, DataLoader(items, batch_size=1), torch.device("cpu"), optimizer)
        # The two batches have 5 and 3 events: event weighting would incorrectly give 1.75.
        self.assertEqual(metrics["diffusion_loss"], 2.)

    def test_residual_alone_trains_both_heads_and_question_vectors_and_decreases(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.1)
        batch = example_batch()
        before = self.model(batch)["diffusion_loss"]
        before.backward()
        for name in ("linear_a.weight", "linear_b.weight", "question_embedding.weight", "difficulty.weight"):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        self.assertTrue((self.model.question_embedding.weight.grad[0] == 0).all())
        optimizer.step()
        after = self.model(batch)["diffusion_loss"]
        self.assertLess(float(after.detach()), float(before.detach()))

    def test_residual_weight_only_changes_total_loss_and_negative_weight_is_rejected(self):
        other = LinearQ(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_weight=.2,
                        train_item_ids=self.model.train_item_ids.tolist())
        other.load_state_dict(self.model.state_dict(), strict=True)
        batch = example_batch()
        default = self.model(batch)
        changed = other(batch)
        torch.testing.assert_close(default["probabilities"], changed["probabilities"], atol=0, rtol=0)
        torch.testing.assert_close(changed["loss"], changed["kt_loss"] + .001 * changed["contrastive_loss"]
                                   + .2 * changed["diffusion_loss"])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            LinearQ(8, 4, diffusion_weight=-.01)

    def test_same_qstar_and_downstream_weights_match_diffuqkt(self):
        diffusion = DiffuQKT(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4)
        for name in ("answer", "retriever", "predictor"):
            getattr(diffusion, name).load_state_dict(getattr(self.model, name).state_dict())
        self.model.eval()
        diffusion.eval()
        batch = example_batch()
        with torch.no_grad():
            enhanced = self.model.question_components(batch)["enhanced"]
            linear_output = self.model(batch)
            with patch.object(diffusion.diffusion, "sample", return_value=enhanced):
                diffusion_output = diffusion(batch)
        for key in ("states", "logits", "probabilities", "kt_loss", "loss"):
            torch.testing.assert_close(linear_output[key], diffusion_output[key], atol=0, rtol=0)

    def test_eval_is_repeatable_and_preserves_temporal_causality_and_padding(self):
        self.model.eval()
        batch = example_batch()
        with torch.no_grad():
            before = self.model(batch, generator=torch.Generator().manual_seed(1))
            repeat = self.model(batch, generator=torch.Generator().manual_seed(99))
            changed = {key: value.clone() for key, value in batch.items()}
            changed["answers"][:, 2:] = 1 - changed["answers"][:, 2:]
            changed["problem_ids"][0, 3:] = torch.tensor([7, 8])
            changed["skill_ids"][0, 3:] = 4
            changed["problem_ids"][1, 3:] = 8
            changed["skill_ids"][1, 3:] = 4
            after = self.model(changed)
        torch.testing.assert_close(before["probabilities"], repeat["probabilities"], atol=0, rtol=0)
        torch.testing.assert_close(before["probabilities"][:, :3], after["probabilities"][:, :3], atol=1e-6, rtol=0)
        self.assertTrue((after["states"][~batch["valid_mask"]] == 0).all())
        self.assertIsNone(before["contrastive_loss"])
        self.assertIsNone(before["diffusion_loss"])

    def test_multiskill_mean_preserves_offsets_and_is_order_invariant(self):
        batch = example_batch()
        first = batch["skill_ids"]
        second = torch.where(first.ne(0), first.remainder(4) + 1, first)
        batch["skill_ids"] = torch.stack((first, second, torch.zeros_like(first)), dim=-1)
        parts = self.model.question_components(batch)
        valid = batch["valid_mask"].unsqueeze(-1)
        expected_c = (self.model.concept(first) + self.model.concept(second)) / 2 * valid
        torch.testing.assert_close(parts["concept"], expected_c)
        shuffled = {key: value.clone() for key, value in batch.items()}
        shuffled["skill_ids"] = shuffled["skill_ids"].flip(-1)
        self.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model(batch)["probabilities"], self.model(shuffled)["probabilities"])


if __name__ == "__main__":
    unittest.main()
