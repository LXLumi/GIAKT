"""User regression: distinguish unevaluated auxiliary losses from zero gradients."""
import unittest
import torch
from torch.utils.data import DataLoader
from src.model.diffuqkt import DiffuQKT
from src.training.trainer import run_epoch
from test_diffuqkt import example_batch


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = DiffuQKT(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4)

    def test_eval_forward_marks_uncomputed_losses_as_unavailable(self):
        self.model.eval()
        with torch.no_grad():
            result = self.model(example_batch(), generator=torch.Generator().manual_seed(2026))
        self.assertIsNone(result["contrastive_loss"])
        self.assertIsNone(result["diffusion_loss"])
        torch.testing.assert_close(result["loss"], result["kt_loss"])

    def test_eval_epoch_reports_prediction_loss_without_auxiliary_zeros(self):
        batch = example_batch()
        items = [{key: value[row] for key, value in batch.items()} for row in range(2)]
        metrics = run_epoch(self.model, DataLoader(items, batch_size=2), torch.device("cpu"))
        self.assertEqual(metrics["loss_mode"], "prediction_only")
        self.assertIsNone(metrics["contrastive_loss"])
        self.assertIsNone(metrics["diffusion_loss"])
        self.assertEqual(metrics["loss"], metrics["kt_loss"])

    def test_each_auxiliary_loss_independently_updates_expected_parameters(self):
        for loss_key, parameter_names in [
            ("contrastive_loss", ["diffusion.denoiser.0.weight",
                                  "retriever.interaction_encoder.attention.v_proj.weight"]),
            ("diffusion_loss", ["diffusion.denoiser.0.weight", "diffusion.time_mlp.0.weight"]),
        ]:
            with self.subTest(loss=loss_key):
                self.model.train()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=.002, weight_decay=0)
                parameters = dict(self.model.named_parameters())
                before = {name: parameters[name].detach().clone() for name in parameter_names}
                result = self.model(example_batch())
                self.assertGreater(float(result[loss_key].detach()), 0)
                self.assertTrue(result[loss_key].requires_grad)
                optimizer.zero_grad(set_to_none=True)
                result[loss_key].backward()
                for name in parameter_names:
                    gradient = parameters[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(float(gradient.abs().sum()), 0, name)
                optimizer.step()
                for name in parameter_names:
                    self.assertFalse(torch.equal(parameters[name].detach(), before[name]), name)

    def test_train_epoch_reports_joint_loss_and_real_auxiliary_values(self):
        batch = example_batch()
        items = [{key: value[row] for key, value in batch.items()} for row in range(2)]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.002, weight_decay=0)
        metrics = run_epoch(self.model, DataLoader(items, batch_size=2), torch.device("cpu"), optimizer)
        self.assertEqual(metrics["loss_mode"], "joint_training")
        self.assertGreater(metrics["contrastive_loss"], 0)
        self.assertGreater(metrics["diffusion_loss"], 0)
        self.assertAlmostEqual(metrics["loss"], metrics["kt_loss"]
            + .001 * metrics["contrastive_loss"] + .01 * metrics["diffusion_loss"], places=6)


if __name__ == "__main__":
    unittest.main()
