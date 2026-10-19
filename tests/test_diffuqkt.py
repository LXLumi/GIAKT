"""Regression checks for temporal leakage, masks, diffusion and gradient flow."""
import unittest
import torch
from src.model.diffuqkt import DiffuQKT, stable_state_contrastive
from src.model.diffuqkt.diffusion import ConditionalQuestionDiffusion


def example_batch():
    return {
        "problem_ids": torch.tensor([[1, 2, 3, 4, 5], [3, 2, 1, 0, 0]]),
        "skill_ids": torch.tensor([[1, 2, 1, 2, 3], [2, 1, 2, 0, 0]]),
        "answers": torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 0]]),
        "student_ids": torch.tensor([1, 2]),
        "valid_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        "prediction_mask": torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 0, 0]], dtype=torch.bool),
    }


class DiffuQKTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = DiffuQKT(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4)

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch, generator=torch.Generator().manual_seed(19))

    def test_current_and_future_answers_do_not_change_prediction(self):
        batch = example_batch()
        before = self.predict(batch)["probabilities"]
        changed = {k: v.clone() for k, v in batch.items()}
        changed["answers"][:, 2:] = 1 - changed["answers"][:, 2:]
        after = self.predict(changed)["probabilities"]
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=0)

    def test_future_question_does_not_change_past_predictions(self):
        batch = example_batch()
        before = self.predict(batch)["probabilities"]
        changed = {k: v.clone() for k, v in batch.items()}
        changed["problem_ids"][0, 3:] = torch.tensor([7, 8])
        changed["skill_ids"][0, 3:] = 4
        after = self.predict(changed)["probabilities"]
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=0)

    def test_padding_values_have_no_effect(self):
        batch = example_batch()
        before = self.predict(batch)
        changed = {k: v.clone() for k, v in batch.items()}
        changed["problem_ids"][1, 3:] = 8
        changed["skill_ids"][1, 3:] = 4
        changed["answers"][1, 3:] = 1
        after = self.predict(changed)
        torch.testing.assert_close(before["probabilities"][batch["valid_mask"]],
                                   after["probabilities"][batch["valid_mask"]])
        self.assertTrue(torch.isfinite(after["states"]).all())
        self.assertTrue((after["states"][~batch["valid_mask"]] == 0).all())

    def test_all_active_components_receive_finite_gradients(self):
        self.model.train()
        result = self.model(example_batch())
        result["loss"].backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        torch.testing.assert_close(result["loss"], result["kt_loss"] + .001 * result["contrastive_loss"]
                                   + .01 * result["diffusion_loss"])

    def test_eval_repeatability_and_checkpoint_buffers(self):
        batch = example_batch()
        before = self.predict(batch)["probabilities"]
        self.model.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(before, self.predict(batch)["probabilities"], atol=0, rtol=0)
        self.assertIn("diffusion.posterior_variance", self.model.state_dict())

    def test_analytic_forward_and_final_reverse_step(self):
        diffusion = ConditionalQuestionDiffusion(8, steps=4)
        original, noise = torch.ones(1, 2, 8), torch.full((1, 2, 8), 2.)
        time = torch.tensor([[0, 3]])
        expected = diffusion.alpha_bar[time].sqrt().unsqueeze(-1) * original
        expected += (1-diffusion.alpha_bar[time]).sqrt().unsqueeze(-1) * noise
        torch.testing.assert_close(diffusion.q_sample(original, time, noise), expected)
        torch.testing.assert_close(diffusion.mean_coef_x0[0], torch.tensor(1.))
        self.assertEqual(float(diffusion.mean_coef_xt[0]), 0.)
        self.assertEqual(float(diffusion.posterior_variance[0]), 0.)

    def test_original_question_representation_is_scalar_and_fully_trainable(self):
        batch = example_batch()
        components = self.model.question_components(batch)
        mask = batch["valid_mask"]
        skills, problems = batch["skill_ids"], batch["problem_ids"]
        expected = (self.model.concept(skills)
                    + self.model.difficulty(problems).sigmoid() * self.model.variation(skills))
        torch.testing.assert_close(components["original"][mask], expected[mask])
        self.assertEqual(components["conditioning"].shape, (2, 5, 1))
        self.assertEqual(self.model.diffusion.denoiser[0].in_features, 3 * 8 + 1)
        self.assertTrue((components["original"][~mask] == 0).all())
        self.model(batch)["loss"].backward()
        for embedding in [self.model.concept, self.model.difficulty, self.model.variation]:
            self.assertTrue(embedding.weight.requires_grad)
            self.assertGreater(float(embedding.weight.grad[1:].abs().sum()), 0)
            self.assertTrue((embedding.weight.grad[0] == 0).all())

    def test_contrastive_matches_paper_and_excludes_same_student(self):
        enhanced = torch.tensor([[[1., 0.]], [[0., 1.]]], requires_grad=True)
        original = torch.tensor([[[1., 0.]], [[0., 1.]]])
        valid = torch.ones(2, 1, dtype=torch.bool)
        loss = stable_state_contrastive(enhanced, original, valid, torch.tensor([1, 2]), 1.)
        torch.testing.assert_close(loss, torch.log1p(torch.exp(torch.tensor(-1.))))
        no_negative = stable_state_contrastive(enhanced, original, valid, torch.tensor([1, 1]), 1.)
        torch.testing.assert_close(no_negative, torch.tensor(0.))
        padded = stable_state_contrastive(enhanced, original, torch.tensor([[True], [False]]),
                                         torch.tensor([1, 2]), 1.)
        torch.testing.assert_close(padded, torch.tensor(0.))


if __name__ == "__main__":
    unittest.main()
