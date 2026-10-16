"""Conditional x0 prediction (paper Eqs. 15-19), rather than noise prediction."""
import math
import torch
from torch import nn


def timestep_embedding(timesteps, dim):
    half = dim // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device) / half)
    argument = timesteps.unsqueeze(-1).float() * frequency
    result = torch.cat([argument.cos(), argument.sin()], dim=-1)
    if dim % 2:
        result = torch.cat([result, torch.zeros_like(result[..., :1])], dim=-1)
    return result


class ConditionalQuestionDiffusion(nn.Module):
    def __init__(self, dim, steps=100, beta_start=1e-4, beta_end=0.02, variance="posterior"):
        super().__init__()
        if steps < 2 or not 0 < beta_start <= beta_end < 1 or variance not in {"posterior", "fixed_large"}:
            raise ValueError("Invalid diffusion schedule or variance mode.")
        self.dim, self.steps, self.variance = dim, steps, variance
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.denoiser = nn.Sequential(nn.Linear(3 * dim + 1, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        # Compute in float64 for schedule accuracy, then register portable float32 buffers.
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        alphas = 1 - betas
        cumulative = alphas.cumprod(dim=0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), cumulative[:-1]])
        posterior = betas * (1 - previous) / (1 - cumulative)
        buffers = {
            "betas": betas, "alpha_bar": cumulative,
            "sqrt_alpha_bar": cumulative.sqrt(), "sqrt_one_minus_alpha_bar": (1-cumulative).sqrt(),
            "posterior_variance": posterior,
            "mean_coef_x0": betas * previous.sqrt() / (1-cumulative),
            "mean_coef_xt": (1-previous) * alphas.sqrt() / (1-cumulative),
            "fixed_large_variance": torch.cat([posterior[1:2], betas[1:]]),
        }
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor.float())

    @staticmethod
    def noise_like(tensor, generator=None):
        return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)

    def predict_x0(self, noisy, time, concept, question_scale):
        if question_scale.shape != noisy.shape[:-1] + (1,):
            raise ValueError("Question difficulty must have one scalar per event.")
        time_rep = self.time_mlp(timestep_embedding(time, self.dim))
        return self.denoiser(torch.cat([concept, noisy, question_scale, time_rep], dim=-1))

    def q_sample(self, original, time, noise):
        return self.sqrt_alpha_bar[time].unsqueeze(-1) * original + (
            self.sqrt_one_minus_alpha_bar[time].unsqueeze(-1) * noise)

    def training_representation(self, original, concept, question_scale, valid):
        time = torch.randint(self.steps, original.shape[:2], device=original.device)
        noisy = self.q_sample(original, time, self.noise_like(original))
        enhanced = self.predict_x0(noisy, time, concept, question_scale) * valid.unsqueeze(-1)
        mse = (enhanced[valid] - original[valid]).square().mean()
        return enhanced, mse

    @torch.no_grad()
    def sample(self, concept, question_scale, valid, generator=None):
        state = self.noise_like(concept, generator) * valid.unsqueeze(-1)
        variances = self.posterior_variance if self.variance == "posterior" else self.fixed_large_variance
        for step in range(self.steps - 1, -1, -1):
            time = torch.full(concept.shape[:2], step, dtype=torch.long, device=concept.device)
            original = self.predict_x0(state, time, concept, question_scale)
            state = self.mean_coef_x0[step] * original + self.mean_coef_xt[step] * state
            if step > 0:
                state = state + variances[step].sqrt() * self.noise_like(state, generator)
            state = state * valid.unsqueeze(-1)
        return state
