"""Fixed-profile x0 diffusion and differentiable DDIM; legacy DDPM uses the original loop."""
import math

import torch
from torch import nn

from ..diffuqkt.diffusion import ConditionalQuestionDiffusion, timestep_embedding


def diffusion_schedule(steps, schedule="cosine", beta_start=1e-4, beta_end=.02):
    if schedule == "cosine":
        grid = torch.linspace(0., 1., steps+1, dtype=torch.float64)
        cumulative = torch.cos((grid+.008)/1.008*math.pi/2).square()
        cumulative = cumulative/cumulative[0]
        betas = (1-cumulative[1:]/cumulative[:-1]).clamp(1e-8, .999)
    elif schedule == "linear":
        if not 0 < beta_start <= beta_end < 1:
            raise ValueError("Invalid linear diffusion beta bounds.")
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    else:
        raise ValueError("diffusion.schedule must be cosine or linear.")
    alphas = 1-betas
    cumulative = alphas.cumprod(0)
    previous = torch.cat([torch.ones(1, dtype=torch.float64), cumulative[:-1]])
    posterior = betas*(1-previous)/(1-cumulative)
    return dict(betas=betas, alpha_bar=cumulative, sqrt_alpha_bar=cumulative.sqrt(),
                sqrt_one_minus_alpha_bar=(1-cumulative).sqrt(), posterior_variance=posterior,
                mean_coef_x0=betas*previous.sqrt()/(1-cumulative),
                mean_coef_xt=(1-previous)*alphas.sqrt()/(1-cumulative),
                fixed_large_variance=torch.cat([posterior[1:2], betas[1:]]))


class ProfileDiffusion(nn.Module):
    VERSION = "profile-ddim-x0-v1"
    q_sample = ConditionalQuestionDiffusion.q_sample

    def __init__(self, profile_dim, structure_dim, residual_dim, steps=100, schedule="cosine",
                 sampling_steps=8, ddim_eta=0., time_embedding_dim=64, hidden_dim=128,
                 beta_start=1e-4, beta_end=.02):
        super().__init__()
        self.profile_dim, self.steps = profile_dim, steps
        self.structure_dim, self.residual_dim = structure_dim, residual_dim
        self.sampling_steps, self.ddim_eta = sampling_steps, ddim_eta
        self.time_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_embedding_dim, time_embedding_dim), nn.SiLU(),
                                      nn.Linear(time_embedding_dim, time_embedding_dim))
        self.denoiser = nn.Sequential(nn.Linear(4*profile_dim+structure_dim+residual_dim+time_embedding_dim, hidden_dim),
                                      nn.GELU(), nn.Linear(hidden_dim, profile_dim))
        for name, value in diffusion_schedule(steps, schedule, beta_start, beta_end).items():
            self.register_buffer(name, value.float())
        grid = torch.linspace(steps-1, 0, sampling_steps, dtype=torch.float64).round().long()
        if grid.unique().numel() != sampling_steps:
            raise ValueError("DDIM time grid must contain no duplicate indices.")
        self.register_buffer("sampling_grid", grid)

    def predict_x0(self, noisy, step, structure, partial, counts, observed, residual_condition=None):
        time = self.time_mlp(timestep_embedding(step, self.time_dim))
        values = [noisy, time, structure, partial, counts.log1p(), observed.to(noisy.dtype)]
        if self.residual_dim:
            if residual_condition is None or residual_condition.shape != structure.shape[:-1]+(self.residual_dim,):
                raise ValueError("Missing or incorrectly sized profile residual condition.")
            values.append(residual_condition)
        elif residual_condition is not None:
            raise ValueError("Profile residual condition is disabled.")
        return self.denoiser(torch.cat(values, dim=-1))

    def sample(self, structure, partial, counts, observed, noise, residual_condition=None, step_noise=None):
        """The same differentiable sampler is used by training and evaluation."""
        state = noise
        for index, step in enumerate(self.sampling_grid.tolist()):
            times = torch.full(state.shape[:-1], step, device=state.device, dtype=torch.long)
            x0 = self.predict_x0(state, times, structure, partial, counts, observed, residual_condition)
            if index == self.sampling_steps-1:
                return x0
            following = int(self.sampling_grid[index+1])
            a_s, a_r = self.alpha_bar[step], self.alpha_bar[following]
            epsilon = (state-a_s.sqrt()*x0)/(1-a_s).sqrt()
            sigma = self.ddim_eta*(((1-a_r)/(1-a_s))*(1-a_s/a_r)).clamp_min(0).sqrt()
            state = a_r.sqrt()*x0+(1-a_r-sigma.square()).clamp_min(0).sqrt()*epsilon
            if self.ddim_eta:
                if step_noise is None:
                    raise ValueError("Stochastic DDIM requires keyed step noise.")
                state = state+sigma*step_noise(index)
        raise RuntimeError("Empty DDIM grid.")

    def denoising_loss(self, dense, dense_counts, dense_observed, structure, partial, counts, observed,
                       residual_condition=None, kappa=5.):
        # Both x0 and the source of noisy x_s are immutable reference evidence.
        target = dense.detach()
        steps = torch.randint(self.steps, target.shape[:-1], device=target.device)
        noisy = self.q_sample(target, steps, torch.randn_like(target))
        prediction = self.predict_x0(noisy, steps, structure, partial, counts, observed, residual_condition)
        reliability = dense_counts.detach()/(dense_counts.detach()+kappa)
        weights = reliability*dense_observed
        per_key = ((prediction-target).square()*weights).sum(-1)/weights.sum(-1).clamp_min(1e-8)
        supported = dense_observed.any(-1)
        return per_key[supported].mean() if supported.any() else prediction.sum()*0


class LegacySampleCall:
    """Call the baseline DDPM loop without its eval-only no_grad decorator."""
    def __init__(self, module, condition, noise_source):
        self.module, self.condition, self.noise_source = module, condition, noise_source

    def __getattr__(self, name):
        return getattr(self.module, name)

    def predict_x0(self, noisy, time, concept, difficulty):
        if self.condition is None:
            return self.module.predict_x0(noisy, time, concept, difficulty)
        return self.module.predict_x0(noisy, time, concept, difficulty, residual_condition=self.condition)

    def noise_like(self, reference, generator=None):
        return self.noise_source(reference)

    def sample(self, concept, difficulty, valid):
        return ConditionalQuestionDiffusion.sample.__wrapped__(self, concept, difficulty, valid)
