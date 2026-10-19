"""An independent residual condition using the baseline diffusion equations."""
import torch
from torch import nn

from ..diffuqkt.diffusion import ConditionalQuestionDiffusion, timestep_embedding


class _ResidualConditionCall:
    """Per-call adapter: inherited loops pass the same R at every denoising step."""

    def __init__(self, module, residual_condition):
        self.module = module
        self.residual_condition = residual_condition

    def __getattr__(self, name):
        return getattr(self.module, name)

    def predict_x0(self, noisy, time, concept, question_scale):
        return self.module.predict_x0(noisy, time, concept, question_scale,
                                      residual_condition=self.residual_condition)


class ResidualConditionDiffusion(ConditionalQuestionDiffusion):
    """Append d residual columns, preserving all existing weights and schedule buffers."""

    def __init__(self, original):
        super().__init__(original.dim, original.steps, float(original.betas[0]),
                         float(original.betas[-1]), original.variance)
        self.to(device=original.betas.device, dtype=original.betas.dtype)
        self.load_state_dict(original.state_dict(), strict=True)
        old = self.denoiser[0]
        expanded = nn.Linear(4*self.dim+1, 2*self.dim, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :old.in_features].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.denoiser[0] = expanded

    @staticmethod
    def _validate_condition(residual_condition, reference):
        if not isinstance(residual_condition, torch.Tensor) or residual_condition.shape != reference.shape:
            raise ValueError("Residual condition must match [batch, time, embedding_dim].")
        if residual_condition.device != reference.device or residual_condition.dtype != reference.dtype:
            raise ValueError("Residual condition must match the input device and dtype.")
        return residual_condition

    def predict_x0(self, noisy, time, concept, question_scale, residual_condition=None):
        residual_condition = self._validate_condition(residual_condition, noisy)
        if question_scale.shape != noisy.shape[:-1]+(1,):
            raise ValueError("Question difficulty must have one scalar per event.")
        time_rep = self.time_mlp(timestep_embedding(time, self.dim))
        base_input = torch.cat([concept, noisy, question_scale, time_rep], dim=-1)
        return self.denoiser(torch.cat([base_input, residual_condition], dim=-1))

    def training_representation(self, original, concept, question_scale, valid, residual_condition=None):
        condition = self._validate_condition(residual_condition, original) * valid.unsqueeze(-1)
        call = _ResidualConditionCall(self, condition)
        return ConditionalQuestionDiffusion.training_representation(call, original, concept, question_scale, valid)

    @torch.no_grad()
    def sample(self, concept, question_scale, valid, generator=None, residual_condition=None):
        condition = self._validate_condition(residual_condition, concept) * valid.unsqueeze(-1)
        call = _ResidualConditionCall(self, condition)
        return ConditionalQuestionDiffusion.sample(call, concept, question_scale, valid, generator)


class _ConcatenatedTargetCall(_ResidualConditionCall):
    """Keep inherited diffusion loops, using 2d noise even when the concept has d entries."""

    def noise_like(self, tensor, generator=None):
        reference = tensor.new_empty(tensor.shape[:-1] + (self.module.output_dim,))
        return self.module.noise_like(reference, generator)


class ConcatenatedTargetDiffusion(ConditionalQuestionDiffusion):
    """Generate a full [Q_sem, R] target while keeping concept/time/condition width d.

    Only the denoiser input/output widths change. The hidden width stays 2d,
    and all schedules, x0 loss and reverse-sampling equations are inherited.
    """

    def __init__(self, original, use_residual_condition):
        super().__init__(original.dim, original.steps, float(original.betas[0]),
                         float(original.betas[-1]), original.variance)
        self.to(device=original.betas.device, dtype=original.betas.dtype)
        self.load_state_dict(original.state_dict(), strict=True)
        self.output_dim = 2 * self.dim
        self.use_residual_condition = use_residual_condition
        old_input, old_output = self.denoiser[0], self.denoiser[2]
        width = (5 if use_residual_condition else 4) * self.dim + 1
        expanded_input = nn.Linear(width, 2 * self.dim, device=old_input.weight.device, dtype=old_input.weight.dtype)
        expanded_output = nn.Linear(2 * self.dim, self.output_dim,
                                    device=old_output.weight.device, dtype=old_output.weight.dtype)
        with torch.no_grad():
            expanded_input.weight.zero_()
            # [C(d), noisy_Q(d), noisy_R(d), D(1), time(d), optional R(d)]
            expanded_input.weight[:, :2*self.dim].copy_(old_input.weight[:, :2*self.dim])
            expanded_input.weight[:, 3*self.dim:4*self.dim+1].copy_(old_input.weight[:, 2*self.dim:])
            expanded_input.bias.copy_(old_input.bias)
            expanded_output.weight.zero_()
            expanded_output.bias.zero_()
            expanded_output.weight[:self.dim].copy_(old_output.weight)
            expanded_output.bias[:self.dim].copy_(old_output.bias)
        self.denoiser[0], self.denoiser[2] = expanded_input, expanded_output

    def _condition(self, residual_condition, concept, valid):
        if self.use_residual_condition:
            return ResidualConditionDiffusion._validate_condition(residual_condition, concept) * valid.unsqueeze(-1)
        if residual_condition is not None:
            raise ValueError("Residual condition is disabled for the target-only route.")
        return None

    def predict_x0(self, noisy, time, concept, question_scale, residual_condition=None):
        if noisy.shape != concept.shape[:-1] + (self.output_dim,) or concept.shape[-1] != self.dim:
            raise ValueError("Concatenated target must have width 2d and concept must have width d.")
        if question_scale.shape != noisy.shape[:-1] + (1,):
            raise ValueError("Question difficulty must have one scalar per event.")
        time_rep = self.time_mlp(timestep_embedding(time, self.dim))
        values = [concept, noisy, question_scale, time_rep]
        if self.use_residual_condition:
            values.append(ResidualConditionDiffusion._validate_condition(residual_condition, concept))
        elif residual_condition is not None:
            raise ValueError("Residual condition is disabled for the target-only route.")
        return self.denoiser(torch.cat(values, dim=-1))

    def training_representation(self, original, concept, question_scale, valid, residual_condition=None):
        condition = self._condition(residual_condition, concept, valid)
        call = _ConcatenatedTargetCall(self, condition)
        return ConditionalQuestionDiffusion.training_representation(call, original, concept, question_scale, valid)

    @torch.no_grad()
    def sample(self, concept, question_scale, valid, generator=None, residual_condition=None):
        condition = self._condition(residual_condition, concept, valid)
        call = _ConcatenatedTargetCall(self, condition)
        return ConditionalQuestionDiffusion.sample(call, concept, question_scale, valid, generator)
