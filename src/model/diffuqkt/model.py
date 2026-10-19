"""DiffuQKT: Rasch question embeddings, conditional diffusion and contrastive KT."""
from .base import QuestionKTBase, stable_state_contrastive
from .diffusion import ConditionalQuestionDiffusion


class DiffuQKT(QuestionKTBase):
    def __init__(self, num_problems, num_skills, embedding_dim=128, num_heads=2, dropout=0.4,
                 diffusion_steps=100, beta_start=1e-4, beta_end=0.02, reverse_variance="posterior",
                 contrastive_negatives="enhanced", contrastive_weight=0.001,
                 diffusion_weight=0.01, temperature=0.3):
        if diffusion_weight < 0:
            raise ValueError("Loss weights must be nonnegative and temperature positive.")
        super().__init__(num_problems, num_skills, embedding_dim, dropout,
                         contrastive_negatives, contrastive_weight, temperature)
        self.diffusion = ConditionalQuestionDiffusion(embedding_dim, diffusion_steps, beta_start,
                                                      beta_end, reverse_variance)
        self.initialize_downstream(embedding_dim, num_heads, dropout)
        self.diffusion_weight = diffusion_weight
        self.representation_loss_weights = {"diffusion_loss": diffusion_weight}

    def enhance_questions(self, components, batch, generator=None):
        concept, condition, original = [components[key] for key in
                                        ["concept", "conditioning", "original"]]
        valid = batch["valid_mask"]
        if self.training:
            enhanced, diffusion_loss = self.diffusion.training_representation(
                original, concept, condition, valid)
        else:
            enhanced = self.diffusion.sample(concept, condition, valid, generator)
            diffusion_loss = None
        return enhanced, {"diffusion_loss": diffusion_loss}
