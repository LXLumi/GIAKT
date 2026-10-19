"""LinearQ: question-specific linear residuals replace conditional diffusion."""
import torch
from torch import nn

from ..diffuqkt.base import QuestionKTBase


class LinearQ(QuestionKTBase):
    """Q=C+D*V; A=W_A*E_q, B=W_B*E_q; Q*=(C+A)+D*(V+B).

    E_q is a trainable question-ID embedding. The two bias-free linear heads
    generate distinct concept/variation offsets for questions sharing a skill.
    Parameters learn from KT, state contrastive learning and a uniform itemwise
    residual constraint over training questions. No diffusion network is used.
    """

    def __init__(self, num_problems, num_skills, embedding_dim=128, num_heads=2, dropout=0.4,
                 question_embedding_dim=None, contrastive_negatives="enhanced",
                 contrastive_weight=0.001, temperature=0.3, diffusion_weight=0.01,
                 train_item_ids=None):
        if diffusion_weight < 0:
            raise ValueError("Residual loss weight must be nonnegative.")
        super().__init__(num_problems, num_skills, embedding_dim, dropout,
                         contrastive_negatives, contrastive_weight, temperature)
        question_dim = embedding_dim if question_embedding_dim is None else question_embedding_dim
        if not isinstance(question_dim, int) or question_dim < 1:
            raise ValueError("question_embedding_dim must be a positive integer or null.")
        self.question_embedding = nn.Embedding(num_problems + 1, question_dim, padding_idx=0)
        self.linear_a = nn.Linear(question_dim, embedding_dim, bias=False)
        self.linear_b = nn.Linear(question_dim, embedding_dim, bias=False)
        # Small, nonzero residuals start near Q and allow both heads and E_q to learn immediately.
        nn.init.normal_(self.question_embedding.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.question_embedding.weight[0].zero_()
        self.initialize_downstream(embedding_dim, num_heads, dropout)
        # Reuse the existing auxiliary-loss slot so the shared model/trainer stays unchanged.
        self.diffusion_weight = diffusion_weight
        self.representation_loss_weights = {"diffusion_loss": diffusion_weight}
        # Epoch reporting averages this global penalty over steps, not interactions.
        self.loss_metric_weights = {"diffusion_loss": 1}
        # The constructor list is saved in checkpoint model_kwargs. Keep state_dict
        # compatible with historical LinearQ checkpoints that have no item buffer.
        self.register_buffer("train_item_ids", torch.empty(0, dtype=torch.long), persistent=False)
        if train_item_ids is not None:
            self.set_training_items(train_item_ids)

    def set_training_items(self, item_ids):
        """Initialize from training data only; never infer the set from a minibatch."""
        ids = torch.as_tensor(item_ids, dtype=torch.long, device=self.question_embedding.weight.device).flatten()
        if ((ids < 0) | (ids > self.question_embedding.num_embeddings - 1)).any():
            raise ValueError("Training item IDs must be valid question vocabulary indices.")
        ids = torch.unique(ids[ids.ne(0)], sorted=True)
        if ids.numel() == 0:
            raise ValueError("The training item set must contain at least one non-padding question.")
        self.train_item_ids = ids

    def prepare_training(self, dataset):
        """Model-specific preparation; return constructor arguments needed for recovery."""
        self.set_training_items(dataset.unique_problem_ids())
        return {"train_item_ids": self.train_item_ids.detach().cpu().tolist()}

    def item_regularization_loss(self):
        """sum_i ||A_i + D_i B_i||^2 / (embedding_dim * number_of_train_items)."""
        if self.train_item_ids.numel() == 0:
            raise RuntimeError("Initialize LinearQ with train_item_ids or set_training_items() before training.")
        question = self.question_embedding(self.train_item_ids)
        scale = self.difficulty(self.train_item_ids).sigmoid()
        residual = self.linear_a(question) + scale * self.linear_b(question)
        return residual.square().mean()

    def question_components(self, batch):
        components = super().question_components(batch)
        valid = batch["valid_mask"].unsqueeze(-1)
        question = self.question_embedding(batch["problem_ids"]) * valid
        offset_a, offset_b = self.linear_a(question) * valid, self.linear_b(question) * valid
        enhanced = ((components["concept"] + offset_a)
                    + components["scale"] * (components["variation"] + offset_b)) * valid
        components.update(question_embedding=question, A=offset_a, B=offset_b, enhanced=enhanced)
        return components

    def enhance_questions(self, components, batch, generator=None):
        enhanced = components["enhanced"]
        residual_loss = None
        if self.training:
            # Every training question contributes once at each parameter update,
            # independent of interaction frequency, current batch or skill assignment.
            residual_loss = self.item_regularization_loss()
        return enhanced, {"diffusion_loss": residual_loss}
