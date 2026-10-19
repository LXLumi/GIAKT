"""Separate answer-independent queries from posterior-conditioned historical values."""
from ..diffuqkt.attention import KnowledgeRetriever


class EvidenceRetriever(KnowledgeRetriever):
    def encode_questions(self, questions, valid):
        return self.question_encoder(questions, questions, questions, valid)

    def forward_with_interactions(self, question_prior, interaction_input, valid, question_context=None):
        questions = self.encode_questions(question_prior, valid) if question_context is None else question_context
        interactions = self.interaction_encoder(interaction_input, interaction_input, interaction_input, valid)
        return self.retrieval(questions, questions, interactions, valid, strict=True)
