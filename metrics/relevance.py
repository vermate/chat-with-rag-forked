"""
Relevance metric - evaluates if answer addresses the question.
TODO: To be implemented
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RelevanceMetric:
    """Evaluates if the answer is relevant to the user's question."""

    def __init__(self):
        """Initialize the relevance metric evaluator."""
        pass

    def evaluate(self, answer: str, question: str, context_docs: List[Document]) -> float:
        """
        Check if the answer is relevant to the user's question.

        TODO: Implement relevance evaluation logic

        Args:
            answer: The generated answer from the RAG system
            question: The user's original question
            context_docs: List of retrieved documents (for context)

        Returns:
            Relevance score between 0 and 1 (currently returns None)
        """
        # TODO: Implement relevance metric
        return None
