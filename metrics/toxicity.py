"""
Toxicity metric - evaluates if answer contains harmful or biased content.
TODO: To be implemented
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ToxicityMetric:
    """Evaluates if the answer contains toxic, harmful, or biased content."""

    def __init__(self):
        """Initialize the toxicity metric evaluator."""
        pass

    def evaluate(self, answer: str, context_docs: List[Document]) -> float:
        """
        Check if the answer contains toxic or harmful content.

        TODO: Implement toxicity evaluation logic

        Args:
            answer: The generated answer from the RAG system
            context_docs: List of retrieved documents (for context)

        Returns:
            Toxicity score between 0 and 1 (lower is better, currently returns None)
        """
        # TODO: Implement toxicity metric
        return None
