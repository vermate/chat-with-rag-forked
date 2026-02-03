"""
Faithfulness metric - evaluates if answer is supported by context.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

import config

logger = logging.getLogger(__name__)


class FaithfulnessMetric:
    """Evaluates if the answer is faithful to (supported by) the retrieved context."""

    def __init__(self):
        """Initialize the faithfulness metric evaluator."""
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.VALIDATION_TEMPERATURE,
        )

    def evaluate(self, answer: str, context_docs: List[Document]) -> float:
        """
        Check if the answer is faithful to (supported by) the retrieved context.

        Args:
            answer: The generated answer from the RAG system
            context_docs: List of retrieved documents that were used to generate the answer

        Returns:
            Faithfulness score between 0 and 1
        """
        # Combine all context documents into a single text
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Create a prompt to evaluate faithfulness
        faithfulness_prompt = f"""You are an expert evaluator. Your task is to assess if an answer is faithful to (supported by) the given context.

Context:
{context_text}

Answer:
{answer}

Evaluate if the answer is factually supported by the context. Consider:
1. Are all claims in the answer backed by information in the context?
2. Does the answer introduce information not present in the context?
3. Does the answer contradict any information in the context?

Rate the faithfulness on a scale of 0.0 to 1.0:
- 1.0: Completely faithful, all information is directly supported by context
- 0.7-0.9: Mostly faithful, minor extrapolations but no contradictions
- 0.4-0.6: Partially faithful, some unsupported claims
- 0.0-0.3: Not faithful, major contradictions or unsupported information

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        try:
            response = self.llm.invoke(faithfulness_prompt)
            score_text = response.content.strip()

            # Extract the numeric score
            score = float(score_text)

            # Ensure score is within valid range
            score = max(0.0, min(1.0, score))

            logger.info(f"Faithfulness score: {score:.2f}")
            return score

        except Exception as e:
            logger.error(f"Error calculating faithfulness score: {e}")
            # Return a default low score if evaluation fails
            return 0.5
