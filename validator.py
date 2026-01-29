import logging
from typing import Dict, List

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGValidator:
    """Validator for RAG responses to ensure quality and reliability."""

    def __init__(self, faithfulness_threshold: float = None):
        """
        Initialize the RAG validator.

        Args:
            faithfulness_threshold: Minimum score for faithfulness to pass (0-1)
                                  If None, uses default from config
        """
        self.faithfulness_threshold = faithfulness_threshold or config.FAITHFULNESS_THRESHOLD
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.VALIDATION_TEMPERATURE,
        )

    def check_faithfulness(self, answer: str, context_docs: List[Document]) -> float:
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

    def validate_response(self, answer: str, question: str, context_docs: List[Document]) -> Dict:
        """
        Validate a RAG response across multiple dimensions.

        Args:
            answer: The generated answer
            question: The user's question
            context_docs: Retrieved context documents

        Returns:
            Dictionary with validation metrics and pass/fail status
        """
        # Calculate faithfulness
        faithfulness_score = self.check_faithfulness(answer, context_docs)

        # Determine if response passes validation
        passes_validation = faithfulness_score >= self.faithfulness_threshold

        # Prepare validation result
        validation_result = {
            "faithfulness": round(faithfulness_score, 2),
            "relevance": None,  # Placeholder for future implementation
            "consistency": None,  # Placeholder for future implementation
            "toxicity": None,  # Placeholder for future implementation
            "pass": passes_validation,
        }

        # Log the validation result
        logger.info(f"Validation result: {validation_result}")

        return validation_result

    def get_improvement_prompt(
        self,
        original_answer: str,
        question: str,
        context_docs: List[Document],
        validation_result: Dict,
    ) -> str:
        """
        Generate a prompt to help the LLM improve a low-quality answer.

        Args:
            original_answer: The original answer that failed validation
            question: The user's question
            context_docs: Retrieved context documents
            validation_result: The validation scores and failures

        Returns:
            Improved prompt for regeneration
        """
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        improvement_prompt = f"""The previous answer failed quality validation. Please regenerate a better answer.

Original Question: {question}

Context (retrieved from URL):
{context_text}

Previous Answer (which failed validation):
{original_answer}

Validation Scores:
- Faithfulness: {validation_result["faithfulness"]} (threshold: {self.faithfulness_threshold})

Issues to address:
- Ensure ALL claims are directly supported by the context
- Do NOT add information not present in the context
- Do NOT contradict any information in the context
- Stay strictly grounded in the provided context

Generate a new, improved answer that is completely faithful to the context. If you cannot answer based on the context, explicitly state that."""

        return improvement_prompt
