"""
Response handler for orchestrating RAG responses with validation.
"""

import logging
from typing import Dict, Tuple

import config
from rag_chains import create_context_retriever_chain, create_conversational_rag_chain
from validator import RAGValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResponseHandler:
    """
    Handles response generation, validation, and regeneration.
    """

    def __init__(self, vector_store, validator: RAGValidator):
        """
        Initialize the response handler.

        Args:
            vector_store: Chroma vector store instance
            validator: RAGValidator instance
        """
        self.vector_store = vector_store
        self.validator = validator

    def generate_response(self, user_input: str, chat_history: list) -> Tuple[str, Dict]:
        """
        Generate a validated response to user input.

        This method:
        1. Generates an initial response using RAG
        2. Validates the response for faithfulness
        3. Regenerates if validation fails (up to max retries)
        4. Returns the final answer and validation metrics

        Args:
            user_input: User's question
            chat_history: List of previous chat messages

        Returns:
            Tuple of (answer, validation_result)
        """
        # Create RAG chains
        retriever_chain = create_context_retriever_chain(self.vector_store)
        rag_chain = create_conversational_rag_chain(retriever_chain)

        # Generate initial response
        logger.info(f"Generating response for: {user_input}")
        response = rag_chain.invoke({"chat_history": chat_history, "input": user_input})

        answer = response["answer"]
        context_docs = response.get("context", [])

        # Validate the response
        validation_result = self.validator.validate_response(answer, user_input, context_docs)

        logger.info(f"Initial validation: {validation_result}")

        # Retry if validation fails
        retry_count = 0
        while not validation_result["pass"] and retry_count < config.MAX_REGENERATION_RETRIES:
            retry_count += 1
            logger.warning(
                f"Answer failed validation. Retry {retry_count}/{config.MAX_REGENERATION_RETRIES}"
            )
            logger.warning(f"Failed answer: {answer}")

            # Regenerate with improved prompt
            answer = self._regenerate_answer(user_input, answer, context_docs, validation_result)

            # Validate regenerated answer
            validation_result = self.validator.validate_response(answer, user_input, context_docs)

            logger.info(f"Retry {retry_count} validation: {validation_result}")

        return answer, validation_result

    def _regenerate_answer(
        self, question: str, original_answer: str, context_docs: list, validation_result: Dict
    ) -> str:
        """
        Regenerate an answer that failed validation.

        Args:
            question: Original user question
            original_answer: Answer that failed validation
            context_docs: Retrieved context documents
            validation_result: Validation metrics from failed attempt

        Returns:
            Regenerated answer
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Get improvement prompt from validator
        improvement_prompt = self.validator.get_improvement_prompt(
            original_answer, question, context_docs, validation_result
        )

        # Initialize LLM with lower temperature for focused regeneration
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.REGENERATION_TEMPERATURE,
        )

        # Generate improved answer
        regenerated_response = llm.invoke(improvement_prompt)
        return regenerated_response.content
