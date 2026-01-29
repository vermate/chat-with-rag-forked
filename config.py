"""
Configuration settings for the RAG application.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-2.5-flash"
RETRIEVER_LLM_MODEL = "google/flan-t5-base"

# Temperature Settings
DEFAULT_TEMPERATURE = 0.7
VALIDATION_TEMPERATURE = 0.0
REGENERATION_TEMPERATURE = 0.3

# Validation Thresholds
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.7
CONSISTENCY_THRESHOLD = 0.7
TOXICITY_THRESHOLD = 0.3  # Lower is better for toxicity

# Retry Configuration
MAX_REGENERATION_RETRIES = 1

# UI Configuration
APP_TITLE = "Chat with RAG App"
INITIAL_GREETING = "Hello, I am a bot. How can I help you?"
