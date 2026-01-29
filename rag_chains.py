"""
RAG chain components for context retrieval and response generation.
"""

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

import config


def create_vectorstore_from_url(url: str):
    """
    Load content from URL and create a vector store.

    Args:
        url: Website URL to load and index

    Returns:
        Chroma vector store with embedded document chunks
    """
    # Load document from URL
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # Create and return vector store
    vector_store = Chroma.from_documents(document_chunks, embedding_model)
    return vector_store


def create_context_retriever_chain(vector_store):
    """
    Create a history-aware retriever chain.

    This chain generates search queries based on chat history and
    retrieves relevant documents from the vector store.

    Args:
        vector_store: Chroma vector store instance

    Returns:
        History-aware retriever chain
    """
    # Initialize LLM for query generation
    llm_pipeline = pipeline("text2text-generation", model=config.RETRIEVER_LLM_MODEL)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Get retriever from vector store
    retriever = vector_store.as_retriever()

    # Create prompt for contextual query generation
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up "
                "in order to get information relevant to the conversation",
            ),
        ]
    )

    # Create and return history-aware retriever
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def create_conversational_rag_chain(retriever_chain):
    """
    Create a conversational RAG chain.

    This chain takes the retrieved context and generates answers
    using the LLM while maintaining conversation history.

    Args:
        retriever_chain: History-aware retriever chain

    Returns:
        Complete RAG chain for conversational question answering
    """
    # Initialize main LLM for response generation
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.DEFAULT_TEMPERATURE,
    )

    # Create prompt for answer generation
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context. \n\nContext:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    # Create document chain
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    # Create and return complete retrieval chain
    rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    return rag_chain
