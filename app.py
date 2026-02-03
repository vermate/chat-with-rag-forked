"""
Main Streamlit application for Chat with RAG.

This application allows users to:
1. Load content from any website URL
2. Ask questions about the content
3. Get validated, high-quality answers
4. View validation metrics for transparency
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from rag_chains import create_vectorstore_from_pdf, create_vectorstore_from_url
from response_handler import ResponseHandler
from ui_components import render_chat_history, render_sidebar
from validator import RAGValidator


def initialize_session_state(source_type: str, source=None):
    """
    Initialize Streamlit session state variables.

    Args:
        source_type: Type of source - "url" or "pdf"
        source: Website URL string or uploaded PDF file
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=config.INITIAL_GREETING),
        ]

    if "vector_store" not in st.session_state:
        with st.spinner(f"Loading content from {source_type}..."):
            if source_type == "url":
                st.session_state.vector_store = create_vectorstore_from_url(source)
            elif source_type == "pdf":
                st.session_state.vector_store = create_vectorstore_from_pdf(source)

    if "validator" not in st.session_state:
        st.session_state.validator = RAGValidator()

    if "response_handler" not in st.session_state:
        st.session_state.response_handler = ResponseHandler(
            st.session_state.vector_store, st.session_state.validator
        )

    if "validation_history" not in st.session_state:
        st.session_state.validation_history = []


def handle_user_input(user_query: str):
    """
    Process user input and generate validated response.

    Args:
        user_query: User's question
    """
    # Generate validated response
    with st.spinner("Generating answer..."):
        answer, validation_result = st.session_state.response_handler.generate_response(
            user_query, st.session_state.chat_history
        )

    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))

    # Store validation result
    st.session_state.validation_history.append(validation_result)


def main():
    """Main application entry point."""

    # Configure page
    st.set_page_config(page_title=config.APP_TITLE, page_icon="💬", layout="wide")
    st.title(config.APP_TITLE)

    # Render sidebar and get input source
    website_url, uploaded_file = render_sidebar()

    # Check if input is provided
    if not website_url and not uploaded_file:
        st.info("Please enter a website URL or upload a PDF file in the sidebar to get started")
        st.markdown("""
        ### How to use:
        1. Choose input method: URL or PDF Upload
        2. Enter a website URL or upload a PDF file
        3. Ask questions about the content
        4. Get AI-powered answers with quality validation
        5. View validation metrics for each response
        """)
        return

    # Determine source type and initialize
    if website_url:
        source_type = "url"
        source = website_url
        # Reset session state if URL changed
        if (
            "current_source" not in st.session_state
            or st.session_state.current_source != website_url
        ):
            st.session_state.clear()
            st.session_state.current_source = website_url
    else:
        source_type = "pdf"
        source = uploaded_file
        # Reset session state if PDF changed
        if (
            "current_source" not in st.session_state
            or st.session_state.current_source != uploaded_file.name
        ):
            st.session_state.clear()
            st.session_state.current_source = uploaded_file.name

    # Initialize session state
    initialize_session_state(source_type, source)

    # Handle user input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        handle_user_input(user_query)

    # Render chat history
    render_chat_history(st.session_state.chat_history, st.session_state.validation_history)


if __name__ == "__main__":
    main()
