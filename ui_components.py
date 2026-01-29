"""
UI components for displaying chat and validation metrics.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


def render_validation_metrics(validation_result: dict):
    """
    Render validation metrics in a neat format.

    Args:
        validation_result: Dictionary containing validation scores
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        faithfulness = validation_result.get("faithfulness")
        if faithfulness is not None:
            color = "🟢" if faithfulness >= 0.7 else "🔴"
            st.metric("Faithfulness", f"{faithfulness:.2f}", delta=color)
        else:
            st.metric("Faithfulness", "N/A")

    with col2:
        relevance = validation_result.get("relevance")
        if relevance is not None:
            color = "🟢" if relevance >= 0.7 else "🔴"
            st.metric("Relevance", f"{relevance:.2f}", delta=color)
        else:
            st.metric("Relevance", "N/A")

    with col3:
        consistency = validation_result.get("consistency")
        if consistency is not None:
            color = "🟢" if consistency >= 0.7 else "🔴"
            st.metric("Consistency", f"{consistency:.2f}", delta=color)
        else:
            st.metric("Consistency", "N/A")

    with col4:
        toxicity = validation_result.get("toxicity")
        if toxicity is not None:
            # For toxicity, lower is better
            color = "🟢" if toxicity <= 0.3 else "🔴"
            st.metric("Toxicity", f"{toxicity:.2f}", delta=color)
        else:
            st.metric("Toxicity", "N/A")

    with col5:
        passed = validation_result.get("pass", False)
        status = "✅ Pass" if passed else "❌ Fail"
        st.markdown(f"**Status:** {status}")


def render_chat_history(chat_history: list, validation_history: list):
    """
    Render the chat history with validation metrics.

    Args:
        chat_history: List of chat messages
        validation_history: List of validation results corresponding to AI messages
    """
    ai_message_index = 0

    for message in chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

                # Show validation metrics (skip initial greeting)
                if ai_message_index > 0:
                    validation_index = ai_message_index - 1
                    if validation_index < len(validation_history):
                        validation = validation_history[validation_index]

                        # Add some spacing
                        st.markdown("---")
                        st.markdown("**Validation Metrics:**")
                        render_validation_metrics(validation)

                ai_message_index += 1

        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


def render_sidebar() -> str:
    """
    Render the sidebar with settings.

    Returns:
        Website URL entered by user
    """
    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")

        with st.expander("ℹ️ About"):
            st.markdown("""
            This RAG application:
            - Loads content from any URL
            - Answers questions based on the content
            - Validates responses for quality
            - Shows validation metrics for transparency
            """)

        with st.expander("📊 Validation Metrics"):
            st.markdown("""
            - **Faithfulness**: Is the answer supported by the URL content?
            - **Relevance**: Does the answer match the question?
            - **Consistency**: Is the answer internally consistent?
            - **Toxicity**: Is the answer safe and unbiased?
            
            Threshold: ≥ 0.7 to pass (except toxicity: ≤ 0.3)
            """)

    return website_url
