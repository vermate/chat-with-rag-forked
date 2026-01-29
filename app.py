import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

load_dotenv()


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    # split doc into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    # initialize the free Hugging Face embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embedding_model)
    return vector_store


def get_context_retriever_chain(vector_store):
    """This function:
    1. uses text2text-generation task on google/flan-t5-base model via hugging face interface using pipelne method which returns a hugging face model
    2. Wraps the hugging face model into langchain compatible model
    3. retrive embedding as vector object
    4. create a prompt to generate context query based n past chat
    5. uses create_history_aware_retriever to chain the prompt to retriver and llm to generate relevant documents
    """
    # initialize a hugging face model wrapper and then wrap it in leangchain's HuggingFacePipeline
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    retriever = vector_store.as_retriever()

    # prompt to generate a contextual query wrt chat history, this query willl be embeded automatically when fed into retriver
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    # create a create_history_aware_retriever
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)  # retriever chain
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    # Initialize Gemini 2.5 Flash (free tier with 1,500 RPD)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context, If you cannot answer the question based on the context, say 'I don't have enough information in the url content to answer the question':\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)  # llm chain
    return create_retrieval_chain(
        retriever_chain, stuff_documents_chain
    )  # chain to combine retriver and llm chain


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_input}
    )
    # print("llm response is ", response)
    return response["answer"]


# app config
st.set_page_config(page_title="Chat with RAG App")
st.title("Chat with RAG App")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
