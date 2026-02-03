"""
PDF loader for handling PDF file uploads and text extraction.
"""

import logging
import tempfile
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_from_upload(uploaded_file) -> List[Document]:
    """
    Load and process a PDF file from Streamlit file upload.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        List of Document objects with text chunks
    """
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load PDF using PyPDFLoader
        logger.info(f"Loading PDF from: {tmp_file_path}")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        logger.info(f"Successfully loaded {len(documents)} pages from PDF")
        return documents

    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise Exception(f"Failed to load PDF file: {str(e)}")


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(document_chunks)} chunks")
    return document_chunks
