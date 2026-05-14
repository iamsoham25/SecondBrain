"""
Chunking utilities.

Purpose:
- split long documents into smaller chunks
- prepare text segments suitable for embedding and retrieval
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 700,
    overlap: int = 100,
) -> List[Document]:
    """
    Split loaded documents into retrievable chunks.

    Args:
        documents: LangChain Document objects with text + metadata.
        chunk_size: Target chunk size. Intended range is 500-800.
        overlap: Number of overlapping units between chunks.
    """
    if not 500 <= chunk_size <= 800:
        raise ValueError("chunk_size must be between 500 and 800.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    # Token-aware splitter (tiktoken) keeps chunk sizes in token units.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)
