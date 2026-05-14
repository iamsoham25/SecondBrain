"""
Retrieval logic module.

This file will contain semantic search behavior against the vector store
and return the most relevant chunks for a user question.
"""

from typing import List

from vector_store import create_vector_store


def retrieve_context(query: str, top_k: int = 5) -> List[str]:
    """Placeholder retriever that will query FAISS in later implementation."""
    store = create_vector_store()
    _ = store
    _ = query
    _ = top_k
    # TODO: Run vector similarity search and return top chunks.
    return []
