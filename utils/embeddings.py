"""
Embedding utilities.

Purpose:
- initialize embedding model(s)
- transform text chunks and queries into vector representations
"""

import os
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Build a Hugging Face Sentence Transformers embedding model (runs locally).

    Default: sentence-transformers/all-MiniLM-L6-v2
    """
    resolved = (model_name or os.getenv("HF_EMBEDDING_MODEL") or DEFAULT_HF_EMBED_MODEL).strip()
    return HuggingFaceEmbeddings(model_name=resolved)


def embed_texts(
    texts: List[str],
    model_name: Optional[str] = None,
) -> List[List[float]]:
    """Convert multiple texts into vector embeddings."""
    model = get_embedding_model(model_name=model_name)
    if not texts:
        return []
    return model.embed_documents(texts)


def embed_query(
    query: str,
    model_name: Optional[str] = None,
) -> List[float]:
    """Convert a single query string into a vector embedding."""
    if not query.strip():
        raise ValueError("Query must not be empty.")
    model = get_embedding_model(model_name=model_name)
    return model.embed_query(query)
