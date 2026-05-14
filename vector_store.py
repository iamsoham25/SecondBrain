"""
Vector store setup and indexing helpers.

This module will manage FAISS initialization, persistence,
and insertion of chunk embeddings.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from utils.embeddings import get_embedding_model

_VECTOR_STORE: Optional[FAISS] = None
RetrievalMode = Literal["hybrid", "semantic", "keyword"]


def _normalize_document_metadata(document: Document) -> Document:
    """
    Ensure required metadata keys exist before indexing.

    FAISS keeps metadata attached to each Document, so normalizing keys here
    guarantees consistent output during retrieval.
    """
    metadata = dict(document.metadata or {})
    metadata.setdefault("file_name", metadata.get("source", "unknown"))
    metadata.setdefault("page_number", metadata.get("page"))
    metadata.setdefault("source", metadata.get("file_name", "unknown"))
    return Document(page_content=document.page_content, metadata=metadata)


def create_vector_store(
    documents: List[Document],
    model_name: Optional[str] = None,
) -> FAISS:
    """
    Create a FAISS vector store from LangChain Documents.

    Args:
        documents: Chunk documents with metadata.
        model_name: Optional Sentence Transformers model id (Hugging Face embeddings).
    """
    global _VECTOR_STORE

    if not documents:
        raise ValueError("No documents provided to create_vector_store.")

    normalized_docs = [_normalize_document_metadata(doc) for doc in documents]
    embedding_model = get_embedding_model(model_name=model_name)

    # FAISS.from_documents embeds page_content and keeps metadata per chunk.
    _VECTOR_STORE = FAISS.from_documents(normalized_docs, embedding_model)
    return _VECTOR_STORE


def save_vector_store(path: str) -> None:
    """Save the active FAISS vector store to local disk."""
    if _VECTOR_STORE is None:
        raise ValueError("Vector store is not initialized. Create or load it first.")

    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    _VECTOR_STORE.save_local(str(target))


def load_vector_store(
    path: str,
    model_name: Optional[str] = None,
) -> FAISS:
    """Load a FAISS vector store from local disk."""
    global _VECTOR_STORE

    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Vector store path does not exist: {path}")

    embedding_model = get_embedding_model(model_name=model_name)
    _VECTOR_STORE = FAISS.load_local(
        folder_path=str(target),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    return _VECTOR_STORE


def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization for lightweight keyword matching."""
    return re.findall(r"\b\w+\b", text.lower())


def _document_key(document: Document) -> str:
    """Stable key used for deduping semantic and keyword hits."""
    metadata = dict(document.metadata or {})
    source = str(metadata.get("source") or metadata.get("file_name") or "")
    page = str(metadata.get("page_number") or metadata.get("page") or "")
    content_head = document.page_content[:200]
    return f"{source}|{page}|{content_head}"


def _keyword_score(query: str, document: Document) -> float:
    """
    Compute a simple lexical relevance score.

    Scoring strategy (kept intentionally simple):
    - token overlap ratio
    - exact phrase boost
    - file name token match boost
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    content = document.page_content or ""
    content_tokens = set(_tokenize(content))
    query_token_set = set(query_tokens)
    overlap_count = len(query_token_set.intersection(content_tokens))

    # Core overlap signal.
    overlap_ratio = overlap_count / max(len(query_token_set), 1)

    # Exact phrase signal helps "exact queries".
    phrase_boost = 0.25 if query.strip().lower() in content.lower() else 0.0

    # Metadata signal from file_name.
    metadata = dict(document.metadata or {})
    file_name = str(metadata.get("file_name") or "").lower()
    file_name_tokens = set(_tokenize(file_name))
    file_name_overlap = len(query_token_set.intersection(file_name_tokens))
    file_name_boost = min(0.2, file_name_overlap * 0.05)

    return overlap_ratio + phrase_boost + file_name_boost


def _keyword_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Run in-memory keyword ranking over indexed FAISS documents."""
    if _VECTOR_STORE is None:
        return []

    docstore = getattr(_VECTOR_STORE, "docstore", None)
    if docstore is None:
        return []

    all_docs_dict = getattr(docstore, "_dict", {})
    if not isinstance(all_docs_dict, dict):
        return []

    scored: List[Dict[str, Any]] = []
    for doc in all_docs_dict.values():
        if not isinstance(doc, Document):
            continue
        score = _keyword_score(query, doc)
        if score <= 0:
            continue
        scored.append({"document": doc, "keyword_score": score})

    scored.sort(key=lambda item: item["keyword_score"], reverse=True)
    return scored[:top_k]


def _semantic_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Run FAISS semantic retrieval with normalized relevance scores."""
    if _VECTOR_STORE is None:
        return []

    # Try relevance API first (0..1 scale), then fallback to distance scores.
    try:
        semantic_hits = _VECTOR_STORE.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            {"document": doc, "semantic_score": float(score)}
            for doc, score in semantic_hits
        ]
    except Exception:
        semantic_hits = _VECTOR_STORE.similarity_search_with_score(query, k=top_k)
        # Convert distance to positive score. Lower distance => higher score.
        return [
            {"document": doc, "semantic_score": 1.0 / (1.0 + float(distance))}
            for doc, distance in semantic_hits
        ]


def _hybrid_search(query: str, top_k: int) -> List[Document]:
    """
    Combine semantic and keyword retrieval, then rerank.

    Weighted score:
    - semantic score: 70%
    - keyword score: 30%
    """
    semantic_hits = _semantic_search(query, top_k=top_k * 2)
    keyword_hits = _keyword_search(query, top_k=top_k * 2)

    merged: Dict[str, Dict[str, Any]] = {}

    # Merge semantic hits.
    for hit in semantic_hits:
        doc = hit["document"]
        key = _document_key(doc)
        merged.setdefault(
            key,
            {
                "document": doc,
                "semantic_score": 0.0,
                "keyword_score": 0.0,
            },
        )
        merged[key]["semantic_score"] = max(
            merged[key]["semantic_score"], float(hit.get("semantic_score", 0.0))
        )

    # Merge keyword hits.
    for hit in keyword_hits:
        doc = hit["document"]
        key = _document_key(doc)
        merged.setdefault(
            key,
            {
                "document": doc,
                "semantic_score": 0.0,
                "keyword_score": 0.0,
            },
        )
        merged[key]["keyword_score"] = max(
            merged[key]["keyword_score"], float(hit.get("keyword_score", 0.0))
        )

    ranked = sorted(
        merged.values(),
        key=lambda item: (0.7 * item["semantic_score"]) + (0.3 * item["keyword_score"]),
        reverse=True,
    )
    return [item["document"] for item in ranked[:top_k]]


def similarity_search(
    query: str,
    top_k: int = 5,
    mode: RetrievalMode = "hybrid",
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
    - semantic FAISS search
    - keyword search
    - merged and reranked results

    Returns chunk text with key metadata fields.
    """
    if _VECTOR_STORE is None:
        raise ValueError("Vector store is not initialized. Load or create it first.")
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    normalized_mode = mode.strip().lower()
    if normalized_mode == "hybrid":
        matches = _hybrid_search(query, top_k=top_k)
    elif normalized_mode == "semantic":
        matches = [hit["document"] for hit in _semantic_search(query, top_k=top_k)]
    elif normalized_mode == "keyword":
        matches = [hit["document"] for hit in _keyword_search(query, top_k=top_k)]
    else:
        raise ValueError("Invalid retrieval mode. Use 'hybrid', 'semantic', or 'keyword'.")
    results: List[Dict[str, Any]] = []
    for match in matches:
        metadata = dict(match.metadata or {})
        results.append(
            {
                "chunk_text": match.page_content,
                "metadata": {
                    "file_name": metadata.get("file_name"),
                    "page_number": metadata.get("page_number"),
                    "source": metadata.get("source"),
                },
            }
        )
    return results


def index_chunks(
    documents: List[Document],
    model_name: Optional[str] = None,
) -> FAISS:
    """
    Backward-compatible indexing entry point.

    This delegates to create_vector_store and returns the initialized store.
    """
    return create_vector_store(
        documents=documents,
        model_name=model_name,
    )


if __name__ == "__main__":
    # Example usage for quick local testing.
    sample_docs = [
        Document(
            page_content="SecondBrain stores and retrieves personal knowledge.",
            metadata={
                "file_name": "notes.txt",
                "page_number": 1,
                "source": "data/notes.txt",
            },
        ),
        Document(
            page_content="FAISS enables fast vector similarity search over chunks.",
            metadata={
                "file_name": "architecture.md",
                "page_number": None,
                "source": "data/architecture.md",
            },
        ),
    ]

    create_vector_store(sample_docs)
    save_vector_store("vector_index")
    load_vector_store("vector_index")
    demo_results = similarity_search("similarity search over FAISS", top_k=2)
    print(demo_results)
