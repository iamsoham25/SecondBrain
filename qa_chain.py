"""
RAG question-answering pipeline.

This module orchestrates:
- structured context construction from retrieved chunks
- strict prompt design for context-only answers
- LLM response generation with source formatting
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_TEMPERATURE = 0.0
MAX_SUMMARY_CONTEXT_CHARS = 12000


def _get_hf_token() -> str:
    return (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or "").strip()


def _hf_inference_configured() -> bool:
    return bool(_get_hf_token())


def _looks_like_hf_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        ("429" in msg)
        or ("rate" in msg)
        or ("503" in msg)
        or ("overload" in msg)
        or ("timeout" in msg)
        or ("quota" in msg)
    )


def _create_hf_chat(repo_id: str, temperature: float) -> ChatHuggingFace:
    """Hugging Face Inference API via huggingface_hub (requires token for reliable use)."""
    token = _get_hf_token() or None
    do_sample = temperature > 0
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=do_sample,
    )
    return ChatHuggingFace(llm=endpoint)


def _split_sentences(text: str) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    # Lightweight sentence split (good enough for fallback mode).
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p and p.strip()]


def _heuristic_answer(
    query: str,
    retrieved_docs: List[Document],
    max_bullets: int = 6,
    max_chars: int = 900,
) -> str:
    """
    Offline fallback answer generator (no external LLM).

    Strategy:
    - score sentences by token overlap with the query
    - pick top distinct sentences
    - return point-wise bullets (extractive, grounded)
    """
    query_tokens = set(_tokenize_for_overlap(query))
    if not query_tokens:
        return "Not found in documents"

    ranked_docs = _select_summary_documents(documents=retrieved_docs, query=query, max_chunks=5)
    if not ranked_docs:
        return "Not found in documents"

    scored_sents: List[tuple[float, str]] = []
    for doc in ranked_docs:
        for sent in _split_sentences(doc.page_content or ""):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            sent_tokens = set(_tokenize_for_overlap(sent))
            if not sent_tokens:
                continue
            overlap = len(query_tokens.intersection(sent_tokens))
            if overlap == 0:
                continue
            # Normalize by query length; add a small preference for shorter sentences.
            score = (overlap / max(len(query_tokens), 1)) + (0.08 / max(len(sent_tokens), 1))
            scored_sents.append((score, sent))

    scored_sents.sort(key=lambda t: t[0], reverse=True)

    def _dedupe_key(s: str) -> str:
        return " ".join(re.findall(r"\b\w+\b", s.lower()))[:220]

    picked: List[str] = []
    seen_keys: set[str] = set()
    for _, sent in scored_sents:
        key = _dedupe_key(sent)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        picked.append(sent)
        if len(picked) >= max_bullets:
            break

    if not picked:
        # If no overlapping sentences, return a short excerpt from the top chunk.
        top = (ranked_docs[0].page_content or "").strip()
        if not top:
            return "Not found in documents"
        excerpt = " ".join(top.split())
        excerpt = (excerpt[:max_chars].rstrip() + "...") if len(excerpt) > max_chars else excerpt
        return f"- {excerpt}"

    # Build bullets and enforce overall character budget.
    bullets: List[str] = []
    remaining = max_chars
    for sent in picked:
        bullet = f"- {sent}"
        if len(bullet) > remaining and remaining > 50:
            bullet = bullet[: max(0, remaining - 3)].rstrip() + "..."
        if len(bullet) <= remaining:
            bullets.append(bullet)
            remaining -= len(bullet) + 1
        if remaining <= 50:
            break

    return "\n".join(bullets) if bullets else "Not found in documents"


def _heuristic_summary(documents: List[Document], max_bullets: int = 6) -> str:
    ranked = _select_summary_documents(documents=documents, query=None, max_chunks=10)
    bullets: List[str] = []
    for doc in ranked:
        for sent in _split_sentences(doc.page_content or ""):
            if len(sent) < 25:
                continue
            bullets.append(sent)
            if len(bullets) >= max_bullets:
                break
        if len(bullets) >= max_bullets:
            break
    if not bullets:
        return "Not found in documents"
    return "\n".join(f"- {b}" for b in bullets)


def _validate_inputs(query: str, retrieved_docs: List[Document]) -> None:
    """Validate query/doc inputs before building the RAG prompt."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if retrieved_docs is None:
        raise ValueError("retrieved_docs must not be None.")
    if not isinstance(retrieved_docs, list):
        raise TypeError("retrieved_docs must be a list of Document objects.")


def _safe_snippet(text: str, max_chars: int = 180) -> str:
    """Create a short one-line snippet suitable for source display."""
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def _format_chunk_for_context(index: int, document: Document) -> str:
    """Render one retrieved chunk with metadata for model context."""
    metadata = dict(document.metadata or {})
    file_name = metadata.get("file_name", "unknown")
    page_number = metadata.get("page_number")
    page_display = page_number if page_number is not None else "N/A"
    chunk_text = (document.page_content or "").strip()
    return (
        f"[Chunk {index}]\n"
        f"File: {file_name}\n"
        f"Page: {page_display}\n"
        f"Content:\n{chunk_text}"
    )


def build_context(retrieved_docs: List[Document], max_chunks: int = 5) -> str:
    """
    Build a structured context block from retrieved chunks.

    Chunks are clearly separated and retain metadata fields used by the prompt.
    """
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than 0.")

    selected_docs = retrieved_docs[:max_chunks]
    if not selected_docs:
        return ""

    chunk_blocks = [
        _format_chunk_for_context(index=i, document=doc)
        for i, doc in enumerate(selected_docs, start=1)
    ]
    return "\n\n--------------------\n\n".join(chunk_blocks)


def build_sources(retrieved_docs: List[Document], max_sources: int = 5) -> List[Dict[str, Any]]:
    """Build response source objects with file, page, and quoted snippet."""
    if max_sources <= 0:
        return []

    sources: List[Dict[str, Any]] = []
    for doc in retrieved_docs[:max_sources]:
        metadata = dict(doc.metadata or {})
        sources.append(
            {
                "file_name": metadata.get("file_name", "unknown"),
                "page_number": metadata.get("page_number"),
                "snippet": _safe_snippet(doc.page_content or ""),
            }
        )
    return sources


def _create_prompt() -> ChatPromptTemplate:
    """Create a strict anti-hallucination prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a retrieval-grounded assistant.\n"
                    "Use ONLY the provided context chunks to answer.\n"
                    "If the answer is not present in the context, respond exactly with:\n"
                    "Not found in documents\n"
                    "Do not use prior knowledge. Do not hallucinate."
                ),
            ),
            (
                "human",
                (
                    "User question:\n"
                    "{query}\n\n"
                    "Retrieved context:\n"
                    "{context}\n\n"
                    "Instructions:\n"
                    "1) Answer only from the context.\n"
                    "2) Keep the answer concise and factual.\n"
                    "3) If the context does not contain the answer, output exactly: Not found in documents"
                ),
            ),
        ]
    )


def _create_summary_prompt() -> ChatPromptTemplate:
    """Create a strict summarization prompt bound to provided context only."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a retrieval-grounded summarization assistant.\n"
                    "Summarize ONLY the provided context.\n"
                    "Do not add external facts.\n"
                    "If the context is empty or insufficient, respond exactly with:\n"
                    "Not found in documents"
                ),
            ),
            (
                "human",
                (
                    "Summarize the key points from these documents in clear bullet points.\n"
                    "Focus on important concepts, definitions, and conclusions.\n\n"
                    "Context:\n"
                    "{context}"
                ),
            ),
        ]
    )


def _tokenize_for_overlap(text: str) -> List[str]:
    """Simple tokenizer used for lightweight relevance sorting."""
    return re.findall(r"\b\w+\b", text.lower())


def _select_summary_documents(
    documents: List[Document],
    query: Optional[str] = None,
    max_chunks: int = 10,
) -> List[Document]:
    """
    Select chunks for summarization.

    - If query is present: use overlap scoring and keep top relevant chunks.
    - If query is absent: keep first N chunks.
    """
    if max_chunks <= 0:
        return []
    if not documents:
        return []
    if not query or not query.strip():
        return documents[:max_chunks]

    query_tokens = set(_tokenize_for_overlap(query))
    if not query_tokens:
        return documents[:max_chunks]

    scored: List[tuple[float, Document]] = []
    for doc in documents:
        content_tokens = set(_tokenize_for_overlap(doc.page_content or ""))
        overlap = len(query_tokens.intersection(content_tokens))
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    ranked_docs = [doc for score, doc in scored if score > 0]
    if not ranked_docs:
        return documents[:max_chunks]
    return ranked_docs[:max_chunks]


def _truncate_documents_for_context(
    documents: List[Document],
    max_total_chars: int = MAX_SUMMARY_CONTEXT_CHARS,
) -> List[Document]:
    """Limit context payload size to reduce token overflow risk."""
    if max_total_chars <= 0:
        return []

    kept: List[Document] = []
    total_chars = 0
    for doc in documents:
        chunk_len = len(doc.page_content or "")
        if chunk_len == 0:
            continue
        if total_chars + chunk_len > max_total_chars:
            break
        kept.append(doc)
        total_chars += chunk_len
    return kept


def summarize_documents(
    documents: List[Document],
    model_name: str = DEFAULT_MODEL,
    query: Optional[str] = None,
    max_chunks: int = 10,
    max_total_chars: int = MAX_SUMMARY_CONTEXT_CHARS,
) -> str:
    """
    Summarize document chunks into clear bullet points.

    Args:
        documents: List of Documents with page_content + metadata.
        model_name: Hugging Face model repo id for Inference API (default: SmolLM2 360M Instruct).
        query: Optional focus query for selecting top relevant chunks.
        max_chunks: Maximum number of chunks to include.
        max_total_chars: Approximate context budget to avoid overflow.
    """
    if documents is None:
        raise ValueError("documents must not be None.")
    if not isinstance(documents, list):
        raise TypeError("documents must be a list of Document objects.")
    if not documents:
        return "Not found in documents"

    selected_docs = _select_summary_documents(
        documents=documents,
        query=query,
        max_chunks=max_chunks,
    )
    constrained_docs = _truncate_documents_for_context(
        documents=selected_docs,
        max_total_chars=max_total_chars,
    )
    if not constrained_docs:
        return "Not found in documents"

    context = build_context(retrieved_docs=constrained_docs, max_chunks=len(constrained_docs))
    if not context.strip():
        return "Not found in documents"

    if not _hf_inference_configured():
        return _heuristic_summary(constrained_docs)

    prompt = _create_summary_prompt()
    llm = _create_hf_chat(repo_id=model_name, temperature=0.0)

    try:
        messages = prompt.format_messages(context=context)
        llm_response = llm.invoke(messages)
    except Exception as exc:
        if _looks_like_hf_transient_error(exc):
            return _heuristic_summary(constrained_docs)
        raise RuntimeError(f"Failed to generate summary from LLM: {exc}") from exc

    summary_text = (llm_response.content or "").strip()
    if not summary_text:
        return "Not found in documents"
    return summary_text


def answer_query(
    query: str,
    retrieved_docs: List[Document],
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_chunks: int = 5,
) -> Dict[str, Any]:
    """
    Run the RAG QA pipeline and return answer + structured sources.

    Returns:
        {
            "answer": "<model answer or Not found in documents>",
            "sources": [
                {"file_name": "...", "page_number": 2, "snippet": "..."},
                ...
            ]
        }
    """
    _validate_inputs(query=query, retrieved_docs=retrieved_docs)

    if not retrieved_docs:
        return {"answer": "Not found in documents", "sources": []}

    context = build_context(retrieved_docs=retrieved_docs, max_chunks=max_chunks)
    if not context.strip():
        return {"answer": "Not found in documents", "sources": []}

    sources = build_sources(retrieved_docs=retrieved_docs, max_sources=max_chunks)

    if not _hf_inference_configured():
        return {"answer": _heuristic_answer(query, retrieved_docs), "sources": sources}

    prompt = _create_prompt()
    llm = _create_hf_chat(repo_id=model_name, temperature=temperature)

    try:
        messages = prompt.format_messages(query=query, context=context)
        llm_response = llm.invoke(messages)
    except Exception as exc:
        if _looks_like_hf_transient_error(exc):
            return {"answer": _heuristic_answer(query, retrieved_docs), "sources": sources}
        raise RuntimeError(f"Failed to generate answer from LLM: {exc}") from exc

    answer_text = (llm_response.content or "").strip()
    if not answer_text:
        answer_text = "Not found in documents"

    # Strong guardrail: normalize uncertain responses to required fallback.
    low_confidence_markers = [
        "i don't know",
        "cannot determine",
        "insufficient information",
        "not enough information",
    ]
    normalized_answer = answer_text.lower()
    if any(marker in normalized_answer for marker in low_confidence_markers):
        answer_text = "Not found in documents"

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    # Test snippet:
    # 1) Set HUGGINGFACEHUB_API_TOKEN for Inference API, or expect heuristic fallback.
    # 2) Run: python qa_chain.py
    sample_docs = [
        Document(
            page_content="SecondBrain uses FAISS for vector similarity search.",
            metadata={"file_name": "notes.pdf", "page_number": 2, "source": "data/notes.pdf"},
        ),
        Document(
            page_content="Chunk overlap is set to 100 tokens during ingestion.",
            metadata={"file_name": "design.docx", "page_number": 5, "source": "data/design.docx"},
        ),
    ]

    result = answer_query(
        query="What vector database does SecondBrain use?",
        retrieved_docs=sample_docs,
        model_name=DEFAULT_MODEL,
    )
    print("Answer:", result["answer"])
    print("\nSources:")
    for idx, src in enumerate(result["sources"], start=1):
        print(
            f'{idx}. {src["file_name"]} (Page {src["page_number"]}): '
            f'"{src["snippet"]}"'
        )
