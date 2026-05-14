"""
Streamlit entry point for the SecondBrain application.

Features:
- upload and process documents into FAISS
- chat with a RAG pipeline
- show grounded sources and chat history
"""

from pathlib import Path
import os
import re
import tempfile
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

from qa_chain import answer_query, summarize_documents
from ingest import run_ingestion
from vector_store import create_vector_store, similarity_search

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)


def _get_hf_token() -> str:
    """Read Hugging Face Hub token used for Inference API chat models."""
    return (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or "").strip().strip(
        '"'
    ).strip("'")


def _init_session_state() -> None:
    """Initialize all session keys used by the app."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("docs_processed", False)
    st.session_state.setdefault("processed_files", [])
    st.session_state.setdefault("retrieval_mode", "Hybrid")
    st.session_state.setdefault("llm_model", "HuggingFaceTB/SmolLM2-360M-Instruct")
    st.session_state.setdefault("processed_chunks", [])
    st.session_state.setdefault("latest_summary", "")
    if st.session_state.llm_model in ("gpt-4o-mini", "gpt-4o"):
        st.session_state.llm_model = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _save_uploaded_files(uploaded_files: List[Any], target_dir: Path) -> List[Path]:
    """Persist uploaded Streamlit files to a temporary folder."""
    saved_paths: List[Path] = []
    for uploaded in uploaded_files:
        file_path = target_dir / uploaded.name
        file_path.write_bytes(uploaded.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def _dict_results_to_documents(results: List[Dict[str, Any]]) -> List[Document]:
    """Convert vector search dict results into Document objects for QA chain."""
    documents: List[Document] = []
    for item in results:
        metadata = dict(item.get("metadata") or {})
        documents.append(
            Document(
                page_content=str(item.get("chunk_text", "")),
                metadata=metadata,
            )
        )
    return documents


def _extract_query_keywords(query: str, max_keywords: int = 8) -> List[str]:
    """Extract lightweight keyword set from user query for snippet highlighting."""
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "who",
        "from",
        "about",
    }
    tokens = re.findall(r"\b\w+\b", (query or "").lower())
    keywords: List[str] = []
    for token in tokens:
        if len(token) <= 2 or token in stopwords or token.isdigit():
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _highlight_keywords(text: str, keywords: List[str]) -> str:
    """Bold query keywords in source snippets using markdown-safe replacement."""
    highlighted = text or ""
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(keyword)})\b", flags=re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted


def _render_sources(sources: List[Dict[str, Any]], query: str = "") -> None:
    """Render source citations under each assistant response."""
    if not sources:
        st.caption("No sources available.")
        return

    keywords = _extract_query_keywords(query)
    with st.expander("Sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            file_name = source.get("file_name", "unknown")
            page_number = source.get("page_number")
            snippet = source.get("snippet", "")
            highlighted_snippet = _highlight_keywords(snippet, keywords)
            page_label = page_number if page_number is not None else "N/A"
            st.markdown(f"{index}. **{file_name}** (Page {page_label})")
            st.markdown(f"> {highlighted_snippet}")


def _process_documents(uploaded_files: List[Any]) -> None:
    """Run ingestion and update FAISS index from uploaded files."""
    if not uploaded_files:
        st.warning("Please upload at least one PDF, DOCX, or TXT file.")
        return

    with st.spinner("Processing documents and building vector store..."):
        with tempfile.TemporaryDirectory(prefix="secondbrain_uploads_") as temp_dir:
            upload_dir = Path(temp_dir)
            saved_paths = _save_uploaded_files(uploaded_files, upload_dir)

            # Ingest documents (load + chunk). This also indexes via ingestion pipeline.
            # NOTE: We do NOT re-create the vector store here; that would embed chunks twice.
            chunks = run_ingestion(str(upload_dir))

            st.session_state.docs_processed = True
            st.session_state.processed_files = [path.name for path in saved_paths]
            st.session_state.processed_chunks = chunks
            st.session_state.latest_summary = ""

    st.success(
        f"Processed {len(st.session_state.processed_files)} files and indexed documents successfully."
    )


def _summarize_processed_documents() -> None:
    """Summarize currently processed chunks and store output in session state."""
    if not st.session_state.docs_processed or not st.session_state.processed_chunks:
        st.error("No processed documents available. Upload files and process them first.")
        return

    with st.spinner("Summarizing processed documents..."):
        summary = summarize_documents(
            documents=st.session_state.processed_chunks,
            model_name=st.session_state.llm_model,
        )
    st.session_state.latest_summary = summary


def _handle_user_query(query: str) -> None:
    """Execute retrieval + QA for a user query and append messages to history."""
    if not st.session_state.docs_processed:
        st.error("No documents processed yet. Upload files and click 'Process Documents' first.")
        return
    if not query.strip():
        st.warning("Please enter a query.")
        return

    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Retrieving context and generating answer..."):
        retrieval_mode = st.session_state.retrieval_mode.lower()
        selected_model = st.session_state.llm_model

        retrieved_results = similarity_search(query, top_k=5, mode=retrieval_mode)
        retrieved_docs = _dict_results_to_documents(retrieved_results)
        qa_result = answer_query(query, retrieved_docs, model_name=selected_model)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": qa_result.get("answer", "Not found in documents"),
            "sources": qa_result.get("sources", []),
            "query": query,
        }
    )


def _render_chat_history() -> None:
    """Render full user/assistant conversation from session state."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown("**Answer**")
            st.markdown(message["content"] or "")
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []), query=message.get("query", ""))


def main() -> None:
    """Render the complete Streamlit UI."""
    st.set_page_config(page_title="SecondBrain", page_icon="🧠", layout="wide")
    _init_session_state()

    st.title("SecondBrain - Personal Knowledge AI")

    with st.sidebar:
        st.header("Document Setup")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Upload one or more PDF, DOCX, or TXT files.",
        )

        if st.button("Process Documents", use_container_width=True):
            try:
                _process_documents(uploaded_files)
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")

        st.divider()
        if st.button("Summarize Documents", use_container_width=True):
            try:
                _summarize_processed_documents()
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")

        if st.session_state.processed_files:
            st.caption("Indexed files:")
            for file_name in st.session_state.processed_files:
                st.write(f"- {file_name}")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

    _render_chat_history()

    if st.session_state.latest_summary:
        st.markdown("### Document Summary")
        st.markdown(st.session_state.latest_summary)

    user_query = st.chat_input("Ask a question about your uploaded documents...")
    if user_query is not None:
        try:
            # Render the new turn immediately (avoids forcing a rerun that can
            # look like "chat disappeared" in some browsers/sessions).
            with st.chat_message("user"):
                st.markdown(user_query)
            _handle_user_query(user_query)
            # Render the assistant message we just appended.
            last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
            if last and last.get("role") == "assistant":
                with st.chat_message("assistant"):
                    st.markdown("**Answer**")
                    st.markdown(last.get("content") or "")
                    _render_sources(last.get("sources", []), query=last.get("query", ""))
        except Exception as exc:
            st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
