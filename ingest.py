"""
Document ingestion module.

Purpose:
- load source documents from disk or other sources
- split documents into chunks
- pass chunks to the vector store for indexing
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from utils.chunking import chunk_documents
from vector_store import index_chunks


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _discover_files(source_path: str) -> List[Path]:
    """Collect all supported files from a file or directory path."""
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [path]

    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise ValueError(
            f"No supported files found in {source_path}. "
            f"Expected one of: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return files


def _get_loader(file_path: Path):
    """Return the appropriate loader for each supported file type."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    if suffix == ".docx":
        return Docx2txtLoader(str(file_path))
    if suffix == ".txt":
        # autodetect_encoding handles common mixed-encoding text files.
        return TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip empty lines for cleaner chunks."""
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(cleaned_lines).strip()


def _normalize_metadata(document: Document, file_path: Path) -> Dict[str, object]:
    """Create consistent chunk metadata across file formats."""
    metadata = dict(document.metadata or {})
    page_value = metadata.get("page", metadata.get("page_number"))

    # Ensure required fields are always present.
    normalized: Dict[str, object] = {
        "file_name": file_path.name,
        "page_number": page_value + 1 if isinstance(page_value, int) else page_value,
        "source": str(file_path),
    }
    return normalized


def _load_file_documents(file_path: Path) -> List[Document]:
    """Load and clean text content for a single file."""
    loader = _get_loader(file_path)
    raw_docs = loader.load()

    cleaned_docs: List[Document] = []
    for raw_doc in raw_docs:
        cleaned_text = _clean_text(raw_doc.page_content)
        if not cleaned_text:
            continue

        metadata = _normalize_metadata(raw_doc, file_path)
        cleaned_docs.append(Document(page_content=cleaned_text, metadata=metadata))

    return cleaned_docs


def _iter_loaded_documents(files: Iterable[Path]) -> Iterable[Document]:
    """Yield documents file-by-file while isolating per-file loader errors."""
    for file_path in files:
        try:
            for doc in _load_file_documents(file_path):
                yield doc
        except Exception as exc:
            # Surface which file failed while continuing other files.
            raise RuntimeError(f"Failed to load '{file_path}': {exc}") from exc


def load_documents(source_path: str) -> List[Document]:
    """
    Load supported documents (PDF, DOCX, TXT) with normalized text/metadata.
    """
    files = _discover_files(source_path)
    documents = list(_iter_loaded_documents(files))
    if not documents:
        raise ValueError(f"No readable text found in supported files under: {source_path}")
    return documents


def run_ingestion(
    source_path: str,
    chunk_size: int = 700,
    overlap: int = 100,
    embedding_model_name: Optional[str] = None,
) -> List[Document]:
    """
    End-to-end ingestion pipeline:
    1) load documents, 2) split into chunks, 3) index chunk text.

    Returns:
        List of chunk Documents, each containing:
        - page_content: chunk text
        - metadata.file_name
        - metadata.page_number (when available)
    """
    docs = load_documents(source_path)
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Chunking produced no chunks; verify input documents.")

    # Index full documents so metadata is preserved in FAISS.
    index_chunks(
        chunks,
        model_name=embedding_model_name,
    )
    return chunks


if __name__ == "__main__":
    # Example execution path; replace "data/" with your corpus location.
    generated_chunks = run_ingestion("data/")
    print(f"Ingested {len(generated_chunks)} chunks.")
