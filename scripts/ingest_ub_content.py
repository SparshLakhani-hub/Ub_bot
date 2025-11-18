"""Ingest UB content files into a local Chroma vector store."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Ensure the project root is importable when running as a script.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.config import (  # noqa: E402
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    UB_COLLECTION_NAME,
    UB_DATA_DIR,
    VECTOR_STORE_DIR,
)
from app.rag_pipeline import embed_texts  # noqa: E402
import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402


load_dotenv()


def iter_text_files(base_dir: Path) -> List[Path]:
    """Recursively find all .txt and .md files under base_dir."""
    return [
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    ]


def extract_title_and_body(text: str) -> Tuple[str, str]:
    """
    Extract a title and body from a document.

    For Markdown, uses the first '# heading' as title; otherwise, the first
    non-empty line becomes the title.
    """
    lines = [line.strip() for line in text.splitlines()]
    title = "Untitled"
    body_lines: List[str] = []

    found_title = False
    for line in lines:
        if not found_title:
            if line.startswith("#"):
                title = line.lstrip("#").strip() or "Untitled"
                found_title = True
                continue
            if line:
                title = line
                found_title = True
                continue
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    return title, body or text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """Simple character-level text splitter for retrieval chunks.

    Ensures progress even for short texts so we do not get stuck in a loop
    when applying overlap.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= length:
            break
        # Move start forward with overlap but never backwards or stuck.
        next_start = end - overlap
        if next_start <= start:
            start = end
        else:
            start = max(0, next_start)

    return chunks


def main():
    """Main ingestion routine."""

    provider = (LLM_PROVIDER or "").lower()

    if provider == "openai":
        if not OPENAI_API_KEY or "YOUR_OPENAI_API_KEY_HERE" in OPENAI_API_KEY:
            print(
                "OPENAI_API_KEY is not set or is still a placeholder. "
                "Please edit .env and set your real key before running ingestion."
            )
            sys.exit(1)
    elif provider == "ollama":
        # No API key required; embeddings will be generated via Ollama.
        pass
    else:
        print(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
        sys.exit(1)

    data_dir = Path(UB_DATA_DIR)
    if not data_dir.exists():
        raise RuntimeError(f"Data directory does not exist: {data_dir}")

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=VECTOR_STORE_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=UB_COLLECTION_NAME,
        metadata={"description": "UB website and document content"},
    )

    files = iter_text_files(data_dir)
    if not files:
        print(
            f"No .txt or .md files found under {data_dir}.\n"
            "Add UB content files to this directory or run scripts/scrape_ub_site.py first."
        )
        sys.exit(1)

    print(f"Found {len(files)} files to ingest from {data_dir}.")

    all_ids: List[str] = []
    all_texts: List[str] = []
    all_metadatas: List[Dict[str, str]] = []

    for file_path in files:
        rel_path = file_path.relative_to(data_dir)
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        title, body = extract_title_and_body(raw_text)
        chunks = split_text_into_chunks(body)
        if not chunks:
            continue

        for idx, chunk in enumerate(chunks):
            doc_id = f"{rel_path.as_posix()}::chunk-{idx}"
            metadata = {
                "source_file": rel_path.as_posix(),
                "title": title,
                # Store URL as empty string if unknown to avoid None in Chroma metadata.
                "url": "",
            }
            all_ids.append(doc_id)
            all_texts.append(chunk)
            all_metadatas.append(metadata)

    if not all_texts:
        print("No text chunks generated. Nothing to ingest.")
        sys.exit(1)

    print(
        f"Prepared {len(all_texts)} text chunks from {len(files)} files.\n"
        f"Storing embeddings in vector DB at '{VECTOR_STORE_DIR}', collection '{UB_COLLECTION_NAME}'."
    )

    batch_size = 64
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_metadatas = all_metadatas[i : i + batch_size]

        try:
            embeddings = embed_texts(batch_texts)
        except RuntimeError as exc:
            if provider == "ollama":
                print(
                    "Could not reach Ollama at OLLAMA_BASE_URL. Make sure Ollama is "
                    "installed, running, and the embedding model is available."
                )
                sys.exit(1)
            raise

        if len(embeddings) != len(batch_texts):
            print(
                "Embedding count does not match chunk count: "
                f"{len(embeddings)} embeddings for {len(batch_texts)} chunks."
            )
            sys.exit(1)

        if not (
            len(batch_ids) == len(batch_texts) == len(batch_metadatas) == len(embeddings)
        ):
            print(
                "Batch length mismatch before collection.add:\n"
                f"- ids: {len(batch_ids)}\n"
                f"- documents: {len(batch_texts)}\n"
                f"- metadatas: {len(batch_metadatas)}\n"
                f"- embeddings: {len(embeddings)}"
            )
            sys.exit(1)

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas,
            embeddings=embeddings,
        )

        print(f"Ingested batch {i // batch_size + 1} ({len(batch_texts)} chunks)")

    print("Ingestion complete. Vector store is ready at:", VECTOR_STORE_DIR)


if __name__ == "__main__":
    main()
