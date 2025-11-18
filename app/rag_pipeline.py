"""RAG pipeline for the UB chatbot."""

from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
import requests

from app.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
    UB_COLLECTION_NAME,
    get_openai_client,
)

# Global Chroma client and collection (lazy-initialized)
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the configured LLM provider.

    Returns a list of embedding vectors, one per input text.
    """
    if not texts:
        return []

    provider = (LLM_PROVIDER or "").lower()

    if provider == "openai":
        client = get_openai_client()
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    if provider == "ollama":
        base_url = (OLLAMA_BASE_URL or "").rstrip("/")
        embeddings: List[List[float]] = []

        for text in texts:
            try:
                resp = requests.post(
                    f"{base_url}/api/embed",
                    json={"model": OLLAMA_EMBED_MODEL, "input": text},
                    timeout=60,
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    "Ollama is not running at OLLAMA_BASE_URL; please install Ollama, "
                    "run `ollama serve`, and pull the required models."
                ) from exc

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama embeddings request failed with status {resp.status_code}: "
                    f"{resp.text}"
                )

            data = resp.json()
            if "embeddings" in data and data["embeddings"]:
                # Standard Ollama embed response: {"embeddings": [[...]]}
                vec = data["embeddings"][0]
            elif "embedding" in data:
                # Fallback: single vector
                vec = data["embedding"]
            else:
                raise RuntimeError(
                    "Unexpected response format from Ollama embeddings API."
                )

            if not isinstance(vec, list) or not vec:
                raise RuntimeError(
                    "Received empty or invalid embedding vector from Ollama."
                )

            embeddings.append(vec)

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"embed_texts: embedding count {len(embeddings)} does not match "
                f"text count {len(texts)}"
            )

        return embeddings

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def chat_completion(messages: List[Dict[str, str]]) -> str:
    """
    Run a chat completion using the configured LLM provider and return the text.
    """
    provider = (LLM_PROVIDER or "").lower()

    if provider == "openai":
        client = get_openai_client()
        completion = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()

    if provider == "ollama":
        base_url = (OLLAMA_BASE_URL or "").rstrip("/")
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": messages,
                    # Ensure a single JSON response instead of a streaming payload.
                    "stream": False,
                },
                timeout=300,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama is not running at OLLAMA_BASE_URL; please install Ollama, "
                "run `ollama serve`, and pull the required models."
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama chat request failed with status {resp.status_code}: "
                f"{resp.text}"
            )

        data = resp.json()

        # Standard Ollama chat format: {"message": {"role": "assistant", "content": "..."}}
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content") or ""
            return content.strip()

        # Fallback for OpenAI-like-compatible structures, if ever returned
        if "choices" in data and data["choices"]:
            message = data["choices"][0].get("message", {})
            content = message.get("content") or ""
            return content.strip()

        raise RuntimeError(
            "Unexpected response format from Ollama chat API; "
            "could not find assistant message content."
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def load_vector_store():
    """
    Initialize and return the Chroma collection for UB documents.

    If the collection does not exist or is empty, ingestion must be run first.
    """
    global _chroma_client, _collection

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=VECTOR_STORE_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=UB_COLLECTION_NAME,
            metadata={"description": "UB website and document content"},
        )

    return _collection


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed the user query, search the vector store, and return the top-k chunks.
    """
    collection = load_vector_store()

    query_embeddings = embed_texts([query])
    if not query_embeddings:
        return []
    query_embedding = query_embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else [
        None
    ] * len(ids)

    matched_docs: List[Dict[str, Any]] = []
    for idx, doc_id in enumerate(ids):
        matched_docs.append(
            {
                "id": doc_id,
                "content": docs[idx],
                "metadata": metadatas[idx] or {},
                "distance": distances[idx],
            }
        )

    return matched_docs


def build_prompt_from_context(
    user_query: str,
    retrieved_docs: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages for the chat completion API using context and optional history.
    """
    system_prompt = (
        "You are UB Bot, a helpful, friendly assistant for the University at Buffalo (UB). "
        "You answer questions for prospective and current students about UB programs, admissions, "
        "housing, campus life, and related topics.\n\n"
        "Use ONLY the context passages provided to you plus obvious, generic admissions knowledge. "
        "Be as clear and concrete as possible in your answers.\n\n"
        "Rules:\n"
        "- Give a direct, helpful answer. Do not just tell the user to 'check the website' or 'contact support'.\n"
        "- When the user asks about applying or the process to apply (for example, 'how do I apply', "
        "'what is the process', or 'I want to take admission'), respond with a short, step-by-step list of the "
        "main steps in the process based on the context and general admissions flow "
        "(review requirements, prepare documents, submit online application, pay fee, track status, etc.).\n"
        "- Do NOT invent specific numbers that are not in the context (no specific GPA cutoffs, deadlines, fees, "
        "or dollar amounts unless they appear in the context).\n"
        "- Do NOT mention or reference 'Source 1', 'Source 2', file names, or .txt documents in your answer. "
        "Just answer as if you know the information.\n"
        "- Keep your tone supportive and student-friendly. You can use bullet points or numbered lists when "
        "describing steps.\n"
        "- Only if important details are clearly missing, add ONE short line at the very end like: "
        "\"For the latest official requirements and deadlines, please confirm on UB’s official website.\""
    )

    if retrieved_docs:
        context_lines: List[str] = [
            "Here are context passages from University at Buffalo information. Use them when answering the question."
        ]
        for idx, doc in enumerate(retrieved_docs, start=1):
            meta = doc.get("metadata", {}) or {}
            title = meta.get("title") or "Untitled section"
            url = meta.get("url")

            header = f"\n[Context {idx}] Title: {title}"
            if url:
                header += f" | URL: {url}"

            context_lines.append(header)
            context_lines.append(doc.get("content", "").strip())
            context_lines.append("---")
        context_text = "\n".join(context_lines)
    else:
        context_text = (
            "No specific context passages were retrieved for this question. "
            "Answer using only obvious, generic admissions knowledge and, if you cannot "
            "answer reliably, say you are not sure and add one short line at the very end like: "
            "\"For the latest official requirements and deadlines, please confirm on UB’s official website.\""
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if retrieved_docs:
        messages.append({"role": "user", "content": context_text})

    if conversation_history:
        messages.extend(conversation_history)

    user_message_content = (
        "User question:\n"
        f"{user_query}\n\n"
        "Answer this question following the instructions above, using the provided context passages "
        "and obvious, generic admissions knowledge. Give a direct, student-friendly answer, and only if "
        "important details are clearly missing, add one short line at the very end suggesting the user "
        "confirm on UB’s official website."
    )

    messages.append({"role": "user", "content": user_message_content})
    return messages


def generate_answer(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
) -> Tuple[str, List[Dict[str, Optional[str]]]]:
    """
    Full RAG pipeline: retrieve, build prompt, call the configured LLM, return answer + sources.
    """
    retrieved_docs = retrieve_relevant_chunks(user_query, top_k=top_k)
    messages = build_prompt_from_context(
        user_query=user_query,
        retrieved_docs=retrieved_docs,
        conversation_history=conversation_history,
    )
    answer = chat_completion(messages)

    sources: List[Dict[str, Optional[str]]] = []
    for doc in retrieved_docs:
        meta = doc.get("metadata", {}) or {}
        sources.append(
            {
                "source_file": meta.get("source_file"),
                "title": meta.get("title"),
                "url": meta.get("url"),
            }
        )

    return answer, sources


def list_sources(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return a small sample of known documents in the vector store for debugging.
    """
    collection = load_vector_store()
    results = collection.get(limit=limit)

    ids = results.get("ids", [])
    metadatas = results.get("metadatas", [])

    out: List[Dict[str, Any]] = []
    for idx, doc_id in enumerate(ids):
        out.append({"id": doc_id, "metadata": metadatas[idx]})
    return out
