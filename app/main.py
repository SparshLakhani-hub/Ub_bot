"""FastAPI application exposing the UB RAG chatbot API."""

from typing import Dict, List, Optional

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import MAX_HISTORY_TURNS
from app.rag_pipeline import generate_answer, list_sources, load_vector_store

app = FastAPI(
    title="UB RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for University at Buffalo.",
    version="0.1.0",
)

# Allow CORS for local development and embedding on arbitrary domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Source(BaseModel):
    source_file: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for maintaining short conversation history.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="User's question or message.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Source]


conversation_store: Dict[str, List[Dict[str, str]]] = {}


@app.on_event("startup")
def startup_event():
    """Initialize the vector store at application startup."""
    try:
        load_vector_store()
    except Exception as exc:  # pragma: no cover
        print(f"Warning: Failed to initialize vector store: {exc}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint for RAG-based UB Bot.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or uuid.uuid4().hex
    history = conversation_store.get(session_id, [])

    try:
        answer, sources = generate_answer(
            user_query=request.message,
            conversation_history=history,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {exc}",
        ) from exc

    history = history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": answer},
    ]

    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history = history[-max_messages:]

    conversation_store[session_id] = history

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=[Source(**s) for s in sources],
    )


@app.get("/sources")
def get_sources(limit: int = 20):
    """Return a sample of known documents in the vector store."""
    try:
        docs = list_sources(limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sources: {exc}",
        ) from exc
    return {"count": len(docs), "documents": docs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
