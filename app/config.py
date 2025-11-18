"""Global configuration and environment loading for the UB RAG bot."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file in the project root if present.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama configuration (used when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# Data and vector store configuration
UB_DATA_DIR = os.getenv("UB_DATA_DIR", str(BASE_DIR / "data" / "ub_pages"))
VECTOR_STORE_DIR = os.getenv(
    "VECTOR_STORE_DIR", str(BASE_DIR / "vector_store" / "ub")
)
UB_COLLECTION_NAME = os.getenv("UB_COLLECTION_NAME", "ub_knowledge")

# Conversation memory settings
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))


def get_openai_client():
    """
    Create and return an OpenAI client using the configured API key.

    Raises:
        RuntimeError: if OPENAI_API_KEY is not configured.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it in your environment or .env file."
        )

    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)
