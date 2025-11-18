# UB RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the University at Buffalo (UB), built with FastAPI, OpenAI/Ollama, and a local Chroma vector database. The bot answers questions for prospective and newly admitted students using UB website content and exposes a simple embeddable web widget.

Repository: https://github.com/SparshLakhani-hub/Ub_bot

## Overview and Features

- RAG-based QA over curated UB pages (admissions, housing, MS CS program, etc.).
- Optional scraper that pulls the **CSE faculty directory** and saves cleaned text into `data/ub_pages/ub_cse_faculty_directory.txt` so the bot can answer questions about professors.
- Local vector store using **ChromaDB** for fast semantic search.
- Pluggable LLM backend:
  - **Ollama** for fully local, no-cost inference.
  - **OpenAI** for hosted models if you have an API key.
- REST API built with **FastAPI** exposing `/chat`, `/health`, and `/sources` endpoints.
- Standalone HTML **chat widget** (`frontend/ub_chat_widget.html`) that can be embedded into other sites.
- Conversation history per session ID so the bot can handle follow-up questions.

---

## Project Structure

```text
.
├─ app/
│  ├─ __init__.py
│  ├─ config.py          # Env vars, paths, model names
│  ├─ rag_pipeline.py    # Vector lookup + LLM calls
│  └─ main.py            # FastAPI app and endpoints
├─ scripts/
│  ├─ ingest_ub_content.py  # Ingest local UB text into Chroma
│  └─ scrape_ub_site.py     # Optional UB web scraper
├─ data/
│  └─ ub_pages/          # Raw UB text/Markdown files (you create/fill)
├─ vector_store/
│  └─ ub/                # Chroma persistent vector DB (created by ingest)
├─ frontend/
│  └─ ub_chat_widget.html   # Embeddable chat widget
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## Prerequisites

- Python 3.9+ (3.10+ recommended)
- EITHER:
  - Ollama installed and running locally (recommended, free local mode), or
  - An OpenAI API key (for using OpenAI-hosted models)

---

## Setup

### 1. Create and activate a virtual environment

On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Copy `.env.example` to `.env` and edit values as needed.

By default, the project is configured to use **Ollama** locally:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text

# Only used if LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

UB_DATA_DIR=./data/ub_pages
VECTOR_STORE_DIR=./vector_store/ub
UB_COLLECTION_NAME=ub_documents
MAX_HISTORY_TURNS=4
```

If you prefer using OpenAI (and have sufficient quota), set:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

and the app will use OpenAI for both embeddings and chat.

---

## Running locally with Ollama (no OpenAI quota)

This mode uses local models via Ollama and does **not** require an OpenAI API key.

1. Install Ollama from the official website and ensure it is running (e.g. `ollama serve`).
2. In a terminal, pull the required models:

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

3. In `.env`, configure:

   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_CHAT_MODEL=llama3.1
   OLLAMA_EMBED_MODEL=nomic-embed-text
   ```

   (Adjust the model names if you use different local models.)

4. Ensure `UB_DATA_DIR` points to a folder containing `.txt` / `.md` UB content files (e.g. `data/ub_pages/`).
5. Build the vector store:

   ```bash
   python scripts/ingest_ub_content.py
   ```

6. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

7. Open `frontend/ub_chat_widget.html` in a browser and confirm:

   ```javascript
   const API_BASE_URL = "http://localhost:8000";
   ```

   You can now chat with the UB bot using local models via Ollama.

---

## Preparing UB Content

### Option A: Use local text/Markdown files

1. Ensure the folder exists:

   ```bash
   mkdir -p data/ub_pages
   ```

2. Place `.txt` and/or `.md` files under `data/ub_pages/`.  
   Each file should represent one UB page or section. The first heading (e.g. `# Title`) or first non-empty line will be treated as the title.

### Option B (Optional): Scrape UB public pages

Edit `scripts/scrape_ub_site.py` to adjust `SEED_URLS`, `ALLOWED_DOMAINS`, and limits if needed, then run:

```bash
python scripts/scrape_ub_site.py
```

This will save cleaned text files into `data/ub_pages/`.

---

## Ingesting Data into the Vector Store

Once `data/ub_pages/` contains UB content, run:

```bash
python scripts/ingest_ub_content.py
```

This script will:

- Read all `.txt` / `.md` files under `UB_DATA_DIR`
- Split content into overlapping chunks
- Generate embeddings using OpenAI
- Store documents + embeddings in a persistent Chroma DB under `VECTOR_STORE_DIR`

You should see logs indicating how many files and chunks were ingested.

---

## Running the API Server

Start the FastAPI app with uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Endpoints

- `GET /health`  
  Returns:

  ```json
  { "status": "ok" }
  ```

- `POST /chat`  
  Request JSON:

  ```json
  {
    "session_id": "optional-string-id",
    "message": "user question here"
  }
  ```

  Response JSON:

  ```json
  {
    "session_id": "generated-or-existing-id",
    "answer": "LLM answer using UB knowledge base",
    "sources": [
      {
        "source_file": "relative/path/to/file.txt",
        "title": "Document title",
        "url": "optional-ub-link"
      }
    ]
  }
  ```

- `GET /sources`  
  Returns a small sample of documents in the vector store (for debugging).

---

## Frontend Chat Widget

1. Ensure the backend is running at `http://localhost:8000`.
2. Open `frontend/ub_chat_widget.html` in a browser.
3. At the top of the script in that file, ensure:

   ```javascript
   const API_BASE_URL = "http://localhost:8000";
   ```

4. You should see a floating “Ask UB Bot” chat box at the bottom-right. Type a UB-related question and click **Send** (or press **Enter**).

The widget:

- Maintains a `session_id` to keep short conversation history
- Shows “UB Bot is thinking…” while waiting for a response
- Displays the answer from the backend

---

## Embedding the Widget in a UB Page

To embed the widget in an existing UB page:

1. Copy the `<style>`, `<div id="ub-chat-widget">`, and `<script>` blocks from `frontend/ub_chat_widget.html`.
2. Paste them into the UB page template.
3. Set `API_BASE_URL` in the script to your deployed backend URL (e.g. `https://your-domain.edu/api`).
4. Deploy both the backend (FastAPI app) and the static HTML.

---

## Notes

- The chatbot is instructed to:
  - Stay on UB-related topics
  - Use only retrieved context and avoid inventing facts
  - Admit uncertainty and point users to the official UB website when needed
- Conversation history is held in memory per-process and is not persisted; restarting the server clears histories.
