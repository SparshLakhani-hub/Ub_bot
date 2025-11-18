# UB RAG Chatbot

> A RAG-powered chatbot for the University at Buffalo that answers prospective student questions using UBâ€™s own web content.

Repository: https://github.com/SparshLakhani-hub/Ub_bot

**Tech stack**
- Python, FastAPI
- ChromaDB (vector store)
- Ollama (Llama 3.1 + `nomic-embed-text`) or OpenAI
- HTML / Tailwind floating chat widget

---

## Overview

UB RAG Chatbot is a retrieval-augmented generation (RAG) assistant focused on the University at Buffalo. It indexes key UB pages (MS CS program, undergraduate admissions, housing and dining, etc.) and uses that content to answer questions from prospective and current students.

Content comes from:
- `scripts/scrape_ub_site.py`, which can crawl UBâ€™s public site (including the **Computer Science & Engineering faculty directory**) and write cleaned text into `data/ub_pages/ub_cse_faculty_directory.txt` and other files.
- `scripts/ingest_ub_content.py`, which embeds all `.txt`/`.md` files in `data/ub_pages/` into a local **ChromaDB** vector store.

The backend is a **FastAPI** app with:
- `GET /health` â€“ basic health check
- `POST /chat` â€“ main RAG chat endpoint (returns `answer` + `sources`)
- `GET /sources` â€“ sample of stored documents for debugging

The frontend is a floating, Tailwind-based chat widget in `frontend/ub_chat_widget.html` that talks to the FastAPI backend at `http://localhost:8000` by default.

---

## Features

- Retrieval-augmented generation (RAG) over UB program, admissions, and housing pages.
- CSE faculty directory scraper for richer answers about Computer Science & Engineering faculty and professors.
- Configurable LLM provider: OpenAI or fully local Ollama.
- Persistent **ChromaDB** vector store on disk.
- Modern floating chat widget that can be embedded into any site.
- Conversation history keyed by `session_id`, so the bot can handle follow-up questions.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### Quickstart (local dev)

```bash
# 1. Clone the repo
git clone https://github.com/SparshLakhani-hub/Ub_bot.git
cd Ub_bot

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (separate terminal)
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text

# 5. Ingest UB content into Chroma
python scripts/scrape_ub_site.py      # optional extra scrape
python scripts/ingest_ub_content.py

# 6. Run the FastAPI backend
uvicorn app.main:app --reload --port 8000

# 7. Open the chat widget in your browser
# (Adjust the path if needed)
# file:///C:/.../Ub_bot/frontend/ub_chat_widget.html
```

The chat widget’s script uses `API_BASE_URL = "http://localhost:8000"` by default, so it will talk to your local FastAPI server without extra configuration.

---

## Project Structure

```text
.
â”œâ”€ app/
â”‚  â”œâ”€ __init__.py
â”‚  â”œâ”€ config.py          # Env vars, paths, model names
â”‚  â”œâ”€ rag_pipeline.py    # Vector lookup + LLM calls
â”‚  â””â”€ main.py            # FastAPI app and endpoints
â”œâ”€ scripts/
â”‚  â”œâ”€ ingest_ub_content.py  # Ingest local UB text into Chroma
â”‚  â””â”€ scrape_ub_site.py     # Optional UB web scraper
â”œâ”€ data/
â”‚  â””â”€ ub_pages/          # Raw UB text/Markdown files (you create/fill)
â”œâ”€ vector_store/
â”‚  â””â”€ ub/                # Chroma persistent vector DB (created by ingest)
â”œâ”€ frontend/
â”‚  â””â”€ ub_chat_widget.html   # Embeddable chat widget
â”œâ”€ .env.example
â”œâ”€ requirements.txt
â””â”€ README.md
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

4. You should see a floating â€œAsk UB Botâ€ chat box at the bottom-right. Type a UB-related question and click **Send** (or press **Enter**).

The widget:

- Maintains a `session_id` to keep short conversation history
- Shows â€œUB Bot is thinkingâ€¦â€ while waiting for a response
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

