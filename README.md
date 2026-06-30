# 📬 Inbox Intelligence

Ask natural-language questions across your inbox and get answers grounded in your actual emails, with citations back to the source message. Also drafts context-aware reply suggestions for any source email, in your choice of tone. Built as a RAG (retrieval-augmented generation) pipeline with a real API backend, a chat frontend, and a Docker-based deployment path.

> "What did my landlord say about the deposit?" → an answer pulled from the actual email, not a guess.

**Live demo:** _add your Cloud Run URL here once deployed_

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  Streamlit  │ HTTP │   FastAPI    │      │  Vector store   │
│  frontend   ├─────►│   backend    ├─────►│  (Chroma)       │
└─────────────┘      └──────┬───────┘      └────────────────┘
                             │                       ▲
                 ┌───────────┴───────────┐           │ embed
                 ▼                       ▼            │
         ┌───────────────┐      ┌────────────────┐    │
         │ Gmail API      │      │ Sample dataset │────┘
         │ (OAuth)        │      │ (no auth)      │
         └───────────────┘      └────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ LLM (Ollama or   │
                    │ Claude/Anthropic)│
                    │ via LangChain    │
                    └──────────────────┘
```

**Flow:** emails are pulled (Gmail or sample data) → chunked → embedded (HuggingFace sentence-transformers) → stored in Chroma → a question gets embedded, the most relevant chunks are retrieved, and an LLM answers using only that retrieved context. Each retrieved source email also exposes a "draft a reply" action, which sends that email's content to the LLM with a tone instruction and returns an editable draft.

## Tools used and why

| Tool | Role |
|---|---|
| **LangChain** | Orchestrates the retrieve → prompt → generate pipeline |
| **HuggingFace** (sentence-transformers) | Local, free embedding model |
| **Chroma** | Vector database for similarity search |
| **Ollama** | Free local LLM for development |
| **Anthropic API (Claude)** | Hosted LLM for the cloud demo |
| **FastAPI** | Backend API (typed, auto-documented) |
| **Streamlit** | Chat frontend |
| **Gmail API + OAuth** | Real inbox ingestion |
| **Docker / docker-compose** | Containerized, portable deployment |

## Quickstart (sample data, no setup)

Runs in a few minutes with zero API keys, using the bundled fake inbox in `data/sample_emails.json`.

```bash
# 1. Install Ollama and pull a model (free, local LLM)
ollama pull llama3.2

# 2. Backend
cp .env.example .env
python -m venv venv && source venv/bin/activate
pip install -r requirements-backend.txt
uvicorn app.main:app --reload

# 3. Frontend (second terminal)
pip install -r requirements-frontend.txt
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL, click **"Sync emails"** (source = `sample`), then ask things like:

- "When is rent deposit getting returned?"
- "What's due this week?"
- "Summarize anything related to my job search"

## Draft reply suggestions

Every source email shown alongside an answer has a "Draft a reply" button. Pick a tone (professional, friendly, or brief) and the LLM generates a context-aware reply body for that specific email — editable in the UI before you copy it into your actual email client.

## Switching to real Gmail

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → create a project.
2. Enable the **Gmail API**.
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID** → Application type: **Desktop app**.
4. Download the JSON and save it as `credentials.json` in the project root (already gitignored).
5. In `.env`, set `DATA_SOURCE=gmail`.
6. Restart the backend and sync with source = `gmail`. A browser window opens to grant read-only access — approve it. A `token.json` is cached afterward so you won't need to re-approve every run.

**Note on public demos:** Google requires app verification before an OAuth app can be used by the general public. For a portfolio demo, keep the app in "Testing" mode in Google Cloud Console and add your own account as a test user — or demo with `DATA_SOURCE=sample`, which needs no verification at all.

## Switching to the Anthropic API (for the hosted demo)

```bash
# in .env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

No code changes needed — `app/rag/chain.py` picks the provider based on this one setting.

## Running with Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2
```

Starts three containers: `ollama`, `backend` (FastAPI + RAG), and `frontend` (Streamlit). Visit `http://localhost:8501`.

## Deploying to the cloud (GCP Cloud Run)

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/inbox-backend -f Dockerfile.backend .
gcloud run deploy inbox-backend \
  --image gcr.io/YOUR_PROJECT_ID/inbox-backend \
  --set-env-vars LLM_PROVIDER=anthropic,ANTHROPIC_API_KEY=sk-ant-...,DATA_SOURCE=sample \
  --region us-central1 --allow-unauthenticated

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/inbox-frontend -f Dockerfile.frontend .
gcloud run deploy inbox-frontend \
  --image gcr.io/YOUR_PROJECT_ID/inbox-frontend \
  --set-env-vars BACKEND_URL=https://inbox-backend-xxxxx.run.app \
  --region us-central1 --allow-unauthenticated
```

For the live demo, use `LLM_PROVIDER=anthropic` (Ollama needs a continuously running host, which doesn't fit Cloud Run) and `DATA_SOURCE=sample` (sidesteps Gmail's app-verification requirement for public access).

## Project structure

```
inbox-intelligence/
├── app/
│   ├── main.py              # FastAPI app: /sync, /ask, /draft-reply
│   ├── config.py            # settings, env-var driven
│   ├── models.py            # request/response schemas
│   ├── tracking.py          # local query logging
│   ├── email_ingest/
│   │   ├── sample_data.py   # loads data/sample_emails.json
│   │   ├── gmail_client.py  # Gmail OAuth + fetch + parse
│   │   └── chunking.py      # email -> LangChain Documents
│   └── rag/
│       ├── embeddings.py    # HuggingFace embedding model
│       ├── vectorstore.py   # Chroma wrapper
│       └── chain.py         # RAG chain + draft reply chain
├── frontend/
│   └── streamlit_app.py
├── data/
│   └── sample_emails.json
├── tests/
│   └── test_api.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements-backend.txt
├── requirements-frontend.txt
└── .env.example
```

## Running tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Possible next steps

- Incremental sync (only embed new emails instead of re-indexing everything each time)
- Scheduled re-embedding to guard against drift as the inbox grows
- Prompt caching for the system prompt to cut latency/cost on repeated queries
- Swap Chroma for a managed vector DB (Pinecone, Weaviate) for multi-user deployment
- Auth support so the deployed app can serve more than one user's inbox
