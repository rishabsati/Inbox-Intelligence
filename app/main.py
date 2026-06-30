"""
FastAPI backend. Two real endpoints:

  POST /sync   -- pulls emails (sample or Gmail), chunks them, embeds
                  them, and indexes them into the vector store.
  POST /ask    -- runs the RAG chain against a question, returns the
                  answer plus the source emails it was grounded in.

This is the same pattern real companies use to expose an AI feature:
a typed, documented HTTP API that any frontend (Streamlit here, but
could be a React app, mobile app, Slack bot, etc.) can call.
"""
import time

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.models import (
    AskRequest,
    AskResponse,
    DraftReplyRequest,
    DraftReplyResponse,
    HealthResponse,
    SourceEmail,
    SyncRequest,
    SyncResponse,
)
from app.email_ingest.chunking import emails_to_documents
from app.email_ingest.sample_data import get_sample_emails
from app.rag.chain import ask_question, draft_reply
from app.rag.vectorstore import index_documents
from app.tracking import tracker

app = FastAPI(
    title="Inbox Intelligence API",
    description="RAG over your email -- ask questions, get grounded answers with sources.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        data_source=settings.DATA_SOURCE,
    )


@app.post("/sync", response_model=SyncResponse)
def sync_emails(req: SyncRequest):
    source = req.source or settings.DATA_SOURCE
    max_results = req.max_results or settings.GMAIL_MAX_RESULTS

    if source == "gmail":
        from app.email_ingest.gmail_client import fetch_recent_emails

        try:
            emails = fetch_recent_emails(max_results=max_results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gmail fetch failed: {e}")
    elif source == "sample":
        emails = get_sample_emails()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown source '{source}', expected 'sample' or 'gmail'.")

    documents = emails_to_documents(emails)
    count = index_documents(documents)

    return SyncResponse(status="success", documents_indexed=count, source=source)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()
    try:
        answer, sources = ask_question(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    latency = time.time() - start

    tracker.log_query(req.question, answer, latency, len(sources))

    return AskResponse(
        answer=answer,
        sources=[SourceEmail(**s) for s in sources],
        latency_seconds=round(latency, 2),
        llm_provider=settings.LLM_PROVIDER,
    )


@app.post("/draft-reply", response_model=DraftReplyResponse)
def draft_reply_endpoint(req: DraftReplyRequest):
    try:
        draft = draft_reply(
            subject=req.subject,
            sender=req.sender,
            date=req.date,
            snippet=req.snippet,
            tone=req.tone,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Draft reply failed: {e}")

    return DraftReplyResponse(draft=draft, tone=req.tone)