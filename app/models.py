from typing import List, Optional
from pydantic import BaseModel


class SyncRequest(BaseModel):
    source: Optional[str] = None
    max_results: Optional[int] = None


class SyncResponse(BaseModel):
    status: str
    documents_indexed: int
    source: str


class AskRequest(BaseModel):
    question: str


class SourceEmail(BaseModel):
    subject: str
    sender: str
    date: str
    snippet: str
    category: str = "Other"


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceEmail]
    latency_seconds: float
    llm_provider: str


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    data_source: str


class DraftReplyRequest(BaseModel):
    subject: str
    sender: str
    date: str
    snippet: str
    tone: str = "professional"


class DraftReplyResponse(BaseModel):
    draft: str
    tone: str