"""
Chroma is the vector database: it stores each email chunk's embedding
and does fast similarity search to find the chunks most relevant to a
question. It persists to disk so you don't have to re-embed everything
on every restart -- only when you explicitly /sync again.
"""
from langchain_chroma import Chroma
from app.rag.embeddings import get_embedding_model
from app.config import settings

_vectorstore = None
_COLLECTION_NAME = "inbox_emails"


def _new_store():
    return Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _new_store()
    return _vectorstore


def index_documents(documents: list) -> int:
    """Wipes the existing collection and re-indexes from scratch.

    Simpler and safer for a demo than diffing/updating individual
    emails. A production version would track which email IDs are
    already indexed and only embed new ones (and re-embed periodically
    to guard against embedding drift).
    """
    global _vectorstore
    store = get_vectorstore()
    store.delete_collection()
    _vectorstore = _new_store()
    _vectorstore.add_documents(documents)
    return len(documents)
