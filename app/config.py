"""
Central configuration. Every setting can be overridden via environment
variables (or a local .env file) so the same code runs unchanged in
development, Docker, and the cloud — only the env vars change.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Data source: "sample" or "gmail" ---
    DATA_SOURCE = os.getenv("DATA_SOURCE", "sample")

    # --- LLM provider: "ollama" (free/local) or "anthropic" (hosted) ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # --- Embeddings (HuggingFace, runs locally, no API key needed) ---
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Vector store ---
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

    # --- Gmail OAuth ---
    GMAIL_CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "./credentials.json")
    GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", "./token.json")
    GMAIL_MAX_RESULTS = int(os.getenv("GMAIL_MAX_RESULTS", "50"))

    # --- Weights & Biases tracking ---
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "inbox-intelligence")
    TRACKING_ENABLED = bool(WANDB_API_KEY)


settings = Settings()
