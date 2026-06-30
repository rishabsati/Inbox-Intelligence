"""
Logs query metrics to Weights & Biases -- this is what makes the
latency-vs-quality tradeoff visible instead of just theoretical. If
WANDB_API_KEY isn't set, tracking silently no-ops so the app still
runs fine without a W&B account.
"""
from app.config import settings


class MetricsTracker:
    def __init__(self):
        self.enabled = settings.TRACKING_ENABLED
        self._wandb = None

        if self.enabled:
            try:
                import wandb

                wandb.init(
                    project=settings.WANDB_PROJECT,
                    config={
                        "llm_provider": settings.LLM_PROVIDER,
                        "embedding_model": settings.EMBEDDING_MODEL,
                        "retrieval_k": settings.RETRIEVAL_K,
                    },
                    reinit=True,
                )
                self._wandb = wandb
            except Exception as e:  # pragma: no cover
                print(f"[tracking] W&B init failed, disabling tracking: {e}")
                self.enabled = False

    def log_query(self, question: str, answer: str, latency_seconds: float, num_sources: int):
        if not self.enabled:
            return
        self._wandb.log(
            {
                "latency_seconds": latency_seconds,
                "num_sources_retrieved": num_sources,
                "question_length": len(question),
                "answer_length": len(answer),
            }
        )


tracker = MetricsTracker()
