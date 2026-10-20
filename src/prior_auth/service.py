"""Composition root: builds the components once and runs one review end to end."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel

from .config import Settings
from .embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder
from .graph import build_graph
from .knowledge_base import GuidelineIndex
from .llm import build_chat_model
from .policies import PolicyRepository
from .schemas import ReviewRequest, ReviewResult
from .store import ReviewStore
from .tools import ToolBox
from .trace import TraceRecorder

log = logging.getLogger(__name__)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)


@dataclass
class ReviewOutcome:
    result: ReviewResult
    trace: dict[str, Any]


class ReviewService:
    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel | None = None,
        embedder: Embedder | None = None,
        store: ReviewStore | None = None,
    ):
        self.settings = settings
        self.policies = PolicyRepository.from_directory(settings.policies_dir)
        self.embedder = embedder or build_embedder(settings)
        self.guidelines = GuidelineIndex(settings.qdrant_url, settings.qdrant_collection, self.embedder)
        self.store = store or ReviewStore(settings.database_path)
        self.llm = llm or build_chat_model(settings)
        toolbox = ToolBox(self.guidelines, settings.record_search_k, settings.guideline_search_k)
        self.graph = build_graph(self.llm, toolbox, self.policies, self.embedder, settings)

    def start(self) -> None:
        """Index the policy corpus. Safe to call on every start (idempotent upsert)."""
        self.guidelines.wait_until_ready()
        self.guidelines.ingest(self.policies.all())

    def review(self, request: ReviewRequest) -> ReviewOutcome:
        review_id = f"PAR-{uuid.uuid4().hex[:10].upper()}"
        trace = TraceRecorder()
        state: dict[str, Any] = {}
        with get_usage_metadata_callback() as usage:
            try:
                state = self.graph.invoke(
                    {"review_id": review_id, "request": request, "trace": trace}, {"recursion_limit": 80}
                )
            finally:
                # The trace is saved even when the run fails: it is the main debugging aid.
                per_model = usage.usage_metadata.values()
                tokens = {
                    "input_tokens": sum(u["input_tokens"] for u in per_model),
                    "output_tokens": sum(u["output_tokens"] for u in per_model),
                }
                trace_body = {
                    "review_id": review_id,
                    "duration_ms": trace.duration_ms,
                    "tool_calls": trace.tool_calls,
                    "llm_turns": state.get("turns", 0),
                    "tokens": tokens,
                    "events": trace.events,
                }
                self.store.save_trace(review_id, trace_body)
        result: ReviewResult = state["result"]
        self.store.save_review(result.review_id, result.model_dump(mode="json"))
        log.info(
            "Review %s: %s in %d ms, %d tool calls, %s tokens",
            result.review_id,
            result.recommendation,
            trace.duration_ms,
            trace.tool_calls,
            tokens,
        )
        return ReviewOutcome(result=result, trace=trace_body)
