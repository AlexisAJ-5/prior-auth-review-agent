"""Smoke test against the real model in .env. Skipped when no LLM is configured (CI, fresh clones)."""

from pathlib import Path

import pytest

from prior_auth.config import Settings
from prior_auth.schemas import ReviewRequest
from prior_auth.service import ReviewService
from prior_auth.store import ReviewStore
from tests.conftest import load_case

LIVE = Settings()


@pytest.mark.live
@pytest.mark.skipif(not LIVE.llm_configured, reason="no LLM configured in the environment or .env")
def test_real_model_completes_a_review(tmp_path: Path) -> None:
    service = ReviewService(
        LIVE.model_copy(update={"qdrant_url": ":memory:"}), store=ReviewStore(tmp_path / "live.db")
    )
    service.start()
    outcome = service.review(ReviewRequest.model_validate(load_case("PA-001")["request"]))
    assert outcome.result.agent_completed
    assert len(outcome.result.criteria) == 4
    assert outcome.trace["tool_calls"] >= 4
