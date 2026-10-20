"""Offline fixtures: a scripted chat model, the hashing embedder and in-memory Qdrant. No network."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from prior_auth.api import create_app
from prior_auth.config import Settings
from prior_auth.embeddings import HashingEmbedder
from prior_auth.policies import PolicyRepository
from prior_auth.service import ReviewService
from prior_auth.store import ReviewStore

ROOT = Path(__file__).resolve().parents[1]
Call = tuple[str, dict[str, Any]]


class ScriptedChatModel:
    """Stands in for the chat model. Replays one list of tool calls per agent turn.

    An argument value "$hit:N" is replaced by the id of the N-th sentence returned by the most
    recent search_clinical_record call, and "$section:N" likewise for guideline sections. That keeps
    scripts independent of the exact ranking.
    """

    def __init__(self, turns: list[list[Call]]):
        self.turns = turns
        self.calls = 0
        self.seen: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": _resolve(args, messages),
                    "id": f"call_{self.calls}_{i}",
                    "type": "tool_call",
                }
                for i, (name, args) in enumerate(turn)
            ],
        )


def _latest(messages: list[BaseMessage], key: str) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            body = json.loads(str(message.content))
            if key in body:
                return body[key]  # type: ignore[no-any-return]
    return []


def _resolve(value: Any, messages: list[BaseMessage]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, messages) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, messages) for v in value]
    if isinstance(value, str) and value.startswith("$hit:"):
        return _latest(messages, "sentences")[int(value[5:])]["id"]
    if isinstance(value, str) and value.startswith("$section:"):
        return _latest(messages, "sections")[int(value[9:])]["section_id"]
    return value


def search(query: str) -> Call:
    return ("search_clinical_record", {"purpose": f"Look for {query}", "query": query})


def guideline(query: str) -> Call:
    return ("search_policy_guidelines", {"purpose": f"Check policy on {query}", "query": query})


def record(cid: str, status: str, evidence: list[str] | None = None) -> Call:
    return (
        "record_criterion",
        {
            "purpose": f"Record {cid}",
            "criterion_id": cid,
            "status": status,
            "evidence_ids": evidence or [],
            "rationale": f"{cid} assessed from the record.",
        },
    )


def finish(summary: str = "All criteria were assessed against the cited evidence.") -> Call:
    return ("finish_review", {"summary": summary, "cited_section_ids": ["$section:0"]})


# A complete, valid run for the lumbar MRI policy (case PA-001).
MRI_APPROVAL_SCRIPT: list[list[Call]] = [
    [search("bowel bladder weakness fever cancer"), guideline("red flag findings")],
    [record("RF1", "not_met", ["$hit:0"]), search("weeks of low back pain")],
    [record("C1", "met", ["$hit:0"]), search("physical therapy naproxen")],
    [record("C2", "met", ["$hit:0"]), search("epidural steroid injection spine surgery")],
    [record("C3", "met", ["$hit:0"]), finish()],
]


def load_case(case_id: str) -> dict[str, Any]:
    return json.loads((ROOT / "evals" / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider="gemini",
        llm_model="scripted",
        llm_api_key="test",  # type: ignore[arg-type]
        qdrant_url=":memory:",
        embedding_backend="hashing",
        policies_dir=ROOT / "data" / "policies",
        examples_dir=ROOT / "evals" / "cases",
        database_path=tmp_path / "reviews.db",
        max_agent_turns=8,
    )


@pytest.fixture
def policies(settings: Settings) -> PolicyRepository:
    return PolicyRepository.from_directory(settings.policies_dir)


@pytest.fixture
def make_service(settings: Settings) -> Callable[[ScriptedChatModel], ReviewService]:
    def _make(llm: ScriptedChatModel) -> ReviewService:
        service = ReviewService(
            settings,
            llm=llm,
            embedder=HashingEmbedder(),  # type: ignore[arg-type]
            store=ReviewStore(settings.database_path),
        )
        service.start()
        return service

    return _make


@pytest.fixture
def make_client(
    make_service: Callable[[ScriptedChatModel], ReviewService],
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(llm: ScriptedChatModel) -> TestClient:
        client = TestClient(create_app(make_service(llm)))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
