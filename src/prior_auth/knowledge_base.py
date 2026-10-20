"""Vector index of policy guideline sections in Qdrant (server in Docker, in-memory for tests and evals)."""

import logging
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from .embeddings import Embedder
from .policies import Policy

log = logging.getLogger(__name__)


def section_text(policy: Policy, heading: str, text: str) -> str:
    return f"{policy.title}. {heading}. {text}"


class GuidelineIndex:
    def __init__(self, url: str, collection: str, embedder: Embedder):
        self.client = (
            QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url, timeout=10)
        )
        self.collection = collection
        self.embedder = embedder

    def wait_until_ready(self, timeout_s: float = 60) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.is_ready():
            if time.monotonic() > deadline:
                raise RuntimeError("Qdrant is not reachable")
            log.info("Waiting for Qdrant...")
            time.sleep(2)

    def is_ready(self) -> bool:
        try:
            self.client.get_collections()
        except Exception:
            return False
        return True

    def ingest(self, policies: list[Policy]) -> int:
        """Embed every policy section. Idempotent: point ids derive from the section id."""
        rows = [(p, s) for p in policies for s in p.sections]
        vectors = self.embedder.embed_documents([section_text(p, s.heading, s.text) for p, s in rows])
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.embedder.dimension, distance=Distance.COSINE),
            )
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, s.section_id)),
                vector=v,
                payload={
                    "section_id": s.section_id,
                    "policy_id": p.policy_id,
                    "heading": s.heading,
                    "text": s.text,
                },
            )
            for (p, s), v in zip(rows, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        log.info("Indexed %d guideline sections into '%s'", len(points), self.collection)
        return len(points)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, query: str, policy_id: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        query_filter = (
            Filter(must=[FieldCondition(key="policy_id", match=MatchValue(value=policy_id))])
            if policy_id
            else None
        )
        hits = self.client.query_points(
            self.collection,
            query=self.embedder.embed_query(query),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        return [
            {
                "section_id": h.payload["section_id"],
                "heading": h.payload["heading"],
                "text": h.payload["text"],
                "score": round(float(h.score), 4),
            }
            for h in hits
            if h.payload
        ]
