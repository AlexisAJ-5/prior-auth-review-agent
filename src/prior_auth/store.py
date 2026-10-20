"""SQLite persistence for review results and traces. Only redacted content is ever stored."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

_TABLES = ("reviews", "traces")


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            for table in _TABLES:
                db.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} (review_id TEXT PRIMARY KEY, body TEXT NOT NULL)"
                )

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call is safe across FastAPI's worker threads.
        return sqlite3.connect(self.path, timeout=10)

    def _put(self, table: str, review_id: str, body: dict[str, Any]) -> None:
        assert table in _TABLES
        with closing(self._connect()) as db, db:
            db.execute(
                f"INSERT OR REPLACE INTO {table} (review_id, body) VALUES (?, ?)",  # noqa: S608
                (review_id, json.dumps(body, default=str)),
            )

    def _get(self, table: str, review_id: str) -> dict[str, Any] | None:
        assert table in _TABLES
        with closing(self._connect()) as db:
            row = db.execute(f"SELECT body FROM {table} WHERE review_id = ?", (review_id,)).fetchone()  # noqa: S608
        return json.loads(row[0]) if row else None

    def save_review(self, review_id: str, body: dict[str, Any]) -> None:
        self._put("reviews", review_id, body)

    def save_trace(self, review_id: str, body: dict[str, Any]) -> None:
        self._put("traces", review_id, body)

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        return self._get("reviews", review_id)

    def get_trace(self, review_id: str) -> dict[str, Any] | None:
        return self._get("traces", review_id)
