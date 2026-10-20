"""FastAPI application: REST endpoints and the static review UI."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import __version__
from .config import get_settings
from .documents import DocumentError, extract_pdf_text
from .llm import PROVIDER_ERRORS, is_quota_exhausted
from .schemas import PlanId, ReviewRequest, ReviewResult, Urgency
from .service import ReviewService

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(service: ReviewService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = service or ReviewService(get_settings())
        await run_in_threadpool(svc.start)
        app.state.service = svc
        yield

    app = FastAPI(title="Prior Authorization Review Agent", version=__version__, lifespan=lifespan)

    def svc() -> ReviewService:
        return app.state.service  # type: ignore[no-any-return]

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready() -> dict[str, Any]:
        """Readiness: policies indexed and an LLM configured."""
        s = svc()
        checks = {"guideline_sections": s.guidelines.count(), "llm_configured": s.settings.llm_configured}
        if not checks["guideline_sections"] or not checks["llm_configured"]:
            raise HTTPException(503, detail=checks)
        return {"status": "ready", **checks}

    @app.get("/v1/policies", tags=["policies"])
    def list_policies() -> list[dict[str, Any]]:
        return [
            {
                "policy_id": p.policy_id,
                "title": p.title,
                "service_code": p.service_code,
                "service_name": p.service_name,
                "applies_to_plans": p.applies_to_plans,
                "criteria": [c.model_dump() for c in p.criteria],
                "logic": p.describe_logic(),
            }
            for p in svc().policies.all()
        ]

    @app.get("/v1/examples", tags=["policies"])
    def list_examples() -> list[dict[str, Any]]:
        """Synthetic example requests (the evaluation cases) for the demo UI."""
        directory = svc().settings.examples_dir
        if not directory.is_dir():
            return []
        cases = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(directory.glob("*.json"))]
        return [{"case_id": c["case_id"], "summary": c["summary"], "request": c["request"]} for c in cases]

    @app.post("/v1/reviews", status_code=201, response_model=ReviewResult, tags=["reviews"])
    async def create_review(
        plan_id: Annotated[PlanId, Form()],
        service_code: Annotated[str, Form()],
        diagnosis_codes: Annotated[str, Form(description="Comma-separated ICD-10-CM codes")],
        clinical_notes: Annotated[str, Form()],
        urgency: Annotated[Urgency, Form()] = "standard",
        document_text: Annotated[
            str | None, Form(description="Optional pasted text of a supporting document")
        ] = None,
        document: Annotated[UploadFile | None, File(description="Optional supporting PDF")] = None,
    ) -> ReviewResult:
        s = svc()
        if not s.settings.llm_configured:
            raise HTTPException(503, "No LLM configured. Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY.")
        if s.policies.get(service_code) is None:
            known = ", ".join(p.service_code for p in s.policies.all())
            raise HTTPException(422, f"Unknown service_code '{service_code}'. Known: {known}")

        document_text = (document_text or "").strip() or None
        if document is not None and document.filename:
            data = await document.read(s.settings.max_document_bytes + 1)
            if len(data) > s.settings.max_document_bytes:
                raise HTTPException(413, "Supporting document is too large.")
            try:
                pdf_text = extract_pdf_text(data, s.settings.max_document_chars)
            except DocumentError as exc:
                raise HTTPException(400, str(exc)) from exc
            document_text = f"{document_text}\n{pdf_text}" if document_text else pdf_text
        try:
            request = ReviewRequest(
                plan_id=plan_id,
                service_code=service_code,
                urgency=urgency,
                diagnosis_codes=diagnosis_codes.split(","),
                clinical_notes=clinical_notes,
                document_text=document_text,
            )
        except ValidationError as exc:
            raise HTTPException(
                422, exc.errors(include_url=False, include_input=False, include_context=False)
            ) from exc

        try:
            outcome = await run_in_threadpool(s.review, request)
        except PROVIDER_ERRORS as exc:
            log.warning("LLM provider error: %s", exc)
            if is_quota_exhausted(exc):
                raise HTTPException(429, "The model's daily quota is used up. Try again later.") from exc
            raise HTTPException(
                502, "The LLM provider failed or rate-limited the request. Try again."
            ) from exc
        return outcome.result

    @app.get("/v1/reviews/{review_id}", response_model=ReviewResult, tags=["reviews"])
    def get_review(review_id: str) -> dict[str, Any]:
        body = svc().store.get_review(review_id)
        if body is None:
            raise HTTPException(404, "Review not found")
        return body

    @app.get("/v1/reviews/{review_id}/trace", tags=["reviews"])
    def get_trace(review_id: str) -> dict[str, Any]:
        body = svc().store.get_trace(review_id)
        if body is None:
            raise HTTPException(404, "Trace not found")
        return body

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
