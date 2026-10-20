from collections.abc import Callable
from io import BytesIO
from typing import Any

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF
from pydantic import SecretStr

from tests.conftest import MRI_APPROVAL_SCRIPT, ScriptedChatModel, load_case

MakeClient = Callable[[ScriptedChatModel], TestClient]


def _form(case_id: str = "PA-001", **overrides: str) -> dict[str, Any]:
    r = load_case(case_id)["request"]
    form = {
        "plan_id": r["plan_id"],
        "service_code": r["service_code"],
        "urgency": r["urgency"],
        "diagnosis_codes": ",".join(r["diagnosis_codes"]),
        "clinical_notes": r["clinical_notes"],
    }
    return {**form, **overrides}


def _pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


@pytest.fixture
def client(make_client: MakeClient) -> TestClient:
    return make_client(ScriptedChatModel(MRI_APPROVAL_SCRIPT))


def test_create_then_fetch_review_and_trace(client: TestClient) -> None:
    created = client.post("/v1/reviews", data=_form())
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["recommendation"] == "recommend_approval"

    fetched = client.get(f"/v1/reviews/{body['review_id']}")
    assert fetched.status_code == 200 and fetched.json()["criteria"] == body["criteria"]
    trace = client.get(f"/v1/reviews/{body['review_id']}/trace").json()
    assert trace["tool_calls"] >= 5 and trace["events"][0]["node"] == "intake"


def test_pdf_text_becomes_citable_document_sentences(client: TestClient) -> None:
    pdf = _pdf("SPINE CLINIC LETTER. The surgeon will perform an epidural steroid injection at L5-S1.")
    response = client.post(
        "/v1/reviews", data=_form(), files={"document": ("letter.pdf", BytesIO(pdf), "application/pdf")}
    )
    assert response.status_code == 201, response.text
    trace = client.get(f"/v1/reviews/{response.json()['review_id']}/trace").json()
    sources = {
        s["source"]
        for e in trace["events"]
        if e.get("tool") == "search_clinical_record"
        for s in e["observation"]["sentences"]
    }
    assert "document" in sources


@pytest.mark.parametrize(
    ("overrides", "files", "status"),
    [
        ({"service_code": "teleportation"}, None, 422),
        ({"diagnosis_codes": "not-a-code"}, None, 422),
        ({"clinical_notes": "too short"}, None, 422),
        ({"plan_id": "platinum"}, None, 422),
        ({}, {"document": ("x.pdf", BytesIO(b"plain text, not a pdf"), "application/pdf")}, 400),
    ],
)
def test_invalid_input_is_rejected(
    client: TestClient, overrides: dict[str, str], files: Any, status: int
) -> None:
    assert client.post("/v1/reviews", data=_form(**overrides), files=files).status_code == status


def test_unknown_review_is_404(client: TestClient) -> None:
    assert client.get("/v1/reviews/PAR-NOPE").status_code == 404
    assert client.get("/v1/reviews/PAR-NOPE/trace").status_code == 404


def test_ops_and_catalog_endpoints(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["guideline_sections"] > 0
    assert {p["policy_id"] for p in client.get("/v1/policies").json()} >= {"MP-RAD-101", "MP-RX-310"}
    assert len(client.get("/v1/examples").json()) == 16
    assert "Prior authorization pre-review" in client.get("/").text


def test_review_is_refused_without_an_llm(make_client: MakeClient) -> None:
    client = make_client(ScriptedChatModel(MRI_APPROVAL_SCRIPT))
    client.app.state.service.settings.llm_api_key = SecretStr("")  # type: ignore[attr-defined]
    client.app.state.service.settings.llm_provider = "openai"  # type: ignore[attr-defined]
    assert client.post("/v1/reviews", data=_form()).status_code == 503
    assert client.get("/ready").status_code == 503
