"""Domain models shared by the API, the agent and the evaluation harness."""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PlanId = Literal["commercial_ppo", "commercial_hmo", "medicare_advantage"]
Urgency = Literal["standard", "expedited"]
CriterionStatus = Literal["met", "not_met", "insufficient_evidence"]
Recommendation = Literal["recommend_approval", "request_more_information", "refer_to_physician_review"]
SentenceSource = Literal["notes", "document"]

ICD10_PATTERN = r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$"


class ReviewRequest(BaseModel):
    """A prior authorization request as received. May contain PHI; never persisted as-is."""

    plan_id: PlanId
    service_code: str = Field(min_length=2, max_length=64)
    diagnosis_codes: list[str] = Field(min_length=1, max_length=12)
    urgency: Urgency = "standard"
    clinical_notes: str = Field(min_length=20)
    document_text: str | None = None

    @field_validator("diagnosis_codes")
    @classmethod
    def _normalize_codes(cls, codes: list[str]) -> list[str]:
        cleaned = [c.strip().upper() for c in codes if c.strip()]
        bad = [c for c in cleaned if not re.match(ICD10_PATTERN, c)]
        if bad:
            raise ValueError(f"Not ICD-10-CM formatted: {', '.join(bad)}")
        return list(dict.fromkeys(cleaned))  # de-duplicate, keep order


class Sentence(BaseModel):
    id: str  # N3 = third sentence of the notes, D2 = second sentence of the attached document
    source: SentenceSource
    text: str


class CriterionAssessment(BaseModel):
    criterion_id: str
    status: CriterionStatus
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str


class CriterionResult(BaseModel):
    criterion_id: str
    criterion_text: str
    status: CriterionStatus
    rationale: str
    evidence: list[Sentence]


class DiagnosisCheck(BaseModel):
    code: str
    supported: bool
    description: str | None = None


class CitedSection(BaseModel):
    section_id: str
    heading: str


class GuardReport(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    replaced_fields: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    review_id: str
    received_at: datetime
    decision_due_by: datetime
    plan_id: PlanId
    service_code: str
    urgency: Urgency
    policy_id: str | None
    recommendation: Recommendation
    recommendation_reasons: list[str]
    criteria: list[CriterionResult]
    missing_information: list[str]
    diagnosis_checks: list[DiagnosisCheck]
    cited_sections: list[CitedSection]
    reviewer_summary: str
    phi_redactions: dict[str, int]
    guard: GuardReport
    agent_completed: bool
    human_review_required: bool
    disclaimer: str = (
        "Decision support only. Synthetic policies. A licensed clinician makes every adverse determination."
    )
