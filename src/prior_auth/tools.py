"""The reviewer agent's tools and the per-review context they act on.

Every tool except `finish_review` takes a `purpose` argument: one sentence, written for the human
reviewer, saying why the call is made. It is logged in the trace, so the trace explains each action
without exposing the model's private reasoning.

Tools validate their arguments and return {"error": ...} instead of raising, so the agent observes
its mistake and can correct it on the next turn.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .knowledge_base import GuidelineIndex
from .policies import Policy
from .record import ClinicalRecord
from .schemas import CriterionAssessment, CriterionStatus, DiagnosisCheck

PURPOSE = Field(description="One sentence for the human reviewer explaining why you are making this call.")


class SearchClinicalRecordArgs(BaseModel):
    purpose: str = PURPOSE
    query: str = Field(
        min_length=3, description="What to look for in the clinical record, in plain clinical words."
    )


class SearchPolicyGuidelinesArgs(BaseModel):
    purpose: str = PURPOSE
    query: str = Field(
        min_length=3, description="The policy question, e.g. 'what counts as conservative therapy'."
    )


class CalculateBmiArgs(BaseModel):
    purpose: str = PURPOSE
    weight_kg: float = Field(gt=20, lt=400)
    height_m: float = Field(gt=1.0, lt=2.5)


class RecordCriterionArgs(BaseModel):
    purpose: str = PURPOSE
    criterion_id: str
    status: CriterionStatus = Field(
        description=(
            "met / not_met need a cited sentence that states the fact. If the record is silent or too "
            "vague on what the criterion needs, use insufficient_evidence, not not_met."
        )
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Sentence ids (N1, D3...) returned by search_clinical_record."
    )
    rationale: str = Field(
        min_length=10, max_length=600, description="Neutral, factual, one or two sentences."
    )


class FinishReviewArgs(BaseModel):
    summary: str = Field(
        min_length=20,
        max_length=1200,
        description="Neutral summary for the clinical reviewer. No decision wording, no treatment advice.",
    )
    cited_section_ids: list[str] = Field(
        min_length=1,
        description="Policy section ids returned by search_policy_guidelines that support the review.",
    )


FINAL_TOOL = "finish_review"

ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_clinical_record": SearchClinicalRecordArgs,
    "search_policy_guidelines": SearchPolicyGuidelinesArgs,
    "calculate_bmi": CalculateBmiArgs,
    "record_criterion": RecordCriterionArgs,
    FINAL_TOOL: FinishReviewArgs,
}

DESCRIPTIONS = {
    "search_clinical_record": "Hybrid semantic + keyword search over the redacted clinical record. "
    "Returns the best-matching sentences with their ids for citation.",
    "search_policy_guidelines": "Semantic search over the sections of the governing medical policy. "
    "Use it to check definitions and thresholds before assessing a criterion.",
    "calculate_bmi": "Compute body mass index from weight in kilograms and height in metres. "
    "Use it instead of doing arithmetic yourself.",
    "record_criterion": "Record your assessment of one policy criterion with cited evidence. "
    "Call again to revise. Every criterion must be recorded before finishing.",
    FINAL_TOOL: "Finish the review once every criterion is recorded. The recommendation itself is computed "
    "by deterministic rules from your criterion assessments.",
}


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI function-calling definitions, generated from the Pydantic argument models."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in ARG_MODELS.items()
    ]


@dataclass
class ReviewContext:
    """Everything one review needs. Holds redacted text only."""

    policy: Policy | None
    record: ClinicalRecord
    diagnosis_checks: list[DiagnosisCheck]
    plan_applies: bool
    phi_counts: dict[str, int]
    seen_sentence_ids: set[str] = field(default_factory=set)
    seen_sections: dict[str, str] = field(default_factory=dict)  # section_id -> heading
    assessments: dict[str, CriterionAssessment] = field(default_factory=dict)
    summary: str | None = None
    cited_section_ids: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.summary is not None

    def statuses(self) -> dict[str, CriterionStatus]:
        return {cid: a.status for cid, a in self.assessments.items()}


class ToolBox:
    def __init__(self, guidelines: GuidelineIndex, record_k: int = 4, guideline_k: int = 3):
        self.guidelines = guidelines
        self.record_k = record_k
        self.guideline_k = guideline_k

    def execute(
        self, ctx: ReviewContext, name: str, raw: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Run one call. Returns (purpose, normalized input, observation)."""
        purpose = str(raw.get("purpose") or f"Call {name}").strip()
        model = ARG_MODELS.get(name)
        if model is None:
            return (
                purpose,
                {"tool": name},
                {"error": f"Unknown tool '{name}'. Available: {', '.join(ARG_MODELS)}"},
            )
        try:
            args = model.model_validate(raw)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return purpose, _without_purpose(raw), {"error": f"Invalid arguments: {detail}"}
        handler = getattr(self, f"_{name}")
        tool_input = _without_purpose(args.model_dump())
        return purpose, tool_input, handler(ctx, args)

    # --- handlers -------------------------------------------------------------------------------

    def _search_clinical_record(self, ctx: ReviewContext, args: SearchClinicalRecordArgs) -> dict[str, Any]:
        hits = ctx.record.search(args.query, k=self.record_k)
        ctx.seen_sentence_ids.update(h.sentence.id for h in hits)
        return {
            "sentences": [
                {"id": h.sentence.id, "source": h.sentence.source, "text": h.sentence.text, "score": h.score}
                for h in hits
            ]
        }

    def _search_policy_guidelines(
        self, ctx: ReviewContext, args: SearchPolicyGuidelinesArgs
    ) -> dict[str, Any]:
        policy_id = ctx.policy.policy_id if ctx.policy else None
        sections = self.guidelines.search(args.query, policy_id=policy_id, limit=self.guideline_k)
        ctx.seen_sections.update({s["section_id"]: s["heading"] for s in sections})
        return {"sections": sections}

    def _calculate_bmi(self, ctx: ReviewContext, args: CalculateBmiArgs) -> dict[str, Any]:
        return {"bmi": round(args.weight_kg / args.height_m**2, 1), "unit": "kg/m2"}

    def _record_criterion(self, ctx: ReviewContext, args: RecordCriterionArgs) -> dict[str, Any]:
        if ctx.policy is None or ctx.policy.criterion(args.criterion_id) is None:
            known = ", ".join(c.id for c in ctx.policy.criteria) if ctx.policy else "none"
            return {"error": f"Unknown criterion '{args.criterion_id}'. Policy criteria: {known}"}
        unseen = [i for i in args.evidence_ids if i not in ctx.seen_sentence_ids]
        if unseen:
            return {
                "error": f"Evidence ids {unseen} were not returned by search_clinical_record. "
                "Search the record first and cite only returned sentence ids."
            }
        if args.status != "insufficient_evidence" and not args.evidence_ids:
            return {
                "error": f"Status '{args.status}' needs at least one evidence id. "
                "If the record is silent, use insufficient_evidence."
            }
        ctx.assessments[args.criterion_id] = CriterionAssessment(
            criterion_id=args.criterion_id,
            status=args.status,
            evidence_ids=list(dict.fromkeys(args.evidence_ids)),
            rationale=args.rationale.strip(),
        )
        remaining = [c.id for c in ctx.policy.criteria if c.id not in ctx.assessments]
        return {"recorded": args.criterion_id, "status": args.status, "remaining_criteria": remaining}

    def _finish_review(self, ctx: ReviewContext, args: FinishReviewArgs) -> dict[str, Any]:
        if ctx.policy is None:
            return {"error": "No policy is loaded for this review."}
        missing = [c.id for c in ctx.policy.criteria if c.id not in ctx.assessments]
        if missing:
            return {"error": f"Record every criterion before finishing. Missing: {', '.join(missing)}"}
        uncited = [s for s in args.cited_section_ids if s not in ctx.seen_sections]
        if uncited:
            return {"error": f"Sections {uncited} were not returned by search_policy_guidelines."}
        ctx.summary = args.summary.strip()
        ctx.cited_section_ids = list(dict.fromkeys(args.cited_section_ids))
        return {"finished": True}


def _without_purpose(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if k != "purpose"}
