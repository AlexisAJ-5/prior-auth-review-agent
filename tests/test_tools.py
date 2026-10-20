import pytest

from prior_auth.determination import check_diagnoses
from prior_auth.embeddings import HashingEmbedder
from prior_auth.knowledge_base import GuidelineIndex
from prior_auth.policies import PolicyRepository
from prior_auth.record import ClinicalRecord, split_sentences
from prior_auth.tools import ReviewContext, ToolBox

NOTES = (
    "Low back pain for 9 weeks. Completed 6 weeks of physical therapy without relief. "
    "No bowel or bladder changes. Spine surgeon plans an epidural steroid injection."
)


@pytest.fixture
def toolbox(policies: PolicyRepository) -> ToolBox:
    index = GuidelineIndex(":memory:", "test", HashingEmbedder())
    index.ingest(policies.all())
    return ToolBox(index)


@pytest.fixture
def ctx(policies: PolicyRepository) -> ReviewContext:
    policy = policies.get("mri_lumbar_spine")
    assert policy is not None
    return ReviewContext(
        policy=policy,
        record=ClinicalRecord(NOTES, "MRI REPORT. Disc bulge at L4-L5.", HashingEmbedder()),
        diagnosis_checks=check_diagnoses(policy, ["M54.42"]),
        plan_applies=True,
        phi_counts={},
    )


def test_sentences_get_stable_ids() -> None:
    record = ClinicalRecord(NOTES, "MRI REPORT. Disc bulge.", HashingEmbedder())
    assert [s.id for s in record.sentences] == ["N1", "N2", "N3", "N4", "D1", "D2"]
    assert split_sentences("Dr. [NAME] saw her.\nSecond line.") == ["Dr. [NAME] saw her.", "Second line."]


def test_record_search_ranks_the_relevant_sentence_first(toolbox: ToolBox, ctx: ReviewContext) -> None:
    _, _, out = toolbox.execute(ctx, "search_clinical_record", {"purpose": "p", "query": "physical therapy"})
    assert out["sentences"][0]["id"] == "N2"
    assert "N2" in ctx.seen_sentence_ids


def test_guideline_search_is_scoped_to_the_policy(toolbox: ToolBox, ctx: ReviewContext) -> None:
    _, _, out = toolbox.execute(
        ctx, "search_policy_guidelines", {"purpose": "p", "query": "conservative therapy"}
    )
    assert out["sections"]
    assert all(s["section_id"].startswith("MP-RAD-101") for s in out["sections"])


def test_record_criterion_rejects_unseen_evidence(toolbox: ToolBox, ctx: ReviewContext) -> None:
    _, _, out = toolbox.execute(
        ctx,
        "record_criterion",
        {
            "purpose": "p",
            "criterion_id": "C2",
            "status": "met",
            "evidence_ids": ["N2"],
            "rationale": "PT for 6 weeks.",
        },
    )
    assert "not returned by search_clinical_record" in out["error"]
    assert "C2" not in ctx.assessments


def test_record_criterion_requires_evidence_unless_insufficient(toolbox: ToolBox, ctx: ReviewContext) -> None:
    _, _, out = toolbox.execute(
        ctx,
        "record_criterion",
        {"purpose": "p", "criterion_id": "C3", "status": "met", "rationale": "Injection planned."},
    )
    assert "needs at least one evidence id" in out["error"]
    _, _, out = toolbox.execute(
        ctx,
        "record_criterion",
        {
            "purpose": "p",
            "criterion_id": "C3",
            "status": "insufficient_evidence",
            "rationale": "Not documented.",
        },
    )
    assert out["remaining_criteria"] == ["RF1", "C1", "C2"]


def test_unknown_criterion_and_bad_arguments_return_errors(toolbox: ToolBox, ctx: ReviewContext) -> None:
    assert (
        "Unknown criterion"
        in toolbox.execute(
            ctx,
            "record_criterion",
            {
                "purpose": "p",
                "criterion_id": "C9",
                "status": "met",
                "evidence_ids": [],
                "rationale": "x" * 12,
            },
        )[2]["error"]
    )
    assert (
        "Invalid arguments"
        in toolbox.execute(ctx, "calculate_bmi", {"purpose": "p", "weight_kg": 70})[2]["error"]
    )
    assert "Unknown tool" in toolbox.execute(ctx, "approve", {})[2]["error"]


def test_finish_requires_all_criteria_and_retrieved_sections(toolbox: ToolBox, ctx: ReviewContext) -> None:
    out = toolbox.execute(
        ctx, "finish_review", {"summary": "Summary of the review.", "cited_section_ids": ["x"]}
    )[2]
    assert "Missing: RF1, C1, C2, C3" in out["error"]
    for cid in ("RF1", "C1", "C2", "C3"):
        toolbox.execute(
            ctx,
            "record_criterion",
            {
                "purpose": "p",
                "criterion_id": cid,
                "status": "insufficient_evidence",
                "rationale": "Not documented.",
            },
        )
    out = toolbox.execute(
        ctx, "finish_review", {"summary": "Summary of the review.", "cited_section_ids": ["MP-RAD-101 §1"]}
    )[2]
    assert "were not returned by search_policy_guidelines" in out["error"]
    assert not ctx.finished


def test_calculate_bmi(toolbox: ToolBox, ctx: ReviewContext) -> None:
    out = toolbox.execute(ctx, "calculate_bmi", {"purpose": "p", "weight_kg": 98, "height_m": 1.69})[2]
    assert out == {"bmi": 34.3, "unit": "kg/m2"}


def test_not_met_with_a_contradicting_fact_is_accepted(toolbox: ToolBox, ctx: ReviewContext) -> None:
    toolbox.execute(ctx, "search_clinical_record", {"purpose": "p", "query": "bowel bladder"})
    out = toolbox.execute(
        ctx,
        "record_criterion",
        {
            "purpose": "p",
            "criterion_id": "RF1",
            "status": "not_met",
            "evidence_ids": ["N3"],
            "rationale": "The note states there are no bowel or bladder changes.",
        },
    )[2]
    assert out["status"] == "not_met"
