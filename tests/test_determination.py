from datetime import UTC, datetime, timedelta

import pytest

from prior_auth.determination import check_diagnoses, decide, decision_due_by, evaluate_logic
from prior_auth.policies import PolicyRepository

M, N, I = "met", "not_met", "insufficient_evidence"
MRI_LOGIC = {"any_of": ["RF1", {"all_of": ["C1", "C2", "C3"]}]}


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({"RF1": M}, M),  # a red flag alone is enough
        ({"RF1": N, "C1": M, "C2": M, "C3": M}, M),
        ({"RF1": N, "C1": M, "C2": I, "C3": M}, I),  # unknown propagates
        ({"RF1": N, "C1": M, "C2": N, "C3": I}, N),  # a failed AND beats an unknown
        ({"RF1": I, "C1": N, "C2": M, "C3": M}, I),  # an unknown red flag keeps the case open
        ({}, I),  # nothing assessed
    ],
)
def test_kleene_logic(statuses: dict[str, str], expected: str) -> None:
    assert evaluate_logic(MRI_LOGIC, statuses) == expected  # type: ignore[arg-type]


def test_decide_outcomes(policies: PolicyRepository) -> None:
    mri = policies.get("mri_lumbar_spine")
    assert mri is not None
    dx_ok = check_diagnoses(mri, ["M54.42"])
    met = {"RF1": N, "C1": M, "C2": M, "C3": M}

    assert decide(mri, "commercial_ppo", dx_ok, met, True)[0] == "recommend_approval"
    assert decide(mri, "commercial_ppo", dx_ok, {**met, "C2": I}, True)[0] == "request_more_information"
    assert decide(mri, "commercial_ppo", dx_ok, {**met, "C2": N}, True)[0] == "refer_to_physician_review"


def test_decide_never_approves_with_unsupported_diagnosis(policies: PolicyRepository) -> None:
    mri = policies.get("mri_lumbar_spine")
    assert mri is not None
    recommendation, reasons = decide(
        mri, "commercial_ppo", check_diagnoses(mri, ["J45.909"]), {"RF1": M}, True
    )
    assert recommendation == "refer_to_physician_review"
    assert any("diagnosis codes" in r for r in reasons)


def test_decide_routes_unfinished_reviews_to_a_physician(policies: PolicyRepository) -> None:
    mri = policies.get("mri_lumbar_spine")
    assert mri is not None
    dx = check_diagnoses(mri, ["M54.42"])
    assert (
        decide(mri, "commercial_ppo", dx, {"C1": I}, agent_completed=False)[0] == "refer_to_physician_review"
    )


def test_decide_checks_plan_applicability(policies: PolicyRepository) -> None:
    glp1 = policies.get("glp1_weight_management")
    assert glp1 is not None
    recommendation, reasons = decide(glp1, "medicare_advantage", [], {}, True)
    assert recommendation == "refer_to_physician_review"
    assert "does not apply" in reasons[0]


def test_decision_timeframes() -> None:
    received = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)
    assert decision_due_by(received, "expedited") - received == timedelta(hours=72)
    assert decision_due_by(received, "standard") - received == timedelta(days=7)
