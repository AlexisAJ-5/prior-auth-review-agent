"""Consistency checks on the synthetic policies and the labelled evaluation cases."""

import json
from pathlib import Path

import pytest

from prior_auth.determination import check_diagnoses, decide
from prior_auth.policies import PolicyRepository
from prior_auth.schemas import ReviewRequest

CASES = sorted((Path(__file__).resolve().parents[1] / "evals" / "cases").glob("*.json"))


def test_policy_sections_are_unique_and_marked_synthetic(policies: PolicyRepository) -> None:
    section_ids = [s.section_id for p in policies.all() for s in p.sections]
    assert len(section_ids) == len(set(section_ids))
    assert all(p.synthetic for p in policies.all())


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_labels_agree_with_the_decision_rules(path: Path, policies: PolicyRepository) -> None:
    """A label set is only useful if the expected recommendation follows from the expected criteria."""
    case = json.loads(path.read_text(encoding="utf-8"))
    request = ReviewRequest.model_validate(case["request"])
    policy = policies.get(request.service_code)
    assert policy is not None
    expected = case["expected"]
    if expected["criteria"]:
        assert set(expected["criteria"]) == {c.id for c in policy.criteria}
    recommendation, _ = decide(
        policy,
        request.plan_id,
        check_diagnoses(policy, request.diagnosis_codes),
        expected["criteria"],
        agent_completed=True,
    )
    assert recommendation == expected["recommendation"]


def test_cases_cover_every_policy_and_outcome(policies: PolicyRepository) -> None:
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in CASES]
    assert {c["request"]["service_code"] for c in cases} == {p.service_code for p in policies.all()}
    assert {c["expected"]["recommendation"] for c in cases} == {
        "recommend_approval",
        "request_more_information",
        "refer_to_physician_review",
    }
