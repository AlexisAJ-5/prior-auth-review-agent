import pytest

from prior_auth.guard import apply_guard, violations
from prior_auth.schemas import CriterionResult


@pytest.mark.parametrize(
    ("text", "issue"),
    [
        ("The request is denied because PT was not completed.", "adverse decision wording"),
        ("This imaging is not medically necessary.", "adverse decision wording"),
        ("We recommend denial of the service.", "adverse decision wording"),
        ("The service will be approved once received.", "approval promise"),
        ("Coverage is guaranteed.", "approval promise"),
        ("The patient should start metformin first.", "treatment advice"),
        ("Call the member at 212-555-0187.", "possible PHI (PHONE)"),
    ],
)
def test_violations_are_detected(text: str, issue: str) -> None:
    assert issue in violations(text)


@pytest.mark.parametrize(
    "text",
    [
        "She denies locking, catching or giving way (N3).",  # clinical "denies" is not a decision
        "Criteria C1 and C2 are met; C3 lacks documentation of conservative therapy.",
        "Patient: [NAME] completed 6 weeks of physical therapy.",
    ],
)
def test_clinical_language_is_not_flagged(text: str) -> None:
    assert violations(text) == []


def _criterion(rationale: str) -> CriterionResult:
    return CriterionResult(
        criterion_id="C1", criterion_text="t", status="met", rationale=rationale, evidence=[]
    )


def test_failing_text_is_replaced_with_a_template() -> None:
    summary, criteria, report = apply_guard("The request is denied.", [_criterion("Call 212-555-0187.")])
    assert summary.startswith("Criteria assessment: C1 met")
    assert "212-555-0187" not in criteria[0].rationale
    assert not report.passed
    assert report.replaced_fields == ["criteria.C1.rationale", "reviewer_summary"]


def test_clean_text_passes_unchanged() -> None:
    summary, criteria, report = apply_guard("All criteria documented.", [_criterion("PT for 6 weeks (N2).")])
    assert (summary, criteria[0].rationale, report.passed) == (
        "All criteria documented.",
        "PT for 6 weeks (N2).",
        True,
    )
