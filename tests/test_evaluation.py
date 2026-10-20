from collections.abc import Callable

from prior_auth.config import Settings
from prior_auth.evaluation import load_cases, raw_identifiers, score, to_markdown
from prior_auth.schemas import ReviewRequest
from prior_auth.service import ReviewService
from tests.conftest import MRI_APPROVAL_SCRIPT, ROOT, ScriptedChatModel


def test_raw_identifiers_come_from_the_phi_detectors() -> None:
    (case,) = load_cases(ROOT / "evals" / "cases", {"PA-001"})
    assert {"Daniel", "Okafor", "00482913", "Hamid"} <= raw_identifiers(case["request"])


def test_scoring_on_a_scripted_run(
    make_service: Callable[[ScriptedChatModel], ReviewService], settings: Settings
) -> None:
    (case,) = load_cases(ROOT / "evals" / "cases", {"PA-001"})
    outcome = make_service(ScriptedChatModel(MRI_APPROVAL_SCRIPT)).review(
        ReviewRequest.model_validate(case["request"])
    )
    run = {
        "case_id": "PA-001",
        "result": outcome.result.model_dump(mode="json"),
        "trace": outcome.trace,
        "phi_leaks": [],
    }
    failed = {"case_id": "PA-002", "error": "RateLimitError"}

    metrics = score([case, case], [run, failed])
    assert metrics["completed_runs"] == 1
    assert metrics["recommendation_accuracy"] == 1.0
    assert metrics["criterion_accuracy"] == 1.0
    assert metrics["false_approvals"] == 0
    assert "| PA-001 | recommend_approval | recommend_approval |" in to_markdown(
        settings, metrics, [case, case], [run, failed]
    )
