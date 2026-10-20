"""Deterministic decision gate. The LLM assesses criteria; this code turns them into a recommendation.

Design rule: the system can recommend approval, ask for more information, or route the case to a
physician. It never issues a denial. Adverse determinations stay with a licensed clinician.
"""

from datetime import datetime, timedelta

from .policies import LogicNode, Policy
from .schemas import CriterionStatus, DiagnosisCheck, PlanId, Recommendation, Urgency

# Decision timeframes for impacted payers under CMS-0057-F (in force from 2026).
DECISION_TIMEFRAMES: dict[Urgency, timedelta] = {
    "expedited": timedelta(hours=72),
    "standard": timedelta(days=7),
}


def decision_due_by(received_at: datetime, urgency: Urgency) -> datetime:
    return received_at + DECISION_TIMEFRAMES[urgency]


def evaluate_logic(node: LogicNode, statuses: dict[str, CriterionStatus]) -> CriterionStatus:
    """Three-valued (Kleene) evaluation: unknown evidence propagates instead of defaulting to 'not met'."""
    if isinstance(node, str):
        return statuses.get(node, "insufficient_evidence")
    ((op, children),) = node.items()
    results = [evaluate_logic(child, statuses) for child in children]
    if op == "all_of":
        if "not_met" in results:
            return "not_met"
        return "insufficient_evidence" if "insufficient_evidence" in results else "met"
    if "met" in results:  # any_of
        return "met"
    return "insufficient_evidence" if "insufficient_evidence" in results else "not_met"


def check_diagnoses(policy: Policy, codes: list[str]) -> list[DiagnosisCheck]:
    return [
        DiagnosisCheck(
            code=c, supported=c in policy.supported_diagnoses, description=policy.supported_diagnoses.get(c)
        )
        for c in codes
    ]


def plan_applies(policy: Policy, plan_id: PlanId) -> bool:
    return plan_id in policy.applies_to_plans


def decide(
    policy: Policy | None,
    plan_id: PlanId,
    diagnosis_checks: list[DiagnosisCheck],
    statuses: dict[str, CriterionStatus],
    agent_completed: bool,
) -> tuple[Recommendation, list[str]]:
    """Return the recommendation and the human-readable reasons behind it."""
    if policy is None:
        return "refer_to_physician_review", ["No medical policy covers the requested service."]
    if not plan_applies(policy, plan_id):
        return "refer_to_physician_review", [
            f"Policy {policy.policy_id} does not apply to plan '{plan_id}'; benefit review needed."
        ]

    reasons: list[str] = []
    if not any(d.supported for d in diagnosis_checks):
        reasons.append(f"None of the submitted diagnosis codes is listed in policy {policy.policy_id}.")
    if not agent_completed:
        reasons.append("The automated review did not finish; unassessed criteria count as missing evidence.")

    outcome = evaluate_logic(policy.logic, statuses)
    logic_text = policy.describe_logic()
    if outcome == "met":
        reasons.insert(0, f"Policy criteria satisfied ({logic_text}).")
    elif outcome == "insufficient_evidence":
        reasons.insert(0, f"The record lacks evidence for criteria needed to satisfy {logic_text}.")
    else:
        reasons.insert(0, f"The documented evidence does not satisfy {logic_text}.")

    if outcome == "met" and len(reasons) == 1:
        return "recommend_approval", reasons
    if outcome == "insufficient_evidence" and agent_completed and len(reasons) == 1:
        return "request_more_information", reasons
    return "refer_to_physician_review", reasons


def missing_information(policy: Policy, statuses: dict[str, CriterionStatus]) -> list[str]:
    return [
        f"{c.id}: {c.text}"
        for c in policy.criteria
        if statuses.get(c.id, "insufficient_evidence") == "insufficient_evidence"
    ]
