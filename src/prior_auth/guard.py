"""Output guard: deterministic checks on every free-text field the LLM wrote.

Failing text is replaced by a template built from the structured assessment, so the API never
returns unsafe wording. Rules are deterministic on purpose: they are cheap, testable and cannot be
talked out of their job. An LLM reviewer could be layered on top (see README, next steps).
"""

import re

from .phi import find_phi
from .schemas import CriterionResult, GuardReport

RULES: dict[str, re.Pattern[str]] = {
    "adverse decision wording": re.compile(
        r"\b(?:request|authori[sz]ation|service|coverage|claim)\s+(?:is|was|will be|should be)\s+"
        r"(?:denied|rejected|declined)\b|\b(?:we|I)\s+(?:deny|reject)\b|\brecommend(?:ing)?\s+(?:a\s+)?"
        r"(?:denial|rejection)\b|\bnot medically necessary\b",
        re.I,
    ),
    "approval promise": re.compile(
        r"\bguarantee[ds]?\b|\b(?:is|will be)\s+(?:approved|covered|authori[sz]ed)\b"
        r"|\bfinal (?:decision|determination)\b",
        re.I,
    ),
    "treatment advice": re.compile(
        r"\b(?:patient|member)\s+should\s+(?:take|start|stop|try|increase|decrease|switch)\b"
        r"|\bI (?:would )?(?:recommend|suggest) (?:taking|starting|stopping|switching)\b",
        re.I,
    ),
}


def violations(text: str) -> list[str]:
    issues = [name for name, pattern in RULES.items() if pattern.search(text)]
    issues += [f"possible PHI ({label})" for label in find_phi(text)]
    return issues


def template_summary(criteria: list[CriterionResult]) -> str:
    parts = [f"{c.criterion_id} {c.status.replace('_', ' ')}" for c in criteria]
    return "Criteria assessment: " + "; ".join(parts) + ". See cited evidence for each criterion."


def apply_guard(
    summary: str, criteria: list[CriterionResult]
) -> tuple[str, list[CriterionResult], GuardReport]:
    issues: list[str] = []
    replaced: list[str] = []

    safe_criteria = []
    for c in criteria:
        found = violations(c.rationale)
        if found:
            issues += [f"{c.criterion_id} rationale: {i}" for i in found]
            replaced.append(f"criteria.{c.criterion_id}.rationale")
            c = c.model_copy(
                update={"rationale": "Rationale withheld by the output guard; see the cited evidence."}
            )
        safe_criteria.append(c)

    found = violations(summary)
    if found:
        issues += [f"summary: {i}" for i in found]
        replaced.append("reviewer_summary")
        summary = template_summary(safe_criteria)

    # `passed` reports whether the model's text was clean. The returned text is safe either way,
    # because failing fields were replaced; the issues are kept for monitoring.
    return summary, safe_criteria, GuardReport(passed=not issues, issues=issues, replaced_fields=replaced)
