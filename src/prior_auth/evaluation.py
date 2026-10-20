"""Offline evaluation on the labelled synthetic cases.

    uv run prior-auth-eval                       # all cases, model from .env
    uv run prior-auth-eval --only PA-001,PA-007  # a subset
    uv run prior-auth-eval --delay 4             # pause between cases (free-tier rate limits)

Writes evals/results/latest.json and evals/results/latest.md.
"""

import argparse
import json
import logging
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Settings
from .llm import PROVIDER_ERRORS, is_quota_exhausted, is_transient
from .phi import DETECTORS
from .schemas import ReviewRequest
from .service import ReviewOutcome, ReviewService
from .store import ReviewStore

RECOMMENDATIONS = ["recommend_approval", "request_more_information", "refer_to_physician_review"]


def load_cases(directory: Path, only: set[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]
    return [c for c in cases if not only or c["case_id"] in only]


def raw_identifiers(request: dict[str, Any]) -> set[str]:
    """Identifier strings present in the raw input, used to prove none of them leaks into outputs."""
    text = request["clinical_notes"] + "\n" + (request.get("document_text") or "")
    values: set[str] = set()
    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            values.update(tok for tok in m.group(det.group).split() if len(tok) > 3)
    return values


def score(cases: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: Counter[tuple[str, str]] = Counter()
    crit_total = crit_correct = 0
    per_status: dict[str, Counter[str]] = {}
    for case, run in zip(cases, runs, strict=True):
        if "error" in run:
            continue
        expected, got = case["expected"]["recommendation"], run["result"]["recommendation"]
        confusion[(expected, got)] += 1
        predicted = {c["criterion_id"]: c["status"] for c in run["result"]["criteria"]}
        for cid, status in case["expected"]["criteria"].items():
            crit_total += 1
            crit_correct += predicted.get(cid) == status
            per_status.setdefault(status, Counter())[predicted.get(cid, "missing")] += 1

    ok = [r for r in runs if "error" not in r]
    reviewed = [r for r in ok if r["result"]["criteria"]]
    correct = sum(n for (e, g), n in confusion.items() if e == g)
    return {
        "cases": len(cases),
        "completed_runs": len(ok),
        "recommendation_accuracy": round(correct / len(ok), 3) if ok else None,
        "criterion_accuracy": round(crit_correct / crit_total, 3) if crit_total else None,
        "false_approvals": sum(n for (e, g), n in confusion.items() if g == "recommend_approval" != e),
        "missed_approvals": sum(n for (e, g), n in confusion.items() if e == "recommend_approval" != g),
        "status_recall": {s: round(c[s] / sum(c.values()), 3) for s, c in per_status.items()},
        "confusion": {f"{e} -> {g}": n for (e, g), n in sorted(confusion.items())},
        "phi_leaks": sum(len(r["phi_leaks"]) for r in ok),
        "guard_replacements": sum(len(r["result"]["guard"]["replaced_fields"]) for r in ok),
        # Only runs where the agent had criteria to assess (a plan outside the policy skips it).
        "agent_completed_rate": round(
            sum(r["result"]["agent_completed"] for r in reviewed) / len(reviewed), 3
        )
        if reviewed
        else None,
        "median_latency_s": round(statistics.median(r["trace"]["duration_ms"] for r in ok) / 1000, 1)
        if ok
        else None,
        "mean_tool_calls": round(statistics.mean(r["trace"]["tool_calls"] for r in ok), 1) if ok else None,
        "mean_input_tokens": round(statistics.mean(r["trace"]["tokens"]["input_tokens"] for r in ok))
        if ok
        else None,
    }


def to_markdown(
    settings: Settings, metrics: dict[str, Any], cases: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> str:
    lines = [
        f"# Evaluation: {settings.llm_provider} / {settings.llm_model}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in metrics.items() if not isinstance(v, dict)],
        "",
        "| Case | Expected | Got | Criteria (expected / got) |",
        "|---|---|---|---|",
    ]
    for case, run in zip(cases, runs, strict=True):
        if "error" in run:
            lines.append(
                f"| {case['case_id']} | {case['expected']['recommendation']} | ERROR: {run['error']} | |"
            )
            continue
        got = {c["criterion_id"]: c["status"] for c in run["result"]["criteria"]}
        crit = ", ".join(f"{cid} {s}/{got.get(cid, '-')}" for cid, s in case["expected"]["criteria"].items())
        lines.append(
            f"| {case['case_id']} | {case['expected']['recommendation']} | "
            f"{run['result']['recommendation']} | {crit} |"
        )
    return "\n".join(lines) + "\n"


def review_with_retries(
    service: ReviewService, case: dict[str, Any], retries: int, wait_s: float
) -> ReviewOutcome:
    """Free tiers return 429/503 under load; retry those with exponential backoff, nothing else."""
    request = ReviewRequest.model_validate(case["request"])
    for attempt in range(retries + 1):
        try:
            return service.review(request)
        except PROVIDER_ERRORS as exc:
            if attempt == retries or not is_transient(exc) or is_quota_exhausted(exc):
                raise
            pause = wait_s * 2**attempt
            print(f"{case['case_id']}: provider busy ({type(exc).__name__}), retrying in {pause:.0f}s")
            time.sleep(pause)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", type=Path, default=Path("evals/cases"))
    parser.add_argument("--only", default="", help="Comma-separated case ids")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between cases")
    parser.add_argument("--out", type=Path, default=Path("evals/results"))
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries per case on transient provider errors"
    )
    parser.add_argument(
        "--retry-wait", type=float, default=30.0, help="Seconds before the first retry (doubles)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = Settings(qdrant_url=":memory:")
    if not settings.llm_configured:
        raise SystemExit("No LLM configured: set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in .env")
    service = ReviewService(settings, store=ReviewStore(args.out / "eval.db"))
    service.start()

    cases = load_cases(args.cases, set(filter(None, args.only.split(","))) or None)
    runs: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        if i and args.delay:
            time.sleep(args.delay)
        try:
            outcome = review_with_retries(service, case, args.retries, args.retry_wait)
        except Exception as exc:  # one failing case must not stop the evaluation
            runs.append({"case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"{case['case_id']}: ERROR {str(exc)[:300]}")
            if is_quota_exhausted(exc):
                print("Daily model quota exhausted: stopping. Re-run after the quota resets.")
                cases = cases[: len(runs)]
                break
            continue
        result = outcome.result.model_dump(mode="json")
        dumped = json.dumps(result) + json.dumps(outcome.trace)
        leaks = sorted(v for v in raw_identifiers(case["request"]) if v in dumped)
        runs.append(
            {"case_id": case["case_id"], "result": result, "trace": outcome.trace, "phi_leaks": leaks}
        )
        print(
            f"{case['case_id']}: expected {case['expected']['recommendation']}, "
            f"got {result['recommendation']}"
        )

    metrics = score(cases, runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latest.json").write_text(
        json.dumps({"model": settings.llm_model, "metrics": metrics, "runs": runs}, indent=2, default=str),
        encoding="utf-8",
    )
    (args.out / "latest.md").write_text(to_markdown(settings, metrics, cases, runs), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
