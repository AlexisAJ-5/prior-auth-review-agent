# CLAUDE.md

Guidance for AI coding agents (Claude Code and similar) working in this repository. Humans can read
it as the short version of the contributor guide.

## Commands

```bash
uv sync                                  # install (Python 3.12+)
uv run pytest                            # offline tests, must stay green and network-free
uv run ruff check . && uv run ruff format --check .
uv run mypy                              # strict mode, no new `type: ignore` without a reason
uv run uvicorn prior_auth.api:app --reload
uv run prior-auth-eval --only PA-001     # live evaluation, needs an LLM in .env
```

Run `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest` before proposing
a change.

## Architecture in one paragraph

`api.py` receives a request and calls `ReviewService.review` (`service.py`). The LangGraph graph in
`graph.py` runs `intake` (PHI redaction, policy, plan and diagnosis checks), then the `reviewer`
LLM node and `run_tools` in a loop until `finish_review` succeeds or the turn limit is hit, then the
deterministic `decide` (`determination.py`) and `guard` (`guard.py`) nodes. Tools live in
`tools.py` and act on a per-review `ReviewContext`. The record is in `record.py`, policy retrieval
in `knowledge_base.py`. Read `docs/adr/` before changing any of these boundaries.

## Invariants: do not break these

1. **No automated denials.** `Recommendation` has exactly three values and none is a denial. Do not
   add one, and do not let model text drive the recommendation: it comes only from
   `determination.decide`.
2. **PHI is redacted in `intake`, before any model call.** Nothing downstream may receive
   `ReviewRequest.clinical_notes` or `document_text`. `test_phi_never_reaches_the_model_or_the_outputs`
   guards this; extend it rather than weaken it.
3. **Citations are verified by code.** Evidence ids must come from `search_clinical_record`
   results, cited sections from `search_policy_guidelines` results. Keep these checks in the tools.
4. **Every model-written string goes through `guard.apply_guard`** before it is returned or stored.
5. **Tests are offline.** Use `ScriptedChatModel` and `HashingEmbedder` from `tests/conftest.py`.
   Anything that needs a real model is marked `@pytest.mark.live` and skips without a key.
6. **Data is synthetic.** Never add real patient data or real payer policy text.

## Conventions

- Python 3.12, type hints everywhere, Pydantic models at boundaries, dataclasses inside.
- Tools return `{"error": ...}` instead of raising, so the agent can observe and correct.
- Every tool except `finish_review` has a `purpose` argument; keep it, it feeds the trace.
- Keep prompts in `reviewer.py`. If you change agent behavior, add or update a scripted test in
  `tests/test_review_flow.py` and re-run the evaluation.
- Commit messages: imperative mood, one logical change per commit (`Add BMI tool`, `Fix PHI regex for
  member ids`).

## Adding a policy

1. Add `data/policies/<POLICY_ID>.json`: criteria with ids, a `logic` tree using only those ids
   (`all_of` / `any_of`), supported ICD-10-CM codes, and guideline `sections` with unique ids.
   `Policy` validation fails if the logic and the criteria disagree.
2. Add at least three labelled cases in `evals/cases/` (one per outcome).
   `tests/test_data.py` checks that labels match the decision rules.
3. Run the evaluation on the new cases.
