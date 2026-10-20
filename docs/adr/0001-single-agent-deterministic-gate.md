# ADR 0001: One LLM agent between deterministic nodes

**Status:** accepted · **Date:** 2026-10

## Context

A prior authorization review has parts that need language understanding (finding and reading
evidence in free-text notes) and parts that must be reproducible and auditable (applying the
policy's AND/OR logic, computing deadlines, blocking unsafe wording). Multi-agent designs often give
each part to its own LLM.

## Decision

Only the criteria review is an LLM agent (a ReAct loop with tools). Intake, decision and output
guard are plain code nodes in the same LangGraph graph. The agent records one assessment per
criterion; `determination.decide` turns those into a recommendation with three-valued logic.

## Consequences

- The same criterion assessments always give the same recommendation, and the rules are unit-tested.
- Fewer LLM calls per review, which matters on free-tier quotas.
- The model cannot "talk its way" to an approval: it never outputs the recommendation.
- Some judgment (is this sentence really evidence of a red flag?) stays with the model, so
  criterion accuracy is the metric to watch in the evaluation.
