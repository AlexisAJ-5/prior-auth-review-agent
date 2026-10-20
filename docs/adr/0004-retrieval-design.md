# ADR 0004: Two retrieval surfaces, with citations checked by code

**Status:** accepted · **Date:** 2026-10

## Context

The agent needs two kinds of context: the policy (definitions, thresholds) and the patient record
(evidence). Putting the whole record in the prompt is simple but does not scale to long records and
makes citations unverifiable.

## Decision

- **Policy guidelines** are embedded once into Qdrant (FastEmbed `bge-small-en-v1.5`) and searched
  with a filter on the governing policy.
- **The record** is split into sentences with stable ids (`N3` for notes, `D1` for the attached
  document) and searched in memory per review, with a hybrid score: cosine similarity plus a lexical
  overlap bonus, because clinical terms ("saddle anesthesia") are precise and small embedders blur
  them.
- `record_criterion` rejects evidence ids that no search returned; `finish_review` rejects policy
  sections that no search returned.

## Consequences

- Every cited sentence exists and was seen by the agent. Hallucinated quotes are impossible by
  construction.
- More tool calls per review than a single prompt.
- If retrieval misses the evidence, the criterion becomes `insufficient_evidence`, which routes to
  a request for information: a safe failure mode.
