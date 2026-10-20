# ADR 0003: Redact PHI before any model call

**Status:** accepted · **Date:** 2026-10

## Context

Clinical notes contain identifiers that the review does not need (names, dates of birth, phone
numbers, record and member numbers, addresses). Sending them to a third-party model, logging them
or storing them widens the exposure for no benefit.

## Decision

The `intake` node redacts identifiers with typed placeholders (`[NAME]`, `[DATE]`...) and drops the
raw request from the graph state. Only redacted text reaches the model, the trace and SQLite. A name
found once is removed everywhere in the note. The guard re-scans model output for identifiers.

## Consequences

- An end-to-end test asserts that none of the raw identifiers of a case appears in the model input,
  the result or the trace; the evaluation reports the same check as `phi_leaks`.
- Absolute dates are lost, so durations must be stated in the note ("for 9 weeks"). This matches
  how criteria are written.
- Rules miss unlabelled names and cities. A clinical NER model is the planned next layer.
