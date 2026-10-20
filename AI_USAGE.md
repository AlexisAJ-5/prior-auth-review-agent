# AI usage

This project was built with an AI pair programmer (Claude, by Anthropic). This file says what was
delegated and how the output was checked, because that is part of how the code should be judged.

## What the AI did

- Drafted most of the initial code, tests and documentation from a design discussed and decided
  together: scope (pre-review, never deny), the graph shape, the tool set and the safeguards.
- Wrote the synthetic policies and labelled cases from that brief.
- Proposed the regex patterns for PHI redaction and the output guard.

## What I decided and checked

- The technical stack. LangGraph, Qdrant, FastEmbed, FastAPI and Docker come from my earlier work
  on agentic systems. I chose them for this project and adapted them to the problem: a single
  agent instead of several, retrieval over the patient record as well as the policies, and a
  provider switch so the project runs on a free Gemini key or fully local with Ollama.
- The product boundaries: no automated denials, PHI redaction before any model call, deterministic
  decision logic, citations verified by code.
- Every file was reviewed before commit. The offline suite (lint, strict typing, tests) runs locally
  and in CI.
- The evaluation was run against a real model, and behavior was adjusted from the results.

## Corrections made during development

- The name detector first redacted the label with the name (`Patient:` disappeared). Fixed with an
  explicit capture group and a regression test.
- Sentence splitting broke on `Dr.`, which separated a clinician name from its sentence. Fixed with
  an abbreviation rule and a test.
- Validation errors from Pydantic were not JSON serializable in the 422 response. Fixed by
  excluding the error context.

## Conventions for AI agents

See [CLAUDE.md](CLAUDE.md): commands, architecture and the invariants a change must respect.
