# ADR 0002: No automated denials

**Status:** accepted · **Date:** 2026-10

## Context

An incorrect denial delays care. Several jurisdictions require that medical-necessity denials be
made by a licensed clinician (for example California SB 1120, 2024), and payers face decision
deadlines (CMS-0057-F: 72 hours expedited, 7 days standard).

## Decision

The system has three outcomes: `recommend_approval`, `request_more_information` and
`refer_to_physician_review`. Anything that is not a clear approval goes to a human, including runs
where the agent did not finish, unsupported diagnosis codes and plans the policy does not cover.
The output guard also blocks adverse-decision wording in model text.

## Consequences

- The worst automated error is a missed approval (extra human work), never a wrongful denial.
- `false_approvals` is tracked separately in the evaluation because it is the costly error for a payer.
- Each result carries `decision_due_by` so reviewers can prioritize.
