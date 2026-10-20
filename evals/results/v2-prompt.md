# Evaluation: gemini / gemini-3.5-flash-lite

| Metric | Value |
|---|---|
| cases | 16 |
| completed_runs | 16 |
| recommendation_accuracy | 0.938 |
| criterion_accuracy | 0.981 |
| false_approvals | 0 |
| missed_approvals | 0 |
| phi_leaks | 0 |
| guard_replacements | 0 |
| agent_completed_rate | 1.0 |
| median_latency_s | 4.6 |
| mean_tool_calls | 8.5 |
| mean_input_tokens | 12667 |

| Case | Expected | Got | Criteria (expected / got) |
|---|---|---|---|
| PA-001 | recommend_approval | recommend_approval | RF1 not_met/not_met, C1 met/met, C2 met/met, C3 met/met |
| PA-002 | recommend_approval | recommend_approval | RF1 met/met, C1 not_met/not_met, C2 not_met/not_met, C3 met/met |
| PA-003 | refer_to_physician_review | refer_to_physician_review | RF1 not_met/not_met, C1 not_met/not_met, C2 not_met/not_met, C3 not_met/not_met |
| PA-004 | recommend_approval | recommend_approval | C1 met/met, C2 met/met, C3 met/met, C4 met/met |
| PA-005 | refer_to_physician_review | refer_to_physician_review | C1 not_met/not_met, C2 met/met, C3 met/met, C4 not_met/not_met |
| PA-006 | request_more_information | request_more_information | C1 met/met, C2 met/met, C3 insufficient_evidence/insufficient_evidence, C4 insufficient_evidence/insufficient_evidence |
| PA-007 | recommend_approval | recommend_approval | C1 met/met, C2 met/met, C3 met/met, C4 met/met |
| PA-008 | request_more_information | request_more_information | C1 met/met, C2 met/met, C3 insufficient_evidence/insufficient_evidence, C4 insufficient_evidence/insufficient_evidence |
| PA-009 | refer_to_physician_review | refer_to_physician_review | C1 met/met, C2 not_met/not_met, C3 met/met, C4 met/met |
| PA-010 | refer_to_physician_review | refer_to_physician_review |  |
| PA-011 | recommend_approval | recommend_approval | C1 met/met, C2 met/met |
| PA-012 | refer_to_physician_review | refer_to_physician_review | C1 met/met, C2 not_met/not_met |
| PA-013 | refer_to_physician_review | refer_to_physician_review | C1 met/met, C2 met/met |
| PA-014 | recommend_approval | recommend_approval | C1 met/met, C2 met/met, C3 met/met |
| PA-015 | refer_to_physician_review | refer_to_physician_review | C1 not_met/not_met, C2 not_met/not_met, C3 not_met/not_met |
| PA-016 | request_more_information | refer_to_physician_review | RF1 not_met/not_met, C1 met/met, C2 insufficient_evidence/not_met, C3 met/met |
