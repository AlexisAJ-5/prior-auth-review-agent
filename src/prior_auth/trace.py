"""Execution trace of a review: graph steps and every tool call with its purpose and observation."""

import time
from datetime import UTC, datetime
from typing import Any


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._started = time.perf_counter()

    def _add(self, event: dict[str, Any]) -> None:
        self.events.append({"seq": len(self.events) + 1, **event, "at": datetime.now(UTC).isoformat()})

    def step(self, node: str, message: str) -> None:
        self._add({"kind": "step", "node": node, "message": message})

    def tool(
        self,
        *,
        turn: int,
        tool: str,
        purpose: str,
        tool_input: dict[str, Any],
        observation: dict[str, Any],
        duration_ms: int,
    ) -> None:
        self._add(
            {
                "kind": "tool_call",
                "turn": turn,
                "tool": tool,
                "purpose": purpose,
                "input": tool_input,
                "observation_summary": summarize(tool, observation),
                "observation": observation,
                "ok": "error" not in observation,
                "duration_ms": duration_ms,
            }
        )

    @property
    def duration_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    @property
    def tool_calls(self) -> int:
        return sum(1 for e in self.events if e["kind"] == "tool_call")


def summarize(tool: str, observation: dict[str, Any]) -> str:
    if "error" in observation:
        return f"Error: {observation['error']}"
    if tool == "search_clinical_record":
        return "Found " + ", ".join(f"{s['id']} ({s['score']:.2f})" for s in observation["sentences"])
    if tool == "search_policy_guidelines":
        return "Found " + ", ".join(s["section_id"] for s in observation["sections"])
    if tool == "calculate_bmi":
        return f"BMI {observation['bmi']} {observation['unit']}"
    if tool == "record_criterion":
        left = observation["remaining_criteria"]
        return f"{observation['recorded']} = {observation['status']}; remaining: {', '.join(left) or 'none'}"
    return "Review finished"
