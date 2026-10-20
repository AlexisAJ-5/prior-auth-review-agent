"""LangGraph wiring. One LLM agent (the ReAct loop) between deterministic intake and decision nodes.

    START -> intake ---------- policy missing or plan not covered ----------+
               |                                                            |
               v                                                            |
           reviewer <------------+   REASON: the model picks tools,         |
               |                 |   each with a stated purpose             |
               v                 |                                          |
           run_tools ------------+   ACT + OBSERVE: validate, execute,      |
               | finished (or turn limit)       trace every call            |
               v                                                            |
           decide  <--------------------------------------------------------+
               |    deterministic: Kleene logic over criteria, never "deny"
               v
            guard -> END        deterministic output checks, template fallback

Raw request text exists only inside `intake`: it is redacted there and dropped from the state.
"""

import json
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from . import determination, guard, phi
from .config import Settings
from .embeddings import Embedder
from .policies import PolicyRepository
from .record import ClinicalRecord
from .reviewer import SYSTEM_PROMPT, opening_message
from .schemas import CitedSection, CriterionResult, GuardReport, ReviewRequest, ReviewResult
from .tools import FINAL_TOOL, ReviewContext, ToolBox, tool_specs
from .trace import TraceRecorder


class ReviewState(TypedDict, total=False):
    review_id: str
    request: ReviewRequest | None  # raw input; set to None by intake
    meta: dict[str, Any]  # non-PHI request fields: plan, service, urgency, received_at, review_id
    ctx: ReviewContext
    trace: TraceRecorder
    messages: Annotated[list[AnyMessage], add_messages]
    turns: int
    result: ReviewResult


def build_graph(
    llm: BaseChatModel, toolbox: ToolBox, policies: PolicyRepository, embedder: Embedder, settings: Settings
) -> CompiledStateGraph:  # type: ignore[type-arg]
    agent_llm = llm.bind_tools(tool_specs(), tool_choice=settings.llm_tool_choice)

    def intake(state: ReviewState) -> dict[str, Any]:
        request = state["request"]
        assert request is not None
        trace = state["trace"]
        notes = phi.redact(request.clinical_notes[: settings.max_notes_chars])
        document = (
            phi.redact(request.document_text[: settings.max_document_chars])
            if request.document_text
            else None
        )
        counts = dict(notes.counts + (document.counts if document else Counter()))

        policy = policies.get(request.service_code)
        checks = determination.check_diagnoses(policy, request.diagnosis_codes) if policy else []
        applies = bool(policy) and determination.plan_applies(policy, request.plan_id)  # type: ignore[arg-type]
        ctx = ReviewContext(
            policy=policy,
            record=ClinicalRecord(notes.text, document.text if document else None, embedder),
            diagnosis_checks=checks,
            plan_applies=applies,
            phi_counts=counts,
        )
        redacted = ", ".join(f"{n} {label}" for label, n in sorted(counts.items())) or "none found"
        trace.step("intake", f"PHI redacted before any model call: {redacted}.")
        trace.step(
            "intake",
            f"Policy: {policy.policy_id if policy else 'none'}; applies to plan: {applies}; "
            f"supported diagnosis codes: {sum(c.supported for c in checks)}/{len(checks)}.",
        )
        meta = {
            "review_id": state["review_id"],
            "received_at": datetime.now(UTC),
            "plan_id": request.plan_id,
            "service_code": request.service_code,
            "urgency": request.urgency,
        }
        return {"request": None, "meta": meta, "ctx": ctx}

    def reviewer(state: ReviewState) -> dict[str, Any]:
        ctx, meta = state["ctx"], state["meta"]
        history = state.get("messages") or [
            SystemMessage(SYSTEM_PROMPT),
            opening_message(ctx, meta["service_code"], meta["plan_id"], meta["urgency"]),
        ]
        reply = agent_llm.invoke(history)
        new = [reply] if state.get("messages") else [*history, reply]
        return {"messages": new, "turns": state.get("turns", 0) + 1}

    def run_tools(state: ReviewState) -> dict[str, Any]:
        ctx, trace, turn = state["ctx"], state["trace"], state["turns"]
        calls = getattr(state["messages"][-1], "tool_calls", [])
        if not calls:
            trace.step("reviewer", "Model replied without a tool call; nudged to continue.")
            return {"messages": [HumanMessage("Continue by calling a tool. Call finish_review when done.")]}
        observations: dict[str, dict[str, Any]] = {}
        # Execute evidence searches first, then assessments, then the final call, so parallel calls
        # in one turn see each other's effects. Replies keep the model's original call order.
        order = {"record_criterion": 1, FINAL_TOOL: 2}
        for call in sorted(calls, key=lambda c: order.get(c["name"], 0)):
            started = time.perf_counter()
            purpose, tool_input, observation = toolbox.execute(ctx, call["name"], call["args"])
            trace.tool(
                turn=turn,
                tool=call["name"],
                purpose=purpose,
                tool_input=tool_input,
                observation=observation,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            observations[call["id"]] = observation
        replies: list[AnyMessage] = [
            ToolMessage(json.dumps(observations[c["id"]]), tool_call_id=c["id"], name=c["name"])
            for c in calls
        ]
        return {"messages": replies}

    def decide(state: ReviewState) -> dict[str, Any]:
        ctx, meta, trace = state["ctx"], state["meta"], state["trace"]
        policy = ctx.policy
        statuses = ctx.statuses()
        if policy and ctx.plan_applies and not ctx.finished:
            trace.step("decide", f"Agent stopped after {state.get('turns', 0)} turns without finishing.")
        recommendation, reasons = determination.decide(
            policy,
            meta["plan_id"],
            ctx.diagnosis_checks,
            statuses,
            agent_completed=ctx.finished or not (policy and ctx.plan_applies),
        )
        trace.step("decide", f"Deterministic gate: {recommendation}. " + " ".join(reasons))

        criteria = []
        if policy and ctx.plan_applies:
            for c in policy.criteria:
                a = ctx.assessments.get(c.id)
                criteria.append(
                    CriterionResult(
                        criterion_id=c.id,
                        criterion_text=c.text,
                        status=a.status if a else "insufficient_evidence",
                        rationale=a.rationale if a else "Not assessed by the agent.",
                        evidence=[s for i in (a.evidence_ids if a else []) if (s := ctx.record.get(i))],
                    )
                )
        result = ReviewResult(
            review_id=meta["review_id"],
            received_at=meta["received_at"],
            decision_due_by=determination.decision_due_by(meta["received_at"], meta["urgency"]),
            plan_id=meta["plan_id"],
            service_code=meta["service_code"],
            urgency=meta["urgency"],
            policy_id=policy.policy_id if policy else None,
            recommendation=recommendation,
            recommendation_reasons=reasons,
            criteria=criteria,
            missing_information=determination.missing_information(policy, statuses)
            if policy and ctx.plan_applies
            else [],
            diagnosis_checks=ctx.diagnosis_checks,
            cited_sections=[
                CitedSection(section_id=s, heading=ctx.seen_sections[s]) for s in ctx.cited_section_ids
            ],
            reviewer_summary=ctx.summary or "No automated summary; see recommendation reasons.",
            phi_redactions=ctx.phi_counts,
            guard=GuardReport(passed=True),
            agent_completed=ctx.finished,
            human_review_required=recommendation != "recommend_approval",
        )
        return {"result": result}

    def output_guard(state: ReviewState) -> dict[str, Any]:
        result = state["result"]
        summary, criteria, report = guard.apply_guard(result.reviewer_summary, result.criteria)
        state["trace"].step(
            "guard",
            "All model-written text passed the output checks."
            if report.passed
            else f"Replaced {', '.join(report.replaced_fields)}: {'; '.join(report.issues)}",
        )
        return {
            "result": result.model_copy(
                update={"reviewer_summary": summary, "criteria": criteria, "guard": report}
            )
        }

    def after_intake(state: ReviewState) -> str:
        ctx = state["ctx"]
        return "reviewer" if ctx.policy and ctx.plan_applies else "decide"

    def after_tools(state: ReviewState) -> str:
        if state["ctx"].finished or state.get("turns", 0) >= settings.max_agent_turns:
            return "decide"
        return "reviewer"

    graph = StateGraph(ReviewState)
    graph.add_node("intake", intake)
    graph.add_node("reviewer", reviewer)
    graph.add_node("run_tools", run_tools)
    graph.add_node("decide", decide)
    graph.add_node("guard", output_guard)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", after_intake, ["reviewer", "decide"])
    graph.add_edge("reviewer", "run_tools")
    graph.add_conditional_edges("run_tools", after_tools, ["reviewer", "decide"])
    graph.add_edge("decide", "guard")
    graph.add_edge("guard", END)
    return graph.compile()
