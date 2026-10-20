"""End-to-end runs of the graph with a scripted model: the ReAct loop, self-correction and safeguards."""

from collections.abc import Callable

from prior_auth.schemas import ReviewRequest
from prior_auth.service import ReviewService
from tests.conftest import MRI_APPROVAL_SCRIPT, ScriptedChatModel, finish, load_case, record, search

MakeService = Callable[[ScriptedChatModel], ReviewService]


def _request(case_id: str) -> ReviewRequest:
    return ReviewRequest.model_validate(load_case(case_id)["request"])


def test_full_react_loop_produces_a_grounded_approval(make_service: MakeService) -> None:
    llm = ScriptedChatModel(MRI_APPROVAL_SCRIPT)
    outcome = make_service(llm).review(_request("PA-001"))
    result = outcome.result

    assert result.recommendation == "recommend_approval"
    assert result.agent_completed and not result.human_review_required
    assert [c.status for c in result.criteria] == ["not_met", "met", "met", "met"]
    assert all(c.evidence for c in result.criteria)  # every assessment cites record sentences
    assert result.cited_sections and result.cited_sections[0].section_id.startswith("MP-RAD-101")
    assert result.decision_due_by > result.received_at
    tools = [e["tool"] for e in outcome.trace["events"] if e["kind"] == "tool_call"]
    assert tools.count("search_clinical_record") == 4 and tools[-1] == "finish_review"
    assert outcome.trace["llm_turns"] == 5


def test_phi_never_reaches_the_model_or_the_outputs(make_service: MakeService) -> None:
    llm = ScriptedChatModel(MRI_APPROVAL_SCRIPT)
    outcome = make_service(llm).review(_request("PA-001"))

    sent_to_model = " ".join(str(m.content) for turn in llm.seen for m in turn)
    stored = outcome.result.model_dump_json() + str(outcome.trace)
    for identifier in ("Okafor", "Daniel", "00482913", "555-0142", "04/12/1974", "Hamid"):
        assert identifier not in sent_to_model
        assert identifier not in stored
    assert outcome.result.phi_redactions == {"NAME": 2, "DATE": 1, "MRN": 1, "PHONE": 1}


def test_agent_corrects_itself_after_a_rejected_call(make_service: MakeService) -> None:
    script = [
        [record("C1", "met", ["N1"])],  # cites a sentence it never retrieved -> error observation
        [search("weeks of back pain")],
        *MRI_APPROVAL_SCRIPT,
    ]
    outcome = make_service(ScriptedChatModel(script)).review(_request("PA-001"))
    first = next(e for e in outcome.trace["events"] if e["kind"] == "tool_call")
    assert not first["ok"] and "not returned by search_clinical_record" in first["observation"]["error"]
    assert outcome.result.recommendation == "recommend_approval"


def test_missing_documentation_requests_information(make_service: MakeService) -> None:
    script = [
        [search("locking catching giving way"), search("conservative therapy")],
        [
            record("C1", "met", ["$hit:0"]),
            record("C3", "insufficient_evidence"),
            search("MRI meniscal tear"),
            ("search_policy_guidelines", {"purpose": "p", "query": "osteoarthritis grade"}),
        ],
        [record("C2", "met", ["$hit:0"]), record("C4", "insufficient_evidence"), finish()],
    ]
    result = make_service(ScriptedChatModel(script)).review(_request("PA-006")).result
    assert result.recommendation == "request_more_information"
    assert [m.split(":")[0] for m in result.missing_information] == ["C3", "C4"]


def test_turn_limit_routes_to_physician_review(make_service: MakeService) -> None:
    looping = ScriptedChatModel([[search("back pain")]])  # never finishes
    outcome = make_service(looping).review(_request("PA-001"))
    assert outcome.result.recommendation == "refer_to_physician_review"
    assert not outcome.result.agent_completed
    assert looping.calls == 8  # max_agent_turns in the test settings
    assert all(c.status == "insufficient_evidence" for c in outcome.result.criteria)


def test_plan_outside_policy_skips_the_model(make_service: MakeService) -> None:
    llm = ScriptedChatModel([[search("unused")]])
    result = make_service(llm).review(_request("PA-010")).result
    assert llm.calls == 0
    assert result.recommendation == "refer_to_physician_review"
    assert "does not apply to plan" in result.recommendation_reasons[0]


def test_unsafe_summary_is_replaced(make_service: MakeService) -> None:
    script = [
        *MRI_APPROVAL_SCRIPT[:-1],
        [record("C3", "met", ["$hit:0"]), finish("Coverage is guaranteed for this member.")],
    ]
    result = make_service(ScriptedChatModel(script)).review(_request("PA-001")).result
    assert "guaranteed" not in result.reviewer_summary
    assert result.guard.replaced_fields == ["reviewer_summary"]
    assert result.recommendation == "recommend_approval"  # the guard changes wording, not the decision
