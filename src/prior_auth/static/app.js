// Minimal client. All model-written text is inserted with textContent, never innerHTML.
const $ = (id) => document.getElementById(id);
const form = $("review-form");
let examples = [];

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

const LABELS = {
  recommend_approval: "Recommend approval",
  request_more_information: "Request more information",
  refer_to_physician_review: "Refer to physician review",
  met: "met", not_met: "not met", insufficient_evidence: "insufficient evidence",
};

async function init() {
  try {
    const [policies, cases] = await Promise.all([
      fetch("/v1/policies").then((r) => r.json()),
      fetch("/v1/examples").then((r) => r.json()),
    ]);
    policies.forEach((p) => $("service").append(el("option", { value: p.service_code }, `${p.service_name} · ${p.policy_id}`)));
    examples = cases;
    cases.forEach((c, i) => $("examples").append(el("option", { value: i }, `${c.case_id} · ${c.summary}`)));
  } catch (err) {
    showError(`Could not load policies: ${err}`);
  }
  const ready = await fetch("/ready");
  $("ready").textContent = ready.ok ? "ready" : "LLM not configured";
  $("ready").className = `pill ${ready.ok ? "pill-ok" : "pill-muted"}`;
}

$("examples").addEventListener("change", (e) => {
  const c = examples[e.target.value];
  if (!c) return;
  const r = c.request;
  form.plan_id.value = r.plan_id;
  form.urgency.value = r.urgency;
  form.service_code.value = r.service_code;
  form.diagnosis_codes.value = r.diagnosis_codes.join(", ");
  form.clinical_notes.value = r.clinical_notes;
  form.document_text.value = r.document_text || "";
});

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  $("submit").disabled = true;
  $("submit").textContent = "Reviewing…";
  try {
    const res = await fetch("/v1/reviews", { method: "POST", body: new FormData(form) });
    const body = await res.json().catch(() => ({ detail: `Server error (${res.status})` }));
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    render(body);
    const trace = await fetch(`/v1/reviews/${body.review_id}/trace`).then((r) => r.json());
    renderTrace(trace);
  } catch (err) {
    showError(err.message);
  } finally {
    $("submit").disabled = false;
    $("submit").textContent = "Run pre-review";
  }
});

function list(target, items) {
  target.replaceChildren(...items.map((t) => el("li", {}, t)));
}

function render(r) {
  $("empty").hidden = true;
  $("result").hidden = false;
  $("review-id").textContent = r.review_id;
  $("recommendation").textContent = LABELS[r.recommendation];
  $("recommendation").className = `badge ${r.recommendation}`;
  $("due").textContent = `Decision due by ${new Date(r.decision_due_by).toLocaleString()} · ${r.policy_id || "no policy"}`;
  list($("reasons"), r.recommendation_reasons);

  $("criteria").replaceChildren(...r.criteria.map((c) => {
    const box = el("div", { class: "criterion" });
    const head = el("div", { class: "criterion-head" });
    head.append(el("span", { class: "criterion-id" }, c.criterion_id), el("span", { class: `status ${c.status}` }, LABELS[c.status]));
    const ev = el("ul", { class: "evidence" });
    c.evidence.forEach((s) => {
      const li = el("li");
      li.append(el("span", { class: "mono" }, s.id), document.createTextNode(s.text));
      ev.append(li);
    });
    box.append(head, el("p", { class: "criterion-text" }, c.criterion_text), el("p", { class: "rationale" }, c.rationale), ev);
    return box;
  }));
  if (!r.criteria.length) $("criteria").replaceChildren(el("p", { class: "muted" }, "Criteria were not assessed for this request."));

  $("missing-block").hidden = !r.missing_information.length;
  list($("missing"), r.missing_information);
  $("summary").textContent = r.reviewer_summary;
  list($("dx"), r.diagnosis_checks.map((d) => `${d.code}: ${d.supported ? "listed" : "not listed"}`));
  list($("sections"), r.cited_sections.map((s) => `${s.section_id} ${s.heading}`));
  const phi = Object.entries(r.phi_redactions).map(([k, v]) => `${v} ${k}`).join(", ");
  $("phi").textContent = phi || "none detected";
  $("guard").textContent = r.guard.passed ? "passed" : `replaced: ${r.guard.replaced_fields.join(", ")}`;
}

function renderTrace(t) {
  $("trace-meta").textContent = `· ${t.llm_turns} model turns, ${t.tool_calls} tool calls, ${(t.duration_ms / 1000).toFixed(1)} s`;
  $("trace").replaceChildren(...t.events.map((e) => {
    const li = el("li");
    if (e.kind === "step") {
      li.append(el("strong", {}, `${e.node}: `), document.createTextNode(e.message));
    } else {
      li.append(el("span", { class: "tool" }, e.tool), document.createTextNode(` · ${e.purpose} → `),
        el("span", { class: e.ok ? "" : "fail" }, e.observation_summary));
    }
    return li;
  }));
}

init();
