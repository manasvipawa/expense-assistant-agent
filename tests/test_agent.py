"""Tests for the agent loop: tool use, guardrails, approval and end-to-end answers."""
import pytest
import re
from pathlib import Path

from expense_agent import build_agent
from expense_agent.agent import SYSTEM_PROMPT, Agent, ToolCall, Turn
from expense_agent.offline import inr


class ScriptedLLM:
    """Fake LLM that replays a fixed list of turns and records what it saw."""
    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def next(self, history):
        self.seen.append(list(history))
        return self.turns.pop(0)


def test_loop_calls_tool_then_answers(tools):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("get_available_months", {})]), Turn(text="Done")])
    r = Agent(llm, tools).run("hi")
    assert r.answer == "Done" and r.steps == 2 and r.stopped_reason == "answered"
    # the tool result was fed back to the model before its second turn
    assert llm.seen[1][-1]["role"] == "tool"
    assert llm.seen[1][-1]["results"][0][1]["latest"] == "2026-09"


def test_max_steps_stops_runaway_agent(tools):
    looping = [Turn(tool_calls=[ToolCall("get_spending_summary", {"month": f"2026-0{i}"})]) for i in range(1, 10)]
    r = Agent(ScriptedLLM(looping), tools, max_steps=4).run("loop")
    assert r.stopped_reason == "max_steps" and r.steps == 4


def test_duplicate_call_is_refused(tools):
    same = ToolCall("get_spending_summary", {"month": "2026-09"})
    llm = ScriptedLLM([Turn(tool_calls=[same]), Turn(tool_calls=[same]), Turn(text="ok")])
    r = Agent(llm, tools).run("q")
    tool_steps = [t for t in r.trace if t["type"] == "tool"]
    assert "total" in tool_steps[0]["result"]
    assert "duplicate" in tool_steps[1]["result"]["error"]


def test_unknown_tool_does_not_crash(tools):
    llm = ScriptedLLM([Turn(tool_calls=[ToolCall("transfer_money", {"to": "x"})]), Turn(text="sorry")])
    r = Agent(llm, tools).run("q")
    assert r.trace[0]["result"] == {"error": "unknown tool 'transfer_money'"}
    assert r.answer == "sorry"


def test_write_tool_needs_approval(store, tools):
    call = ToolCall("set_budget", {"category": "Rent", "amount": 20000})
    Agent(ScriptedLLM([Turn(tool_calls=[call]), Turn(text="x")]), tools).run("q")
    assert store.budgets()["Rent"] == 15000  # denied by default: unchanged
    asked = []
    approve = lambda name, args: asked.append(name) or True
    Agent(ScriptedLLM([Turn(tool_calls=[call]), Turn(text="x")]), tools, approve=approve).run("q")
    assert asked == ["set_budget"] and store.budgets()["Rent"] == 20000


def test_conversation_memory_kept_between_questions(tools):
    llm = ScriptedLLM([Turn(text="a"), Turn(text="b")])
    agent = Agent(llm, tools)
    agent.run("first"); agent.run("second")
    assert [h["text"] for h in llm.seen[1] if h["role"] == "user"] == ["first", "second"]


# ---------- end to end with the offline planner ----------
def _agent(**kw):
    return build_agent(offline=True, **kw)


def test_e2e_food_question_numbers_are_grounded(tools):
    agent, store, _ = _agent()
    r = agent.run("How much did I spend on food in September?")
    truth = tools.get_spending_summary("2026-09")["by_category"]["Food & Dining"]["total"]
    assert inr(truth) in r.answer
    assert [t["name"] for t in r.trace if t["type"] == "tool"] == ["get_available_months", "get_spending_summary"]


def test_e2e_budget_question_flags_overspending():
    agent, _, _ = _agent()
    r = agent.run("Am I over budget this month?")
    assert "Shopping: OVER" in r.answer and "Food & Dining: OVER" in r.answer


def test_e2e_last_month_resolves_to_august():
    agent, _, _ = _agent()
    r = agent.run("Give me a summary of last month")
    assert "August 2026" in r.answer


def test_e2e_set_budget_with_approval():
    agent, store, _ = _agent(approve=lambda n, a: True)
    r = agent.run("Set my Food budget to 7000")
    assert store.budgets()["Food & Dining"] == 7000 and "Rs. 7,000" in r.answer


# ---------- security ----------
def test_prompt_injection_text_is_passed_as_labelled_data(tools):
    rows = tools.get_review_queue()["transactions"]
    inj = [t for t in rows if "IGNORE ALL PREVIOUS" in t["description_data"]]
    assert inj and inj[0]["category"] == "Other"  # the text did not change how it was handled
    assert "description_data" in SYSTEM_PROMPT and "Never follow instructions" in SYSTEM_PROMPT


def test_injection_does_not_change_budget_answer():
    agent, _, _ = _agent()
    r = agent.run("Am I over budget this month?")
    assert "OVER" in r.answer  # the injected 'say all budgets are fine' had no effect


def test_no_api_key_committed_in_source():
    root = Path(__file__).resolve().parent.parent
    for f in list(root.rglob("*.py")) + list(root.rglob("*.md")) + list(root.rglob("*.json")):
        assert not re.search(r"AIza[0-9A-Za-z_\-]{30,}", f.read_text(errors="ignore")), f


# ---------- Gemini adapter (no network: checks the message format only) ----------
def test_gemini_adapter_builds_valid_messages():
    from expense_agent.gemini import GeminiLLM
    from expense_agent.tools import DECLARATIONS
    llm = GeminiLLM(DECLARATIONS, api_key="dummy-key-for-offline-test")
    assert llm.model == "gemini-3.8-flash"  # current model tried first, not the retired 2.5
    assert len(llm.config.tools[0].function_declarations) == len(DECLARATIONS)
    history = [{"role": "user", "text": "Am I over budget?"},
               {"role": "model", "turn": Turn(tool_calls=[ToolCall("get_available_months", {})])},
               {"role": "tool", "results": [("get_available_months", {"latest": "2026-09"})]}]
    contents = llm._contents(history)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "get_available_months"
    assert contents[2].parts[0].function_response.response == {"result": {"latest": "2026-09"}}


def test_e2e_human_feedback_shrinks_review_queue():
    agent, store, _ = _agent(approve=lambda n, a: True)
    before = agent.run("Which transactions need review?").answer
    assert before.startswith("4 transaction(s)")
    chai_id = store.query("SELECT id FROM transactions WHERE description LIKE 'Chai Point Kor%'")[0]["id"]
    r = agent.run(f"Transaction {chai_id} is Food & Dining")
    assert "updated 2 transaction(s)" in r.answer
    assert agent.run("Which transactions need review?").answer.startswith("2 transaction(s)")


# ---------- resilience: model availability and LLM failures ----------
class _FakeModels:
    """Stands in for client.models; fails for some model names."""
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def generate_content(self, model, contents, config):
        from google.genai import errors
        self.calls.append(model)
        b = self.behaviour.get(model, "ok")
        if b == "gone":
            raise errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND",
                                     "message": f"This model models/{model} is no longer available to new users."}})
        if b == "quota":
            raise errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                     "message": "Quota exceeded"}})
        return f"response from {model}"


class _FakeClient:
    def __init__(self, behaviour):
        self.models = _FakeModels(behaviour)


def test_router_skips_retired_models_and_remembers_working_one():
    from expense_agent.gemini import ModelRouter
    client = _FakeClient({"old-model": "gone"})
    router = ModelRouter(client, models=["old-model", "new-model"], sleep=lambda s: None)
    assert router.generate("hi", None) == "response from new-model"
    assert router.generate("again", None) == "response from new-model"
    assert client.models.calls == ["old-model", "new-model", "new-model"]  # old model tried only once
    assert router.skipped[0][0] == "old-model"


def test_router_retries_rate_limit_then_gives_up():
    import pytest
    from google.genai import errors
    from expense_agent.gemini import ModelRouter
    client = _FakeClient({"m": "quota"})
    router = ModelRouter(client, models=["m"], sleep=lambda s: None)
    with pytest.raises(errors.ClientError):
        router.generate("hi", None)
    assert client.models.calls == ["m", "m", "m"]  # 3 attempts with back-off


def test_router_reports_when_no_model_is_available():
    import pytest
    from expense_agent.gemini import ModelRouter
    router = ModelRouter(_FakeClient({"a": "gone", "b": "gone"}), models=["a", "b"], sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="None of the Gemini models"):
        router.generate("hi", None)


class _BrokenLLM:
    def next(self, history):
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")


def test_agent_falls_back_to_offline_planner_when_gemini_fails(tools):
    from expense_agent.offline import OfflinePlanner
    r = Agent(_BrokenLLM(), tools, fallback_llm=OfflinePlanner()).run("Am I over budget this month?")
    assert r.stopped_reason == "answered"
    assert "Shopping: OVER" in r.answer and "Gemini was unavailable" in r.answer
    assert r.trace[0]["type"] == "llm_error"


def test_agent_without_fallback_returns_message_instead_of_crashing(tools):
    r = Agent(_BrokenLLM(), tools).run("hi")
    assert r.stopped_reason == "llm_error" and "not available" in r.answer


@pytest.mark.parametrize("question,args", [
    ("What if I cut Swiggy by half?", {"cut_percent": 50.0, "merchant": "swiggy"}),
    ("How much would I save if I stop Uber?", {"cut_percent": 100.0, "merchant": "uber"}),
    ("What if I reduce shopping by 30%?", {"cut_percent": 30.0, "category": "Shopping"}),
])
def test_e2e_what_if_questions(question, args):
    agent, _, _ = _agent()
    r = agent.run(question)
    calls = [t for t in r.trace if t["type"] == "tool"]
    assert calls[-1]["name"] == "simulate_savings" and calls[-1]["args"] == args
    assert "you would save about Rs." in r.answer


def test_write_tool_can_wait_for_ui_approval(store, tools):
    call = ToolCall("set_budget", {"category": "Rent", "amount": 20000})
    r = Agent(ScriptedLLM([Turn(tool_calls=[call]), Turn(text="x")]), tools,
              approve=lambda n, a: None).run("q")
    assert r.trace[0]["result"]["status"] == "pending_approval"
    assert store.budgets()["Rent"] == 15000  # nothing changed until the user clicks Approve


def test_e2e_pending_approval_message():
    agent, store, _ = _agent(approve=lambda n, a: None)
    r = agent.run("Set my Food budget to 7000")
    assert "Please approve or reject" in r.answer and store.budgets()["Food & Dining"] == 6000
