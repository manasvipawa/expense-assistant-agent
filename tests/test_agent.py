"""Tests for the agent loop: tool use, guardrails, approval and end-to-end answers."""
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
