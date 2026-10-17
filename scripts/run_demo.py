"""Run a fixed demo conversation and save the full transcript (with every tool call).

  python scripts/run_demo.py            # Gemini if GEMINI_API_KEY is set
  python scripts/run_demo.py --offline  # rule-based planner
Output: evidence/demo_transcript_<mode>.md
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expense_agent import build_agent  # noqa: E402

SCRIPT = [
    ("How much did I spend on food in September?", True),
    ("Am I over budget this month?", True),
    ("Any unusual transactions?", True),
    ("Compare August and September", True),
    ("Which transactions need review?", True),
    ("Transaction 87 is Food & Dining", True),  # human feedback, approved
    ("Which transactions need review?", True),  # queue should now be shorter
    ("Set my Shopping budget to 5000", False),  # human says NO
    ("Set my Food budget to 7000", True),       # human says YES
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    decision = {"value": True}
    agent, store, report = build_agent(offline=True if a.offline else None,
                                       approve=lambda n, args: decision["value"])
    mode = type(agent.llm).__name__
    out = [f"# Demo transcript ({mode})", f"Run at {datetime.now():%Y-%m-%d %H:%M}. "
           f"Loaded {report['loaded']} transactions, {len(report['skipped'])} skipped.", ""]
    for i, (q, approve) in enumerate(SCRIPT, 1):
        decision["value"] = approve
        agent.reset()  # each test question is independent
        r = agent.run(q)
        out += [f"## Test {i}: {q}", "", "**Tool calls:**", ""]
        for t in r.trace:
            if t["type"] == "tool":
                res = json.dumps(t["result"])
                out.append(f"- step {t['step']}: `{t['name']}({json.dumps(t['args'])})` → "
                           f"`{res[:200]}{'…' if len(res) > 200 else ''}`")
        if any(t.get("name") in ("set_budget", "confirm_category") for t in r.trace):
            out.append(f"- human approval given: **{'yes' if approve else 'no'}**")
        out += ["", f"**Agent answer** ({r.steps} LLM steps, {r.stopped_reason}):", "", "```", r.answer, "```", ""]
        print(f"[{i}] {q}\n{r.answer}\n")
    out.append(f"Budgets after the session: {json.dumps(store.budgets())}")
    dest = ROOT / "evidence" / f"demo_transcript_{mode}.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
