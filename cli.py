"""Command-line chat with the expense agent.

  python cli.py                         # interactive chat (Gemini if GEMINI_API_KEY is set)
  python cli.py --offline               # no API key needed
  python cli.py -q "Am I over budget?"  # single question
  python cli.py --csv my_statement.csv --trace
"""
import argparse
import json

from expense_agent import SAMPLE_CSV, build_agent


def ask_user(name, args):
    print(f"\n[approval needed] The agent wants to run {name} with {json.dumps(args)}")
    return input("Allow? (y/n): ").strip().lower() == "y"


def show_trace(result):
    for t in result.trace:
        if t["type"] == "tool":
            res = json.dumps(t["result"])
            print(f"  step {t['step']}: {t['name']}({json.dumps(t['args'])}) -> {res[:160]}{'...' if len(res) > 160 else ''}")
    print(f"  ({result.steps} LLM step(s), stopped: {result.stopped_reason})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(SAMPLE_CSV))
    p.add_argument("--offline", action="store_true", help="run without Gemini")
    p.add_argument("-q", "--question")
    p.add_argument("--trace", action="store_true", help="print each tool call")
    a = p.parse_args()

    agent, store, report = build_agent(a.csv, offline=True if a.offline else None, approve=ask_user)
    mode = type(agent.llm).__name__
    print(f"Loaded {report['loaded']} transactions ({len(report['skipped'])} skipped). LLM: {mode}")

    questions = [a.question] if a.question else None
    while True:
        q = questions.pop(0) if questions else (None if a.question else input("\nYou: ").strip())
        if not q or q.lower() in {"exit", "quit"}:
            break
        result = agent.run(q)
        print(f"\nAgent: {result.answer}")
        if a.trace:
            show_trace(result)


if __name__ == "__main__":
    main()
