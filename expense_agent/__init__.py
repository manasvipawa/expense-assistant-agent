"""Expense Assistant Agent: an agentic AI that analyses a bank statement."""
import os
from pathlib import Path

from .agent import Agent, deny_all
from .offline import OfflinePlanner
from .store import Store
from .tools import DECLARATIONS, ExpenseTools

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "sample_transactions.csv"
SAMPLE_BUDGETS = ROOT / "data" / "budgets.json"


def load_env(path=ROOT / ".env"):
    """Minimal .env reader so the API key never has to be typed into code."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_agent(csv_path=SAMPLE_CSV, budgets_path=SAMPLE_BUDGETS, offline=None,
                approve=deny_all, db_path=":memory:"):
    """Wire store + tools + LLM together. Uses Gemini if a key is set, else offline."""
    load_env()
    if offline is None:
        offline = not os.environ.get("GEMINI_API_KEY")
    store = Store(db_path)
    classifier = None
    if not offline:
        from .gemini import GeminiLLM, ModelRouter, _client, make_gemini_classifier
        router = ModelRouter(_client())  # shared, so the working model is found only once
        classifier = make_gemini_classifier(router=router)
        llm = GeminiLLM(DECLARATIONS, router=router)
    else:
        llm = OfflinePlanner()
    load_report = store.load_csv(csv_path, llm_classifier=classifier)
    if budgets_path:
        store.load_budgets(budgets_path)
    agent = Agent(llm, ExpenseTools(store), approve=approve,
                  fallback_llm=None if offline else OfflinePlanner())
    return agent, store, load_report
