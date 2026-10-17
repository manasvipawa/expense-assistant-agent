import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from expense_agent import SAMPLE_BUDGETS, SAMPLE_CSV  # noqa: E402
from expense_agent.store import Store  # noqa: E402
from expense_agent.tools import ExpenseTools  # noqa: E402


@pytest.fixture
def store():
    s = Store()
    s.load_csv(SAMPLE_CSV)
    s.load_budgets(SAMPLE_BUDGETS)
    return s


@pytest.fixture
def tools(store):
    return ExpenseTools(store)


@pytest.fixture
def raw_rows():
    """The CSV read independently of the app, used as ground truth."""
    with open(SAMPLE_CSV, encoding="utf-8") as f:
        return [dict(r, amount=float(r["amount"])) for r in csv.DictReader(f)]
