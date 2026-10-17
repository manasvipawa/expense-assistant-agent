"""Tests for CSV loading, categorisation and every tool."""
import pytest

from expense_agent.categorizer import categorize
from expense_agent.store import CSVFormatError, Store


# ---------- loading & validation ----------
def test_sample_loads_every_row(store, raw_rows):
    assert len(store.query("SELECT id FROM transactions")) == len(raw_rows)


def test_invalid_rows_are_skipped_not_stored(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,description,amount\n"
                 "2026-09-01,Swiggy Order,250\n"
                 "01/09/2026,Bad date,100\n"
                 "2026-09-02,Negative,-50\n"
                 "2026-09-03,Not a number,abc\n"
                 "2026-09-04,,100\n")
    s = Store()
    report = s.load_csv(p)
    assert report["loaded"] == 1
    assert [line for line, _ in report["skipped"]] == [3, 4, 5, 6]


def test_missing_column_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,amount\n2026-09-01,100\n")
    with pytest.raises(CSVFormatError):
        Store().load_csv(p)


# ---------- categoriser ----------
@pytest.mark.parametrize("desc,expected", [
    ("Swiggy Order", "Food & Dining"),
    ("Uber Trip", "Transport"),
    ("BESCOM Electricity Bill", "Bills & Utilities"),
    ("House Rent - Landlord Transfer", "Rent"),
    ("Coca Cola vending machine", "Other"),  # 'ola' inside 'Cola' must NOT mean Transport
])
def test_keyword_rules(desc, expected):
    assert categorize(desc)[0] == expected


def test_unknown_merchant_goes_to_review():
    cat, conf, src = categorize("Sharma Tailors")
    assert (cat, conf, src) == ("Other", 0.0, "none")


def test_llm_fallback_and_invalid_llm_output():
    assert categorize("Sharma Tailors", llm_classifier=lambda d: ("Shopping", 0.8))[:2] == ("Shopping", 0.8)
    # an LLM that invents a category is ignored
    assert categorize("Sharma Tailors", llm_classifier=lambda d: ("Luxury", 0.9))[0] == "Other"
    # an LLM that crashes does not break loading
    def boom(d):
        raise RuntimeError("API down")
    assert categorize("Sharma Tailors", llm_classifier=boom)[0] == "Other"


def test_human_feedback_is_learned(store, tools):
    queue = tools.get_review_queue()["transactions"]
    chai = [t for t in queue if "Chai Point" in t["description_data"]]
    assert len(chai) == 2
    out = tools.confirm_category(chai[0]["id"], "Food & Dining", remember=True)
    assert out["updated"] == 2  # the other Chai Point row was fixed by the learned rule
    remaining = [t["description_data"] for t in tools.get_review_queue()["transactions"]]
    assert not any("Chai Point" in d for d in remaining)
    assert categorize("Chai Point HSR Layout", store.learned_rules())[2] == "human"


# ---------- tools: numbers must match an independent calculation ----------
def test_summary_matches_ground_truth(tools, raw_rows):
    expected = round(sum(r["amount"] for r in raw_rows if r["date"].startswith("2026-09")), 2)
    out = tools.get_spending_summary("2026-09")
    assert out["total"] == expected
    assert round(sum(c["total"] for c in out["by_category"].values()), 2) == expected


def test_budget_status(tools):
    out = {b["category"]: b for b in tools.check_budgets("2026-09")["budgets"]}
    assert out["Shopping"]["status"] == "over"
    assert out["Food & Dining"]["status"] == "over"
    assert out["Bills & Utilities"]["status"] == "ok"
    for b in out.values():
        assert b["over_by"] == round(max(0, b["spent"] - b["budget"]), 2)


def test_anomaly_detector_finds_the_smartphone(tools):
    found = tools.detect_anomalies("2026-09")["anomalies"]
    assert found[0]["amount"] == 18999
    # no ordinary food order or rent payment should be flagged
    assert not any(a["category"] in ("Food & Dining", "Rent") for a in found)


def test_compare_months(tools):
    out = tools.compare_months("2026-08", "2026-09")
    a = tools.get_spending_summary("2026-08")["total"]
    b = tools.get_spending_summary("2026-09")["total"]
    assert out["total_change"] == round(b - a, 2)


def test_find_transactions_filters(tools):
    out = tools.find_transactions(month="2026-09", category="Shopping")
    assert all(t["category"] == "Shopping" for t in out["transactions"])
    assert out["transactions"][0]["amount"] >= out["transactions"][-1]["amount"]


@pytest.mark.parametrize("name,args", [
    ("get_spending_summary", {"month": "Sept"}),
    ("get_spending_summary", {"month": "2026-13"}),
    ("set_budget", {"category": "Luxury", "amount": 100}),
    ("set_budget", {"category": "Rent", "amount": -5}),
    ("find_transactions", {"category": "Pets"}),
    ("get_spending_summary", {}),
    ("delete_everything", {}),
])
def test_bad_tool_calls_return_errors_not_exceptions(tools, name, args):
    assert "error" in tools.execute(name, args)
