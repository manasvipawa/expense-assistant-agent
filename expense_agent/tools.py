"""The tools the agent can call.

Each tool does one well-defined job and returns plain JSON, so the LLM never
has to do arithmetic over raw rows itself: the numbers in every answer come
from these functions. Write tools (set_budget, confirm_category) are marked so
the agent loop asks the human for approval before running them.
"""
import re
import statistics

from .categorizer import CATEGORIES, REVIEW_THRESHOLD

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
ANOMALY_Z = 2.5
MIN_HISTORY = 4
WARNING_PCT = 90

_month = {"type": "string", "description": "Month in YYYY-MM format, e.g. 2026-09"}
_category = {"type": "string", "enum": CATEGORIES}

DECLARATIONS = [
    {"name": "get_available_months",
     "description": "List the months that have transactions and the latest month. Call this first when the user does not name a month.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_spending_summary",
     "description": "Total spending per category and overall for one month.",
     "parameters": {"type": "object", "properties": {"month": _month}, "required": ["month"]}},
    {"name": "check_budgets",
     "description": "Compare each category's spending in a month with its budget. Status is ok, warning (>=90% used) or over.",
     "parameters": {"type": "object", "properties": {"month": _month}, "required": ["month"]}},
    {"name": "find_transactions",
     "description": "Search transactions. All filters optional. Returns at most 20 rows, largest first.",
     "parameters": {"type": "object", "properties": {
         "month": _month, "category": _category,
         "merchant": {"type": "string", "description": "Text to match in the description"},
         "min_amount": {"type": "number"}}}},
    {"name": "detect_anomalies",
     "description": "Find transactions in a month that are statistically unusual for their category (z-score >= 2.5 versus that category's history).",
     "parameters": {"type": "object", "properties": {"month": _month}, "required": ["month"]}},
    {"name": "compare_months",
     "description": "Category-by-category spending change between two months.",
     "parameters": {"type": "object", "properties": {"month_a": _month, "month_b": _month},
                    "required": ["month_a", "month_b"]}},
    {"name": "simulate_savings",
     "description": "What-if simulator. Estimates how much the user would save by cutting spending on a category OR a merchant by a percentage, over 3, 6 and 12 months, and what that does to the latest month's budget. Use for questions like 'what if I cut Swiggy by half?'. Give either category or merchant, not both.",
     "parameters": {"type": "object", "properties": {
         "cut_percent": {"type": "number", "description": "Reduction from 1 to 100 (100 = stop completely)"},
         "category": _category,
         "merchant": {"type": "string", "description": "Merchant text to match, e.g. swiggy"}},
         "required": ["cut_percent"]}},
    {"name": "get_review_queue",
     "description": "Transactions whose category the system is unsure about and which need a human to confirm.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "set_budget",
     "description": "Create or change the monthly budget for a category. Requires user approval.",
     "parameters": {"type": "object", "properties": {"category": _category, "amount": {"type": "number"}},
                    "required": ["category", "amount"]}},
    {"name": "confirm_category",
     "description": "Set the correct category for a transaction from the review queue. If remember is true the merchant is learned for the future. Requires user approval.",
     "parameters": {"type": "object", "properties": {
         "transaction_id": {"type": "integer"}, "category": _category,
         "remember": {"type": "boolean"}}, "required": ["transaction_id", "category"]}},
]
WRITE_TOOLS = {"set_budget", "confirm_category"}
TOOL_NAMES = {d["name"] for d in DECLARATIONS}


class ToolError(ValueError):
    pass


def _check_month(m):
    if not isinstance(m, str) or not MONTH_RE.match(m):
        raise ToolError(f"month must look like 2026-09, got {m!r}")
    return m


def _txn_view(r):
    # Descriptions are untrusted text from the bank file: label them as data.
    return {"id": r["id"], "date": r["date"], "description_data": r["description"],
            "amount": r["amount"], "category": r["category"]}


class ExpenseTools:
    def __init__(self, store):
        self.store = store

    def execute(self, name: str, args: dict) -> dict:
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except (ToolError, ValueError) as e:
            return {"error": str(e)}

    # ----- read tools -----
    def get_available_months(self):
        months = self.store.months()
        return {"months": months, "latest": months[-1] if months else None}

    def _totals(self, month):
        rows = self.store.query(
            "SELECT category, ROUND(SUM(amount),2) AS total, COUNT(*) AS n FROM transactions "
            "WHERE month=? GROUP BY category ORDER BY total DESC", (month,))
        return {r["category"]: {"total": r["total"], "count": r["n"]} for r in rows}

    def get_spending_summary(self, month):
        month = _check_month(month)
        by_cat = self._totals(month)
        if not by_cat:
            return {"month": month, "error": "no transactions for this month"}
        total = round(sum(v["total"] for v in by_cat.values()), 2)
        return {"month": month, "total": total, "by_category": by_cat}

    def check_budgets(self, month):
        month = _check_month(month)
        totals = self._totals(month)
        out = []
        for cat, budget in sorted(self.store.budgets().items()):
            spent = totals.get(cat, {}).get("total", 0.0)
            pct = round(100 * spent / budget, 1)
            status = "over" if spent > budget else "warning" if pct >= WARNING_PCT else "ok"
            out.append({"category": cat, "spent": spent, "budget": budget, "percent_used": pct,
                        "status": status, "over_by": round(max(0, spent - budget), 2)})
        no_budget = sorted(set(totals) - set(self.store.budgets()))
        return {"month": month, "budgets": out, "categories_without_budget": no_budget}

    def find_transactions(self, month=None, category=None, merchant=None, min_amount=None):
        sql, params = "SELECT * FROM transactions WHERE 1=1", []
        if month is not None:
            sql += " AND month=?"; params.append(_check_month(month))
        if category is not None:
            if category not in CATEGORIES:
                raise ToolError(f"unknown category {category!r}")
            sql += " AND category=?"; params.append(category)
        if merchant:
            sql += " AND lower(description) LIKE ?"; params.append(f"%{str(merchant).lower()[:50]}%")
        if min_amount is not None:
            sql += " AND amount>=?"; params.append(float(min_amount))
        rows = self.store.query(sql + " ORDER BY amount DESC", params)
        return {"count": len(rows), "total": round(sum(r["amount"] for r in rows), 2),
                "transactions": [_txn_view(r) for r in rows[:20]]}

    def detect_anomalies(self, month):
        month = _check_month(month)
        all_rows = self.store.query("SELECT * FROM transactions")
        by_cat = {}
        for r in all_rows:
            by_cat.setdefault(r["category"], []).append(r)
        flagged = []
        for r in all_rows:
            if r["month"] != month:
                continue
            others = [o["amount"] for o in by_cat[r["category"]] if o["id"] != r["id"]]
            if len(others) < MIN_HISTORY:
                continue
            mean, sd = statistics.mean(others), statistics.pstdev(others)
            if sd == 0:
                continue
            z = (r["amount"] - mean) / sd
            if z >= ANOMALY_Z:
                flagged.append({**_txn_view(r), "z_score": round(z, 2),
                                "category_average": round(mean, 2)})
        flagged.sort(key=lambda x: -x["z_score"])
        return {"month": month, "threshold_z": ANOMALY_Z, "anomalies": flagged}

    def compare_months(self, month_a, month_b):
        a, b = self._totals(_check_month(month_a)), self._totals(_check_month(month_b))
        rows = []
        for cat in sorted(set(a) | set(b)):
            ta, tb = a.get(cat, {}).get("total", 0.0), b.get(cat, {}).get("total", 0.0)
            rows.append({"category": cat, month_a: ta, month_b: tb, "change": round(tb - ta, 2),
                         "change_pct": round(100 * (tb - ta) / ta, 1) if ta else None})
        rows.sort(key=lambda r: -abs(r["change"]))
        tot_a = round(sum(v["total"] for v in a.values()), 2)
        tot_b = round(sum(v["total"] for v in b.values()), 2)
        return {"month_a": month_a, "month_b": month_b, "total_a": tot_a, "total_b": tot_b,
                "total_change": round(tot_b - tot_a, 2), "by_category": rows}

    def get_review_queue(self):
        rows = self.store.query(
            "SELECT * FROM transactions WHERE confidence < ? ORDER BY date", (REVIEW_THRESHOLD,))
        return {"count": len(rows), "transactions": [
            {**_txn_view(r), "confidence": r["confidence"]} for r in rows]}

    def simulate_savings(self, cut_percent, category=None, merchant=None):
        cut = float(cut_percent)
        if not 0 < cut <= 100:
            raise ToolError("cut_percent must be between 1 and 100")
        if (category is None) == (not merchant):
            raise ToolError("give either a category or a merchant (exactly one)")
        if category is not None:
            if category not in CATEGORIES:
                raise ToolError(f"unknown category {category!r}")
            where, params, target = "category=?", [category], category
        else:
            merchant = str(merchant).strip().lower()[:50]
            where, params, target = "lower(description) LIKE ?", [f"%{merchant}%"], merchant
        months = self.store.months()
        if not months:
            raise ToolError("no transactions loaded")
        rows = self.store.query(
            f"SELECT month, category, SUM(amount) AS total, COUNT(*) AS n FROM transactions "
            f"WHERE {where} GROUP BY month, category", params)
        if not rows:
            return {"target": target, "error": "no matching transactions"}
        by_month = {m: 0.0 for m in months}
        for r in rows:
            by_month[r["month"]] += r["total"]
        baseline = sum(by_month.values()) / len(months)
        saving = baseline * cut / 100
        latest = months[-1]
        main_cat = category or max(rows, key=lambda r: r["total"])["category"]
        cat_spent = self._totals(latest).get(main_cat, {}).get("total", 0.0)
        new_spent = cat_spent - by_month[latest] * cut / 100
        budget = self.store.budgets().get(main_cat)

        def status(x):
            if budget is None:
                return "no budget"
            return "over" if x > budget else "warning" if 100 * x / budget >= WARNING_PCT else "ok"

        return {
            "target": target, "target_type": "category" if category else "merchant",
            "cut_percent": cut, "months_of_data": len(months),
            "average_monthly_spend": round(baseline, 2),
            "saving_per_month": round(saving, 2),
            "saving_3_months": round(saving * 3, 2), "saving_6_months": round(saving * 6, 2),
            "saving_12_months": round(saving * 12, 2),
            "latest_month": latest, "budget_category": main_cat, "budget": budget,
            "latest_category_spend": round(cat_spent, 2), "latest_category_spend_after_cut": round(new_spent, 2),
            "budget_status_before": status(cat_spent), "budget_status_after": status(new_spent),
        }

    # ----- write tools (approval required) -----
    def set_budget(self, category, amount):
        old = self.store.budgets().get(category)
        self.store.set_budget(category, amount)
        return {"category": category, "old_budget": old, "new_budget": float(amount)}

    def confirm_category(self, transaction_id, category, remember=True):
        return self.store.recategorize(int(transaction_id), category, bool(remember))
