"""OfflinePlanner: a rule-based stand-in for the LLM.

It lets the whole agent run with no API key (demo fallback, CI tests) and makes
the test results reproducible. It follows the same protocol as GeminiLLM: look
at the conversation, either request tool calls or return a final answer. It
only understands a fixed set of question types; Gemini handles free-form ones.
"""
import calendar
import re

from .agent import ToolCall, Turn

CATEGORY_WORDS = {
    "Food & Dining": ["food", "dining", "eating out", "swiggy", "zomato", "restaurant", "coffee"],
    "Groceries": ["grocer", "bigbasket", "blinkit"],
    "Transport": ["transport", "travel", "cab", "uber", "ola", "commute", "metro"],
    "Shopping": ["shopping", "amazon", "myntra", "clothes"],
    "Bills & Utilities": ["bill", "utilit", "electricity", "internet", "broadband", "phone"],
    "Rent": ["rent"],
    "Entertainment": ["entertainment", "movie", "netflix", "spotify", "subscription"],
    "Health": ["health", "medic", "pharmacy", "doctor"],
    "Education": ["education", "course", "tuition"],
}
MONTH_NAMES = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTH_NAMES.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def inr(x):
    """Format with Indian digit grouping: 123456.0 -> 'Rs. 1,23,456'."""
    neg, x = x < 0, abs(round(x))
    s = str(int(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"{'-' if neg else ''}Rs. {s}"


def month_label(m):
    y, mm = m.split("-")
    return f"{calendar.month_name[int(mm)]} {y}"


def detect_intent(q):
    q = q.lower()
    if "budget" in q and re.search(r"\b(set|change|increase|raise|lower|reduce|make)\b", q) and re.search(r"\d", q):
        return "set_budget"
    if re.search(r"transaction\s*#?\d+\s+(is|as|should be|to)\b", q):
        return "confirm"
    if re.search(r"what if|what-if|if i (cut|reduce|stop|halve|spend less)|cut (down )?(my |on )?\w+.* by|"
                 r"how much (would|could|will|can) i save|save if", q):
        return "whatif"
    if re.search(r"unusual|anomal|suspicious|strange|odd|outlier", q):
        return "anomalies"
    if re.search(r"compare|compared|\bvs\b|versus|than last|change from", q):
        return "compare"
    if re.search(r"review|uncategor|not sure|unsure|unknown", q):
        return "review"
    if re.search(r"budget|overspend|over spend|over budget|limit", q):
        return "budgets"
    if re.search(r"\b(list|show|which|biggest|largest|top)\b", q):
        return "find"
    return "summary"


def detect_category(q):
    q = q.lower()
    for cat, words in CATEGORY_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}", q) for w in words):
            return cat
    return None


def explicit_months(q, available):
    found = re.findall(r"\b(\d{4}-\d{2})\b", q)
    for word in re.findall(r"[a-z]+", q.lower()):
        if word in MONTH_NAMES:
            num = f"{MONTH_NAMES[word]:02d}"
            match = [m for m in available if m.endswith("-" + num)]
            if match:
                found.append(match[-1])
    seen = []
    for m in found:
        if m not in seen:
            seen.append(m)
    return seen


class OfflinePlanner:
    def next(self, history):
        last_user = max(i for i, h in enumerate(history) if h["role"] == "user")
        question = history[last_user]["text"]
        results = {}
        for h in history[last_user + 1:]:
            if h["role"] == "tool":
                for name, res in h["results"]:
                    results[name] = res
        intent = detect_intent(question)

        # Step 1: find out which months exist (unless already known)
        if "get_available_months" not in results:
            return Turn(tool_calls=[ToolCall("get_available_months", {})])
        available = results["get_available_months"]["months"]
        if not available:
            return Turn(text="There are no transactions loaded yet. Please upload a bank statement CSV.")
        named = explicit_months(question, available)
        if "last month" in question.lower() and len(available) >= 2:
            named = [available[-2]]
        month = named[0] if named else available[-1]

        # Step 2: call the tool that answers the question
        tool, args = {
            "summary": ("get_spending_summary", {"month": month}),
            "budgets": ("check_budgets", {"month": month}),
            "anomalies": ("detect_anomalies", {"month": month}),
            "review": ("get_review_queue", {}),
            "find": ("find_transactions", self._find_args(question, month)),
            "compare": ("compare_months", self._compare_args(named, available)),
            "set_budget": ("set_budget", self._budget_args(question)),
            "confirm": ("confirm_category", self._confirm_args(question)),
            "whatif": ("simulate_savings", self._whatif_args(question)),
        }[intent]
        if tool not in results:
            if args is None:
                return Turn(text="I could not work out the details. Try 'Set my Food budget to 7000', "
                                 "'Transaction 87 is Food & Dining' or 'What if I cut Swiggy by half?'.")
            return Turn(tool_calls=[ToolCall(tool, args)])

        # Step 3: write the answer from the tool result
        return Turn(text=self._compose(intent, question, results[tool]))

    # ----- argument helpers -----
    def _find_args(self, q, month):
        args = {"month": month}
        cat = detect_category(q)
        if cat:
            args["category"] = cat
        return args

    def _compare_args(self, named, available):
        if len(named) >= 2:
            a, b = sorted(named[:2])
        elif len(named) == 1 and named[0] != available[0]:
            b = named[0]
            a = available[available.index(b) - 1]
        else:
            a, b = (available[-2], available[-1]) if len(available) >= 2 else (available[-1],) * 2
        return {"month_a": a, "month_b": b}

    def _whatif_args(self, q):
        ql = q.lower()
        if re.search(r"\b(stop|quit|cancel|no more|completely)\b", ql):
            cut = 100.0
        elif re.search(r"\bhalf|halve\b", ql):
            cut = 50.0
        elif m := re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent)", ql):
            cut = float(m.group(1))
        elif "third" in ql:
            cut = 33.0
        elif "quarter" in ql:
            cut = 25.0
        else:
            cut = 50.0
        from .categorizer import KEYWORD_RULES
        merchants = sorted((k for ks in KEYWORD_RULES.values() for k in ks), key=len, reverse=True)
        for m in merchants:
            if len(m) > 3 and re.search(rf"\b{re.escape(m)}\b", ql):
                return {"cut_percent": cut, "merchant": m}
        cat = detect_category(q)
        if not cat:
            return None
        return {"cut_percent": cut, "category": cat}

    def _confirm_args(self, q):
        m = re.search(r"transaction\s*#?(\d+)", q.lower())
        cat = detect_category(q)
        if not m or not cat:
            return None
        return {"transaction_id": int(m.group(1)), "category": cat, "remember": True}

    def _budget_args(self, q):
        cat = detect_category(q)
        nums = re.findall(r"\d[\d,]*", q)
        if not cat or not nums:
            return None
        return {"category": cat, "amount": float(nums[-1].replace(",", ""))}

    # ----- answer writer -----
    def _compose(self, intent, q, r):
        if "error" in r:
            return f"I could not complete that: {r['error']}."
        if r.get("status") == "pending_approval":
            p = r["proposed"]
            if r["tool"] == "set_budget":
                what = f"set your {p['category']} budget to {inr(p['amount'])} a month"
            else:
                what = f"mark transaction #{p['transaction_id']} as {p['category']} and remember this merchant"
            return f"I've prepared this change: {what}. Please approve or reject it below; nothing has changed yet."
        if intent == "summary":
            label, cat = month_label(r["month"]), detect_category(q)
            if cat:
                c = r["by_category"].get(cat)
                if not c:
                    return f"You had no {cat} spending in {label}."
                share = 100 * c["total"] / r["total"]
                return (f"In {label} you spent {inr(c['total'])} on {cat} across {c['count']} "
                        f"transactions, which is {share:.0f}% of your total spending of {inr(r['total'])}.")
            top = list(r["by_category"].items())[:4]
            parts = ", ".join(f"{k} {inr(v['total'])}" for k, v in top)
            return f"In {label} you spent {inr(r['total'])} in total. The biggest categories were {parts}."
        if intent == "budgets":
            over = [b for b in r["budgets"] if b["status"] == "over"]
            warn = [b for b in r["budgets"] if b["status"] == "warning"]
            label = month_label(r["month"])
            if not over and not warn:
                return f"Good news: every category is within budget for {label}."
            lines = [f"Budget check for {label}:"]
            for b in over:
                lines.append(f"- {b['category']}: OVER by {inr(b['over_by'])} "
                             f"(spent {inr(b['spent'])} of {inr(b['budget'])}, {b['percent_used']}%).")
            for b in warn:
                lines.append(f"- {b['category']}: close to the limit, {b['percent_used']}% used "
                             f"({inr(b['spent'])} of {inr(b['budget'])}).")
            worst = max(over or warn, key=lambda b: b["percent_used"])
            lines.append("")
            lines.append(f"Suggestion: look at your {worst['category']} transactions first; "
                         "that is where the biggest overrun is.")
            return "\n".join(lines)
        if intent == "anomalies":
            label = month_label(r["month"])
            if not r["anomalies"]:
                return f"Nothing unusual found in {label}."
            lines = [f"I found {len(r['anomalies'])} unusual transaction(s) in {label}:"]
            for a in r["anomalies"]:
                lines.append(f"- {a['date']}: {inr(a['amount'])} at \"{a['description_data']}\" ({a['category']}). "
                             f"Your usual {a['category']} transaction is about {inr(a['category_average'])} "
                             f"(z-score {a['z_score']}).")
            lines.append("")
            lines.append("If you do not recognise any of these, check with your bank.")
            return "\n".join(lines)
        if intent == "compare":
            a, b = r["month_a"], r["month_b"]
            direction = "more" if r["total_change"] > 0 else "less"
            lines = [f"You spent {inr(abs(r['total_change']))} {direction} in {month_label(b)} "
                     f"({inr(r['total_b'])}) than in {month_label(a)} ({inr(r['total_a'])})."]
            for c in r["by_category"][:3]:
                if c["change"]:
                    pct = f" ({c['change_pct']:+.0f}%)" if c["change_pct"] is not None else ""
                    lines.append(f"- {c['category']}: {'up' if c['change'] > 0 else 'down'} "
                                 f"{inr(abs(c['change']))}{pct}")
            return "\n".join(lines)
        if intent == "review":
            if not r["count"]:
                return "Nothing needs review: every transaction has a confident category."
            lines = [f"{r['count']} transaction(s) need you to confirm the category:"]
            for t in r["transactions"]:
                lines.append(f"- #{t['id']} {t['date']} {inr(t['amount'])} \"{t['description_data']}\" "
                             f"(currently {t['category']})")
            lines.append("")
            lines.append("Tell me the right category, e.g. 'transaction 12 is Food & Dining'.")
            return "\n".join(lines)
        if intent == "find":
            if not r["count"]:
                return "No matching transactions."
            lines = [f"{r['count']} matching transaction(s), {inr(r['total'])} in total. Largest:"]
            for t in r["transactions"][:5]:
                lines.append(f"- {t['date']} {inr(t['amount'])} {t['description_data']}")
            return "\n".join(lines)
        if intent == "whatif":
            name = r["target"].title() if r["target_type"] == "merchant" else r["target"]
            action = (f"stopped {name} completely" if r["cut_percent"] >= 100
                      else f"cut {name} spending by {r['cut_percent']:.0f}%")
            lines = [f"If you {action}, you would save about "
                     f"{inr(r['saving_per_month'])} a month: {inr(r['saving_3_months'])} in 3 months, "
                     f"{inr(r['saving_6_months'])} in 6 months and {inr(r['saving_12_months'])} in a year.",
                     f"This is based on your average of {inr(r['average_monthly_spend'])} a month "
                     f"over {r['months_of_data']} month(s)."]
            if r["budget"]:
                before, after = r["budget_status_before"], r["budget_status_after"]
                verdict = {("over", "ok"): "back within budget", ("over", "warning"): "back within budget, but close to the limit",
                           ("over", "over"): "still over budget"}.get(
                    (before, after), {"ok": "within budget", "warning": "close to the limit",
                                      "over": "over budget"}.get(after, after))
                lines.append(f"In {month_label(r['latest_month'])}, {r['budget_category']} would drop from "
                             f"{inr(r['latest_category_spend'])} to {inr(r['latest_category_spend_after_cut'])}, "
                             f"which is {verdict} ({inr(r['budget'])}).")
            return "\n\n".join(lines)
        if intent == "confirm":
            n = r["updated"]
            return (f"Saved. I updated {n} transaction(s) and will remember this merchant "
                    "for future statements.")
        if intent == "set_budget":
            old = inr(r["old_budget"]) if r.get("old_budget") else "none"
            return f"Done. {r['category']} budget changed from {old} to {inr(r['new_budget'])} per month."
        return str(r)
