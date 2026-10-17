"""SQLite storage plus validated CSV import.

All SQL uses parameter binding (no string formatting), and every CSV row is
validated before it is stored, so a malformed or hostile statement file cannot
corrupt the database or the agent's numbers.
"""
import csv
import json
import re
import sqlite3
from datetime import datetime

from .categorizer import CATEGORIES, categorize, merchant_key

MAX_ROWS = 10_000
MAX_AMOUNT = 10_000_000
MAX_DESC_LEN = 200
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CSVFormatError(ValueError):
    pass


def clean_description(text: str) -> str:
    return _CONTROL.sub(" ", text or "").strip()[:MAX_DESC_LEN]


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS transactions(
                id INTEGER PRIMARY KEY, date TEXT NOT NULL, month TEXT NOT NULL,
                description TEXT NOT NULL, amount REAL NOT NULL,
                category TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS budgets(category TEXT PRIMARY KEY, amount REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS learned_rules(keyword TEXT PRIMARY KEY, category TEXT NOT NULL);
            """
        )

    # ---------- loading ----------
    def load_csv(self, path, llm_classifier=None) -> dict:
        loaded, skipped = 0, []
        learned = self.learned_rules()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.strip().lower() for c in (reader.fieldnames or [])}
            missing = {"date", "description", "amount"} - cols
            if missing:
                raise CSVFormatError(f"CSV is missing column(s): {sorted(missing)}")
            for line_no, raw in enumerate(reader, start=2):
                if loaded >= MAX_ROWS:
                    skipped.append((line_no, "row limit reached"))
                    break
                row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                try:
                    date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                except ValueError:
                    skipped.append((line_no, f"bad date {row['date']!r}"))
                    continue
                try:
                    amount = float(row["amount"].replace(",", ""))
                except ValueError:
                    skipped.append((line_no, f"bad amount {row['amount']!r}"))
                    continue
                if not (0 < amount <= MAX_AMOUNT):
                    skipped.append((line_no, f"amount out of range {amount}"))
                    continue
                desc = clean_description(row["description"])
                if not desc:
                    skipped.append((line_no, "empty description"))
                    continue
                cat, conf, src = categorize(desc, learned, llm_classifier)
                self.conn.execute(
                    "INSERT INTO transactions(date, month, description, amount, category, confidence, source)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (date.isoformat(), date.strftime("%Y-%m"), desc, round(amount, 2), cat, conf, src),
                )
                loaded += 1
        self.conn.commit()
        return {"loaded": loaded, "skipped": skipped}

    def load_budgets(self, path) -> None:
        with open(path, encoding="utf-8") as f:
            for cat, amt in json.load(f).items():
                self.set_budget(cat, amt)

    # ---------- writes ----------
    def set_budget(self, category: str, amount: float) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        amount = float(amount)
        if not (0 < amount <= MAX_AMOUNT):
            raise ValueError("budget must be between 0 and 1,00,00,000")
        self.conn.execute(
            "INSERT INTO budgets(category, amount) VALUES (?,?) "
            "ON CONFLICT(category) DO UPDATE SET amount=excluded.amount",
            (category, amount),
        )
        self.conn.commit()

    def recategorize(self, txn_id: int, category: str, remember: bool) -> dict:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        row = self.conn.execute("SELECT * FROM transactions WHERE id=?", (txn_id,)).fetchone()
        if row is None:
            raise ValueError(f"no transaction with id {txn_id}")
        self.conn.execute(
            "UPDATE transactions SET category=?, confidence=1.0, source='human' WHERE id=?",
            (category, txn_id),
        )
        updated = 1
        if remember:
            key = merchant_key(row["description"])
            self.conn.execute(
                "INSERT INTO learned_rules(keyword, category) VALUES (?,?) "
                "ON CONFLICT(keyword) DO UPDATE SET category=excluded.category",
                (key, category),
            )
            # apply the new rule to other low-confidence rows from the same merchant
            cur = self.conn.execute(
                "UPDATE transactions SET category=?, confidence=1.0, source='human' "
                "WHERE confidence < 0.7 AND lower(description) LIKE ?",
                (category, f"%{key}%"),
            )
            updated += cur.rowcount
        self.conn.commit()
        return {"updated": updated}

    # ---------- reads ----------
    def learned_rules(self) -> dict:
        return {r["keyword"]: r["category"] for r in self.conn.execute("SELECT * FROM learned_rules")}

    def budgets(self) -> dict:
        return {r["category"]: r["amount"] for r in self.conn.execute("SELECT * FROM budgets")}

    def months(self) -> list:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT month FROM transactions ORDER BY month")]

    def query(self, sql: str, params=()) -> list:
        return [dict(r) for r in self.conn.execute(sql, params)]
