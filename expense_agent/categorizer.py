"""Transaction categoriser: learned rules -> keyword rules -> LLM fallback.

Order matters:
1. Learned rules come from human corrections (feedback loop), so they win.
2. Keyword rules are free, fast and deterministic; they cover most merchants.
3. The LLM is only asked about merchants nobody recognises, which keeps
   API calls (and cost / rate-limit pressure on the free tier) low.
Anything below REVIEW_THRESHOLD confidence is sent to the human review queue.
"""
import re
from typing import Callable, Optional

CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Bills & Utilities",
    "Rent", "Entertainment", "Health", "Education", "Other",
]
REVIEW_THRESHOLD = 0.7

KEYWORD_RULES = {
    "Food & Dining": ["swiggy", "zomato", "starbucks", "dominos", "restaurant", "cafe", "mcdonalds", "kfc"],
    "Groceries": ["bigbasket", "blinkit", "zepto", "dmart", "reliance fresh", "grocery", "groceries"],
    "Transport": ["uber", "ola", "rapido", "metro", "irctc", "petrol", "indian oil", "fastag"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "decathlon", "nykaa"],
    "Bills & Utilities": ["electricity", "bescom", "airtel", "jio", "broadband", "water bill", "gas bill"],
    "Rent": ["rent"],
    "Entertainment": ["netflix", "spotify", "bookmyshow", "pvr", "hotstar", "prime video"],
    "Health": ["apollo", "pharmacy", "pharmeasy", "1mg", "clinic", "hospital"],
    "Education": ["coursera", "udemy", "tuition", "college fee"],
}
_COMPILED = {
    cat: [re.compile(rf"\b{re.escape(k)}\b") for k in kws] for cat, kws in KEYWORD_RULES.items()
}

# classifier(description) -> (category, confidence)
LLMClassifier = Callable[[str], tuple]


def merchant_key(description: str) -> str:
    """Normalised merchant key used for learned rules (first two words)."""
    words = re.findall(r"[a-z]+", description.lower())
    return " ".join(words[:2])


def categorize(description: str, learned: Optional[dict] = None,
               llm_classifier: Optional[LLMClassifier] = None) -> tuple:
    """Return (category, confidence, source)."""
    text = description.lower()
    for key, cat in (learned or {}).items():
        if key and key in text:
            return cat, 1.0, "human"
    for cat, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            return cat, 0.95, "rule"
    if llm_classifier is not None:
        try:
            cat, conf = llm_classifier(description)
            if cat in CATEGORIES:
                return cat, max(0.0, min(float(conf), 1.0)), "llm"
        except Exception:  # LLM failure must never break loading
            pass
    return "Other", 0.0, "none"
