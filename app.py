"""Streamlit web front end for the Expense Assistant Agent.   Run:  streamlit run app.py

Pages (tabs):
  Overview      - hero with an "ask the agent" bar and this month's receipt, insight strip,
                  budget cards, charts and unusual transactions
  Ask the agent - chat; each answer shows the agent's tool steps as a timeline;
                  data changes wait for Approve / Reject buttons (human in the loop)
  What-if       - interactive savings simulator (same tool the agent uses)
  Review        - confirm uncertain categories; the agent learns each merchant

Every figure on screen comes from the same tool functions the agent calls.
"""
import html
import json
import os
import re
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

from expense_agent import SAMPLE_CSV, build_agent, load_env
from expense_agent.categorizer import CATEGORIES, merchant_key
from expense_agent.offline import inr as _inr, month_label

st.set_page_config(page_title="Expense Assistant Agent", page_icon="💸", layout="wide",
                   initial_sidebar_state="collapsed")
load_env()

# ---------------- design tokens ----------------
PAPER, INK, INK2, MUTED = "#f6f2ea", "#17302e", "#4c5a58", "#8a8f8b"
TEAL, TEAL_SOFT, TEAL_MID, LINE = "#0f5a54", "#e3eeeb", "#9cc5bf", "#e6dfd2"
STATUS = {"ok": ("#2f8a3e", "On track"), "warning": ("#c98500", "Near limit"),
          "over": ("#c23b35", "Over budget")}


def inr(x):
    return _inr(x).replace("Rs. ", "₹")


def show(text):
    """Display agent text with the rupee sign (tools and agent use 'Rs.' in plain text)."""
    return text.replace("Rs. ", "₹")


ICON_PATHS = {
    "Food & Dining": '<path d="M7 3v8M5 3v5a2 2 0 0 0 4 0V3M7 11v10M17 3c-2 0-3 2-3 5s1 4 3 4v9"/>',
    "Groceries": '<circle cx="9" cy="20" r="1.3"/><circle cx="18" cy="20" r="1.3"/><path d="M2 3h3l2.4 11.5h11.2L21 7H6"/>',
    "Transport": '<path d="M5 17h14l-1.5-6.5A2 2 0 0 0 15.6 9H8.4a2 2 0 0 0-1.9 1.5z"/><circle cx="7.5" cy="17.5" r="1.6"/><circle cx="16.5" cy="17.5" r="1.6"/>',
    "Shopping": '<path d="M5 8h14l-1 13H6z"/><path d="M9 8V6a3 3 0 0 1 6 0v2"/>',
    "Bills & Utilities": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "Rent": '<path d="M3 10.5 12 3l9 7.5V21H3z"/><path d="M9 21v-6h6v6"/>',
    "Entertainment": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "Health": '<path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.5-7 10-7 10z"/>',
    "Education": '<path d="M4 5h6a2 2 0 0 1 2 2v13a2 2 0 0 0-2-2H4zM20 5h-6a2 2 0 0 0-2 2v13a2 2 0 0 1 2-2h6z"/>',
    "Other": '<circle cx="12" cy="12" r="8"/><path d="M8 12h8"/>',
}


def icon(cat, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{TEAL}" '
            f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{ICON_PATHS.get(cat, ICON_PATHS["Other"])}</svg>')


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea {{font-family: 'Inter', system-ui, sans-serif;}}
.stApp {{background: {PAPER};}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 1.2rem; max-width: 1180px;}}
h1, h2, h3, .serif {{font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em;}}
p, li {{color: {INK};}}
a {{color: {TEAL};}}
:focus-visible {{outline: 2px solid {TEAL} !important; outline-offset: 2px;}}

/* top bar */
.brand {{display:flex; align-items:center; gap:10px; font-family:'Fraunces',serif; font-size:1.45rem;
        font-weight:600; color:{INK}; padding-top:4px;}}
.brand .mark {{width:30px; height:22px; background:{TEAL}; border-radius:5px; position:relative;}}
.brand .mark:before, .brand .mark:after {{content:""; position:absolute; top:7px; width:8px; height:8px;
        border-radius:50%; background:{PAPER};}}
.brand .mark:before {{left:-4px;}} .brand .mark:after {{right:-4px;}}

[data-testid="stToolbar"] {{display: none;}}
/* tabs as navigation */
div[data-baseweb="tab-list"] {{gap: 28px; border-bottom: 1px solid {LINE};}}
button[data-baseweb="tab"] {{background: transparent !important; padding: 10px 2px !important;}}
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] [role="tab"] [data-testid="stMarkdownContainer"] p {{font-size: 1.06rem !important;
        color: {INK2}; font-weight: 500; line-height: 1.6;}}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{color: {INK} !important;}}
button[data-baseweb="tab"][aria-selected="true"] p {{color: {INK};}}
div[data-baseweb="tab-highlight"] {{background: {TEAL} !important; height: 2px !important;}}
div[data-baseweb="tab-border"] {{display:none;}}

/* buttons */
.stButton button, .stDownloadButton button, div[data-testid="stPopover"] button {{border-radius: 10px;
        border: 1px solid {LINE}; background: #fff; color: {INK}; font-weight: 500;}}
.stButton button:hover {{border-color: {TEAL}; color: {TEAL};}}
button[data-testid^="stBaseButton-primary"], div[data-testid="stFormSubmitButton"] button {{background: {TEAL} !important;
        color: #fff !important; border: none !important; border-radius: 10px; font-weight: 600;}}
button[data-testid^="stBaseButton-primary"] p, div[data-testid="stFormSubmitButton"] button p {{color: #fff !important;
        font-weight: 600;}}
button[data-testid^="stBaseButton-primary"]:hover, div[data-testid="stFormSubmitButton"] button:hover {{background: #0b4843 !important;}}
.st-key-answer {{border-left: 4px solid {TEAL} !important;}}

/* hero */
.kicker {{color: {TEAL}; font-weight: 600; font-size: .8rem; letter-spacing: .14em; text-transform: uppercase;
         margin: 26px 0 10px 0;}}
.hero-title {{font-family: 'Fraunces', serif; font-size: clamp(2.4rem, 4.6vw, 3.9rem); line-height: 1.04;
             font-weight: 600; color: {INK}; margin: 0 0 18px 0;}}
.hero-title em {{color: {TEAL}; font-style: italic; font-weight: 500;}}
.hero-sub {{font-size: 1.12rem; line-height: 1.6; color: {INK2}; max-width: 34em; margin-bottom: 22px;}}

/* ask bar (a styled st.form) */
div[data-testid="stForm"] {{background:#fff; border:1px solid {LINE}; border-radius:14px; padding: 10px 10px 10px 18px;
        box-shadow: 0 10px 30px -18px rgba(23,48,46,.35);}}
div[data-testid="stForm"] input {{font-size: 1.05rem; border: none !important; background: transparent !important;}}
div[data-testid="stForm"] div[data-baseweb="input"] {{border: none !important; background: transparent !important;}}
div[data-testid="stForm"] label p {{font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:{INK2};
        font-weight:600;}}
.try {{color: {INK2}; font-size: .92rem; margin: 2px 0 0 0;}}

/* receipt */
.receipt-wrap {{padding: 22px 6px 0 18px;}}
.receipt {{background:#fffdf8; border-radius: 6px 6px 0 0; padding: 26px 28px 20px; position: relative;
          box-shadow: 0 24px 50px -28px rgba(23,48,46,.45); transform: rotate(1.4deg);
          font-family: 'Inter', sans-serif; color: {INK};}}
.receipt:after {{content:""; position:absolute; left:0; right:0; bottom:-12px; height:12px;
          background: linear-gradient(-45deg, transparent 8px, #fffdf8 0) 0 0 / 16px 12px repeat-x,
                      linear-gradient(45deg, transparent 8px, #fffdf8 0) 0 0 / 16px 12px repeat-x;}}
.receipt .r-head {{display:flex; justify-content:space-between; font-size:.8rem; color:{MUTED};
          letter-spacing:.08em; text-transform: uppercase;}}
.receipt .r-total {{font-family:'Fraunces',serif; font-size: 3rem; font-weight:600; line-height:1; margin: 14px 0 4px;}}
.receipt .r-delta {{font-size:.9rem; color:{INK2}; margin-bottom: 16px;}}
.receipt .r-row {{display:flex; align-items:baseline; gap:8px; font-size:.95rem; padding: 5px 0;}}
.receipt .r-row .dots {{flex:1; border-bottom: 1.5px dotted #cfc6b5; transform: translateY(-4px);}}
.receipt .r-cut {{border-top: 1.5px dashed #d8cfbf; margin: 14px -28px 12px;}}
.receipt .r-foot {{display:flex; justify-content:space-between; align-items:center; font-size:.85rem; color:{INK2};}}
.stamp {{border: 2.5px solid {STATUS['over'][0]}; color: {STATUS['over'][0]}; border-radius: 8px; padding: 4px 10px;
        font-weight: 700; letter-spacing: .06em; text-transform: uppercase; font-size: .78rem;
        transform: rotate(-8deg); display:inline-block; opacity:.9;}}
.stamp.ok {{border-color: {STATUS['ok'][0]}; color: {STATUS['ok'][0]};}}

/* insight strip */
.strip {{margin: 34px calc(50% - 50vw) 0; border-top:1px solid {LINE}; border-bottom:1px solid {LINE};
        background: #fbf8f2; overflow:hidden; white-space:nowrap;}}
.strip .track {{display:inline-block; padding: 14px 0; animation: scroll 60s linear infinite;}}
.strip:hover .track {{animation-play-state: paused;}}
.strip span.item {{margin: 0 26px; color: {INK2}; font-size: .97rem;}}
.strip b {{color: {TEAL}; font-weight: 600;}}
.strip .dot {{color: #d39a3a;}}
@keyframes scroll {{from {{transform: translateX(0);}} to {{transform: translateX(-50%);}}}}
@media (prefers-reduced-motion: reduce) {{.strip .track {{animation: none;}}}}

/* sections & cards */
.sec {{display:flex; justify-content:space-between; align-items:flex-end; margin: 46px 0 16px;}}
.sec h2 {{font-size: 2rem; margin:0; font-weight:600;}}
.sec h2 em {{color:{TEAL}; font-weight:500;}}
.sec .note {{color:{INK2}; font-size:.95rem;}}
.cards {{display:grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 16px;}}
.card {{background:#fff; border:1px solid {LINE}; border-radius:14px; padding: 18px 20px;}}
.card .name {{font-weight:600; font-size:1.05rem; margin: 12px 0 4px; color:{INK};}}
.card .amt {{color:{INK2}; font-size:.92rem;}}
.card .bar {{height:6px; border-radius:3px; background:#efe9de; margin: 12px 0 8px; overflow:hidden;}}
.card .bar i {{display:block; height:100%; border-radius:3px;}}
.card .st {{font-size:.82rem; font-weight:600;}}
.odd {{display:flex; gap:16px; align-items:flex-start;}}
.odd .big {{font-family:'Fraunces',serif; font-size:1.7rem; font-weight:600; color:{INK}; min-width: 120px;}}
.odd .why {{color:{INK2}; font-size:.92rem;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background:#fff; border-radius:14px; border-color:{LINE} !important;}}
div[data-testid="stMetric"] {{background:#fff; border:1px solid {LINE}; border-radius:14px; padding: 14px 18px;}}
div[data-testid="stMetricValue"] {{font-family:'Fraunces',serif; color:{INK};}}

/* agent */
.answer-card {{background:#fff; border:1px solid {LINE}; border-left: 4px solid {TEAL}; border-radius: 12px;
        padding: 16px 20px; margin-top: 16px;}}
.answer-card .q {{color:{INK2}; font-size:.9rem; margin-bottom:6px;}}
.step {{border-left: 2px solid {TEAL_MID}; padding: 4px 0 10px 14px; margin-left: 6px; position: relative;}}
.step:before {{content:""; position:absolute; left:-6px; top:8px; width:10px; height:10px; border-radius:50%;
        background:{TEAL}; box-shadow: 0 0 0 3px #fff;}}
.step.err:before {{background:#c98500;}} .step.ans:before {{background:#2f8a3e;}}
.step code, .tool {{font-size:.84rem; background:{TEAL_SOFT}; color:{TEAL}; padding:1px 7px; border-radius:6px;}}
.step .sum {{color:{INK2}; font-size:.9rem;}}
.approve {{background:#fff8ea; border:1px solid #f0d9a8; border-radius:12px; padding:12px 16px; margin: 8px 0;}}
div[data-testid="stChatMessage"] {{background: transparent;}}
</style>
""", unsafe_allow_html=True)


# ---------------- data / agent setup ----------------
def ui_approver(name, args):
    """Called by the agent before a data change. Approved actions run; others wait for buttons."""
    key = name + json.dumps(args, sort_keys=True)
    if key in st.session_state.get("approved_keys", set()):
        return True
    st.session_state.setdefault("new_pending", []).append({"name": name, "args": args})
    return None


def load(path, use_gemini, source):
    agent, store, report = build_agent(path, offline=not use_gemini, approve=ui_approver)
    st.session_state.update(agent=agent, store=store, report=report, chat=[], source=source,
                            approved_keys=set(), pending=[], last_answer=None)


has_key = bool(os.environ.get("GEMINI_API_KEY"))
if "use_gemini" not in st.session_state:
    st.session_state.use_gemini = has_key
if "agent" not in st.session_state:
    load(str(SAMPLE_CSV), st.session_state.use_gemini, "Sample statement")

# ---------------- top bar ----------------
bar_l, bar_r = st.columns([3, 2], vertical_alignment="center")
bar_l.markdown('<div class="brand"><span class="mark"></span>Expense Assistant</div>', unsafe_allow_html=True)
with bar_r:
    b1, b2 = st.columns([1.25, 1])
    with b1.popover("Upload statement", width="stretch"):
        st.markdown("**Your bank statement (CSV)**")
        st.caption("Columns: date (YYYY-MM-DD), description, amount. Rows that don't fit are skipped and listed.")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        st.toggle("Use Gemini", key="use_gemini", disabled=not has_key,
                  help="Needs GEMINI_API_KEY. Off = rule-based offline planner.")
        if st.button("Analyse this file", type="primary", disabled=up is None, width="stretch"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(up.getvalue()); tmp.close()
            load(tmp.name, st.session_state.use_gemini, up.name)
            st.rerun()
    if b2.button("Use sample data", width="stretch"):
        load(str(SAMPLE_CSV), st.session_state.use_gemini, "Sample statement")
        st.rerun()

store, agent = st.session_state.store, st.session_state.agent
tools = agent.tools
months = store.months()
report = st.session_state.report
model = getattr(agent.llm, "model", None) or "offline planner"

if not months:
    st.warning("No transactions could be read from this file. Check that it has date, description "
               "and amount columns, then upload it again.")
    st.stop()

queue_count = tools.get_review_queue()["count"]
tab_home, tab_chat, tab_what, tab_review = st.tabs(
    ["Overview", "Ask the agent", "What-if", f"Review ({queue_count})" if queue_count else "Review"])


def axis_cfg(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure(font="Inter")
            .configure_axis(gridColor="#ece6da", domainColor="#d8cfbf", labelColor=MUTED, titleColor=INK2,
                            labelFontSize=12, titleFontSize=12, tickColor="#d8cfbf"))


# ================= shared agent helpers =================
TOOL_SUMMARY = {
    "get_available_months": lambda r: f"latest month is {month_label(r['latest'])}" if r.get("latest") else "",
    "get_spending_summary": lambda r: f"total {inr(r['total'])} across {len(r['by_category'])} categories" if "total" in r else "",
    "check_budgets": lambda r: f"{sum(b['status'] == 'over' for b in r['budgets'])} over budget, "
                               f"{sum(b['status'] == 'warning' for b in r['budgets'])} near the limit",
    "find_transactions": lambda r: f"{r['count']} transactions, {inr(r['total'])}",
    "detect_anomalies": lambda r: f"{len(r['anomalies'])} flagged",
    "compare_months": lambda r: f"change of {inr(r['total_change'])}",
    "get_review_queue": lambda r: f"{r['count']} need review",
    "simulate_savings": lambda r: f"saves {inr(r['saving_per_month'])} a month" if "saving_per_month" in r else "",
    "set_budget": lambda r: "waiting for your approval" if r.get("status") else f"budget is now {inr(r['new_budget'])}",
    "confirm_category": lambda r: "waiting for your approval" if r.get("status") else f"{r['updated']} updated",
}


def render_trace(trace):
    parts = []
    for t in trace:
        if t["type"] == "tool":
            res = t["result"]
            if "error" in res:
                summary = res["error"]
            else:
                try:
                    summary = TOOL_SUMMARY.get(t["name"], lambda r: "")(res)
                except Exception:
                    summary = ""
            args = ", ".join(f"{k}={v}" for k, v in t["args"].items())
            parts.append(f'<div class="step"><b>Step {t["step"]}</b>&nbsp; <span class="tool">{t["name"]}'
                         f'({html.escape(args)})</span><br><span class="sum">{html.escape(summary)}</span></div>')
        elif t["type"] == "llm_error":
            parts.append(f'<div class="step err"><b>Step {t["step"]}</b>&nbsp; Gemini failed, so the offline '
                         f'planner took over<br><span class="sum">{html.escape(t["text"][:160])}</span></div>')
        elif t["type"] == "answer":
            parts.append(f'<div class="step ans"><b>Step {t["step"]}</b>&nbsp; wrote the answer from these results</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def ask(question):
    st.session_state.new_pending = []
    with st.spinner("The agent is checking your statement…"):
        result = agent.run(question)
    st.session_state.chat.append({"role": "user", "text": question})
    st.session_state.chat.append({"role": "assistant", "text": result.answer, "trace": result.trace})
    st.session_state.last_answer = {"q": question, "a": result.answer, "trace": result.trace}
    for p in st.session_state.new_pending:
        p["id"] = len(st.session_state.chat) * 100 + len(st.session_state.pending)
        st.session_state.pending.append(p)


def describe(p):
    a = p["args"]
    if p["name"] == "set_budget":
        old = store.budgets().get(a.get("category"))
        return (f"Set the **{a.get('category')}** budget to **{inr(float(a.get('amount', 0)))}** a month"
                + (f" (now {inr(old)})" if old else ""))
    return f"Mark transaction **#{a.get('transaction_id')}** as **{a.get('category')}** and remember this merchant"


def approval_cards(prefix):
    for p in list(st.session_state.pending):
        detail = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(describe(p)))
        st.markdown(f'<div class="approve"><b>Waiting for your approval.</b> The agent wants to:<br>{detail}</div>',
                    unsafe_allow_html=True)
        c1, c2, _ = st.columns([0.9, 0.9, 5])
        if c1.button("Approve", key=f"{prefix}ok{p['id']}", type="primary"):
            res = tools.execute(p["name"], p["args"])
            st.session_state.approved_keys.add(p["name"] + json.dumps(p["args"], sort_keys=True))
            st.session_state.pending.remove(p)
            text = (f"Couldn't apply the change: {res['error']}" if "error" in res else
                    (f"Approved. {res['category']} budget is now {inr(res['new_budget'])}."
                     if p["name"] == "set_budget" else
                     f"Approved. Updated {res['updated']} transaction(s) and learned the merchant."))
            st.session_state.chat.append({"role": "assistant", "text": text})
            st.session_state.last_answer = {"q": "Approval", "a": text, "trace": None}
            st.rerun()
        if c2.button("Reject", key=f"{prefix}no{p['id']}"):
            st.session_state.pending.remove(p)
            st.session_state.chat.append({"role": "assistant", "text": "Rejected. Nothing was changed."})
            st.session_state.last_answer = {"q": "Approval", "a": "Rejected. Nothing was changed.", "trace": None}
            st.rerun()


# ================= OVERVIEW =================
with tab_home:
    month = months[-1]
    if len(months) > 1:
        month = st.session_state.get("ov_month", months[-1])
    summ = tools.get_spending_summary(month)
    budgets = tools.check_budgets(month)
    anomalies = tools.detect_anomalies(month)["anomalies"]
    total = summ.get("total", 0.0)
    idx = months.index(month)
    prev_total = tools.get_spending_summary(months[idx - 1]).get("total", 0.0) if idx > 0 else None
    over = [b for b in budgets["budgets"] if b["status"] == "over"]

    hero_l, hero_r = st.columns([1.25, 1], gap="large")
    with hero_l:
        span = f"{month_label(months[0])} – {month_label(months[-1])}" if len(months) > 1 else month_label(months[0])
        st.markdown(f"""
            <div class="kicker">{html.escape(st.session_state.source)} · {span}</div>
            <div class="hero-title">See where your money <em>actually</em> went.</div>
            <div class="hero-sub">Ask in plain English. The agent checks your budgets, spots unusual spending
            and works out how to save, with every number calculated from your statement.</div>""",
                    unsafe_allow_html=True)
        with st.form("hero_ask", clear_on_submit=True, border=False):
            fc1, fc2 = st.columns([4, 1.1], vertical_alignment="bottom")
            q = fc1.text_input("Ask the agent", placeholder="e.g. Am I over budget this month?")
            sent = fc2.form_submit_button("Ask", width="stretch")
        st.markdown('<p class="try">Try:</p>', unsafe_allow_html=True)
        chips = {"Over budget?": "Am I over budget this month?",
                 "Halve Swiggy?": "What if I cut Swiggy by half?",
                 "Anything unusual?": "Any unusual transactions?"}
        cc = st.columns(len(chips))
        for i, (label, question) in enumerate(chips.items()):
            if cc[i].button(label, key=f"chip{i}", width="stretch"):
                q, sent = question, True
        if sent and q and q.strip():
            ask(q.strip())
            st.rerun()

    with hero_r:
        rows = list(summ.get("by_category", {}).items())[:5]
        row_html = "".join(f'<div class="r-row"><span>{html.escape(k)}</span><span class="dots"></span>'
                           f'<span>{inr(v["total"])}</span></div>' for k, v in rows)
        delta = ""
        if prev_total:
            pct = 100 * (total - prev_total) / prev_total
            delta = f"{'up' if pct > 0 else 'down'} {abs(pct):.0f}% on {month_label(months[idx - 1]).split()[0]}"
        stamp = (f'<span class="stamp">{len(over)} over budget</span>' if over
                 else '<span class="stamp ok">All within budget</span>')
        st.markdown(f"""
          <div class="receipt-wrap"><div class="receipt" role="img"
               aria-label="Spending receipt for {month_label(month)}: total {inr(total)}">
            <div class="r-head"><span>Statement summary</span><span>{month_label(month)}</span></div>
            <div class="r-total">{inr(total)}</div>
            <div class="r-delta">{summ.get('by_category') and sum(v['count'] for v in summ['by_category'].values())} transactions{(' · ' + delta) if delta else ''}</div>
            {row_html}
            <div class="r-cut"></div>
            <div class="r-foot"><span>{len(anomalies)} unusual · {queue_count} to review</span>{stamp}</div>
          </div></div>""", unsafe_allow_html=True)

    if st.session_state.get("last_answer"):
        la = st.session_state.last_answer
        with st.container(border=True, key="answer"):
            if la["q"] != "Approval":
                st.markdown(f'<div class="q" style="color:{INK2};font-size:.9rem">You asked: '
                            f'{html.escape(la["q"])}</div>', unsafe_allow_html=True)
            st.markdown(show(la["a"]))
            approval_cards("home")
            if la.get("trace"):
                with st.expander("How the agent answered"):
                    render_trace(la["trace"])

    # insight strip
    items = []
    for b in over:
        items.append(f"<b>{html.escape(b['category'])}</b> at {b['percent_used']:.0f}% of budget")
    for a in anomalies:
        items.append(f"<b>{int(a['date'][8:])} {month_label(a['date'][:7]).split()[0][:3]}</b> "
                     f"{inr(a['amount'])} {html.escape(a['description_data'][:40])} looks unusual")
    if prev_total:
        items.append(f"<b>{month_label(month).split()[0]}</b> total {inr(total)} vs {inr(prev_total)} the month before")
    biggest = max(summ.get("by_category", {}).items(), key=lambda kv: kv[1]["total"], default=None)
    if biggest:
        items.append(f"<b>{html.escape(biggest[0])}</b> is your biggest category at {inr(biggest[1]['total'])}")
    if queue_count:
        items.append(f"<b>{queue_count}</b> transactions need a category from you")
    strip = '<span class="dot">•</span>'.join(f'<span class="item">{t}</span>' for t in items)
    st.markdown(f'<div class="strip" aria-label="Highlights"><div class="track">{strip}<span class="dot">•</span>'
                f'{strip}<span class="dot">•</span></div></div>', unsafe_allow_html=True)

    # budgets
    s1, s2 = st.columns([3, 1], vertical_alignment="bottom")
    s1.markdown(f'<div class="sec"><h2>Budgets in <em>{month_label(month).split()[0]}</em></h2></div>',
                unsafe_allow_html=True)
    if len(months) > 1:
        s2.selectbox("Month", months[::-1], format_func=month_label, key="ov_month")
    cards = ""
    for b in sorted(budgets["budgets"], key=lambda b: -b["percent_used"]):
        color, label = STATUS[b["status"]]
        cards += f"""<div class="card">{icon(b['category'])}
            <div class="name">{html.escape(b['category'])}</div>
            <div class="amt">{inr(b['spent'])} of {inr(b['budget'])}</div>
            <div class="bar"><i style="width:{min(b['percent_used'], 100):.0f}%;background:{color}"></i></div>
            <div class="st" style="color:{color}">{label} · {b['percent_used']:.0f}%</div></div>"""
    st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

    # charts
    st.markdown('<div class="sec"><h2>Where it <em>went</em></h2>'
                '<span class="note">Hover a bar for details</span></div>', unsafe_allow_html=True)
    g1, g2 = st.columns([1.2, 1], gap="large")
    with g1:
        cat_df = pd.DataFrame([{"Category": k, "Spent": v["total"], "Transactions": v["count"]}
                               for k, v in summ.get("by_category", {}).items()])
        box = st.container(border=True)
        box.markdown(f"**By category, {month_label(month)}**")
        if not cat_df.empty:
            cat_df["Label"] = cat_df["Spent"].map(inr)
            base = alt.Chart(cat_df).encode(
                y=alt.Y("Category:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK, labelFontSize=13)),
                x=alt.X("Spent:Q", title=None, axis=None, scale=alt.Scale(domain=[0, cat_df["Spent"].max() * 1.28])),
                tooltip=["Category", alt.Tooltip("Label:N", title="Spent"), "Transactions"])
            bars = base.mark_bar(color=TEAL, cornerRadiusEnd=4, height=14)
            text = base.mark_text(align="left", dx=6, color=INK2, fontSize=12).encode(text="Label:N")
            box.altair_chart(axis_cfg((bars + text).properties(height=32 * len(cat_df) + 10)),
                             width="stretch")
    with g2:
        trend = pd.DataFrame([{"Month": month_label(m).split()[0][:3] + " " + m[2:4], "order": i,
                               "Total": tools.get_spending_summary(m).get("total", 0.0),
                               "Selected": m == month} for i, m in enumerate(months)])
        trend["Label"] = trend["Total"].map(inr)
        box = st.container(border=True)
        box.markdown("**Month by month**")
        chart = alt.Chart(trend).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=44).encode(
            x=alt.X("Month:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0, labelColor=INK)),
            y=alt.Y("Total:Q", title=None, axis=alt.Axis(format="~s", tickCount=4)),
            color=alt.condition("datum.Selected", alt.value(TEAL), alt.value(TEAL_MID)),
            tooltip=["Month", alt.Tooltip("Label:N", title="Total")])
        box.altair_chart(axis_cfg(chart.properties(height=250)), width="stretch")

    st.markdown('<div class="sec"><h2>Worth a <em>second look</em></h2>'
                '<span class="note">Far above what you usually spend in that category</span></div>',
                unsafe_allow_html=True)
    if anomalies:
        ocols = st.columns(min(len(anomalies), 3))
        for i, a in enumerate(anomalies[:3]):
            with ocols[i].container(border=True):
                st.markdown(f"""<div class="odd"><div class="big">{inr(a['amount'])}</div><div>
                    <div><b>{html.escape(a['description_data'])}</b></div>
                    <div class="why">{a['date']} · {html.escape(a['category'])}<br>
                    Usually about {inr(a['category_average'])} (z-score {a['z_score']})</div></div></div>""",
                            unsafe_allow_html=True)
    else:
        st.caption("Nothing unusual this month.")
    if report["skipped"]:
        with st.expander(f"{len(report['skipped'])} rows in your file were skipped"):
            for line, why in report["skipped"]:
                st.caption(f"Line {line}: {why}")
    st.caption(f"{report['loaded']} transactions · model: {model} · every figure comes from the same "
               "tools the agent uses.")


# ================= CHAT =================
with tab_chat:
    st.markdown('<div class="sec" style="margin-top:22px"><h2>Ask the <em>agent</em></h2>'
                '<span class="note">It asks before changing anything</span></div>', unsafe_allow_html=True)
    examples = ["Am I over budget this month?", "Any unusual transactions?", "What if I cut Swiggy by half?",
                "Compare this month with last month", "Which transactions need review?", "Set my Food budget to 7000"]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.queued = ex

    if not st.session_state.chat:
        with st.chat_message("assistant", avatar="🧾"):
            st.markdown("Ask me about your spending. I can check budgets, find unusual transactions, compare "
                        "months and work out what cutting back would save. I'll ask before changing anything.")
    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "🧾"):
            st.markdown(show(m["text"]))
            if m.get("trace"):
                n_tools = sum(t["type"] == "tool" for t in m["trace"])
                with st.expander(f"How the agent answered ({n_tools} tool call{'s' if n_tools != 1 else ''})"):
                    render_trace(m["trace"])
    approval_cards("chat")

    q = st.chat_input("Ask about your spending…")
    q = q or st.session_state.pop("queued", None)
    if q:
        ask(q)
        st.rerun()


# ================= WHAT-IF =================
with tab_what:
    st.markdown('<div class="sec" style="margin-top:22px"><h2>What if you <em>cut back?</em></h2>'
                '<span class="note">Same simulate_savings tool the agent uses</span></div>', unsafe_allow_html=True)
    merch = store.query("SELECT description, SUM(amount) AS total FROM transactions GROUP BY description")
    m_tot = {}
    for row in merch:
        k = merchant_key(row["description"])
        if k:
            m_tot[k] = m_tot.get(k, 0) + row["total"]
    top_merchants = [k for k, _ in sorted(m_tot.items(), key=lambda kv: -kv[1])][:20]

    c1, c2, c3 = st.columns([1, 1.4, 1.6])
    kind = c1.radio("Cut back on", ["A merchant", "A category"], horizontal=True)
    if kind == "A merchant":
        default = top_merchants.index("swiggy order") if "swiggy order" in top_merchants else 0
        target = c2.selectbox("Merchant", top_merchants, index=default, format_func=str.title)
        args = {"merchant": target}
    else:
        cats = [c for c in CATEGORIES if c in {k for m_ in months for k in tools._totals(m_)}]
        target = c2.selectbox("Category", cats, index=cats.index("Food & Dining") if "Food & Dining" in cats else 0)
        args = {"category": target}
    cut = c3.slider("Cut by", 10, 100, 50, step=5, format="%d%%")
    sim = tools.execute("simulate_savings", {"cut_percent": cut, **args})

    if "error" in sim:
        st.info(f"Can't simulate that: {sim['error']}.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Saved a month", inr(sim["saving_per_month"]))
        s2.metric("In 6 months", inr(sim["saving_6_months"]))
        s3.metric("In a year", inr(sim["saving_12_months"]))
        if sim["budget"]:
            before, after = sim["budget_status_before"], sim["budget_status_after"]
            s4.metric(f"{sim['budget_category']} budget", STATUS[after][1],
                      f"was {STATUS[before][1].lower()}" if before != after else None, delta_color="off")
        g1, g2 = st.columns([1.3, 1], gap="large")
        with g1:
            cum = pd.DataFrame({"Month": list(range(1, 13)),
                                "Saved": [round(sim["saving_per_month"] * i, 2) for i in range(1, 13)]})
            cum["Label"] = cum["Saved"].map(inr)
            box = st.container(border=True)
            box.markdown("**Money saved over the next 12 months**")
            area = alt.Chart(cum).encode(
                x=alt.X("Month:Q", title="Months from now", scale=alt.Scale(domain=[1, 12]),
                        axis=alt.Axis(tickCount=12, format="d")),
                y=alt.Y("Saved:Q", title=None, axis=alt.Axis(format="~s", tickCount=4)),
                tooltip=[alt.Tooltip("Month:Q", title="Month"), alt.Tooltip("Label:N", title="Saved")])
            box.altair_chart(axis_cfg((area.mark_area(color=TEAL, opacity=0.12)
                                       + area.mark_line(color=TEAL, strokeWidth=2)
                                       + area.mark_point(color=TEAL, size=40, filled=True)).properties(height=240)),
                             width="stretch")
        with g2:
            if sim["budget"]:
                bdf = pd.DataFrame([
                    {"When": "Now", "Spent": sim["latest_category_spend"]},
                    {"When": f"After {cut}% cut", "Spent": sim["latest_category_spend_after_cut"]}])
                bdf["Label"] = bdf["Spent"].map(inr)
                box = st.container(border=True)
                box.markdown(f"**{sim['budget_category']} in {month_label(sim['latest_month'])} against its budget**")
                bars = alt.Chart(bdf).mark_bar(cornerRadiusEnd=4, height=24).encode(
                    y=alt.Y("When:N", title=None, sort=None, axis=alt.Axis(labelColor=INK, labelFontSize=13)),
                    x=alt.X("Spent:Q", title=None, axis=alt.Axis(format="~s", tickCount=4),
                            scale=alt.Scale(domain=[0, max(bdf["Spent"].max(), sim["budget"]) * 1.3])),
                    color=alt.Color("When:N", scale=alt.Scale(domain=["Now", f"After {cut}% cut"],
                                                              range=[TEAL_MID, TEAL]), legend=None),
                    tooltip=["When", alt.Tooltip("Label:N", title="Spent")])
                labels = bars.mark_text(align="left", dx=6).encode(text="Label:N", color=alt.value(INK2))
                rule = alt.Chart(pd.DataFrame({"b": [sim["budget"]]})).mark_rule(
                    color=STATUS["over"][0], strokeDash=[4, 3], strokeWidth=2).encode(x="b:Q")
                box.altair_chart(axis_cfg((bars + labels + rule).properties(height=130)), width="stretch")
                box.caption(f"Dashed line: budget of {inr(sim['budget'])}.")
        st.caption(f"Based on your average of {inr(sim['average_monthly_spend'])} a month over "
                   f"{sim['months_of_data']} month(s). One-off big purchases raise the average.")


# ================= REVIEW =================
with tab_review:
    queue = tools.get_review_queue()
    st.markdown('<div class="sec" style="margin-top:22px"><h2>Help the agent <em>learn</em></h2></div>',
                unsafe_allow_html=True)
    st.markdown("These didn't match any merchant the agent knows. Pick the right category and it will remember "
                "the merchant, fixing matching transactions now and in future statements.")
    if not queue["count"]:
        st.success("Nothing to review. Every transaction has a category the agent is confident about.")
    for t in queue["transactions"]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1.6, 0.9], vertical_alignment="center")
            c1.markdown(f"**{html.escape(t['description_data'])}**  \n"
                        f"<span style='color:{INK2};font-size:.9rem'>#{t['id']} · {t['date']} · {inr(t['amount'])}</span>",
                        unsafe_allow_html=True)
            choice = c2.selectbox("Category", CATEGORIES, index=CATEGORIES.index(t["category"]),
                                  key=f"rq{t['id']}", label_visibility="collapsed")
            if c3.button("Save", key=f"rqb{t['id']}", width="stretch", type="primary"):
                res = tools.confirm_category(t["id"], choice, remember=True)
                st.toast(f"Saved. Updated {res['updated']} transaction(s) and learned the merchant.")
                st.rerun()
    learned = store.learned_rules()
    if learned:
        st.markdown("**Merchants the agent has learned from you**")
        st.dataframe(pd.DataFrame([{"Merchant text": k, "Category": v} for k, v in learned.items()]),
                     hide_index=True, width="stretch")
