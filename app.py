"""Streamlit web front end for the Expense Assistant Agent.   Run:  streamlit run app.py

Tabs:
  Dashboard     - KPIs, budget status, spending by category, monthly trend, anomalies
  Ask the agent - chat; every answer shows the agent's tool steps as a timeline;
                  data changes wait for Approve / Reject buttons (human in the loop)
  What-if       - interactive savings simulator (same tool the agent uses)
  Review queue  - confirm uncertain categories; the agent learns each merchant

The dashboard and what-if tab call the same tool functions as the agent, so the
numbers on screen and in the agent's answers always agree.
"""
import html
import json
import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

from expense_agent import SAMPLE_CSV, build_agent, load_env
from expense_agent.categorizer import CATEGORIES, merchant_key
from expense_agent.offline import inr, month_label

st.set_page_config(page_title="Expense Assistant Agent", page_icon="💸", layout="wide")
load_env()

# ---------------- palette (validated reference palette, light mode) ----------------
BLUE, BLUE_LIGHT = "#2a78d6", "#b7d3f6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STATUS = {"ok": ("#0ca30c", "✅", "On track"), "warning": ("#fab219", "⚠️", "Near limit"),
          "over": ("#d03b3b", "⛔", "Over budget")}

st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem; max-width: 1200px;}}
  .hero h1 {{font-size: 2.1rem; margin: 0 0 .2rem 0;}}
  .hero p {{color: {INK2}; margin: 0 0 .6rem 0; font-size: 1.02rem;}}
  .badge {{display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem;
          background: #eef4fc; color: #1c5cab; margin-right: 6px; border: 1px solid #cde2fb;}}
  .card {{background: #fff; border: 1px solid rgba(11,11,11,.08); border-radius: 12px;
         padding: 16px 18px; margin-bottom: 12px;}}
  .card h4 {{margin: 0 0 10px 0; font-size: 1rem;}}
  div[data-testid="stVerticalBlockBorderWrapper"] {{background:#fff; border-radius:12px;}}
  h4 {{font-size: 1rem !important; padding-bottom: .2rem !important;}}
  div[data-testid="stMetric"] {{background:#fff; border:1px solid rgba(11,11,11,.08);
         border-radius:12px; padding: 12px 16px;}}
  .brow {{margin: 0 0 12px 0;}}
  .brow .top {{display:flex; justify-content:space-between; font-size:.9rem;}}
  .brow .top .amt {{color:{INK2};}}
  .track {{height: 8px; background: #efeee9; border-radius: 4px; margin-top: 5px; overflow: hidden;}}
  .fill {{height: 100%; border-radius: 4px;}}
  .tag {{font-size: .78rem; color: {INK2};}}
  .step {{border-left: 3px solid {BLUE_LIGHT}; padding: 4px 0 8px 12px; margin-left: 6px; position: relative;}}
  .step:before {{content:""; position:absolute; left:-7px; top:8px; width:11px; height:11px;
                border-radius:50%; background:{BLUE}; border: 2px solid #fff;}}
  .step.err:before {{background:#fab219;}}
  .step.ans:before {{background:#0ca30c;}}
  .step .tool {{font-family: ui-monospace, Consolas, monospace; font-size:.85rem; background:#eef4fc;
               color:#1c5cab; padding:1px 6px; border-radius:6px;}}
  .step .sum {{color:{INK2}; font-size:.88rem;}}
  .approve {{background:#fffaf0; border:1px solid #fbe3b0; border-radius:12px; padding:12px 16px; margin: 6px 0;}}
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
                            approved_keys=set(), pending=[])


with st.sidebar:
    st.markdown("### ⚙️ Setup")
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    use_gemini = st.toggle("Use Gemini", value=has_key, disabled=not has_key,
                           help="Needs GEMINI_API_KEY in .env or Streamlit secrets. Off = rule-based offline planner.")
    upload = st.file_uploader("Bank statement CSV", type="csv",
                              help="Columns: date (YYYY-MM-DD), description, amount")
    c1, c2 = st.columns(2)
    if c1.button("Load file", disabled=upload is None, use_container_width=True):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(upload.getvalue()); tmp.close()
        load(tmp.name, use_gemini, upload.name)
    if c2.button("Sample data", use_container_width=True) or "agent" not in st.session_state:
        load(str(SAMPLE_CSV), use_gemini, "sample statement (Jul–Sep 2026)")

    r = st.session_state.report
    llm = st.session_state.agent.llm
    model = getattr(llm, "model", None) or "rule-based planner"
    st.markdown(f"**Data:** {st.session_state.source}  \n**Rows:** {r['loaded']} loaded, "
                f"{len(r['skipped'])} skipped  \n**Model:** `{model}`")
    if r["skipped"]:
        with st.expander(f"⚠️ {len(r['skipped'])} rows skipped"):
            for line, why in r["skipped"]:
                st.caption(f"Line {line}: {why}")
    st.divider()
    st.markdown("**How it works**")
    st.caption("Gemini plans which tools to call; the tools do every calculation on your data; "
               "changes to your data wait for your approval. Open *How the agent answered* "
               "under any reply to see each step.")

store, agent = st.session_state.store, st.session_state.agent
tools = agent.tools
months = store.months()

# ---------------- header ----------------
st.markdown(
    f"""<div class="hero"><h1>💸 Expense Assistant Agent</h1>
    <p>An AI agent that analyses your bank statement, explains where your money goes,
    and simulates how to save, with every number calculated from your data.</p>
    <span class="badge">ReAct agent · 10 tools</span><span class="badge">Human-in-the-loop</span>
    <span class="badge">Model: {html.escape(model)}</span></div>""", unsafe_allow_html=True)

if not months:
    st.warning("No transactions loaded. Upload a CSV in the sidebar.")
    st.stop()

tab_dash, tab_chat, tab_what, tab_review = st.tabs(
    ["📊 Dashboard", "💬 Ask the agent", "🧮 What-if simulator", "🗂️ Review queue"])


def axis_cfg(chart):
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(gridColor=GRID, domainColor="#c3c2b7", labelColor=MUTED, titleColor=INK2,
                            labelFontSize=12, titleFontSize=12, tickColor="#c3c2b7"))


# ================= DASHBOARD =================
with tab_dash:
    month = st.columns([1, 3])[0].selectbox("Month", months[::-1], format_func=month_label, key="dash_month")
    idx = months.index(month)
    summ = tools.get_spending_summary(month)
    budgets = tools.check_budgets(month)
    anomalies = tools.detect_anomalies(month)["anomalies"]
    queue = tools.get_review_queue()

    total = summ.get("total", 0.0)
    delta = None
    if idx > 0:
        prev = tools.get_spending_summary(months[idx - 1]).get("total", 0.0)
        if prev:
            delta = f"{100 * (total - prev) / prev:+.0f}% vs {month_label(months[idx - 1]).split()[0]}"
    over = [b for b in budgets["budgets"] if b["status"] == "over"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total spent", inr(total), delta, delta_color="inverse")
    k2.metric("Categories over budget", f"{len(over)} of {len(budgets['budgets'])}",
              help=", ".join(b["category"] for b in over) or "none")
    k3.metric("Unusual transactions", len(anomalies), delta_color="off")
    k4.metric("Need your review", queue["count"], delta_color="off")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        rows = sorted(budgets["budgets"], key=lambda b: -b["percent_used"])
        body = ""
        for b in rows:
            color, icon, label = STATUS[b["status"]]
            width = min(b["percent_used"], 100)
            body += f"""<div class="brow"><div class="top"><span><b>{html.escape(b['category'])}</b>
              <span class="tag">{icon} {label}</span></span>
              <span class="amt">{inr(b['spent'])} of {inr(b['budget'])} · {b['percent_used']:.0f}%</span></div>
              <div class="track"><div class="fill" style="width:{width}%;background:{color}"></div></div></div>"""
        box = st.container(border=True)
        box.markdown(f"#### Budget status · {month_label(month)}")
        box.markdown(body, unsafe_allow_html=True)
    with right:
        cat_df = pd.DataFrame([{"Category": k, "Spent": v["total"], "Transactions": v["count"]}
                               for k, v in summ.get("by_category", {}).items()])
        box = st.container(border=True)
        box.markdown("#### Where the money went")
        if not cat_df.empty:
            cat_df["Label"] = cat_df["Spent"].map(inr)
            base = alt.Chart(cat_df).encode(
                y=alt.Y("Category:N", sort="-x", title=None),
                x=alt.X("Spent:Q", title=None, axis=alt.Axis(format="~s", tickCount=4),
                        scale=alt.Scale(domain=[0, cat_df["Spent"].max() * 1.25])),
                tooltip=["Category", alt.Tooltip("Label:N", title="Spent"), "Transactions"])
            bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, height=16)
            text = base.mark_text(align="left", dx=6, color=INK2, fontSize=12).encode(text="Label:N")
            box.altair_chart(axis_cfg((bars + text).properties(height=34 * len(cat_df) + 20)),
                             use_container_width=True)

    t_left, t_right = st.columns([1.15, 1], gap="large")
    with t_left:
        trend = pd.DataFrame([{"Month": month_label(m), "order": i,
                               "Total": tools.get_spending_summary(m).get("total", 0.0),
                               "Selected": m == month} for i, m in enumerate(months)])
        trend["Label"] = trend["Total"].map(inr)
        box = st.container(border=True)
        box.markdown("#### Monthly spending")
        chart = alt.Chart(trend).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=48).encode(
            x=alt.X("Month:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Total:Q", title=None, axis=alt.Axis(format="~s", tickCount=4)),
            color=alt.condition("datum.Selected", alt.value(BLUE), alt.value(BLUE_LIGHT)),
            tooltip=["Month", alt.Tooltip("Label:N", title="Total")])
        box.altair_chart(axis_cfg(chart.properties(height=230)), use_container_width=True)
    with t_right:
        box = st.container(border=True)
        box.markdown("#### 🔎 Unusual transactions")
        if anomalies:
            for a in anomalies:
                box.markdown(f"**{inr(a['amount'])}** · {html.escape(a['description_data'])}  \n"
                            f"<span class='tag'>{a['date']} · {a['category']} · usual ≈ "
                            f"{inr(a['category_average'])} · z = {a['z_score']}</span>", unsafe_allow_html=True)
        else:
            box.caption("Nothing unusual this month.")
    st.caption("Every figure on this page comes from the same tool functions the agent calls.")


# ================= CHAT =================
TOOL_SUMMARY = {
    "get_available_months": lambda r: f"latest month {r.get('latest')}",
    "get_spending_summary": lambda r: f"total {inr(r['total'])} across {len(r['by_category'])} categories" if "total" in r else "",
    "check_budgets": lambda r: f"{sum(b['status'] == 'over' for b in r['budgets'])} over, "
                               f"{sum(b['status'] == 'warning' for b in r['budgets'])} near limit",
    "find_transactions": lambda r: f"{r['count']} transactions, {inr(r['total'])}",
    "detect_anomalies": lambda r: f"{len(r['anomalies'])} flagged",
    "compare_months": lambda r: f"change {inr(r['total_change'])}",
    "get_review_queue": lambda r: f"{r['count']} need review",
    "simulate_savings": lambda r: f"saves {inr(r['saving_per_month'])}/month" if "saving_per_month" in r else "",
    "set_budget": lambda r: "waiting for approval" if r.get("status") else f"budget now {inr(r['new_budget'])}",
    "confirm_category": lambda r: "waiting for approval" if r.get("status") else f"{r['updated']} updated",
}


def render_trace(trace):
    parts = []
    for t in trace:
        if t["type"] == "tool":
            res = t["result"]
            if "error" in res:
                summary = "⚠️ " + res["error"]
            else:
                try:
                    summary = TOOL_SUMMARY.get(t["name"], lambda r: "")(res)
                except Exception:
                    summary = ""
            args = ", ".join(f"{k}={v}" for k, v in t["args"].items())
            parts.append(f'<div class="step"><b>Step {t["step"]}</b> · <span class="tool">{t["name"]}'
                         f'({html.escape(args)})</span><br><span class="sum">→ {html.escape(summary)}</span></div>')
        elif t["type"] == "llm_error":
            parts.append(f'<div class="step err"><b>Step {t["step"]}</b> · model error, switched to offline planner'
                         f'<br><span class="sum">{html.escape(t["text"][:160])}</span></div>')
        elif t["type"] == "answer":
            parts.append(f'<div class="step ans"><b>Step {t["step"]}</b> · wrote the answer from the tool results</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    with st.popover("Raw tool data"):
        st.json(trace)


def ask(question):
    st.session_state.new_pending = []
    with st.spinner("The agent is working…"):
        result = agent.run(question)
    st.session_state.chat.append({"role": "user", "text": question})
    st.session_state.chat.append({"role": "assistant", "text": result.answer, "trace": result.trace})
    for p in st.session_state.new_pending:
        p["id"] = len(st.session_state.chat) * 100 + len(st.session_state.pending)
        st.session_state.pending.append(p)


def describe(p):
    a = p["args"]
    if p["name"] == "set_budget":
        old = store.budgets().get(a.get("category"))
        return (f"Set the **{a.get('category')}** budget to **{inr(float(a.get('amount', 0)))}** a month"
                + (f" (currently {inr(old)})" if old else ""))
    return f"Mark transaction **#{a.get('transaction_id')}** as **{a.get('category')}** and remember this merchant"


with tab_chat:
    st.markdown("**Try one of these, or type your own question:**")
    examples = ["Am I over budget this month?", "Any unusual transactions?", "What if I cut Swiggy by half?",
                "Compare this month with last month", "Which transactions need review?", "Set my Food budget to 7000"]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.queued = ex

    if not st.session_state.chat:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Hi! I can answer questions about your spending, check your budgets, spot unusual "
                        "transactions and simulate savings. I'll ask before changing anything.")
    for i, m in enumerate(st.session_state.chat):
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["text"])
            if m.get("trace"):
                n_tools = sum(t["type"] == "tool" for t in m["trace"])
                with st.expander(f"🔍 How the agent answered ({n_tools} tool call{'s' if n_tools != 1 else ''})"):
                    render_trace(m["trace"])

    for p in list(st.session_state.pending):
        st.markdown(f'<div class="approve">✋ <b>Approval needed</b> · the agent wants to:</div>',
                    unsafe_allow_html=True)
        st.markdown(describe(p))
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("✅ Approve", key=f"ok{p['id']}", type="primary"):
            res = tools.execute(p["name"], p["args"])
            st.session_state.approved_keys.add(p["name"] + json.dumps(p["args"], sort_keys=True))
            st.session_state.pending.remove(p)
            text = (f"❌ Could not apply the change: {res['error']}" if "error" in res else
                    f"✅ Approved and applied. " + (f"{res['category']} budget is now {inr(res['new_budget'])}."
                                                   if p["name"] == "set_budget" else
                                                   f"Updated {res['updated']} transaction(s); merchant learned."))
            st.session_state.chat.append({"role": "assistant", "text": text})
            st.rerun()
        if b2.button("Reject", key=f"no{p['id']}"):
            st.session_state.pending.remove(p)
            st.session_state.chat.append({"role": "assistant", "text": "Rejected. Nothing was changed."})
            st.rerun()

    q = st.chat_input("Ask about your spending…")
    q = q or st.session_state.pop("queued", None)
    if q:
        ask(q)
        st.rerun()


# ================= WHAT-IF =================
with tab_what:
    st.markdown("Move the slider to see what cutting back would save. This is the same "
                "`simulate_savings` tool the agent uses when you ask a *what if* question.")
    merch = store.query("SELECT description, SUM(amount) AS total FROM transactions GROUP BY description")
    m_tot = {}
    for row in merch:
        k = merchant_key(row["description"])
        if k:
            m_tot[k] = m_tot.get(k, 0) + row["total"]
    top_merchants = [k for k, _ in sorted(m_tot.items(), key=lambda kv: -kv[1])][:20]

    c1, c2, c3 = st.columns([1, 1.4, 1.6])
    kind = c1.radio("Cut back on", ["Merchant", "Category"], horizontal=True)
    if kind == "Merchant":
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
        st.info(sim["error"])
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Saved per month", inr(sim["saving_per_month"]))
        s2.metric("In 6 months", inr(sim["saving_6_months"]))
        s3.metric("In 1 year", inr(sim["saving_12_months"]))
        if sim["budget"]:
            before, after = sim["budget_status_before"], sim["budget_status_after"]
            s4.metric(f"{sim['budget_category']} budget", STATUS[after][2],
                      f"was {STATUS[before][2].lower()}" if before != after else None, delta_color="off")
        g1, g2 = st.columns([1.3, 1], gap="large")
        with g1:
            cum = pd.DataFrame({"Month": list(range(1, 13)),
                                "Saved": [round(sim["saving_per_month"] * i, 2) for i in range(1, 13)]})
            cum["Label"] = cum["Saved"].map(inr)
            box = st.container(border=True)
            box.markdown("#### Money saved over the next 12 months")
            area = alt.Chart(cum).encode(
                x=alt.X("Month:Q", title="Months from now", scale=alt.Scale(domain=[1, 12]),
                        axis=alt.Axis(tickCount=12, format="d")),
                y=alt.Y("Saved:Q", title=None, axis=alt.Axis(format="~s", tickCount=4)),
                tooltip=[alt.Tooltip("Month:Q", title="Month"), alt.Tooltip("Label:N", title="Saved")])
            box.altair_chart(axis_cfg((area.mark_area(color=BLUE, opacity=0.15)
                                      + area.mark_line(color=BLUE, strokeWidth=2)
                                      + area.mark_point(color=BLUE, size=40, filled=True)).properties(height=240)),
                             use_container_width=True)
        with g2:
            if sim["budget"]:
                bdf = pd.DataFrame([
                    {"When": "Now", "Spent": sim["latest_category_spend"]},
                    {"When": f"After {cut}% cut", "Spent": sim["latest_category_spend_after_cut"]}])
                bdf["Label"] = bdf["Spent"].map(inr)
                box = st.container(border=True)
                box.markdown(f"#### {sim['budget_category']} in {month_label(sim['latest_month'])} vs budget")
                bars = alt.Chart(bdf).mark_bar(cornerRadiusEnd=4, height=26).encode(
                    y=alt.Y("When:N", title=None, sort=None),
                    x=alt.X("Spent:Q", title=None, axis=alt.Axis(format="~s", tickCount=4),
                            scale=alt.Scale(domain=[0, max(bdf["Spent"].max(), sim["budget"]) * 1.3])),
                    color=alt.Color("When:N", scale=alt.Scale(domain=["Now", f"After {cut}% cut"],
                                    range=[BLUE_LIGHT, BLUE]), legend=None),
                    tooltip=["When", alt.Tooltip("Label:N", title="Spent")])
                labels = bars.mark_text(align="left", dx=6, color=INK2).encode(text="Label:N", color=alt.value(INK2))
                rule = alt.Chart(pd.DataFrame({"b": [sim["budget"]]})).mark_rule(
                    color="#d03b3b", strokeDash=[4, 3], strokeWidth=2).encode(x="b:Q")
                box.altair_chart(axis_cfg((bars + labels + rule).properties(height=130)), use_container_width=True)
                box.caption(f"Red dashed line = budget ({inr(sim['budget'])}).")
        st.caption(f"Based on your average of {inr(sim['average_monthly_spend'])} a month over "
                   f"{sim['months_of_data']} month(s) of data. One-off big purchases raise the average.")


# ================= REVIEW QUEUE =================
with tab_review:
    queue = tools.get_review_queue()
    st.markdown("These transactions didn't match any known merchant, so the agent isn't sure of the category. "
                "Confirm the right one, and the agent **remembers the merchant** and fixes matching rows "
                "now and in future statements.")
    if not queue["count"]:
        st.success("Nothing to review: every transaction has a confident category.")
    for t in queue["transactions"]:
        c1, c2, c3 = st.columns([3, 1.6, 0.8])
        c1.markdown(f"**{html.escape(t['description_data'])}**  \n"
                    f"<span class='tag'>#{t['id']} · {t['date']} · {inr(t['amount'])}</span>", unsafe_allow_html=True)
        choice = c2.selectbox("Category", CATEGORIES, index=CATEGORIES.index(t["category"]),
                              key=f"rq{t['id']}", label_visibility="collapsed")
        if c3.button("Confirm", key=f"rqb{t['id']}", use_container_width=True):
            res = tools.confirm_category(t["id"], choice, remember=True)
            st.toast(f"Saved. {res['updated']} transaction(s) updated and the merchant was learned.")
            st.rerun()
    learned = store.learned_rules()
    if learned:
        st.markdown("**Merchants the agent has learned from you**")
        st.dataframe(pd.DataFrame([{"Merchant text": k, "Category": v} for k, v in learned.items()]),
                     hide_index=True, use_container_width=True)
