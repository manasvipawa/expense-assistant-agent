"""Streamlit web front end.   Run:  streamlit run app.py"""
import os
import tempfile

import pandas as pd
import streamlit as st

from expense_agent import SAMPLE_CSV, build_agent, load_env

st.set_page_config(page_title="Expense Assistant Agent", page_icon="💸")
st.title("💸 Expense Assistant Agent")
load_env()

with st.sidebar:
    st.header("Setup")
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    use_gemini = st.toggle("Use Gemini", value=has_key, disabled=not has_key,
                           help="Set GEMINI_API_KEY in .env to enable")
    upload = st.file_uploader("Bank statement CSV (date, description, amount)", type="csv")
    allow_writes = st.checkbox("Allow the agent to change budgets / categories", value=False)
    if st.button("Reload data") or "agent" not in st.session_state:
        path = str(SAMPLE_CSV)
        if upload is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(upload.getvalue()); tmp.close()
            path = tmp.name
        agent, store, report = build_agent(
            path, offline=not use_gemini,
            approve=lambda name, args: st.session_state.get("allow_writes", False))
        st.session_state.update(agent=agent, store=store, report=report, chat=[])
    st.session_state.allow_writes = allow_writes
    r = st.session_state.report
    st.caption(f"{r['loaded']} transactions loaded, {len(r['skipped'])} rows skipped. "
               f"LLM: {type(st.session_state.agent.llm).__name__}")

store = st.session_state.store
with st.expander("Spending by month and category"):
    df = pd.DataFrame(store.query("SELECT month, category, SUM(amount) AS total FROM transactions GROUP BY 1,2"))
    if not df.empty:
        st.bar_chart(df.pivot(index="month", columns="category", values="total").fillna(0))

for role, text, trace in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        if trace:
            with st.expander("Agent steps"):
                st.json(trace)

q = st.chat_input("Ask about your spending, e.g. 'Am I over budget this month?'")
if q:
    st.session_state.chat.append(("user", q, None))
    with st.spinner("Thinking..."):
        result = st.session_state.agent.run(q)
    st.session_state.chat.append(("assistant", result.answer, result.trace))
    st.rerun()
