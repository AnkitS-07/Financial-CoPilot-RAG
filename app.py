"""
app.py
------
Finance Copilot — a personal finance dashboard with a built-in RAG chatbot.

Covers the full loop: enter your income/expenses, see a savings projection
and a single 0-100 Financial Health Score, stress-test an investment plan
with a Monte Carlo simulation, check live market prices, then ask a
chatbot plain-English questions about your own numbers (grounded in what's
actually on the page, not a canned FAQ) and export a summary report.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from engine import (
    financial_health_score,
    monte_carlo_investment,
    months_to_goal,
    monthly_savings,
    recommendations,
    savings_projection,
)
from rag import FinanceRAG, build_documents

st.set_page_config(page_title="Finance Copilot", page_icon="💰", layout="wide")
st.title("💰 Finance Copilot")
st.caption("Plan your savings, simulate investments, and ask questions about your own numbers.")

# Sidebar — all core inputs live here so every tab shares the same numbers
with st.sidebar:
    st.header("Your Numbers")
    income = st.number_input("Monthly income", min_value=0.0, value=5000.0, step=100.0)
    current_savings = st.number_input("Current savings", min_value=0.0, value=2000.0, step=100.0)
    savings_goal = st.number_input("Savings goal", min_value=0.0, value=20000.0, step=500.0)

    st.subheader("Monthly Expenses")
    default_categories = {"Rent": 1200.0, "Groceries": 400.0, "Entertainment": 150.0, "Other": 300.0}
    expense_categories = {}
    for cat, default in default_categories.items():
        expense_categories[cat] = st.number_input(cat, min_value=0.0, value=default, step=25.0)
    total_expenses = sum(expense_categories.values())

    st.subheader("Investing")
    risk = st.selectbox("Risk tolerance", ["Low", "Medium", "High"], index=1)
    num_investments = st.slider("Number of investment types you hold", 0, 6, 2)

# Shared calculations (used across tabs and by the RAG chatbot)
saving = monthly_savings(income, total_expenses)
eta_months = months_to_goal(current_savings, savings_goal, saving)
score = financial_health_score(income, total_expenses, savings_goal, current_savings, num_investments)
tips = recommendations(income, total_expenses, risk, score)

profile = {
    "income": income,
    "expenses": total_expenses,
    "monthly_saving": saving,
    "goal": savings_goal,
    "current_savings": current_savings,
    "months_to_goal": eta_months if eta_months is not None else "unreachable at current pace",
    "risk": risk,
    "score": score["score"],
    "score_label": score["label"],
    "score_breakdown": score["breakdown"],
    "expense_categories": expense_categories,
}

tab_dashboard, tab_invest, tab_expenses, tab_chat, tab_report = st.tabs(
    ["📊 Dashboard", "💹 Investment Simulator", "🧾 Expense Tracker", "🤖 Ask My Finances", "📄 Report"]
)

# Dashboard
with tab_dashboard:
    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly savings", f"${saving:,.0f}")
    col2.metric("Months to goal", eta_months if eta_months is not None else "—")
    col3.metric("Health score", f"{score['score']}/100", score["label"])

    st.subheader("Financial Health Breakdown")
    breakdown_df = pd.DataFrame(score["breakdown"].items(), columns=["Component", "Points"])
    st.bar_chart(breakdown_df.set_index("Component"))

    st.subheader("Projected Savings Growth")
    months_ahead = st.slider("Months to project", 6, 60, 24)
    proj = savings_projection(current_savings, saving, months=months_ahead)
    st.plotly_chart(px.line(proj, x="Month", y="Projected Balance"), use_container_width=True)

    st.subheader("Recommendations")
    for tip in tips:
        st.info(tip)

# Investment Simulator
with tab_invest:
    st.subheader("Monte Carlo Investment Simulation")
    principal = st.number_input("Amount to invest", min_value=0.0, value=max(current_savings, 1000.0), step=100.0)
    years = st.slider("Investment horizon (years)", 1, 30, 10)

    sim = monte_carlo_investment(principal, years, risk)
    months_idx = sim.index
    percentiles = sim.quantile([0.1, 0.5, 0.9], axis=1).T
    percentiles.columns = ["10th percentile", "Median", "90th percentile"]
    percentiles["Month"] = months_idx

    fig = px.line(percentiles, x="Month", y=["10th percentile", "Median", "90th percentile"],
                  title=f"{risk}-risk simulation over {years} years (300 runs)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Median outcome: ${percentiles['Median'].iloc[-1]:,.0f} · "
               f"Range (10th–90th pct): ${percentiles['10th percentile'].iloc[-1]:,.0f} – "
               f"${percentiles['90th percentile'].iloc[-1]:,.0f}")

    st.subheader("Live Market Snapshot")
    tickers = st.text_input("Tickers (comma-separated)", "AAPL, BTC-USD, VOO")
    period = st.select_slider("History window", options=["5d", "1mo", "3mo", "6mo", "1y"], value="1mo")
    if st.button("Fetch prices"):
        symbols = [t.strip() for t in tickers.split(",") if t.strip()]
        try:
            data = yf.download(symbols, period=period, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(name=symbols[0])
            latest = data.iloc[-1]
            cols = st.columns(len(latest))
            for col, (ticker, price) in zip(cols, latest.items()):
                col.metric(ticker, f"${price:,.2f}")

            chart_df = data.reset_index().melt(id_vars=data.index.name or "Date",
                                                var_name="Ticker", value_name="Price")
            fig = px.line(chart_df, x=chart_df.columns[0], y="Price", color="Ticker",
                          title=f"Closing price — last {period}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't fetch market data right now ({e}). Check your tickers or connection.")

# Expense Tracker
with tab_expenses:
    st.subheader("Where Your Money Goes")
    exp_df = pd.DataFrame(expense_categories.items(), columns=["Category", "Amount"])
    st.plotly_chart(px.pie(exp_df, names="Category", values="Amount"), use_container_width=True)
    st.dataframe(exp_df, use_container_width=True, hide_index=True)

# RAG Chatbot
with tab_chat:
    st.subheader("Ask questions about your own financial picture")
    st.caption("Answers are grounded only in the numbers on this page, plus any CSV you upload below.")

    uploaded_holdings = st.file_uploader("Optional: upload holdings.csv", type="csv", key="holdings")
    uploaded_trades = st.file_uploader("Optional: upload trades.csv", type="csv", key="trades")
    holdings_df = pd.read_csv(uploaded_holdings) if uploaded_holdings else None
    trades_df = pd.read_csv(uploaded_trades) if uploaded_trades else None

    @st.cache_resource(show_spinner="Loading models (first run only)...")
    def load_rag_pipeline():
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        return FinanceRAG(embedder, generator, tokenizer)

    question = st.text_input("Your question", placeholder="e.g. What's my savings rate, and how healthy is it?")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            rag = load_rag_pipeline()
            docs = build_documents(profile, holdings_df, trades_df)
            rag.index(docs)
            result = rag.ask(question)
        st.success(result.answer)
        with st.expander("Sources used"):
            for s in result.sources:
                st.write("•", s)

# Report
with tab_report:
    st.subheader("Downloadable Summary")
    report_lines = [
        "# Finance Copilot Report",
        f"\n**Health score:** {score['score']}/100 ({score['label']})",
        f"\n**Monthly income:** ${income:,.2f}  \n**Monthly expenses:** ${total_expenses:,.2f}  \n**Monthly savings:** ${saving:,.2f}",
        f"\n**Savings goal:** ${savings_goal:,.2f} (currently ${current_savings:,.2f}, ETA: {profile['months_to_goal']} months)",
        f"\n**Risk tolerance:** {risk}",
        "\n## Recommendations",
    ] + [f"- {t}" for t in tips]
    report_md = "\n".join(report_lines)
    st.markdown(report_md)
    st.download_button("Download report as Markdown", report_md, file_name="finance_report.md")
