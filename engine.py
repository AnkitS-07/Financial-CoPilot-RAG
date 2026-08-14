"""
engine.py
---------
Pure-Python finance calculations used by the app. No Streamlit imports here
on purpose — keeps the math testable and reusable.
"""

import numpy as np
import pandas as pd

RISK_PROFILES = {
    "Low":    {"annual_return": 0.05, "annual_vol": 0.06},
    "Medium": {"annual_return": 0.09, "annual_vol": 0.14},
    "High":   {"annual_return": 0.13, "annual_vol": 0.25},
}


def monthly_savings(income: float, expenses: float) -> float:
    return max(income - expenses, 0.0)


def months_to_goal(current_savings: float, goal: float, monthly_saving: float,
                    monthly_growth: float = 0.02) -> int | None:
    """How many months (compounding monthly_growth on the balance) to hit `goal`.
    Returns None if the goal is unreachable (no savings and no starting balance).
    """
    if current_savings >= goal:
        return 0
    if monthly_saving <= 0 and current_savings <= 0:
        return None

    balance = current_savings
    months = 0
    max_months = 100 * 12  # 100-year safety cap so this can't loop forever
    while balance < goal and months < max_months:
        balance = balance * (1 + monthly_growth) + monthly_saving
        months += 1
    return months if balance >= goal else None


def savings_projection(current_savings: float, monthly_saving: float,
                        months: int = 24, monthly_growth: float = 0.02) -> pd.DataFrame:
    """Month-by-month projected balance, for charting."""
    balances = [current_savings]
    for _ in range(months):
        balances.append(balances[-1] * (1 + monthly_growth) + monthly_saving)
    return pd.DataFrame({"Month": range(months + 1), "Projected Balance": balances})


def monte_carlo_investment(principal: float, years: int, risk: str,
                            n_sims: int = 300, seed: int | None = 42) -> pd.DataFrame:
    """Simulate `n_sims` random walks of an investment using a simple
    geometric-Brownian-motion-style monthly return draw. Returns a DataFrame
    of shape (months+1, n_sims) — one column per simulated path.
    """
    profile = RISK_PROFILES.get(risk, RISK_PROFILES["Medium"])
    monthly_return = profile["annual_return"] / 12
    monthly_vol = profile["annual_vol"] / np.sqrt(12)
    months = years * 12

    rng = np.random.default_rng(seed)
    paths = np.zeros((months + 1, n_sims))
    paths[0] = principal
    for m in range(1, months + 1):
        shocks = rng.normal(monthly_return, monthly_vol, n_sims)
        paths[m] = paths[m - 1] * (1 + shocks)
    return pd.DataFrame(paths, columns=[f"sim_{i}" for i in range(n_sims)])


def financial_health_score(income: float, expenses: float, savings_goal: float,
                            current_savings: float, num_investments: int) -> dict:
    """A lightweight, transparent 0-100 score that gives the whole app a
    single headline number, and gives the RAG chatbot something concrete
    to explain when asked "how healthy are my finances?".

    Weighting (kept intentionally simple, not a black box):
      - Savings rate (income saved each month)      -> 45 pts
      - Expense ratio (expenses / income, lower better) -> 30 pts
      - Goal progress (current_savings / savings_goal)  -> 15 pts
      - Diversification (has >1 investment types tracked) -> 10 pts
    """
    savings_rate = monthly_savings(income, expenses) / income if income > 0 else 0
    expense_ratio = expenses / income if income > 0 else 1
    goal_progress = min(current_savings / savings_goal, 1.0) if savings_goal > 0 else 0

    savings_pts = min(savings_rate / 0.30, 1.0) * 45          # 30% savings rate = full marks
    expense_pts = max(0.0, 1 - min(expense_ratio, 1.0)) * 30
    goal_pts = goal_progress * 15
    diversification_pts = min(num_investments / 3, 1.0) * 10

    total = round(savings_pts + expense_pts + goal_pts + diversification_pts)

    if total >= 80:
        label = "Excellent"
    elif total >= 60:
        label = "Good"
    elif total >= 40:
        label = "Fair"
    else:
        label = "Needs Attention"

    return {
        "score": total,
        "label": label,
        "breakdown": {
            "Savings rate": round(savings_pts, 1),
            "Expense control": round(expense_pts, 1),
            "Goal progress": round(goal_pts, 1),
            "Diversification": round(diversification_pts, 1),
        },
    }


def recommendations(income: float, expenses: float, risk: str, score: dict) -> list[str]:
    """Simple rule-based tips — deliberately not ML, tied directly to the
    health score so the advice always matches what the score is telling you.
    """
    tips = []
    ratio = expenses / income if income > 0 else 1

    if ratio > 0.8:
        tips.append("Your expenses are eating over 80% of your income — look for the top 1-2 categories to trim first.")
    elif ratio > 0.6:
        tips.append("You're saving something every month, but there's room to push your savings rate higher.")
    else:
        tips.append("Strong expense discipline — you're saving a healthy share of your income.")

    if risk == "Low" and score["score"] > 70:
        tips.append("Your finances look stable enough that a slightly higher-risk allocation could accelerate growth, if you're comfortable with more volatility.")
    if risk == "High" and score["score"] < 40:
        tips.append("A high-risk allocation combined with a low health score is a risky pairing — consider building a cash buffer before adding volatility.")

    if score["breakdown"]["Diversification"] < 5:
        tips.append("You're tracking very few investment types — spreading across more assets tends to smooth out returns.")

    return tips
