# 💰 Finance Copilot

A personal finance dashboard with a built-in chatbot that answers
questions about *your own* numbers — plan your savings, stress-test an
investment strategy, track live market prices, and just ask when you want
a plain-English read on your situation instead of digging through charts.

## Features

- **Dashboard** — income/expense tracking, a projected savings curve, and a
  transparent 0–100 **Financial Health Score** with a visible breakdown
  (savings rate, expense control, goal progress, diversification) so the
  number is never a black box.
- **Investment Simulator** — a Monte Carlo simulation (300 runs) showing a
  10th/50th/90th percentile outcome band for your chosen risk tolerance and
  horizon, plus a live market snapshot with real price history charts.
- **Ask My Finances** — a retrieval-augmented chatbot (sentence-transformers + FAISS 
  (Facebook AI Similarity Search) + FLAN-T5 (Fine-tuned Language Net
  Text-to-Text Transfer Transformer), all open-source, no API key needed) that 
  answers questions grounded in your actual dashboard numbers. You can also 
  upload your own holdings/trades CSVs and ask about those too.
- **Report** — a one-click downloadable Markdown summary of everything
  above.

## Why the chatbot isn't just a gimmick

Most "AI chatbot" bolt-ons answer from a static FAQ or a canned CSV. Here,
`rag.py::build_documents()` turns your *live* income, expenses, savings
goal, and health score into small retrievable text snippets every time you
recalculate — so asking "how healthy are my finances and why?" gets an
answer grounded in the numbers actually on your screen, not a template.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads two small open-source models (`all-MiniLM-L6-v2` and
`google/flan-t5-base`) for the chatbot — free, local, no API key required.

## Project structure

```
.
├── app.py            # Streamlit UI, ties everything together
├── engine.py          # Pure-Python finance math (savings, Monte Carlo, health score)
├── rag.py             # Retrieval-augmented chatbot
├── sample_data/       # Example holdings/trades CSVs for the chat tab
└── requirements.txt
```

## Try asking the chatbot

- "What's my monthly savings rate?"
- "How healthy are my finances and why?"
- "Which fund had the better YTD P&L?" (with sample data loaded)
- "How many months until I hit my savings goal?"
