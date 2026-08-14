"""
rag.py
------
Retrieval-Augmented Generation over the user's own financial picture.

Pipeline: turn what's known about the user (income, expenses, savings goal,
health score, and optionally uploaded holdings/trades) into short text
"documents" -> embed them with sentence-transformers -> retrieve the closest
matches with FAISS -> generate a grounded answer with a small instruction-
tuned model (FLAN-T5). Because `build_documents()` indexes the live
dashboard state rather than only static files, the chatbot can answer
things like "what's my savings rate?" without any CSV upload at all.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_documents(profile: dict, holdings_df: pd.DataFrame | None = None,
                     trades_df: pd.DataFrame | None = None) -> list[str]:
    """Turn everything we know about the user into short text snippets.
    Short, single-fact documents retrieve far more precisely than one big
    blob of text — that's the same trick the original RAG notebook used for
    holdings/trades rows.
    """
    docs = []

    # live dashboard facts
    docs.append(f"Monthly income is {profile['income']:.2f}.")
    docs.append(f"Monthly expenses total {profile['expenses']:.2f}.")
    docs.append(f"Monthly savings (income minus expenses) is {profile['monthly_saving']:.2f}.")
    docs.append(f"Savings goal is {profile['goal']:.2f}, current savings are {profile['current_savings']:.2f}.")
    docs.append(f"Estimated months to reach the savings goal: {profile['months_to_goal']}.")
    docs.append(f"Selected investment risk tolerance is {profile['risk']}.")
    docs.append(f"Financial health score is {profile['score']} out of 100, rated {profile['score_label']}.")
    for category, pts in profile["score_breakdown"].items():
        docs.append(f"Health score component '{category}' contributed {pts} points.")
    for category, amount in profile["expense_categories"].items():
        docs.append(f"Expense category '{category}' costs {amount:.2f} per month.")

    # optional uploaded fund data (holdings/trades CSVs)
    if holdings_df is not None and not holdings_df.empty:
        for _, row in holdings_df.iterrows():
            docs.append(
                f"Holding: portfolio {row.get('PortfolioName', 'N/A')} holds "
                f"{row.get('SecName', 'N/A')}, quantity {row.get('Qty', 'N/A')}, "
                f"market value {row.get('MV_Base', 'N/A')}, YTD P&L {row.get('PL_YTD', 'N/A')}."
            )
        for fund, group in holdings_df.groupby(holdings_df.get("PortfolioName", pd.Series(dtype=str))):
            total_mv = group.get("MV_Base", pd.Series(dtype=float)).sum()
            total_pl = group.get("PL_YTD", pd.Series(dtype=float)).sum()
            docs.append(f"Fund summary: {fund} has total market value {total_mv:.2f} and total YTD P&L {total_pl:.2f}.")

    if trades_df is not None and not trades_df.empty:
        for _, row in trades_df.iterrows():
            docs.append(
                f"Trade: {row.get('TradeTypeName', 'N/A')} of {row.get('Ticker', 'N/A')} "
                f"in portfolio {row.get('PortfolioName', 'N/A')}, quantity {row.get('Quantity', 'N/A')}, "
                f"price {row.get('Price', 'N/A')}, date {row.get('TradeDate', 'N/A')}."
            )

    return docs


@dataclass
class RAGResult:
    answer: str
    sources: list[str]


class FinanceRAG:
    """Thin wrapper around sentence-transformers + FAISS + FLAN-T5.
    Models are passed in already-loaded so Streamlit can cache them once
    with @st.cache_resource instead of reloading per query.
    """

    def __init__(self, embedder, generator, tokenizer):
        self.embedder = embedder
        self.generator = generator
        self.tokenizer = tokenizer
        self._docs: list[str] = []
        self._index = None

    def index(self, docs: list[str]) -> None:
        import faiss  # local import so the app can run in "no-RAG" mode without faiss installed

        self._docs = docs
        embeddings = self.embedder.encode(docs, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        self._index.add(np.array(embeddings, dtype="float32"))

    def ask(self, question: str, top_k: int = 5) -> RAGResult:
        if self._index is None or not self._docs:
            return RAGResult("I don't have any financial data indexed yet.", [])

        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        scores, idxs = self._index.search(np.array(q_emb, dtype="float32"), top_k)
        retrieved = [self._docs[i] for i in idxs[0] if i != -1]

        if not retrieved:
            return RAGResult("Sorry, I can't find that in your financial data.", [])

        context = " ".join(retrieved[:top_k])
        prompt = (
            "Answer the question using only the context below. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = self.generator.generate(**inputs, max_new_tokens=128)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return RAGResult(answer.strip() or "Sorry, I can't find that in your financial data.", retrieved)
