"""
Retrieval over the Portuguese wine catalog (Kaggle Wine Reviews, country == Portugal).

This is "RAG without a vector database": TF-IDF over the tasting notes plus simple filters
parsed from the question (colour, max price, region). For ~5,000 wines it is fast, needs no
extra service, and every step is easy to explain in a 5-minute presentation.

The retrieved wines are pasted into the prompt as {{context}}. A grounded prompt must then
recommend ONLY from that list — which is what stops the model inventing wines.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src import config

REGIONS = ["douro", "vinho verde", "alentejo", "alentejano", "dão", "dao", "bairrada",
           "lisboa", "tejo", "setúbal", "setubal", "península de setúbal", "madeira",
           "porto", "port", "beira interior", "algarve", "minho"]

COLOUR_WORDS = {
    "red": ["red", "tinto"],
    "white": ["white", "branco"],
    "rosé": ["rosé", "rose"],
}


class Catalog:
    def __init__(self, wines: pd.DataFrame):
        self.wines = wines.reset_index(drop=True).copy()
        text = (self.wines["title"].fillna("") + " " + self.wines["variety"].fillna("") + " "
                + self.wines["province"].fillna("") + " " + self.wines["description"].fillna(""))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(text)

    # ----------------------------------------------------------------------------------
    @staticmethod
    def parse_filters(query: str) -> dict:
        q = query.lower()
        filters = {}
        for colour, words in COLOUR_WORDS.items():
            if any(re.search(rf"\b{w}\b", q) for w in words):
                filters["colour"] = colour
                break
        price = re.search(r"(?:under|below|less than|max|<|até)\s*(?:€|eur|\$)?\s*(\d+)", q)
        if price:
            filters["max_price"] = float(price.group(1))
        for region in REGIONS:
            # Word boundaries, or "port" would match every question about "Portuguese" wine.
            if re.search(rf"\b{re.escape(region)}\b", q):
                filters["region"] = region
                break
        return filters

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        filters = self.parse_filters(query)
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        df = self.wines.assign(score=scores)

        if "colour" in filters:
            df = df[df["colour"] == filters["colour"]]
        if "max_price" in filters:
            df = df[df["price"].fillna(1e9) <= filters["max_price"]]
        if "region" in filters:
            r = filters["region"]
            mask = (df["province"].fillna("").str.lower().str.contains(r)
                    | df["region_1"].fillna("").str.lower().str.contains(r)
                    | df["title"].fillna("").str.lower().str.contains(r))
            if mask.any():  # a region filter that empties the list is worse than none
                df = df[mask]

        # Relevance first, then critic score as tie-breaker.
        return df.sort_values(["score", "points"], ascending=False).head(k)

    @staticmethod
    def to_context(results: pd.DataFrame) -> str:
        if results.empty:
            return "(no matching wines in the catalog)"
        lines = []
        for _, w in results.iterrows():
            price = f"€{w['price']:.0f}" if pd.notna(w["price"]) else "price n/a"
            lines.append(f"- {w['title']} | {w['variety']} | {w['province']} | {w['points']} pts | "
                         f"{price} | {w['description']}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def load(path: str = str(config.PORTUGAL_CATALOG)) -> Catalog | None:
    """The shared catalog, or None if `python -m src.data --kaggle` has not been run."""
    if not Path(path).exists():
        return None
    return Catalog(pd.read_csv(path))
