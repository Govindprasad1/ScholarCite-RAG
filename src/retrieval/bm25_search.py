"""
STAGE 5a — BM25 Keyword Search (Hybrid Retrieval, part 1)

Goal: catch exact-term matches (acronyms, equation names, specific
numbers) that vector/semantic search sometimes misses.

Key functions to build:

    build_bm25_index(chunks: list[dict]) -> BM25Okapi
    bm25_query(index, chunks: list[dict], query_text: str, top_k: int) -> list[dict]

Uses the rank_bm25 library. This step comes AFTER you have basic
vector retrieval working (Stage 2-3) — don't build this first.
"""
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_bm25_index(chunks: list[dict]):
    """
    TODO: from rank_bm25 import BM25Okapi
    tokenized = [c["text"].split() for c in chunks]
    return BM25Okapi(tokenized)
    """
    raise NotImplementedError("Stage 5a: build BM25 index")


def bm25_query(index, chunks: list[dict], query_text: str, top_k: int = 20) -> list[dict]:
    """
    TODO: score = index.get_scores(query_text.split())
    Return top_k chunks sorted by score, same normalized shape as vector_store.query()
    """
    raise NotImplementedError("Stage 5a: implement BM25 query")
