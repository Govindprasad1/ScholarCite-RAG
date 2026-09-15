"""
STAGE 5b — Reranking (Hybrid Retrieval, part 2)

Goal: after merging vector + BM25 candidates (~30-40 chunks), use a
cross-encoder reranker to re-score them more precisely and keep only
the best final_top_k (config.yaml -> retrieval.final_top_k).

A cross-encoder looks at (query, chunk) together — more accurate than
the bi-encoder embedding model, but slower, which is why it's only run
on the smaller merged candidate set, not the whole document.

Key function to build:

    rerank(query_text: str, candidates: list[dict], top_k: int) -> list[dict]

Model: BAAI/bge-reranker-base (config.yaml -> retrieval.reranker_model)
"""
from functools import lru_cache
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()


@lru_cache(maxsize=1)
def get_reranker_model():
    """
    TODO: from FlagEmbedding import FlagReranker
    return FlagReranker(cfg["retrieval"]["reranker_model"], use_fp16=False)
    """
    raise NotImplementedError("Stage 5b: load reranker model")


def rerank(query_text: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    TODO: score each (query_text, candidate["text"]) pair with the reranker,
    sort descending, return top_k candidates.
    """
    raise NotImplementedError("Stage 5b: implement reranking")
