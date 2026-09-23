"""
Stage 5b — Cross-Encoder Reranking

A bi-encoder (Stage 2's embedding model) embeds the query and each
chunk SEPARATELY and compares vectors — fast, scalable, but less
precise. A cross-encoder looks at the query and chunk TOGETHER in one
pass, which is far more accurate but too slow to run over an entire
document — which is exactly why it's only applied here, to the small
merged candidate pool from hybrid.py, not the whole corpus.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    config = load_config()
    model_name = config["retrieval"]["reranker_model"]
    logger.info(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name)


def rerank(question: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score a candidate pool with the cross-encoder and return the
    top_k best, most relevant chunks.

    Note: BAAI/bge-reranker-base's CrossEncoder already applies a
    sigmoid activation internally, so model.predict() returns scores
    already in a 0-1 relevance range — no manual activation needed
    here. (Confirmed by diagnostic: raw outputs were already small
    positive decimals, not unbounded logits.)
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    reranked = []
    for chunk, score in top:
        chunk_copy = dict(chunk)
        chunk_copy["rerank_score"] = float(score)
        reranked.append(chunk_copy)
    return reranked