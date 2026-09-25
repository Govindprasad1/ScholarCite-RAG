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
    model_name = config["retrieval"]["local_reranker_model"]
    logger.info(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name, max_length=512)

import math

def rerank(question: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score candidates with the cross-encoder, then blend with the
    original RRF score as a safety net. ms-marco-MiniLM outputs
    unbounded logits (can be strongly positive or negative), so we
    pass them through a sigmoid to normalize into 0-1 before blending
    with RRF's naturally small fractional scores — otherwise the two
    scales would be incomparable.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(question, c["text"]) for c in candidates]
    raw_scores = model.predict(pairs)
    normalized_rerank = [1 / (1 + math.exp(-s)) for s in raw_scores]

    rrf_scores = [c.get("rrf_score", 0.0) for c in candidates]
    max_rrf = max(rrf_scores) if rrf_scores else 1.0
    normalized_rrf = [s / max_rrf if max_rrf > 0 else 0.0 for s in rrf_scores]

    # Reranker carries most of the weight since it's shown strong,
    # confident discrimination on real test cases; RRF acts only as a
    # light tiebreaker/safety net.
    RERANK_WEIGHT = 0.85
    RRF_WEIGHT = 0.15

    combined = []
    for candidate, r_score, norm_rerank, norm_rrf in zip(candidates, raw_scores, normalized_rerank, normalized_rrf):
        blended = (RERANK_WEIGHT * norm_rerank) + (RRF_WEIGHT * norm_rrf)
        combined.append((candidate, r_score, blended))

    combined.sort(key=lambda x: x[2], reverse=True)
    top = combined[:top_k]

    reranked = []
    for candidate, r_score, blended in top:
        chunk_copy = dict(candidate)
        chunk_copy["rerank_score"] = float(r_score)
        chunk_copy["combined_score"] = float(blended)
        reranked.append(chunk_copy)
    return reranked