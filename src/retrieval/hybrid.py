"""
Stage 5c — Hybrid Retrieval

Combines vector search (Stage 2) and BM25 (Stage 5a) using Reciprocal
Rank Fusion, then optionally reranks the merged pool (Stage 5b) to
produce the final set of chunks used for generation.

Why RRF instead of just averaging raw scores: vector similarity scores
(cosine, 0-1 range) and BM25 scores (unbounded, corpus-dependent) live
on completely different scales — averaging them directly would let
whichever score happens to have bigger numbers dominate. RRF sidesteps
this entirely by using each chunk's RANK (1st, 2nd, 3rd...) in each
list instead of its raw score, which is scale-independent by design.
"""
from src.retrieval.vector_store import query_similar
from src.retrieval.bm25_search import bm25_query
from src.retrieval.reranker import rerank
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _chunk_key(chunk: dict) -> tuple:
    """A stable identity for a chunk, used to deduplicate across the
    two ranked lists (same chunk can appear in both vector and BM25
    results)."""
    meta = chunk["metadata"]
    return (meta["page_number"], meta["section"], chunk["text"])

def _reciprocal_rank_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """
    RRF score for a chunk = sum over each list it appears in of
    1 / (k + rank). The score is attached as "rrf_score" so it can
    optionally be blended with the reranker's score later — a safety
    net in case a future reranker/document combination produces a
    weak or ambiguous top score, even though our current model
    (ms-marco-MiniLM-L-6-v2) has shown strong, confident separation.
    """
    scores: dict[tuple, float] = {}
    chunk_lookup: dict[tuple, dict] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunk_lookup[key] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        key = _chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunk_lookup[key] = chunk

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for key, rrf_score in merged:
        chunk_copy = dict(chunk_lookup[key])
        chunk_copy["rrf_score"] = rrf_score
        result.append(chunk_copy)
    return result

def hybrid_query(question: str, doc_id: str) -> list[dict]:
    """
    The main entrypoint Stage 4's retrieve_node calls. Falls back to
    pure vector search if hybrid/reranking are disabled in config.yaml
    — this makes it easy to A/B these techniques during Stage 7 eval.
    """
    config = load_config()
    retrieval_cfg = config["retrieval"]

    if not retrieval_cfg.get("use_hybrid", True):
        return query_similar(question, top_k=retrieval_cfg["final_top_k"], doc_id=doc_id)

    vector_results = query_similar(question, top_k=retrieval_cfg["vector_top_k"], doc_id=doc_id)
    bm25_results = bm25_query(question, doc_id=doc_id, top_k=retrieval_cfg["bm25_top_k"])

    logger.info(f"Vector hits: {len(vector_results)}, BM25 hits: {len(bm25_results)}")

    merged = _reciprocal_rank_fusion(vector_results, bm25_results, k=retrieval_cfg["rrf_k"])

    if retrieval_cfg.get("use_reranker", True):
        final = rerank(question, merged, top_k=retrieval_cfg["final_top_k"])
    else:
        final = merged[:retrieval_cfg["final_top_k"]]

    return final