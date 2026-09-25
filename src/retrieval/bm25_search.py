"""
Stage 5a — BM25 Keyword Search

Catches exact-term matches (acronyms, specific numbers, named methods)
that vector/semantic search sometimes misses, since embeddings capture
general meaning rather than precise tokens.

Note: BM25 has no persistence of its own — the index is rebuilt from
Chroma's stored chunks on each query. This is fine at this project's
scale (a handful of documents, each with tens to low hundreds of
chunks); if this ever needed to scale to huge corpora, the index would
be cached instead of rebuilt per call.
"""
from rank_bm25 import BM25Okapi
from functools import lru_cache
from src.retrieval.vector_store import get_all_chunks_for_doc
from src.utils.logger import get_logger

logger = get_logger(__name__)


import re

def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation, so 'heads,' and 'heads' match."""
    return re.findall(r"\b\w+\b", text.lower())


def bm25_query(question: str, doc_id: str, top_k: int = 15) -> list[dict]:
    bm25, chunks = _get_bm25_index(doc_id)
    if bm25 is None or not chunks:
        logger.warning(f"No chunks found for doc_id={doc_id} — BM25 index is empty.")
        return []

    tokenized_query = _tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = scored_chunks[:top_k]

    return [{"text": c["text"], "metadata": c["metadata"], "score": float(score)}
            for c, score in top_chunks]
@lru_cache(maxsize=8)
def _get_bm25_index(doc_id: str):
    """
    Cached per doc_id — avoids rebuilding the BM25 index from scratch
    on every single query within the same session. Must be invalidated
    (see invalidate_bm25_cache below) whenever a document's chunks
    change, e.g. on delete/re-upload.

    Returns (None, []) if the document has no chunks at all — BM25Okapi
    crashes with a ZeroDivisionError on an empty corpus, so we short-
    circuit before ever constructing it.
    """
    chunks = get_all_chunks_for_doc(doc_id)
    if not chunks:
        return None, []

    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus), chunks
def invalidate_bm25_cache(doc_id: str = None) -> None:
    """
    Call this after add_chunks() or delete_document() for a doc_id, so
    a stale cached index doesn't linger. lru_cache doesn't support
    per-key clearing, so we clear the whole cache — fine at this
    project's scale (a handful of documents at once).
    """
    _get_bm25_index.cache_clear()