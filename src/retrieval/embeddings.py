"""
Stage 2 — Embeddings
Wraps a sentence-transformers model so the rest of the app never talks
to the model directly. Model loading is lazy + cached so it only loads
once per process, not once per call.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load (once) and cache the embedding model defined in config.yaml."""
    config = load_config()
    model_name = config["embeddings"]["model_name"]
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings into vectors.
    bge-small-en-v1.5 doesn't require a special prefix for passages
    (only some bge variants need a query-side instruction prefix, and
    we keep that logic in embed_query below to stay explicit about it).
    """
    if not texts:
        return []
    config = load_config()
    model = get_embedding_model()
    batch_size = config["embeddings"].get("batch_size", 32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # normalized vectors make cosine similarity a simple dot product
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string (used at question-time, not ingestion-time)."""
    return embed_texts([query])[0]