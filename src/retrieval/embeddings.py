"""
STAGE 2a — Embeddings

Goal: convert text (chunks and queries) into vectors using a local,
free, open-source embedding model — no API key needed.

Key function/class to build:

    get_embedding_model() -> SentenceTransformer
    embed_texts(texts: list[str]) -> list[list[float]]

Model: BAAI/bge-small-en-v1.5 (set in config.yaml under embedding.model_name)
Runs on CPU fine for a project this size.

IMPORTANT: always use the SAME model for indexing chunks and embedding
queries at search time — never mix embedding models.
"""
from functools import lru_cache
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    TODO (build this together):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg["embedding"]["model_name"], device=cfg["embedding"]["device"])
    """
    raise NotImplementedError("Stage 2a: load the embedding model")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    TODO: model = get_embedding_model(); return model.encode(texts).tolist()
    """
    raise NotImplementedError("Stage 2a: implement embed_texts")
