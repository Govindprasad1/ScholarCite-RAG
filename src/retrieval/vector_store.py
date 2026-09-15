"""
STAGE 2b — Vector Store (ChromaDB)

Goal: persist chunk embeddings + metadata to disk, and query the top-k
nearest neighbors for a given query embedding.

Key functions to build:

    get_collection() -> chromadb.Collection
    add_chunks(chunks: list[dict]) -> None
    query(query_text: str, top_k: int) -> list[dict]

Chroma persists to disk at config.yaml's vector_store.persist_directory,
so re-running the app doesn't require re-embedding every time.
"""
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()


def get_collection():
    """
    TODO (build this together):
    import chromadb
    client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_directory"])
    return client.get_or_create_collection(cfg["vector_store"]["collection_name"])
    """
    raise NotImplementedError("Stage 2b: set up Chroma persistent client + collection")


def add_chunks(chunks: list[dict]) -> None:
    """
    TODO: embed chunk texts (via embeddings.embed_texts), then
    collection.add(ids=..., embeddings=..., documents=..., metadatas=...)
    """
    raise NotImplementedError("Stage 2b: implement add_chunks")


def query(query_text: str, top_k: int = 20) -> list[dict]:
    """
    TODO: embed query_text, then collection.query(query_embeddings=..., n_results=top_k)
    Return a normalized list of {"text": ..., "metadata": ..., "score": ...}
    """
    raise NotImplementedError("Stage 2b: implement vector query")
