"""
Stage 2 — Vector Store
Thin wrapper around a persistent ChromaDB collection. Nothing else in
the codebase should import chromadb directly — everything goes through
here, so the vector DB could be swapped later (e.g. for Qdrant) without
touching any other stage.
"""

import uuid
from typing import List, Dict, Any

import chromadb

from src.retrieval.embeddings import embed_texts, embed_query
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_collection():
    """Open (or create) the persistent Chroma collection defined in config.yaml."""
    config = load_config()
    persist_dir = config["vector_store"]["persist_directory"]
    collection_name = config["vector_store"]["collection_name"]

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks(chunks: List[Dict[str, Any]], doc_id: str) -> int:
    """
    Embed and store a list of Stage 1 chunks in the vector store.
    Each chunk looks like:
        {"text": "...", "metadata": {"page_number": 1, "section": "Abstract"}}

    doc_id tags every chunk with which source PDF it came from, so
    searches can later be scoped to one document, or a document's
    chunks can be deleted on re-upload.
    """
    if not chunks:
        logger.warning("add_chunks called with an empty chunk list.")
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    ids = [f"{doc_id}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = []
    for c in chunks:
        meta = dict(c["metadata"])
        meta["doc_id"] = doc_id
        metadatas.append(meta)

    collection = get_collection()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(f"Added {len(chunks)} chunks to vector store for doc_id={doc_id}")
    return len(chunks)


def query_similar(question: str, top_k: int = None, doc_id: str = None) -> List[Dict[str, Any]]:
    """Return the top-k most similar chunks to a question, optionally scoped to one doc_id."""
    config = load_config()
    if top_k is None:
        top_k = config["vector_store"].get("top_k", 5)

    query_embedding = embed_query(question)
    collection = get_collection()

    where_filter = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    hits = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": text,
            "metadata": meta,
            "score": 1 - distance,  # cosine distance -> similarity, higher is better
        })
    return hits


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a given document (useful for re-uploads)."""
    collection = get_collection()
    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted all chunks for doc_id={doc_id}")

def get_all_chunks_for_doc(doc_id: str) -> list[dict]:
    """
    Fetch every chunk belonging to a document, regardless of similarity
    to any query — needed by BM25, which builds its index over the full
    document rather than a pre-filtered candidate set.
    """
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})

    chunks = []
    for text, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({"text": text, "metadata": meta})
    return chunks