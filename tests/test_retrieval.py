"""
Tests for Stage 2 & 5 (embeddings, vector store, BM25, reranker).

Run with: pytest tests/test_retrieval.py -v
"""
import pytest


def test_embed_texts_returns_correct_dimensionality():
    pytest.skip("TODO: implement once embeddings.embed_texts() is built")


def test_vector_query_returns_top_k_results():
    pytest.skip("TODO: implement once vector_store.query() is built")


def test_known_query_retrieves_expected_chunk():
    """
    The most important retrieval test: given a question you know the
    answer to (from a real test PDF), does retrieval actually return
    the chunk containing that answer?
    """
    pytest.skip("TODO: implement once full retrieval pipeline is built")


def test_reranker_improves_ordering():
    pytest.skip("TODO: implement once reranker.rerank() is built")
