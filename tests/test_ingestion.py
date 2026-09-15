"""
Tests for Stage 1 (ingestion + chunking).

Run with: pytest tests/test_ingestion.py -v

TODO: put a small sample PDF at tests/fixtures/sample.pdf before writing
these tests for real.
"""
import pytest


def test_extract_pages_returns_page_numbers():
    """Each extracted page dict should have 'page_number' and 'text' keys."""
    pytest.skip("TODO: implement once pdf_parser.extract_pages() is built")


def test_chunk_pages_respects_chunk_size():
    """Chunks should not wildly exceed configured chunk_size."""
    pytest.skip("TODO: implement once chunker.chunk_pages() is built")


def test_chunk_metadata_includes_page_number():
    """Every chunk must carry page_number metadata for citations to work later."""
    pytest.skip("TODO: implement once chunker.chunk_pages() is built")
