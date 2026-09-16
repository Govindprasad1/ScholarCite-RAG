"""
Tests for Stage 1 (ingestion + chunking).
Run with: pytest tests/test_ingestion.py -v
"""
from pathlib import Path
from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_extract_pages_returns_correct_page_count():
    pages = extract_pages(str(FIXTURE_PATH))
    assert len(pages) == 3


def test_extract_pages_returns_page_numbers_starting_at_one():
    pages = extract_pages(str(FIXTURE_PATH))
    assert [p["page_number"] for p in pages] == [1, 2, 3]


def test_extract_pages_contains_expected_text():
    pages = extract_pages(str(FIXTURE_PATH))
    assert "retrieval augmented generation" in pages[0]["text"].lower()
    assert "references" in pages[2]["text"].lower()


def test_chunk_pages_detects_sections():
    pages = extract_pages(str(FIXTURE_PATH))
    chunks = chunk_pages(pages)
    sections = {c["metadata"]["section"] for c in chunks}
    assert {"Abstract", "Introduction", "Method", "Experiments", "References"}.issubset(sections)


def test_chunk_metadata_includes_page_number():
    pages = extract_pages(str(FIXTURE_PATH))
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert "page_number" in chunk["metadata"]
        assert isinstance(chunk["metadata"]["page_number"], int)


def test_abstract_chunk_maps_to_page_one():
    pages = extract_pages(str(FIXTURE_PATH))
    chunks = chunk_pages(pages)
    abstract_chunks = [c for c in chunks if c["metadata"]["section"] == "Abstract"]
    assert len(abstract_chunks) >= 1
    assert abstract_chunks[0]["metadata"]["page_number"] == 1


def test_references_chunk_maps_to_page_three():
    pages = extract_pages(str(FIXTURE_PATH))
    chunks = chunk_pages(pages)
    ref_chunks = [c for c in chunks if c["metadata"]["section"] == "References"]
    assert len(ref_chunks) >= 1
    assert ref_chunks[0]["metadata"]["page_number"] == 3


def test_no_chunk_is_empty():
    pages = extract_pages(str(FIXTURE_PATH))
    chunks = chunk_pages(pages)
    for chunk in chunks:
        assert chunk["text"].strip() != ""