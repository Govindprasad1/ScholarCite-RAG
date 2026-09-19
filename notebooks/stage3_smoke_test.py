"""
Manual smoke test for Stage 3 — not part of the test suite.
Run with: uv run python notebooks/stage3_smoke_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks
from src.llm.generate import answer_question

pages = extract_pages("tests/fixtures/sample.pdf")
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id="smoke_test_stage3")
print(f"Indexed {len(chunks)} chunks.\n")

questions = [
    "What method does the paper propose?",
    "What was the faithfulness improvement reported?",
    "What is the capital of France?",  # should be refused — not in the paper
]

for q in questions:
    print(f"Q: {q}")
    result = answer_question(q, doc_id="smoke_test_stage3")
    print(f"A: {result['answer']}\n")
    print("-" * 60)