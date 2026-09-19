"""
Manual smoke test for Stage 2 — not part of the test suite.
Run with: uv run python notebooks/stage2_smoke_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, query_similar

pages = extract_pages("tests/fixtures/sample.pdf")
chunks = chunk_pages(pages)
print(f"Ingested {len(chunks)} chunks from sample.pdf")

add_chunks(chunks, doc_id="smoke_test_doc")
print("Chunks embedded and stored.")

question = "What dataset or method did the paper use?"
results = query_similar(question, top_k=3, doc_id="smoke_test_doc")

print(f"\nQuery: {question}\n")
for i, r in enumerate(results, 1):
    meta = r["metadata"]
    print(f"[{i}] score={r['score']:.3f} | page={meta['page_number']} | section={meta['section']}")
    print(f"    {r['text'][:150]}...\n")