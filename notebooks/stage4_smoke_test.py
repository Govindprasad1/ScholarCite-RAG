"""
Manual smoke test for Stage 4 — proves the verification loop catches
unsupported claims, now runnable against any real PDF.

Run with: uv run python notebooks/stage4_smoke_test.py path/to/paper.pdf
"""
import sys
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.graph.build_graph import run

if len(sys.argv) < 2:
    print("Usage: uv run python notebooks/stage4_smoke_test.py <path_to_pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
doc_id = f"smoke_test_stage4_{uuid.uuid4().hex[:8]}"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)
print(f"Indexed {len(chunks)} chunks from {pdf_path}\n")

# Edit these to fit whatever paper you're testing:
questions = [
    "How many attention heads are used in the base Transformer model?",   # answerable — should verify cleanly
    "What optimizer and learning rate schedule did they use for training?",  # answerable
    "What does BLEU measure?",                                            # topically related but NOT stated in the paper — good test of refusal
    "What is the capital of France?",                                     # completely unrelated — should refuse immediately
]

for q in questions:
    print(f"Q: {q}")
    result = run(q, doc_id=doc_id)
    print(f"Status: {result['status']} | Attempts: {result['attempts']}")
    print(f"Final Answer: {result['final_answer']}\n")
    if result.get("verification"):
        for claim in result["verification"].get("claims", []):
            mark = "✅" if claim["supported"] else "❌"
            print(f"  {mark} {claim['claim']}")
    print("-" * 70)

delete_document(doc_id)
print(f"\nCleaned up smoke test data for doc_id={doc_id}")