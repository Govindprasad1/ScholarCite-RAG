"""
Diagnostic — reproduces the blank-answer bug for a specific question.
Run with: uv run python notebooks/diagnose_blank_answer.py data/uploads/AIAYN.pdf
"""
import sys
from pathlib import Path

# Add project root to Python import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uuid

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.graph.build_graph import run as run_pipeline

pdf_path = sys.argv[1]
doc_id = f"diag_blank_{uuid.uuid4().hex[:8]}"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)

question = "What hyperparameters differ between the base and big Transformer models, and where is each one specified?"
result = run_pipeline(question, doc_id=doc_id)

print(f"final_answer: {result['final_answer']!r}")
print(f"raw answer field: {result.get('answer')!r}")
print(f"status: {result['status']}")
print(f"attempts: {result['attempts']}")
print(f"verification: {result.get('verification')}")

delete_document(doc_id)