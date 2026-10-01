"""
Diagnostic — checks why the Regularization question fails to retrieve
relevant content, even though the paper contains a Regularization section.
Run with: uv run python notebooks/diagnose_regularization.py data/uploads/AIAYN.pdf
"""
import sys
import uuid
import sys
from pathlib import Path

# Add project root to Python import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document, query_similar
from src.retrieval.bm25_search import bm25_query
from src.retrieval.hybrid import hybrid_query

pdf_path = sys.argv[1]
doc_id = f"diag_reg_{uuid.uuid4().hex[:8]}"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)
print("=== Chunks containing 'regulariz' (ground truth) ===")
for i, c in enumerate(chunks):
    if "regulariz" in c["text"].lower():
        print(f"Chunk {i} | page={c['metadata']['page_number']} | section={c['metadata']['section']}")
        print(f"Full length: {len(c['text'])} characters")
        print(f"FULL TEXT:\n{c['text']}\n")
        print("=" * 80)
question = "Summarize the three types of regularization used during training"

print("=== Vector search top 10 ===")
for r in query_similar(question, top_k=10, doc_id=doc_id):
    has_reg = "regulariz" in r["text"].lower()
    print(f"  score={r['score']:.3f} page={r['metadata']['page_number']}{'  <-- REG' if has_reg else ''}")

print("\n=== BM25 top 10 ===")
for r in bm25_query(question, doc_id=doc_id, top_k=10):
    has_reg = "regulariz" in r["text"].lower()
    print(f"  score={r['score']:.3f} page={r['metadata']['page_number']}{'  <-- REG' if has_reg else ''}")

print("\n=== Final hybrid output ===")
for r in hybrid_query(question, doc_id=doc_id):
    has_reg = "regulariz" in r["text"].lower()
    print(f"  rerank_score={r.get('rerank_score', 0):.3f} page={r['metadata']['page_number']}{'  <-- REG' if has_reg else ''}")

delete_document(doc_id)