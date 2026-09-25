"""
Diagnostic — finds exactly which chunk(s) contain the Adam optimizer
details, and checks their rank in vector search, BM25, and the final
reranked hybrid output for the Query 2 question.

Run with: uv run python notebooks/diagnose_query2.py data/uploads/AIAYN.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, query_similar, delete_document
from src.retrieval.bm25_search import bm25_query
from src.retrieval.hybrid import hybrid_query

pdf_path = sys.argv[1]
doc_id = f"diag_q2_{uuid.uuid4().hex[:8]}"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)

# Step A: which raw chunks actually contain "Adam" — the ground truth
print("=== Chunks containing 'Adam optimizer' (ground truth) ===")
for i, c in enumerate(chunks):
    if "adam" in c["text"].lower():
        print(f"  Chunk index {i} | page={c['metadata']['page_number']} | section={c['metadata']['section']}")
        print(f"  Text: {c['text'][:200]}...\n")

question = "What optimizer and learning rate schedule did they use for training?"

# Step B: where does it rank in plain vector search (wide net)
print("\n=== Vector search top 10 (checking if Adam chunk appears) ===")
vector_results = query_similar(question, top_k=10, doc_id=doc_id)
for rank, r in enumerate(vector_results, 1):
    has_adam = "adam" in r["text"].lower()
    marker = "  <-- CONTAINS ADAM" if has_adam else ""
    print(f"  #{rank} score={r['score']:.3f} | page={r['metadata']['page_number']}{marker}")

# Step C: where does it rank in BM25 (wide net)
print("\n=== BM25 top 10 (checking if Adam chunk appears) ===")
bm25_results = bm25_query(question, doc_id=doc_id, top_k=10)
for rank, r in enumerate(bm25_results, 1):
    has_adam = "adam" in r["text"].lower()
    marker = "  <-- CONTAINS ADAM" if has_adam else ""
    print(f"  #{rank} score={r['score']:.3f} | page={r['metadata']['page_number']}{marker}")

# Step D: final hybrid output (current config)
print("\n=== Final hybrid output (current config) ===")
hybrid_results = hybrid_query(question, doc_id=doc_id)
for rank, r in enumerate(hybrid_results, 1):
    has_adam = "adam" in r["text"].lower()
    marker = "  <-- CONTAINS ADAM" if has_adam else ""
    print(f"  #{rank} rerank_score={r.get('rerank_score', 0):.3f} | page={r['metadata']['page_number']}{marker}")

# Check if the Adam chunk might be getting truncated by the reranker
adam_chunk = chunks[30]
print(f"\n=== Adam chunk full length ===")
print(f"Character count: {len(adam_chunk['text'])}")
print(f"Approx token count: {len(adam_chunk['text']) // 4}")
print(f"\nFull text:\n{adam_chunk['text']}")

from src.retrieval.reranker import get_reranker

model = get_reranker()

print(f"\nReranker max_length setting: {model.max_seq_length}")
delete_document(doc_id)