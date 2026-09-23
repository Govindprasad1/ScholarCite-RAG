"""
Stage 5 quality smoke test — compares pure vector search vs. full
hybrid+rerank on a REAL document, using queries specifically chosen to
expose where each method's strengths/weaknesses show up.

Run with: uv run python notebooks/stage5_smoke_test.py path/to/paper.pdf
"""
import sys
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, query_similar, delete_document
from src.retrieval.hybrid import hybrid_query

if len(sys.argv) < 2:
    print("Usage: uv run python notebooks/stage5_smoke_test.py <path_to_pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
doc_id = f"smoke_test_{uuid.uuid4().hex[:8]}"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)
print(f"Indexed {len(chunks)} chunks from {pdf_path}\n")
print("=" * 80)

# Deliberately varied query types to stress-test different strengths:
queries = {
    "Semantic/paraphrase (vector search's strength)":
        "What is the main idea behind this architecture?",

    "Exact acronym/term (BM25's strength)":
    "What optimizer and learning rate schedule did they use for training?",

    "Exact number (BM25's strength)":
    "How many attention heads are used in the base Transformer model?",

    "Deliberately unanswerable (should show weak/irrelevant matches)":
    "What is the capital of France?",
}

print("⚠️  EDIT the queries dict above with real terms/numbers from YOUR paper before running for best results.\n")

for label, question in queries.items():
    print(f"### {label}")
    print(f"Q: {question}\n")

    print("--- Pure vector search ---")
    vector_only = query_similar(question, top_k=3, doc_id=doc_id)
    for r in vector_only:
        preview = r["text"][:100].replace("\n", " ")
        print(f"  score={r['score']:.3f} | page={r['metadata']['page_number']} | {preview}...")

    print("\n--- Hybrid (vector + BM25 + rerank) ---")
    hybrid = hybrid_query(question, doc_id=doc_id)
    for r in hybrid:
        preview = r["text"].replace("\n", " ")
        rerank_score = r.get("rerank_score", 0)
        print(f"  rerank_score={rerank_score:.3f} | page={r['metadata']['page_number']} | {preview}...")

    print("\n" + "=" * 80 + "\n")

# Clean up after ourselves — this was a throwaway smoke test document
delete_document(doc_id)
print(f"Cleaned up smoke test data for doc_id={doc_id}")
