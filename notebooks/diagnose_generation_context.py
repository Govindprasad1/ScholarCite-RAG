"""
Diagnostic — shows EXACTLY what chunks get passed to the generation
LLM for a specific question, so we can see if the right context made
it through, and if not, why the LLM said it couldn't answer.

Run with: uv run python notebooks/diagnose_generation_context.py data/uploads/AIAYN.pdf
"""
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.retrieval.hybrid import hybrid_query
from src.llm.prompts import format_context, GENERATION_PROMPT_TEMPLATE
from src.utils.config import load_config

pdf_path = sys.argv[1]
doc_id = f"diag_ctx_{uuid.uuid4().hex[:8]}"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)

cfg = load_config()
print(f"Current config final_top_k: {cfg['retrieval']['final_top_k']}\n")

question = "What optimizer and learning rate schedule did they use for training?"
retrieved = hybrid_query(question, doc_id=doc_id)

print(f"Number of chunks actually retrieved: {len(retrieved)}\n")
for i, r in enumerate(retrieved, 1):
    has_adam = "adam" in r["text"].lower()
    marker = "  <-- CONTAINS ADAM" if has_adam else ""
    print(f"Chunk {i} | page={r['metadata']['page_number']} | rerank={r.get('rerank_score', 'n/a'):.3f} | combined={r.get('combined_score', 'n/a'):.3f}{marker}")
print("\n=== FULL PROMPT SENT TO THE LLM ===\n")
context = format_context(retrieved)
prompt = GENERATION_PROMPT_TEMPLATE.format(context=context, question=question)
print(prompt)

delete_document(doc_id)