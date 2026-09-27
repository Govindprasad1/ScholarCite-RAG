"""
Stage 7 — Ragas Evaluation
...
Note: requires langchain-community<0.4 — ragas 0.4.x's internal LLM
factory imports ChatVertexAI from a langchain_community submodule that
was reorganized/removed in the 0.4.x "sunset" release. Pinned in
pyproject.toml until ragas updates its internal import path.
"""
"""
Stage 7 — Full evaluation run: Ragas scoring + MLflow logging.


Run with: uv run python notebooks/run_stage7_eval.py data/uploads/AIAYN.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.eval.ragas_eval import run_evaluation
from src.eval.mlflow_tracking import log_run

if len(sys.argv) < 2:
    print("Usage: uv run python notebooks/run_stage7_eval.py <path_to_pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
doc_id = "stage7_eval_doc"

pages = extract_pages(pdf_path)
chunks = chunk_pages(pages)
add_chunks(chunks, doc_id=doc_id)
print(f"Indexed {len(chunks)} chunks.\n")

scores, df = run_evaluation(doc_id)

print("\n=== AGGREGATE RAGAS SCORES ===")
for k, v in scores.items():
    if v is None:
        print(f"  {k}: N/A (0 valid samples)")
    elif isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")

log_run(scores, run_name="baseline_chunk300_hybrid_on")

delete_document(doc_id)
print("\nLogged to MLflow. Run 'mlflow ui' to view the dashboard.")