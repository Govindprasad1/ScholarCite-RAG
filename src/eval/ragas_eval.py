"""
Stage 7 — Ragas Evaluation

Runs the full pipeline (ingestion already done once per doc; retrieval +
generation + verification per question) against a hand-labeled test set,
then scores the results with Ragas' four core RAG metrics:

  - Faithfulness: is the answer grounded in the retrieved context?
  - Response Relevancy: does the answer address the question asked?
  - Context Precision: are relevant chunks ranked near the top?
  - Context Recall: did retrieval fetch everything needed to answer?

Uses your existing Groq LLM (via LangchainLLMWrapper) as the judge and
your existing local bge-small embeddings (via LangchainEmbeddingsWrapper)
for the relevancy metric — no extra API keys needed beyond what you
already have.
"""
import json
from pathlib import Path
import time
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from src.graph.build_graph import run as run_pipeline
from src.llm.groq_client import get_llm
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_dataset.json"


def load_eval_questions() -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_samples(doc_id: str) -> list[SingleTurnSample]:
    """
    Runs the real pipeline for every question in the test set and builds
    Ragas SingleTurnSamples from the actual retrieved context + answer —
    this evaluates your ACTUAL system behavior, not a mocked shortcut.
    """
    questions = load_eval_questions()
    #questions = questions[:3]  # uncomment to test with fewer questions first
    samples = []

    for item in questions:
        logger.info(f"Running pipeline for eval question: {item['question']}")
        result = run_pipeline(item["question"], doc_id=doc_id)

        retrieved_texts = [c["text"] for c in result.get("retrieved_chunks", [])]
        answer = result.get("final_answer") or "I cannot answer this from the given context."

        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=retrieved_texts if retrieved_texts else ["(no context retrieved)"],
            reference=item["reference"],
        ))
        time.sleep(2)

    return samples

from ragas.run_config import RunConfig

def run_evaluation(doc_id: str) -> dict:
    samples = collect_samples(doc_id)
    dataset = EvaluationDataset(samples=samples)

    generation_llm = get_llm(role="generation")
    judge_llm = LangchainLLMWrapper(generation_llm)

    config = load_config()
    embedding_model_name = config["embeddings"]["model_name"]
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=embedding_model_name)
    )

    # Low concurrency + generous retries/timeout to stay under Groq's
    # free-tier rate limits, since Ragas fires many extra judge LLM
    # calls per sample (multiple per metric).
    run_config = RunConfig(max_workers=2, max_retries=5, timeout=180)

    logger.info(f"Running Ragas evaluation on {len(samples)} samples...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),  # Groq's API only supports n=1 completions per request
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    logger.info(f"\n{df.to_string()}")

    # Report both the mean AND how many non-null samples backed it,
    # so a thin/unreliable average is visible rather than hidden.
    aggregate = {}
    for col in ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]:
        if col in df.columns:
            valid = df[col].dropna()
            aggregate[col] = float(valid.mean()) if len(valid) > 0 else None
            aggregate[f"{col}_n_valid"] = len(valid)

    return aggregate, df


if __name__ == "__main__":
    import sys
    from src.ingestion.pdf_parser import extract_pages
    from src.ingestion.chunker import chunk_pages
    from src.retrieval.vector_store import add_chunks, delete_document

    if len(sys.argv) < 2:
        print("Usage: uv run python -m src.eval.ragas_eval <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc_id = "eval_run_doc"

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    add_chunks(chunks, doc_id=doc_id)

    scores, df = run_evaluation(doc_id)

    print("\n=== AGGREGATE SCORES ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    delete_document(doc_id)