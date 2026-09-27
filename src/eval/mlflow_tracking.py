"""
Stage 7 — MLflow Experiment Tracking

Logs each evaluation run's configuration (chunk size, top_k, reranker
model, etc.) alongside its Ragas scores, so different pipeline
configurations can be compared side by side over time.
"""
import mlflow

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def log_run(scores: dict, run_name: str = None) -> None:
    config = load_config()
    mlflow_config = config.get("mlflow", {})

    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "./mlruns"))
    mlflow.set_experiment(mlflow_config.get("experiment_name", "scholarcite-rag"))

    # mlflow.log_metrics cannot accept None values — filter them out,
    # but keep a note in params so it's visible a metric failed to compute.
    numeric_scores = {k: v for k, v in scores.items() if v is not None}
    failed_metrics = [k for k, v in scores.items() if v is None]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "chunk_size": config["chunking"]["chunk_size"],
            "chunk_overlap": config["chunking"]["chunk_overlap"],
            "chunking_strategy": config["chunking"]["strategy"],
            "vector_top_k": config["retrieval"]["vector_top_k"],
            "bm25_top_k": config["retrieval"]["bm25_top_k"],
            "final_top_k": config["retrieval"]["final_top_k"],
            "use_hybrid": config["retrieval"]["use_hybrid"],
            "use_reranker": config["retrieval"]["use_reranker"],
            "reranker_model": config["retrieval"]["local_reranker_model"],
            "generation_model": config["llm"]["generation_model"],
            "verification_model": config["llm"]["verification_model"],
            "failed_metrics": ",".join(failed_metrics) if failed_metrics else "none",
        })
        mlflow.log_metrics(numeric_scores)
        logger.info(f"Logged run to MLflow: {numeric_scores} (failed: {failed_metrics})")