"""
MLOps — Experiment Tracking with MLflow

Goal: log each pipeline "configuration" (chunk size, embedding model,
retrieval mode, LLM) alongside the resulting eval scores from
langsmith_eval.run_evaluation(), so you can compare runs over time.

This directly answers the common interview question: "how do you track
and compare experiments?"

Key function to build:

    log_run(config: dict, metrics: dict) -> None
        Example:
            log_run(
                config={"chunk_size": 500, "embedding_model": "bge-small", "use_hybrid": True},
                metrics={"retrieval_precision": 0.85, "faithfulness": 0.90},
            )
"""
import mlflow
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()


def log_run(config: dict, metrics: dict) -> None:
    """
    TODO (build this together):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
    """
    raise NotImplementedError("MLOps stage: implement MLflow run logging")
