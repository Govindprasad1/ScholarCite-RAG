"""
Diagnostic — confirms MLflow runs exist and shows them side by side,
bypassing the browser UI comparison feature entirely.
Run with: uv run python notebooks/check_mlflow_runs.py
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiments = mlflow.search_experiments()

for exp in experiments:
    if exp.name != "scholarcite-rag":
        continue
    print(f"Experiment: {exp.name} (id={exp.experiment_id})\n")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time ASC"])

    metric_cols = [c for c in runs.columns if c.startswith("metrics.") and "_n_valid" not in c]
    param_cols = ["params.use_hybrid", "params.use_reranker", "params.chunk_size",
                  "params.generation_model", "params.reranker_model"]

    display_cols = ["tags.mlflow.runName"] + [c for c in param_cols if c in runs.columns] + metric_cols
    display_cols = [c for c in display_cols if c in runs.columns]

    print(runs[display_cols].to_string())