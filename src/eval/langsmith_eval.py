"""
STAGE 7 — Evaluation with LangSmith

Goal: build a small hand-labeled test set (10-20 question/answer pairs
from a paper you know well) and run automated evaluators for:
  - retrieval precision (did we fetch the chunk containing the answer?)
  - faithfulness (does the generated answer match the source chunks?)

Key pieces to build:

    TEST_SET: list[dict]  e.g. [{"question": "...", "expected_answer": "...", "expected_page": 3}]
    run_evaluation(app) -> dict  (aggregate scores, also logged to MLflow — see mlflow_tracking.py)

LangSmith tracing itself just needs these env vars set (see .env.example):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=...
    LANGCHAIN_PROJECT=ScholarCite-RAG
Once set, every LangChain/LangGraph call is automatically traced — no
extra code needed for basic tracing, only for the structured eval below.
"""
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEST_SET: list[dict] = [
    # TODO: fill in with real Q&A pairs once you have a test PDF.
    # {"question": "What method does the paper propose?", "expected_page": 2},
]


def run_evaluation(app) -> dict:
    """
    TODO: for each item in TEST_SET, run app.invoke(...), compare the
    retrieved page(s) and answer faithfulness against expectations,
    aggregate into {"retrieval_precision": ..., "faithfulness": ...}
    """
    raise NotImplementedError("Stage 7: implement evaluation loop")
