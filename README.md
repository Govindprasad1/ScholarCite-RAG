# ScholarCite RAG

**A Citation-Grounded RAG Assistant for Academic Documents**

Answers questions about research papers, textbooks, and other academic
PDFs with **page/section-level citations**, and **rejects claims it
can't verify** against the source document — built on a fully
free/open-source stack.

> 🚧 This repo is currently a scaffold. Modules contain docstrings and
> `TODO`s describing exactly what to build at each stage. See
> `scholarcite-rag-architecture.md` for the full system design.

## Tech Stack
- **Ingestion**: PyMuPDF
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, free)
- **Vector Store**: ChromaDB
- **Hybrid Retrieval**: Vector search + BM25 + `bge-reranker-base`
- **Orchestration**: LangGraph (retrieve → generate → verify → decide)
- **LLM**: Groq (free tier, Llama 3.x)
- **Evaluation**: LangSmith
- **Experiment Tracking**: MLflow
- **UI**: Streamlit
- **MLOps**: Docker, GitHub Actions CI, pytest

## Setup

1. **Clone & create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up API keys** (both free)
   ```bash
   cp .env.example .env
   ```
   - Groq key: https://console.groq.com/keys
   - LangSmith key: https://smith.langchain.com/settings

   Fill both into `.env`.

3. **Run tests** (mostly skipped until you build each stage)
   ```bash
   pytest tests/ -v
   ```

4. **Run the app** (once built)
   ```bash
   streamlit run app.py
   ```

   Or with Docker:
   ```bash
   docker compose up --build
   ```

## Build Order

Follow this order — each stage is a real, working milestone, not just a step:

| Stage | What | Files |
|---|---|---|
| 1 | PDF parsing + chunking | `src/ingestion/` |
| 2 | Embeddings + vector store | `src/retrieval/embeddings.py`, `vector_store.py` |
| 3 | Basic retrieve → generate chain | `src/llm/groq_client.py` |
| 4 | LangGraph verification loop | `src/graph/` |
| 5 | Hybrid search (BM25) + reranking | `src/retrieval/bm25_search.py`, `reranker.py` |
| 6 | Config-driven pipeline | `config.yaml` (already scaffolded) |
| 7 | LangSmith evaluation | `src/eval/langsmith_eval.py` |
| — | MLflow experiment tracking | `src/eval/mlflow_tracking.py` |
| 8 | Streamlit UI | `app.py` |
| — | Docker + CI | `Dockerfile`, `.github/workflows/ci.yml` (already scaffolded) |
| 9 | Deploy to HuggingFace Spaces | — |

## Project Structure

```
scholarcite-rag/
├── app.py                      # Streamlit entrypoint
├── config.yaml                 # All tunable pipeline settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml    # Test-on-push CI
├── src/
│   ├── ingestion/               # Stage 1
│   ├── retrieval/                # Stage 2, 5
│   ├── graph/                    # Stage 4
│   ├── llm/                      # Stage 3
│   ├── eval/                     # Stage 7 + MLflow
│   └── utils/                    # config loader, logger
├── tests/                       # pytest, mirrors src/ structure
├── data/
│   ├── uploads/                  # gitignored — uploaded PDFs
│   └── chroma_db/                 # gitignored — persisted vector store
└── notebooks/                   # scratch space for prototyping each stage
```

## Notes on Free-Tier Constraints
- Groq's free tier has request-rate limits — fine for demo/dev use, but
  don't hammer it in tight test loops.
- Embeddings and reranking run locally on CPU — fine for this project's
  scale, just don't expect GPU-speed throughput.
