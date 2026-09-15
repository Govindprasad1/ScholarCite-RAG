# ScholarCite RAG — System Architecture
### A Citation-Grounded RAG Assistant for Academic Documents

## 1. Problem Statement
Students/researchers need to ask questions about dense papers/textbooks and get
answers that are **traceable to a specific page/section** — not confident-sounding
guesses. The system must also flag when it *cannot* answer from the provided
material, rather than hallucinating.

## 2. High-Level Architecture

```
                         ┌─────────────────────────┐
                         │   User (Streamlit UI)    │
                         │  - Upload PDF            │
                         │  - Ask question           │
                         │  - View answer + citations│
                         └────────────┬─────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │      INGESTION          │
                          │  PyMuPDF → page text    │
                          │  Section-aware chunking │
                          │  Metadata: page, section│
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │      EMBEDDING          │
                          │  bge-small-en-v1.5      │
                          │  (sentence-transformers)│
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │     VECTOR STORE        │
                          │      ChromaDB           │
                          │  (persisted to disk)    │
                          └───────────┬────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │         HYBRID RETRIEVAL             │
                    │  Vector search (Chroma)  +  BM25     │
                    │  → merge → rerank (bge-reranker)     │
                    │  → top-k chunks with metadata        │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │     LANGGRAPH ORCHESTRATION         │
                    │                                     │
                    │   [Retrieve] → [Generate Answer]    │
                    │         → [Verify Grounding]        │
                    │         → if fail: [Regenerate /     │
                    │              Flag Unsupported]      │
                    │         → if pass: [Return Answer]  │
                    └─────────────────┬─────────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │   LLM (Groq — Llama 3.x) │
                          │  used for both generation │
                          │  and verification calls   │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │   Answer + Citations     │
                          │  [Page X, Section Y]     │
                          │   returned to UI          │
                          └────────────────────────┘

        (parallel) ──► LangSmith: traces every run, stores eval dataset,
                        scores retrieval precision + faithfulness
```

## 3. Component Responsibilities

| Component | Responsibility | Tech |
|---|---|---|
| Ingestion | Extract text per page, preserve structure, attach metadata | PyMuPDF |
| Chunking | Split into coherent, section-aware chunks with overlap | LangChain splitters |
| Embedding | Convert chunks + queries into vectors | sentence-transformers (bge-small) |
| Vector Store | Store & retrieve nearest-neighbor chunks | ChromaDB |
| Keyword Search | Catch exact-term matches vector search misses | rank_bm25 |
| Reranker | Re-score top candidates for precision | bge-reranker-base |
| Orchestration | Control flow: retrieve → generate → verify → decide | LangGraph |
| LLM | Generation + self-verification | Groq API (Llama 3.x) |
| Evaluation | Quantify retrieval/faithfulness quality | LangSmith |
| UI | Upload, ask, display answer + citations | Streamlit |
| Deployment | Host the live app | HuggingFace Spaces |

## 4. Data Flow (Single Query Lifecycle)

1. User uploads PDF → ingested once, chunked, embedded, stored in Chroma (cached — not repeated per question).
2. User asks a question.
3. Question embedded → vector search (top ~20) + BM25 search (top ~20) → merged.
4. Reranker scores merged candidates → keep top 5.
5. LangGraph "Generate" node: LLM answers **using only these 5 chunks**, citing page/section per claim.
6. LangGraph "Verify" node: separate LLM call checks each claim against the same 5 chunks → pass/fail per claim.
7. If any claim fails → either strip it, flag it to the user, or trigger one regeneration attempt.
8. Final answer + citations returned to UI.
9. Entire run logged to LangSmith (inputs, retrieved chunks, generated answer, verification result).

## 5. Repo / Folder Structure (what we'll build)

```
scholarcite-rag/
├── app.py                     # Streamlit entrypoint
├── requirements.txt
├── .env.example                # GROQ_API_KEY, LANGCHAIN_API_KEY (no real keys committed)
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py       # PyMuPDF extraction, page-aware
│   │   └── chunker.py          # section-aware chunking
│   ├── retrieval/
│   │   ├── embeddings.py       # bge-small wrapper
│   │   ├── vector_store.py     # Chroma setup/query
│   │   ├── bm25_search.py      # keyword search
│   │   └── reranker.py         # bge-reranker wrapper
│   ├── graph/
│   │   ├── state.py            # LangGraph state schema
│   │   ├── nodes.py            # retrieve/generate/verify nodes
│   │   └── build_graph.py      # wires nodes into the graph
│   ├── llm/
│   │   └── groq_client.py      # Groq API wrapper
│   └── eval/
│       └── langsmith_eval.py   # test set + evaluators
├── data/
│   └── (uploaded PDFs, chroma persistence dir — gitignored)
└── README.md                   # project story, architecture diagram, results
```

## 6. Build Order (matches architecture, bottom-up)

1. Ingestion + chunking (`src/ingestion/`)
2. Embeddings + vector store (`src/retrieval/embeddings.py`, `vector_store.py`)
3. Basic retrieve → generate chain (no LangGraph yet — just prove it works)
4. Wrap it in LangGraph with a single "generate" node
5. Add the "verify" node + conditional edge (the hard part)
6. Add BM25 + reranker into the retrieval step
7. LangSmith tracing + eval dataset
8. Streamlit UI
9. Deploy to HuggingFace Spaces

## 7. Environment & Tools We'll Set Up Next
- Python virtual environment
- VS Code + Python extension + a few QoL extensions
- `.env` for API keys (Groq, LangSmith)
- Git repo initialized from day 1 (commit as you go — good practice + shows progress history on GitHub)
