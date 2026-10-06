# ScholarCite RAG — Complete Project Summary

**A Citation-Grounded, Self-Verifying RAG Assistant for Academic Documents**

This document is a full, stage-by-stage record of the project: what was
built, which files exist and what each one does, and every significant
issue debugged along the way — intended as both a personal reference
and material for interview walkthroughs.

---

## Overview

ScholarCite RAG lets a user upload any academic PDF (research paper,
textbook chapter, notes) and ask questions about it. Every answer is:
- **Grounded** — generated only from retrieved excerpts of the actual document
- **Cited** — tagged with the exact page and section it came from
- **Self-verified** — independently fact-checked against the source by a second LLM pass before being shown to the user

Built entirely on a free, open-source/free-tier stack: PyMuPDF, local
embeddings, ChromaDB, BM25, a local cross-encoder reranker, Groq's free
LLM API, LangGraph, LangSmith, Ragas, MLflow, and Streamlit — deployed
on AWS EC2.

---

## Stage 1 — Ingestion (PDF Parsing + Chunking)

**Goal:** turn a raw PDF into clean, labeled text chunks that can be
traced back to an exact page and section.

**Files created:**
- `src/ingestion/pdf_parser.py` — `extract_pages()`: opens a PDF with PyMuPDF, returns a list of `{page_number, text}` dicts, one per page.
- `src/ingestion/chunker.py` — `chunk_pages()`: the most-iterated file in the whole project. Detects section headers via a layered strategy — a hardcoded pattern for common academic/textbook headers, a subsection pattern (`5.3 Optimizer`), and a **generalizable, frequency-based fallback** that flags any short, capitalized, standalone line as a heading candidate unless it's statistically too common across the document (a signal it's a recurring table/figure label, not a real heading). Splits text into token-budgeted chunks with overlap, drops near-empty extraction artifacts, and merges a section whose last line ends in `:` with whatever follows (so a sentence introducing a list isn't severed from that list).
- `src/utils/config.py`, `src/utils/logger.py` — shared config loader and structured logger used by every later stage.
- `tests/fixtures/generate_sample_pdf.py`, `generate_notes_pdf.py` — generate small synthetic PDFs (a fake paper, fake textbook notes) used as deterministic test fixtures.
- `tests/test_ingestion.py` — 8+ tests covering page extraction, section detection, metadata correctness, and generalization to textbook-style headings.

**Key issues resolved:**
- A Windows **Anaconda vs. standalone Python 3.13** DLL conflict (`c10.dll` init failure) blocked `torch` entirely — root-caused via systematic elimination (VC++ Redistributable, CUDA vs CPU build, folder location, antivirus, `uv` link mode) down to Anaconda's bundled MKL/OpenMP libraries conflicting with PyTorch's own; fixed by rebuilding the venv on a clean, standalone Python 3.13 install.
- `tiktoken`'s token counter required a network download on first use — replaced with a dependency-free `len(text)//4` approximation.
- Iteratively tightened the generic heading-detection fallback against real-world false positives (table headers like "BLEU", "Model", line-wrap artifacts like "Law", "American") using a document-size-aware frequency heuristic (absolute count **and** ratio) that generalizes across both short notes and long papers without hardcoding a word list.

---

## Stage 2 — Embeddings + Vector Store

**Goal:** make chunks searchable by meaning, not just keywords.

**Files created:**
- `src/retrieval/embeddings.py` — `embed_texts()`/`embed_query()`: wraps `sentence-transformers`' `BAAI/bge-small-en-v1.5`, cached via `lru_cache` so the model loads once per process.
- `src/retrieval/vector_store.py` — `add_chunks()`, `query_similar()`, `delete_document()`, `get_all_chunks_for_doc()`: thin wrapper around a persistent ChromaDB collection; every chunk is tagged with a `doc_id` so searches/deletes can be scoped to one uploaded document.
- `tests/test_retrieval.py` (started here, extended in later stages).

**Key decision:** ChromaDB runs embedded, session-scoped per uploaded document — deliberately not a shared, permanently-persisted corpus, since the app's use case is "upload your own paper," not a fixed library.

**Key issue resolved:** a documentation-comment artifact (`# ... existing code stays the same ...`) got pasted literally into `add_chunks()`, silently deleting its real body — chunks were never actually written to the vector store, causing every downstream retrieval call to return empty. Found via a minimal, fixture-free diagnostic script and fixed by restoring the full embed → ID/metadata construction → `collection.add()` logic.

---

## Stage 3 — Basic Retrieve → Generate Chain

**Goal:** prove the simplest end-to-end path works before adding orchestration complexity.

**Files created:**
- `src/llm/groq_client.py` — `get_llm(role)`: wraps `ChatGroq`, supporting separate **generation** and **verification** model roles (set independently in config), each cached separately.
- `src/llm/prompts.py` — `format_context()` and `GENERATION_PROMPT_TEMPLATE`: shared between the straight-line chain and the later LangGraph nodes, so there's only one definition of "how to ask for a cited answer."
- `src/llm/generate.py` — `answer_question()`: retrieve → format → prompt → `llm.invoke()` → return answer + source chunks.
- `tests/test_generation.py`.

**Key issue resolved:** Groq deprecated `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` mid-project (a real platform change, not a bug) — migrated to `openai/gpt-oss-120b` (generation) and `openai/gpt-oss-20b` (verification), which turned out to be a genuine upgrade (MoE architecture, larger context window, Apache 2.0). Added `scripts/check_groq_models.py` as a standing safeguard against future silent deprecations.

---

## Stage 4 — LangGraph Self-Verification Loop

**Goal:** stop trusting the LLM's first answer — independently fact-check it before showing it to the user. This is the project's core differentiator.

**Files created:**
- `src/graph/state.py` — `GraphState`: the shared state dict threaded through every node (question, retrieved_chunks, answer, attempts, verification, final_answer, status).
- `src/graph/nodes.py` — `retrieve_node`, `generate_node`, `verify_node`, `decide_node`, `no_context_node`. `verify_node` prompts a *separate* LLM to extract every factual claim from the generated answer and check each one against the retrieved chunks, returning structured JSON (parsed defensively — fails closed to "unsupported" if parsing breaks). `decide_node` finalizes on success, retries generation (with a fresh, wider retrieval pass) on failure, or flags unsupported claims after exhausting retries.
- `src/graph/build_graph.py` — `build_app()` wires the nodes into a LangGraph `StateGraph` with conditional edges; `run()` (final-result only) and `stream_run()` (yields progress after every node, used later for Stage 8's live visualization) are both exposed.
- `tests/test_graph.py` — includes a mocked-LLM unit test that proves `verify_node` genuinely flags a fabricated claim, independent of any specific model's behavior on a given day.

**Key issue resolved:** none structural — this stage worked cleanly on first real build, validated immediately by a smoke test showing correct refusal behavior and correct per-claim ✅/❌ verification output.

---

## Stage 5 — Hybrid Search + Reranking

**Goal:** catch exact-term/number matches that pure semantic search misses, and get a more precise final ranking.

**Files created:**
- `src/retrieval/bm25_search.py` — `bm25_query()`: BM25 keyword scoring over a document's full chunk set, cached per `doc_id` (invalidated on add/delete), with punctuation-aware tokenization and an empty-corpus guard (prevents a `ZeroDivisionError` when querying a document with zero chunks).
- `src/retrieval/reranker.py` — `rerank()`: cross-encoder reranking, blending the reranker's score with the original hybrid (RRF) rank so one model's occasional scoring quirk can't completely override strong agreement from both underlying retrieval methods.
- `src/retrieval/hybrid.py` — `hybrid_query()`: merges vector + BM25 results via Reciprocal Rank Fusion, then reranks the merged pool; the single entrypoint Stage 4's `retrieve_node` calls.

**Key issues resolved (this was the most heavily debugged stage):**
- A reranker score "double-sigmoid" bug (manually applying sigmoid to an already-sigmoid-activated score) flattened all relevance scores toward ~0.5 — diagnosed via a raw-logit check, fixed by removing the redundant activation.
- The initial reranker model (`bge-reranker-base`) gave a real, high-relevance chunk (containing the Adam optimizer's exact hyperparameters) a low score relative to topically-similar-but-wrong chunks — a known weakness of general-purpose rerankers on formula/symbol-dense text. Root-cause confirmed via a dedicated diagnostic comparing vector/BM25/rerank rankings side by side. Fixed by switching to `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0, purpose-built for passage relevance) — which resolved it with a large, confident score margin — plus layering the RRF-blended scoring as a defense-in-depth safety net.
- Verified, via real adversarial testing on "Attention Is All You Need," that hybrid+rerank correctly (a) excludes a semantically-similar-but-wrong chunk that vector-only ranked in its top 3, (b) gives sharper, more confident scoring on exact-fact lookups, and (c) drives scores to ~0.000 on a genuinely unrelated question, versus vector-only's misleadingly confident ~0.45.

---

## Stage 6 — Config-Driven Pipeline

**Goal:** audit for hardcoded "magic numbers" and move them into `config.yaml`.

No new files — an audit pass across `chunker.py` (heading-detection thresholds) and `reranker.py` (RRF/rerank blend weights), confirming every tunable value lives in one place for easy experimentation in Stage 7.

---

## Stage 7 — Evaluation (Ragas + MLflow + LangSmith)

**Goal:** replace manual spot-checking with quantified, repeatable metrics.

**Files created:**
- `tests/fixtures/eval_dataset.json` — 10 hand-verified Q&A pairs against "Attention Is All You Need," including two deliberately unanswerable questions to test refusal behavior.
- `src/eval/ragas_eval.py` — `collect_samples()` runs the real pipeline (not a mock) for each test question, builds Ragas `SingleTurnSample`s from the actual retrieved context and answer; `run_evaluation()` scores them with Faithfulness, Response Relevancy, Context Precision, and Context Recall, using the project's own Groq LLM as judge and local embeddings for relevancy — with low concurrency and generous retries/timeouts to survive Groq's free-tier limits.
- `src/eval/mlflow_tracking.py` — `log_run()`: logs the current config (chunk size, models, hybrid/reranker toggles) alongside the resulting scores to a local MLflow SQLite backend, filtering out any metric that failed to compute (rather than crashing on `None`).
- `notebooks/run_stage7_eval.py`, `check_mlflow_runs.py`, `diagnose_relevancy.py` — orchestration and diagnostic scripts.

**Key issues resolved:**
- `ragas`'s internal LLM factory imported a `langchain_community` submodule that had been reorganized away in a newer release — fixed by pinning `langchain-community<0.4`.
- Ragas' `ResponseRelevancy` metric generates multiple reverse-engineered questions per answer by default (`strictness=3`), which calls the LLM with a multi-completion `n` parameter Groq's API doesn't support — fixed with `strictness=1`.
- MLflow's file-based tracking backend was deprecated mid-project — switched to a SQLite backend (`sqlite:///mlflow.db`).
- Discovered that Ragas scores a correctly-issued refusal ("I cannot answer this...") as `0` across all four metrics, since there are no factual claims to check — this understated the aggregate score on the two deliberately-unanswerable test questions. Documented explicitly rather than treated as a pipeline failure: on the 8 genuinely answerable questions, scores were materially higher (faithfulness ~0.88, relevancy ~0.92, precision ~0.98) than the blended 10-question aggregate.
- A genuine Faithfulness hallucination was caught by Ragas during testing (`dmodel = 102` invented in a list of real table values) — Stage 4's own verification loop had already flagged it "partially supported," and the finding led to a stricter no-fabrication rule added to the generation prompt.
- Ran a real before/after comparison (vector-only vs. hybrid+larger-chunks) via MLflow, showing every metric improved with hybrid enabled.

---

## Stage 8 — Streamlit UI (with Live Pipeline Visualization)

**Goal:** a usable, polished interface — plus an animated, real-time view of the pipeline actually executing.

**Files created/replaced:**
- `app.py` — full Streamlit app: PDF upload (session-scoped indexing, properly clearable via a dynamic uploader key), a chat interface, graceful error handling for rate limits/timeouts (never a raw traceback), a custom dark/glassmorphism visual theme, and **`run_with_live_visualization()`** — drives `stream_run()` from Stage 4 and progressively renders a flow diagram (Retrieve → Generate → Verify → Finalize) with real relevance-score bars for retrieved chunks and a live ✅/❌ checklist of each claim as verification genuinely completes, not a simulated delay.
- `.streamlit/config.toml` — custom theme colors.

**Key issues resolved:**
- A `NoneType` crash in the streaming generator, caused by LangGraph occasionally emitting a `None` node update during a retry loop — fixed by skipping `None` updates instead of calling `dict.update(None)`.
- A false "this answer may have been cut off" warning on answers correctly ending in a citation bracket `]` — the truncation-check's allowed-endings list only included a full-width bracket character; added the standard ASCII `]`.
- "Clear document" silently re-indexed the same file immediately, because Streamlit's file-uploader widget retains its file across reruns unless its `key` changes — fixed with a dynamic, incrementing uploader key.
- Hiding Streamlit's default header (to remove branding) also hid the sidebar's own collapse/reopen arrow, which lives inside that header — fixed by hiding only the specific branding elements, not the whole header.
- A genuinely blank-looking answer in the UI turned out to be a **truncated markdown table** (hit `max_tokens`), which Streamlit's markdown renderer silently fails to render when malformed — fixed with a higher `max_tokens` (3072), a prompt rule discouraging markdown tables in favor of plain prose, and a UI-level fallback warning for any answer that still looks incomplete.

---

## Stage 9 — Docker, CI, and Deployment

**Goal:** reproducible packaging and a real, public live deployment.

**Files created/updated:**
- `Dockerfile` — multi-stage `uv sync` (dependencies first with `--no-install-project` for Docker layer caching, then the full project once source code is copied in) on `python:3.13-slim`.
- `docker-compose.yml`, `.dockerignore`.
- `.github/workflows/ci.yml` — runs the full pytest suite on every push via `uv`, plus a second job that confirms the Docker image still builds.

**Key issues resolved:**
- `ci.yml` initially targeted `requirements.txt`/pip/Python 3.11 from the original scaffold, long after the project migrated to `uv`/Python 3.13 — updated to match.
- A YAML syntax error ("a sequence was not expected") from a stray character at the top of the workflow file — fixed by recreating the file cleanly.
- The first Docker build failed with "Expected a Python module at src/..." — `uv sync` was running before `src/` existed in the build context (it only gets installed on `COPY . .` afterward); fixed by splitting the sync into a dependency-only pass (`--no-install-project`) before the code copy, and a full sync after.
- The resulting image was unexpectedly ~3.5GB because `torch`'s default Linux wheel bundles the full CUDA/cuDNN/Triton stack even for a CPU-only app — fixed by pinning an explicit CPU-only PyTorch index per platform in `pyproject.toml` and regenerating the lockfile.
- Investigated HuggingFace Spaces for deployment, but discovered (via live verification) a July 2026 policy change requiring a paid PRO subscription to host **both** Docker and Gradio Spaces on free compute — only Static Spaces remain free. Considered switching the UI to Gradio, but confirmed this wouldn't help, since the constraint is the hosting platform's policy, not the UI framework. Also evaluated Streamlit Community Cloud, but its 1GB guaranteed RAM limit is a real risk for this app's embedding+reranker memory footprint.
- **Final decision:** deployed to **AWS EC2** (Amazon Linux 2023, `t3.small`, 2 vCPU / 2GB RAM, 20GB gp3 EBS), with a 2GB swap file for memory headroom, Docker installed directly on the instance, the image built locally and transferred via `docker save`/`scp`/`docker load`, and run with `--restart unless-stopped` for resilience — chosen deliberately based on the app's actual memory profile rather than accepting a platform's default/convenient option.

---

## Technology Stack Summary

| Layer | Technology | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Reliable text + page-level extraction |
| Chunking | Custom (regex + frequency heuristics) | Generalizes across document types without a fixed vocabulary |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | Free, fast, strong retrieval benchmark performance |
| Vector store | ChromaDB (embedded) | Simple, free, sufficient at this scale |
| Keyword search | `rank_bm25` | Catches exact-term matches embeddings miss |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache 2.0, purpose-built for passage relevance, proven via real adversarial testing |
| Orchestration | LangGraph | State machine for retrieve/generate/verify/retry |
| LLM | Groq (`openai/gpt-oss-120b` / `-20b`) | Free tier, fast inference, MoE architecture |
| Evaluation | Ragas + MLflow + LangSmith | RAG-specific metrics, experiment tracking, tracing |
| UI | Streamlit | Fast to build, now with a custom animated pipeline visualization |
| Packaging | Docker + `uv` | Reproducible builds |
| CI | GitHub Actions | Test + Docker build verification on every push |
| Deployment | AWS EC2 | Chosen deliberately based on the app's real memory requirements |

---

## Known Limitations (stated explicitly, not hidden)

- Section detection can occasionally misattribute content that falls between a detected header and interstitial text (e.g. author-contribution footnotes).
- Inline subsection headings embedded mid-paragraph (rather than on their own line) in dense, tightly-typeset PDFs are not always separated from surrounding text.
- A sentence ending in ":" that introduces a list can, in rare cases involving an intervening page-footer number, still be split from that list's content — the system correctly refuses in this case rather than answering incompletely.
- Ragas' automated metrics score correctly-issued refusals as 0 across the board, understating aggregate scores when a test set includes deliberately unanswerable questions.
- No OCR support — assumes text-based (not scanned) PDFs.
- Single-document scope by design — no cross-document synthesis.
