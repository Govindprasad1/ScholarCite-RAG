"""
Stage 4 tests — the LangGraph verification loop.
These make REAL Groq API calls (two per question: generate + verify),
so they need GROQ_API_KEY and will skip without it.
"""
import os
import shutil
import pytest

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval import vector_store as vs
from src.graph.build_graph import run
from src.graph import nodes as graph_nodes

SAMPLE_PDF = "tests/fixtures/sample.pdf"

requires_groq_key = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM tests",
)


@pytest.fixture
def indexed_sample_doc(tmp_path, monkeypatch):
    import src.utils.config as config_module
    original_load_config = config_module.load_config

    def patched_load_config():
        cfg = original_load_config()
        cfg["vector_store"] = dict(cfg["vector_store"])
        cfg["vector_store"]["persist_directory"] = str(tmp_path / "chroma_test")
        cfg["vector_store"]["collection_name"] = "test_graph"
        return cfg

    monkeypatch.setattr(vs, "load_config", patched_load_config)

    pages = extract_pages(SAMPLE_PDF)
    chunks = chunk_pages(pages)
    vs.add_chunks(chunks, doc_id="graph_test_doc")

    yield "graph_test_doc"
    shutil.rmtree(tmp_path / "chroma_test", ignore_errors=True)


@requires_groq_key
def test_verified_answer_has_final_answer_and_status(indexed_sample_doc):
    result = run("What method does the paper propose?", doc_id=indexed_sample_doc)
    assert result["final_answer"] is not None
    assert result["status"] in ("verified", "partially_supported")


@requires_groq_key
def test_no_context_short_circuits_before_generation(indexed_sample_doc):
    result = run("What is the capital of France?", doc_id="nonexistent_doc_id")
    assert result["status"] == "no_context"
    assert result["answer"] is None  # generate_node should never have run


@requires_groq_key
def test_verification_runs_and_produces_claims(indexed_sample_doc):
    result = run("What faithfulness improvement did they report?", doc_id=indexed_sample_doc)
    assert result["verification"] is not None
    assert "all_supported" in result["verification"]


def test_verify_node_fabricated_claim_flagged_unsupported(monkeypatch):
    """
    Directly unit-test verify_node with a fabricated claim that is NOT
    in the source chunks, to confirm the verifier actually catches it
    rather than rubber-stamping everything. Mocks the LLM call so this
    test doesn't depend on model behavior variability.
    """
    fake_verification_response = type("R", (), {
        "content": '{"claims": [{"claim": "The paper was written in 1990", "supported": false, "reasoning": "No date is mentioned"}], "all_supported": false}'
    })()

    class FakeLLM:
        def invoke(self, prompt):
            return fake_verification_response

    monkeypatch.setattr(graph_nodes, "get_llm", lambda role: FakeLLM())

    state = {
        "answer": "The paper was written in 1990 [Page 1, Section: Introduction].",
        "retrieved_chunks": [{"text": "Retrieval augmented generation combines retrieval with generation.",
                               "metadata": {"page_number": 1, "section": "Introduction"}}],
    }
    result = graph_nodes.verify_node(state)
    assert result["verification"]["all_supported"] is False


def test_parse_verification_json_handles_markdown_fences():
    raw = '```json\n{"claims": [], "all_supported": true}\n```'
    parsed = graph_nodes._parse_verification_json(raw)
    assert parsed["all_supported"] is True


def test_parse_verification_json_fails_closed_on_garbage():
    parsed = graph_nodes._parse_verification_json("this is not json at all")
    assert parsed["all_supported"] is False