"""
Tests for Stage 4 (LangGraph pipeline: generate + verify + citations).

Run with: pytest tests/test_graph.py -v

These are the tests that make your project's hardest, most differentiating
feature (reject-if-unsupported verification) provably correct — worth
spending real time here.
"""
import pytest


def test_generate_node_includes_citations():
    """Generated answers should contain [Page X] style citations."""
    pytest.skip("TODO: implement once nodes.generate_node() is built")


def test_verify_node_flags_unsupported_claim():
    """
    Feed a fabricated claim not present in the source chunks and confirm
    verify_node correctly marks it as unsupported.
    """
    pytest.skip("TODO: implement once nodes.verify_node() is built")


def test_verify_node_passes_supported_claim():
    """A claim directly quoted from a source chunk should pass verification."""
    pytest.skip("TODO: implement once nodes.verify_node() is built")


def test_graph_terminates_after_max_attempts():
    """Ensure the retry loop doesn't run forever if verification keeps failing."""
    pytest.skip("TODO: implement once build_graph.py routing is built")
