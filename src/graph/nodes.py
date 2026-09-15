"""
STAGE 4b — LangGraph Nodes

Goal: implement each step of the pipeline as a node function that takes
GraphState in and returns a (partial) GraphState update.

Nodes to build:

    retrieve_node(state: GraphState) -> dict
        - calls hybrid retrieval (vector_store + bm25 + reranker)
        - sets state["retrieved_chunks"]

    generate_node(state: GraphState) -> dict
        - prompts the LLM with retrieved_chunks (with page/section metadata)
        - instructs it to cite [Page X, Section Y] per claim, and say
          "I cannot answer from the given context" if unsupported
        - sets state["answer"]

    verify_node(state: GraphState) -> dict
        - separate LLM call: checks each claim in state["answer"] against
          state["retrieved_chunks"]
        - sets state["verification_result"] = {"all_supported": bool, ...}

    decide_node(state: GraphState) -> dict
        - if verification passed -> state["final_answer"] = state["answer"]
        - if failed and attempts < max_regenerate_attempts -> loop back to generate
        - if failed and out of attempts -> flag unsupported parts to the user
"""
from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_node(state: GraphState) -> dict:
    raise NotImplementedError("Stage 4b: implement retrieve_node")


def generate_node(state: GraphState) -> dict:
    raise NotImplementedError("Stage 4b: implement generate_node (with citation prompting)")


def verify_node(state: GraphState) -> dict:
    raise NotImplementedError("Stage 4b: implement verify_node (reject-if-unsupported check)")


def decide_node(state: GraphState) -> dict:
    raise NotImplementedError("Stage 4b: implement decide_node (loop or finalize)")
