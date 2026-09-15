"""
STAGE 4c — Wiring the Graph

Goal: connect the nodes from nodes.py into an actual LangGraph
StateGraph, with a conditional edge after verify_node that routes
back to generate_node (retry) or forward to END, based on
verification_result.

Key function to build:

    build_app() -> CompiledGraph
        Compile once (e.g. at app startup), then call
        app.invoke({"question": ..., "attempts": 0}) per user query.
"""
from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.graph import nodes
from src.utils.logger import get_logger

logger = get_logger(__name__)


def route_after_verify(state: GraphState) -> str:
    """
    TODO: return "generate" (retry) if verification failed and attempts
    remain, else return "end".
    """
    raise NotImplementedError("Stage 4c: implement routing logic")


def build_app():
    """
    TODO (build this together):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("verify", nodes.verify_node)
    graph.add_node("decide", nodes.decide_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"generate": "generate", "end": "decide"})
    graph.add_edge("decide", END)

    return graph.compile()
    """
    raise NotImplementedError("Stage 4c: wire up the StateGraph")
