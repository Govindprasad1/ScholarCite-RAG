"""
Stage 4 — Wiring the Graph

Flow:
    START -> retrieve -> [conditional: chunks found?]
                              |-- no  --> no_context -> END
                              |-- yes --> generate -> verify -> decide
                                                                  |
                                          [conditional: verified or out of attempts?]
                                                  |-- retry --> generate (loop)
                                                  |-- done  --> END
"""
from langgraph.graph import StateGraph, START, END

from src.graph.state import GraphState
from src.graph import nodes


def _route_after_retrieve(state: GraphState) -> str:
    return "generate" if state["retrieved_chunks"] else "no_context"


def _route_after_decide(state: GraphState) -> str:
    # If decide_node didn't set final_answer, it means: retry.
    if state.get("final_answer") is not None:
        return "end"
    return "retry"


def build_app():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("verify", nodes.verify_node)
    graph.add_node("decide", nodes.decide_node)
    graph.add_node("no_context", nodes.no_context_node)

    graph.add_edge(START, "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "no_context": "no_context"},
    )

    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "decide")

    graph.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"retry": "generate", "end": END},
    )

    graph.add_edge("no_context", END)

    return graph.compile()


def run(question: str, doc_id: str, top_k: int = 5, max_attempts: int = 2) -> dict:
    """
    Convenience entrypoint — builds a fresh state, invokes the compiled
    graph, and returns the final result.
    """
    app = build_app()
    initial_state: GraphState = {
        "question": question,
        "doc_id": doc_id,
        "top_k": top_k,
        "max_attempts": max_attempts,
        "retrieved_chunks": [],
        "answer": None,
        "attempts": 0,
        "verification": None,
        "final_answer": None,
        "status": None,
    }
    result = app.invoke(initial_state)
    return result