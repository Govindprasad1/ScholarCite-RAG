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


def _initial_state(question: str, doc_id: str, top_k: int, max_attempts: int) -> GraphState:
    return {
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


def run(question: str, doc_id: str, top_k: int = 5, max_attempts: int = 2) -> dict:
    """
    Convenience entrypoint — builds a fresh state, invokes the compiled
    graph, and returns the final result. Use this when you don't need
    intermediate progress (e.g. in scripts, tests, evaluation).
    """
    app = build_app()
    initial_state = _initial_state(question, doc_id, top_k, max_attempts)
    return app.invoke(initial_state)


def stream_run(question: str, doc_id: str, top_k: int = 5, max_attempts: int = 2):
    """
    Generator version of run() — yields (node_name, accumulated_state)
    after EACH node finishes, instead of only returning the final
    result. This is what powers the live pipeline visualization: the
    UI can react to retrieval completing, then generation, then
    verification, as they genuinely happen — not a simulated delay.

    Usage:
        for node_name, state in stream_run(question, doc_id):
            ... update UI based on node_name and state ...
        final_state = state  # last yielded state is the final one
    """
    app = build_app()
    state = _initial_state(question, doc_id, top_k, max_attempts)

    for event in app.stream(state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_output is None:
                # LangGraph can occasionally emit a None update during
                # certain internal transitions (e.g. retry loops) —
                # dict.update(None) would crash with "'NoneType' object
                # is not iterable", so skip it rather than updating.
                continue
            state.update(node_output)
            yield node_name, dict(state)