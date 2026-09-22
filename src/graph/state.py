"""
Stage 4 — Shared Graph State

This TypedDict is the single object that flows through every node in
the LangGraph pipeline. Each node reads whatever fields it needs and
returns a dict of the fields it wants to update — LangGraph merges
that into the running state automatically.
"""
from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict):
    # --- inputs (set once, at graph.invoke() time) ---
    question: str
    doc_id: str
    top_k: int
    max_attempts: int

    # --- populated by retrieve_node ---
    retrieved_chunks: List[Dict[str, Any]]

    # --- populated by generate_node ---
    answer: Optional[str]
    attempts: int

    # --- populated by verify_node ---
    verification: Optional[Dict[str, Any]]  # {"claims": [...], "all_supported": bool}

    # --- populated by decide_node (what the caller actually reads) ---
    final_answer: Optional[str]
    status: Optional[str]  # "verified" | "partially_supported" | "no_context"