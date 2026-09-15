"""
STAGE 4a — LangGraph State Schema

Goal: define the shared "state" object that flows through every node
in the graph (retrieve -> generate -> verify -> decide).

This is the NEW mental model piece (see architecture doc) — take time
to actually understand this before coding it: state is just a typed
dict that each node reads from and writes back to.
"""
from typing import TypedDict, List, Dict, Optional


class GraphState(TypedDict):
    question: str
    retrieved_chunks: List[Dict]      # [{"text": ..., "metadata": {...}}, ...]
    answer: Optional[str]
    verification_result: Optional[Dict]   # {"all_supported": bool, "unsupported_claims": [...]}
    attempts: int                     # how many generate/verify loops have run
    final_answer: Optional[str]       # what actually gets returned to the user
