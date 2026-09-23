"""
Stage 4 — Graph Nodes

Four nodes make up the pipeline:
  retrieve_node -> generate_node -> verify_node -> decide_node

decide_node either finalizes the answer, or (via the conditional edge
wired in build_graph.py) sends the flow back to generate_node for
another attempt, up to max_attempts.
"""
import json

from src.graph.state import GraphState
from src.llm.groq_client import get_llm
from src.llm.prompts import format_context, GENERATION_PROMPT_TEMPLATE, VERIFICATION_PROMPT_TEMPLATE
from src.retrieval.vector_store import query_similar
from src.utils.logger import get_logger

logger = get_logger(__name__)

REFUSAL_PHRASE = "I cannot answer this from the given context."

from src.retrieval.hybrid import hybrid_query

def retrieve_node(state: GraphState) -> dict:
    """Fetch relevant chunks using hybrid (vector + BM25 + rerank) retrieval."""
    chunks = hybrid_query(state["question"], doc_id=state["doc_id"])
    logger.info(f"Retrieved {len(chunks)} chunks for question: {state['question']}")
    return {"retrieved_chunks": chunks}

def generate_node(state: GraphState) -> dict:
    attempts = state.get("attempts", 0) + 1

    # On retry attempts, widen retrieval instead of reusing the same chunks —
    # a failed verification often means the first retrieval missed the
    # actual supporting context, not just that generation hallucinated.
    if attempts > 1:
        from src.retrieval.hybrid import hybrid_query
        logger.info(f"Retry attempt {attempts}: re-retrieving with wider net")
        wider_chunks = hybrid_query(state["question"], doc_id=state["doc_id"])
        chunks_to_use = wider_chunks if wider_chunks else state["retrieved_chunks"]
    else:
        chunks_to_use = state["retrieved_chunks"]

    context = format_context(chunks_to_use)
    prompt = GENERATION_PROMPT_TEMPLATE.format(context=context, question=state["question"])

    llm = get_llm(role="generation")
    response = llm.invoke(prompt)

    logger.info(f"Generation attempt {attempts}: {response.content[:100]}...")
    return {"answer": response.content, "attempts": attempts, "retrieved_chunks": chunks_to_use}


def _parse_verification_json(raw_text: str) -> dict:
    """
    Parse the verifier's JSON response defensively — LLMs occasionally
    wrap JSON in markdown fences despite instructions not to. If
    parsing fails entirely, we treat it as a failed verification
    (fail-safe: assume unsupported rather than assume supported).
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if "all_supported" not in parsed or "claims" not in parsed:
            raise ValueError("Missing expected keys in verification JSON")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse verification JSON ({e}); treating as unsupported. Raw: {raw_text[:200]}")
        return {"claims": [], "all_supported": False, "parse_error": True}


def verify_node(state: GraphState) -> dict:
    """
    Fact-check the generated answer against the SAME retrieved chunks
    used to generate it. Uses a separate (cheaper/faster) model — this
    is the core of the "self-verifying" design.
    """
    answer = state["answer"]

    # A correct refusal never needs fact-checking — nothing to verify.
    if answer.strip() == REFUSAL_PHRASE:
        return {"verification": {"claims": [], "all_supported": True}}

    context = format_context(state["retrieved_chunks"])
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(context=context, answer=answer)

    llm = get_llm(role="verification")
    response = llm.invoke(prompt)
    result = _parse_verification_json(response.content)

    logger.info(f"Verification result: all_supported={result.get('all_supported')}, "
                f"claims_checked={len(result.get('claims', []))}")
    return {"verification": result}


def decide_node(state: GraphState) -> dict:
    """
    Decide the outcome based on verification result and attempts used.
    This node itself does NOT loop — it just sets state. The actual
    routing decision (retry vs. finalize) happens in build_graph.py's
    conditional edge, which reads the fields this node sets.
    """
    verification = state["verification"]

    if verification["all_supported"]:
        return {"final_answer": state["answer"], "status": "verified"}

    if state["attempts"] < state["max_attempts"]:
        # Not finalizing yet — the conditional edge will route back to generate
        logger.info(f"Verification failed on attempt {state['attempts']}, retrying...")
        return {}

    # Out of attempts — finalize, but clearly flag what wasn't supported
    unsupported = [c["claim"] for c in verification["claims"] if not c["supported"]]
    flagged_answer = state["answer"]
    if unsupported:
        flagged_note = "\n\n⚠️ Note: the following claims could not be verified against the source document:\n" + \
                        "\n".join(f"- {c}" for c in unsupported)
        flagged_answer += flagged_note

    return {"final_answer": flagged_answer, "status": "partially_supported"}


def no_context_node(state: GraphState) -> dict:
    """Reached when retrieval finds nothing at all — skip generation entirely."""
    return {"final_answer": REFUSAL_PHRASE, "status": "no_context"}