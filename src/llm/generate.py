"""
Stage 3 — Basic Generation
Wires Stage 2's retrieval together with Stage 3's LLM client into a
single straight-line function: question in, cited answer out. No
verification loop yet — that's Stage 4. This is the "happy path"
baseline to prove the whole chain works before adding complexity.
"""
from typing import Dict, Any

from src.llm.groq_client import get_llm
from src.retrieval.vector_store import query_similar
from src.utils.logger import get_logger

from src.llm.prompts import format_context, GENERATION_PROMPT_TEMPLATE


logger = get_logger(__name__)


def _format_context(chunks: list[dict]) -> str:
    """
    Turn retrieved chunks into a labeled block the LLM can cite from.
    Each chunk is clearly tagged with its page/section so the model
    has no excuse not to cite correctly.
    """
    blocks = []
    for chunk in chunks:
        meta = chunk["metadata"]
        label = f"[Page {meta['page_number']}, Section: {meta['section']}]"
        blocks.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


PROMPT_TEMPLATE = """You are a research assistant answering questions strictly based on the provided document excerpts.

RULES:
- Only use information from the excerpts below. Do not use outside knowledge.
- After every claim, cite its source like this: [Page X, Section: Y]
- If the excerpts do not contain enough information to answer, say exactly: "I cannot answer this from the given context."
- Do not guess or fill in gaps with assumptions.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER (with citations):"""


def answer_question(question: str, doc_id: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full Stage 3 pipeline: retrieve relevant chunks, prompt the LLM,
    return the answer plus which chunks were used (for transparency/debugging).
    """

# ... inside answer_question(), replace:
#   context = _format_context(chunks)
#   prompt = PROMPT_TEMPLATE.format(...)
# with:
    

    chunks = query_similar(question, top_k=top_k, doc_id=doc_id)

    if not chunks:
        return {
            "answer": "I cannot answer this from the given context.",
            "source_chunks": [],
        }

    context = format_context(chunks)
    prompt = GENERATION_PROMPT_TEMPLATE.format(context=context, question=question)

        
    llm = get_llm(role="generation")
    logger.info(f"Sending question to LLM: {question}")
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "source_chunks": chunks,
    }