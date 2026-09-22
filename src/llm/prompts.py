"""
Shared prompt construction, used by both Stage 3's straight-line
answer_question() and Stage 4's LangGraph nodes. Keeping this in one
place means the generation prompt can never silently drift between
the two — there's only one definition of "how we ask the LLM to
answer with citations."
"""

def format_context(chunks: list[dict]) -> str:
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


GENERATION_PROMPT_TEMPLATE = """You are a research assistant answering questions strictly based on the provided document excerpts.

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


VERIFICATION_PROMPT_TEMPLATE = """You are a strict fact-checker. Your job is to verify whether an answer's claims are actually supported by the given source excerpts — nothing else.

SOURCE EXCERPTS:
{context}

ANSWER TO VERIFY:
{answer}

INSTRUCTIONS:
1. Break the answer down into its individual factual claims.
2. For each claim, decide if it is directly supported by the source excerpts above.
   - "supported": the excerpts state this or something that clearly implies it.
   - "unsupported": the excerpts do not contain this information, or it contradicts them.
3. A citation tag like [Page X, Section: Y] being present does NOT automatically make a claim supported — check the actual content, not just whether a citation was included.
4. Respond with ONLY valid JSON, no markdown formatting, no explanation outside the JSON, in exactly this shape:

{{
  "claims": [
    {{"claim": "short restatement of the claim", "supported": true, "reasoning": "one sentence why"}},
    {{"claim": "...", "supported": false, "reasoning": "..."}}
  ],
  "all_supported": true
}}

"all_supported" must be true only if EVERY claim is supported. If the answer is exactly "I cannot answer this from the given context.", treat it as trivially fully supported (no claims to check) and return an empty claims list with all_supported: true.
"""