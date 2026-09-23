"""
STAGE 1b — Chunking (COMPLETE)

Splits extracted page text into coherent chunks, tagged with metadata
(page_number, section) so every chunk can be traced back to exactly
where it came from in the document.
"""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()

SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\s+)?"
    r"(Abstract|Introduction|Related Work|Background|"
    r"Method(?:ology)?|Experiments?|Results?|Discussion|"
    r"Conclusion|References|Acknowledge?ments?|Appendix|"
    # --- added for textbooks/notes ---
    r"Chapter\s*\d*|Unit\s*\d*|Lesson\s*\d*|Module\s*\d*|"
    r"Overview|Summary|Key\s*Terms?|Key\s*Concepts?|Definitions?|"
    r"Examples?|Exercises?|Review\s*Questions?|Practice\s*Problems?|"
    r"Learning\s*Objectives?|Notes?|Recap)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)



def _token_len(text: str) -> int:
    """
    Approximate token count without an external tokenizer download.
    ~4 characters per token is a standard rule of thumb for English.
    """
    return max(1, len(text) // 4)


GENERIC_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\d*\s+)?"      # optional numbering like "1.", "2.3"
    r"([A-Z][A-Za-z0-9\s\-:]{2,50})\s*$"  # short, capitalized standalone line
)


def _detect_section(text: str) -> str | None:
    stripped = text.strip()
    match = SECTION_HEADER_PATTERN.match(stripped)
    if match:
        return match.group(1).title()

    # Fallback: a short, standalone, capitalized line is very likely a
    # heading in notes/textbooks even if it's not on our known-word list
    # (e.g. "Photosynthesis", "The French Revolution", "Newton's Laws").
    generic_match = GENERIC_HEADING_PATTERN.match(stripped)
    if generic_match and len(stripped.split()) <= 8 and not stripped.endswith((".", ",")):
        return generic_match.group(1).strip()

    return None


def _split_into_sections(pages: list[dict]) -> list[dict]:
    blocks = []
    current_section = "Unlabeled"
    current_text_lines = []
    current_page = pages[0]["page_number"] if pages else None

    def flush():
        if current_text_lines:
            blocks.append({
                "section": current_section,
                "text": "\n".join(current_text_lines).strip(),
                "page_number": current_page,
            })

    for page in pages:
        for line in page["text"].split("\n"):
            detected = _detect_section(line)
            if detected:
                flush()
                current_section = detected
                current_text_lines = []
                current_page = page["page_number"]
            else:
                if not current_text_lines:
                    current_page = page["page_number"]
                current_text_lines.append(line)
    flush()

    return [b for b in blocks if b["text"]]

MIN_CHUNK_CHARS = 40  # skip chunks too short to carry meaningful content
                       # (e.g. stray page numbers, running headers caught in isolation)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Returns:
        [{"text": "...", "metadata": {"page_number": N, "section": "..."}}, ...]

    Chunks shorter than MIN_CHUNK_CHARS are dropped — these are almost
    always extraction artifacts (a lone page number, a stray header
    fragment) rather than real content, and they pollute retrieval by
    sometimes ranking highly on pure coincidence despite carrying no
    useful information.
    """
    MIN_CHUNK_CHARS = cfg["chunking"].get("min_chunk_chars", 40)
    
    chunk_size = cfg["chunking"]["chunk_size"]
    chunk_overlap = cfg["chunking"]["chunk_overlap"]
    strategy = cfg["chunking"]["strategy"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    skipped = 0

    if strategy == "section_aware":
        blocks = _split_into_sections(pages)
        logger.info("Detected %d section blocks", len(blocks))
        for block in blocks:
            for sub_text in splitter.split_text(block["text"]):
                if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                    skipped += 1
                    continue
                chunks.append({
                    "text": sub_text,
                    "metadata": {
                        "page_number": block["page_number"],
                        "section": block["section"],
                    },
                })
    else:
        for page in pages:
            for sub_text in splitter.split_text(page["text"]):
                if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                    skipped += 1
                    continue
                chunks.append({
                    "text": sub_text,
                    "metadata": {
                        "page_number": page["page_number"],
                        "section": "Unlabeled",
                    },
                })

    if skipped:
        logger.info("Skipped %d chunks shorter than %d characters (likely extraction artifacts)",
                    skipped, MIN_CHUNK_CHARS)

    logger.info("Produced %d chunks (strategy=%s, chunk_size=%d tokens)",
                len(chunks), strategy, chunk_size)
    return chunks

if __name__ == "__main__":
    import sys
    from src.ingestion.pdf_parser import extract_pages

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.chunker <path_to_pdf>")
        sys.exit(1)

    pages = extract_pages(sys.argv[1])
    chunks = chunk_pages(pages)

    print(f"\nTotal chunks: {len(chunks)}\n")
    for c in chunks[:5]:
        print(f"--- Page {c['metadata']['page_number']} | Section: {c['metadata']['section']} ---")
        print(c["text"][:200])
        print()