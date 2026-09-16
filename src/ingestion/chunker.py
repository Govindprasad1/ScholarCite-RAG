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
    r"Conclusion|References|Acknowledge?ments?|Appendix)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _token_len(text: str) -> int:
    """
    Approximate token count without an external tokenizer download.
    ~4 characters per token is a standard rule of thumb for English.
    """
    return max(1, len(text) // 4)


def _detect_section(text: str) -> str | None:
    match = SECTION_HEADER_PATTERN.match(text.strip())
    return match.group(1).title() if match else None


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


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Returns:
        [{"text": "...", "metadata": {"page_number": N, "section": "..."}}, ...]
    """
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

    if strategy == "section_aware":
        blocks = _split_into_sections(pages)
        logger.info("Detected %d section blocks", len(blocks))
        for block in blocks:
            for sub_text in splitter.split_text(block["text"]):
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
                chunks.append({
                    "text": sub_text,
                    "metadata": {
                        "page_number": page["page_number"],
                        "section": "Unlabeled",
                    },
                })

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