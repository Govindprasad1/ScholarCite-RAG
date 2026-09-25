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
from collections import Counter
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
    r"^\s*(?:\d+\.?\d*\.?\d*\s+)?"
    r"([A-Z][A-Za-z0-9\s\-:]{2,50})\s*$"
)
SUBSECTION_HEADER_PATTERN = re.compile(
    r"^\s*(\d+\.\d+)\s+([A-Z][A-Za-z\s]{2,40})\s*$",
    re.MULTILINE,
)

# Common words we don't require to be capitalized in a "headline case" check
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "and", "or", "for", "to", "with",
    "is", "are", "our", "its", "their", "his", "her", "by", "as", "at",
    "from", "this", "that", "we", "be",
}


def _looks_like_headline_case(text: str) -> bool:
    """Every non-stopword should start with an uppercase letter — real
    headings are written this way; a wrapped mid-sentence line usually
    isn't (e.g. 'the projections are parameter matrices' fails this)."""
    words = text.split()
    for w in words:
        clean = w.strip(":,")
        if clean.lower() in _STOPWORDS:
            continue
        if not clean or not clean[0].isupper():
            return False
    return True



def _build_word_frequencies(pages: list[dict]) -> Counter:
    """
    Count how often each lowercase word appears across the WHOLE
    document body. Used to detect table/figure labels generically:
    a genuine one-off heading is usually a distinctive phrase, while
    table/figure labels (e.g. 'Model', 'Layer', a repeated metric name)
    tend to be common words that also appear constantly as ordinary
    text throughout the document. This generalizes across any PDF,
    since it's based on frequency behavior, not a fixed word list.
    """
    freq = Counter()
    for page in pages:
        words = re.findall(r"[a-zA-Z\-]+", page["text"].lower())
        freq.update(words)
    return freq

def _is_plausible_heading(
    line: str,
    prev_line: str,
    prev_was_heading: bool,
    word_freq: Counter,
    total_words: int,
) -> bool:
    """
    Guards the generic heading fallback against false positives using
    signals that generalize across any document:

    1. All-caps short tokens (abbreviations) rejected.
    2. Purely numeric tokens (table data) rejected.
    3. Must look like proper headline case.
    4. For SHORT (1-2 word) candidates: reject if the word is BOTH
       frequent in absolute terms AND frequent relative to the
       document's size. Two conditions are required (not just ratio)
       because ratio alone breaks down on short documents — a word
       appearing 3 times in a 40-word document is 7.5% of the
       vocabulary but is still clearly a legitimate topic word, not a
       recurring table label. Requiring a minimum absolute occurrence
       count too prevents small documents from being over-filtered,
       while the ratio still catches genuinely repetitive vocabulary
       in longer documents (e.g. 'Model' appearing 40+ times in a
       15-page paper).
    """
    words = line.split()
    if not words:
        return False

    first_token = words[0].strip(":,")
    if first_token.isupper() and len(first_token) >= 2 and len(words) <= 3:
        return False
    if any(w.strip(":,").isdigit() for w in words):
        return False
    if not _looks_like_headline_case(line):
        return False

    if len(words) <= 2:
        MIN_ABSOLUTE_OCCURRENCES = 5   # word must appear at least this many times...
        FREQ_RATIO_THRESHOLD = 0.0015  # ...AND exceed this ratio of total vocabulary

        for w in words:
            clean = w.strip(":,").lower()
            if not clean:
                continue
            occurrences = word_freq.get(clean, 0)
            ratio = occurrences / total_words if total_words > 0 else 0
            if occurrences >= MIN_ABSOLUTE_OCCURRENCES and ratio > FREQ_RATIO_THRESHOLD:
                return False

        prev_stripped = prev_line.strip()
        prev_ends_sentence = prev_stripped == "" or prev_stripped[-1] in ".!?:;"
        return prev_ends_sentence or prev_was_heading

    return True

def _detect_section(
    line: str,
    prev_line: str,
    prev_was_heading: bool,
    word_freq: Counter,
    total_words: int,
) -> str | None:
    stripped = line.strip()

    match = SECTION_HEADER_PATTERN.match(stripped)
    if match:
        return match.group(1).title()

    sub_match = SUBSECTION_HEADER_PATTERN.match(stripped)
    if sub_match:
        return f"{sub_match.group(1)} {sub_match.group(2).strip()}"

    generic_match = GENERIC_HEADING_PATTERN.match(stripped)
    if (generic_match
            and len(stripped.split()) <= 8
            and not stripped.endswith((".", ","))
            and _is_plausible_heading(stripped, prev_line, prev_was_heading, word_freq, total_words)):
        return generic_match.group(1).strip()

    return None


def _split_into_sections(pages: list[dict]) -> list[dict]:
    word_freq = _build_word_frequencies(pages)
    total_words = sum(word_freq.values())

    blocks = []
    current_section = "Unlabeled"
    current_text_lines = []
    current_page = pages[0]["page_number"] if pages else None

    prev_line = ""
    prev_was_heading = False

    def flush():
        if current_text_lines:
            blocks.append({
                "section": current_section,
                "text": "\n".join(current_text_lines).strip(),
                "page_number": current_page,
            })

    for page in pages:
        for line in page["text"].split("\n"):
            detected = _detect_section(line, prev_line, prev_was_heading, word_freq, total_words)
            if detected:
                flush()
                current_section = detected
                current_text_lines = []
                current_page = page["page_number"]
                prev_was_heading = True
            else:
                if not current_text_lines:
                    current_page = page["page_number"]
                current_text_lines.append(line)
                if line.strip():
                    prev_was_heading = False
            if line.strip():
                prev_line = line
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
""" KNOWN LimitaTION:Subsection headings that appear inline with surrounding paragraph
  text (common in dense, tightly-typeset 2-column PDFs) are not
  separated from their preceding content — text extraction flattens
  layout, and only headings appearing on their own line are detected.
  A more complete fix would use font-size/style-based layout analysis
  rather than line-pattern matching."""