"""
STAGE 1a — PDF Parsing (COMPLETE)

Extracts text from a PDF page-by-page, keeping the page number attached
to every piece of text — this is what makes citations like "[Page 7]"
possible later in the pipeline.
"""
import fitz  # PyMuPDF
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        [{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}, ...]
    """
    logger.info("Opening PDF: %s", pdf_path)
    doc = fitz.open(pdf_path)

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page_number": i + 1, "text": text})

    doc.close()
    logger.info("Extracted %d pages from %s", len(pages), pdf_path)

    empty_pages = [p["page_number"] for p in pages if not p["text"].strip()]
    if empty_pages:
        logger.warning(
            "Pages with no extractable text (possibly scanned images): %s",
            empty_pages,
        )

    return pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pdf_parser <path_to_pdf>")
        sys.exit(1)

    result = extract_pages(sys.argv[1])
    print(f"\nTotal pages: {len(result)}\n")
    print("--- Preview of first page ---")
    print(result[0]["text"][:500])