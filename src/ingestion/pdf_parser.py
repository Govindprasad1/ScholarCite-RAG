"""
STAGE 1a — PDF Parsing

Goal: extract text from a PDF page-by-page, keeping the page number
attached to every piece of text (needed for citations later).

We'll implement this together using PyMuPDF (fitz). Key function to build:

    extract_pages(pdf_path: str) -> list[dict]
        Returns: [{"page_number": 1, "text": "..."}, {"page_number": 2, ...}, ...]

Things to watch for (see architecture doc):
- Two-column research papers can read left-to-right across columns and
  scramble sentence order — test on a real 2-column paper early.
- Keep raw text per page separate; don't concatenate the whole PDF into
  one string, or you lose the ability to cite page numbers.
"""
import fitz  # PyMuPDF
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_pages(pdf_path: str) -> list[dict]:
    """
    TODO (build this together):
    1. Open the PDF with fitz.open(pdf_path)
    2. Loop through pages, call page.get_text()
    3. Return a list of {"page_number": i+1, "text": page_text}
    """
    raise NotImplementedError("Stage 1a: implement PDF page extraction")
