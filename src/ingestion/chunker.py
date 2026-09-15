"""
STAGE 1b — Chunking

Goal: split extracted page text into coherent chunks, with metadata
(page_number, section_title if detectable) attached to each chunk.

Key function to build:

    chunk_pages(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]
        Input: output of pdf_parser.extract_pages()
        Returns: [{"text": "...", "metadata": {"page_number": 1, "section": "Introduction"}}, ...]

Things to watch for (see architecture doc):
- Don't split purely on character count — try to respect paragraph/section
  boundaries first (section_aware strategy in config.yaml).
- Use config.yaml's chunking.chunk_size / chunk_overlap instead of
  hardcoding numbers here.
"""
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    TODO (build this together):
    1. Use LangChain's RecursiveCharacterTextSplitter (or a section-aware
       splitter) with cfg["chunking"]["chunk_size"] / ["chunk_overlap"].
    2. For each page's text, split into chunks.
    3. Attach {"page_number": ..., "section": ...} metadata to each chunk.
    4. Return the flat list of chunks across all pages.
    """
    raise NotImplementedError("Stage 1b: implement section-aware chunking")
