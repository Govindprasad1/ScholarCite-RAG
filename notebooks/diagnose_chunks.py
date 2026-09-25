"""
Diagnostic script — prints every chunk's section label and a short
preview, for inspecting chunking quality on a real PDF.

Run with: uv run python notebooks/diagnose_chunks.py data/uploads/AIAYN.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys
from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages

if len(sys.argv) < 2:
    print("Usage: uv run python notebooks/diagnose_chunks.py <path_to_pdf>")
    sys.exit(1)

pages = extract_pages(sys.argv[1])
chunks = chunk_pages(pages)

print(f"Total chunks: {len(chunks)}\n")
for i, c in enumerate(chunks):
    section = c["metadata"]["section"]
    page = c["metadata"]["page_number"]
    preview = c["text"][:50].replace("\n", " ")
    print(f"{i:2d}. page={page:2d} | section={section!r:35s} | {preview}...")