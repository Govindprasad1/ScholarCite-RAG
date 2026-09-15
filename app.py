"""
STAGE 8 — Streamlit UI (build this LAST, after the pipeline works
end-to-end in a plain script/notebook first — see architecture doc).

Planned UI:
    - st.file_uploader for PDF upload
    - Ingest + index the PDF (cache with st.cache_resource so it only
      runs once per uploaded file, not on every question)
    - st.text_input / st.chat_input for the question
    - Display the answer with inline citations
    - Show a small "sources" panel listing the retrieved chunks used
    - Optional: an "admin" tab showing latency + faithfulness stats
      from MLflow (see MLOps additions)
"""
import streamlit as st
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()

st.set_page_config(page_title=cfg["app"]["title"], page_icon="📄")
st.title(cfg["app"]["title"])
st.caption("A Citation-Grounded RAG Assistant for Academic Documents")

st.info(
    "🚧 This is a scaffold. Build the pipeline stage-by-stage "
    "(see README.md and the architecture doc) before wiring up this UI."
)

# TODO Stage 8:
# uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
# if uploaded_file:
#     ... ingest, chunk, embed, index (cache this!) ...
#     question = st.text_input("Ask a question about the paper")
#     if question:
#         ... run the LangGraph app, display answer + citations ...
