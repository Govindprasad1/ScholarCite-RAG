""""
Stage 8 — Streamlit UI

Wraps the full pipeline (ingest -> hybrid retrieve -> generate -> verify)
in an interactive web app: upload a PDF, ask questions, see cited,
verified answers with the source chunks shown.
"""
import uuid
import streamlit as st

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.graph.build_graph import run as run_pipeline
from src.utils.config import load_config

config = load_config()

st.set_page_config(page_title=config["app"]["title"], page_icon="📄", layout="wide")

# --- Session state setup ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title(f"📄 {config['app']['title']}")
st.caption("A Citation-Grounded RAG Assistant for Academic Documents")

# --- Sidebar: upload ---
with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Processing document..."):
                temp_path = f"/tmp/{uuid.uuid4().hex}_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if st.session_state.doc_id:
                    delete_document(st.session_state.doc_id)

                pages = extract_pages(temp_path)
                chunks = chunk_pages(pages)
                new_doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                add_chunks(chunks, doc_id=new_doc_id)

                st.session_state.doc_id = new_doc_id
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []

            st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")
        else:
            st.info(f"Currently loaded: {uploaded_file.name}")

    if st.session_state.doc_id:
        if st.button("Clear document"):
            delete_document(st.session_state.doc_id)
            st.session_state.doc_id = None
            st.session_state.doc_name = None
            st.session_state.chat_history = []
            st.rerun()

# --- Main area: chat ---
if not st.session_state.doc_id:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            status_label = {"verified": "✅ Verified", "partially_supported": "⚠️ Partially supported", "no_context": "ℹ️ No relevant context"}
            st.caption(status_label.get(entry["status"], entry["status"]))
            if entry.get("chunks"):
                with st.expander("View source chunks used"):
                    for c in entry["chunks"]:
                        meta = c["metadata"]
                        st.markdown(f"**Page {meta['page_number']}, Section: {meta['section']}**")
                        st.text(c["text"][:300] + "...")

    question = st.chat_input("Ask a question about the document")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving, generating, and verifying..."):
                result = run_pipeline(question, doc_id=st.session_state.doc_id)
            st.write(result["final_answer"])
            status_label = {"verified": "✅ Verified", "partially_supported": "⚠️ Partially supported", "no_context": "ℹ️ No relevant context"}
            st.caption(status_label.get(result["status"], result["status"]))

            if result.get("retrieved_chunks"):
                with st.expander("View source chunks used"):
                    for c in result["retrieved_chunks"]:
                        meta = c["metadata"]
                        st.markdown(f"**Page {meta['page_number']}, Section: {meta['section']}**")
                        st.text(c["text"][:300] + "...")

        st.session_state.chat_history.append({
            "question": question,
            "answer": result["final_answer"],
            "status": result["status"],
            "chunks": result.get("retrieved_chunks", []),
        })