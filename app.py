"""
Stage 8 — Streamlit UI (final, consolidated)

Upload a PDF, ask questions, get citation-grounded, self-verified
answers with source excerpts shown. Includes graceful error handling
and a polished, non-default visual style.
"""
import uuid
import tempfile
import streamlit as st

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.graph.build_graph import run as run_pipeline
from src.utils.config import load_config

config = load_config()
st.set_page_config(page_title=config["app"]["title"], page_icon="📄", layout="wide")

# ---------- Styling ----------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at 10% 0%, #1a1f3a 0%, #0a0e1a 45%, #0a0e1a 100%); }
    .block-container { max-width: 880px; padding-top: 1.5rem; }
    .scholarcite-header { display: flex; align-items: center; gap: 16px; padding: 20px 0 8px 0; }
    .scholarcite-logo {
        width: 48px; height: 48px; border-radius: 14px;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; box-shadow: 0 8px 24px rgba(129,140,248,0.35);
    }
    .scholarcite-title { font-size: 1.6rem; font-weight: 800; color: #F1F5F9; margin: 0; letter-spacing: -0.02em; }
    .scholarcite-subtitle { font-size: 0.9rem; color: #94A3B8; margin: 0; }
    .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
    .chunk-card { background: rgba(129,140,248,0.06); border-left: 3px solid #818CF8; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
    .chunk-meta { color: #C4B5FD; font-weight: 700; font-size: 0.8rem; margin-bottom: 4px; }
    .chunk-text { color: #94A3B8; line-height: 1.5; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
    [data-testid="stChatInput"] textarea { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.12) !important; background: rgba(255,255,255,0.03) !important; }
    section[data-testid="stSidebar"] { background: rgba(10,14,26,0.6); border-right: 1px solid rgba(255,255,255,0.06); }
    .stButton button {
        border-radius: 10px !important; border: 1px solid rgba(129,140,248,0.4) !important;
        background: rgba(129,140,248,0.1) !important; color: #C4B5FD !important; font-weight: 600 !important;
    }
    .stButton button:hover { background: rgba(129,140,248,0.2) !important; border-color: #818CF8 !important; }
    [data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.08) !important; background: rgba(255,255,255,0.02) !important; }
    @keyframes pulse-glow { 0%, 100% { opacity: 0.6; } 50% { opacity: 1; } }
    .thinking-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #818CF8; animation: pulse-glow 1.2s ease-in-out infinite; margin-right: 6px; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

STATUS_STYLES = {
    "verified": ("✅", "Verified", "#34D399", "rgba(52,211,153,0.12)"),
    "partially_supported": ("⚠️", "Partially Supported", "#FBBF24", "rgba(251,191,36,0.12)"),
    "no_context": ("ℹ️", "No Relevant Context", "#94A3B8", "rgba(148,163,184,0.12)"),
}


def render_status_badge(status: str):
    icon, label, color, bg = STATUS_STYLES.get(status, ("•", status, "#94A3B8", "rgba(148,163,184,0.12)"))
    st.markdown(f'<span class="status-badge" style="color:{color}; background:{bg};">{icon} {label}</span>',
                unsafe_allow_html=True)


def render_chunks(chunks: list):
    if not chunks:
        return
    with st.expander(f"📚 View {len(chunks)} source excerpt(s)"):
        for c in chunks:
            meta = c["metadata"]
            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-meta">📍 Page {meta['page_number']} · {meta['section']}</div>
                <div class="chunk-text">{c['text'][:280]}...</div>
            </div>
            """, unsafe_allow_html=True)


def render_error(e: Exception):
    error_message = str(e)
    if "rate_limit" in error_message.lower() or "429" in error_message:
        st.error("⚠️ **Rate limit reached.** This app uses Groq's free tier, which has daily usage limits. Please try again later.")
    elif "timeout" in error_message.lower():
        st.error("⚠️ **Request timed out.** Please try asking your question again.")
    else:
        st.error("⚠️ **Something went wrong.** Please try again or rephrase your question.")
    with st.expander("Technical details (for debugging)"):
        st.code(error_message)


def render_answer_text(answer_text: str):
    text = answer_text if answer_text else "(No answer text was generated.)"
    st.write(text)
    if text.strip() and not text.rstrip().endswith((".", "!", "?", "」", ")", "|", "%", "】")):
        st.caption("⚠️ This answer may have been cut off before completing.")


# ---------- Session state ----------
for key, default in [("doc_id", None), ("doc_name", None), ("chat_history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Header ----------
st.markdown(f"""
<div class="scholarcite-header">
    <div class="scholarcite-logo">📄</div>
    <div>
        <p class="scholarcite-title">{config['app']['title']}</p>
        <p class="scholarcite-subtitle">Citation-grounded answers from your academic documents</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar: upload ----------
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <h3 style="margin-bottom: 4px;">📤 Document</h3>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 0;">Upload a PDF to begin asking questions</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None and st.session_state.doc_name != uploaded_file.name:
        with st.spinner("Processing document..."):
            try:
                temp_dir = tempfile.gettempdir()
                temp_path = f"{temp_dir}/{uuid.uuid4().hex}_{uploaded_file.name}"
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
            except Exception as e:
                st.error("⚠️ Failed to process this document. Please try a different PDF.")
                with st.expander("Technical details"):
                    st.code(str(e))
                st.stop()

    if st.session_state.doc_id:
        if st.button("Clear document"):
            delete_document(st.session_state.doc_id)
            st.session_state.doc_id = None
            st.session_state.doc_name = None
            st.session_state.chat_history = []
            st.rerun()

# ---------- Main: chat ----------
if not st.session_state.doc_id:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            render_answer_text(entry["answer"])
            render_status_badge(entry["status"])
            render_chunks(entry.get("chunks", []))

    question = st.chat_input("Ask a question about the document")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            result = None
            placeholder = st.empty()
            placeholder.markdown('<div><span class="thinking-dot"></span>Retrieving, generating, and verifying...</div>', unsafe_allow_html=True)
            try:
                result = run_pipeline(question, doc_id=st.session_state.doc_id)
            except Exception as e:
                placeholder.empty()
                render_error(e)
            else:
                placeholder.empty()

            if result is not None:
                render_answer_text(result["final_answer"])
                render_status_badge(result["status"])
                render_chunks(result.get("retrieved_chunks", []))

        if result is not None:
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["final_answer"],
                "status": result["status"],
                "chunks": result.get("retrieved_chunks", []),
            })