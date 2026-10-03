"""
Stage 8 — Streamlit UI (with live animated pipeline visualization)

Upload a PDF, ask questions, and watch the actual retrieve -> generate
-> verify -> decide pipeline animate in real time as it executes,
before showing the final citation-grounded, self-verified answer.
"""
import math
import time
import uuid
import tempfile
import streamlit as st

from src.ingestion.pdf_parser import extract_pages
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, delete_document
from src.graph.build_graph import stream_run
from src.utils.config import load_config

config = load_config()
st.set_page_config(page_title=config["app"]["title"], page_icon="📄", layout="wide")

# ========================= Styling =========================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at 10% 0%, #1a1f3a 0%, #0a0e1a 45%, #0a0e1a 100%); }
    .block-container { max-width: 900px; padding-top: 1.5rem; }

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

    [data-testid="stChatInput"] textarea { border-radius: 14px !important; border: 1px solid rgba(255,255,255,0.12) !important; background: rgba(255,255,255,0.03) !important; }
    section[data-testid="stSidebar"] { background: rgba(10,14,26,0.6); border-right: 1px solid rgba(255,255,255,0.06); }
    .stButton button {
        border-radius: 10px !important; border: 1px solid rgba(129,140,248,0.4) !important;
        background: rgba(129,140,248,0.1) !important; color: #C4B5FD !important; font-weight: 600 !important;
    }
    .stButton button:hover { background: rgba(129,140,248,0.2) !important; border-color: #818CF8 !important; }
    [data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.08) !important; background: rgba(255,255,255,0.02) !important; }

    /* IMPORTANT: only hide the specific branding elements below —
       never hide the whole <header>, since Streamlit's sidebar
       collapse/expand arrow lives inside it. Hiding the full header
       would make a collapsed sidebar impossible to reopen. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ---- Pipeline flow diagram ---- */
    .pipeline-flow { display: flex; align-items: flex-start; margin: 10px 0 24px 0; }
    .pipeline-stage { display: flex; flex-direction: column; align-items: center; gap: 6px; width: 80px; }
    .pipeline-icon {
        width: 44px; height: 44px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-size: 19px;
        background: rgba(148,163,184,0.08); border: 2px solid rgba(148,163,184,0.25);
        transition: all .3s ease;
    }
    .pipeline-stage.active .pipeline-icon {
        border-color: #818CF8; background: rgba(129,140,248,0.18);
        animation: pulse-ring 1.1s ease-in-out infinite;
    }
    .pipeline-stage.done .pipeline-icon { border-color: #34D399; background: rgba(52,211,153,0.15); }
    .pipeline-label { font-size: 0.68rem; color: #64748B; font-weight: 600; text-align: center; }
    .pipeline-stage.active .pipeline-label { color: #C4B5FD; }
    .pipeline-stage.done .pipeline-label { color: #34D399; }
    .pipeline-connector { flex: 1; height: 2px; background: rgba(148,163,184,0.15); margin-top: 22px; transition: background .3s ease; }
    .pipeline-connector.done { background: #34D399; }
    @keyframes pulse-ring { 0%,100% { box-shadow: 0 0 8px rgba(129,140,248,0.35); } 50% { box-shadow: 0 0 20px rgba(129,140,248,0.85); } }

    /* ---- Live chunk reveal cards ---- */
    .chunk-reveal-card {
        background: rgba(129,140,248,0.05); border: 1px solid rgba(129,140,248,0.15);
        border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
        animation: slideInFade 0.4s ease forwards; opacity: 0;
    }
    .chunk-reveal-meta { font-size: 0.78rem; color: #C4B5FD; font-weight: 700; margin-bottom: 5px; }
    .bar-track { height: 5px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg,#818CF8,#C084FC); border-radius: 3px; }
    @keyframes slideInFade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    /* ---- Live claim verification reveal ---- */
    .claim-row {
        display: flex; align-items: flex-start; gap: 8px; padding: 6px 0;
        font-size: 0.85rem; color: #CBD5E1;
        animation: slideInFade 0.35s ease forwards; opacity: 0;
    }

    /* ---- Source chunk cards (final, in expander) ---- */
    .chunk-card { background: rgba(129,140,248,0.06); border-left: 3px solid #818CF8; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
    .chunk-meta { color: #C4B5FD; font-weight: 700; font-size: 0.8rem; margin-bottom: 4px; }
    .chunk-text { color: #94A3B8; line-height: 1.5; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

STATUS_STYLES = {
    "verified": ("✅", "Verified", "#34D399", "rgba(52,211,153,0.12)"),
    "partially_supported": ("⚠️", "Partially Supported", "#FBBF24", "rgba(251,191,36,0.12)"),
    "no_context": ("ℹ️", "No Relevant Context", "#94A3B8", "rgba(148,163,184,0.12)"),
}

PIPELINE_STAGES = [
    ("retrieve", "🔍", "Retrieve"),
    ("generate", "✍️", "Generate"),
    ("verify", "🛡️", "Verify"),
    ("decide", "🏁", "Finalize"),
]

# Characters that legitimately end a complete answer — includes both
# ASCII "]" (our standard citation format "[Page X, Section: Y]") and
# the full-width "】" in case the model ever uses that style instead.
VALID_ENDING_CHARS = (".", "!", "?", "」", ")", "|", "%", "】", "]")


# ========================= Helper renderers =========================

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
    if text.strip() and not text.rstrip().endswith(VALID_ENDING_CHARS):
        st.caption("⚠️ This answer may have been cut off before completing.")


def _score_to_percent(chunk: dict) -> int:
    """Normalize whatever score field a chunk has into a 0-100 bar width."""
    if "rerank_score" in chunk:
        raw = chunk["rerank_score"]
        pct = 1 / (1 + math.exp(-raw))
    elif "score" in chunk:
        pct = chunk["score"]
    else:
        pct = 0.5
    return max(3, min(100, int(pct * 100)))


def render_pipeline_html(active_node: str, completed_nodes: set) -> str:
    parts = ['<div class="pipeline-flow">']
    for i, (key, icon, label) in enumerate(PIPELINE_STAGES):
        css_class = "done" if key in completed_nodes else ("active" if key == active_node else "")
        parts.append(f'''
            <div class="pipeline-stage {css_class}">
                <div class="pipeline-icon">{icon}</div>
                <div class="pipeline-label">{label}</div>
            </div>
        ''')
        if i < len(PIPELINE_STAGES) - 1:
            connector_done = "done" if key in completed_nodes else ""
            parts.append(f'<div class="pipeline-connector {connector_done}"></div>')
    parts.append("</div>")
    return "".join(parts)


def render_chunk_reveal_html(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        pct = _score_to_percent(c)
        delay = i * 0.08
        parts.append(f'''
            <div class="chunk-reveal-card" style="animation-delay:{delay}s">
                <div class="chunk-reveal-meta">📍 Page {meta['page_number']} · {meta['section']}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
            </div>
        ''')
    return "".join(parts)


def render_claims_reveal_html(claims: list) -> str:
    parts = []
    for i, claim in enumerate(claims):
        icon = "✅" if claim.get("supported") else "❌"
        delay = i * 0.12
        parts.append(f'''
            <div class="claim-row" style="animation-delay:{delay}s">
                <span>{icon}</span><span>{claim.get("claim", "")}</span>
            </div>
        ''')
    return "".join(parts)


def run_with_live_visualization(question: str, doc_id: str):
    pipeline_slot = st.empty()
    chunks_slot = st.empty()
    claims_slot = st.empty()

    completed = set()
    final_state = None

    pipeline_slot.markdown(render_pipeline_html("retrieve", completed), unsafe_allow_html=True)

    for node_name, state in stream_run(question, doc_id=doc_id):
        final_state = state

        if node_name == "retrieve":
            completed.add("retrieve")
            pipeline_slot.markdown(render_pipeline_html("generate", completed), unsafe_allow_html=True)
            chunks_slot.markdown(render_chunk_reveal_html(state.get("retrieved_chunks", [])), unsafe_allow_html=True)

        elif node_name == "generate":
            completed.add("generate")
            pipeline_slot.markdown(render_pipeline_html("verify", completed), unsafe_allow_html=True)

        elif node_name == "verify":
            completed.add("verify")
            pipeline_slot.markdown(render_pipeline_html("decide", completed), unsafe_allow_html=True)
            verification = state.get("verification") or {}
            claims_slot.markdown(render_claims_reveal_html(verification.get("claims", [])), unsafe_allow_html=True)

        elif node_name == "decide":
            completed.add("decide")
            pipeline_slot.markdown(render_pipeline_html("", completed), unsafe_allow_html=True)

        elif node_name == "no_context":
            completed.update({"retrieve"})
            pipeline_slot.markdown(render_pipeline_html("", completed), unsafe_allow_html=True)

    time.sleep(0.4)
    pipeline_slot.empty()
    chunks_slot.empty()
    claims_slot.empty()

    return final_state


# ========================= Session state =========================
for key, default in [
    ("doc_id", None),
    ("doc_name", None),
    ("chat_history", []),
    ("uploader_key", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ========================= Header =========================
st.markdown(f"""
<div class="scholarcite-header">
    <div class="scholarcite-logo">📄</div>
    <div>
        <p class="scholarcite-title">{config['app']['title']}</p>
        <p class="scholarcite-subtitle">Citation-grounded answers from your academic documents</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ========================= Sidebar: upload =========================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <h3 style="margin-bottom: 4px;">📤 Document</h3>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 0;">Upload a PDF to begin asking questions</p>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic key: changing this forces Streamlit to treat the uploader
    # as a brand-new widget with no file, which is what actually clears
    # it on "Clear document" — without this, the widget silently keeps
    # holding the old file across reruns and re-indexes it immediately.
    uploaded_file = st.file_uploader(
        "", type="pdf", label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

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
            st.session_state.uploader_key += 1  # forces a fresh, empty uploader widget
            st.rerun()

# ========================= Main: chat =========================
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
            final_state = None
            try:
                final_state = run_with_live_visualization(question, st.session_state.doc_id)
            except Exception as e:
                render_error(e)

            if final_state is not None:
                render_answer_text(final_state["final_answer"])
                render_status_badge(final_state["status"])
                render_chunks(final_state.get("retrieved_chunks", []))

        if final_state is not None:
            st.session_state.chat_history.append({
                "question": question,
                "answer": final_state["final_answer"],
                "status": final_state["status"],
                "chunks": final_state.get("retrieved_chunks", []),
            })