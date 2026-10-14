"""

app/streamlit_app.py

---------------------

Thin UI layer. Wires adapters -> pipelines -> orchestrators.

No business logic lives here.

"""

from __future__ import annotations

import os
# Configure extended timeout for Hugging Face model weight downloads
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import tempfile

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from lecture_memory.config.settings import settings

from lecture_memory.observability.logger import get_logger

from lecture_memory.domain.enums import VerificationOutcome

from lecture_memory.adapters.opencv_frame_selector import (

    OpenCVFrameSelector

)

# Qwen adapters

from lecture_memory.adapters.qwen_runtime import QwenRuntime

from lecture_memory.adapters.qwen_vision import QwenVisionAdapter

from lecture_memory.adapters.qwen_generator import QwenGeneratorAdapter

from lecture_memory.adapters.simple_event_former import (

    SimpleVisualEventFormer

)

from lecture_memory.adapters.simple_aligner import SimpleTemporalAligner

from lecture_memory.adapters.simple_verifier import SimpleEvidenceVerifier

from lecture_memory.adapters.sentence_embedder import SentenceEmbedder

from lecture_memory.adapters.pinecone_vector_store import PineconeVectorStore
from lecture_memory.adapters.chroma_vector_store import ChromaVectorStore
from lecture_memory.adapters.link_downloader import VideoLinkDownloader, DownloadedVideoInfo
from lecture_memory.adapters.temporal_retriever import TemporalRetriever

from lecture_memory.pipeline.ingestion import VideoIngestionPipeline

from lecture_memory.pipeline.audio import AudioProcessingPipeline

from lecture_memory.pipeline.visual import VisualProcessingPipeline

from lecture_memory.pipeline.alignment import CrossModalAlignmentPipeline

from lecture_memory.pipeline.indexing import MemoryIndexingPipeline

from lecture_memory.pipeline.qa import QuestionAnsweringPipeline

from lecture_memory.orchestrator.ingest_orchestrator import (

    IngestOrchestrator

)

from lecture_memory.orchestrator.qa_orchestrator import QAOrchestrator

from lecture_memory.orchestrator.job_state import JobStateManager

from lecture_memory.storage.media_store import MediaStore

logger = get_logger(__name__)

# ==============================================================

# Page configuration

# ==============================================================

st.set_page_config(

    page_title="CS Lecture Memory",

    layout="wide",

)

st.title("🎓 Multimodal CS Lecture Memory")

st.caption(

    "Upload a lecture video → ask questions → "

    "get grounded answers with speech, visual and temporal context."

)

# ==============================================================

# Helpers

# ==============================================================

def _get_whisper_transcriber():

    try:

        from lecture_memory.adapters.whisper_transcriber import (

            WhisperTranscriber

        )

        return WhisperTranscriber()

    except ImportError:

        return None

def _format_time(seconds: float | None) -> str:

    if seconds is None:

        return "—"

    seconds = max(0.0, seconds)

    minutes = int(seconds // 60)

    remaining = int(seconds % 60)

    return f"{minutes:02d}:{remaining:02d}"

def _build_concept_lookup(

    vector_store,

    video_id: str | None,

):

    """

    Build a lookup from internal concept IDs to human-readable

    lecture concepts.

    UUIDs remain internal. The UI uses this lookup only to display

    meaningful lecture evidence.

    """

    if not video_id:

        return {}

    try:

        concepts = vector_store.get_by_video(

            video_id

        )

        return {

            concept.concept_id: concept

            for concept in concepts

        }

    except Exception as exc:

        logger.warning(

            f"ui_concept_lookup_failed | error={exc}"

        )

        return {}

# ----------------------------------------------------------
# CSS: YouTube-style answer panel
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    .ts-chip {
        display: inline-block;
        background: #1e3a5f;
        color: #7ec8f7;
        border: 1px solid #2a5080;
        border-radius: 6px;
        padding: 1px 8px;
        font-size: 0.78rem;
        font-family: monospace;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
        white-space: nowrap;
    }
    .concept-bullet {
        margin: 6px 0;
        padding: 8px 14px;
        background: #111827;
        border-left: 3px solid #3b82f6;
        border-radius: 0 6px 6px 0;
        line-height: 1.55;
    }
    .concept-bullet b { color: #e2e8f0; }
    .ev-card {
        background: #0f1929;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
    }
    .ev-header { color: #94a3b8; font-size: 0.78rem; margin-bottom: 4px; }
    .ev-body   { color: #cbd5e1; font-size: 0.9rem; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_evidence_details(
    result,
    vector_store,
) -> None:
    """Render YouTube-style concept evidence cards with timestamp chips."""
    answer = result.answer
    if not answer.evidence_ids:
        return

    concept_lookup = _build_concept_lookup(
        vector_store,
        answer.video_id,
    )
    if not concept_lookup:
        return

    concepts = []
    for evidence_id in answer.evidence_ids:
        concept = concept_lookup.get(evidence_id)
        if concept is not None:
            concepts.append(concept)

    if not concepts:
        return

    with st.expander(
        f"🎞️ Lecture Evidence  ·  {len(concepts)} relevant section(s)",
        expanded=True,
    ):
        for idx, concept in enumerate(concepts, start=1):
            start = _format_time(concept.start_time)
            end   = _format_time(concept.end_time)
            topic = concept.topic or concept.concept or "Lecture section"

            # ── Header chip row ──────────────────────────────
            st.markdown(
                f"""
                <div class="concept-bullet">
                <b>{idx}. {topic}</b>
                <span class="ts-chip">⏱ {start} – {end}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_s, col_v = st.columns(2)

            with col_s:
                if concept.speech_text and concept.speech_text.strip():
                    st.markdown(
                        "<div class='ev-card'>"
                        "<div class='ev-header'>🗣️ Lecturer said</div>"
                        f"<div class='ev-body'>{concept.speech_text.strip()}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            with col_v:
                if concept.visual_description and concept.visual_description.strip():
                    st.markdown(
                        "<div class='ev-card'>"
                        "<div class='ev-header'>👁️ On screen</div>"
                        f"<div class='ev-body'>{concept.visual_description.strip()}</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            if idx < len(concepts):
                st.markdown("<hr style='border-color:#1e3a5f;margin:8px 0'>", unsafe_allow_html=True)


def _render_result(
    result,
    vector_store,
) -> None:
    """YouTube-style answer panel."""
    answer  = result.answer
    outcome = result.overall_outcome

    icon_map = {
        VerificationOutcome.SUPPORTED:            "✅",
        VerificationOutcome.PARTIAL:              "⚠️",
        VerificationOutcome.UNSUPPORTED:          "❌",
        VerificationOutcome.INSUFFICIENT_EVIDENCE: "🔍",
    }
    icon = icon_map.get(outcome, "❓")

    displayed_answer = result.verified_answer or answer.answer

    # ── Main answer ──────────────────────────────────────────
    st.markdown(f"#### {icon} Answer")

    if displayed_answer:
        # Render each sentence / bullet on its own line so it
        # visually resembles the YouTube right-panel layout.
        paragraphs = [p.strip() for p in displayed_answer.split("\n") if p.strip()]
        for para in paragraphs:
            st.markdown(para)

    # ── Overall timestamp range ───────────────────────────────
    if answer.timestamp_start is not None:
        t_start = _format_time(answer.timestamp_start)
        t_end   = _format_time(answer.timestamp_end)
        st.markdown(
            f"<span class='ts-chip' style='font-size:0.85rem;'>⏱ {t_start} – {t_end}</span>"
            "&nbsp;&nbsp;<span style='color:#64748b;font-size:0.8rem;'>overall lecture range</span>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Explanation ───────────────────────────────────────────
    if answer.explanation:
        with st.expander("📖 How this answer was constructed", expanded=False):
            st.write(answer.explanation)

    # ── Lecture evidence sections ─────────────────────────────
    _render_evidence_details(result, vector_store)

    # ── Claim verification (collapsed by default) ─────────────
    if result.claim_verifications:
        with st.expander(f"🔬 Claim Verification ({outcome.value})", expanded=False):
            for verification in result.claim_verifications:
                if verification.outcome == VerificationOutcome.SUPPORTED:
                    cv_icon = "✅"
                elif verification.outcome == VerificationOutcome.PARTIAL:
                    cv_icon = "⚠️"
                else:
                    cv_icon = "❌"
                st.markdown(f"{cv_icon} {verification.claim}")
                st.caption(f"Support confidence: {verification.confidence:.0%}")
                st.divider()

# ==============================================================
# Dependency wiring — cached so models load once per session
# ==============================================================

@st.cache_resource
def get_vector_store():
    if settings.vector_store_backend.lower() == "pinecone" and settings.pinecone_api_key:
        logger.info(f"Using Pinecone vector store (index: {settings.pinecone_index_name})")
        return PineconeVectorStore()
    logger.info("Using local Chroma vector store")
    return ChromaVectorStore()

@st.cache_resource
def get_embedder():
    return SentenceEmbedder()

@st.cache_resource
def get_qwen_runtime():
    return QwenRuntime(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct"
    )

@st.cache_resource
def build_ingest_orchestrator():
    media_store = MediaStore()
    embedder = get_embedder()
    vector_store = get_vector_store()
    qwen_runtime = get_qwen_runtime()

    ingest_orchestrator = IngestOrchestrator(
        ingestion=VideoIngestionPipeline(),
        audio=AudioProcessingPipeline(
            transcriber=_get_whisper_transcriber()
        ),
        visual=VisualProcessingPipeline(
            frame_selector=OpenCVFrameSelector(),
            vision_model=QwenVisionAdapter(
                qwen_runtime
            ),
            event_former=SimpleVisualEventFormer(),
            media_store=media_store,
        ),
        alignment=CrossModalAlignmentPipeline(
            aligner=SimpleTemporalAligner()
        ),
        indexing=MemoryIndexingPipeline(
            embedder=embedder,
            vector_store=vector_store,
        ),
        media_store=media_store,
        job_manager=JobStateManager(),
    )

    return (
        ingest_orchestrator,
        vector_store,
        embedder,
        qwen_runtime,
    )

@st.cache_resource

def build_qa_orchestrator(

    _vector_store,

    _embedder,

    _qwen_runtime,

):

    return QAOrchestrator(

        qa_pipeline=QuestionAnsweringPipeline(

            retriever=TemporalRetriever(

                embedder=_embedder,

                vector_store=_vector_store,

            ),

            generator=QwenGeneratorAdapter(

                _qwen_runtime

            ),

            verifier=SimpleEvidenceVerifier(),

        )

    )

# ==============================================================

# Sidebar — upload + ingest

# ==============================================================
# Sidebar — upload + link ingestion
# ==============================================================

with st.sidebar:
    st.header("📥 Ingest Lecture")

    ingest_source = st.radio(
        "Ingest Source",
        options=["📁 File Upload", "🔗 Video URL / Link"],
        horizontal=True,
    )

    uploaded = None
    video_url = ""

    if ingest_source == "📁 File Upload":
        uploaded = st.file_uploader(
            "Upload video file",
            type=[
                "mp4",
                "mov",
                "avi",
                "mkv",
                "webm",
            ],
        )
    else:
        video_url = st.text_input(
            "Video URL / Link",
            placeholder="https://www.youtube.com/watch?v=... or direct .mp4 link",
            help="Supports YouTube, Vimeo, Loom, Coursera, Google Drive, or direct MP4/WebM URLs.",
        )
        if video_url.strip():
            if VideoLinkDownloader.is_valid_url(video_url):
                st.caption("✅ Valid URL format detected")
            else:
                st.caption("⚠️ Please enter a valid http:// or https:// URL")

    subject = st.text_input(
        "Subject (optional)",
        placeholder="e.g. Computer Networks",
        help="Will auto-populate from video title if using a URL.",
    )

    ingest_clicked = st.button(
        "🚀 Process Video",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.header("⚙️ Architecture & Settings")

    vs_name = (
        f"🌲 Pinecone (`{settings.pinecone_index_name}`)"
        if settings.vector_store_backend.lower() == "pinecone" and settings.pinecone_api_key
        else "💾 Local ChromaDB"
    )
    st.caption(f"Vector Database: {vs_name}")

    st.caption(
        "Vision model: "
        "`Qwen/Qwen2.5-VL-3B-Instruct`"
    )

    st.caption(
        f"Embedding model: "
        f"`{settings.embedding_model}`"
    )

    st.caption(
        f"Frame interval: "
        f"`{settings.frame_interval_sec}s`"
    )

# ==============================================================
# Session state
# ==============================================================

if "ingested_video_id" not in st.session_state:
    st.session_state.ingested_video_id = None

if "job_log" not in st.session_state:
    st.session_state.job_log = []

# ==============================================================
# Ingest
# ==============================================================

if ingest_clicked:
    video_to_process = None
    video_display_name = ""
    is_temp_file = False

    if ingest_source == "📁 File Upload":
        if uploaded is None:
            st.sidebar.error("Please upload a video file first.")
        else:
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uploaded.name)[1],
                delete=False,
            ) as tmp:
                tmp.write(uploaded.getbuffer())
                video_to_process = tmp.name
                video_display_name = uploaded.name
                is_temp_file = True
    else:
        if not video_url.strip() or not VideoLinkDownloader.is_valid_url(video_url):
            st.sidebar.error("Please enter a valid video URL.")
        else:
            dl_progress = st.sidebar.progress(0.0, text="Initializing download...")

            def _on_dl_progress(pct: float, msg: str):
                dl_progress.progress(min(max(pct, 0.0), 1.0), text=msg)

            try:
                downloader = VideoLinkDownloader()
                with st.spinner("Downloading video from link..."):
                    dl_info = downloader.download(
                        video_url.strip(),
                        progress_callback=_on_dl_progress,
                    )
                video_to_process = dl_info.file_path
                video_display_name = dl_info.title
                if not subject and dl_info.title:
                    subject = dl_info.title
                dl_progress.empty()
            except Exception as dl_err:
                err_str = str(dl_err)
                if "403" in err_str or "Forbidden" in err_str:
                    st.sidebar.error("❌ YouTube blocked the download (HTTP 403).")
                    st.sidebar.info(
                        "**Fix: Export cookies from your browser**\n\n"
                        "1. Install the **[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)** Chrome extension.\n"
                        "2. Open **youtube.com**, sign in to your Google account.\n"
                        "3. Click the extension icon → Export cookies → save as `cookies.txt`.\n"
                        "4. Put `cookies.txt` in the project root:\n"
                        "   `d:\\\\Mainproject\\\\VLLM-branch-qwen-changes (1)\\\\VLLM-branch-qwen-changes\\\\cookies.txt`\n"
                        "5. Retry the download."
                    )
                else:
                    st.sidebar.error(f"❌ Failed to download video: {dl_err}")
                logger.error(f"video_url_download_failed | {dl_err}")

    if video_to_process:
        (
            ingest_orch,
            vector_store,
            embedder,
            qwen_runtime,
        ) = build_ingest_orchestrator()

        with st.spinner("Processing video through the full pipeline…"):
            try:
                job = ingest_orch.run(
                    video_to_process,
                    subject=subject or None,
                )
                st.session_state.ingested_video_id = job.video_id
                st.sidebar.success("✅ Lecture processed successfully.")
                st.session_state.job_log.append(
                    f"Ingested: {video_display_name} → {job.video_id} | status={job.status}"
                )
            except Exception as exc:
                st.sidebar.error(f"❌ Ingestion failed: {exc}")
                logger.error(f"ui_ingest_error | {exc}")
            finally:
                if is_temp_file and os.path.exists(video_to_process):
                    try:
                        os.unlink(video_to_process)
                    except Exception:
                        pass

# ==============================================================

# Tabs

# ==============================================================

tab_qa, tab_library, tab_eval = st.tabs(

    [

        "💬 Ask",

        "📚 Library",

        "📊 Evaluate",

    ]

)

# ==============================================================
# QA Tab
# ==============================================================

with tab_qa:

    # ── Video selector ────────────────────────────────────────
    _vs_for_qa = get_vector_store()
    _known_video_ids: list[str] = []
    try:
        if hasattr(_vs_for_qa, "list_videos"):
            _known_video_ids = _vs_for_qa.list_videos()
        else:
            # Fallback: derive from a broad search (ChromaDB)
            _dummy_emb = [0.0] * 384
            _all_c = _vs_for_qa.search(_dummy_emb, top_k=200)
            _known_video_ids = sorted({c.video_id for c in _all_c if c.video_id})
    except Exception:
        _known_video_ids = []

    _video_options = ["🌐 All videos"] + _known_video_ids

    # If a video was just ingested, pre-select it
    _default_idx = 0
    if st.session_state.ingested_video_id and st.session_state.ingested_video_id in _known_video_ids:
        _default_idx = _known_video_ids.index(st.session_state.ingested_video_id) + 1  # +1 for "All videos"

    _selected_video_label = st.selectbox(
        "🎬 Filter by video",
        options=_video_options,
        index=_default_idx,
        help="Select a specific ingested lecture or leave on 'All videos' to search across everything.",
    )
    video_id_filter = "" if _selected_video_label == "🌐 All videos" else _selected_video_label

    if video_id_filter:
        st.caption(f"🔒 Searching within: `{video_id_filter}`")
    else:
        st.caption("🔍 Searching across all ingested lectures")

    question = st.text_area(
        "Your question",
        placeholder="What is the TCP three-way handshake?",
        height=90,
    )

    ask_clicked = st.button("🔍 Ask", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            (
                ingest_orch,
                vector_store,
                embedder,
                qwen_runtime,
            ) = build_ingest_orchestrator()

            qa_orch = build_qa_orchestrator(
                vector_store,
                embedder,
                qwen_runtime,
            )

            with st.spinner("Retrieving multimodal lecture context and generating answer…"):
                try:
                    result = qa_orch.ask(
                        question.strip(),
                        video_id=video_id_filter.strip() or None,
                    )
                    _render_result(result, vector_store)
                except Exception as exc:
                    st.error(f"❌ QA failed: {exc}")
                    logger.error(f"ui_qa_error | {exc}")

# ==============================================================
# Library / Browse Videos Tab
# ==============================================================

with tab_library:

    st.subheader("📂 Browse Videos in Pinecone DB")

    lib_vs = get_vector_store()

    # ── Collect video IDs ─────────────────────────────────────
    lib_video_ids: list[str] = []
    lib_all_concepts: list = []
    try:
        if hasattr(lib_vs, "list_videos"):
            lib_video_ids = lib_vs.list_videos()
        else:
            _d = [0.0] * 384
            lib_all_concepts = lib_vs.search(_d, top_k=500)
            lib_video_ids = sorted({c.video_id for c in lib_all_concepts if c.video_id})
    except Exception:
        pass

    if not lib_video_ids:
        st.info("No videos ingested yet. Upload one using the sidebar.")
    else:
        st.caption(f"{len(lib_video_ids)} lecture(s) stored in Pinecone")
        st.markdown("---")

        # ── Session state: which video is expanded ────────────
        if "lib_selected_vid" not in st.session_state:
            st.session_state.lib_selected_vid = None

        # ── Left panel: video cards ───────────────────────────
        col_list, col_detail = st.columns([1, 2], gap="large")

        with col_list:
            st.markdown("### 🎬 Videos")
            for vid in lib_video_ids:
                # Fetch summary info (from already-fetched concepts if available)
                vid_concepts = [c for c in lib_all_concepts if c.video_id == vid]
                if not vid_concepts:
                    try:
                        vid_concepts = lib_vs.get_by_video(vid)
                    except Exception:
                        vid_concepts = []

                subjects = sorted({c.subject for c in vid_concepts if c.subject})
                duration = max((c.end_time for c in vid_concepts), default=0.0)
                n_topics = len({c.topic for c in vid_concepts if c.topic})

                # Card styling
                is_selected = st.session_state.lib_selected_vid == vid
                card_bg = "#1e3a5f" if is_selected else "#111827"
                border_c = "#3b82f6" if is_selected else "#1e293b"

                label_top = subjects[0] if subjects else "Unknown subject"
                label_id  = vid[:8] + "…"  # show truncated UUID

                st.markdown(
                    f"""
                    <div style="background:{card_bg};border:1px solid {border_c};
                    border-radius:8px;padding:10px 14px;margin-bottom:8px;cursor:pointer">
                    <div style="color:#e2e8f0;font-weight:600;font-size:0.95rem">{label_top}</div>
                    <div style="color:#64748b;font-size:0.75rem;margin-top:2px">
                      🆔 {label_id} &nbsp;|&nbsp; ⏱ {_format_time(duration)} &nbsp;|&nbsp; 📌 {len(vid_concepts)} concepts
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"{'✅ ' if is_selected else ''}View  {label_id}",
                    key=f"lib_sel_{vid}",
                    use_container_width=True,
                ):
                    st.session_state.lib_selected_vid = vid
                    st.rerun()

        # ── Right panel: concept timeline ─────────────────────
        with col_detail:
            sel = st.session_state.lib_selected_vid

            if sel is None:
                st.markdown(
                    "<div style='color:#475569;margin-top:60px;text-align:center'>"
                    "← Select a video to browse its concepts"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Fetch concepts for selected video
                sel_concepts: list = [c for c in lib_all_concepts if c.video_id == sel]
                if not sel_concepts:
                    try:
                        sel_concepts = lib_vs.get_by_video(sel)
                    except Exception:
                        sel_concepts = []

                sel_concepts.sort(key=lambda c: c.start_time)

                subjects_sel = sorted({c.subject for c in sel_concepts if c.subject})
                topics_sel   = sorted({c.topic   for c in sel_concepts if c.topic})
                duration_sel = max((c.end_time for c in sel_concepts), default=0.0)

                # ── Header ───────────────────────────────────
                st.markdown(
                    f"### {'  ·  '.join(subjects_sel) if subjects_sel else 'Lecture'}"
                )
                st.caption(f"Video ID: `{sel}`")

                m1, m2, m3 = st.columns(3)
                m1.metric("Concepts",  len(sel_concepts))
                m2.metric("Duration",  _format_time(duration_sel))
                m3.metric("Topics",    len(topics_sel))

                if topics_sel:
                    with st.expander("📌 Topics in this lecture", expanded=False):
                        for t in topics_sel:
                            st.markdown(f"- {t}")

                # ── Delete button ─────────────────────────────
                if st.button(f"🗑️ Delete this lecture", key=f"lib_del_{sel}"):
                    try:
                        lib_vs.delete_by_video(sel)
                        st.session_state.lib_selected_vid = None
                        st.success("Lecture deleted from Pinecone.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

                st.markdown("---")
                st.markdown("#### 📅 Concept Timeline")

                # ── Concept timeline ──────────────────────────
                for concept in sel_concepts:
                    t_start = _format_time(concept.start_time)
                    t_end   = _format_time(concept.end_time)
                    topic_label = concept.topic or concept.concept or "—"

                    with st.expander(
                        f"{t_start} – {t_end}  ·  {topic_label}",
                        expanded=False,
                    ):
                        c_sp, c_vi = st.columns(2)
                        with c_sp:
                            if concept.speech_text and concept.speech_text.strip():
                                st.markdown(
                                    "<div class='ev-card'>"
                                    "<div class='ev-header'>🗣️ Lecturer said</div>"
                                    f"<div class='ev-body'>{concept.speech_text.strip()}</div>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption("No speech recorded.")
                        with c_vi:
                            if concept.visual_description and concept.visual_description.strip():
                                st.markdown(
                                    "<div class='ev-card'>"
                                    "<div class='ev-header'>👁️ On screen</div>"
                                    f"<div class='ev-body'>{concept.visual_description.strip()}</div>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.caption("No visual description.")

# ==============================================================

# Evaluation Tab

# ==============================================================

with tab_eval:

    st.subheader(

        "Pipeline Evaluation"

    )

    st.caption(

        "Define question/answer pairs with known relevant "

        "concept IDs and expected timestamps. "

        "The harness computes Precision@K, Recall@K, MRR, "

        "timestamp error, answer overlap, groundedness rate, "

        "and unsupported claim rate."

    )

    eval_video_id = st.text_input(

        "Video ID to evaluate against",

        value=(

            st.session_state.ingested_video_id

            or ""

        ),

        key="eval_vid",

    )

    st.markdown(

        "**Sample questions** "

        "(one per line, format: `question | expected_answer`)"

    )

    sample_text = st.text_area(

        "Samples",

        placeholder=(

            "What is the TCP three-way handshake? | "

            "SYN SYN-ACK ACK\n"

            "What is a deadlock? | circular wait condition"

        ),

        height=150,

        key="eval_samples",

    )

    run_eval = st.button(

        "▶️ Run Evaluation",

        type="primary",

    )

    if run_eval:

        if not sample_text.strip():

            st.warning(

                "Enter at least one sample."

            )

        else:

            from lecture_memory.evaluation.harness import (

                EvaluationHarness,

                QASample,

            )

            samples = []

            for line in sample_text.strip().splitlines():

                parts = line.split(

                    "|",

                    1,

                )

                q = parts[0].strip()

                expected = (

                    parts[1].strip()

                    if len(parts) > 1

                    else ""

                )

                if q:

                    samples.append(

                        QASample(

                            question=q,

                            expected_answer=expected,

                            video_id=(

                                eval_video_id.strip()

                                or None

                            ),

                        )

                    )

            (

                ingest_orch,

                vector_store,

                embedder,

                qwen_runtime,

            ) = build_ingest_orchestrator()

            qa_orch = build_qa_orchestrator(

                vector_store,

                embedder,

                qwen_runtime,

            )

            harness = EvaluationHarness(

                qa_orch

            )

            with st.spinner(

                f"Evaluating {len(samples)} sample(s)…"

            ):

                try:

                    report = harness.run(

                        samples

                    )

                    # --------------------------------------------------

                    # Summary metrics

                    # --------------------------------------------------

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(

                        "Precision@K",

                        f"{report.mean_precision_at_k:.3f}",

                    )

                    m2.metric(

                        "Recall@K",

                        f"{report.mean_recall_at_k:.3f}",

                    )

                    m3.metric(

                        "MRR",

                        f"{report.mrr:.3f}",

                    )

                    m4.metric(

                        "Groundedness",

                        f"{report.groundedness_rate:.1%}",

                    )

                    m5, m6, m7 = st.columns(3)

                    m5.metric(

                        "Answer Overlap",

                        f"{report.mean_answer_overlap:.3f}",

                    )

                    m6.metric(

                        "Unsupported Rate",

                        f"{report.mean_unsupported_claim_rate:.1%}",

                    )

                    m7.metric(

                        "Mean Latency",

                        f"{report.mean_latency_sec:.2f}s",

                    )

                    # --------------------------------------------------

                    # Per-sample breakdown

                    # --------------------------------------------------

                    with st.expander(

                        "Per-sample results"

                    ):

                        for sr in report.samples:

                            st.markdown(

                                f"**Q:** {sr.question}"

                            )

                            st.markdown(

                                f"**A:** {sr.predicted_answer}"

                            )

                            cols = st.columns(4)

                            cols[0].caption(

                                f"P@K: "

                                f"{sr.precision_at_k:.2f}"

                            )

                            cols[1].caption(

                                f"R@K: "

                                f"{sr.recall_at_k:.2f}"

                            )

                            cols[2].caption(

                                f"RR: "

                                f"{sr.reciprocal_rank:.2f}"

                            )

                            cols[3].caption(

                                f"Overlap: "

                                f"{sr.answer_overlap:.2f}"

                            )

                            st.divider()

                except Exception as exc:

                    st.error(

                        f"Evaluation failed: {exc}"

                    )

                    logger.error(

                        f"ui_eval_error | {exc}"

                    )

# ==============================================================

# Job log

# ==============================================================

if st.session_state.job_log:

    with st.expander(

        "📋 Processing Log"

    ):

        for line in st.session_state.job_log:

            st.text(line)