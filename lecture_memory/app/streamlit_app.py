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

def _render_evidence_details(

    result,

    vector_store,

) -> None:

    answer = result.answer

    if not answer.evidence_ids:

        return

    concept_lookup = _build_concept_lookup(

        vector_store,

        answer.video_id,

    )

    if not concept_lookup:

        return

    # ----------------------------------------------------------

    # Keep the order returned by the QA pipeline.

    # ----------------------------------------------------------

    concepts = []

    for evidence_id in answer.evidence_ids:

        concept = concept_lookup.get(

            evidence_id

        )

        if concept is not None:

            concepts.append(concept)

    if not concepts:

        return

    with st.expander(

        f"🎞️ Lecture Evidence ({len(concepts)} relevant sections)"

    ):

        st.caption(

            "These sections show how the answer is grounded "

            "in the lecture's speech, visuals and temporal context."

        )

        for index, concept in enumerate(

            concepts,

            start=1,

        ):

            start = _format_time(

                concept.start_time

            )

            end = _format_time(

                concept.end_time

            )

            st.markdown(

                f"### {index}. Lecture section — {start} to {end}"

            )

            # --------------------------------------------------

            # Topic

            # --------------------------------------------------

            if concept.topic:

                st.markdown(

                    f"**Topic:** {concept.topic}"

                )

            # --------------------------------------------------

            # Speech

            # --------------------------------------------------

            if concept.speech_text:

                st.markdown(

                    "**🗣️ What the lecturer said**"

                )

                st.write(

                    concept.speech_text

                )

            # --------------------------------------------------

            # Visual

            # --------------------------------------------------

            if concept.visual_description:

                st.markdown(

                    "**👁️ What was shown visually**"

                )

                st.write(

                    concept.visual_description

                )

            # --------------------------------------------------

            # Temporal / visual-change information

            # --------------------------------------------------

            if (

                concept.visual_change_score > 0

                or concept.selection_reasons

            ):

                details = []

                if concept.selection_reasons:

                    reasons = ", ".join(

                        str(reason)

                        for reason in concept.selection_reasons

                    )

                    details.append(

                        f"Frame selection: {reasons}"

                    )

                if concept.visual_change_score > 0:

                    details.append(

                        "Visual change score: "

                        f"{concept.visual_change_score:.2f}"

                    )

                if details:

                    st.caption(

                        " • ".join(details)

                    )

            st.divider()

def _render_result(

    result,

    vector_store,

) -> None:

    answer = result.answer

    outcome = result.overall_outcome

    icon = {

        VerificationOutcome.SUPPORTED: "✅",

        VerificationOutcome.PARTIAL: "⚠️",

        VerificationOutcome.UNSUPPORTED: "❌",

        VerificationOutcome.INSUFFICIENT_EVIDENCE: "🔍",

    }.get(

        outcome,

        "❓",

    )

    # ==========================================================

    # Answer

    # ==========================================================

    st.subheader(

        f"{icon} Answer"

    )

    displayed_answer = (

        result.verified_answer

        or answer.answer

    )

    if displayed_answer:

        st.info(

            displayed_answer

        )

    # ==========================================================

    # Explanation

    # ==========================================================

    if answer.explanation:

        with st.expander(

            "📖 Explanation",

            expanded=True,

        ):

            st.write(

                answer.explanation

            )

    # ==========================================================

    # Lecture time range

    # ==========================================================

    if answer.timestamp_start is not None:

        st.markdown(

            "### ⏱️ Relevant lecture section"

        )

        c1, c2 = st.columns(2)

        c1.metric(

            "Start",

            _format_time(

                answer.timestamp_start

            ),

        )

        c2.metric(

            "End",

            _format_time(

                answer.timestamp_end

            ),

        )

        st.caption(

            "The time range covers the retrieved lecture "

            "sections used to construct the answer."

        )

    # ==========================================================

    # Multimodal lecture evidence

    # ==========================================================

    _render_evidence_details(

        result,

        vector_store,

    )

    # ==========================================================

    # Claim verification

    #

    # Internal evidence IDs are deliberately hidden from the

    # student-facing UI.

    # ==========================================================

    if result.claim_verifications:

        with st.expander(

            f"🔬 Answer Verification ({outcome.value})"

        ):

            for verification in result.claim_verifications:

                if (

                    verification.outcome

                    == VerificationOutcome.SUPPORTED

                ):

                    cv_icon = "✅"

                elif (

                    verification.outcome

                    == VerificationOutcome.PARTIAL

                ):

                    cv_icon = "⚠️"

                else:

                    cv_icon = "❌"

                st.markdown(

                    f"{cv_icon} {verification.claim}"

                )

                st.caption(

                    f"Support confidence: "

                    f"{verification.confidence:.0%}"

                )

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

    video_id_filter = st.text_input(

        "Filter by video ID "

        "(optional — leave blank to search all ingested lectures)",

        value=(

            st.session_state.ingested_video_id

            or ""

        ),

    )

    question = st.text_area(

        "Your question",

        placeholder=(

            "What is the TCP three-way handshake?"

        ),

    )

    ask_clicked = st.button(

        "🔍 Ask",

        type="primary",

    )

    if ask_clicked:

        if not question.strip():

            st.warning(

                "Please enter a question."

            )

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

            with st.spinner(

                "Retrieving multimodal lecture context "

                "and generating answer…"

            ):

                try:

                    result = qa_orch.ask(

                        question.strip(),

                        video_id=(

                            video_id_filter.strip()

                            or None

                        ),

                    )

                    _render_result(

                        result,

                        vector_store,

                    )

                except Exception as exc:

                    st.error(

                        f"❌ QA failed: {exc}"

                    )

                    logger.error(

                        f"ui_qa_error | {exc}"

                    )

# ==============================================================

# Library Tab

# ==============================================================

with tab_library:

    st.subheader(

        "Ingested Videos"

    )

    vector_store = get_vector_store()

    all_concepts = []

    try:

        # Collect concepts to derive unique video IDs.

        #

        # ChromaVectorStore does not currently expose a

        # list_videos() method, so we use the existing search

        # interface as a lightweight peek.

        dummy_emb = [0.0] * 384

        all_concepts = vector_store.search(

            dummy_emb,

            top_k=200,

        )

    except Exception:

        pass

    video_ids = sorted(

        {

            concept.video_id

            for concept in all_concepts

        }

    )

    if not video_ids:

        st.info(

            "No videos ingested yet. "

            "Upload one using the sidebar."

        )

    else:

        for vid in video_ids:

            concepts = [

                concept

                for concept in all_concepts

                if concept.video_id == vid

            ]

            subjects = {

                concept.subject

                for concept in concepts

                if concept.subject

            }

            topics = {

                concept.topic

                for concept in concepts

                if concept.topic

            }

            duration = (

                max(

                    concept.end_time

                    for concept in concepts

                )

                if concepts

                else 0.0

            )

            with st.expander(

                f"🎬 {vid}"

            ):

                col1, col2, col3 = st.columns(3)

                col1.metric(

                    "Concepts",

                    len(concepts),

                )

                col2.metric(

                    "Duration",

                    _format_time(duration),

                )

                col3.metric(

                    "Topics",

                    len(topics),

                )

                if subjects:

                    st.caption(

                        "Subjects: "

                        + ", ".join(

                            sorted(subjects)

                        )

                    )

                if topics:

                    st.caption(

                        "Topics: "

                        + ", ".join(

                            sorted(topics)

                        )

                    )

                if st.button(

                    f"🗑️ Delete {vid}",

                    key=f"del_{vid}",

                ):

                    try:

                        vector_store.delete_by_video(

                            vid

                        )

                        st.success(

                            "Deleted lecture from "

                            "the vector store."

                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(

                            f"Delete failed: {exc}"

                        )

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