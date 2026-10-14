# Batch ingestion of all .mp4 lecture videos in a folder
#
# The script uses the existing IngestOrchestrator pipeline:
#   - Validate video & extract metadata
#   - Extract audio → transcribe (via Whisper if configured)
#   - Frame selection → vision model → event formation
#   - Cross‑modal alignment
#   - Index concepts into the Pinecone vector store
#
# Place your .mp4 files in the folder given by `VIDEO_DIR`.
# Run the script from the repository root.

import pathlib
import sys
import traceback

# --------------------------------------------------------------
# Project imports – adjust the PYTHONPATH if you run the script
# from the repository root (recommended).
# --------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lecture_memory.orchestrator.ingest_orchestrator import IngestOrchestrator
from lecture_memory.pipeline.ingestion import VideoIngestionPipeline
from lecture_memory.pipeline.audio import AudioProcessingPipeline
from lecture_memory.pipeline.visual import VisualProcessingPipeline
from lecture_memory.pipeline.alignment import CrossModalAlignmentPipeline
from lecture_memory.pipeline.indexing import MemoryIndexingPipeline
from lecture_memory.storage.media_store import MediaStore
from lecture_memory.adapter_factory import (
    get_frame_selector,
    get_vision_model,
    get_event_former,
    get_speech_transcriber,
)
from lecture_memory.observability.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
VIDEO_DIR = PROJECT_ROOT / "data" / "batch_videos"  # folder with your .mp4 files
MAX_VIDEO_COUNT = None  # Set an integer to limit the batch, or None for all

# -----------------------------------------------------------------
# Helper: build the orchestrator exactly as the Streamlit app does
# -----------------------------------------------------------------
def build_orchestrator() -> IngestOrchestrator:
    ingestion = VideoIngestionPipeline()
    audio = AudioProcessingPipeline(transcriber=get_speech_transcriber())
    visual = VisualProcessingPipeline(
        frame_selector=get_frame_selector(),
        vision_model=get_vision_model(),
        event_former=get_event_former(),
        media_store=MediaStore(),
    )
    alignment = CrossModalAlignmentPipeline()
    indexing = MemoryIndexingPipeline()
    media_store = MediaStore()
    return IngestOrchestrator(
        ingestion=ingestion,
        audio=audio,
        visual=visual,
        alignment=alignment,
        indexing=indexing,
        media_store=media_store,
    )


def main() -> None:
    if not VIDEO_DIR.is_dir():
        logger.error(f"❌ Video folder does not exist: {VIDEO_DIR}")
        sys.exit(1)

    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not mp4_files:
        logger.info(f"🟡 No .mp4 files found in {VIDEO_DIR}")
        return

    if MAX_VIDEO_COUNT:
        mp4_files = mp4_files[:MAX_VIDEO_COUNT]

    logger.info(f"🚀 Starting ingestion of {len(mp4_files)} video(s) from {VIDEO_DIR}")

    orchestrator = build_orchestrator()

    for idx, video_path in enumerate(mp4_files, start=1):
        logger.info(f"\n=== [{idx}/{len(mp4_files)}] Ingesting: {video_path.name} ===")
        try:
            job = orchestrator.run(str(video_path))
            logger.info(
                f"✅ Completed: video_id={job.video_id} | status={job.status.name}"
            )
        except Exception as exc:
            logger.error(
                f"❌ Failed to ingest {video_path.name}: {exc}\n{traceback.format_exc()}"
            )
            continue

    logger.info("\n🎉 Batch ingestion finished.")


if __name__ == "__main__":
    main()
