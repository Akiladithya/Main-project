
"""
config/settings.py

----------------------------------

All configuration in one place.
Loaded from environment / .env file.

No module should hardcode values that belong here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ============================================================
    # API KEYS
    # ============================================================

    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
    )

    pinecone_api_key: str = Field(
        default="pcsk_2eYYhQ_H2eB8vnzGNfH9cMmLS1zHwGsofnX7HNt7gWpD3Y186FMY1tfuNp2UxEg4rZh4j7",
        alias="PINECONE_API_KEY",
    )

    pinecone_index_name: str = Field(
        default="lecture-concepts",
        alias="PINECONE_INDEX_NAME",
    )

    vector_store_backend: str = Field(
        default="pinecone",
        alias="VECTOR_STORE_BACKEND",
    )

    # ============================================================
    # MODELS
    # ============================================================

    # Kept for backward compatibility with the existing
    # configuration/interface.
    #
    # The actual local vision and generation runtime currently
    # uses Qwen2.5-VL through QwenRuntime.
    vision_model: str = "gemini-2.5-flash"
    llm_model: str = "gemini-2.5-flash"
    subject_extraction_model: str = "gemini-2.5-flash"

    embedding_model: str = "all-MiniLM-L6-v2"

    # ============================================================
    # FRAME SELECTION
    # ============================================================

    # One candidate every 5 seconds.
    #
    # This is only CHEAP candidate sampling.
    # The OpenCV selector then chooses the most useful frames.
    frame_interval_sec: float = 5.0

    # Hard upper limit on expensive Qwen vision calls.
    #
    # Suitable for the RTX 2050 4 GB setup.
    max_frames_per_video: int = 60

    # Histogram difference threshold.
    #
    # Higher = only stronger visual changes are prioritized.
    # Lower = more sensitive to smaller changes.
    visual_change_threshold: float = 0.30

    # ============================================================
    # VIDEO INGESTION
    # ============================================================

    max_video_duration_sec: float = 1800.0  # 30 minutes

    supported_formats: list[str] = [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    ]

    # ============================================================
    # RETRIEVAL
    # ============================================================

    retrieval_top_k: int = 10

    temporal_expansion_sec: float = 30.0

    # ============================================================
    # STORAGE
    # ============================================================

    media_storage_dir: str = "data/media"

    frames_storage_dir: str = "data/frames"

    vector_store_dir: str = "data/vectorstore"

    # ============================================================
    # OBSERVABILITY
    # ============================================================

    log_level: str = "INFO"

    log_format: str = "json"

    # ============================================================
    # PROCESSING
    # ============================================================

    processing_version: str = "2.0"

    schema_version: str = "1.0"

    # ============================================================
    # PYDANTIC SETTINGS CONFIGURATION
    # ============================================================

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


# Singleton — import this everywhere.
settings = Settings()

