from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class SelectionReason(str, Enum):
    UNIFORM_SAMPLE = "uniform_sample"
    VISUAL_CHANGE = "visual_change"
    SCENE_BOUNDARY = "scene_boundary"
    FORCED_KEYFRAME = "forced_keyframe"


class VerificationOutcome(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ModalityType(str, Enum):
    SPEECH = "speech"
    VISUAL = "visual"
    MULTIMODAL = "multimodal"
