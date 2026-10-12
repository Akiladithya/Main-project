
"""
adapters/qwen_vision.py

Implements IVisionModel using the shared Qwen2.5-VL runtime.

The Qwen model itself is loaded only once by QwenRuntime.

This adapter:

- analyzes selected lecture frames using Qwen2.5-VL
- extracts visual descriptions, entities, and CS concepts
- preserves frame timestamp
- preserves frame selection reason
- preserves visual change score

The vision stage is optimized for low-VRAM local inference:
- controlled image resolution
- short structured generation
- shared Qwen runtime
- no batching
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info

from lecture_memory.domain.models import FrameEvidence, VisualEvidence
from lecture_memory.interfaces.vision import IVisionModel
from lecture_memory.observability.logger import get_logger
from lecture_memory.adapters.qwen_runtime import QwenRuntime


logger = get_logger(__name__)


# ============================================================
# VISION PROCESSING LIMITS
# ============================================================

# Qwen2.5-VL processes images using visual tokens.
#
# Keeping the maximum visual-token budget moderate is important
# for a 4 GB VRAM GPU.
#
# 28 x 28 is the visual patch size used by Qwen2.5-VL.
_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 768 * 28 * 28

# The answer only needs Caption + Entities + Concepts.
# 120 tokens is enough for this structured output.
_MAX_NEW_TOKENS = 120


# ============================================================
# QWEN VISION PROMPT
# ============================================================

_PROMPT = """You are analyzing one frame from a Computer Science lecture.

Timestamp: {timestamp:.1f}s

{context_block}

Describe ONLY information visibly present in the frame.

Focus on:
- diagrams and architecture
- trees, graphs, data structures
- flowcharts and state machines
- code and pseudocode
- algorithms and procedures
- formulas and equations
- tables, lists and labels
- relationships between visible components
- important Computer Science terms

For code, identify the visible operation or structure.
For diagrams, identify components and their relationships.
For tables, identify the type of information shown.

Do not invent information.

Respond exactly as:

Caption: <one concise sentence>
Entities: <comma-separated visible terms>
Concepts: <comma-separated CS concepts>
"""

_CONTEXT_BLOCK = (
    'Previous frame context: "{context}"'
)


class QwenVisionAdapter(IVisionModel):

    def __init__(self, runtime: QwenRuntime) -> None:
        """
        Reuse the shared Qwen runtime.

        The model is loaded only once.
        """

        self._runtime = runtime
        self._processor = runtime.processor
        self._model = runtime.model

        logger.info("qwen_vision_adapter_ready")

    # ========================================================
    # FRAME DESCRIPTION
    # ========================================================

    def describe_frame(
        self,
        frame: FrameEvidence,
        context: Optional[str] = None,
    ) -> VisualEvidence:

        # ----------------------------------------------------
        # Build previous-frame context
        # ----------------------------------------------------

        context_block = (
            _CONTEXT_BLOCK.format(context=context)
            if context
            else "No previous frame context."
        )

        prompt = _PROMPT.format(
            timestamp=frame.timestamp,
            context_block=context_block,
        )

        # ----------------------------------------------------
        # Retry Qwen vision up to 3 times.
        #
        # Retries are kept because a single GPU inference
        # failure should not destroy the complete video job.
        # ----------------------------------------------------

        for attempt in range(3):

            try:

                # ------------------------------------------------
                # Construct multimodal Qwen message
                # ------------------------------------------------

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": frame.image_uri,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ]

                # ------------------------------------------------
                # Prepare chat template
                # ------------------------------------------------

                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(
                    messages
                )

                # ------------------------------------------------
                # Process image with controlled visual resolution.
                #
                # This is important for the 4 GB RTX 2050.
                #
                # We do NOT manually resize the saved frame.
                # The processor performs the visual-token
                # resizing while preserving the original image.
                # ------------------------------------------------

                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=_MIN_PIXELS,
                    max_pixels=_MAX_PIXELS,
                )

                # ------------------------------------------------
                # Move inputs to Qwen's device
                # ------------------------------------------------

                inputs = inputs.to(self._model.device)

                # ------------------------------------------------
                # Generate concise visual description
                # ------------------------------------------------

                with torch.inference_mode():

                    generated_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=_MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )

                # ------------------------------------------------
                # Remove input/prompt tokens
                # ------------------------------------------------

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(
                        inputs.input_ids,
                        generated_ids,
                    )
                ]

                # ------------------------------------------------
                # Decode Qwen output
                # ------------------------------------------------

                output_text = self._processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                raw = output_text[0].strip()

                # ------------------------------------------------
                # Parse structured response
                # ------------------------------------------------

                parsed = self._parse(raw)

                # ------------------------------------------------
                # Preserve FrameEvidence metadata.
                #
                # This metadata is later used by:
                #
                # frame
                #   ↓
                # visual evidence
                #   ↓
                # multimodal concept
                #   ↓
                # retrieval/reranking
                #
                # So the visual-change information is not lost.
                # ------------------------------------------------

                return VisualEvidence(
                    visual_id=str(uuid.uuid4()),
                    frame_id=frame.frame_id,
                    video_id=frame.video_id,
                    timestamp=frame.timestamp,
                    caption=(
                        parsed["caption"]
                        or "(no caption)"
                    ),
                    entities=parsed["entities"],
                    concept_candidates=parsed["concepts"],
                    selection_reason=frame.selection_reason,
                    visual_change_score=frame.visual_change_score,
                )

            except Exception as exc:

                logger.warning(
                    f"vision_retry | "
                    f"attempt={attempt + 1} | "
                    f"frame_id={frame.frame_id} | "
                    f"error={exc}"
                )

                # Short backoff between retries.
                time.sleep(1.5 * (attempt + 1))

        # --------------------------------------------------------
        # All retries failed
        # --------------------------------------------------------

        raise RuntimeError(
            f"Qwen vision model failed for frame "
            f"{frame.frame_id}"
        )

    # ========================================================
    # RESPONSE PARSER
    # ========================================================

    def _parse(self, text: str) -> dict:

        result = {
            "caption": "",
            "entities": [],
            "concepts": [],
        }

        for line in text.splitlines():

            line = line.strip()

            if line.lower().startswith("caption:"):

                result["caption"] = (
                    line.split(":", 1)[1].strip()
                )

            elif line.lower().startswith("entities:"):

                result["entities"] = [
                    entity.strip()
                    for entity in (
                        line.split(":", 1)[1].split(",")
                    )
                    if entity.strip()
                ]

            elif line.lower().startswith("concepts:"):

                result["concepts"] = [
                    concept.strip()
                    for concept in (
                        line.split(":", 1)[1].split(",")
                    )
                    if concept.strip()
                ]

        return result

