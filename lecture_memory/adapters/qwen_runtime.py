"""
adapters/qwen_runtime.py

Shared Qwen2.5-VL runtime.

The model is loaded exactly once and reused by:
- QwenVisionAdapter
- QwenGenerator

Designed for lightweight local inference on a 4 GB GPU.
"""

from __future__ import annotations

import json
import os
import struct
# Ensure Hugging Face hub uses a generous 10-minute timeout for large weight downloads
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import torch
import transformers.modeling_utils
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Windows Paging File & Memory-Mapping Workaround
# ---------------------------------------------------------------------------
_SAFE_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_orig_load_state_dict = transformers.modeling_utils.load_state_dict


def _patched_load_state_dict(checkpoint_file, **kwargs):
    """
    Load safetensors checkpoint files via standard binary streaming I/O instead of mmap.
    This avoids OS Error 1455 (paging file limit) on Windows.
    """
    if isinstance(checkpoint_file, str) and checkpoint_file.endswith(".safetensors") and os.path.exists(checkpoint_file):
        map_location = kwargs.get("map_location", "cpu")
        with open(checkpoint_file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = json.loads(f.read(header_len).decode("utf-8"))
            header_end = 8 + header_len
            state_dict = {}
            for k, v in header_json.items():
                if k == "__metadata__":
                    continue
                dtype = _SAFE_DTYPE_MAP.get(v.get("dtype"), torch.float32)
                shape = v["shape"]
                if map_location == "meta":
                    state_dict[k] = torch.empty(shape, dtype=dtype, device="meta")
                else:
                    start, end = v["data_offsets"]
                    f.seek(header_end + start)
                    t = torch.empty(shape, dtype=dtype, device="cpu")
                    f.readinto(t.view(torch.uint8).numpy())
                    state_dict[k] = t
            return state_dict
    return _orig_load_state_dict(checkpoint_file, **kwargs)


# Apply patches to transformers to ensure smooth local loading on Windows
transformers.modeling_utils.load_state_dict = _patched_load_state_dict
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


class QwenRuntime:
    """
    Shared Qwen2.5-VL runtime.

    The model and processor are loaded once so that
    vision processing and answer generation reuse the
    same model instance.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ) -> None:
        # Check for local model directory in the project first
        local_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "Qwen2.5-VL-3B-Instruct",
        )
        if os.path.exists(os.path.join(local_dir, "config.json")):
            self.model_id = local_dir
            print(f"Using local project model directory: {self.model_id}")
        else:
            self.model_id = model_id
            print(f"Loading Qwen runtime: {model_id}")

        # ========================================================
        # DEVICE SELECTION
        # ========================================================
        from lecture_memory.config.settings import settings as _settings
        if _settings.use_cpu:
            # Force CPU execution
            self.device = "cpu"
            print("CPU mode enabled via settings; loading model on CPU.")
        else:
            # GPU mode: ensure CUDA is available and report the device.
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. Qwen2.5-VL is configured to run on the GPU."
                )
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            # No need to set a read‑only property; the model's device will be used later.


        # ========================================================
        # 4-BIT QUANTIZATION
        # ========================================================

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # ========================================================
        # PROCESSOR
        # ========================================================

        print(f"Loading Qwen processor from: {self.model_id}...")

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                local_files_only=True,
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
            )

        print("Processor loaded.")

        # ========================================================
        # MODEL
        # ========================================================

        print(f"Loading Qwen model in 4-bit from: {self.model_id}...")

        is_local = os.path.isdir(self.model_id)
        # Determine device_map based on settings
        device_map = {"": 0} if not _settings.use_cpu else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=is_local,
        )

        # ========================================================
        # INFERENCE MODE
        # ========================================================

        self.model.eval()

        print("Qwen runtime loaded successfully.")

        # ========================================================
        # DIAGNOSTICS
        # ========================================================

        print(
            f"Qwen primary device: {self.model.device}"
        )

        allocated_gb = (
            torch.cuda.memory_allocated() / (1024 ** 3)
        )

        reserved_gb = (
            torch.cuda.memory_reserved() / (1024 ** 3)
        )

        total_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"CUDA memory allocated: "
            f"{allocated_gb:.2f} GB"
        )

        print(
            f"CUDA memory reserved: "
            f"{reserved_gb:.2f} GB"
        )

        print(
            f"CUDA total VRAM: "
            f"{total_gb:.2f} GB"
        )

    @property
    def device(self):
        """
        Return the model's primary device.
        """
        return self.model.device