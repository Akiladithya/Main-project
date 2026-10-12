"""
download_model.py
-----------------
Downloads the Qwen2.5-VL-3B-Instruct model directly into the local
project folder on D: drive (d:\Mainproject\...\models\Qwen2.5-VL-3B-Instruct).

Benefits:
- Saves precious OS drive (C:) space (which only has ~2.4 GB free).
- Stores all 7.5 GB of weights on D: drive (which has >53 GB free).
- Enables 100% offline, self-contained local execution.
"""

import os
import sys

# Generous download timeout for large weights
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

from huggingface_hub import snapshot_download

def main():
    repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    target_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "Qwen2.5-VL-3B-Instruct",
    )
    os.makedirs(target_dir, exist_ok=True)

    print(f"============================================================")
    print(f"Downloading {repo_id}")
    print(f"Target folder: {target_dir}")
    print(f"Drive: D: (High capacity)")
    print(f"============================================================")
    print("This will resume any existing downloads directly into the local folder.")

    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            resume_download=True,
            etag_timeout=120,
            max_workers=2,  # Prevent socket exhaustion on Windows
        )
        print(f"\n✅ Model is fully downloaded and ready locally at:\n{path}")
    except Exception as e:
        print(f"\n❌ Download error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
