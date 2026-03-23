#!/usr/bin/env python3
"""Download Qwen3-TTS model weights from HuggingFace Hub.

Usage:
    python qwen3tts/setup/download_models.py
    python qwen3tts/setup/download_models.py --model-dir ~/models/Qwen3-TTS
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_model(model_dir: str, repo_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice") -> None:
    from huggingface_hub import snapshot_download

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} → {model_path} ...", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(model_path))
    print(f"Done. Model saved to {model_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS model")
    parser.add_argument("--model-dir", default=str(Path.home() / "models" / "Qwen3-TTS"))
    parser.add_argument("--repo-id",   default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        help="HuggingFace repo ID. Options: Qwen/Qwen3-TTS-12Hz-1.7B-Base, "
                             "Qwen/Qwen3-TTS-12Hz-0.6B-Base, Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    args = parser.parse_args()
    download_model(args.model_dir, args.repo_id)


if __name__ == "__main__":
    main()
