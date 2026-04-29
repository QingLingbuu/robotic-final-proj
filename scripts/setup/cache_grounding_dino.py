"""Download and verify the configured Grounding DINO model cache."""

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_CONFIG = "configs/vision.yaml"


def load_vision_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def cache_model(model_id, force_download=False):
    print(f"Caching model: {model_id}")
    cache_path = snapshot_download(
        repo_id=model_id,
        force_download=force_download,
        local_files_only=False,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ],
    )
    print(f"Snapshot path: {cache_path}")
    return cache_path


def verify_local_load(model_id):
    print("Verifying local-only load...")
    AutoProcessor.from_pretrained(model_id, local_files_only=True)
    AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        local_files_only=True,
    )
    print("Local-only Grounding DINO load succeeded.")


def main():
    parser = argparse.ArgumentParser(
        description="Cache the Grounding DINO model used by configs/vision.yaml."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to vision YAML config.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model_id from the vision config.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading model files.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    vision_config = load_vision_config(config_path)
    model_id = args.model_id or vision_config["model_id"]

    cache_model(model_id=model_id, force_download=args.force_download)
    verify_local_load(model_id=model_id)


if __name__ == "__main__":
    main()

