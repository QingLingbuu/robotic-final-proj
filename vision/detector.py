"""Detector backends for text-conditioned object detection."""

from dataclasses import dataclass
from importlib import metadata
import inspect

import numpy as np


@dataclass
class Detection:
    """Single 2D detection result."""

    label: str
    score: float
    box_xyxy: list[float]


class GroundingDinoDetector:
    """Grounding DINO detector wrapper built on Hugging Face Transformers."""

    def __init__(
        self,
        model_id,
        device="cpu",
        box_threshold=0.35,
        text_threshold=0.25,
        local_files_only=False,
    ):
        self._validate_runtime_dependencies()
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Grounding DINO backend imports failed after dependency checks. "
                "Reinstall torch, pillow, and transformers in the active environment."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=bool(local_files_only),
            )
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                local_files_only=bool(local_files_only),
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    "Grounding DINO model cache is missing or incomplete. "
                    "Run `python scripts/cache_grounding_dino.py` once, then retry."
                ) from exc
            raise
        self._device = device
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)

        if device != "cpu":
            self._model = self._model.to(device)
        self._model.eval()

    @staticmethod
    def _parse_version(version_text):
        parts = []
        for segment in str(version_text).split("."):
            digits = "".join(char for char in segment if char.isdigit())
            if not digits:
                break
            parts.append(int(digits))
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    @classmethod
    def _require_package(cls, package_name, install_hint):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError as exc:
            raise ImportError(
                f"Grounding DINO backend requires `{package_name}`. Install it with: {install_hint}"
            ) from exc

    @classmethod
    def _validate_runtime_dependencies(cls):
        cls._require_package("torch", "pip install torch")
        cls._require_package("Pillow", "pip install pillow")
        transformers_version = cls._require_package(
            "transformers",
            "pip install transformers",
        )

        min_version = (4, 41, 0)
        parsed_version = cls._parse_version(transformers_version)
        if parsed_version < min_version:
            raise ImportError(
                "Grounding DINO backend requires `transformers>=4.41.0` for "
                "zero-shot object detection support. "
                f"Current version: {transformers_version}. "
                "Upgrade with: pip install -U \"transformers>=4.41.0\""
            )

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Installed `transformers` does not expose Grounding DINO zero-shot "
                "object detection APIs. "
                f"Detected version: {transformers_version}. "
                "Upgrade with: pip install -U \"transformers>=4.41.0\""
            ) from exc

    def detect(self, rgb_image, labels):
        """Run text-conditioned detection and return normalized Detection items."""
        if rgb_image is None:
            raise ValueError("rgb_image is required for detection.")
        normalized_labels = [str(label).strip().lower() for label in labels if str(label).strip()]
        if not normalized_labels:
            return []

        text_prompt = ". ".join(normalized_labels) + "."
        image = self._image_cls.fromarray(np.asarray(rgb_image, dtype=np.uint8))
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        post_process_kwargs = {
            "outputs": outputs,
            "input_ids": inputs["input_ids"],
            "text_threshold": self._text_threshold,
            "target_sizes": [image.size[::-1]],
        }
        signature = inspect.signature(
            self._processor.post_process_grounded_object_detection
        )
        if "box_threshold" in signature.parameters:
            post_process_kwargs["box_threshold"] = self._box_threshold
        else:
            post_process_kwargs["threshold"] = self._box_threshold

        results = self._processor.post_process_grounded_object_detection(
            **post_process_kwargs
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            box_values = box.detach().cpu().tolist()
            detections.append(
                Detection(
                    label=str(label).strip().lower(),
                    score=float(score),
                    box_xyxy=[float(value) for value in box_values],
                )
            )
        return detections


def build_detector(config):
    """Build a detector backend from config."""
    backend = str(config.get("backend", "grounding_dino")).strip().lower()
    if backend != "grounding_dino":
        raise ValueError(f"Unsupported vision backend: {backend}")

    return GroundingDinoDetector(
        model_id=config["model_id"],
        device=config.get("device", "cpu"),
        box_threshold=config.get("box_threshold", 0.35),
        text_threshold=config.get("text_threshold", 0.25),
        local_files_only=config.get("local_files_only", False),
    )
