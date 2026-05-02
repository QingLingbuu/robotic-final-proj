"""Small image IO helpers for vision demo scripts."""

from pathlib import Path

import numpy as np


def save_rgb_frame(rgb_image, output_path):
    """Save an RGB uint8 frame to disk."""
    from PIL import Image

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(rgb_image)
    if frame.dtype.kind == "f":
        max_value = float(np.nanmax(frame)) if frame.size else 0.0
        if max_value <= 1.0:
            frame = np.clip(frame * 255.0, 0.0, 255.0)
        else:
            frame = np.clip(frame, 0.0, 255.0)
    frame = np.asarray(frame, dtype=np.uint8)
    Image.fromarray(frame).save(path)
    return str(path)
