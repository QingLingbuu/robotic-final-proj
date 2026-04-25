"""Helpers for building and validating the detected_objects schema."""

import time

MAX_OBSTACLES = 16
VALID_STATUS = {"ready", "processing", "error"}


def _normalize_position(position):
    if position is None or len(position) != 3:
        raise ValueError("Position must contain exactly 3 coordinates.")
    return [float(value) for value in position]


def build_detected_objects(
    target_label,
    target_pos,
    target_conf,
    obstacles=None,
    status="ready",
    conf_thresh=0.0,
):
    """Build a detected_objects payload that matches the required schema."""
    if status not in VALID_STATUS:
        raise ValueError(f"Invalid status: {status}")

    ranked_obstacles = sorted(
        obstacles or [],
        key=lambda obstacle: float(obstacle.get("conf", 0.0)),
        reverse=True,
    )[:MAX_OBSTACLES]

    normalized_obstacles = [
        {
            "label": str(obstacle["label"]),
            "pos": _normalize_position(obstacle["pos"]),
            "id": int(obstacle["id"]),
        }
        for obstacle in ranked_obstacles
    ]

    normalized_status = status
    if float(target_conf) < float(conf_thresh):
        normalized_status = "error"

    return {
        "target": {
            "label": str(target_label),
            "pos": _normalize_position(target_pos),
            "conf": float(target_conf),
            "timestamp": time.monotonic(),
        },
        "obstacles": normalized_obstacles,
        "status": normalized_status,
    }


def validate_detected_objects(detected_objects):
    """Validate the queue payload against the required schema."""
    if not isinstance(detected_objects, dict):
        return False

    target = detected_objects.get("target")
    obstacles = detected_objects.get("obstacles")
    status = detected_objects.get("status")

    if not isinstance(target, dict) or not isinstance(obstacles, list):
        return False
    if status not in VALID_STATUS:
        return False
    if set(target.keys()) != {"label", "pos", "conf", "timestamp"}:
        return False
    if len(target["pos"]) != 3 or len(obstacles) > MAX_OBSTACLES:
        return False

    for obstacle in obstacles:
        if set(obstacle.keys()) != {"label", "pos", "id"}:
            return False
        if len(obstacle["pos"]) != 3:
            return False

    return True
