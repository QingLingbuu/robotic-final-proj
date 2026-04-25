"""Minimal perception producer for queue-driven integration tests."""

from ipc.perception_queue import publish_detected_objects
from vision.detected_objects import build_detected_objects


def make_demo_detection(conf_thresh, target_conf=0.95, obstacle_count=1):
    """Create a minimal detected_objects payload for integration wiring."""
    obstacles = [
        {
            "label": f"demo_obstacle_{index}",
            "pos": [0.2 + 0.01 * index, 0.0, 0.4],
            "id": index + 1,
            "conf": max(0.0, 0.8 - 0.01 * index),
        }
        for index in range(obstacle_count)
    ]
    return build_detected_objects(
        target_label="demo_target",
        target_pos=[0.0, 0.0, 0.5],
        target_conf=target_conf,
        obstacles=obstacles,
        status="ready",
        conf_thresh=conf_thresh,
    )


def publish_demo_detection(
    perception_queue, conf_thresh, target_conf=0.95, obstacle_count=1
):
    """Publish a generated detected_objects payload to the shared queue."""
    detected_objects = make_demo_detection(
        conf_thresh=conf_thresh,
        target_conf=target_conf,
        obstacle_count=obstacle_count,
    )
    publish_detected_objects(perception_queue, detected_objects)
    return detected_objects
