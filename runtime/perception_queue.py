"""Queue-based IPC helpers for perception-to-logic handoff."""

from multiprocessing import Queue
from queue import Empty

PERCEPTION_QUEUE_NAME = "perception_queue"
PERCEPTION_QUEUE_TIMEOUT_SEC = 0.2


def create_perception_queue() -> Queue:
    """Create the required multiprocessing queue."""
    return Queue()


def publish_detected_objects(perception_queue: Queue, detected_objects) -> None:
    """Write a detected_objects payload to the queue."""
    perception_queue.put(detected_objects)


def read_detected_objects(
    perception_queue: Queue, timeout: float = PERCEPTION_QUEUE_TIMEOUT_SEC
):
    """Read a detected_objects payload or raise queue.Empty on stale data."""
    return perception_queue.get(timeout=timeout)

