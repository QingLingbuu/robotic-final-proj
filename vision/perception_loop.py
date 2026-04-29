"""Perception loop that turns RGB-D observations into detected_objects payloads."""

import threading
import time

import numpy as np

from runtime.perception_queue import publish_detected_objects
from vision.coord_transform import camera_to_world, pixel_to_3d
from vision.detected_objects import MAX_OBSTACLES, build_detected_objects
from vision.detector import build_detector


def _sample_depth_at_center(depth_image, center_u, center_v, radius):
    """Estimate a stable depth value around a box center using a small local window."""
    height, width = depth_image.shape[:2]
    u = int(np.clip(round(center_u), 0, width - 1))
    v = int(np.clip(round(center_v), 0, height - 1))
    radius = max(int(radius), 0)

    u_min = max(u - radius, 0)
    u_max = min(u + radius + 1, width)
    v_min = max(v - radius, 0)
    v_max = min(v + radius + 1, height)

    patch = np.asarray(depth_image[v_min:v_max, u_min:u_max], dtype=float).reshape(-1)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _bbox_center(box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    return (float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0


class VisionPerceptionLoop:
    """Run text-conditioned detection and publish queue payloads."""

    def __init__(self, vision_config, camera_config, conf_thresh):
        self.vision_config = vision_config
        self.camera_config = camera_config
        self.conf_thresh = float(conf_thresh)
        self.detector = build_detector(vision_config)
        self.target_labels = [str(label).lower() for label in vision_config["target_labels"]]
        self.obstacle_labels = [
            str(label).lower() for label in vision_config.get("obstacle_labels", [])
        ]
        self.depth_sample_radius = int(vision_config.get("depth_sample_radius", 2))
        self.max_obstacles = min(
            int(vision_config.get("max_obstacles", MAX_OBSTACLES)),
            MAX_OBSTACLES,
        )

    def _detection_to_world_pos(self, detection, depth_image):
        fx = float(self.camera_config["fx"])
        fy = float(self.camera_config["fy"])
        cx = float(self.camera_config["cx"])
        cy = float(self.camera_config["cy"])
        transform = self.camera_config["T_world_cam"]
        rotation = np.array(transform["rotation"], dtype=float)
        translation = np.array(transform["translation"], dtype=float)

        center_u, center_v = _bbox_center(detection.box_xyxy)
        depth = _sample_depth_at_center(
            depth_image,
            center_u=center_u,
            center_v=center_v,
            radius=self.depth_sample_radius,
        )
        if depth is None:
            return None

        point_cam = pixel_to_3d(center_u, center_v, depth, fx, fy, cx, cy)
        point_world = camera_to_world(point_cam, rotation, translation)
        return [float(value) for value in point_world.tolist()]

    def infer_detected_objects(self, rgb_image, depth_image):
        """Infer target and obstacles from one RGB-D frame."""
        candidate_labels = self.target_labels + self.obstacle_labels
        detections = self.detector.detect(rgb_image, candidate_labels)

        target_candidates = []
        obstacle_candidates = []
        for detection in detections:
            world_pos = self._detection_to_world_pos(detection, depth_image)
            if world_pos is None:
                continue

            item = {
                "label": detection.label,
                "pos": world_pos,
                "conf": float(detection.score),
            }
            if detection.label in self.target_labels:
                target_candidates.append(item)
            elif detection.label in self.obstacle_labels:
                obstacle_candidates.append(item)

        if not target_candidates:
            return build_detected_objects(
                target_label="unknown",
                target_pos=[0.0, 0.0, 0.0],
                target_conf=0.0,
                obstacles=[],
                status="error",
                conf_thresh=self.conf_thresh,
            )

        target_candidates.sort(key=lambda item: item["conf"], reverse=True)
        obstacle_candidates.sort(key=lambda item: item["conf"], reverse=True)
        target = target_candidates[0]

        obstacles = []
        for index, obstacle in enumerate(obstacle_candidates[: self.max_obstacles], start=1):
            obstacles.append(
                {
                    "label": obstacle["label"],
                    "pos": obstacle["pos"],
                    "id": index,
                    "conf": obstacle["conf"],
                }
            )

        return build_detected_objects(
            target_label=target["label"],
            target_pos=target["pos"],
            target_conf=target["conf"],
            obstacles=obstacles,
            status="ready",
            conf_thresh=self.conf_thresh,
        )

    def publish_from_observation(self, perception_queue, rgb_image, depth_image):
        """Run one perception pass and publish the standardized payload."""
        detected_objects = self.infer_detected_objects(rgb_image, depth_image)
        publish_detected_objects(perception_queue, detected_objects)
        return detected_objects


class VisionPerceptionWorker:
    """Background publisher that continuously pushes RGB-D detections into the queue."""

    def __init__(self, env, perception_loop, perception_queue, publish_period_sec=0.2):
        self.env = env
        self.perception_loop = perception_loop
        self.perception_queue = perception_queue
        self.publish_period_sec = float(publish_period_sec)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._latest_detection = None
        self._latest_error = None

    def start(self):
        """Start the background publisher once."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="vision-perception-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the background publisher and wait briefly for shutdown."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.publish_period_sec * 2.0))

    def get_latest_detection(self):
        """Return the latest published payload, if any."""
        with self._lock:
            return self._latest_detection

    def get_latest_error(self):
        """Return the latest worker error, if any."""
        with self._lock:
            return self._latest_error

    def wait_for_initial_detection(self, timeout_sec):
        """Wait for the first successful payload or raise the latest worker error."""
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_error is not None:
                    raise RuntimeError(
                        f"Vision worker failed before first payload: {self._latest_error}"
                    ) from self._latest_error
                if self._latest_detection is not None:
                    return self._latest_detection
            time.sleep(min(self.publish_period_sec, 0.1))
        return None

    def _run(self):
        while not self._stop_event.is_set():
            try:
                rgb, depth, _ = self.env.get_observation()
                detected_objects = self.perception_loop.publish_from_observation(
                    perception_queue=self.perception_queue,
                    rgb_image=rgb,
                    depth_image=depth,
                )
                with self._lock:
                    self._latest_detection = detected_objects
                    self._latest_error = None
            except Exception as exc:
                with self._lock:
                    self._latest_error = exc
            finally:
                self._stop_event.wait(self.publish_period_sec)
