"""Perception loop that turns RGB-D observations into detected_objects payloads."""

import threading
import time

import numpy as np

from runtime.perception_queue import publish_detected_objects
from vision.coord_transform import camera_to_world, pixel_to_3d, pixels_to_world
from vision.detected_objects import MAX_OBSTACLES, build_detected_objects
from vision.detector import build_detector


DEFAULT_TOP_DOWN_ORIENTATION = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


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


def _clip_bbox(box_xyxy, image_shape):
    height, width = image_shape[:2]
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    u_min = max(int(np.floor(min(x0, x1))), 0)
    u_max = min(int(np.ceil(max(x0, x1))), width - 1)
    v_min = max(int(np.floor(min(y0, y1))), 0)
    v_max = min(int(np.ceil(max(y0, y1))), height - 1)
    return u_min, u_max, v_min, v_max


def _quantile_bounds(values, low_q, high_q):
    if values.size == 0:
        return None, None
    return float(np.quantile(values, low_q)), float(np.quantile(values, high_q))


def _normalize_vector(vector):
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return np.asarray(vector, dtype=float) / norm


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
        self.candidate_types = [
            str(candidate_type).strip().lower()
            for candidate_type in vision_config.get("candidate_types", ["top_down"])
            if str(candidate_type).strip()
        ]
        self.default_gripper_width = float(vision_config.get("default_gripper_width", 0.04))
        self.top_down_surface_quantile = float(vision_config.get("top_down_surface_quantile", 0.85))
        self.top_down_penetration_offset = float(vision_config.get("top_down_penetration_offset", 0.01))
        self.top_down_width_margin = float(vision_config.get("top_down_width_margin", 0.01))
        self.top_down_width_min = float(vision_config.get("top_down_width_min", self.default_gripper_width))
        self.handle_gripper_width = float(
            vision_config.get("handle_gripper_width", self.default_gripper_width)
        )
        self.handle_width_margin = float(vision_config.get("handle_width_margin", 0.01))
        self.handle_width_min = float(vision_config.get("handle_width_min", 0.02))
        self.handle_labels = {
            str(label).strip().lower()
            for label in vision_config.get("handle_labels", ["cup", "mug"])
        }
        self.handle_min_points = int(vision_config.get("handle_min_points", 8))
        self.handle_radial_ratio = float(vision_config.get("handle_radial_ratio", 0.15))
        self.handle_radial_offset = float(vision_config.get("handle_radial_offset", 0.01))
        handle_height_quantiles = vision_config.get("handle_height_quantiles", [0.25, 0.85])
        self.handle_height_quantiles = (
            float(handle_height_quantiles[0]),
            float(handle_height_quantiles[1]),
        )

    def _detection_to_world_pos(self, detection, depth_image):
        center_u, center_v = _bbox_center(detection.box_xyxy)
        depth = _sample_depth_at_center(
            depth_image,
            center_u=center_u,
            center_v=center_v,
            radius=self.depth_sample_radius,
        )
        if depth is None:
            return None

        camera_to_world_transform = self.camera_config.get("camera_to_world_transform")
        if camera_to_world_transform is not None:
            world_point = pixels_to_world(
                pixels_rc=np.array([[center_v, center_u]], dtype=float),
                depth_image=depth_image,
                camera_to_world_transform=np.asarray(camera_to_world_transform, dtype=float),
            )[0]
            return [float(value) for value in np.asarray(world_point, dtype=float).tolist()]

        fx = float(self.camera_config["fx"])
        fy = float(self.camera_config["fy"])
        cx = float(self.camera_config["cx"])
        cy = float(self.camera_config["cy"])
        transform = self.camera_config["T_world_cam"]
        rotation = np.array(transform["rotation"], dtype=float)
        translation = np.array(transform["translation"], dtype=float)
        point_cam = pixel_to_3d(center_u, center_v, depth, fx, fy, cx, cy)
        point_world = camera_to_world(point_cam, rotation, translation)
        return [float(value) for value in point_world.tolist()]

    def _extract_bbox_point_cloud(self, detection, depth_image):
        camera_to_world_transform = self.camera_config.get("camera_to_world_transform")
        if camera_to_world_transform is not None:
            u_min, u_max, v_min, v_max = _clip_bbox(detection.box_xyxy, depth_image.shape)
            pixels = []
            for v in range(v_min, v_max + 1):
                for u in range(u_min, u_max + 1):
                    depth = float(depth_image[v, u])
                    if not np.isfinite(depth) or depth <= 0.0:
                        continue
                    pixels.append([float(v), float(u)])
            if not pixels:
                return None
            return pixels_to_world(
                pixels_rc=np.asarray(pixels, dtype=float),
                depth_image=depth_image,
                camera_to_world_transform=np.asarray(camera_to_world_transform, dtype=float),
            )

        fx = float(self.camera_config["fx"])
        fy = float(self.camera_config["fy"])
        cx = float(self.camera_config["cx"])
        cy = float(self.camera_config["cy"])
        transform = self.camera_config["T_world_cam"]
        rotation = np.array(transform["rotation"], dtype=float)
        translation = np.array(transform["translation"], dtype=float)
        u_min, u_max, v_min, v_max = _clip_bbox(detection.box_xyxy, depth_image.shape)

        points = []
        for v in range(v_min, v_max + 1):
            for u in range(u_min, u_max + 1):
                depth = float(depth_image[v, u])
                if not np.isfinite(depth) or depth <= 0.0:
                    continue
                point_cam = pixel_to_3d(float(u), float(v), depth, fx, fy, cx, cy)
                point_world = camera_to_world(point_cam, rotation, translation)
                points.append(point_world)
        if not points:
            return None
        return np.asarray(points, dtype=float)

    def _estimate_top_down_candidate(self, target, point_cloud, candidate_id):
        if point_cloud is None or point_cloud.shape[0] == 0:
            return {
                "id": candidate_id,
                "pos": list(target["pos"]),
                "orientation": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                "gripper_width": self.default_gripper_width,
                "score": float(target["conf"]),
                "grasp_type": "top_down",
            }

        z_values = point_cloud[:, 2]
        z_top = float(np.max(z_values))
        z_threshold = float(np.quantile(z_values, self.top_down_surface_quantile))
        top_points = point_cloud[z_values >= z_threshold]
        if top_points.shape[0] == 0:
            top_points = point_cloud
        top_center = np.mean(top_points[:, :2], axis=0)
        top_down_pos = [
            float(top_center[0]),
            float(top_center[1]),
            float(z_top - self.top_down_penetration_offset),
        ]
        top_xy = top_points[:, :2]
        xy_span = np.max(top_xy, axis=0) - np.min(top_xy, axis=0)
        estimated_width = max(self.top_down_width_min, float(np.min(xy_span)) + self.top_down_width_margin)
        top_ratio = top_points.shape[0] / point_cloud.shape[0]
        z_band = max(z_top - z_threshold, 1e-6)
        top_surface_quality = min(1.0, z_band / max(z_top, 1e-6))
        score = min(1.0, float(target["conf"]) * (0.45 + 0.30 * top_ratio + 0.25 * top_surface_quality))
        return {
            "id": candidate_id,
            "pos": top_down_pos,
            "orientation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            "gripper_width": estimated_width,
            "score": score,
            "grasp_type": "top_down",
        }

    def _estimate_handle_candidate(self, target, point_cloud, candidate_id):
        if point_cloud is None or point_cloud.shape[0] < self.handle_min_points:
            return None

        xy = point_cloud[:, :2]
        z = point_cloud[:, 2]
        center_xy = np.median(xy, axis=0)
        radial = np.linalg.norm(xy - center_xy, axis=1)
        if radial.size == 0:
            return None

        radial_body = float(np.median(radial))
        radial_threshold = radial_body + max(
            self.handle_radial_offset,
            self.handle_radial_ratio * max(radial_body, 1e-6),
        )
        z_low, z_high = _quantile_bounds(z, *self.handle_height_quantiles)
        if z_low is None or z_high is None:
            return None

        handle_mask = (radial > radial_threshold) & (z >= z_low) & (z <= z_high)
        handle_points = point_cloud[handle_mask]
        if handle_points.shape[0] < self.handle_min_points:
            return None

        handle_center = np.mean(handle_points, axis=0)
        outward_xy = handle_center[:2] - center_xy
        approach_axis = _normalize_vector(np.array([-outward_xy[0], -outward_xy[1], 0.0], dtype=float))
        if approach_axis is None:
            return None
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        closing_axis = _normalize_vector(np.cross(vertical_axis, approach_axis))
        if closing_axis is None:
            return None
        lateral_axis = _normalize_vector(np.cross(approach_axis, closing_axis))
        if lateral_axis is None:
            return None
        closing_projection = handle_points @ closing_axis
        closing_span = float(np.max(closing_projection) - np.min(closing_projection))
        estimated_width = max(self.handle_width_min, closing_span + self.handle_width_margin)
        handle_ratio = handle_points.shape[0] / point_cloud.shape[0]
        mean_handle_radial = float(np.mean(radial[handle_mask]))
        protrusion_strength = max(0.0, mean_handle_radial - radial_body)
        protrusion_quality = protrusion_strength / (protrusion_strength + radial_body + 1e-6)
        score = min(
            1.0,
            float(target["conf"]) * (0.35 + 0.35 * handle_ratio + 0.30 * protrusion_quality),
        )
        return {
            "id": candidate_id,
            "pos": [float(value) for value in handle_center.tolist()],
            "orientation": [
                [float(value) for value in closing_axis.tolist()],
                [float(value) for value in lateral_axis.tolist()],
                [float(value) for value in approach_axis.tolist()],
            ],
            "gripper_width": estimated_width,
            "score": score,
            "grasp_type": "handle_grasp",
        }

    def _build_grasp_candidates(self, target, point_cloud=None):
        candidate_types = getattr(self, "candidate_types", None)
        if candidate_types is None:
            candidate_types = [
                str(candidate_type).strip().lower()
                for candidate_type in self.vision_config.get("candidate_types", ["top_down"])
                if str(candidate_type).strip()
            ]
        default_gripper_width = float(
            getattr(
                self,
                "default_gripper_width",
                self.vision_config.get("default_gripper_width", 0.04),
            )
        )
        candidates = []
        next_id = 1
        for grasp_type in candidate_types:
            if grasp_type == "top_down":
                candidates.append(self._estimate_top_down_candidate(target, point_cloud, next_id))
                next_id += 1
                continue
            if grasp_type == "handle_grasp" and target["label"] in self.handle_labels:
                handle_candidate = self._estimate_handle_candidate(target, point_cloud, next_id)
                if handle_candidate is not None:
                    candidates.append(handle_candidate)
                    next_id += 1
        return candidates

    def infer_detected_objects(self, rgb_image, depth_image):
        """Infer target and obstacles from one RGB-D frame."""
        candidate_labels = self.target_labels + self.obstacle_labels
        detections = self.detector.detect(rgb_image, candidate_labels)

        target_candidates = []
        obstacle_candidates = []
        detection_point_clouds = {}
        for detection in detections:
            world_pos = self._detection_to_world_pos(detection, depth_image)
            if world_pos is None:
                continue

            point_cloud = self._extract_bbox_point_cloud(detection, depth_image)
            detection_label = str(detection.label).strip().lower()
            item = {
                "label": detection_label,
                "pos": world_pos,
                "conf": float(detection.score),
            }
            if detection_label in self.target_labels:
                target_candidates.append(item)
                detection_point_clouds[id(item)] = point_cloud
            elif detection_label in self.obstacle_labels:
                obstacle_candidates.append(item)

        if not target_candidates:
            return build_detected_objects(
                target_label="unknown",
                target_pos=[0.0, 0.0, 0.0],
                target_conf=0.0,
                obstacles=[],
                status="error",
                conf_thresh=self.conf_thresh,
                grasp_candidates=[],
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
            grasp_candidates=self._build_grasp_candidates(
                target,
                detection_point_clouds.get(id(target)),
            ),
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
