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


def _clip_bbox(box_xyxy, image_shape, padding_fraction=0.0):
    height, width = image_shape[:2]
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    x_min = min(x0, x1)
    x_max = max(x0, x1)
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    pad_x = (x_max - x_min) * float(padding_fraction)
    pad_y = (y_max - y_min) * float(padding_fraction)
    u_min = max(int(np.floor(x_min - pad_x)), 0)
    u_max = min(int(np.ceil(x_max + pad_x)), width - 1)
    v_min = max(int(np.floor(y_min - pad_y)), 0)
    v_max = min(int(np.ceil(y_max + pad_y)), height - 1)
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
        self.handle_top_down_inset = float(vision_config.get("handle_top_down_inset", 0.0))
        self.handle_top_down_surface_quantile = float(
            vision_config.get("handle_top_down_surface_quantile", 0.50)
        )
        self.handle_top_down_penetration_offset = float(
            vision_config.get("handle_top_down_penetration_offset", 0.012)
        )
        self.handle_top_down_width_margin = float(
            vision_config.get("handle_top_down_width_margin", 0.008)
        )
        self.handle_top_down_width_min = float(vision_config.get("handle_top_down_width_min", 0.015))
        self.handle_top_down_width_max = float(vision_config.get("handle_top_down_width_max", 0.08))
        self.handle_labels = {
            str(label).strip().lower()
            for label in vision_config.get("handle_labels", ["cup", "mug"])
        }
        self.handle_min_points = int(vision_config.get("handle_min_points", 8))
        self.handle_radial_ratio = float(vision_config.get("handle_radial_ratio", 0.15))
        self.handle_radial_offset = float(vision_config.get("handle_radial_offset", 0.01))
        self.handle_side_band_fraction = float(vision_config.get("handle_side_band_fraction", 0.35))
        self.handle_bbox_padding_fraction = float(vision_config.get("handle_bbox_padding_fraction", 0.15))
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

    def _bbox_padding_fraction_for_detection(self, detection):
        label = str(detection.label).strip().lower()
        if label in self.handle_labels:
            return self.handle_bbox_padding_fraction
        return 0.0

    def _bbox_diagnostics(self, detection, depth_image):
        raw = _clip_bbox(detection.box_xyxy, depth_image.shape, padding_fraction=0.0)
        padded = _clip_bbox(
            detection.box_xyxy,
            depth_image.shape,
            padding_fraction=self._bbox_padding_fraction_for_detection(detection),
        )
        return {
            "raw_xyxy": [int(raw[0]), int(raw[2]), int(raw[1]), int(raw[3])],
            "padded_xyxy": [int(padded[0]), int(padded[2]), int(padded[1]), int(padded[3])],
            "padding_fraction": float(self._bbox_padding_fraction_for_detection(detection)),
            "image_shape_hw": [int(depth_image.shape[0]), int(depth_image.shape[1])],
        }

    def _extract_bbox_point_cloud(self, detection, depth_image, side_band=None):
        camera_to_world_transform = self.camera_config.get("camera_to_world_transform")
        if camera_to_world_transform is not None:
            u_min, u_max, v_min, v_max = _clip_bbox(
                detection.box_xyxy,
                depth_image.shape,
                padding_fraction=self._bbox_padding_fraction_for_detection(detection),
            )
            if side_band is not None:
                band_u_min, band_u_max = self._bbox_side_band_u_range(u_min, u_max, side_band)
                u_min, u_max = band_u_min, band_u_max
            pixels = []
            for v, u in self._foreground_depth_pixels_in_bbox(depth_image, u_min, u_max, v_min, v_max):
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
        u_min, u_max, v_min, v_max = _clip_bbox(
            detection.box_xyxy,
            depth_image.shape,
            padding_fraction=self._bbox_padding_fraction_for_detection(detection),
        )
        if side_band is not None:
            band_u_min, band_u_max = self._bbox_side_band_u_range(u_min, u_max, side_band)
            u_min, u_max = band_u_min, band_u_max

        points = []
        for v, u in self._foreground_depth_pixels_in_bbox(depth_image, u_min, u_max, v_min, v_max):
            depth = float(depth_image[v, u])
            point_cam = pixel_to_3d(float(u), float(v), depth, fx, fy, cx, cy)
            point_world = camera_to_world(point_cam, rotation, translation)
            points.append(point_world)
        if not points:
            return None
        return np.asarray(points, dtype=float)

    def _bbox_side_band_u_range(self, u_min, u_max, side_band):
        width = int(u_max) - int(u_min) + 1
        band_width = max(1, int(round(width * float(self.handle_side_band_fraction))))
        if side_band == "left":
            return int(u_min), min(int(u_max), int(u_min) + band_width - 1)
        if side_band == "right":
            return max(int(u_min), int(u_max) - band_width + 1), int(u_max)
        raise ValueError(f"Unknown side band: {side_band}")

    def _foreground_depth_pixels_in_bbox(self, depth_image, u_min, u_max, v_min, v_max):
        crop = np.asarray(depth_image[v_min : v_max + 1, u_min : u_max + 1], dtype=float)
        valid = np.isfinite(crop) & (crop > 0.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            return []

        height, width = crop.shape
        center_u = (width - 1) / 2.0
        center_v = (height - 1) / 2.0
        valid_rc = np.argwhere(valid)
        seed_idx = int(
            np.argmin((valid_rc[:, 1] - center_u) ** 2 + (valid_rc[:, 0] - center_v) ** 2)
        )
        seed_v, seed_u = [int(value) for value in valid_rc[seed_idx]]
        seed_depth = float(crop[seed_v, seed_u])

        valid_depths = crop[valid]
        depth_iqr = float(np.quantile(valid_depths, 0.75) - np.quantile(valid_depths, 0.25))
        depth_jump = max(0.025, min(0.060, 0.35 * depth_iqr))
        seed_window = max(0.060, min(0.140, 3.0 * depth_jump))

        component = np.zeros_like(valid, dtype=bool)
        stack = [(seed_v, seed_u)]
        component[seed_v, seed_u] = True
        while stack:
            cur_v, cur_u = stack.pop()
            cur_depth = float(crop[cur_v, cur_u])
            for next_v, next_u in (
                (cur_v - 1, cur_u),
                (cur_v + 1, cur_u),
                (cur_v, cur_u - 1),
                (cur_v, cur_u + 1),
            ):
                if next_v < 0 or next_v >= height or next_u < 0 or next_u >= width:
                    continue
                if component[next_v, next_u] or not valid[next_v, next_u]:
                    continue
                next_depth = float(crop[next_v, next_u])
                if abs(next_depth - cur_depth) > depth_jump:
                    continue
                if abs(next_depth - seed_depth) > seed_window:
                    continue
                component[next_v, next_u] = True
                stack.append((next_v, next_u))

        component_count = int(np.count_nonzero(component))
        if component_count < max(self.handle_min_points, int(0.10 * valid_count)):
            component = valid
            component_count = valid_count

        self._last_bbox_foreground_filter = {
            "valid_depth_count": valid_count,
            "foreground_depth_count": component_count,
            "depth_jump_threshold": float(depth_jump),
            "seed_depth": float(seed_depth),
        }
        return [
            (int(v_min + local_v), int(u_min + local_u))
            for local_v, local_u in np.argwhere(component)
        ]

    def _choose_handle_side_point_cloud(self, detection, depth_image, target):
        left_cloud = self._extract_bbox_point_cloud(detection, depth_image, side_band="left")
        right_cloud = self._extract_bbox_point_cloud(detection, depth_image, side_band="right")
        candidates = []
        target_pos = np.asarray(target.get("pos", [np.nan, np.nan, np.nan]), dtype=float)
        target_xy = target_pos[:2] if np.all(np.isfinite(target_pos[:2])) else None
        for side_name, side_cloud in [("left", left_cloud), ("right", right_cloud)]:
            if side_cloud is None or side_cloud.shape[0] < self.handle_min_points:
                continue
            center_xy = np.mean(side_cloud[:, :2], axis=0)
            if target_xy is None:
                score = float(side_cloud.shape[0])
            else:
                score = float(np.linalg.norm(center_xy - target_xy))
            candidates.append((score, side_name, side_cloud))
        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, side_name, side_cloud = candidates[0]
        return side_cloud, side_name

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
        if hasattr(self, "_grasp_candidate_diagnostics"):
            self._grasp_candidate_diagnostics[str(candidate_id)] = {
                "grasp_type": "handle_grasp",
                "point_cloud_count": int(point_cloud.shape[0]),
                "handle_point_count": int(handle_points.shape[0]),
                "handle_ratio": float(handle_ratio),
                "body_center_xy": [float(value) for value in center_xy.tolist()],
                "handle_center": [float(value) for value in handle_center.tolist()],
                "radial_body": float(radial_body),
                "radial_threshold": float(radial_threshold),
                "z_low": float(z_low),
                "z_high": float(z_high),
                "mean_handle_radial": float(mean_handle_radial),
                "protrusion_strength": float(protrusion_strength),
                "protrusion_quality": float(protrusion_quality),
                "closing_span": float(closing_span),
                "estimated_width": float(estimated_width),
                "approach_axis": [float(value) for value in approach_axis.tolist()],
                "closing_axis": [float(value) for value in closing_axis.tolist()],
                "lateral_axis": [float(value) for value in lateral_axis.tolist()],
            }
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

    def _estimate_handle_top_down_candidate(self, target, point_cloud, candidate_id, handle_point_cloud=None):
        if point_cloud is None or point_cloud.shape[0] < self.handle_min_points:
            return None

        candidate_cloud = point_cloud
        xy = point_cloud[:, :2]
        z = point_cloud[:, 2]
        target_pos = np.asarray(target.get("pos", [np.nan, np.nan, np.nan]), dtype=float)
        z_low, z_high = _quantile_bounds(z, *self.handle_height_quantiles)
        if z_low is None or z_high is None:
            return None

        z_band_mask = (z >= z_low) & (z <= z_high)
        z_band_points = point_cloud[z_band_mask]
        if z_band_points.shape[0] < self.handle_min_points:
            return None
        z_band_xy = z_band_points[:, :2]

        top_body_z = float(np.quantile(z, self.top_down_surface_quantile))
        top_body_points = point_cloud[z >= top_body_z]
        if top_body_points.shape[0] >= self.handle_min_points:
            coarse_center_xy = np.mean(top_body_points[:, :2], axis=0)
            body_center_source = "top_surface_mean"
        else:
            coarse_center_xy = np.median(z_band_xy, axis=0)
            body_center_source = "z_band_median"
        coarse_radial = np.linalg.norm(z_band_xy - coarse_center_xy, axis=1)
        body_fit_mask = coarse_radial <= float(np.quantile(coarse_radial, 0.55))
        body_fit_xy = z_band_xy[body_fit_mask]
        if body_fit_xy.shape[0] < self.handle_min_points:
            body_fit_xy = z_band_xy

        if top_body_points.shape[0] >= self.handle_min_points:
            body_center_xy = coarse_center_xy
        else:
            body_center_xy = np.median(body_fit_xy, axis=0)
        body_radial = np.linalg.norm(body_fit_xy - body_center_xy, axis=1)
        radial_body = float(np.quantile(body_radial, 0.75))
        radial = np.linalg.norm(xy - body_center_xy, axis=1)
        residual_margin = max(self.handle_radial_offset, 0.10 * radial_body)
        radial_threshold = max(0.02, 0.75 * radial_body + residual_margin)
        if z_band_points.shape[0] >= 100:
            max_handle_radial = radial_body + 0.065
        else:
            max_handle_radial = radial_body + 0.20
        handle_z_floor = float(z_low + 0.35 * (z_high - z_low))
        residual_mask = (
            z_band_mask
            & (z >= handle_z_floor)
            & (radial >= radial_threshold)
            & (radial <= max_handle_radial)
        )

        residual_points = point_cloud[residual_mask]
        if residual_points.shape[0] < self.handle_min_points:
            if hasattr(self, "_grasp_candidate_diagnostics"):
                self._grasp_candidate_diagnostics[str(candidate_id)] = {
                    "grasp_type": "handle_top_down",
                    "rejected": True,
                    "reject_reason": "insufficient_body_residual_points",
                    "point_cloud_count": int(point_cloud.shape[0]),
                    "candidate_point_cloud_count": int(candidate_cloud.shape[0]),
                    "handle_point_count": int(residual_points.shape[0]),
                    "body_center_xy": [float(value) for value in body_center_xy.tolist()],
                    "radial_body": float(radial_body),
                    "radial_threshold": float(radial_threshold),
                    "max_handle_radial": float(max_handle_radial),
                    "handle_z_floor": float(handle_z_floor),
                    "z_low": float(z_low),
                    "z_high": float(z_high),
                    "used_handle_side_cloud": False,
                    "handle_point_source": "body_model_residual_cluster",
                }
            return None

        residual_xy = residual_points[:, :2]
        residual_rel_xy = residual_xy - body_center_xy
        residual_angles = np.arctan2(residual_rel_xy[:, 1], residual_rel_xy[:, 0])
        residual_radial = np.linalg.norm(residual_rel_xy, axis=1)
        residual_z = residual_points[:, 2]
        sector_count = 48
        sector_width = 2.0 * np.pi / float(sector_count)
        min_sector_points = max(self.handle_min_points, int(0.10 * residual_points.shape[0]))
        best_sector = None
        for sector_idx in range(sector_count):
            sector_center = -np.pi + (sector_idx + 0.5) * sector_width
            angular_delta = np.arctan2(
                np.sin(residual_angles - sector_center),
                np.cos(residual_angles - sector_center),
            )
            sector_mask = np.abs(angular_delta) <= sector_width
            if np.count_nonzero(sector_mask) < min_sector_points:
                continue
            sector_mean_radial = float(np.mean(residual_radial[sector_mask]))
            sector_median_z = float(np.median(residual_z[sector_mask]))
            z_quality = (sector_median_z - handle_z_floor) / max(z_high - handle_z_floor, 1e-6)
            z_quality = float(np.clip(z_quality, 0.0, 1.0))
            count_quality = min(1.0, float(np.count_nonzero(sector_mask)) / max(residual_points.shape[0], 1))
            sector_score = float(sector_mean_radial * (0.70 + 0.30 * z_quality) + 0.015 * count_quality)
            if best_sector is None or sector_score > best_sector["score"]:
                best_sector = {
                    "center": float(sector_center),
                    "score": sector_score,
                    "mean_radial": sector_mean_radial,
                    "median_z": sector_median_z,
                    "z_quality": z_quality,
                    "mask": sector_mask,
                    "point_count": int(np.count_nonzero(sector_mask)),
                }
        if best_sector is None:
            if hasattr(self, "_grasp_candidate_diagnostics"):
                self._grasp_candidate_diagnostics[str(candidate_id)] = {
                    "grasp_type": "handle_top_down",
                    "rejected": True,
                    "reject_reason": "no_body_residual_angle_cluster",
                    "point_cloud_count": int(point_cloud.shape[0]),
                    "candidate_point_cloud_count": int(candidate_cloud.shape[0]),
                    "handle_point_count": int(residual_points.shape[0]),
                    "body_center_xy": [float(value) for value in body_center_xy.tolist()],
                    "radial_body": float(radial_body),
                    "radial_threshold": float(radial_threshold),
                    "max_handle_radial": float(max_handle_radial),
                    "handle_z_floor": float(handle_z_floor),
                    "z_low": float(z_low),
                    "z_high": float(z_high),
                    "used_handle_side_cloud": False,
                    "handle_point_source": "body_model_residual_cluster",
                }
            return None

        handle_points = residual_points[best_sector["mask"]]
        handle_mask = residual_mask
        handle_point_source = "body_model_residual_cluster"
        radial_center_xy = body_center_xy
        body_center_source = f"body_model_inlier_median_from_{body_center_source}"
        if handle_points.shape[0] < self.handle_min_points:
            if hasattr(self, "_grasp_candidate_diagnostics"):
                self._grasp_candidate_diagnostics[str(candidate_id)] = {
                    "grasp_type": "handle_top_down",
                    "rejected": True,
                    "reject_reason": "insufficient_handle_points",
                    "point_cloud_count": int(point_cloud.shape[0]),
                    "candidate_point_cloud_count": int(candidate_cloud.shape[0]),
                    "handle_point_count": int(handle_points.shape[0]),
                    "radial_center_xy": [float(value) for value in radial_center_xy.tolist()],
                    "body_center_xy": [float(value) for value in body_center_xy.tolist()],
                    "radial_body": float(radial_body),
                    "radial_threshold": float(radial_threshold),
                    "max_handle_radial": float(max_handle_radial),
                    "handle_z_floor": float(handle_z_floor),
                    "z_low": float(z_low),
                    "z_high": float(z_high),
                    "used_handle_side_cloud": False,
                    "handle_point_source": handle_point_source,
                }
            return None

        handle_center = np.mean(handle_points, axis=0)
        relative_xy = handle_points[:, :2] - body_center_xy
        tail_distance = np.linalg.norm(relative_xy, axis=1)
        tail_threshold = float(np.quantile(tail_distance, 0.75))
        tail_points = handle_points[tail_distance >= tail_threshold]
        min_tail_points = max(1, self.handle_min_points // 4)
        if tail_points.shape[0] >= min_tail_points:
            handle_direction_center_xy = np.median(tail_points[:, :2], axis=0)
            handle_direction_source = "handle_tail_outer_points"
        else:
            handle_direction_center_xy = handle_center[:2]
            handle_direction_source = "handle_center_fallback"

        outward_xy = handle_direction_center_xy - body_center_xy
        outward_axis_xy = _normalize_vector(outward_xy)
        if outward_axis_xy is None:
            return None

        closing_axis = _normalize_vector(np.array([outward_axis_xy[0], outward_axis_xy[1], 0.0], dtype=float))
        orientation_source = "handle_tail_outward_radial"
        approach_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        lateral_axis = _normalize_vector(np.cross(approach_axis, closing_axis))
        if closing_axis is None or lateral_axis is None:
            return None

        top_z = float(np.quantile(handle_points[:, 2], self.handle_top_down_surface_quantile))
        grasp_xy = handle_center[:2]
        projection = handle_points @ closing_axis
        thickness = float(np.quantile(projection, 0.90) - np.quantile(projection, 0.10))
        raw_gripper_width = max(self.handle_top_down_width_min, thickness + self.handle_top_down_width_margin)
        gripper_width = min(raw_gripper_width, self.handle_top_down_width_max)
        if thickness > self.handle_top_down_width_max:
            if hasattr(self, "_grasp_candidate_diagnostics"):
                self._grasp_candidate_diagnostics[str(candidate_id)] = {
                    "grasp_type": "handle_top_down",
                    "rejected": True,
                    "reject_reason": "width_exceeds_max",
                    "point_cloud_count": int(point_cloud.shape[0]),
                    "candidate_point_cloud_count": int(candidate_cloud.shape[0]),
                    "handle_point_count": int(handle_points.shape[0]),
                    "radial_center_xy": [float(value) for value in radial_center_xy.tolist()],
                    "body_center_xy": [float(value) for value in body_center_xy.tolist()],
                    "body_center_source": body_center_source,
                    "handle_center": [float(value) for value in handle_center.tolist()],
                    "handle_direction_center_xy": [float(value) for value in handle_direction_center_xy.tolist()],
                    "handle_direction_source": handle_direction_source,
                    "handle_tail_point_count": int(tail_points.shape[0]),
                    "handle_tail_distance_threshold": float(tail_threshold),
                    "outward_axis_xy": [float(value) for value in outward_axis_xy.tolist()],
                    "orientation_source": orientation_source,
                    "radial_body": float(radial_body),
                    "radial_threshold": float(radial_threshold),
                    "max_handle_radial": float(max_handle_radial),
                    "handle_z_floor": float(handle_z_floor),
                    "z_low": float(z_low),
                    "z_high": float(z_high),
                    "top_z": float(top_z),
                    "thickness": float(thickness),
                    "estimated_width": float(raw_gripper_width),
                    "max_width": float(self.handle_top_down_width_max),
                    "used_handle_side_cloud": False,
                    "residual_point_count": int(residual_points.shape[0]),
                    "residual_sector_center_deg": float(np.degrees(best_sector["center"])),
                    "residual_sector_score": float(best_sector["score"]),
                    "residual_sector_mean_radial": float(best_sector["mean_radial"]),
                    "residual_sector_median_z": float(best_sector["median_z"]),
                    "residual_sector_z_quality": float(best_sector["z_quality"]),
                    "residual_sector_point_count": int(best_sector["point_count"]),
                    "handle_point_source": handle_point_source,
                }
            return None

        top_band = handle_points[handle_points[:, 2] >= top_z]
        top_surface_quality = min(1.0, top_band.shape[0] / max(handle_points.shape[0], 1))
        handle_ratio = handle_points.shape[0] / point_cloud.shape[0]
        mean_handle_radial = float(np.mean(np.linalg.norm(handle_points[:, :2] - body_center_xy, axis=1)))
        protrusion_strength = float(np.linalg.norm(handle_center[:2] - body_center_xy))
        protrusion_quality = protrusion_strength / (protrusion_strength + radial_body + 1e-6)
        thickness_quality = max(0.0, 1.0 - abs(gripper_width - 0.03) / 0.03)
        score = min(
            1.0,
            float(target["conf"])
            * (
                0.45
                + 0.20 * handle_ratio
                + 0.15 * protrusion_quality
                + 0.10 * thickness_quality
                + 0.10 * top_surface_quality
            ),
        )
        if hasattr(self, "_grasp_candidate_diagnostics"):
            self._grasp_candidate_diagnostics[str(candidate_id)] = {
                "grasp_type": "handle_top_down",
                "point_cloud_count": int(point_cloud.shape[0]),
                "candidate_point_cloud_count": int(candidate_cloud.shape[0]),
                "handle_point_count": int(handle_points.shape[0]),
                "handle_ratio": float(handle_ratio),
                "radial_center_xy": [float(value) for value in radial_center_xy.tolist()],
                "body_center_xy": [float(value) for value in body_center_xy.tolist()],
                "body_center_source": body_center_source,
                "handle_center": [float(value) for value in handle_center.tolist()],
                "grasp_xy": [float(value) for value in grasp_xy.tolist()],
                "grasp_xy_source": "handle_point_cloud_center",
                "handle_direction_center_xy": [float(value) for value in handle_direction_center_xy.tolist()],
                "handle_direction_source": handle_direction_source,
                "handle_tail_point_count": int(tail_points.shape[0]),
                "handle_tail_distance_threshold": float(tail_threshold),
                "outward_axis_xy": [float(value) for value in outward_axis_xy.tolist()],
                "radial_body": float(radial_body),
                "radial_threshold": float(radial_threshold),
                "max_handle_radial": float(max_handle_radial),
                "handle_z_floor": float(handle_z_floor),
                "z_low": float(z_low),
                "z_high": float(z_high),
                "top_z": float(top_z),
                "top_surface_quality": float(top_surface_quality),
                "mean_handle_radial": float(mean_handle_radial),
                "protrusion_strength": float(protrusion_strength),
                "protrusion_quality": float(protrusion_quality),
                "thickness": float(thickness),
                "thickness_quality": float(thickness_quality),
                "estimated_width": float(raw_gripper_width),
                "selected_width": float(gripper_width),
                "approach_axis": [float(value) for value in approach_axis.tolist()],
                "closing_axis": [float(value) for value in closing_axis.tolist()],
                "orientation_source": orientation_source,
                "lateral_axis": [float(value) for value in lateral_axis.tolist()],
                "used_handle_side_cloud": False,
                "residual_point_count": int(residual_points.shape[0]),
                "residual_sector_center_deg": float(np.degrees(best_sector["center"])),
                "residual_sector_score": float(best_sector["score"]),
                "residual_sector_mean_radial": float(best_sector["mean_radial"]),
                "residual_sector_median_z": float(best_sector["median_z"]),
                "residual_sector_z_quality": float(best_sector["z_quality"]),
                "residual_sector_point_count": int(best_sector["point_count"]),
                "handle_point_source": handle_point_source,
            }
        return {
            "id": candidate_id,
            "pos": [
                float(grasp_xy[0]),
                float(grasp_xy[1]),
                float(top_z),
            ],
            "orientation": [
                [float(value) for value in closing_axis.tolist()],
                [float(value) for value in lateral_axis.tolist()],
                [float(value) for value in approach_axis.tolist()],
            ],
            "gripper_width": gripper_width,
            "score": score,
            "grasp_type": "handle_top_down",
        }

    def _build_grasp_candidates(self, target, point_cloud=None, handle_point_cloud=None, handle_side_band=None):
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
        if hasattr(self, "_grasp_candidate_diagnostics"):
            self._grasp_candidate_diagnostics = {}
        for grasp_type in candidate_types:
            if grasp_type == "top_down":
                candidates.append(self._estimate_top_down_candidate(target, point_cloud, next_id))
                next_id += 1
                continue
            if grasp_type == "handle_grasp" and target["label"] in self.handle_labels:
                handle_candidate = self._estimate_handle_candidate(
                    target,
                    handle_point_cloud if handle_point_cloud is not None else point_cloud,
                    next_id,
                )
                if handle_candidate is not None:
                    if hasattr(self, "_grasp_candidate_diagnostics") and str(next_id) in self._grasp_candidate_diagnostics:
                        self._grasp_candidate_diagnostics[str(next_id)]["bbox_side_band"] = handle_side_band
                    candidates.append(handle_candidate)
                    next_id += 1
                continue
            if grasp_type == "handle_top_down" and target["label"] in self.handle_labels:
                diagnostics_id = next_id
                handle_top_down_candidate = self._estimate_handle_top_down_candidate(
                    target,
                    point_cloud,
                    next_id,
                    handle_point_cloud=handle_point_cloud,
                )
                if handle_top_down_candidate is not None:
                    if hasattr(self, "_grasp_candidate_diagnostics") and str(next_id) in self._grasp_candidate_diagnostics:
                        self._grasp_candidate_diagnostics[str(next_id)]["bbox_side_band"] = handle_side_band
                    candidates.append(handle_top_down_candidate)
                    next_id += 1
                elif hasattr(self, "_grasp_candidate_diagnostics") and str(diagnostics_id) in self._grasp_candidate_diagnostics:
                    next_id += 1
                continue
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
            foreground_filter = dict(getattr(self, "_last_bbox_foreground_filter", {}))
            handle_point_cloud = None
            handle_side_band = None
            detection_label = str(detection.label).strip().lower()
            item = {
                "label": detection_label,
                "pos": world_pos,
                "conf": float(detection.score),
            }
            if detection_label in self.target_labels:
                if detection_label in self.handle_labels:
                    handle_point_cloud, handle_side_band = self._choose_handle_side_point_cloud(
                        detection,
                        depth_image,
                        item,
                    )
                target_candidates.append(item)
                detection_point_clouds[id(item)] = point_cloud
                detection_point_clouds[(id(item), "handle")] = handle_point_cloud
                detection_point_clouds[(id(item), "handle_side_band")] = handle_side_band
                bbox_diagnostics = self._bbox_diagnostics(detection, depth_image)
                bbox_diagnostics["foreground_depth_filter"] = foreground_filter
                detection_point_clouds[(id(item), "bbox")] = bbox_diagnostics
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

        grasp_candidates = self._build_grasp_candidates(
            target,
            detection_point_clouds.get(id(target)),
            handle_point_cloud=detection_point_clouds.get((id(target), "handle")),
            handle_side_band=detection_point_clouds.get((id(target), "handle_side_band")),
        )
        if hasattr(self, "_grasp_candidate_diagnostics"):
            self._grasp_candidate_diagnostics["target_bbox"] = detection_point_clouds.get((id(target), "bbox"))

        return build_detected_objects(
            target_label=target["label"],
            target_pos=target["pos"],
            target_conf=target["conf"],
            obstacles=obstacles,
            status="ready",
            conf_thresh=self.conf_thresh,
            grasp_candidates=grasp_candidates,
        )

    def infer_detected_objects_with_diagnostics(self, rgb_image, depth_image):
        self._grasp_candidate_diagnostics = {}
        detected_objects = self.infer_detected_objects(rgb_image, depth_image)
        diagnostics = {
            "grasp_candidates": dict(getattr(self, "_grasp_candidate_diagnostics", {})),
        }
        if hasattr(self, "_grasp_candidate_diagnostics"):
            delattr(self, "_grasp_candidate_diagnostics")
        return detected_objects, diagnostics

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
