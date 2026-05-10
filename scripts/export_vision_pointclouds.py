"""Export RGB-D, point cloud, and grasp candidate visualizations to outputs/."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import robocasa  # noqa: F401 - import registers RoboCasa envs into robosuite.make
import robosuite

from scripts.demo_robocasa_reach_onscreen import (
    build_env_config,
    get_rgbd,
    load_yaml,
    resolve_camera_config,
)
from vision.detector import Detection
from vision.coord_transform import world_to_camera
from vision.perception_loop import VisionPerceptionLoop
from vision.sorting_policy import classify_drinkware_targets


DEFAULT_LEFT_SINK_LAYOUT_ID = 1
CUP_MUG_SORTING_TASKS = {"CupMugSorting", "CupMugSortingClean"}
RANDOM_CUP_MUG_SORTING_TASKS = {"CupMugSortingRandom"}


def _normalize_rgb(rgb):
    frame = np.asarray(rgb)
    if frame.dtype.kind == "f":
        max_value = float(np.nanmax(frame)) if frame.size else 0.0
        if max_value <= 1.0:
            frame = np.clip(frame * 255.0, 0.0, 255.0)
        else:
            frame = np.clip(frame, 0.0, 255.0)
    return np.asarray(frame, dtype=np.uint8)


def _save_depth_png(depth, output_path):
    depth_map = np.asarray(depth, dtype=float)
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    if not np.any(valid):
        image = np.zeros(depth_map.shape, dtype=np.uint8)
    else:
        valid_depth = depth_map[valid]
        d_min = float(valid_depth.min())
        d_max = float(valid_depth.max())
        if d_max - d_min <= 1e-8:
            normalized = np.zeros(depth_map.shape, dtype=float)
        else:
            normalized = (depth_map - d_min) / (d_max - d_min)
        normalized[~valid] = 0.0
        image = np.asarray(np.clip(normalized * 255.0, 0.0, 255.0), dtype=np.uint8)
    Image.fromarray(image).save(output_path)


def _draw_detection_overlay(rgb, detections, payload, output_path):
    image = Image.fromarray(_normalize_rgb(rgb))
    drawer = ImageDraw.Draw(image)
    for index, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = [float(value) for value in detection.box_xyxy]
        drawer.rectangle((x0, y0, x1, y1), outline=(255, 80, 80), width=3)
        drawer.text(
            (x0 + 4.0, max(0.0, y0 - 18.0)),
            f"{index}:{detection.label} {float(detection.score):.2f}",
            fill=(255, 80, 80),
        )
    for candidate in payload.get("grasp_candidates", []):
        pos = candidate.get("pos")
        if pos is None:
            continue
        drawer.text((8.0, 8.0 + 18.0 * int(candidate["id"])), str(candidate["grasp_type"]), fill=(255, 255, 0))
    image.save(output_path)


def _label_color(label, strategy=None):
    normalized = str(label).strip().lower()
    if strategy == "handle_top_down" or normalized == "mug":
        return (255, 110, 110)
    if strategy == "top_down" or normalized in {"cup", "glass cup"}:
        return (110, 220, 255)
    return (255, 215, 90)


def _world_points_to_pixels(points, camera_config):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    rotation = np.asarray(camera_config["T_world_cam"]["rotation"], dtype=float)
    translation = np.asarray(camera_config["T_world_cam"]["translation"], dtype=float)
    fx = float(camera_config["fx"])
    fy = float(camera_config["fy"])
    cx = float(camera_config["cx"])
    cy = float(camera_config["cy"])
    pixel_rows = []
    depth_values = []
    for point_world in pts:
        point_cam = world_to_camera(point_world, rotation, translation)
        z = float(point_cam[2])
        if z <= 1e-8:
            continue
        u = float(point_cam[0] * fx / z + cx)
        v = float(point_cam[1] * fy / z + cy)
        pixel_rows.append([u, v])
        depth_values.append(z)
    if not pixel_rows:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    return np.asarray(pixel_rows, dtype=float), np.asarray(depth_values, dtype=float)


def _depth_to_color(depth_value, depth_min, depth_max):
    if depth_max - depth_min <= 1e-8:
        t = 0.5
    else:
        t = float(np.clip((depth_value - depth_min) / (depth_max - depth_min), 0.0, 1.0))
    red = int(255 * (1.0 - t))
    green = int(180 * (1.0 - abs(t - 0.5) * 2.0))
    blue = int(255 * t)
    return red, green, blue


def _draw_world_axes_overlay(drawer, pixel, orientation, axis_length=36.0):
    axes = np.asarray(orientation, dtype=float)
    colors = [(255, 90, 90), (90, 255, 140), (90, 160, 255)]
    origin_x = float(pixel[0])
    origin_y = float(pixel[1])
    for axis_index in range(min(3, axes.shape[0])):
        axis = axes[axis_index]
        end_x = origin_x + float(axis[0]) * axis_length
        end_y = origin_y - float(axis[1]) * axis_length
        drawer.line((origin_x, origin_y, end_x, end_y), fill=colors[axis_index], width=3)


def _render_camera_reprojection(
    rgb,
    scene_points,
    camera_config,
    output_path,
    title,
    candidate_points=None,
    candidate_payloads=None,
    target_points=None,
    handle_points=None,
):
    image = Image.fromarray(_normalize_rgb(rgb)).convert("RGB")
    drawer = ImageDraw.Draw(image, "RGBA")
    pixels, depths = _world_points_to_pixels(scene_points, camera_config)
    if depths.size > 0:
        depth_min = float(np.min(depths))
        depth_max = float(np.max(depths))
        order = np.argsort(depths)[::-1]
        for index in order:
            x, y = pixels[index]
            color = _depth_to_color(float(depths[index]), depth_min, depth_max)
            drawer.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=(*color, 150))

    if target_points is not None and len(target_points) > 0:
        target_pixels, _ = _world_points_to_pixels(target_points, camera_config)
        for x, y in target_pixels:
            drawer.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(255, 230, 80, 220))

    if handle_points is not None and len(handle_points) > 0:
        handle_pixels, _ = _world_points_to_pixels(handle_points, camera_config)
        for x, y in handle_pixels:
            drawer.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 120, 80, 230))

    if candidate_points is not None and len(candidate_points) > 0:
        candidate_pixels, _ = _world_points_to_pixels(candidate_points, camera_config)
        for idx, (x, y) in enumerate(candidate_pixels):
            drawer.ellipse((x - 5.0, y - 5.0, x + 5.0, y + 5.0), outline=(255, 70, 70, 255), width=2)
            if candidate_payloads is not None and idx < len(candidate_payloads):
                candidate = candidate_payloads[idx]
                drawer.text((x + 6.0, y - 8.0), f"{candidate['id']}:{candidate['grasp_type']}", fill=(255, 255, 255, 255))
                _draw_world_axes_overlay(drawer, (x, y), candidate["orientation"])

    drawer.rectangle((8, 8, 340, 30), fill=(0, 0, 0, 160))
    drawer.text((14, 12), title, fill=(255, 255, 255, 255))
    image.save(output_path)


def _render_grasp_axes_on_rgb(rgb, detections, payload, camera_config, output_path):
    image = Image.fromarray(_normalize_rgb(rgb)).convert("RGB")
    drawer = ImageDraw.Draw(image, "RGBA")
    for detection in detections:
        x0, y0, x1, y1 = [float(value) for value in detection.box_xyxy]
        drawer.rectangle((x0, y0, x1, y1), outline=(255, 110, 110, 255), width=3)
    candidates = list(payload.get("grasp_candidates", []))
    candidate_points = np.asarray([candidate["pos"] for candidate in candidates], dtype=float) if candidates else np.zeros((0, 3), dtype=float)
    candidate_pixels, _ = _world_points_to_pixels(candidate_points, camera_config)
    for idx, (x, y) in enumerate(candidate_pixels):
        candidate = candidates[idx]
        drawer.ellipse((x - 6.0, y - 6.0, x + 6.0, y + 6.0), outline=(255, 80, 80, 255), width=3)
        drawer.text((x + 8.0, y - 8.0), f"{candidate['id']} {candidate['grasp_type']}", fill=(255, 255, 255, 255))
        _draw_world_axes_overlay(drawer, (x, y), candidate["orientation"], axis_length=42.0)
    image.save(output_path)


def _draw_multi_target_overlay(rgb, targets, assignments, camera_config, output_path):
    image = Image.fromarray(_normalize_rgb(rgb)).convert("RGB")
    drawer = ImageDraw.Draw(image, "RGBA")
    for index, target in enumerate(targets):
        assignment = assignments[index] if index < len(assignments) else {}
        label = str(target.get("label", "unknown"))
        strategy = assignment.get("strategy")
        color = _label_color(label, strategy=strategy)
        candidates = list(target.get("grasp_candidates", []))
        if not candidates:
            continue
        candidate_points = np.asarray([candidate["pos"] for candidate in candidates], dtype=float)
        candidate_pixels, _ = _world_points_to_pixels(candidate_points, camera_config)
        for cand_idx, (x, y) in enumerate(candidate_pixels):
            candidate = candidates[cand_idx]
            drawer.ellipse((x - 6.0, y - 6.0, x + 6.0, y + 6.0), outline=(*color, 255), width=3)
            drawer.text(
                (x + 8.0, y - 10.0),
                f"{label}:{candidate['grasp_type']}",
                fill=(255, 255, 255, 255),
            )
            _draw_world_axes_overlay(drawer, (x, y), candidate["orientation"], axis_length=36.0)
        target_pos = target.get("pos")
        if target_pos is not None:
            target_pixel, _ = _world_points_to_pixels(np.asarray([target_pos], dtype=float), camera_config)
            if target_pixel.shape[0] > 0:
                x, y = target_pixel[0]
                drawer.ellipse((x - 4.0, y - 4.0, x + 4.0, y + 4.0), fill=(*color, 220))
                drawer.text(
                    (x + 8.0, y + 8.0),
                    f"{label} -> {strategy or 'none'}",
                    fill=(*color, 255),
                )
    drawer.rectangle((8, 8, 460, 30), fill=(0, 0, 0, 160))
    drawer.text((14, 12), "multi-target cup/mug overlay", fill=(255, 255, 255, 255))
    image.save(output_path)


def _collect_scene_point_cloud(loop, depth_image, stride):
    h, w = depth_image.shape[:2]
    pixels = np.array(
        [[float(v), float(u)] for v in range(0, h, stride) for u in range(0, w, stride)],
        dtype=float,
    )
    valid_mask = np.isfinite(depth_image[::stride, ::stride].reshape(-1)) & (
        depth_image[::stride, ::stride].reshape(-1) > 0.0
    )
    pixels = pixels[valid_mask]
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=float)
    camera_to_world_transform = loop.camera_config.get("camera_to_world_transform")
    if camera_to_world_transform is not None:
        from vision.coord_transform import pixels_to_world

        return pixels_to_world(
            pixels_rc=pixels,
            depth_image=depth_image,
            camera_to_world_transform=np.asarray(camera_to_world_transform, dtype=float),
        )

    fx = float(loop.camera_config["fx"])
    fy = float(loop.camera_config["fy"])
    cx = float(loop.camera_config["cx"])
    cy = float(loop.camera_config["cy"])
    transform = loop.camera_config["T_world_cam"]
    rotation = np.array(transform["rotation"], dtype=float)
    translation = np.array(transform["translation"], dtype=float)
    from vision.coord_transform import camera_to_world, pixel_to_3d

    points = []
    for row, col in pixels:
        depth = float(depth_image[int(round(row)), int(round(col))])
        point_cam = pixel_to_3d(float(col), float(row), depth, fx, fy, cx, cy)
        points.append(camera_to_world(point_cam, rotation, translation))
    return np.asarray(points, dtype=float)


def _save_ascii_ply(points, output_path, colors=None):
    pts = np.asarray(points, dtype=float)
    rgb = None if colors is None else np.asarray(colors, dtype=np.uint8)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(pts)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if rgb is not None:
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for index, point in enumerate(pts):
            if rgb is None:
                handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            else:
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(rgb[index, 0])} {int(rgb[index, 1])} {int(rgb[index, 2])}\n"
                )


def _project_points(points, axis_x, axis_y, size):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float), None
    coords = pts[:, [axis_x, axis_y]]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    margin = 0.08 * span
    mins -= margin
    maxs += margin
    span = np.maximum(maxs - mins, 1e-6)
    usable = float(size - 1)
    scaled = (coords - mins) / span
    px = scaled[:, 0] * usable
    py = usable - scaled[:, 1] * usable
    return np.column_stack([px, py]), {"mins": mins.tolist(), "maxs": maxs.tolist()}


def _render_point_cloud_projection(
    base_points,
    output_path,
    axis_x,
    axis_y,
    title,
    highlight_points=None,
    secondary_points=None,
):
    size = 900
    image = Image.new("RGB", (size, size), (18, 20, 26))
    drawer = ImageDraw.Draw(image)
    projected, bounds = _project_points(base_points, axis_x, axis_y, size)
    if projected.shape[0] > 0:
        for x, y in projected:
            drawer.ellipse((x - 1.0, y - 1.0, x + 1.0, y + 1.0), fill=(110, 180, 255))
    if secondary_points is not None and len(secondary_points) > 0 and bounds is not None:
        secondary_projected, _ = _project_points(
            np.vstack([base_points, secondary_points]),
            axis_x,
            axis_y,
            size,
        )
        offset = len(np.asarray(base_points))
        for x, y in secondary_projected[offset:]:
            drawer.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), fill=(255, 190, 90))
    if highlight_points is not None and len(highlight_points) > 0 and bounds is not None:
        combined = np.vstack([base_points, highlight_points])
        highlight_projected, _ = _project_points(combined, axis_x, axis_y, size)
        offset = len(np.asarray(base_points))
        for x, y in highlight_projected[offset:]:
            drawer.ellipse((x - 5.0, y - 5.0, x + 5.0, y + 5.0), outline=(255, 80, 80), width=2)
    drawer.text((16, 12), title, fill=(255, 255, 255))
    image.save(output_path)


def _find_target_detection(detections, payload):
    target = payload.get("target", {})
    target_label = str(target.get("label", "")).strip().lower()
    target_conf = float(target.get("conf", 0.0))
    matches = [
        detection
        for detection in detections
        if str(detection.label).strip().lower() == target_label
        and abs(float(detection.score) - target_conf) < 1e-6
    ]
    if matches:
        return matches[0]
    for detection in detections:
        if str(detection.label).strip().lower() == target_label:
            return detection
    return None


def _find_matching_detection(detections, target):
    target_label = str(target.get("label", "")).strip().lower()
    target_conf = float(target.get("conf", 0.0))
    target_pos = target.get("pos")
    candidates = [
        detection for detection in detections if str(detection.label).strip().lower() == target_label
    ]
    if not candidates:
        return None
    exact_conf = [d for d in candidates if abs(float(d.score) - target_conf) < 1e-6]
    if len(exact_conf) == 1:
        return exact_conf[0]
    if target_pos is None:
        return max(candidates, key=lambda item: float(item.score))
    return max(candidates, key=lambda item: float(item.score))


def _slugify_label(label):
    text = str(label).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_") or "unknown"


def _build_export_env_config(task_name):
    env_name = str(task_name).replace("robocasa/", "", 1)
    layout_id = DEFAULT_LEFT_SINK_LAYOUT_ID if env_name in CUP_MUG_SORTING_TASKS else None
    style_id = 1 if env_name in CUP_MUG_SORTING_TASKS else None
    layout_and_style_ids = None
    if env_name in CUP_MUG_SORTING_TASKS and layout_id is not None and style_id is not None:
        layout_and_style_ids = [[int(layout_id), int(style_id)]]
    return build_env_config(
        task_name=task_name,
        camera_name="robot0_agentview_center",
        width=640,
        height=480,
        layout=layout_id,
        style=style_id,
        seed=None,
        layout_and_style_ids=layout_and_style_ids,
    )


def export_visualizations(args):
    static_camera_config = load_yaml(PROJECT_ROOT / "configs" / "camera.yaml")
    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    multi_target_mode = str(args.target_label).strip().lower() in {"all", "drinkware", "cup+mug", "cup_and_mug"}
    env_name = str(args.task).replace("robocasa/", "", 1)
    if multi_target_mode:
        task_config_path = PROJECT_ROOT / "configs" / "tasks" / "cup_mug_sorting.yaml"
        if env_name in RANDOM_CUP_MUG_SORTING_TASKS:
            task_config_path = PROJECT_ROOT / "configs" / "tasks" / "cup_mug_sorting_random.yaml"
        task_config = load_yaml(task_config_path)
        vision_config["target_labels"] = list(task_config.get("labels", ["mug", "cup", "glass cup"]))
    else:
        vision_config["target_labels"] = [args.target_label]
    vision_config["obstacle_labels"] = []
    if multi_target_mode or "mug" in args.target_label.lower() or "cup" in args.target_label.lower():
        vision_config["candidate_types"] = ["top_down", "handle_top_down"]
        vision_config["handle_labels"] = ["cup", "mug", "glass cup"]
    if args.allow_download:
        vision_config["local_files_only"] = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = _build_export_env_config(args.task)
    env = robosuite.make(**env_config)
    try:
        obs = env.reset()
        env.render()
        time.sleep(0.05)
        rgb, depth = get_rgbd(obs, camera_name="robot0_agentview_center", sim=env.sim)
        camera_config = resolve_camera_config(
            env,
            camera_name="robot0_agentview_center",
            width=640,
            height=480,
        )
        if rgb is None or depth is None:
            raise RuntimeError("RGB-D observation unavailable for point cloud export.")

        fx = float(camera_config["fx"])
        fy = float(camera_config["fy"])
        cx = float(camera_config["cx"])
        cy = float(camera_config["cy"])
        loop = VisionPerceptionLoop(
            vision_config=vision_config,
            camera_config=camera_config,
            conf_thresh=float(thresholds_config["CONF_THRESH"]),
        )
        detections = loop.detector.detect(rgb, loop.target_labels + loop.obstacle_labels)
        scene_points = _collect_scene_point_cloud(loop, np.asarray(depth, dtype=float), stride=max(1, args.scene_stride))
        if multi_target_mode:
            all_targets_summary = loop.infer_all_targets_with_diagnostics(rgb, depth)
            targets = list(all_targets_summary.get("targets", []))
            if not targets:
                raise RuntimeError("Multi-target drinkware mode found no targets.")
            assignments = classify_drinkware_targets(targets)
            payload = {
                "target": {"label": "drinkware", "pos": [0.0, 0.0, 0.0], "conf": 1.0},
                "grasp_candidates": [],
                "obstacles": [],
                "status": all_targets_summary.get("status", "error"),
            }
            diagnostics = {"multi_target": all_targets_summary, "assignments": assignments}
            target_points = np.zeros((0, 3), dtype=float)
            handle_points = np.zeros((0, 3), dtype=float)
            candidate_point_rows = []
            per_target_exports = []
            for index, target in enumerate(targets):
                detection = _find_matching_detection(detections, target)
                point_cloud = None if detection is None else loop._extract_bbox_point_cloud(detection, depth)
                if point_cloud is not None and len(point_cloud) > 0:
                    target_points = np.vstack([target_points, point_cloud]) if target_points.size else np.asarray(point_cloud, dtype=float)
                handle_cloud = None
                if detection is not None and str(target.get("label", "")).strip().lower() in loop.handle_labels:
                    handle_cloud, _ = loop._choose_handle_side_point_cloud(detection, depth, target)
                    if handle_cloud is not None and len(handle_cloud) > 0:
                        handle_points = np.vstack([handle_points, handle_cloud]) if handle_points.size else np.asarray(handle_cloud, dtype=float)
                candidates = list(target.get("grasp_candidates", []))
                for candidate in candidates:
                    candidate_point_rows.append(candidate["pos"])
                per_target_exports.append(
                    {
                        "target": target,
                        "assignment": assignments[index] if index < len(assignments) else {},
                        "point_cloud_count": 0 if point_cloud is None else int(np.asarray(point_cloud).shape[0]),
                        "handle_point_cloud_count": 0 if handle_cloud is None else int(np.asarray(handle_cloud).shape[0]),
                    }
                )
            candidate_points = np.asarray(candidate_point_rows, dtype=float) if candidate_point_rows else np.zeros((0, 3), dtype=float)
        else:
            payload, diagnostics = loop.infer_detected_objects_with_diagnostics(rgb, depth)
            if payload.get("status") != "ready":
                raise RuntimeError(f"Vision payload not ready: status={payload.get('status')}")

            target_detection = _find_target_detection(detections, payload)
            if target_detection is None:
                raise RuntimeError("Failed to map selected payload target back to a detection box.")
            target_points = loop._extract_bbox_point_cloud(target_detection, depth)
            if target_points is None:
                target_points = np.zeros((0, 3), dtype=float)
            target = payload["target"]
            handle_points = None
            if str(target.get("label", "")).strip().lower() in loop.handle_labels:
                handle_points, _ = loop._choose_handle_side_point_cloud(target_detection, depth, target)

            candidate_points = np.asarray(
                [candidate["pos"] for candidate in payload.get("grasp_candidates", [])],
                dtype=float,
            )
            if candidate_points.ndim == 1 and candidate_points.size > 0:
                candidate_points = candidate_points[None, :]

        rgb_path = output_dir / "rgb.png"
        depth_path = output_dir / "depth.png"
        overlay_path = output_dir / "rgb_detection_overlay.png"
        multi_overlay_path = output_dir / "rgb_multi_target_overlay.png"
        reprojection_path = output_dir / "camera_reprojection_scene.png"
        reprojection_target_path = output_dir / "camera_reprojection_target.png"
        grasp_overlay_path = output_dir / "rgb_grasp_axes_overlay.png"
        scene_ply_path = output_dir / "scene_point_cloud.ply"
        target_ply_path = output_dir / "target_bbox_point_cloud.ply"
        handle_ply_path = output_dir / "target_handle_side_point_cloud.ply"
        summary_path = output_dir / "summary.json"

        Image.fromarray(_normalize_rgb(rgb)).save(rgb_path)
        _save_depth_png(depth, depth_path)
        _draw_detection_overlay(rgb, detections, payload, overlay_path)
        if multi_target_mode:
            _draw_multi_target_overlay(
                rgb=rgb,
                targets=targets,
                assignments=assignments,
                camera_config=camera_config,
                output_path=multi_overlay_path,
            )
            _render_camera_reprojection(
                rgb=rgb,
                scene_points=scene_points,
                camera_config=camera_config,
                output_path=reprojection_path,
                title="scene point cloud reprojected to camera",
                candidate_points=candidate_points if candidate_points.size > 0 else None,
                candidate_payloads=[],
                target_points=target_points if target_points.size > 0 else None,
                handle_points=handle_points if handle_points.size > 0 else None,
            )
            _render_camera_reprojection(
                rgb=rgb,
                scene_points=target_points if target_points.size > 0 else scene_points,
                camera_config=camera_config,
                output_path=reprojection_target_path,
                title="cup+mug target clouds reprojected to camera",
                candidate_points=candidate_points if candidate_points.size > 0 else None,
                candidate_payloads=[],
                target_points=target_points if target_points.size > 0 else None,
                handle_points=handle_points if handle_points.size > 0 else None,
            )
            _draw_multi_target_overlay(
                rgb=rgb,
                targets=targets,
                assignments=assignments,
                camera_config=camera_config,
                output_path=grasp_overlay_path,
            )
        else:
            _render_camera_reprojection(
                rgb=rgb,
                scene_points=scene_points,
                camera_config=camera_config,
                output_path=reprojection_path,
                title="scene point cloud reprojected to camera",
                candidate_points=candidate_points if candidate_points.size > 0 else None,
                candidate_payloads=payload.get("grasp_candidates", []),
            )
            _render_camera_reprojection(
                rgb=rgb,
                scene_points=target_points,
                camera_config=camera_config,
                output_path=reprojection_target_path,
                title="target bbox point cloud reprojected to camera",
                candidate_points=candidate_points if candidate_points.size > 0 else None,
                candidate_payloads=payload.get("grasp_candidates", []),
                target_points=target_points,
                handle_points=handle_points,
            )
            _render_grasp_axes_on_rgb(
                rgb=rgb,
                detections=detections,
                payload=payload,
                camera_config=camera_config,
                output_path=grasp_overlay_path,
            )
        _save_ascii_ply(scene_points, scene_ply_path)
        _save_ascii_ply(target_points, target_ply_path)
        if handle_points is not None and len(handle_points) > 0:
            _save_ascii_ply(handle_points, handle_ply_path)

        for name, points in [("scene", scene_points), ("target", target_points)]:
            secondary = handle_points if name == "target" and handle_points is not None else None
            for axis_x, axis_y, suffix in [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]:
                _render_point_cloud_projection(
                    base_points=points,
                    output_path=output_dir / f"{name}_point_cloud_{suffix}.png",
                    axis_x=axis_x,
                    axis_y=axis_y,
                    title=f"{name} point cloud {suffix}",
                    highlight_points=candidate_points if candidate_points.size > 0 else None,
                    secondary_points=secondary,
                )

        summary = {
            "task": args.task,
            "target_label": args.target_label,
            "multi_target_mode": bool(multi_target_mode),
            "camera": {
                "camera_name": "robot0_agentview_center",
                "active_camera_name": "robot0_agentview_center",
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            },
            "env_config": env_config,
            "raw_detections": [
                {
                    "label": detection.label,
                    "score": float(detection.score),
                    "box_xyxy": [float(value) for value in detection.box_xyxy],
                }
                for detection in detections
            ],
            "payload": payload,
            "diagnostics": diagnostics,
            "multi_target_exports": per_target_exports if multi_target_mode else None,
            "scene_point_count": int(scene_points.shape[0]),
            "target_point_count": int(target_points.shape[0]),
            "handle_point_count": 0 if handle_points is None else int(np.asarray(handle_points).shape[0]),
            "artifacts": {
                "rgb_png": str(rgb_path),
                "depth_png": str(depth_path),
                "overlay_png": str(overlay_path),
                "multi_target_overlay_png": None if not multi_target_mode else str(multi_overlay_path),
                "camera_reprojection_scene_png": str(reprojection_path),
                "camera_reprojection_target_png": str(reprojection_target_path),
                "rgb_grasp_axes_overlay_png": str(grasp_overlay_path),
                "scene_ply": str(scene_ply_path),
                "target_ply": str(target_ply_path),
                "handle_ply": None if not handle_ply_path.exists() else str(handle_ply_path),
            },
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary_path
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export RGB-D, point cloud, and grasp candidate visualizations."
    )
    parser.add_argument("--task", default="robocasa/CoffeeSetupMug")
    parser.add_argument("--target-label", default="mug")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "vision" / "pointcloud_export"),
    )
    parser.add_argument("--scene-stride", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    summary_path = export_visualizations(args)
    print(json.dumps({"summary_json": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
