"""Run a RoboCasa vision smoke on mug / cup tasks without robosuite-only execution logic."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import yaml
from PIL import Image, ImageDraw

from arm.env_wrapper import RobocasaEnvWrapper
from vision.perception_loop import VisionPerceptionLoop

try:
    import imageio
except ModuleNotFoundError:
    imageio = None

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def summarize_payload(payload):
    target = payload.get("target", {})
    candidates = payload.get("grasp_candidates", [])
    return {
        "status": payload.get("status"),
        "target": {
            "label": target.get("label"),
            "conf": target.get("conf"),
            "pos": target.get("pos"),
        },
        "candidate_count": len(candidates),
        "candidates": [
            {
                "id": candidate["id"],
                "grasp_type": candidate["grasp_type"],
                "score": candidate["score"],
                "gripper_width": candidate["gripper_width"],
                "pos": candidate["pos"],
            }
            for candidate in candidates
        ],
        "obstacle_count": len(payload.get("obstacles", [])),
    }


def _safe_stem(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def _to_json_ready(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_run_stem(task_name):
    return f"{_safe_stem(task_name)}_{time.strftime('%Y%m%d_%H%M%S')}"


def _draw_overlay_text(image_array, lines):
    image = Image.fromarray(np.asarray(image_array, dtype=np.uint8))
    drawer = ImageDraw.Draw(image)
    text_y = 8.0
    for line in lines:
        drawer.rectangle((8.0, text_y, 520.0, text_y + 18.0), fill=(0, 0, 0))
        drawer.text((12.0, text_y + 2.0), str(line), fill=(255, 255, 0))
        text_y += 20.0
    return np.asarray(image, dtype=np.uint8)


def _frame_has_visible_content(frame, mean_threshold=3.0, std_threshold=2.0):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.size == 0:
        return False
    return float(frame.mean()) >= float(mean_threshold) or float(frame.std()) >= float(std_threshold)


def _normalize_rgb_uint8(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype.kind == "f":
        max_value = float(np.nanmax(frame)) if frame.size else 0.0
        if max_value <= 1.0:
            frame = np.clip(frame * 255.0, 0.0, 255.0)
        else:
            frame = np.clip(frame, 0.0, 255.0)
    return np.asarray(frame, dtype=np.uint8)


def _normalize_depth_array(depth):
    depth = np.asarray(depth, dtype=float)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def _resolve_initial_rgbd(env):
    rgb, depth, proprio = env.get_observation()
    if rgb is not None and depth is not None:
        return rgb, depth, proprio

    raw_obs = env.get_raw_observation()
    if not isinstance(raw_obs, dict):
        return rgb, depth, proprio

    active_camera_name = getattr(env, "active_camera_name", None)
    active_rgb_key = getattr(env, "active_rgb_key", None)
    active_depth_key = getattr(env, "active_depth_key", None)

    rgb_candidate_keys = []
    depth_candidate_keys = []
    if active_rgb_key is not None:
        rgb_candidate_keys.append(str(active_rgb_key))
    if active_depth_key is not None:
        depth_candidate_keys.append(str(active_depth_key))
    if active_camera_name is not None:
        rgb_candidate_keys.append(f"{active_camera_name}_image")
        depth_candidate_keys.append(f"{active_camera_name}_depth")
        rgb_candidate_keys.append(f"video.{active_camera_name}")
        depth_candidate_keys.append(f"video.{active_camera_name}_depth")
    rgb_candidate_keys.extend(
        [
            "robot0_agentview_left_image",
            "robot0_agentview_right_image",
            "robot0_eye_in_hand_image",
            "robot0_agentview_center_image",
            "frontview_image",
        ]
    )
    depth_candidate_keys.extend(
        [
            "robot0_agentview_left_depth",
            "robot0_agentview_right_depth",
            "robot0_eye_in_hand_depth",
            "robot0_agentview_center_depth",
            "frontview_depth",
        ]
    )

    if rgb is None:
        for key in rgb_candidate_keys:
            if key in raw_obs:
                rgb = _normalize_rgb_uint8(raw_obs[key])
                env.active_rgb_key = key
                break

    if depth is None:
        for key in depth_candidate_keys:
            if key in raw_obs:
                depth = _normalize_depth_array(raw_obs[key])
                env.active_depth_key = key
                break

    return rgb, depth, proprio


def create_frame_sink(save_dir, run_stem, save_video=False, video_path=None, save_frames=False, save_gif=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = save_dir / f"{run_stem}_frames"
    if save_frames or save_gif:
        frames_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    resolved_video_path = None
    if save_video:
        default_video_path = save_dir / f"{run_stem}.avi"
        resolved_video_path = Path(video_path) if video_path is not None else default_video_path
        video_writer = {"path": resolved_video_path}

    return {
        "save_frames": bool(save_frames),
        "save_gif": bool(save_gif),
        "save_video": bool(save_video),
        "frames_dir": frames_dir,
        "frame_index": 0,
        "gif_frames": [],
        "video_writer": video_writer,
        "video_path": resolved_video_path,
        "preview_path": None,
    }


def get_video_frame_rgb(env):
    latest_rgb_frame = getattr(env, "_latest_rgb_frame", None)
    if latest_rgb_frame is not None:
        frame = _normalize_rgb_uint8(latest_rgb_frame)
        if _frame_has_visible_content(frame):
            setattr(env, "_last_video_frame_key", "cached_live_rgb")
            setattr(
                env,
                "_last_video_frame_stats",
                {
                    "mean": float(frame.mean()),
                    "std": float(frame.std()),
                },
            )
            return frame

    sim = getattr(env, "get_sim", lambda: None)()
    active_camera_name = getattr(env, "active_camera_name", None) or getattr(env, "camera_name", None)
    if sim is not None and active_camera_name is not None:
        try:
            frame = sim.render(
                camera_name=str(active_camera_name),
                height=int(getattr(env, "camera_height", 480)),
                width=int(getattr(env, "camera_width", 640)),
            )
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = _normalize_rgb_uint8(frame[::-1].copy())
                if _frame_has_visible_content(frame):
                    setattr(env, "_last_video_frame_key", f"sim.render:{active_camera_name}")
                    setattr(
                        env,
                        "_last_video_frame_stats",
                        {
                            "mean": float(frame.mean()),
                            "std": float(frame.std()),
                        },
                    )
                    return frame
        except Exception:
            pass

    raw_obs = env.get_raw_observation()
    if isinstance(raw_obs, dict):
        candidate_keys = []
        if active_camera_name is not None:
            candidate_keys.append(f"{active_camera_name}_image")
        active_rgb_key = getattr(env, "active_rgb_key", None)
        if active_rgb_key is not None:
            candidate_keys.append(str(active_rgb_key))
        candidate_keys.extend(
            [
                "robot0_agentview_left_image",
                "robot0_agentview_right_image",
                "robot0_eye_in_hand_image",
                "robot0_agentview_center_image",
                "frontview_image",
            ]
        )
        for key in candidate_keys:
            if key not in raw_obs:
                continue
            frame = np.asarray(raw_obs[key])
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = _normalize_rgb_uint8(frame)
                if _frame_has_visible_content(frame):
                    setattr(env, "_last_video_frame_key", key)
                    setattr(
                        env,
                        "_last_video_frame_stats",
                        {
                            "mean": float(frame.mean()),
                            "std": float(frame.std()),
                        },
                    )
                    return frame

    rgb, _, _ = env.get_observation()
    if rgb is None:
        return None
    frame = _normalize_rgb_uint8(rgb)
    setattr(env, "_last_video_frame_key", "wrapper_rgb")
    setattr(
        env,
        "_last_video_frame_stats",
        {
            "mean": float(frame.mean()),
            "std": float(frame.std()),
        },
    )
    return frame


def capture_visual_frame(env, frame_sink, lines):
    if frame_sink is None:
        return
    frame = get_video_frame_rgb(env)
    if frame is None:
        return
    rendered_frame = _draw_overlay_text(frame, lines)
    frame_sink["frame_index"] += 1
    frame_index = frame_sink["frame_index"]

    if frame_sink.get("preview_path") is None:
        preview_path = frame_sink["frames_dir"].parent / f"{frame_sink['frames_dir'].stem}-preview.png"
        Image.fromarray(rendered_frame).save(preview_path)
        frame_sink["preview_path"] = preview_path

    if frame_sink.get("save_frames") or frame_sink.get("save_gif"):
        frame_path = frame_sink["frames_dir"] / f"frame_{frame_index:04d}.png"
        Image.fromarray(rendered_frame).save(frame_path)

    if frame_sink.get("save_gif"):
        frame_sink["gif_frames"].append(Image.fromarray(rendered_frame))


def finalize_frame_sink(frame_sink):
    if frame_sink is None:
        return
    if frame_sink.get("save_gif") and frame_sink["gif_frames"]:
        gif_path = frame_sink["frames_dir"].parent / f"{frame_sink['frames_dir'].stem}.gif"
        first_frame, *rest_frames = frame_sink["gif_frames"]
        first_frame.save(
            gif_path,
            save_all=True,
            append_images=rest_frames,
            duration=50,
            loop=0,
        )
        frame_sink["gif_path"] = gif_path


def save_visualization(save_dir, task_name, rgb, raw_detections, payload=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_stem = build_run_stem(task_name)
    image_path = save_dir / f"{run_stem}.png"
    json_path = save_dir / f"{run_stem}.json"

    image = Image.fromarray(rgb.astype("uint8"))
    drawer = ImageDraw.Draw(image)

    for index, detection in enumerate(raw_detections, start=1):
        x0, y0, x1, y1 = [float(value) for value in detection.box_xyxy]
        drawer.rectangle((x0, y0, x1, y1), outline=(255, 80, 80), width=3)
        drawer.text(
            (x0 + 4.0, max(0.0, y0 - 18.0)),
            f"{index}:{detection.label} {float(detection.score):.2f}",
            fill=(255, 80, 80),
        )

    summary_lines = [f"task={task_name}", f"detections={len(raw_detections)}"]
    if payload is not None:
        target = payload.get("target", {})
        summary_lines.append(
            "target="
            f"{target.get('label', 'unknown')} "
            f"conf={float(target.get('conf', 0.0)):.2f}"
        )
        target_pos = target.get("pos")
        if target_pos is not None:
            summary_lines.append(
                "target_pos="
                f"({float(target_pos[0]):.3f}, {float(target_pos[1]):.3f}, {float(target_pos[2]):.3f})"
            )
        for candidate in payload.get("grasp_candidates", []):
            candidate_pos = candidate.get("pos", [0.0, 0.0, 0.0])
            summary_lines.append(
                f"cand{candidate['id']} {candidate['grasp_type']} "
                f"s={float(candidate['score']):.2f} "
                f"w={float(candidate['gripper_width']):.3f}"
            )
            summary_lines.append(
                "  pos="
                f"({float(candidate_pos[0]):.3f}, {float(candidate_pos[1]):.3f}, {float(candidate_pos[2]):.3f})"
            )

    text_y = 8.0
    for line in summary_lines:
        drawer.rectangle((8.0, text_y, 420.0, text_y + 18.0), fill=(0, 0, 0))
        drawer.text((12.0, text_y + 2.0), line, fill=(255, 255, 0))
        text_y += 20.0

    image.save(image_path)
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "task_name": task_name,
                "raw_detections": [
                    {
                        "label": detection.label,
                        "score": float(detection.score),
                        "box_xyxy": [float(value) for value in detection.box_xyxy],
                    }
                    for detection in raw_detections
                ],
                "payload": payload,
            },
            file_handle,
            indent=2,
            ensure_ascii=False,
            default=_to_json_ready,
        )
    return run_stem, image_path, json_path


def build_action_slices(action_layout):
    slices = {}
    cursor = 0
    for key, width in action_layout:
        slices[str(key)] = slice(cursor, cursor + int(width))
        cursor += int(width)
    return slices


def build_reach_action(action_dim, action_slices, position_delta, gripper_close=0.0):
    action = np.zeros(int(action_dim), dtype=np.float32)
    if "action.end_effector_position" in action_slices:
        action[action_slices["action.end_effector_position"]] = np.asarray(
            position_delta,
            dtype=np.float32,
        )
    elif action.shape[0] >= 3:
        action[:3] = np.asarray(position_delta, dtype=np.float32)

    if "action.gripper_close" in action_slices:
        action[action_slices["action.gripper_close"]] = float(gripper_close)
    if "action.end_effector_rotation" in action_slices:
        action[action_slices["action.end_effector_rotation"]] = 0.0
    if "action.base_motion" in action_slices:
        action[action_slices["action.base_motion"]] = 0.0
    if "action.control_mode" in action_slices:
        action[action_slices["action.control_mode"]] = 0.0
    return action


def get_robot0_eef_pos(env):
    raw_obs = env.get_raw_observation()
    if not isinstance(raw_obs, dict) or "robot0_eef_pos" not in raw_obs:
        raise RuntimeError("RoboCasa raw observation does not expose robot0_eef_pos.")
    return np.asarray(raw_obs["robot0_eef_pos"], dtype=float)


def choose_candidate(payload, candidate_id=None):
    candidates = payload.get("grasp_candidates", [])
    if not candidates:
        return None
    if candidate_id is not None:
        for candidate in candidates:
            if int(candidate["id"]) == int(candidate_id):
                return candidate
        raise ValueError(f"Candidate id {candidate_id} was not present in the payload.")
    return max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))


def _apply_position_pulse(env, action_slices, axis_index, magnitude, steps, frame_sink=None):
    for _ in range(int(steps)):
        position_delta = np.zeros(3, dtype=np.float32)
        position_delta[int(axis_index)] = float(magnitude)
        action = build_reach_action(
            action_dim=env.action_dim,
            action_slices=action_slices,
            position_delta=position_delta,
            gripper_close=0.0,
        )
        _, _, done, info = env.step(action)
        capture_visual_frame(
            env,
            frame_sink,
            [
                f"calibration axis={axis_index}",
                f"pulse={float(magnitude):+.4f}",
            ],
        )
        if done:
            return False, info
    return True, None


def calibrate_position_action_mapping(env, pulse_magnitude=0.01, pulse_steps=4, frame_sink=None):
    action_slices = build_action_slices(env.get_action_layout())
    basis_columns = []
    probe_logs = []
    for axis_index in range(3):
        start_pos = get_robot0_eef_pos(env)
        ok, info = _apply_position_pulse(
            env,
            action_slices=action_slices,
            axis_index=axis_index,
            magnitude=float(pulse_magnitude),
            steps=int(pulse_steps),
            frame_sink=frame_sink,
        )
        pos_after_positive = get_robot0_eef_pos(env)
        if not ok:
            return None, {
                "calibration_ok": False,
                "failed_axis": axis_index,
                "failed_phase": "positive_pulse",
                "info": info,
            }

        ok, info = _apply_position_pulse(
            env,
            action_slices=action_slices,
            axis_index=axis_index,
            magnitude=-float(pulse_magnitude),
            steps=int(pulse_steps * 2),
            frame_sink=frame_sink,
        )
        pos_after_negative = get_robot0_eef_pos(env)
        if not ok:
            return None, {
                "calibration_ok": False,
                "failed_axis": axis_index,
                "failed_phase": "negative_pulse",
                "info": info,
            }

        ok, info = _apply_position_pulse(
            env,
            action_slices=action_slices,
            axis_index=axis_index,
            magnitude=float(pulse_magnitude),
            steps=int(pulse_steps),
            frame_sink=frame_sink,
        )
        recovered_pos = get_robot0_eef_pos(env)
        if not ok:
            return None, {
                "calibration_ok": False,
                "failed_axis": axis_index,
                "failed_phase": "recovery_pulse",
                "info": info,
            }

        world_delta = (pos_after_positive - pos_after_negative) / 2.0
        command_delta = float(pulse_magnitude) * float(pulse_steps)
        basis_column = world_delta / max(command_delta, 1e-8)
        basis_columns.append(basis_column)
        probe_logs.append(
            {
                "axis_index": axis_index,
                "start_pos": start_pos.tolist(),
                "pos_after_positive": pos_after_positive.tolist(),
                "pos_after_negative": pos_after_negative.tolist(),
                "recovered_pos": recovered_pos.tolist(),
                "estimated_world_delta_per_unit_action": basis_column.tolist(),
            }
        )

    mapping = np.column_stack(basis_columns)
    return mapping, {
        "calibration_ok": True,
        "pulse_magnitude": float(pulse_magnitude),
        "pulse_steps": int(pulse_steps),
        "world_delta_from_action": mapping.tolist(),
        "probe_logs": probe_logs,
    }


def execute_reach_smoke(
    env,
    candidate,
    hover_offset=0.08,
    settle_offset=0.03,
    max_steps=120,
    position_gain=6.0,
    step_limit=0.03,
    reach_tolerance=0.04,
    action_mapping=None,
    frame_sink=None,
):
    action_slices = build_action_slices(env.get_action_layout())
    candidate_pos = np.asarray(candidate["pos"], dtype=float)
    hover_target = candidate_pos + np.array([0.0, 0.0, float(hover_offset)], dtype=float)
    settle_target = candidate_pos + np.array([0.0, 0.0, float(settle_offset)], dtype=float)
    start_pos = get_robot0_eef_pos(env)

    phases = [("hover", hover_target), ("settle", settle_target)]
    history = []
    reached = True
    for phase_name, phase_target in phases:
        phase_success = False
        for step_idx in range(int(max_steps)):
            current_pos = get_robot0_eef_pos(env)
            error = phase_target - current_pos
            error_norm = float(np.linalg.norm(error))
            history.append(
                {
                    "phase": phase_name,
                    "step": step_idx,
                    "eef_pos": current_pos.tolist(),
                    "target_pos": phase_target.tolist(),
                    "error_norm": error_norm,
                }
            )
            if error_norm <= float(reach_tolerance):
                phase_success = True
                break
            desired_world_delta = np.clip(
                error * float(position_gain),
                -float(step_limit),
                float(step_limit),
            )
            if action_mapping is not None:
                position_delta = np.linalg.pinv(np.asarray(action_mapping, dtype=float)) @ desired_world_delta
                position_delta = np.clip(position_delta, -float(step_limit), float(step_limit))
            else:
                position_delta = desired_world_delta
            action = build_reach_action(
                action_dim=env.action_dim,
                action_slices=action_slices,
                position_delta=position_delta,
                gripper_close=0.0,
            )
            _, _, done, info = env.step(action)
            capture_visual_frame(
                env,
                frame_sink,
                [
                    f"reach phase={phase_name}",
                    f"candidate={candidate['id']} {candidate['grasp_type']}",
                    f"error={error_norm:.4f}",
                ],
            )
            if done:
                reached = False
                history.append(
                    {
                        "phase": phase_name,
                        "step": step_idx,
                        "terminated": True,
                        "info": info,
                    }
                )
                break
        if not phase_success:
            reached = False
            if env.is_episode_terminated():
                break

    final_pos = get_robot0_eef_pos(env)
    return {
        "reach_success": bool(reached),
        "candidate_id": int(candidate["id"]),
        "candidate_type": candidate["grasp_type"],
        "candidate_pos": candidate_pos.tolist(),
        "hover_target": hover_target.tolist(),
        "settle_target": settle_target.tolist(),
        "robot0_eef_start": start_pos.tolist(),
        "robot0_eef_final": final_pos.tolist(),
        "final_error_to_settle": float(np.linalg.norm(settle_target - final_pos)),
        "action_layout": env.get_action_layout(),
        "action_mapping": None if action_mapping is None else np.asarray(action_mapping, dtype=float).tolist(),
        "history_tail": history[-10:],
    }


def build_task_config(task_name):
    return {
        "task_name": task_name,
        "split": "all",
        "render_mode": None,
        "use_camera_obs": True,
        "camera_names": "robot0_agentview_center",
        "camera_heights": 480,
        "camera_widths": 640,
        "camera_depths": True,
    }


def main():
    parser = argparse.ArgumentParser(description="RoboCasa mug / cup vision smoke test")
    parser.add_argument(
        "--task",
        default="robocasa/CoffeeSetupMug",
        help="RoboCasa task name, e.g. robocasa/CoffeeSetupMug or robocasa/CoffeeServeMug",
    )
    parser.add_argument(
        "--target-label",
        default="mug",
        help="Vision target label, e.g. mug, cup, glass cup",
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow model download if cache is missing")
    parser.add_argument(
        "--save-dir",
        default=str(PROJECT_ROOT / "outputs" / "vision" / "robocasa_smoke"),
        help="Directory used to save overlay PNG and payload JSON",
    )
    parser.add_argument(
        "--execute-reach",
        action="store_true",
        help="After visual inference, run a minimal open-gripper reach smoke toward a selected candidate.",
    )
    parser.add_argument(
        "--candidate-id",
        type=int,
        default=None,
        help="Optional candidate id to use for reach; defaults to the highest-score candidate.",
    )
    parser.add_argument(
        "--skip-axis-calibration",
        action="store_true",
        help="Use naive world-delta-as-action control instead of calibrating the RoboCasa action axes first.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Keep legacy video export enabled when possible. Frame export is the recommended path.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional explicit legacy video path.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save the full rollout as a PNG frame sequence.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save the rollout as an animated GIF.",
    )
    args = parser.parse_args()

    static_camera_config = load_yaml(PROJECT_ROOT / "configs" / "camera.yaml")
    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    vision_config["target_labels"] = [args.target_label]
    vision_config["obstacle_labels"] = []
    if "mug" in args.target_label.lower() or "cup" in args.target_label.lower():
        vision_config["candidate_types"] = ["top_down", "handle_grasp"]
        vision_config["handle_labels"] = ["cup", "mug", "glass cup"]
    if args.allow_download:
        vision_config["local_files_only"] = False

    total_stages = 6 if args.execute_reach else 5

    print(f"Stage 1/{total_stages}: create RoboCasa environment")
    print(json.dumps(build_task_config(args.task), indent=2))
    env = RobocasaEnvWrapper(config=build_task_config(args.task), camera_config=static_camera_config)

    print(f"Stage 2/{total_stages}: fetch camera observation")
    rgb, depth, proprio = env.get_observation()
    if rgb is None:
        raw_obs = env.get_raw_observation()
        obs_keys = sorted(raw_obs.keys()) if isinstance(raw_obs, dict) else None
        raise RuntimeError(
            "RoboCasa wrapper did not expose an RGB observation. "
            "This task / camera config is not yet compatible with the vision smoke. "
            f"Available raw observation keys: {obs_keys}"
        )
    fx, fy, cx, cy = env.get_camera_intrinsics()
    rotation, translation = env.get_camera_extrinsics()
    camera_to_world_transform = env.get_camera_to_world_transform()
    camera_config = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_to_world_transform": camera_to_world_transform.tolist(),
        "T_world_cam": {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
    }
    print(
        json.dumps(
            {
                "camera_name": env.camera_name,
                "active_camera_name": env.active_camera_name,
                "active_rgb_key": env.active_rgb_key,
                "active_depth_key": env.active_depth_key,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "rgb_shape": list(rgb.shape),
                "depth_shape": None if depth is None else list(depth.shape),
                "flat_observation_dim": int(proprio["flat_observation"].shape[0]),
                "raw_obs_keys": (
                    sorted(env.get_raw_observation().keys())
                    if isinstance(env.get_raw_observation(), dict)
                    else None
                ),
            },
            indent=2,
        )
    )

    print(f"Stage 3/{total_stages}: run raw detection")
    loop = VisionPerceptionLoop(
        vision_config=vision_config,
        camera_config=camera_config,
        conf_thresh=float(thresholds_config["CONF_THRESH"]),
    )
    raw_detections = loop.detector.detect(
        rgb,
        loop.target_labels + loop.obstacle_labels,
    )
    print(
        json.dumps(
            {
                "raw_detections": [
                    {
                        "label": detection.label,
                        "score": float(detection.score),
                        "box_xyxy": [float(value) for value in detection.box_xyxy],
                    }
                    for detection in raw_detections
                ]
            },
            indent=2,
        )
    )

    print(f"Stage 4/{total_stages}: build detection payload")
    payload = None
    if depth is None:
        print("Depth observation unavailable; skipping 3D payload construction.")
    else:
        payload = loop.infer_detected_objects(rgb, depth)
        print(json.dumps(summarize_payload(payload), indent=2))

    print(f"Stage 5/{total_stages}: save visualization artifacts")
    run_stem, image_path, json_path = save_visualization(
        save_dir=args.save_dir,
        task_name=args.task,
        rgb=rgb,
        raw_detections=raw_detections,
        payload=payload,
    )
    frame_sink = None
    video_path = None
    if args.save_video or args.save_frames or args.save_gif:
        frame_sink = create_frame_sink(
            save_dir=args.save_dir,
            run_stem=run_stem,
            save_video=args.save_video,
            video_path=args.video_path,
            save_frames=args.save_frames or args.save_gif,
            save_gif=args.save_gif,
        )
        video_path = frame_sink.get("video_path")
        capture_visual_frame(
            env,
            frame_sink,
            [
                f"task={args.task}",
                "stage=post-detection",
            ],
        )
        print(
            json.dumps(
                {
                    "video_capture_key": getattr(env, "_last_video_frame_key", None),
                    "video_frame_stats": getattr(env, "_last_video_frame_stats", None),
                    "frame_preview_png": (
                        None
                        if frame_sink.get("preview_path") is None
                        else str(frame_sink["preview_path"])
                    ),
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "overlay_png": str(image_path),
                "payload_json": str(json_path),
                "video_path": None if video_path is None else str(video_path),
                "frames_dir": None if frame_sink is None else str(frame_sink["frames_dir"]),
            },
            indent=2,
        )
    )

    try:
        if args.execute_reach:
            print(f"Stage 6/{total_stages}: execute minimal reach smoke")
            if payload is None:
                raise RuntimeError("Reach smoke requires a depth-backed payload, but depth was unavailable.")
            candidate = choose_candidate(payload, candidate_id=args.candidate_id)
            if candidate is None:
                raise RuntimeError("Reach smoke requires at least one grasp candidate.")
            action_mapping = None
            calibration_summary = None
            if not args.skip_axis_calibration:
                action_mapping, calibration_summary = calibrate_position_action_mapping(
                    env,
                    frame_sink=frame_sink,
                )
                print(json.dumps({"axis_calibration": calibration_summary}, indent=2))
                if action_mapping is None:
                    raise RuntimeError("Axis calibration failed before reach smoke.")
            reach_summary = execute_reach_smoke(
                env,
                candidate,
                action_mapping=action_mapping,
                frame_sink=frame_sink,
            )
            print(json.dumps(reach_summary, indent=2))
    finally:
        if frame_sink is not None:
            finalize_frame_sink(frame_sink)
            summary = {
                "frames_dir": str(frame_sink["frames_dir"]),
                "frame_count": int(frame_sink["frame_index"]),
                "gif_path": None
                if frame_sink.get("gif_path") is None
                else str(frame_sink["gif_path"]),
                "frame_preview_png": None
                if frame_sink.get("preview_path") is None
                else str(frame_sink["preview_path"]),
            }
            print(json.dumps({"frame_export": summary}, indent=2))


if __name__ == "__main__":
    main()
