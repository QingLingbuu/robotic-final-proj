"""Run a vision-driven RoboCasa reach while showing the live onscreen viewer."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import robocasa  # noqa: F401 - import registers RoboCasa envs into robosuite.make
import robosuite
import robosuite.utils.camera_utils as robosuite_camera_utils
from robosuite.controllers import load_composite_controller_config
from termcolor import colored

from vision.perception_loop import VisionPerceptionLoop


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def normalize_rgb_uint8(frame):
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


def normalize_depth(depth, sim):
    depth = np.asarray(depth, dtype=float)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if np.all(np.isfinite(depth)) and np.all(depth >= 0.0) and np.all(depth <= 1.0):
        depth = robosuite_camera_utils.get_real_depth_map(sim=sim, depth_map=depth)
    return np.asarray(depth, dtype=float)


def build_env_config(task_name, camera_name, width, height, layout=None, style=None):
    return {
        "env_name": task_name.replace("robocasa/", "", 1),
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "camera_names": [camera_name],
        "camera_widths": [int(width)],
        "camera_heights": [int(height)],
        "camera_depths": [True],
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "render_camera": "robot0_frontview",
        "ignore_done": True,
        "use_camera_obs": True,
        "use_object_obs": True,
        "control_freq": 20,
        "renderer": "mjviewer",
        "layout_ids": layout,
        "style_ids": style,
        "obj_instance_split": None,
        "layout_and_style_ids": None,
        "translucent_robot": True,
    }


def resolve_camera_config(env, camera_name, width, height):
    intrinsic = robosuite_camera_utils.get_camera_intrinsic_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = robosuite_camera_utils.get_camera_extrinsic_matrix(
        sim=env.sim,
        camera_name=camera_name,
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    transform = robosuite_camera_utils.get_camera_transform_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    return {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "camera_to_world_transform": np.linalg.inv(np.asarray(transform, dtype=float)).tolist(),
        "T_world_cam": {
            "rotation": np.asarray(world_to_camera[:3, :3], dtype=float).tolist(),
            "translation": np.asarray(world_to_camera[:3, 3], dtype=float).tolist(),
        },
    }


def get_rgbd(obs, camera_name, sim):
    rgb_key = f"{camera_name}_image"
    depth_key = f"{camera_name}_depth"
    if rgb_key not in obs:
        raise RuntimeError(f"Observation missing RGB key {rgb_key!r}. Available keys: {sorted(obs.keys())}")
    if depth_key not in obs:
        raise RuntimeError(f"Observation missing depth key {depth_key!r}. Available keys: {sorted(obs.keys())}")
    rgb = normalize_rgb_uint8(obs[rgb_key][::-1].copy())
    depth = normalize_depth(obs[depth_key][::-1].copy(), sim=sim)
    return rgb, depth


def build_flat_reach_action(env, position_delta, gripper_close=0.0):
    action = np.zeros(env.action_dim, dtype=np.float32)
    # PandaOmron default composite controller layout is:
    # [eef pos(3), eef rot(3), gripper(1), base(4), base_mode(1)]
    action[0:3] = np.asarray(position_delta, dtype=np.float32)
    action[6] = float(gripper_close)
    return action


def get_robot0_eef_pos(obs):
    if "robot0_eef_pos" not in obs:
        raise RuntimeError(f"Observation missing robot0_eef_pos. Available keys: {sorted(obs.keys())}")
    return np.asarray(obs["robot0_eef_pos"], dtype=float)


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


def calibrate_position_action_mapping(env, obs, pulse_magnitude=0.01, pulse_steps=4, render_sleep_sec=0.02):
    basis_columns = []
    probe_logs = []
    current_obs = obs
    for axis_index in range(3):
        start_pos = get_robot0_eef_pos(current_obs)

        def pulse(magnitude, steps):
            nonlocal current_obs
            for _ in range(int(steps)):
                delta = np.zeros(3, dtype=np.float32)
                delta[axis_index] = float(magnitude)
                current_obs, _, _, _ = env.step(build_flat_reach_action(env, delta))
                env.render()
                if render_sleep_sec > 0.0:
                    time.sleep(render_sleep_sec)

        pulse(float(pulse_magnitude), int(pulse_steps))
        pos_after_positive = get_robot0_eef_pos(current_obs)
        pulse(-float(pulse_magnitude), int(pulse_steps * 2))
        pos_after_negative = get_robot0_eef_pos(current_obs)
        pulse(float(pulse_magnitude), int(pulse_steps))
        recovered_pos = get_robot0_eef_pos(current_obs)

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
    return mapping, current_obs, {
        "calibration_ok": True,
        "pulse_magnitude": float(pulse_magnitude),
        "pulse_steps": int(pulse_steps),
        "world_delta_from_action": mapping.tolist(),
        "probe_logs": probe_logs,
    }


def execute_reach(env, obs, candidate, action_mapping, hover_offset=0.08, settle_offset=0.03, max_steps=120, position_gain=6.0, step_limit=0.03, reach_tolerance=0.04, render_sleep_sec=0.02):
    candidate_pos = np.asarray(candidate["pos"], dtype=float)
    hover_target = candidate_pos + np.array([0.0, 0.0, float(hover_offset)], dtype=float)
    settle_target = candidate_pos + np.array([0.0, 0.0, float(settle_offset)], dtype=float)

    current_obs = obs
    history = []
    reached = True
    for phase_name, phase_target in [("hover", hover_target), ("settle", settle_target)]:
        phase_success = False
        for step_idx in range(int(max_steps)):
            current_pos = get_robot0_eef_pos(current_obs)
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
            current_obs, _, _, _ = env.step(build_flat_reach_action(env, position_delta))
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

        if not phase_success:
            reached = False

    final_pos = get_robot0_eef_pos(current_obs)
    return current_obs, {
        "reach_success": bool(reached),
        "candidate_id": int(candidate["id"]),
        "candidate_type": candidate["grasp_type"],
        "candidate_pos": candidate_pos.tolist(),
        "hover_target": hover_target.tolist(),
        "settle_target": settle_target.tolist(),
        "robot0_eef_final": final_pos.tolist(),
        "final_error_to_settle": float(np.linalg.norm(settle_target - final_pos)),
        "history_tail": history[-10:],
    }


def main():
    parser = argparse.ArgumentParser(description="Run a RoboCasa reach in the live onscreen viewer.")
    parser.add_argument("--task", default="robocasa/CoffeeSetupMug")
    parser.add_argument("--target-label", default="mug")
    parser.add_argument("--candidate-id", type=int, default=None)
    parser.add_argument("--camera-name", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-axis-calibration", action="store_true")
    parser.add_argument("--render-sleep-sec", type=float, default=0.02)
    parser.add_argument("--keep-open-sec", type=float, default=8.0)
    args = parser.parse_args()

    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    vision_config["target_labels"] = [args.target_label]
    vision_config["obstacle_labels"] = []
    if "mug" in args.target_label.lower() or "cup" in args.target_label.lower():
        vision_config["candidate_types"] = ["top_down", "handle_grasp"]
        vision_config["handle_labels"] = ["cup", "mug", "glass cup"]
    if args.allow_download:
        vision_config["local_files_only"] = False

    config = build_env_config(
        task_name=args.task,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        layout=args.layout,
        style=args.style,
    )

    print(colored("Initializing onscreen RoboCasa environment...", "yellow"))
    print(json.dumps(config, indent=2))
    env = robosuite.make(**config)

    obs = env.reset()
    env.render()
    time.sleep(max(args.render_sleep_sec, 0.05))

    rgb, depth = get_rgbd(obs, camera_name=args.camera_name, sim=env.sim)
    camera_config = resolve_camera_config(env, camera_name=args.camera_name, width=args.width, height=args.height)
    loop = VisionPerceptionLoop(
        vision_config=vision_config,
        camera_config=camera_config,
        conf_thresh=float(thresholds_config["CONF_THRESH"]),
    )
    payload = loop.infer_detected_objects(rgb, depth)
    candidate = choose_candidate(payload, candidate_id=args.candidate_id)
    if candidate is None:
        raise RuntimeError("No grasp candidate was produced for the current scene.")

    print(json.dumps({"candidate": candidate, "payload_status": payload.get("status")}, indent=2))

    action_mapping = None
    calibration_summary = None
    current_obs = obs
    if not args.skip_axis_calibration:
        action_mapping, current_obs, calibration_summary = calibrate_position_action_mapping(
            env,
            obs=current_obs,
            render_sleep_sec=float(args.render_sleep_sec),
        )
        print(json.dumps({"axis_calibration": calibration_summary}, indent=2))

    current_obs, reach_summary = execute_reach(
        env,
        obs=current_obs,
        candidate=candidate,
        action_mapping=action_mapping,
        render_sleep_sec=float(args.render_sleep_sec),
    )
    print(json.dumps({"reach_summary": reach_summary}, indent=2))

    keep_open_sec = max(float(args.keep_open_sec), 0.0)
    if keep_open_sec > 0.0:
        print(colored(f"Keeping viewer open for {keep_open_sec:.1f}s", "yellow"))
        end_time = time.time() + keep_open_sec
        while time.time() < end_time:
            env.render()
            time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
