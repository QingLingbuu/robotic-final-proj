"""Action-space probing helpers for RoboCasa execution debugging."""

import time

import numpy as np

from arm.robocasa_execution import get_robot0_eef_pos, read_gripper_width


def build_axis_pulse_action(env, action_index, magnitude):
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[int(action_index)] = float(magnitude)
    return action


def summarize_action_effect(start_obs, end_obs):
    start_pos = get_robot0_eef_pos(start_obs)
    end_pos = get_robot0_eef_pos(end_obs)
    start_width = read_gripper_width(start_obs)
    end_width = read_gripper_width(end_obs)
    width_delta = None
    if start_width is not None and end_width is not None:
        width_delta = float(end_width - start_width)
    return {
        "eef_delta": (end_pos - start_pos).tolist(),
        "eef_delta_norm": float(np.linalg.norm(end_pos - start_pos)),
        "gripper_width_delta": width_delta,
        "changed_numeric_obs_keys": changed_numeric_obs_keys(start_obs, end_obs),
    }


def changed_numeric_obs_keys(start_obs, end_obs, threshold=1e-6, max_keys=24):
    changed_keys = []
    for key in sorted(set(start_obs.keys()) & set(end_obs.keys())):
        try:
            start_value = np.asarray(start_obs[key], dtype=float)
            end_value = np.asarray(end_obs[key], dtype=float)
        except (TypeError, ValueError):
            continue
        if start_value.shape != end_value.shape or start_value.size == 0:
            continue
        delta_norm = float(np.linalg.norm(end_value - start_value))
        if delta_norm > float(threshold):
            changed_keys.append({"key": key, "delta_norm": delta_norm})
        if len(changed_keys) >= int(max_keys):
            break
    return changed_keys


def infer_action_slices(axis_results, eef_threshold=1e-4, gripper_threshold=1e-4):
    eef_indices = []
    gripper_indices = []
    base_indices = []
    torso_indices = []
    for result in axis_results:
        if abs(float(result.get("positive", {}).get("eef_delta_norm", 0.0))) > float(eef_threshold):
            eef_indices.append(int(result["action_index"]))
        width_delta = result.get("positive", {}).get("gripper_width_delta")
        if width_delta is not None and abs(float(width_delta)) > float(gripper_threshold):
            gripper_indices.append(int(result["action_index"]))
        changed_keys = {
            item["key"] for item in result.get("positive", {}).get("changed_numeric_obs_keys", [])
        }
        changed_keys.update(
            item["key"] for item in result.get("negative", {}).get("changed_numeric_obs_keys", [])
        )
        if "robot0_base_pos" in changed_keys or "robot0_base_quat" in changed_keys:
            base_indices.append(int(result["action_index"]))
        if any("torso" in key.lower() for key in changed_keys):
            torso_indices.append(int(result["action_index"]))
    return {
        "eef_effect_indices": eef_indices,
        "gripper_effect_indices": gripper_indices,
        "base_effect_indices": base_indices,
        "torso_effect_indices": torso_indices,
        "recommended_base_action_slice": [7, 10],
        "recommended_torso_action_index": None,
        "torso_note": "No dedicated torso observation key was detected; keep torso disabled unless manually verified.",
        "expected_right_arm_position_slice": [0, 1, 2],
        "expected_right_arm_orientation_slice": [3, 4, 5],
        "expected_right_gripper_index": 6,
    }


def diagnose_action_space(env, obs, pulse_magnitude=0.1, pulse_steps=4, render_sleep_sec=0.0):
    current_obs = obs
    axis_results = []
    for action_index in range(int(env.action_dim)):
        start_obs = current_obs
        for _ in range(int(pulse_steps)):
            current_obs, _, _, _ = env.step(build_axis_pulse_action(env, action_index, pulse_magnitude))
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)
        positive_obs = current_obs

        for _ in range(int(pulse_steps * 2)):
            current_obs, _, _, _ = env.step(build_axis_pulse_action(env, action_index, -pulse_magnitude))
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)
        negative_obs = current_obs

        for _ in range(int(pulse_steps)):
            current_obs, _, _, _ = env.step(build_axis_pulse_action(env, action_index, pulse_magnitude))
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

        axis_results.append(
            {
                "action_index": int(action_index),
                "positive": summarize_action_effect(start_obs, positive_obs),
                "negative": summarize_action_effect(positive_obs, negative_obs),
            }
        )

    return current_obs, {
        "action_dim": int(env.action_dim),
        "pulse_magnitude": float(pulse_magnitude),
        "pulse_steps": int(pulse_steps),
        "axis_results": axis_results,
        "inferred_slices": infer_action_slices(axis_results),
    }
