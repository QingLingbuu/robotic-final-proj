"""Experimental base / torso preposition helpers for PandaOmron RoboCasa demos."""

import time

import numpy as np

from arm.calibration import extract_base_to_world_rotation
from arm.robocasa_execution import get_robot0_eef_pos

DEFAULT_BASE_ACTION_SLICE = (7, 10)
DEFAULT_TORSO_ACTION_INDEX = None
BASE_PREPOSITION_GAIN = 4.0
TORSO_PREPOSITION_GAIN = 6.0
BASE_ACTION_LIMIT = 0.35
TORSO_ACTION_LIMIT = 0.25
BASE_MAPPING_TRUST = 0.25
BASE_MAX_WORLD_DELTA = 0.03


def build_base_torso_action(env, base_xy=None, torso=0.0, base_slice=DEFAULT_BASE_ACTION_SLICE, torso_index=DEFAULT_TORSO_ACTION_INDEX):
    action = np.zeros(env.action_dim, dtype=np.float32)
    start, end = base_slice
    if base_xy is not None:
        base_width = max(int(end) - int(start), 0)
        base_command = np.zeros(base_width, dtype=np.float32)
        base_values = np.asarray(base_xy, dtype=np.float32).reshape(-1)
        base_command[: min(base_width, base_values.size)] = base_values[:base_width]
        action[int(start):int(end)] = base_command
    if torso_index is not None and int(torso_index) < int(env.action_dim):
        action[int(torso_index)] = float(torso)
    return action


def get_robot0_base_pos(obs):
    if "robot0_base_pos" not in obs:
        raise RuntimeError(f"Observation missing robot0_base_pos. Available keys: {sorted(obs.keys())}")
    return np.asarray(obs["robot0_base_pos"], dtype=float)


def calibrate_base_action_mapping(
    env,
    obs,
    base_slice=DEFAULT_BASE_ACTION_SLICE,
    pulse_magnitude=0.1,
    pulse_steps=4,
    render_sleep_sec=0.0,
):
    start, end = base_slice
    current_obs = obs
    eef_columns = []
    base_columns = []
    probe_logs = []

    for action_index in range(int(start), int(end)):
        start_eef_pos = get_robot0_eef_pos(current_obs)
        start_base_pos = get_robot0_base_pos(current_obs)

        def pulse(magnitude, steps):
            nonlocal current_obs
            for _ in range(int(steps)):
                action = np.zeros(env.action_dim, dtype=np.float32)
                action[int(action_index)] = float(magnitude)
                current_obs, _, _, _ = env.step(action)
                env.render()
                if render_sleep_sec > 0.0:
                    time.sleep(render_sleep_sec)

        pulse(float(pulse_magnitude), int(pulse_steps))
        positive_eef_pos = get_robot0_eef_pos(current_obs)
        positive_base_pos = get_robot0_base_pos(current_obs)
        pulse(-float(pulse_magnitude), int(pulse_steps * 2))
        negative_eef_pos = get_robot0_eef_pos(current_obs)
        negative_base_pos = get_robot0_base_pos(current_obs)
        pulse(float(pulse_magnitude), int(pulse_steps))
        recovered_eef_pos = get_robot0_eef_pos(current_obs)
        recovered_base_pos = get_robot0_base_pos(current_obs)

        command_delta = float(pulse_magnitude) * float(pulse_steps)
        eef_column = ((positive_eef_pos[:2] - negative_eef_pos[:2]) / 2.0) / max(command_delta, 1e-8)
        base_column = ((positive_base_pos[:2] - negative_base_pos[:2]) / 2.0) / max(command_delta, 1e-8)
        eef_columns.append(eef_column)
        base_columns.append(base_column)
        probe_logs.append(
            {
                "action_index": int(action_index),
                "start_eef_pos": start_eef_pos.tolist(),
                "positive_eef_pos": positive_eef_pos.tolist(),
                "negative_eef_pos": negative_eef_pos.tolist(),
                "recovered_eef_pos": recovered_eef_pos.tolist(),
                "start_base_pos": start_base_pos.tolist(),
                "positive_base_pos": positive_base_pos.tolist(),
                "negative_base_pos": negative_base_pos.tolist(),
                "recovered_base_pos": recovered_base_pos.tolist(),
                "estimated_eef_xy_delta_per_unit_action": eef_column.tolist(),
                "estimated_base_xy_delta_per_unit_action": base_column.tolist(),
            }
        )

    eef_mapping = np.column_stack(eef_columns) if eef_columns else np.zeros((2, 0), dtype=float)
    base_mapping = np.column_stack(base_columns) if base_columns else np.zeros((2, 0), dtype=float)
    return current_obs, {
        "calibration_ok": True,
        "base_slice": [int(start), int(end)],
        "pulse_magnitude": float(pulse_magnitude),
        "pulse_steps": int(pulse_steps),
        "eef_xy_delta_from_base_action": eef_mapping.tolist(),
        "base_xy_delta_from_base_action": base_mapping.tolist(),
        "probe_logs": probe_logs,
    }


def compute_preposition_command(
    eef_pos,
    target_pos,
    action_mapping=None,
    desired_xy_standoff=0.18,
    desired_z=1.12,
    xy_deadband=0.06,
    z_deadband=0.04,
    base_gain=BASE_PREPOSITION_GAIN,
    torso_gain=TORSO_PREPOSITION_GAIN,
    base_action_limit=BASE_ACTION_LIMIT,
    torso_action_limit=TORSO_ACTION_LIMIT,
    base_action_mapping=None,
    action_steps=1,
    base_mapping_trust=BASE_MAPPING_TRUST,
    max_base_world_delta=BASE_MAX_WORLD_DELTA,
):
    eef = np.asarray(eef_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    xy_error_world = target[:2] - eef[:2]
    xy_distance = float(np.linalg.norm(xy_error_world))
    desired_base_world_delta = np.zeros(2, dtype=float)
    if xy_distance > float(desired_xy_standoff) + float(xy_deadband):
        desired_base_world_delta = xy_error_world * ((xy_distance - float(desired_xy_standoff)) / max(xy_distance, 1e-8))

    desired_norm = float(np.linalg.norm(desired_base_world_delta))
    if desired_norm > float(max_base_world_delta) > 0.0:
        desired_base_world_delta = desired_base_world_delta * (float(max_base_world_delta) / desired_norm)

    if base_action_mapping is not None:
        mapping = np.asarray(base_action_mapping, dtype=float)
        if mapping.ndim != 2 or mapping.shape[0] != 2:
            raise ValueError("base_action_mapping must have shape (2, action_width)")
        scaled_mapping = mapping * max(float(action_steps), 1.0)
        base_xy = (
            np.linalg.pinv(scaled_mapping)
            @ desired_base_world_delta
            * float(base_gain)
            * float(base_mapping_trust)
        )
    else:
        base_xy = desired_base_world_delta * float(base_gain)

    rotation = None if base_action_mapping is not None else extract_base_to_world_rotation(action_mapping)
    if rotation is not None:
        base_xyz = rotation.T @ np.array([base_xy[0], base_xy[1], 0.0], dtype=float)
        base_xy = base_xyz[:2]

    z_error = float(target[2] - float(desired_z))
    torso = 0.0 if abs(z_error) <= float(z_deadband) else z_error * float(torso_gain)
    base_xy = np.clip(base_xy, -float(base_action_limit), float(base_action_limit))
    torso = float(np.clip(torso, -float(torso_action_limit), float(torso_action_limit)))
    base_command_norm = float(np.linalg.norm(base_xy))
    return {
        "base_xy": base_xy.tolist(),
        "desired_base_world_delta": desired_base_world_delta.tolist(),
        "max_base_world_delta": float(max_base_world_delta),
        "base_mapping_used": base_action_mapping is not None,
        "base_mapping_action_steps": int(action_steps),
        "base_mapping_trust": float(base_mapping_trust),
        "torso": torso,
        "xy_distance": xy_distance,
        "z_error": z_error,
        "desired_xy_standoff": float(desired_xy_standoff),
        "xy_deadband": float(xy_deadband),
        "base_gain": float(base_gain),
        "base_action_limit": float(base_action_limit),
        "base_command_norm": base_command_norm,
        "should_move": bool(np.linalg.norm(base_xy) > 1e-6 or abs(torso) > 1e-6),
    }


def compute_xy_distance(eef_pos, target_pos):
    eef = np.asarray(eef_pos, dtype=float)
    target = np.asarray(target_pos, dtype=float)
    return float(np.linalg.norm(target[:2] - eef[:2]))


def preposition_base_torso(
    env,
    obs,
    target_pos,
    action_mapping=None,
    steps=30,
    render_sleep_sec=0.005,
    base_slice=DEFAULT_BASE_ACTION_SLICE,
    torso_index=DEFAULT_TORSO_ACTION_INDEX,
    desired_xy_standoff=0.18,
    desired_z=1.12,
    xy_deadband=0.06,
    z_deadband=0.04,
    base_gain=BASE_PREPOSITION_GAIN,
    torso_gain=TORSO_PREPOSITION_GAIN,
    base_action_limit=BASE_ACTION_LIMIT,
    torso_action_limit=TORSO_ACTION_LIMIT,
    base_action_mapping=None,
    base_mapping_trust=BASE_MAPPING_TRUST,
    max_base_world_delta=BASE_MAX_WORLD_DELTA,
):
    start_pos = get_robot0_eef_pos(obs)
    command = compute_preposition_command(
        start_pos,
        target_pos,
        action_mapping=action_mapping,
        desired_xy_standoff=desired_xy_standoff,
        desired_z=desired_z,
        xy_deadband=xy_deadband,
        z_deadband=z_deadband,
        base_gain=base_gain,
        torso_gain=torso_gain,
        base_action_limit=base_action_limit,
        torso_action_limit=torso_action_limit,
        base_action_mapping=base_action_mapping,
        action_steps=steps,
        base_mapping_trust=base_mapping_trust,
        max_base_world_delta=max_base_world_delta,
    )
    base_should_move = float(np.linalg.norm(np.asarray(command["base_xy"], dtype=float))) > 1e-6
    torso_should_move = torso_index is not None and abs(float(command["torso"])) > 1e-6
    effective_should_move = bool(base_should_move or torso_should_move)
    current_obs = obs
    action = build_base_torso_action(
        env,
        base_xy=command["base_xy"],
        torso=command["torso"],
        base_slice=base_slice,
        torso_index=torso_index,
    )
    if effective_should_move:
        for step in range(int(steps)):
            current_obs, _, _, _ = env.step(action)
            if step % 3 == 0:
                env.render()
                if render_sleep_sec > 0.0:
                    time.sleep(render_sleep_sec)

    end_pos = get_robot0_eef_pos(current_obs)
    xy_distance_before = compute_xy_distance(start_pos, target_pos)
    xy_distance_after = compute_xy_distance(end_pos, target_pos)
    return current_obs, {
        "enabled": True,
        "base_slice": [int(base_slice[0]), int(base_slice[1])],
        "torso_index": None if torso_index is None else int(torso_index),
        "steps": int(steps),
        "target_pos": np.asarray(target_pos, dtype=float).tolist(),
        "eef_before": start_pos.tolist(),
        "eef_after": end_pos.tolist(),
        "command": command,
        "torso_command_ignored": bool(torso_index is None and abs(float(command["torso"])) > 1e-6),
        "effective_should_move": effective_should_move,
        "action_norm": float(np.linalg.norm(action)),
        "xy_distance_before": xy_distance_before,
        "xy_distance_after": xy_distance_after,
        "xy_distance_delta": float(xy_distance_after - xy_distance_before),
        "xy_distance_improved": bool(xy_distance_after < xy_distance_before),
        "eef_delta": (end_pos - start_pos).tolist(),
    }
