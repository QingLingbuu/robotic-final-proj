"""Calibration helpers for RoboCasa position action axes."""

import time

import numpy as np

from arm.robocasa_execution import build_flat_reach_action, get_robot0_eef_pos


def extract_base_to_world_rotation(action_mapping):
    """Extract the base-to-world rotation from calibrated position action axes."""
    if action_mapping is None:
        return None
    mapping = np.asarray(action_mapping, dtype=float)
    u, _, vt = np.linalg.svd(mapping)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


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
