"""Small RoboCasa execution helpers shared by demo and tests."""

import numpy as np


def build_flat_reach_action(env, position_delta, orientation_delta=None, gripper_close=0.0):
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[0:3] = np.asarray(position_delta, dtype=np.float32)
    if orientation_delta is not None:
        action[3:6] = np.asarray(orientation_delta, dtype=np.float32)
    action[6] = float(gripper_close)
    return action


def get_robot0_eef_pos(obs):
    if "robot0_eef_pos" not in obs:
        raise RuntimeError(f"Observation missing robot0_eef_pos. Available keys: {sorted(obs.keys())}")
    return np.asarray(obs["robot0_eef_pos"], dtype=float)


def get_robot0_eef_quat(obs):
    key = "robot0_eef_quat_site"
    if key not in obs:
        key = "robot0_eef_quat"
    if key not in obs:
        raise RuntimeError(f"Observation missing eef_quat keys. Available keys: {sorted(obs.keys())}")
    return np.asarray(obs[key], dtype=float)


def read_gripper_width(obs):
    qpos = obs.get("robot0_gripper_qpos")
    if qpos is None or len(qpos) < 2:
        return None
    return float(abs(qpos[0]) + abs(qpos[1]))
