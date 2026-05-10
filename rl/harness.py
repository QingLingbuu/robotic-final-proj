"""Minimal dry-run and real harness for the milestone-1 RL baseline contract."""

import json
from typing import Any, cast
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from arm.env_wrapper import create_env_wrapper
from runtime.bootstrap import resolve_repo_path
from rl.contracts import load_rl_config, summarize_rl_config, validate_rl_config
from rl.cup_mug_metrics import (
    aggregate_episode_records,
    build_episode_record,
    build_rl_ordering_run_payload,
    write_ordering_metrics,
)
from rl.cup_mug_observation import build_cup_mug_ordering_observation
from rl.cup_mug_policies import select_action
from rl.cup_mug_runner import CupMugOrderingEpisodeRunner


CUP_ORDERING_FAILURE_TYPE_CODES = {
    None: -1.0,
    "": -1.0,
    "all_actions_invalid": 0.0,
    "invalid_action": 1.0,
    "grasp_failure": 2.0,
    "placement_failure": 3.0,
    "timeout": 4.0,
    "collision": 5.0,
    "reach": 6.0,
    "close": 7.0,
    "lift": 8.0,
    "place": 9.0,
}
CUP_ORDERING_CHECKPOINT_MODEL_CACHE = {}

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback as _SB3BaseCallback
    BaseCallback: Any = _SB3BaseCallback
except ModuleNotFoundError:
    PPO = None
    BaseCallback: Any = object

try:
    import gymnasium as gym
    GymEnvBase: Any = gym.Env
except ModuleNotFoundError:
    gym = None
    GymEnvBase: Any = object

try:
    import imageio
except ModuleNotFoundError:
    imageio = None


class RLVectorEnvAdapter(GymEnvBase):
    """SB3-compatible wrapper around the minimal RoboCasa env contract."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config, render_enabled=None):
        self._config = config
        self._backend = config["backend"]
        env_config = dict(config["env"])
        if render_enabled is not None and config["backend"] != "robocasa":
            env_config["has_renderer"] = bool(render_enabled)
            env_config["has_offscreen_renderer"] = False if render_enabled else bool(env_config.get("has_offscreen_renderer", False))
        self._wrapped = create_env_wrapper(backend=config["backend"], config=env_config)
        initial_obs = self._build_observation_vector()

        try:
            from gymnasium import spaces
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("gymnasium must be installed for real RL training/eval.") from exc

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(config["action"]["dimensions"]),),
            dtype=np.float32,
        )
        self._step_count = 0
        self._initial_object_height = self._get_object_height()
        self._last_object_height = self._get_object_height()
        self._last_distance = None
        self._last_gripper_width = self._get_gripper_width()
        self._last_object_pos = self._get_target_pos()

    def _build_observation_vector(self):
        if self._backend == "robocasa":
            obs = getattr(self._wrapped, "obs", {}) or {}
            chunks = []
            for key, value in obs.items():
                if not str(key).startswith("state."):
                    continue
                chunks.append(np.asarray(value, dtype=np.float32).reshape(-1))
            if chunks:
                return np.concatenate(chunks, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)
        return self._wrapped.get_flat_observation()

    def _get_target_pos(self):
        raw_obs = getattr(self._wrapped, "get_raw_observation", lambda: {})() or {}
        for key in ("obj_pos", "cube_pos", "object_pos", "pot_pos"):
            if key in raw_obs:
                return np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)[:3]
        obs = getattr(self._wrapped, "obs", {}) or {}
        for key in ("cube_pos", "object_pos", "pot_pos"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32).reshape(-1)[:3]
        return None

    def _get_eef_pos(self):
        raw_obs = getattr(self._wrapped, "get_raw_observation", lambda: {})() or {}
        if "robot0_eef_pos" in raw_obs:
            return np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
        obs = getattr(self._wrapped, "obs", {}) or {}
        if "robot0_eef_pos" not in obs:
            return None
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]

    def _get_gripper_to_object_vec(self):
        raw_obs = getattr(self._wrapped, "get_raw_observation", lambda: {})() or {}
        for key in ("obj_to_robot0_eef_pos", "gripper_to_cube_pos"):
            if key in raw_obs:
                return np.asarray(raw_obs[key], dtype=np.float32).reshape(-1)[:3]
        obs = getattr(self._wrapped, "obs", {}) or {}
        if "gripper_to_cube_pos" in obs:
            return np.asarray(obs["gripper_to_cube_pos"], dtype=np.float32).reshape(-1)[:3]
        return None

    def _get_object_height(self):
        target_pos = self._get_target_pos()
        if target_pos is None or len(target_pos) < 3:
            return None
        return float(target_pos[2])

    def _get_gripper_width(self):
        raw_obs = getattr(self._wrapped, "get_raw_observation", lambda: {})() or {}
        if "robot0_gripper_qpos" in raw_obs:
            qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
            return float(np.abs(qpos).sum())
        obs = getattr(self._wrapped, "obs", {}) or {}
        if "robot0_gripper_qpos" not in obs:
            return None
        qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        return float(np.abs(qpos).sum())

    def _shape_reward(self, base_reward, done):
        reward = float(base_reward)
        coeffs = self._config.get("reward", {}).get("coefficients", {})

        target_pos = self._get_target_pos()
        eef_pos = self._get_eef_pos()
        gripper_to_object = self._get_gripper_to_object_vec()
        distance = None
        if gripper_to_object is not None:
            distance = float(np.linalg.norm(gripper_to_object))
        elif target_pos is not None and eef_pos is not None:
            distance = float(np.linalg.norm(target_pos - eef_pos))

        if distance is not None:
            if self._last_distance is not None:
                distance_improvement = self._last_distance - distance
                reward += float(coeffs.get("approach_target", 0.0)) * distance_improvement * 10.0
            reward += float(coeffs.get("approach_target", 0.0)) * max(0.0, 0.2 - distance)
            self._last_distance = distance

        gripper_width = self._get_gripper_width()
        if distance is not None and distance < 0.08 and gripper_width is not None:
            if self._last_gripper_width is not None:
                closure_delta = self._last_gripper_width - gripper_width
                reward += float(coeffs.get("successful_close", 0.0)) * max(0.0, closure_delta) * 5.0
            if gripper_width < 0.04:
                reward += float(coeffs.get("successful_close", 0.0))
        if gripper_width is not None:
            self._last_gripper_width = gripper_width

        current_height = self._get_object_height()
        current_object_pos = self._get_target_pos()
        if current_height is not None and self._last_object_height is not None:
            height_delta = max(0.0, current_height - self._last_object_height)
            reward += float(coeffs.get("successful_lift", 0.0)) * height_delta * 10.0
        if current_height is not None and self._initial_object_height is not None:
            lift_from_start = max(0.0, current_height - self._initial_object_height)
            reward += float(coeffs.get("successful_lift", 0.0)) * lift_from_start * 2.0
        if current_object_pos is not None and self._last_object_pos is not None:
            object_motion = float(np.linalg.norm(current_object_pos - self._last_object_pos))
            if distance is not None and distance < 0.08 and gripper_width is not None:
                if gripper_width < 0.04 and object_motion > 0.002:
                    reward += float(coeffs.get("successful_close", 0.0)) * object_motion * 2.0
                elif gripper_width > 0.06 and object_motion > 0.01:
                    reward -= abs(float(coeffs.get("collision_penalty", 0.0))) * object_motion * 5.0
        if current_height is not None:
            if (
                self._last_object_height is not None
                and self._initial_object_height is not None
                and self._last_object_height > self._initial_object_height + 0.03
                and current_height < self._last_object_height - 0.01
            ):
                reward -= abs(float(coeffs.get("collision_penalty", 0.0))) * 2.0
            self._last_object_height = current_height
        if current_object_pos is not None:
            self._last_object_pos = current_object_pos

        reward += float(coeffs.get("timeout_penalty", 0.0))

        return reward

    def _is_grasp_success(self):
        distance = self._last_distance
        gripper_width = self._get_gripper_width()
        current_height = self._get_object_height()
        if current_height is None or self._initial_object_height is None or gripper_width is None:
            return False
        lift_from_start = current_height - self._initial_object_height
        close_enough = distance is not None and distance < 0.10
        gripper_closed = gripper_width < 0.05
        lifted = lift_from_start > 0.03
        return bool(close_enough and gripper_closed and lifted)

    def reset(self, seed=None, options=None):
        del options
        self._wrapped.reset(seed=seed)
        self._step_count = 0
        self._initial_object_height = self._get_object_height()
        self._last_object_height = self._get_object_height()
        target_pos = self._get_target_pos()
        eef_pos = self._get_eef_pos()
        gripper_to_object = self._get_gripper_to_object_vec()
        if gripper_to_object is not None:
            self._last_distance = float(np.linalg.norm(gripper_to_object))
        else:
            self._last_distance = None if target_pos is None or eef_pos is None else float(np.linalg.norm(target_pos - eef_pos))
        self._last_gripper_width = self._get_gripper_width()
        self._last_object_pos = self._get_target_pos()
        return self._build_observation_vector(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._backend == "robocasa" and action.shape[-1] == 4:
            full_action = np.zeros(12, dtype=np.float32)
            full_action[0] = action[0]  # action.gripper_close
            full_action[1:4] = action[1:4]  # action.end_effector_position
            action = full_action

        _, reward, done, info = self._wrapped.step(action)
        self._step_count += 1
        obs = self._build_observation_vector()
        terminated = bool(done)
        truncated = False
        shaped_reward = self._shape_reward(reward, terminated or truncated)
        if not terminated and not truncated and self._is_grasp_success():
            terminated = True
            info = dict(info)
            info["success"] = True
            shaped_reward += float(self._config.get("reward", {}).get("coefficients", {}).get("successful_lift", 0.0)) * 2.0
        return obs, float(shaped_reward), terminated, truncated, info

    def render(self):
        return self._wrapped.render()

    def close(self):
        env = getattr(self._wrapped, "env", None)
        if env is not None and hasattr(env, "close"):
            env.close()


class CupOrderingTrainEnv(GymEnvBase):
    """Minimal discrete-action PPO environment for cup-ordering policy training."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, config):
        try:
            from gymnasium import spaces
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("gymnasium must be installed for cup-ordering PPO training.") from exc

        self._config = config
        self._max_targets = int(config["max_targets"])
        self._action_count = int(config["action_space"]["n"])
        self._seed_index = 0
        self._episode_seed = None
        self._rng = random.Random()
        self._last_observation = None
        self._runner = CupMugOrderingEpisodeRunner(
            build_cup_mug_ordering_observation,
            self._training_executor,
            reward_weights=config["reward"]["coefficients"],
            allowed_failed_attempts=int(config.get("train", {}).get("allowed_failed_attempts", 2)),
            max_targets=self._max_targets,
        )
        bootstrap_seed = int(config["eval"]["seed_set"][0])
        bootstrap_observation = self._reset_episode(bootstrap_seed)
        bootstrap_vector = flatten_cup_ordering_observation(bootstrap_observation)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=bootstrap_vector.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._action_count)

    def reset(self, seed=None, options=None):
        del options
        if seed is None:
            seed_set = list(self._config["eval"]["seed_set"])
            seed = int(seed_set[self._seed_index % len(seed_set)])
            self._seed_index += 1
        observation = self._reset_episode(int(seed))
        return flatten_cup_ordering_observation(observation), {"seed": int(seed)}

    def step(self, action):
        resolved_action = int(np.asarray(action).reshape(-1)[0])
        result = self._runner.step(resolved_action)
        observation = result["observation"]
        self._last_observation = observation
        info = {
            "success": bool(result["success"]),
            "failure_type": result["failure_type"],
            "step_log": result["step_log"],
        }
        if result["summary"] is not None:
            info["episode_summary"] = result["summary"]
        return flatten_cup_ordering_observation(observation), float(result["reward"]), bool(result["done"]), False, info

    def render(self):
        if self._last_observation is None:
            return None
        frame = _draw_cup_ordering_frame(
            self._last_observation,
            title=f"cup_ordering train seed={self._episode_seed}",
        )
        return np.asarray(frame)

    def close(self):
        return None

    def _reset_episode(self, seed):
        self._episode_seed = int(seed)
        self._rng.seed(self._episode_seed)
        targets, assignments = _cup_ordering_fixture(self._episode_seed)
        observation = self._runner.reset(
            targets,
            assignments=assignments,
            sorting_metadata=_cup_ordering_sorting_metadata(),
            episode_id=f"train-seed-{self._episode_seed}",
            seed=self._episode_seed,
        )
        self._last_observation = observation
        return observation

    def _training_executor(self, slot, observation, step_index):
        del observation
        return _simulate_cup_ordering_execution(slot, step_index, self._rng)


def build_selector_observation(payload, fsm_state, max_candidates):
    """Build a padded, selector-friendly observation from a perception payload."""
    candidates = list((payload or {}).get("grasp_candidates") or [])
    resolved_max = max(int(max_candidates), 0)
    features = []
    for candidate in candidates[:resolved_max]:
        features.append(
            {
                "id": int(candidate["id"]),
                "score": float(candidate["score"]),
                "gripper_width": float(candidate["gripper_width"]),
                "pos": [float(value) for value in candidate["pos"]],
                "grasp_type": str(candidate["grasp_type"]),
                "mask": 1.0,
            }
        )
    while len(features) < resolved_max:
        features.append(
            {
                "id": -1,
                "score": 0.0,
                "gripper_width": 0.0,
                "pos": [0.0, 0.0, 0.0],
                "grasp_type": "padding",
                "mask": 0.0,
            }
        )
    return {
        "fsm_state": str(fsm_state),
        "candidate_count": min(len(candidates), resolved_max),
        "candidate_features": features,
    }


class ProgressCallback(BaseCallback):
    """Simple heartbeat callback so long RoboCasa runs show visible progress."""

    def __init__(self, print_freq):
        super().__init__()
        self.print_freq = max(int(print_freq), 1)
        self.history = []

    def _on_step(self) -> bool:
        logger_values = dict(getattr(self.logger, "name_to_value", {}) or {})
        record = {"timesteps": int(self.num_timesteps)}
        for key in (
            "rollout/ep_len_mean",
            "rollout/ep_rew_mean",
            "train/approx_kl",
            "train/clip_fraction",
            "train/entropy_loss",
            "train/explained_variance",
            "train/learning_rate",
            "train/loss",
            "train/policy_gradient_loss",
            "train/value_loss",
        ):
            value = logger_values.get(key)
            record[key] = None if value is None else float(value)
        if any(value is not None for key, value in record.items() if key != "timesteps"):
            self.history.append(record)
        if self.num_timesteps % self.print_freq == 0:
            print(f"[train] timesteps={self.num_timesteps}", flush=True)
        return True


def ensure_artifact_dirs(config):
    root_dir = resolve_repo_path(config["artifacts"]["root_dir"])
    checkpoint_dir = root_dir / config["artifacts"]["checkpoint_dir"]
    metrics_dir = root_dir / config["artifacts"]["metrics_dir"]
    video_dir = root_dir / config["artifacts"]["video_dir"]
    report_dir_name = config["artifacts"].get("report_dir")
    report_dir = root_dir / report_dir_name if report_dir_name else None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root_dir": root_dir,
        "checkpoint_dir": checkpoint_dir,
        "metrics_dir": metrics_dir,
        "video_dir": video_dir,
        "report_dir": report_dir,
    }


def _is_cup_ordering_config(config):
    return config.get("mode") == "cup_ordering"


def _cup_ordering_scene_config(config):
    scene = dict(config.get("scene", {}))
    return {
        "layout_and_style_ids": list(scene.get("layout_and_style_ids", [])),
        "num_mugs": int(scene.get("num_mugs", 0)),
        "num_cups": int(scene.get("num_cups", 0)),
    }


def _cup_ordering_sorting_metadata():
    return {
        "mug_1": {"has_handle": True, "target_zone": "handled", "recommended_grasp": "handle_top_down"},
        "mug_2": {"has_handle": True, "target_zone": "handled", "recommended_grasp": "handle_top_down"},
        "cup_1": {"has_handle": False, "target_zone": "plain", "recommended_grasp": "top_down"},
        "cup_2": {"has_handle": False, "target_zone": "plain", "recommended_grasp": "top_down"},
        "cup_3": {"has_handle": False, "target_zone": "plain", "recommended_grasp": "top_down"},
    }


def _cup_ordering_fixture(seed):
    rng = random.Random(int(seed))
    object_specs = [
        ("mug_1", "mug", True),
        ("mug_2", "mug", True),
        ("cup_1", "cup", False),
        ("cup_2", "cup", False),
        ("cup_3", "cup", False),
    ]
    targets = []
    assignments = []
    for index, (name, label, has_handle) in enumerate(object_specs):
        conf = round(0.72 + 0.02 * rng.random() + 0.015 * index, 4)
        candidate_score = round(0.66 + 0.03 * rng.random() + 0.02 * index, 4)
        reachability = round(0.55 + 0.04 * rng.random() + 0.03 * (4 - index), 4)
        pos = [round(0.35 + 0.08 * index, 4), round(-0.2 + 0.05 * ((index % 2) * 2 - 1), 4), 0.95]
        targets.append(
            {
                "label": label,
                "conf": conf,
                "pos": pos,
                "visible": True,
                "sim_object_name": name,
                "sim_has_handle": has_handle,
                "grasp_candidates": [
                    {
                        "id": index + 1,
                        "grasp_type": "handle_top_down" if has_handle else "top_down",
                        "score": candidate_score,
                    }
                ],
            }
        )
        assignments.append(
            {
                "label": label,
                "sim_object_name": name,
                "has_handle": has_handle,
                "strategy": "handle_top_down" if has_handle else "top_down",
                "candidate_score": candidate_score,
                "reachability": reachability,
                "distance_to_place": round(0.1 + 0.02 * index, 4),
                "policy": None,
            }
        )
    return targets, assignments


def _simulate_cup_ordering_execution(slot, step_index, rng):
    candidate_score = float(slot.get("candidate_score", 0.0) or 0.0)
    confidence = float(slot.get("conf", 0.0) or 0.0)
    reachability = float(slot.get("reachability", 0.0) or 0.0)
    retry_count = int(slot.get("retry_count", 0) or 0)
    handle_bonus = 0.08 if bool(slot.get("has_handle")) and int(step_index) < 2 else 0.0
    success_score = (0.45 * candidate_score) + (0.30 * confidence) + (0.25 * reachability) + handle_bonus
    success_score -= 0.12 * retry_count
    success_probability = max(0.05, min(0.98, success_score))
    sampled_success = rng.random() < success_probability
    failure_type = None
    if not sampled_success:
        failure_type = "grasp_failure" if candidate_score < 0.75 or reachability < 0.7 else "placement_failure"
    return {
        "low_level_success": sampled_success,
        "correct_zone": sampled_success,
        "failure_type": failure_type,
        "trace": {
            "selected_slot": int(slot["slot_index"]),
            "object_id": slot.get("object_id"),
            "step_index": int(step_index),
            "success_probability": round(success_probability, 4),
            "sampled_success": bool(sampled_success),
            "grasp_strategy": slot.get("recommended_grasp"),
            "place_zone": slot.get("place_zone_id"),
        },
    }


def _cup_ordering_executor(slot, observation, step_index):
    del observation
    rng = random.Random(int(step_index) + int(slot.get("slot_index", 0)))
    return _simulate_cup_ordering_execution(slot, step_index, rng)


def _cup_ordering_rl_action(observation, seed, step_index):
    valid_actions = [index for index, value in enumerate(observation.get("action_mask", [])) if bool(value)]
    if not valid_actions:
        return None, {"policy": "rl", "reason": "all_actions_invalid", "valid_actions": []}

    def slot_for(index):
        for slot in observation.get("slots", []):
            if int(slot.get("slot_index", -1)) == int(index):
                return slot
        return {}

    ranked = []
    for action_index in valid_actions:
        slot = slot_for(action_index)
        handle_rank = 0 if slot.get("has_handle") is True else 1
        ranked.append(
            (
                handle_rank,
                -float(slot.get("candidate_score", 0.0) or 0.0),
                -float(slot.get("conf", 0.0) or 0.0),
                -float(slot.get("reachability", 0.0) or 0.0),
                int(slot.get("retry_count", 0) or 0),
                int(action_index),
            )
        )
    ranked.sort()
    action = int(ranked[0][-1])
    return int(action), {
        "policy": "rl",
        "reason": "sanity_stub_policy",
        "valid_actions": list(valid_actions),
        "stub_seed": int(seed),
        "stub_step_index": int(step_index),
        "ranked_actions": [int(item[-1]) for item in ranked],
        "selection_mode": "handle_first_quality_biased_stub",
    }


def _cup_ordering_valid_actions(observation):
    return [index for index, value in enumerate(observation.get("action_mask", [])) if bool(value)]


def _bool_feature(value, unknown=-1.0):
    if value is None:
        return float(unknown)
    return 1.0 if bool(value) else 0.0


def _failure_type_code(value):
    return float(CUP_ORDERING_FAILURE_TYPE_CODES.get(value, 99.0))


def flatten_cup_ordering_observation(observation):
    """Flatten the fixed 5-slot cup-ordering observation for checkpoint policies."""

    features = []
    slots = list(observation.get("slots", []))[:5]
    for slot in slots:
        pos = list(slot.get("pos") or [0.0, 0.0, 0.0])
        pos = (pos[:3] + [0.0] * max(0, 3 - len(pos)))[:3]
        features.extend(
            [
                float(slot.get("slot_index", 0) or 0),
                _bool_feature(slot.get("visible")),
                _bool_feature(slot.get("finished")),
                _bool_feature(slot.get("has_handle")),
                float(slot.get("type_id", -1) or -1),
                float(slot.get("conf", 0.0) or 0.0),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(slot.get("candidate_score", 0.0) or 0.0),
                float(slot.get("reachability", 0.0) or 0.0),
                float(slot.get("place_zone_id", -1) or -1),
                float(slot.get("retry_count", 0) or 0),
                _bool_feature(slot.get("valid_action")),
            ]
        )
    features.extend([0.0] * max(0, 5 - len(slots)) * 14)

    global_payload = dict(observation.get("global", {}) or {})
    last_action = global_payload.get("last_action")
    features.extend(
        [
            float(global_payload.get("remaining_count", 0) or 0),
            float(global_payload.get("completed_count", 0) or 0),
            float(global_payload.get("step_index", 0) or 0),
            -1.0 if last_action is None else float(last_action),
            _bool_feature(global_payload.get("last_success")),
            _failure_type_code(global_payload.get("last_failure_type")),
        ]
    )
    action_mask = list(observation.get("action_mask", []))[:5]
    action_mask = action_mask + [0] * max(0, 5 - len(action_mask))
    features.extend([1.0 if bool(value) else 0.0 for value in action_mask[:5]])
    return np.asarray(features, dtype=np.float32)


def _coerce_discrete_action(raw_action):
    raw = np.asarray(raw_action).reshape(-1)
    if raw.size == 0:
        raise RuntimeError("Checkpoint policy returned an empty action.")
    return int(raw[0])


def _load_cup_ordering_checkpoint_model(checkpoint_path):
    if PPO is None:
        raise ModuleNotFoundError("stable_baselines3 must be installed for cup-ordering checkpoint inference.")
    resolved_path = str(resolve_repo_path(str(checkpoint_path)))
    if resolved_path not in CUP_ORDERING_CHECKPOINT_MODEL_CACHE:
        CUP_ORDERING_CHECKPOINT_MODEL_CACHE[resolved_path] = PPO.load(resolved_path)
    return CUP_ORDERING_CHECKPOINT_MODEL_CACHE[resolved_path]


def _predict_cup_ordering_checkpoint_action(observation, metadata):
    valid_actions = _cup_ordering_valid_actions(observation)
    if not valid_actions:
        return None, {
            "policy": "rl",
            "reason": "all_actions_invalid",
            "valid_actions": [],
            "rl_policy_mode": "checkpoint",
            "rl_checkpoint": metadata.get("checkpoint_path"),
            "checkpoint_interface": metadata,
        }

    observation_vector = flatten_cup_ordering_observation(observation)
    model = _load_cup_ordering_checkpoint_model(metadata["checkpoint_path"])
    raw_action, _states = model.predict(observation_vector, deterministic=True)
    action = _coerce_discrete_action(raw_action)
    return action, {
        "policy": "rl",
        "reason": "checkpoint_policy",
        "valid_actions": list(valid_actions),
        "predicted_action": int(action),
        "predicted_action_valid": int(action) in set(valid_actions),
        "selection_mode": "checkpoint_inference",
        "observation_vector_length": int(observation_vector.shape[0]),
        "rl_policy_mode": "checkpoint",
        "rl_checkpoint": metadata.get("checkpoint_path"),
        "checkpoint_interface": metadata,
    }


def inspect_cup_ordering_checkpoint_artifact(checkpoint_path):
    if checkpoint_path is None or str(checkpoint_path).strip() in {"", "cup-ordering-dry-run"}:
        return {
            "policy_mode": "stub",
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "artifact_kind": "cup_ordering_stub_policy",
            "schema_version": 1,
            "supports_inference": False,
            "unsupported_reason": "sanity_stub_policy",
        }

    resolved_path = resolve_repo_path(str(checkpoint_path))
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cup-ordering checkpoint not found: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if payload.get("mode") != "cup_ordering":
            raise ValueError(f"Cup-ordering checkpoint JSON missing mode='cup_ordering': {resolved_path}")
        return {
            "policy_mode": "checkpoint",
            "checkpoint_path": str(resolved_path),
            "artifact_kind": str(payload.get("artifact_kind") or "cup_ordering_sanity_checkpoint"),
            "schema_version": int(payload.get("schema_version", 1)),
            "supports_inference": bool(payload.get("supports_inference", False)),
            "unsupported_reason": str(
                payload.get("unsupported_reason") or "checkpoint_artifact_has_no_online_inference_backend"
            ),
        }

    if suffix == ".zip":
        return {
            "policy_mode": "checkpoint",
            "checkpoint_path": str(resolved_path),
            "artifact_kind": "cup_ordering_policy_checkpoint_zip",
            "schema_version": 1,
            "supports_inference": True,
            "unsupported_reason": None,
        }

    raise ValueError(f"Unsupported cup-ordering checkpoint format: {resolved_path}")


def select_cup_ordering_rl_action(observation, seed, step_index, checkpoint_path=None, checkpoint_metadata=None):
    metadata = checkpoint_metadata or inspect_cup_ordering_checkpoint_artifact(checkpoint_path)
    if metadata.get("policy_mode") == "checkpoint":
        if not metadata.get("supports_inference"):
            raise RuntimeError(
                "Cup-ordering checkpoint interface is reserved but inference is unavailable: "
                f"{metadata.get('unsupported_reason')} ({metadata.get('checkpoint_path')})"
            )
        return _predict_cup_ordering_checkpoint_action(observation, metadata)

    action, debug = _cup_ordering_rl_action(observation, seed=seed, step_index=step_index)
    debug["rl_policy_mode"] = metadata.get("policy_mode", "stub")
    debug["rl_checkpoint"] = metadata.get("checkpoint_path")
    debug["checkpoint_interface"] = metadata
    return action, debug


def classify_cup_ordering_failure(*, failure_reason=None, failure_phase=None, retry_refresh_failure_reason=None):
    reason = str(failure_reason or retry_refresh_failure_reason or "")
    phase = str(failure_phase or "")

    if reason == "rl_checkpoint_inference_unsupported":
        return "checkpoint_interface"
    if reason == "all_actions_invalid":
        return "policy_no_valid_action"
    if reason in {"slot_out_of_range", "slot_invalid", "target_missing_for_slot"}:
        return "slot_resolution"
    if phase == "candidate_build" or phase == "candidate_refresh":
        return "candidate_build"
    if phase in {"reach", "close", "lift", "place"}:
        return f"execution_{phase}"
    if reason.startswith("Handle candidate") or reason.startswith("Top-down candidate"):
        return "candidate_threshold"
    if reason.startswith("No reachable candidate") or reason.startswith("Could not resolve live target"):
        return "candidate_resolution"
    if reason:
        return "runtime_error"
    return None


def _cup_ordering_frame_dir(config, dirs, stem):
    return dirs["root_dir"] / "frames" / stem


def _build_cup_ordering_real_render_env_config(config, seed):
    controller_configs = None
    try:
        from robosuite.controllers import load_composite_controller_config
        controller_configs = load_composite_controller_config(robot="PandaOmron")
    except ModuleNotFoundError:
        controller_configs = None

    return {
        "task_name": config["task_name"],
        "split": None,
        "robots": "PandaOmron",
        "camera_names": "robot0_agentview_center",
        "camera_widths": 640,
        "camera_heights": 480,
        "camera_depths": True,
        "render_mode": None,
        "layout_ids": None,
        "style_ids": None,
        "layout_and_style_ids": list(config.get("scene", {}).get("layout_and_style_ids", [])),
        "seed": None if seed is None else int(seed),
        "obj_instance_split": None,
        "translucent_robot": True,
    }


def _close_wrapper_env(wrapper):
    env = getattr(wrapper, "env", None)
    if env is not None and hasattr(env, "close"):
        env.close()


def _try_capture_real_cup_ordering_background_frame(config, seed):
    try:
        env_config = _build_cup_ordering_real_render_env_config(config, seed=seed)
        wrapper = create_env_wrapper(backend="robocasa", config=env_config, camera_config={})
    except Exception as exc:  # noqa: BLE001 - best effort background acquisition
        return None, {"frame_source": "synthetic_fallback", "real_render_error": f"wrapper_init_failed: {exc}"}

    try:
        wrapper.reset(seed=seed)
        wrapper.render()
        rgb, _, _ = wrapper.get_observation()
        if rgb is None:
            return None, {"frame_source": "synthetic_fallback", "real_render_error": "wrapper_returned_no_rgb"}
        return np.asarray(rgb, dtype=np.uint8), {"frame_source": "real_robocasa_camera", "real_render_error": None}
    except Exception as exc:  # noqa: BLE001 - best effort background acquisition
        return None, {"frame_source": "synthetic_fallback", "real_render_error": f"wrapper_capture_failed: {exc}"}
    finally:
        _close_wrapper_env(wrapper)


def _draw_cup_ordering_frame(observation, step_log=None, title=None, background_frame=None):
    width = 1000
    height = 620
    if background_frame is not None:
        background = Image.fromarray(np.asarray(background_frame, dtype=np.uint8)).convert("RGB").resize((width, 340))
        image = Image.new("RGB", (width, height), color=(245, 247, 250))
        image.paste(background, (0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 340, width, height], fill=(245, 247, 250))
        top = 390
    else:
        image = Image.new("RGB", (width, height), color=(245, 247, 250))
        draw = ImageDraw.Draw(image)
        top = 60
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), title or "cup_ordering dry-run", fill=(20, 20, 20))
    slot_width = 180
    for slot in observation.get("slots", []):
        slot_index = int(slot.get("slot_index", 0))
        left = 20 + slot_index * (slot_width + 10)
        right = left + slot_width
        bottom = top + 160
        is_selected = step_log is not None and int(step_log.get("selected_slot", -1)) == slot_index
        has_handle = slot.get("has_handle") is True
        fill = (222, 242, 255) if has_handle else (255, 240, 220)
        if not bool(slot.get("valid_action")):
            fill = (228, 228, 228)
        if is_selected:
            fill = (190, 235, 190)
        draw.rounded_rectangle([left, top, right, bottom], radius=14, fill=fill, outline=(80, 80, 80), width=2)
        draw.text((left + 10, top + 10), f"slot {slot_index}", fill=(30, 30, 30))
        draw.text((left + 10, top + 38), f"id: {slot.get('object_id')}", fill=(30, 30, 30))
        draw.text((left + 10, top + 66), f"grasp: {slot.get('recommended_grasp')}", fill=(30, 30, 30))
        draw.text((left + 10, top + 94), f"zone: {slot.get('place_zone_id')}", fill=(30, 30, 30))
        draw.text((left + 10, top + 122), f"score: {slot.get('candidate_score')}", fill=(30, 30, 30))
        draw.text((left + 10, top + 150), f"valid: {slot.get('valid_action')}", fill=(30, 30, 30))
    if step_log is not None:
        draw.text((20, height - 30), f"selected_slot={step_log.get('selected_slot')} object={step_log.get('object_id')} reward={step_log.get('reward')}", fill=(20, 20, 20))
    return np.asarray(image, dtype=np.uint8)


def _append_video_frame(video_writer, frame):
    if video_writer is not None:
        video_writer.append_data(np.asarray(frame, dtype=np.uint8))


def _save_frame_png(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(output_path)
    return output_path


def _cup_ordering_report_path(config, dirs, stem):
    report_dir = dirs.get("report_dir") or dirs["root_dir"]
    return report_dir / f"{stem}.json"


def _cup_ordering_sidecar_path(config, dirs, stem):
    report_dir = dirs.get("report_dir") or dirs["root_dir"]
    return report_dir / f"{stem}.sidecar.json"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_episode_sidecar(policy_name, summary, background_metadata, frame_paths, video_output_path=None):
    step_logs = list(summary.get("step_logs", []))
    return {
        "policy": policy_name,
        "episode_id": summary.get("episode_id"),
        "selected_order": list(summary.get("selected_order", [])),
        "object_ids": [entry.get("object_id") for entry in step_logs],
        "has_handle": [entry.get("has_handle") for entry in step_logs],
        "grasp_strategy": [entry.get("grasp_strategy") for entry in step_logs],
        "place_zone": [entry.get("place_zone") for entry in step_logs],
        "success": bool(summary.get("success")),
        "failure_type": [entry.get("failure_type") for entry in step_logs],
        "frame_paths": list(frame_paths),
        "video_output_path": video_output_path,
        **background_metadata,
    }


def _run_cup_ordering_dry_run(
    config,
    policy_name,
    episodes,
    seed_start,
    save_report=False,
    report_stem=None,
    render_override=None,
    save_video=False,
    video_path=None,
    checkpoint_path=None,
    checkpoint_metadata=None,
):
    dirs = ensure_artifact_dirs(config)
    scene_config = _cup_ordering_scene_config(config)
    sorting_metadata = _cup_ordering_sorting_metadata()
    episode_records = []
    should_render = bool(render_override)
    report_payload = {
        "mode": "cup_ordering",
        "policy": policy_name,
        "episodes": int(episodes),
        "scene_config": scene_config,
        "episode_traces": [],
        "sidecar_paths": [],
        "video_output_path": None,
        "video_skip_reason": None,
    }
    frame_dir = _cup_ordering_frame_dir(config, dirs, f"{policy_name}-episodes-{int(episodes)}") if should_render else None
    video_writer = None
    resolved_video_path = None
    if save_video:
        if imageio is None:
            report_payload["video_skip_reason"] = "imageio_not_installed"
        else:
            if video_path is None:
                resolved_video_path = dirs["video_dir"] / f"{policy_name}-episodes-{int(episodes)}.mp4"
            else:
                resolved_video_path = resolve_repo_path(video_path)
            resolved_video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(str(resolved_video_path), fps=2)
            report_payload["video_output_path"] = str(resolved_video_path)
    seeds = [int(seed_start) + index for index in range(int(episodes))]
    for episode_index, seed in enumerate(seeds, start=1):
        targets, assignments = _cup_ordering_fixture(seed)
        episode_rng = random.Random(int(seed))
        background_frame = None
        background_metadata = {"frame_source": "synthetic_fallback", "real_render_error": None}
        if should_render:
            background_frame, background_metadata = _try_capture_real_cup_ordering_background_frame(config, seed=seed)
        runner = CupMugOrderingEpisodeRunner(
            build_cup_mug_ordering_observation,
            lambda slot, observation, step_index, rng=episode_rng: _simulate_cup_ordering_execution(slot, step_index, rng),
            reward_weights=config["reward"]["coefficients"],
            allowed_failed_attempts=0,
            max_targets=int(config["max_targets"]),
        )
        runner.reset(targets, assignments, sorting_metadata, episode_id=f"episode-{episode_index:03d}", seed=seed)
        done = False
        episode_frame_paths = []
        while not done:
            observation = runner.get_observation()
            if str(policy_name) == "rl":
                action, debug = select_cup_ordering_rl_action(
                    observation,
                    seed=seed,
                    step_index=runner._episode_state["step_index"],
                    checkpoint_path=checkpoint_path,
                    checkpoint_metadata=checkpoint_metadata,
                )
            else:
                action, debug = select_action(observation, policy_name, rng=random.Random(seed + runner._episode_state["step_index"]))
            if action is None:
                result = runner.step(0)
            else:
                result = runner.step(action)
                if result["step_log"] is not None:
                    result["step_log"]["policy"] = policy_name
                    result["step_log"]["policy_debug"] = debug
            if should_render and result["step_log"] is not None:
                frame = _draw_cup_ordering_frame(
                    result["observation"],
                    step_log=result["step_log"],
                    title=f"{policy_name} episode {episode_index:03d} seed={seed}",
                    background_frame=background_frame,
                )
                frame_path = _save_frame_png(frame, frame_dir / f"episode-{episode_index:03d}" / f"step-{result['step_log']['step_index']:03d}.png")
                episode_frame_paths.append(str(frame_path))
                _append_video_frame(video_writer, frame)
            done = bool(result["done"])
        summary = result["summary"]
        episode_records.append(build_episode_record(summary, policy=policy_name, duration_sec=0.0))
        sidecar_stem = f"{policy_name}-episode-{episode_index:03d}"
        sidecar_payload = _build_episode_sidecar(
            policy_name=policy_name,
            summary=summary,
            background_metadata=background_metadata,
            frame_paths=episode_frame_paths,
            video_output_path=str(resolved_video_path) if resolved_video_path is not None else None,
        )
        sidecar_path = _write_json(_cup_ordering_sidecar_path(config, dirs, sidecar_stem), sidecar_payload)
        report_payload["episode_traces"].append(
            {
                "episode_id": summary["episode_id"],
                "seed": seed,
                "selected_order": summary["selected_order"],
                "step_logs": summary["step_logs"],
                "success": summary["success"],
                "frame_paths": episode_frame_paths,
                "sidecar_path": str(sidecar_path),
                **background_metadata,
            }
        )
        report_payload["sidecar_paths"].append(str(sidecar_path))

    metrics_payload = aggregate_episode_records(
        episode_records,
        policy=policy_name,
        episodes=int(episodes),
        seed_start=int(seed_start),
        max_targets=int(config["max_targets"]),
        scene_config=scene_config,
    )
    metrics_path = write_ordering_metrics(
        metrics_payload,
        output_dir=dirs["metrics_dir"],
        file_name=f"{policy_name}-episodes-{int(episodes)}.json",
    )
    report_path = None
    if save_report:
        report_stem = report_stem or f"{policy_name}-episodes-{int(episodes)}-trace"
        report_path = _write_json(_cup_ordering_report_path(config, dirs, report_stem), report_payload)
    if video_writer is not None:
        video_writer.close()
    return metrics_path, metrics_payload, report_path, resolved_video_path


def _write_cup_ordering_train_artifact(config, step_count=None):
    dirs = ensure_artifact_dirs(config)
    resolved_steps = int(step_count if step_count is not None else config.get("max_targets", 5))
    checkpoint_path = dirs["checkpoint_dir"] / f"cup-ordering-sanity-step-{resolved_steps}.ckpt.json"
    payload = {
        "mode": "cup_ordering",
        "backend": config["backend"],
        "task_name": config["task_name"],
        "artifact_kind": config["policies"]["checkpoint_interface"]["artifact_kind"],
        "schema_version": int(config["policies"]["checkpoint_interface"]["schema_version"]),
        "action_space": f"Discrete({int(config['action_space']['n'])})",
        "max_targets": int(config["max_targets"]),
        "step_count": resolved_steps,
        "sanity": True,
        "supports_inference": False,
        "unsupported_reason": "cup_ordering_sanity_artifact_has_no_model_weights",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return checkpoint_path


def _write_cup_ordering_training_summary(config, checkpoint_path, total_timesteps):
    dirs = ensure_artifact_dirs(config)
    summary_path = dirs["metrics_dir"] / f"cup-ordering-train-{int(total_timesteps)}.json"
    payload = {
        "mode": "cup_ordering",
        "backend": config["backend"],
        "task_name": config["task_name"],
        "checkpoint_path": str(checkpoint_path),
        "total_timesteps": int(total_timesteps),
        "n_steps": int(config["train"]["n_steps"]),
        "seed_set": list(config["eval"]["seed_set"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _write_training_history(config, total_timesteps, callback, mode="generic"):
    dirs = ensure_artifact_dirs(config)
    history_path = dirs["metrics_dir"] / f"{mode}-train-history-{int(total_timesteps)}.json"
    payload = {
        "mode": mode,
        "task_name": config["task_name"],
        "backend": config["backend"],
        "total_timesteps": int(total_timesteps),
        "records": list(getattr(callback, "history", []) or []),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return history_path


def read_ordering_trace_report(config, policy_name, episodes):
    dirs = ensure_artifact_dirs(config)
    report_path = _cup_ordering_report_path(config, dirs, f"{policy_name}-episodes-{int(episodes)}-trace")
    if not report_path.exists():
        raise FileNotFoundError(f"Cup-ordering trace report not found: {report_path}")
    return report_path


def get_latest_checkpoint(config):
    dirs = ensure_artifact_dirs(config)
    candidates = sorted(dirs["checkpoint_dir"].glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No trained PPO checkpoints found in {dirs['checkpoint_dir']}")
    return candidates[0]


def write_smoke_checkpoint(config, step_count=None):
    dirs = ensure_artifact_dirs(config)
    resolved_steps = config["train"]["smoke_steps"] if step_count is None else int(step_count)
    checkpoint_path = dirs["checkpoint_dir"] / f"{config['task_name']}-smoke-step-{resolved_steps}.ckpt.json"
    checkpoint_payload = {
        "task_name": config["task_name"],
        "backend": config["backend"],
        "step_count": resolved_steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_topology": config["policy"]["topology"],
        "shared_policy": bool(config["policy"]["shared_policy"]),
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")
    return checkpoint_path


def write_eval_metrics(config, checkpoint_path, seed_set=None, success_rate=0.0, episode_return=0.0):
    dirs = ensure_artifact_dirs(config)
    resolved_seed_set = list(config["eval"]["seed_set"] if seed_set is None else seed_set)
    metrics_path = dirs["metrics_dir"] / f"{config['task_name']}-eval-metrics.json"
    payload = {
        "task_name": config["task_name"],
        "backend": config["backend"],
        "checkpoint_path": str(checkpoint_path),
        "seed": resolved_seed_set[0],
        "seed_set": resolved_seed_set,
        "success_rate": float(success_rate),
        "episode_return": float(episode_return),
        "episode_length": int(config["termination"]["max_steps"]),
        "failure_modes_triggered": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics_path


def validate_and_summarize_config(config_path):
    config = validate_rl_config(load_rl_config(config_path))
    return summarize_rl_config(config)


def run_train_smoke(config_path, dry_run=False, steps=None):
    config = validate_rl_config(load_rl_config(config_path))
    if _is_cup_ordering_config(config):
        if not dry_run:
            if PPO is None:
                raise ModuleNotFoundError("stable_baselines3 must be installed for cup-ordering PPO training.")
            env = CupOrderingTrainEnv(config)
            dirs = ensure_artifact_dirs(config)
            total_timesteps = int(config["train"].get("total_timesteps", config["train"]["smoke_steps"]))
            if steps is not None:
                total_timesteps = int(steps)
            n_steps = int(config["train"].get("n_steps", 64))
            progress_print_freq = int(config["train"].get("progress_print_freq", 20))
            print(
                f"[train] mode=cup_ordering task={config['task_name']} total_timesteps={total_timesteps} n_steps={n_steps}",
                flush=True,
            )
            progress_callback = ProgressCallback(progress_print_freq)
            model = PPO("MlpPolicy", cast(Any, env), verbose=1, n_steps=n_steps)
            model.learn(total_timesteps=total_timesteps, callback=cast(Any, progress_callback))
            checkpoint_path = dirs["checkpoint_dir"] / f"cup-ordering-ppo-{total_timesteps}.zip"
            model.save(str(checkpoint_path))
            env.close()
            _write_cup_ordering_training_summary(config, checkpoint_path, total_timesteps)
            _write_training_history(config, total_timesteps, progress_callback, mode="cup_ordering")
            return checkpoint_path
        return _write_cup_ordering_train_artifact(config, step_count=steps)
    if not dry_run:
        if PPO is None:
            raise ModuleNotFoundError("stable_baselines3 must be installed for real RL training.")
        env = RLVectorEnvAdapter(config, render_enabled=False)
        dirs = ensure_artifact_dirs(config)
        total_timesteps = int(config["train"].get("total_timesteps", config["train"]["smoke_steps"]))
        if steps is not None:
            total_timesteps = int(steps)
        n_steps = int(config["train"].get("n_steps", 64))
        progress_print_freq = int(config["train"].get("progress_print_freq", 20))
        print(
            f"[train] backend={config['backend']} task={config['task_name']} total_timesteps={total_timesteps} n_steps={n_steps}",
            flush=True,
        )
        progress_callback = ProgressCallback(progress_print_freq)
        model = PPO("MlpPolicy", cast(Any, env), verbose=1, n_steps=n_steps)
        model.learn(total_timesteps=total_timesteps, callback=cast(Any, progress_callback))
        checkpoint_path = dirs["checkpoint_dir"] / f"{config['task_name']}-ppo-{total_timesteps}.zip"
        model.save(str(checkpoint_path))
        env.close()
        _write_training_history(config, total_timesteps, progress_callback, mode="generic")
        return checkpoint_path
    return write_smoke_checkpoint(config, step_count=steps)


def run_eval_smoke(config_path, checkpoint_path, dry_run=False, render_override=None, save_video=False, video_path=None, policy_name=None, episodes=None, seed=None):
    config = validate_rl_config(load_rl_config(config_path))
    if _is_cup_ordering_config(config):
        resolved_policy = str(policy_name or "random")
        resolved_episodes = int(episodes if episodes is not None else config["eval"]["episodes_per_policy"])
        resolved_seed = int(seed if seed is not None else config["eval"]["seed_set"][0])
        checkpoint_metadata = None
        if resolved_policy == "rl":
            checkpoint_metadata = inspect_cup_ordering_checkpoint_artifact(checkpoint_path)
            if checkpoint_metadata.get("policy_mode") == "checkpoint" and not checkpoint_metadata.get("supports_inference"):
                raise RuntimeError(
                    "Cup-ordering checkpoint interface is reserved but dry-run eval cannot execute real inference: "
                    f"{checkpoint_metadata.get('unsupported_reason')} ({checkpoint_metadata.get('checkpoint_path')})"
                )
        report_stem = f"{resolved_policy}-episodes-{resolved_episodes}-trace"
        metrics_path, _, _, _ = _run_cup_ordering_dry_run(
            config,
            policy_name=resolved_policy,
            episodes=resolved_episodes,
            seed_start=resolved_seed,
            save_report=True,
            report_stem=report_stem,
            render_override=render_override,
            save_video=save_video,
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
        )
        return metrics_path
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dry_run:
        if PPO is None:
            raise ModuleNotFoundError("stable_baselines3 must be installed for real RL evaluation.")
        should_render = config["render"]["enabled"] if render_override is None else bool(render_override)
        env = RLVectorEnvAdapter(config, render_enabled=should_render)
        model = PPO.load(str(checkpoint), env=cast(Any, env))
        success_count = 0
        episode_returns = []
        max_steps = int(config["termination"]["max_steps"])
        video_writer = None
        resolved_video_path = None
        video_frame_key = config.get("render", {}).get("frame_key")
        if save_video:
            if imageio is None:
                raise ModuleNotFoundError("imageio must be installed to save evaluation video.")
            if video_path is None:
                dirs = ensure_artifact_dirs(config)
                resolved_video_path = dirs["video_dir"] / f"{config['task_name']}-eval-rollout.mp4"
            else:
                resolved_video_path = resolve_repo_path(video_path)
            resolved_video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(str(resolved_video_path), fps=20)
        for seed in config["eval"]["seed_set"][: int(config["eval"]["episodes"])]:
            print(f"[eval] starting episode seed={seed}")
            obs, _ = env.reset(seed=seed)
            done = False
            episode_return = 0.0
            step_count = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                frame = env.render() if (should_render or save_video) else None
                if save_video and video_frame_key:
                    raw_obs = getattr(env._wrapped, "obs", {}) or {}
                    preferred_frame = raw_obs.get(video_frame_key)
                    if preferred_frame is not None:
                        frame = preferred_frame
                if save_video and frame is not None and video_writer is not None:
                    video_writer.append_data(frame)
                episode_return += float(reward)
                step_count += 1
                if step_count % 10 == 0:
                    print(f"[eval] seed={seed} step={step_count} return={episode_return:.4f} terminated={terminated} truncated={truncated}")
                if step_count >= max_steps:
                    truncated = True
                done = bool(terminated or truncated)
            print(f"[eval] finished seed={seed} steps={step_count} return={episode_return:.4f}")
            episode_returns.append(episode_return)
            if episode_return > 0:
                success_count += 1
        env.close()
        if video_writer is not None:
            video_writer.close()
            print(f"[eval] video saved to {resolved_video_path}")
        success_rate = success_count / max(len(episode_returns), 1)
        mean_return = sum(episode_returns) / max(len(episode_returns), 1)
        return write_eval_metrics(config, checkpoint, success_rate=success_rate, episode_return=mean_return)
    return write_eval_metrics(config, checkpoint)
