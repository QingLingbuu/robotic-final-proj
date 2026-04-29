"""Minimal dry-run and real harness for the milestone-1 RL baseline contract."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arm.env_wrapper import create_env_wrapper
from rl.contracts import load_rl_config, summarize_rl_config, validate_rl_config

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:
    PPO = None
    BaseCallback = object

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None

try:
    import imageio
except ModuleNotFoundError:
    imageio = None


class RLVectorEnvAdapter(gym.Env if gym is not None else object):
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


class ProgressCallback(BaseCallback):
    """Simple heartbeat callback so long RoboCasa runs show visible progress."""

    def __init__(self, print_freq):
        super().__init__()
        self.print_freq = max(int(print_freq), 1)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[train] timesteps={self.num_timesteps}", flush=True)
        return True


def ensure_artifact_dirs(config):
    root_dir = Path(config["artifacts"]["root_dir"])
    checkpoint_dir = root_dir / config["artifacts"]["checkpoint_dir"]
    metrics_dir = root_dir / config["artifacts"]["metrics_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return {"root_dir": root_dir, "checkpoint_dir": checkpoint_dir, "metrics_dir": metrics_dir}


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
        model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps)
        model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(progress_print_freq))
        checkpoint_path = dirs["checkpoint_dir"] / f"{config['task_name']}-ppo-{total_timesteps}.zip"
        model.save(str(checkpoint_path))
        env.close()
        return checkpoint_path
    return write_smoke_checkpoint(config, step_count=steps)


def run_eval_smoke(config_path, checkpoint_path, dry_run=False, render_override=None, save_video=False, video_path=None):
    config = validate_rl_config(load_rl_config(config_path))
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not dry_run:
        if PPO is None:
            raise ModuleNotFoundError("stable_baselines3 must be installed for real RL evaluation.")
        should_render = config["render"]["enabled"] if render_override is None else bool(render_override)
        env = RLVectorEnvAdapter(config, render_enabled=should_render)
        model = PPO.load(str(checkpoint), env=env)
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
                resolved_video_path = dirs["metrics_dir"] / f"{config['task_name']}-eval-rollout.mp4"
            else:
                resolved_video_path = Path(video_path)
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
                if save_video and frame is not None:
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
