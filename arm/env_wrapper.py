"""Backend-agnostic environment wrapper interfaces and robosuite adapter."""

from abc import ABC, abstractmethod

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = None

try:
    import robocasa  # noqa: F401 - import registers gym environments
except ModuleNotFoundError:
    robocasa = None

try:
    import robosuite as suite
except ModuleNotFoundError:
    suite = None


class BaseEnvWrapper(ABC):
    """Backend-agnostic manipulation environment contract."""

    backend_name = "base"

    def __init__(self):
        self.obs = None
        self.done = False
        self.action_dim = 0
        self.camera_config = {}
        self._episode_terminated = False
        self._seed = None

    @abstractmethod
    def get_observation(self):
        """Return (rgb, depth, proprioception) from the current environment state."""

    @abstractmethod
    def step(self, action):
        """Execute one action step and return (obs, reward, done, info)."""

    @abstractmethod
    def reset(self, seed=None):
        """Reset the environment and return the first observation."""

    @abstractmethod
    def render(self):
        """Render the environment."""

    @abstractmethod
    def get_camera_intrinsics(self):
        """Return camera intrinsics for the active environment."""

    @abstractmethod
    def get_camera_extrinsics(self):
        """Return camera extrinsics for the active environment."""

    @abstractmethod
    def get_object_dynamics_summary(self):
        """Return lightweight object dynamics metadata for diagnostics."""

    @abstractmethod
    def arm_safe_retract(self):
        """Move both arms to a safe neutral pose."""

    def set_seed(self, seed):
        """Persist a deterministic seed to be applied on the next reset."""
        self._seed = None if seed is None else int(seed)

    def get_seed(self):
        """Return the currently configured deterministic seed."""
        return self._seed

    def is_episode_terminated(self):
        """Return whether the current episode has terminated."""
        return bool(self._episode_terminated)


class RobosuiteEnvWrapper(BaseEnvWrapper):
    """robosuite adapter for Lift task with two Panda robots."""

    backend_name = "robosuite"

    def __init__(self, config=None, camera_config=None):
        """Initialize the TwoArmLift environment with Panda robots."""
        super().__init__()
        if suite is None:
            raise ModuleNotFoundError(
                "robosuite is required to initialize RobosuiteEnvWrapper. "
                "Install the project runtime dependencies first."
            )
        default_config = {
            "env_name": "Lift",
            "robots": ["Panda", "Panda"],
            "env_configuration": "parallel",
            "has_renderer": True,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "use_object_obs": True,
            "camera_names": ["frontview"],
            "camera_widths": [640],
            "camera_heights": [480],
            "camera_depths": True,
        }

        if config:
            default_config.update(config)

        self.camera_config = camera_config or {}
        self.env = suite.make(**default_config)
        self.obs = self._reset_env()
        self.action_dim = self.env.action_dim
        self.home_eef_positions = self._capture_eef_positions()
        self._viewer = None

    def _reset_env(self, seed=None):
        resolved_seed = self._seed if seed is None else int(seed)
        if resolved_seed is not None:
            try:
                return self.env.reset(seed=resolved_seed)
            except TypeError:
                pass
        return self.env.reset()

    def _capture_eef_positions(self):
        positions = {
            "robot0": np.array(self.obs["robot0_eef_pos"], dtype=float).copy(),
        }
        if "robot1_eef_pos" in self.obs:
            positions["robot1"] = np.array(self.obs["robot1_eef_pos"], dtype=float).copy()
        return positions

    def _capture_joint_positions(self):
        joint_positions = {
            "robot0": np.array(self.obs["robot0_joint_pos"], dtype=float).copy(),
        }
        if "robot1_joint_pos" in self.obs:
            joint_positions["robot1"] = np.array(self.obs["robot1_joint_pos"], dtype=float).copy()
        return joint_positions

    def get_observation(self):
        """Get current observation from environment."""
        rgb = self.obs.get("frontview_image")
        depth = self.obs.get("frontview_depth")

        proprioception = {
            "robot0_eef_pos": self.obs["robot0_eef_pos"],
            "robot0_eef_quat": self.obs["robot0_eef_quat"],
            "robot0_joint_pos": self.obs["robot0_joint_pos"],
            "robot0_gripper_qpos": self.obs["robot0_gripper_qpos"],
        }
        if "robot1_eef_pos" in self.obs:
            proprioception["robot1_eef_pos"] = self.obs["robot1_eef_pos"]
        if "robot1_eef_quat" in self.obs:
            proprioception["robot1_eef_quat"] = self.obs["robot1_eef_quat"]
        if "robot1_joint_pos" in self.obs:
            proprioception["robot1_joint_pos"] = self.obs["robot1_joint_pos"]
        if "robot1_gripper_qpos" in self.obs:
            proprioception["robot1_gripper_qpos"] = self.obs["robot1_gripper_qpos"]

        return rgb, depth, proprioception

    def get_flat_observation(self):
        """Return a flat privileged-state observation vector for RL baselines."""
        obs_parts = [
            np.asarray(self.obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            np.asarray(self.obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
            np.asarray(self.obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ]
        if "cube_pos" in self.obs:
            obs_parts.append(np.asarray(self.obs["cube_pos"], dtype=np.float32).reshape(-1))
        elif "object_pos" in self.obs:
            obs_parts.append(np.asarray(self.obs["object_pos"], dtype=np.float32).reshape(-1))
        elif "pot_pos" in self.obs:
            obs_parts.append(np.asarray(self.obs["pot_pos"], dtype=np.float32).reshape(-1))
        return np.concatenate(obs_parts, dtype=np.float32)

    def get_raw_observation(self):
        return self.obs

    def get_action_layout(self):
        return [("action", self.action_dim)]

    def step(self, action):
        """Execute action and return observation."""
        if self.done or self._episode_terminated:
            return self.obs, 0.0, True, {"terminated": True}

        try:
            self.obs, reward, done, info = self.env.step(action)
        except ValueError as exc:
            if "executing action in terminated episode" not in str(exc):
                raise
            self._episode_terminated = True
            self.done = True
            return self.obs, 0.0, True, {"terminated": True}

        self._episode_terminated = bool(done)
        self.done = bool(done)
        return self.obs, reward, done, info

    def get_camera_intrinsics(self):
        """Return intrinsics from config or a future runtime camera API."""
        required_keys = ("fx", "fy", "cx", "cy")
        if all(key in self.camera_config for key in required_keys):
            return tuple(float(self.camera_config[key]) for key in required_keys)

        raise ValueError(
            "Camera intrinsics must come from configs/camera.yaml "
            "or a runtime RoboCamera API."
        )

    def get_camera_extrinsics(self):
        """Return T_world_cam from config until the runtime camera API is wired in."""
        transform = self.camera_config.get("T_world_cam")
        if transform is None:
            raise ValueError(
                "Camera extrinsics must come from configs/camera.yaml "
                "or a runtime RoboCamera API."
            )

        rotation = np.array(transform["rotation"], dtype=float)
        translation = np.array(transform["translation"], dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("T_world_cam must contain a 3x3 rotation and 3D translation.")
        return rotation, translation

    def get_object_dynamics_summary(self):
        """Return a lightweight MuJoCo summary for object mobility checks."""
        model = self.env.sim.model
        summary = {
            "matching_bodies": [],
            "matching_joints": [],
        }

        body_names = getattr(model, "body_names", [])
        body_mass = getattr(model, "body_mass", [])
        for body_id, body_name in enumerate(body_names):
            lowered = body_name.lower()
            if "pot" in lowered or "object" in lowered or "cube" in lowered:
                mass = None
                if body_id < len(body_mass):
                    mass = float(body_mass[body_id])
                summary["matching_bodies"].append(
                    {
                        "name": body_name,
                        "mass": mass,
                    }
                )

        joint_names = getattr(model, "joint_names", [])
        joint_type = getattr(model, "jnt_type", [])
        joint_type_names = {
            0: "free",
            1: "ball",
            2: "slide",
            3: "hinge",
        }
        for joint_id, joint_name in enumerate(joint_names):
            lowered = joint_name.lower()
            if "pot" in lowered or "object" in lowered or "cube" in lowered:
                raw_type = None
                type_name = None
                if joint_id < len(joint_type):
                    raw_type = int(joint_type[joint_id])
                    type_name = joint_type_names.get(raw_type, str(raw_type))
                summary["matching_joints"].append(
                    {
                        "name": joint_name,
                        "type": type_name,
                    }
                )

        return summary

    def apply_action(self, arm1_pos, arm2_pos):
        """Apply a joint-space action to both arms."""
        action = np.concatenate([arm1_pos, arm2_pos])
        return self.step(action)

    def arm_safe_retract(self):
        """Return both arms to a neutral zero-action pose immediately."""
        if self.done or self._episode_terminated:
            return False

        action = np.zeros(self.action_dim)
        self.obs, _, done, _ = self.step(action)
        if done:
            return False
        for _ in range(10):
            self.obs, _, done, _ = self.step(action)
            if done:
                return False
        return True

    def reset(self, seed=None):
        """Reset the environment and return the first observation."""
        if seed is not None:
            self.set_seed(seed)
        self.obs = self._reset_env(seed=seed)
        self.done = False
        self.home_eef_positions = self._capture_eef_positions()
        self.home_joint_positions = self._capture_joint_positions()
        self.home_joint_positions = self._capture_joint_positions()
        self._episode_terminated = False
        return self.get_observation()

    def render(self):
        """Render the environment."""
        self.env.render()


class RobocasaEnvWrapper(BaseEnvWrapper):
    """Minimal RoboCasa Gymnasium adapter for single-arm RL training and visualization."""

    backend_name = "robocasa"

    def __init__(self, config=None, camera_config=None):
        super().__init__()
        if gym is None or robocasa is None:
            raise ModuleNotFoundError(
                "robocasa and gymnasium must be installed in the separate RoboCasa/RL environment."
            )

        default_config = {
            "task_name": "robocasa/PickPlaceCounterToCabinet",
            "render_mode": None,
        }
        if config:
            default_config.update(config)

        self.camera_config = camera_config or {}
        self.task_name = default_config["task_name"]
        self.render_mode = default_config.get("render_mode")
        env_kwargs = {key: value for key, value in default_config.items() if key not in {"task_name", "render_mode"}}
        if self.render_mode is not None:
            env_kwargs["render_mode"] = self.render_mode
        self.env = gym.make(self.task_name, **env_kwargs)
        self._action_keys = []
        self.action_dim = self._infer_action_dim(self.env.action_space)
        self.obs, self._last_info = self._reset_env()

    def _infer_action_dim(self, action_space):
        if hasattr(action_space, "shape") and action_space.shape is not None:
            return int(np.prod(action_space.shape))
        if hasattr(action_space, "spaces"):
            total_dim = 0
            self._action_keys = []
            for key, subspace in action_space.spaces.items():
                self._action_keys.append((key, int(np.prod(subspace.shape))))
                total_dim += int(np.prod(subspace.shape))
            return total_dim
        raise ValueError(f"Unsupported RoboCasa action space: {action_space}")

    def format_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self._action_keys:
            return action
        formatted = {}
        cursor = 0
        for key, width in self._action_keys:
            chunk = action[cursor : cursor + width].astype(np.float32)
            formatted[key] = chunk
            cursor += width
        return formatted

    def _reset_env(self, seed=None):
        resolved_seed = self._seed if seed is None else int(seed)
        if resolved_seed is not None:
            return self.env.reset(seed=resolved_seed)
        return self.env.reset()

    def _flatten_obs(self, obs):
        if isinstance(obs, dict):
            chunks = []
            for value in obs.values():
                try:
                    array_value = np.asarray(value, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    continue
                chunks.append(array_value)
            if chunks:
                return np.concatenate(chunks, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def get_observation(self):
        flat_obs = self._flatten_obs(self.obs)
        proprioception = {"flat_observation": flat_obs}
        return None, None, proprioception

    def get_flat_observation(self):
        return self._flatten_obs(self.obs)

    def get_raw_observation(self):
        base_env = getattr(self.env, "unwrapped", None)
        if base_env is not None and hasattr(base_env, "env") and hasattr(base_env.env, "_get_observations"):
            try:
                return base_env.env._get_observations(force_update=True)
            except Exception:  # noqa: BLE001 - best-effort raw observation path for reward shaping
                pass
        return self.obs

    def get_action_layout(self):
        return list(self._action_keys)

    def step(self, action):
        if self.done or self._episode_terminated:
            return self.obs, 0.0, True, {"terminated": True}

        self.obs, reward, terminated, truncated, info = self.env.step(self.format_action(action))
        self._episode_terminated = bool(terminated or truncated)
        self.done = bool(terminated or truncated)
        return self.obs, reward, self.done, info

    def get_camera_intrinsics(self):
        required_keys = ("fx", "fy", "cx", "cy")
        if all(key in self.camera_config for key in required_keys):
            return tuple(float(self.camera_config[key]) for key in required_keys)
        raise ValueError("RoboCasa camera intrinsics must come from config or runtime camera metadata.")

    def get_camera_extrinsics(self):
        transform = self.camera_config.get("T_world_cam")
        if transform is None:
            raise ValueError("RoboCasa camera extrinsics must come from config or runtime camera metadata.")
        rotation = np.array(transform["rotation"], dtype=float)
        translation = np.array(transform["translation"], dtype=float)
        return rotation, translation

    def get_object_dynamics_summary(self):
        return {"task_name": self.task_name, "observation_type": type(self.obs).__name__}

    def arm_safe_retract(self):
        return not self.done

    def reset(self, seed=None):
        if seed is not None:
            self.set_seed(seed)
        self.obs, self._last_info = self._reset_env(seed=seed)
        self.done = False
        self._episode_terminated = False
        return self.get_observation()

    def render(self):
        return self.env.render()


def _create_robocasa_wrapper(config=None, camera_config=None):
    return RobocasaEnvWrapper(config=config, camera_config=camera_config)


def create_env_wrapper(backend="robosuite", config=None, camera_config=None):
    """Create a backend-specific environment wrapper through a stable contract."""
    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "robosuite":
        return RobosuiteEnvWrapper(config=config, camera_config=camera_config)
    if normalized_backend == "robocasa":
        return _create_robocasa_wrapper(config=config, camera_config=camera_config)
    raise ValueError(f"Unsupported environment backend: {backend}")
