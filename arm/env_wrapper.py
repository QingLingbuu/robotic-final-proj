"""Robosuite environment wrapper for two-arm manipulation task."""

import numpy as np
import robosuite as suite


class RobosuiteEnvWrapper:
    """Environment wrapper for TwoArmLift task with two Panda robots."""

    def __init__(self, config=None, camera_config=None):
        """Initialize the TwoArmLift environment with Panda robots."""
        default_config = {
            "env_name": "TwoArmLift",
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
        self.obs = self.env.reset()
        self.done = False
        self.action_dim = self.env.action_dim
        self.home_eef_positions = self._capture_eef_positions()
        self._viewer = None
        self._episode_terminated = False

    def _capture_eef_positions(self):
        return {
            "robot0": np.array(self.obs["robot0_eef_pos"], dtype=float).copy(),
            "robot1": np.array(self.obs["robot1_eef_pos"], dtype=float).copy(),
        }

    def _capture_joint_positions(self):
        return {
            "robot0": np.array(self.obs["robot0_joint_pos"], dtype=float).copy(),
            "robot1": np.array(self.obs["robot1_joint_pos"], dtype=float).copy(),
        }

    def get_observation(self):
        """Get current observation from environment."""
        rgb = self.obs["frontview_image"]
        depth = self.obs["frontview_depth"]

        proprioception = {
            "robot0_eef_pos": self.obs["robot0_eef_pos"],
            "robot0_eef_quat": self.obs["robot0_eef_quat"],
            "robot0_joint_pos": self.obs["robot0_joint_pos"],
            "robot1_eef_pos": self.obs["robot1_eef_pos"],
            "robot1_eef_quat": self.obs["robot1_eef_quat"],
            "robot1_joint_pos": self.obs["robot1_joint_pos"],
            "robot0_gripper_qpos": self.obs["robot0_gripper_qpos"],
            "robot1_gripper_qpos": self.obs["robot1_gripper_qpos"],
        }

        return rgb, depth, proprioception

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

    def is_episode_terminated(self):
        """Return whether the underlying robosuite episode has terminated."""
        return bool(self._episode_terminated)

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

    def reset(self):
        """Reset the environment and return the first observation."""
        self.obs = self.env.reset()
        self.done = False
        self.home_eef_positions = self._capture_eef_positions()
        self.home_joint_positions = self._capture_joint_positions()
        self.home_joint_positions = self._capture_joint_positions()
        self._episode_terminated = False
        return self.get_observation()

    def render(self):
        """Render the environment."""
        self.env.render()
