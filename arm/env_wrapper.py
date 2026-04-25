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
            "has_renderer": False,
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

    def get_observation(self):
        """Get current observation from environment."""
        rgb = self.obs["frontview_image"]
        depth = self.obs["frontview_depth"]

        proprioception = {
            "arm1_eef_pos": self.obs["robot0_eef_pos"],
            "arm1_eef_quat": self.obs["robot0_eef_quat"],
            "arm1_joints": self.obs["robot0_joint_pos"],
            "arm2_eef_pos": self.obs["robot1_eef_pos"],
            "arm2_eef_quat": self.obs["robot1_eef_quat"],
            "arm2_joints": self.obs["robot1_joint_pos"],
        }

        return rgb, depth, proprioception

    def get_camera_intrinsics(self):
        """Return intrinsics from config or a future runtime camera API."""
        required_keys = ("fx", "fy", "cx", "cy")
        if all(key in self.camera_config for key in required_keys):
            return tuple(float(self.camera_config[key]) for key in required_keys)

        raise ValueError(
            "Camera intrinsics must come from configs/camera.yaml "
            "or a runtime RoboCamera API."
        )

    def apply_action(self, arm1_pos, arm2_pos):
        """Apply a joint-space action to both arms."""
        action = np.concatenate([arm1_pos, arm2_pos])
        self.obs, reward, done, info = self.env.step(action)
        return self.obs, reward, done, info

    def arm_safe_retract(self):
        """Return both arms to a neutral zero-action pose immediately."""
        action = np.zeros(self.env.action_dim)
        self.obs, _, _, _ = self.env.step(action)
        for _ in range(10):
            self.obs, _, _, _ = self.env.step(action)

    def reset(self):
        """Reset the environment and return the first observation."""
        self.obs = self.env.reset()
        return self.get_observation()

    def render(self):
        """Render the environment."""
        self.env.render()
