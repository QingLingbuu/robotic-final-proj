"""Robosuite environment wrapper for two-arm manipulation task."""

import numpy as np
import robosuite as suite


class RobosuiteEnvWrapper:
    """Environment wrapper for TwoArmLift task with two Panda robots."""
    
    def __init__(self, config=None):
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
            "camera_depths": True,  # 关键：开启深度图
        }
        
        if config:
            default_config.update(config)
        
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
        fx, fy = 525.0, 525.0
        cx, cy = 319.5, 239.5
        return fx, fy, cx, cy
    
    def apply_action(self, arm1_pos, arm2_pos):
        action = np.concatenate([arm1_pos, arm2_pos])
        self.obs, reward, done, info = self.env.step(action)
        return self.obs, reward, done, info
    
    def arm_safe_retract(self):
        action = np.zeros(self.env.action_dim)
        self.obs, _, _, _ = self.env.step(action)
        for _ in range(10):
            self.obs, _, _, _ = self.env.step(action)
    
    def reset(self):
        self.obs = self.env.reset()
        return self.get_observation()
    
    def render(self):
        self.env.render()

