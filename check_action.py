import numpy as np
import robosuite as suite

env = suite.make(
    "TwoArmLift",
    robots=["Panda", "Panda"],
    env_configuration="parallel",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
)
print("action_dim:", env.action_dim)

obs = env.reset()
action = np.zeros(14)
obs, r, done, info = env.step(action)
print("Zero action works!")

action = np.zeros(14)
action[0] = 0.1
obs, r, done, info = env.step(action)
print("Non-zero action works!")

print("\nGripper keys:", [k for k in obs.keys() if 'gripper' in k])