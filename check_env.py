import numpy as np
import yaml
from arm.env_wrapper import RobosuiteEnvWrapper

camera_config = yaml.safe_load(open("configs/camera.yaml"))
env = RobosuiteEnvWrapper(camera_config=camera_config)

print("=== Initial Robot Positions ===")
print(f"robot0: {env.obs['robot0_eef_pos']}")
print(f"robot1: {env.obs['robot1_eef_pos']}")

print("\n=== Objects ===")
obj_state = env.obs.get('object-state', [])
num_objects = len(obj_state) // 7
for i in range(num_objects):
    pos = obj_state[i*7:(i*7+3)]
    quat = obj_state[(i*7+3):(i*7+7)]
    print(f"Object {i}: pos={pos}, quat={quat}")

print("\n=== Gripper qpos ===")
print(f"robot0_gripper_qpos: {env.obs['robot0_gripper_qpos']}")
print(f"robot1_gripper_qpos: {env.obs['robot1_gripper_qpos']}")

print("\n=== Testing gripper open ===")
action = np.zeros(14)
action[6] = -1.0
for _ in range(20):
    env.obs, _, _, _ = env.step(action)
print(f"After gripper close: robot0_gripper_qpos = {env.obs['robot0_gripper_qpos']}")

print("\n=== Testing gripper open ===")
action = np.zeros(14)
action[6] = 1.0
for _ in range(20):
    env.obs, _, _, _ = env.step(action)
print(f"After gripper open: robot0_gripper_qpos = {env.obs['robot0_gripper_qpos']}")