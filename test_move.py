import numpy as np
import time
from arm.env_wrapper import RobosuiteEnvWrapper

env = RobosuiteEnvWrapper(camera_config={})
print(f"Action dim: {env.action_dim}")
print(f"Initial robot0_eef_pos: {env.obs['robot0_eef_pos']}")

print("\n--- Test 1: Move robot0 with small delta ---")
for i in range(50):
    action = np.zeros(env.action_dim)
    action[0] = -0.02
    action[6] = 0.0
    env.obs, reward, done, info = env.step(action)
    if i % 10 == 0:
        print(f"  Step {i}: eef_pos = {env.obs['robot0_eef_pos'][:2]}")
    if done:
        print("  Done!")
        break

print(f"\nFinal robot0_eef_pos: {env.obs['robot0_eef_pos']}")
print("\nKeeping render window open...")
print("Close window or press Ctrl+C to exit.")
try:
    while True:
        env.render()
        time.sleep(0.03)
except KeyboardInterrupt:
    print("\nExiting...")
except Exception as e:
    print(f"Exited: {e}")