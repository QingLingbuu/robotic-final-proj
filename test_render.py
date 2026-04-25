import time
from arm.env_wrapper import RobosuiteEnvWrapper

env = RobosuiteEnvWrapper(camera_config={})
print("Launching MuJoCo viewer...")

viewer = env.view()
print("Viewer launched, waiting for close...")
while viewer.is_running():
    time.sleep(0.1)

print("Viewer closed!")