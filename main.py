"""Main entry point for robotic manipulation final project."""

import time
from datetime import datetime

import yaml

from arm.env_wrapper import RobosuiteEnvWrapper
from fsm.demo_cycles import build_run_context, run_demo_cycles
from fsm.state_machine import State, TaskStateMachine
from ipc.perception_queue import PERCEPTION_QUEUE_NAME, create_perception_queue
from logs.run_logger import build_run_log, write_run_log


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def main():
    """Run a minimal integration path across config, env, IPC, and FSM."""
    start_time = time.monotonic()
    print("Starting robotic manipulation project...")
    print("=" * 60)

    camera_config = load_config("configs/camera.yaml")
    thresholds_config = load_config("configs/thresholds.yaml")
    conf_thresh = thresholds_config["CONF_THRESH"]

    print("Loaded configs:")
    print(f"  Camera: {camera_config}")
    print(f"  Thresholds: {thresholds_config}")

    print("\nInitializing robosuite environment...")
    env = RobosuiteEnvWrapper(camera_config=camera_config)

    print("\nGetting initial observation...")
    rgb, depth, proprio = env.reset()
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Arm 1 EEF pos: {proprio['arm1_eef_pos']}")
    print(f"  Arm 2 EEF pos: {proprio['arm2_eef_pos']}")

    fx, fy, cx, cy = env.get_camera_intrinsics()
    print(f"\nCamera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    perception_queue = create_perception_queue()
    print(f"\nInitialized {PERCEPTION_QUEUE_NAME}.")

    fsm = TaskStateMachine()
    fsm.transition_to(State.PLANNING)
    print(f"\nFSM initialized in state: {fsm.get_current_state().value}")

    task_completed = run_demo_cycles(
        fsm=fsm,
        perception_queue=perception_queue,
        conf_thresh=conf_thresh,
    )

    print("\nTesting safe arm retract...")
    env.arm_safe_retract()
    print("  Arm retract executed")

    if task_completed:
        fsm.transition_to(State.SUCCESS)

    counts = fsm.get_failure_counts()
    duration_sec = time.monotonic() - start_time
    run_log = build_run_log(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        config_version="camera-thresholds-v1",
        scene_config={
            "seed": 42,
            "object_count": 1,
            "scene_id": "demo_frontview_loop",
        },
        success=task_completed,
        counts=counts,
        duration_sec=duration_sec,
        context=build_run_context(fsm, task_completed),
    )
    log_path = write_run_log(run_log)

    print("\nFailure counters:", counts)
    print(f"Run log written to: {log_path}")
    print("\n" + "=" * 60)
    print(f"Task completed! Final state: {fsm.get_current_state().value}")
    print("\nMinimal integration path completed.")


if __name__ == "__main__":
    main()
