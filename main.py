"""Main entry point for robotic manipulation final project."""

import time
from datetime import datetime

import numpy as np
import yaml

from arm.controller import (
    execute_dual_handle_lift,
    execute_dual_handle_transfer,
    execute_grasp,
    execute_push,
    execute_single_grasp_transfer,
    check_object_at_place,
    get_default_grasp_target,
    get_gripper_width,
    get_handle_targets,
    get_primary_object_pos,
    home_arms,
)
from arm.env_wrapper import RobosuiteEnvWrapper
from fsm.demo_cycles import build_run_context, run_demo_cycles
from fsm.state_machine import State, TaskStateMachine
from ipc.perception_queue import PERCEPTION_QUEUE_NAME, create_perception_queue
from logs.run_logger import build_run_log, write_run_log

KEEP_RENDER_OPEN = False  # Set to True to keep render window open after test
GRASP_MODE = "dual"  # "dual"双臂夹取 or "single"单臂夹取
ENABLE_PLACE_TEST = True
SINGLE_GRASP_Z_OFFSET = -0.04
PLACE_TARGET_OFFSET = [0.2, 0.2, 0.0]
PLACE_RETRY_ATTEMPTS = 2
PUSH_DIRECTION = [0.0, 1.0]


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
    print(f"  Action dim: {env.action_dim}")
    print(f"  Object dynamics: {env.get_object_dynamics_summary()}")

    print("\nGetting initial observation...")
    rgb, depth, proprio = env.reset()
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Arm 1 EEF pos: {proprio['robot0_eef_pos']}")
    print(f"  Arm 2 EEF pos: {proprio['robot1_eef_pos']}")

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

    print("\n--- TEST 1: real grasp control ---")
    if GRASP_MODE not in ("dual", "single"):
        raise ValueError('GRASP_MODE must be "dual" or "single".')

    print(f"Before grasp: robot0_eef_pos = {env.obs['robot0_eef_pos']}")
    primary_object_pos = get_primary_object_pos(env.obs)
    place_target_pos = None
    left_handle_pos, right_handle_pos = get_handle_targets(env.obs)
    grasp_target_pos = get_default_grasp_target(env.obs, arm_idx=0)
    if grasp_target_pos is None:
        raise RuntimeError("No object or handle position found in robosuite observations.")
    print(f"Primary object pos: {primary_object_pos}")
    print(f"Grasp mode: {GRASP_MODE}")
    if GRASP_MODE == "dual" and left_handle_pos is not None and right_handle_pos is not None:
        print(f"Left handle target: {left_handle_pos}")
        print(f"Right handle target: {right_handle_pos}")
        print("Dual-arm grasp target z offset: -0.04 m")
        print(
            "Distance robot0 EEF -> left handle: "
            f"{np.linalg.norm(env.obs['robot0_eef_pos'] - left_handle_pos):.4f} m"
        )
        print(
            "Distance robot1 EEF -> right handle: "
            f"{np.linalg.norm(env.obs['robot1_eef_pos'] - right_handle_pos):.4f} m"
        )
        print("Dual-arm lifting from both handles...")
        if ENABLE_PLACE_TEST:
            place_target_pos = primary_object_pos + np.array(PLACE_TARGET_OFFSET)
            print(f"Place target pos: {place_target_pos}")
            grasp_success = execute_dual_handle_transfer(
                env,
                left_handle_pos,
                right_handle_pos,
                place_target_pos,
            )
        else:
            grasp_success = execute_dual_handle_lift(
                env,
                left_handle_pos,
                right_handle_pos,
            )
    else:
        single_grasp_pos = grasp_target_pos + np.array([0.0, 0.0, SINGLE_GRASP_Z_OFFSET])
        print(f"Raw single-arm grasp target: {grasp_target_pos}")
        print(f"Single-arm grasp target z offset: {SINGLE_GRASP_Z_OFFSET} m")
        print(
            "Distance robot0 EEF -> grasp target: "
            f"{np.linalg.norm(env.obs['robot0_eef_pos'] - single_grasp_pos):.4f} m"
        )
        print(f"Grasping at {single_grasp_pos}...")
        if ENABLE_PLACE_TEST:
            place_target_pos = primary_object_pos + np.array(PLACE_TARGET_OFFSET)
            print(f"Place target pos: {place_target_pos}")
            grasp_success = execute_single_grasp_transfer(
                env,
                single_grasp_pos,
                place_target_pos,
                arm_idx=0,
            )
        else:
            grasp_success = execute_grasp(env, single_grasp_pos, arm_idx=0)
    print(f"  Grasp result: {'SUCCESS' if grasp_success else 'FAILED'}")
    print(f"Object pos after grasp: {get_primary_object_pos(env.obs)}")
    print(f"After grasp: robot0_eef_pos = {env.obs['robot0_eef_pos']}")
    print(f"After grasp: robot1_eef_pos = {env.obs['robot1_eef_pos']}")
    print(f"After grasp: robot0_gripper_width = {get_gripper_width(env, 0):.4f}")
    print(f"After grasp: robot1_gripper_width = {get_gripper_width(env, 1):.4f}")
    if place_target_pos is not None:
        place_check = False
        for retry_idx in range(PLACE_RETRY_ATTEMPTS + 1):
            place_check = check_object_at_place(env, place_target_pos)
            print(
                "Object near place target before home: "
                f"{'YES' if place_check else 'NO'}"
            )
            if place_check:
                grasp_success = True
                break
            if retry_idx >= PLACE_RETRY_ATTEMPTS:
                break

            print(
                "Place target missed; retrying pick-place "
                f"({retry_idx + 1}/{PLACE_RETRY_ATTEMPTS})..."
            )
            if GRASP_MODE == "dual":
                left_handle_pos, right_handle_pos = get_handle_targets(env.obs)
                if left_handle_pos is None or right_handle_pos is None:
                    print("  Retry skipped: handle targets unavailable.")
                    break
                grasp_success = execute_dual_handle_transfer(
                    env,
                    left_handle_pos,
                    right_handle_pos,
                    place_target_pos,
                )
            else:
                grasp_target_pos = get_default_grasp_target(env.obs, arm_idx=0)
                if grasp_target_pos is None:
                    print("  Retry skipped: single-arm grasp target unavailable.")
                    break
                single_grasp_pos = grasp_target_pos + np.array(
                    [0.0, 0.0, SINGLE_GRASP_Z_OFFSET]
                )
                grasp_success = execute_single_grasp_transfer(
                    env,
                    single_grasp_pos,
                    place_target_pos,
                    arm_idx=0,
                )

    print("Returning arms home after grasp task...")
    home_success = home_arms(env)
    print(f"  Home result: {'SUCCESS' if home_success else 'FAILED'}")

    print("\n--- TEST 2: real push control ---")
    env.reset()
    push_target_pos = get_primary_object_pos(env.obs)
    if push_target_pos is None:
        raise RuntimeError("No object position found in robosuite observations.")
    print(f"Object pos before push: {push_target_pos}")
    print(
        "Distance robot1 EEF -> push target: "
        f"{np.linalg.norm(env.obs['robot1_eef_pos'] - push_target_pos):.4f} m"
    )
    print(f"Push direction: {PUSH_DIRECTION}")
    print(f"Pushing obstacle at {push_target_pos}...")
    push_success = execute_push(env, push_target_pos, PUSH_DIRECTION, arm_idx=1)
    print(f"  Push result: {'SUCCESS' if push_success else 'FAILED'}")
    print(f"Object pos after push: {get_primary_object_pos(env.obs)}")
    print(
        "Distance robot1 EEF -> push target after attempt: "
        f"{np.linalg.norm(env.obs['robot1_eef_pos'] - push_target_pos):.4f} m"
    )
    print(
        "Object XY displacement after push: "
        f"{np.linalg.norm(get_primary_object_pos(env.obs)[:2] - push_target_pos[:2]):.4f} m"
    )
    print(f"After push: robot1_gripper_width = {get_gripper_width(env, 1):.4f}")
    print("Returning arms home after push task...")
    home_success = home_arms(env)
    print(f"  Home result: {'SUCCESS' if home_success else 'FAILED'}")

    print("\nTesting safe arm retract...")
    env.arm_safe_retract()
    print("  Arm retract executed")

    if KEEP_RENDER_OPEN:
        print("\nKeeping render window open...")
        print("Close render window or press Ctrl+C to exit.")
        try:
            while True:
                env.render()
                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\nExiting via Ctrl+C...")
        except Exception:
            print("\nRender window closed, exiting...")
    else:
        env.render()
        print("\nRender complete, closing...")

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
