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
    compute_visual_dual_grasp_targets,
    get_default_grasp_target,
    get_gripper_width,
    get_handle_targets,
    get_primary_object_pos,
    home_arms,
)
from arm.env_wrapper import RobosuiteEnvWrapper
from fsm.demo_cycles import build_run_context, run_demo_cycles
from fsm.perception_cycle import run_live_perception_cycle
from fsm.state_machine import State, TaskStateMachine
from ipc.perception_queue import PERCEPTION_QUEUE_NAME, create_perception_queue
from logs.run_logger import build_run_log, write_run_log
from vision.perception_loop import VisionPerceptionLoop, VisionPerceptionWorker

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


def vision_payload_ready(detected_objects):
    """Return whether the payload contains a planning-usable target."""
    return bool(detected_objects) and detected_objects.get("status") == "ready"


def get_target_pos_from_detection(detected_objects):
    """Return the vision target position when available."""
    if not vision_payload_ready(detected_objects):
        return None
    return np.array(detected_objects["target"]["pos"], dtype=float)


def get_obstacle_pos_from_detection(detected_objects):
    """Return the first vision obstacle position when available."""
    if not vision_payload_ready(detected_objects):
        return None

    obstacles = detected_objects.get("obstacles", [])
    if not obstacles:
        return None
    return np.array(obstacles[0]["pos"], dtype=float)


def get_obstacle_count_from_detection(detected_objects):
    """Return the number of planning obstacles in the vision payload."""
    if not vision_payload_ready(detected_objects):
        return 0
    return len(detected_objects.get("obstacles", []))


def build_execution_summary():
    """Create a mutable summary for logging which data sources drove actions."""
    return {
        "planning_source": "demo",
        "clearing_source": None,
        "grasp_source": None,
        "push_source": None,
        "grasp_mode": GRASP_MODE,
        "vision_target_pos": None,
        "grasp_targets": None,
        "object_pos_before_grasp": None,
        "object_pos_after_grasp": None,
    }


def print_detection_summary(prefix, detected_objects):
    """Print a compact summary of the latest perception payload."""
    if detected_objects is None:
        print(f"{prefix}: no payload")
        return
    print(f"{prefix}: status={detected_objects['status']}")
    print(f"{prefix}: target={detected_objects['target']}")
    print(f"{prefix}: obstacles={len(detected_objects['obstacles'])}")


def select_dual_grasp_targets(env, latest_detection, vision_config):
    """Select dual-arm grasp targets from vision first, then observation handles."""
    vision_target_pos = get_target_pos_from_detection(latest_detection)
    if vision_target_pos is not None:
        corrected_target_pos = np.array(vision_target_pos, dtype=float)
        primary_object_pos = get_primary_object_pos(env.obs)
        if primary_object_pos is not None:
            alpha = float(vision_config.get("sim_xy_correction_alpha", 1.0))
            alpha = min(max(alpha, 0.0), 1.0)
            corrected_target_pos[:2] = (
                alpha * corrected_target_pos[:2]
                + (1.0 - alpha) * np.array(primary_object_pos[:2], dtype=float)
            )
        if vision_config.get("use_sim_height_correction", False):
            if primary_object_pos is not None:
                corrected_target_pos[2] = float(primary_object_pos[2])
        left_target, right_target = compute_visual_dual_grasp_targets(
            target_pos=corrected_target_pos,
            robot0_eef_pos=env.obs["robot0_eef_pos"],
            robot1_eef_pos=env.obs["robot1_eef_pos"],
            lateral_offset=vision_config.get("dual_grasp_lateral_offset", 0.09),
            vertical_offset=vision_config.get("dual_grasp_height_offset", -0.04),
            axis_mode=vision_config.get("dual_grasp_axis", "robots"),
        )
        return left_target, right_target, "vision"

    left_target, right_target = get_handle_targets(env.obs)
    if left_target is None or right_target is None:
        return None, None, None
    return left_target, right_target, "handles"


def initialize_vision_system(env, camera_config, vision_config, conf_thresh, perception_queue):
    """Build and warm up the live vision loop."""
    print("\nInitializing vision perception loop...")
    perception_loop = None
    perception_worker = None
    latest_detection = None
    try:
        perception_loop = VisionPerceptionLoop(
            vision_config=vision_config,
            camera_config=camera_config,
            conf_thresh=conf_thresh,
        )
        perception_worker = VisionPerceptionWorker(
            env=env,
            perception_loop=perception_loop,
            perception_queue=perception_queue,
            publish_period_sec=vision_config.get("publish_period_sec", 0.2),
        )
        perception_worker.start()
        latest_detection = perception_worker.wait_for_initial_detection(
            timeout_sec=vision_config.get("initial_publish_timeout_sec", 30.0)
        )
        if latest_detection is None:
            latest_error = perception_worker.get_latest_error()
            if latest_error is not None:
                raise latest_error
            raise RuntimeError(
                "Vision worker did not publish an initial payload before timeout. "
                "Increase `initial_publish_timeout_sec` or inspect worker inference speed."
            )
        print_detection_summary("  Vision payload", latest_detection)
        return perception_loop, perception_worker, latest_detection
    except Exception as exc:
        if perception_worker is not None:
            perception_worker.stop()
        print(f"  Vision loop unavailable: {exc}")
        print("  Continuing with demo FSM cycles only.")
        return None, None, None


def run_planning_phase(
    env,
    fsm,
    perception_queue,
    conf_thresh,
    perception_loop,
    perception_worker,
):
    """Drive planning from live perception when available, else run demo cycles."""
    latest_detection = None
    if perception_loop is not None:
        if perception_worker is not None:
            perception_worker.stop()
        latest_detection, live_cycle_ok = run_live_perception_cycle(
            fsm=fsm,
            perception_queue=perception_queue,
            env=env,
            perception_loop=perception_loop,
        )
        print_detection_summary("Live perception payload", latest_detection)
        return latest_detection, live_cycle_ok and fsm.get_current_state() == State.GRASPING

    task_completed = run_demo_cycles(
        fsm=fsm,
        perception_queue=perception_queue,
        conf_thresh=conf_thresh,
    )
    return latest_detection, task_completed


def run_clearing_phase(
    env, fsm, perception_queue, perception_loop, latest_detection, execution_summary
):
    """Clear the highest-priority obstacle and refresh planning."""
    if perception_loop is None or fsm.get_current_state() != State.CLEARING:
        return latest_detection, fsm.get_current_state() == State.GRASPING

    print("\n--- PLANNING: clearing obstacle selected by vision ---")
    execution_summary["planning_source"] = "vision"
    obstacle_count = get_obstacle_count_from_detection(latest_detection)
    print(f"Detected obstacles before clearing: {obstacle_count}")
    push_target_pos = get_obstacle_pos_from_detection(latest_detection)
    if push_target_pos is None:
        print("No obstacle position available from vision; routing to RETRY_PUSH.")
        fsm.handle_push_blocked()
        return latest_detection, False

    print(f"Clearing obstacle from vision at {push_target_pos}")
    execution_summary["clearing_source"] = "vision"
    push_success = execute_push(env, push_target_pos, PUSH_DIRECTION, arm_idx=1)
    print(f"  Planned clearing result: {'SUCCESS' if push_success else 'FAILED'}")
    if not push_success:
        fsm.handle_push_blocked()
        return latest_detection, False

    fsm.transition_to(State.PLANNING)
    latest_detection, task_completed = run_live_perception_cycle(
        fsm=fsm,
        perception_queue=perception_queue,
        env=env,
        perception_loop=perception_loop,
    )
    print_detection_summary("Post-clearing vision payload", latest_detection)
    return latest_detection, task_completed and fsm.get_current_state() == State.GRASPING


def run_grasp_phase(env, fsm, latest_detection, vision_config, execution_summary):
    """Execute grasping using the current planning state and latest detection."""
    print("\n--- TEST 1: real grasp control ---")
    if GRASP_MODE not in ("dual", "single"):
        raise ValueError('GRASP_MODE must be "dual" or "single".')

    if fsm.get_current_state() not in (State.GRASPING, State.PLANNING):
        print(f"Skipping grasp because FSM is in {fsm.get_current_state().value}.")
        return False, None

    print(f"Before grasp: robot0_eef_pos = {env.obs['robot0_eef_pos']}")
    primary_object_pos = get_primary_object_pos(env.obs)
    if primary_object_pos is not None:
        execution_summary["object_pos_before_grasp"] = [
            float(value) for value in np.array(primary_object_pos, dtype=float).tolist()
        ]
    vision_target_pos = get_target_pos_from_detection(latest_detection)
    place_target_pos = None
    left_handle_pos, right_handle_pos = get_handle_targets(env.obs)
    grasp_target_pos = get_default_grasp_target(env.obs, arm_idx=0)
    if grasp_target_pos is None:
        raise RuntimeError("No object or handle position found in robosuite observations.")
    if vision_target_pos is not None:
        execution_summary["vision_target_pos"] = [
            float(value) for value in np.array(vision_target_pos, dtype=float).tolist()
        ]
        print(f"Vision target world pos: {vision_target_pos}")
    print(f"Primary object pos: {primary_object_pos}")
    print(f"Grasp mode: {GRASP_MODE}")

    if GRASP_MODE == "dual":
        left_handle_pos, right_handle_pos, dual_source = select_dual_grasp_targets(
            env=env,
            latest_detection=latest_detection,
            vision_config=vision_config,
        )
        if dual_source is None:
            raise RuntimeError("Dual-arm grasp requested but no handle or vision targets are available.")

        execution_summary["grasp_source"] = dual_source
        execution_summary["grasp_targets"] = {
            "left": [float(value) for value in np.array(left_handle_pos, dtype=float).tolist()],
            "right": [float(value) for value in np.array(right_handle_pos, dtype=float).tolist()],
        }
        print(f"Dual-arm grasp source: {dual_source}")
        print(f"Left grasp target: {left_handle_pos}")
        print(f"Right grasp target: {right_handle_pos}")
        print(
            "Distance robot0 EEF -> left target: "
            f"{np.linalg.norm(env.obs['robot0_eef_pos'] - left_handle_pos):.4f} m"
        )
        print(
            "Distance robot1 EEF -> right target: "
            f"{np.linalg.norm(env.obs['robot1_eef_pos'] - right_handle_pos):.4f} m"
        )
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
        single_source = "observation"
        if vision_target_pos is not None:
            grasp_target_pos = vision_target_pos
            single_source = "vision"
        execution_summary["grasp_source"] = single_source
        single_grasp_pos = grasp_target_pos + np.array([0.0, 0.0, SINGLE_GRASP_Z_OFFSET])
        execution_summary["grasp_targets"] = {
            "single": [float(value) for value in np.array(single_grasp_pos, dtype=float).tolist()]
        }
        print(f"Raw single-arm grasp target ({single_source}): {grasp_target_pos}")
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

    return grasp_success, place_target_pos


def main():
    """Run a minimal integration path across config, env, IPC, and FSM."""
    start_time = time.monotonic()
    print("Starting robotic manipulation project...")
    print("=" * 60)

    camera_config = load_config("configs/camera.yaml")
    thresholds_config = load_config("configs/thresholds.yaml")
    vision_config = load_config("configs/vision.yaml")
    conf_thresh = thresholds_config["CONF_THRESH"]

    print("Loaded configs:")
    print(f"  Camera: {camera_config}")
    print(f"  Thresholds: {thresholds_config}")
    print(f"  Vision: {vision_config}")

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

    perception_loop, perception_worker, latest_detection = initialize_vision_system(
        env=env,
        camera_config=camera_config,
        vision_config=vision_config,
        conf_thresh=conf_thresh,
        perception_queue=perception_queue,
    )

    fsm = TaskStateMachine()
    execution_summary = build_execution_summary()
    fsm.transition_to(State.PLANNING)
    print(f"\nFSM initialized in state: {fsm.get_current_state().value}")

    latest_detection, task_completed = run_planning_phase(
        env=env,
        fsm=fsm,
        perception_queue=perception_queue,
        conf_thresh=conf_thresh,
        perception_loop=perception_loop,
        perception_worker=perception_worker,
    )
    if perception_loop is not None:
        execution_summary["planning_source"] = "vision"
    latest_detection, task_completed = run_clearing_phase(
        env=env,
        fsm=fsm,
        perception_queue=perception_queue,
        perception_loop=perception_loop,
        latest_detection=latest_detection,
        execution_summary=execution_summary,
    )
    planning_completed = task_completed
    grasp_success, place_target_pos = run_grasp_phase(
        env=env,
        fsm=fsm,
        latest_detection=latest_detection,
        vision_config=vision_config,
        execution_summary=execution_summary,
    )

    print(f"  Grasp result: {'SUCCESS' if grasp_success else 'FAILED'}")
    object_pos_after_grasp = get_primary_object_pos(env.obs)
    if object_pos_after_grasp is not None:
        execution_summary["object_pos_after_grasp"] = [
            float(value) for value in np.array(object_pos_after_grasp, dtype=float).tolist()
        ]
    print(f"Object pos after grasp: {object_pos_after_grasp}")
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
                left_handle_pos, right_handle_pos, dual_source = select_dual_grasp_targets(
                    env=env,
                    latest_detection=latest_detection,
                    vision_config=vision_config,
                )
                if dual_source is None:
                    print("  Retry skipped: dual-arm grasp targets unavailable.")
                    break
                execution_summary["grasp_source"] = dual_source
                print(f"  Retry dual-arm grasp source: {dual_source}")
                grasp_success = execute_dual_handle_transfer(
                    env,
                    left_handle_pos,
                    right_handle_pos,
                    place_target_pos,
                )
            else:
                grasp_target_pos = get_default_grasp_target(env.obs, arm_idx=0)
                single_source = "observation"
                vision_target_pos = get_target_pos_from_detection(latest_detection)
                if vision_target_pos is not None:
                    grasp_target_pos = vision_target_pos
                    single_source = "vision"
                if grasp_target_pos is None:
                    print("  Retry skipped: single-arm grasp target unavailable.")
                    break
                execution_summary["grasp_source"] = single_source
                single_grasp_pos = grasp_target_pos + np.array(
                    [0.0, 0.0, SINGLE_GRASP_Z_OFFSET]
                )
                grasp_success = execute_single_grasp_transfer(
                    env,
                    single_grasp_pos,
                    place_target_pos,
                    arm_idx=0,
                )

    manipulation_success = grasp_success
    if place_target_pos is not None:
        manipulation_success = bool(grasp_success and place_check)

    if manipulation_success:
        if not fsm.is_terminal():
            fsm.transition_to(State.VERIFYING)
            fsm.transition_to(State.SUCCESS)
    elif not fsm.is_terminal():
        fsm.handle_empty_grasp()

    print("Returning arms home after grasp task...")
    home_success = home_arms(env)
    print(f"  Home result: {'SUCCESS' if home_success else 'FAILED'}")

    print("\n--- TEST 2: real push control ---")
    env.reset()
    refreshed_detection = latest_detection
    push_target_pos = get_obstacle_pos_from_detection(refreshed_detection)
    push_source = "vision obstacle"
    if push_target_pos is None and perception_loop is not None:
        try:
            rgb, depth, _ = env.get_observation()
            refreshed_detection = perception_loop.publish_from_observation(
                perception_queue=perception_queue,
                rgb_image=rgb,
                depth_image=depth,
            )
            push_target_pos = get_obstacle_pos_from_detection(refreshed_detection)
        except Exception as exc:
            print(f"Vision refresh before push unavailable: {exc}")
    if push_target_pos is None:
        push_target_pos = get_primary_object_pos(env.obs)
        push_source = "environment observation"
    if push_target_pos is None:
        raise RuntimeError("No object position found in robosuite observations.")
    execution_summary["push_source"] = push_source
    print(f"Push target source: {push_source}")
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

    run_success = bool(planning_completed and manipulation_success)

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
        success=run_success,
        counts=counts,
        duration_sec=duration_sec,
        context=build_run_context(fsm, run_success),
        execution_summary=execution_summary,
    )
    log_path = write_run_log(run_log)

    print("\nFailure counters:", counts)
    print(f"Run log written to: {log_path}")
    print("\n" + "=" * 60)
    print(f"Task completed! Final state: {fsm.get_current_state().value}")
    print("\nMinimal integration path completed.")


if __name__ == "__main__":
    main()
