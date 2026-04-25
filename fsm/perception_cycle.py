"""Perception-to-FSM cycle helpers for live RGB-D observations."""

from ipc.perception_queue import Empty, read_detected_objects
from vision.detected_objects import validate_detected_objects

from fsm.state_machine import State


def choose_next_state_from_detection(detected_objects):
    """Choose the next planning state from a validated perception payload."""
    if detected_objects.get("obstacles"):
        return State.CLEARING
    return State.GRASPING


def consume_perception_queue(fsm, perception_queue):
    """Consume one queue payload and update the FSM according to perception status."""
    try:
        latest_detection = read_detected_objects(perception_queue)
    except Empty:
        print("  Queue read timed out; routing to RETRY_SENSING.")
        fsm.handle_perception_timeout()
        return None, False

    if not validate_detected_objects(latest_detection):
        print("  Payload schema invalid; routing to RETRY_SENSING.")
        fsm.handle_invalid_perception()
        return latest_detection, False

    if latest_detection["status"] != "ready":
        print(
            "  Payload not ready "
            f"(status={latest_detection['status']}); routing to RETRY_SENSING."
        )
        fsm.handle_invalid_perception()
        return latest_detection, False

    next_state = choose_next_state_from_detection(latest_detection)
    print(f"  Payload valid and ready; transitioning to {next_state.value}.")
    fsm.transition_to(next_state)
    return latest_detection, True


def run_live_perception_cycle(
    fsm,
    perception_queue,
    env=None,
    perception_loop=None,
    max_attempts=None,
):
    """Run live vision inference and feed the result into the FSM."""
    attempts_allowed = max_attempts
    if attempts_allowed is None:
        attempts_allowed = fsm.max_planning_retries

    latest_detection = None
    for attempt_idx in range(1, attempts_allowed + 1):
        print(f"\nRunning live perception cycle: attempt {attempt_idx}/{attempts_allowed}")
        fsm.transition_to(State.PLANNING)

        if env is not None and perception_loop is not None:
            rgb, depth, _ = env.get_observation()
            latest_detection = perception_loop.publish_from_observation(
                perception_queue=perception_queue,
                rgb_image=rgb,
                depth_image=depth,
            )
            print(
                "  Published live payload with "
                f"status={latest_detection['status']} "
                f"and conf={latest_detection['target']['conf']:.2f}"
            )

        latest_detection, cycle_ok = consume_perception_queue(fsm, perception_queue)
        if cycle_ok:
            return latest_detection, True
        if fsm.is_terminal():
            break

    return latest_detection, False
