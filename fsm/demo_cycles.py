"""Minimal demo cycle drivers for perception, clearing, and grasp paths."""

from ipc.perception_queue import publish_detected_objects
from vision.detected_objects import build_detected_objects

from fsm.perception_cycle import consume_perception_queue
from fsm.state_machine import State


def run_demo_cycles(fsm, perception_queue, conf_thresh):
    """Run the project demo cycles and return overall completion status."""
    ready_cycle_ok = run_perception_cycle(
        fsm=fsm,
        perception_queue=perception_queue,
        conf_thresh=conf_thresh,
        label="ready sample",
        target_conf=0.95,
        publish_sample=True,
    )
    low_conf_cycle_ok = run_perception_cycle(
        fsm=fsm,
        perception_queue=perception_queue,
        conf_thresh=conf_thresh,
        label="low-confidence sample",
        target_conf=0.25,
        publish_sample=True,
    )
    stale_cycle_ok = run_perception_cycle(
        fsm=fsm,
        perception_queue=perception_queue,
        conf_thresh=conf_thresh,
        label="stale queue sample",
        target_conf=0.0,
        publish_sample=False,
    )
    clearing_cycle_ok = run_clearing_cycle(
        fsm=fsm,
        label="blocked clearing sample",
        blocked=True,
    )
    grasp_cycle_ok = run_grasp_cycle(
        fsm=fsm,
        label="empty grasp sample",
        grasp_empty=True,
    )
    return (
        ready_cycle_ok
        and low_conf_cycle_ok
        and stale_cycle_ok
        and clearing_cycle_ok
        and grasp_cycle_ok
    )


def run_perception_cycle(fsm, perception_queue, conf_thresh, label, target_conf, publish_sample):
    """Exercise one perception-to-FSM integration cycle."""
    print(f"\nRunning perception cycle: {label}")

    if publish_sample:
        detected_objects = build_detected_objects(
            conf_thresh=conf_thresh,
            target_label="demo_target",
            target_pos=[0.0, 0.0, 0.5],
            target_conf=target_conf,
            obstacles=[
                {
                    "label": "demo_obstacle_0",
                    "pos": [0.2, 0.0, 0.4],
                    "id": 1,
                    "conf": 0.8,
                }
            ],
            status="ready",
        )
        publish_detected_objects(perception_queue, detected_objects)
        print(
            "  Published payload with "
            f"status={detected_objects['status']} "
            f"and conf={detected_objects['target']['conf']:.2f}"
        )

    _, cycle_ok = consume_perception_queue(fsm, perception_queue)
    if not cycle_ok:
        return False

    fsm.transition_to(State.PLANNING)
    return True


def run_clearing_cycle(fsm, label, blocked):
    """Exercise one clearing-to-planning transition or retry path."""
    print(f"\nRunning clearing cycle: {label}")
    fsm.transition_to(State.CLEARING)

    if blocked:
        print("  Clearing step blocked; routing to RETRY_PUSH.")
        fsm.handle_push_blocked()
        return False

    print("  Clearing step completed; returning to PLANNING.")
    fsm.transition_to(State.PLANNING)
    return True


def run_grasp_cycle(fsm, label, grasp_empty):
    """Exercise one grasp verification path or retry path."""
    print(f"\nRunning grasp cycle: {label}")
    fsm.transition_to(State.GRASPING)

    if grasp_empty:
        print("  Grasp verification empty; routing to RETRY_GRASP.")
        fsm.handle_empty_grasp()
        return False

    print("  Grasp closed successfully; transitioning through VERIFYING.")
    fsm.transition_to(State.VERIFYING)
    fsm.transition_to(State.PLANNING)
    return True


def build_run_context(fsm, task_completed):
    """Summarize the final loop outcome for the experiment log."""
    state = fsm.get_current_state()
    if task_completed:
        return "Perception, clearing, and grasp cycles completed without retries."
    if state == State.FAILED:
        return "Retry budget exhausted while handling invalid or stale perception data."
    if state == State.EMERGENCY:
        return "Execution entered EMERGENCY after a safety event."
    return (
        "Demo loop exercised perception, clearing, and grasp recovery paths but did not complete the "
        f"full manipulation task; final state={state.value}."
    )
