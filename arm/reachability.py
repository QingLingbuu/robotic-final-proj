"""Reach summary diagnosis helpers for execution debugging."""

import numpy as np

from planner.candidates import PANDA_GRIPPER_MAX_WIDTH


def _tail_values(history_tail, key):
    values = []
    for entry in history_tail or []:
        value = entry.get(key)
        if value is not None:
            values.append(value)
    return values


def _action_saturated(actions, threshold=0.98):
    for action in actions:
        if np.max(np.abs(np.asarray(action, dtype=float))) >= float(threshold):
            return True
    return False


def diagnose_reachability(reach_summary, candidate=None, success_tolerance=0.015):
    history_tail = reach_summary.get("history_tail", []) if reach_summary else []
    position_actions = _tail_values(history_tail, "position_delta_action")
    orientation_actions = _tail_values(history_tail, "orientation_delta_action")
    final_error = reach_summary.get("final_error_to_settle")
    if final_error is None:
        final_error = reach_summary.get("final_error_to_candidate")

    gripper_width = None if candidate is None else candidate.get("gripper_width")
    gripper_width_risk = False
    if gripper_width is not None:
        gripper_width_risk = float(gripper_width) > PANDA_GRIPPER_MAX_WIDTH

    position_saturated = _action_saturated(position_actions)
    orientation_saturated = _action_saturated(orientation_actions)
    near_success = False
    if final_error is not None:
        near_success = float(final_error) <= float(success_tolerance) * 1.5

    if reach_summary.get("reach_success"):
        likely_cause = "reachable"
    elif gripper_width_risk:
        likely_cause = "candidate_gripper_width_exceeds_panda_limit"
    elif position_saturated and orientation_saturated:
        likely_cause = "pose_target_likely_outside_current_arm_workspace"
    elif position_saturated:
        likely_cause = "position_control_saturated_or_base_torso_needed"
    elif orientation_saturated:
        likely_cause = "orientation_target_hard_or_unreachable"
    elif near_success:
        likely_cause = "near_success_tolerance_or_settle_height_issue"
    else:
        likely_cause = "needs_action_space_or_reachability_probe"

    return {
        "reach_success": bool(reach_summary.get("reach_success", False)),
        "candidate_type": reach_summary.get("candidate_type"),
        "final_error": None if final_error is None else float(final_error),
        "near_success": bool(near_success),
        "position_saturated": bool(position_saturated),
        "orientation_saturated": bool(orientation_saturated),
        "gripper_width": None if gripper_width is None else float(gripper_width),
        "gripper_width_risk": bool(gripper_width_risk),
        "likely_cause": likely_cause,
    }
