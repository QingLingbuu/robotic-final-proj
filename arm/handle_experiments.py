"""Experimental handle-grasp reach paths for RoboCasa demos."""

import json
import sys
import time

import numpy as np
import robosuite.utils.transform_utils as T

from arm.calibration import extract_base_to_world_rotation
from arm.handle_phases import HANDLE_STANDOFF, build_handle_reach_phases, build_oblique_reach_phases
from arm.robocasa_execution import build_flat_reach_action, get_robot0_eef_pos, get_robot0_eef_quat
from arm.robocasa_primitives import GRIPPER_OPEN, compute_position_delta
from arm.handle_validation import summarize_handle_reach_validation
from planner.candidates import normalize_vector

HANDLE_ORI_GAIN = 1.5
HANDLE_ORI_MAX_ACTION = 0.5
HANDLE_REACH_ABORT_POS_ERROR = 0.12

def build_target_rotation_for_handle(candidate, current_quat=None):
    closing = np.asarray(candidate["orientation"][0], dtype=float)
    lateral = np.asarray(candidate["orientation"][1], dtype=float)
    approach = np.asarray(candidate["orientation"][2], dtype=float)

    def make_rotation(closing_axis, lateral_axis, approach_axis):
        rotation = np.column_stack([lateral_axis, closing_axis, approach_axis])
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return rotation

    rotation_a = make_rotation(closing, lateral, approach)
    rotation_b = make_rotation(-closing, -lateral, approach)

    if current_quat is None:
        return rotation_a

    current_rot = T.quat2mat(current_quat)

    def angle_between(target_rot, current_rotation):
        error_rot = target_rot @ current_rotation.T
        trace_val = np.clip((np.trace(error_rot) - 1.0) / 2.0, -1.0, 1.0)
        return np.arccos(trace_val)

    angle_a = angle_between(rotation_a, current_rot)
    angle_b = angle_between(rotation_b, current_rot)
    return rotation_a if angle_a <= angle_b else rotation_b


def compute_orientation_action(current_quat, target_rot, base_to_world_rot=None, gain=HANDLE_ORI_GAIN, max_action=HANDLE_ORI_MAX_ACTION):
    current_rot = T.quat2mat(current_quat)
    error_rot = target_rot @ current_rot.T
    trace_val = np.clip((np.trace(error_rot) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace_val)
    if angle < 1e-4:
        return np.zeros(3, dtype=np.float32), 0.0
    axis = np.array(
        [
            error_rot[2, 1] - error_rot[1, 2],
            error_rot[0, 2] - error_rot[2, 0],
            error_rot[1, 0] - error_rot[0, 1],
        ]
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=np.float32), float(angle)
    axis = axis / axis_norm
    error_axis_angle_world = axis * angle
    if base_to_world_rot is not None:
        error_axis_angle = base_to_world_rot.T @ error_axis_angle_world
    else:
        error_axis_angle = error_axis_angle_world
    action = np.clip(error_axis_angle * gain, -max_action, max_action)
    return np.asarray(action, dtype=np.float32), float(angle)


def reach_phase(
    env,
    current_obs,
    target_pos,
    target_rot,
    action_mapping,
    phase_name,
    max_steps,
    position_gain,
    step_limit,
    pos_tolerance,
    ori_tolerance,
    render_sleep_sec,
    history,
    base_to_world_rot=None,
    abort_pos_error=None,
):
    phase_success = False
    desired_world_delta = np.zeros(3, dtype=float)
    abort_reason = None
    for step_idx in range(int(max_steps)):
        current_pos = get_robot0_eef_pos(current_obs)
        current_quat = get_robot0_eef_quat(current_obs)
        pos_error = target_pos - current_pos
        pos_error_norm = float(np.linalg.norm(pos_error))

        if target_rot is not None:
            ori_action, ori_error_norm = compute_orientation_action(
                current_quat,
                target_rot,
                base_to_world_rot=base_to_world_rot,
            )
        else:
            ori_action = None
            ori_error_norm = 0.0

        history.append(
            {
                "phase": phase_name,
                "step": step_idx,
                "eef_pos": current_pos.tolist(),
                "target_pos": target_pos.tolist(),
                "pos_error_norm": pos_error_norm,
                "ori_error_norm": ori_error_norm,
            }
        )

        pos_ok = pos_error_norm <= float(pos_tolerance)
        ori_ok = ori_error_norm < float(ori_tolerance)
        if pos_ok and ori_ok:
            phase_success = True
            break

        if abort_pos_error is not None and target_rot is not None and pos_error_norm > float(abort_pos_error):
            abort_reason = "position_drift_during_orientation_alignment"
            history[-1]["abort_reason"] = abort_reason
            break

        if pos_tolerance < 900.0:
            desired_world_delta, position_delta = compute_position_delta(
                pos_error,
                action_mapping=action_mapping,
                position_gain=position_gain,
                step_limit=step_limit,
            )
        else:
            position_delta = np.zeros(3, dtype=np.float32)

        history[-1].update(
            {
                "desired_world_delta": np.asarray(desired_world_delta, dtype=float).tolist(),
                "position_delta_action": np.asarray(position_delta, dtype=float).tolist(),
                "orientation_delta_action": None if ori_action is None else np.asarray(ori_action, dtype=float).tolist(),
            }
        )

        current_obs, _, _, _ = env.step(
            build_flat_reach_action(
                env,
                position_delta,
                orientation_delta=ori_action,
                gripper_close=GRIPPER_OPEN,
            )
        )
        if step_idx % 3 == 0:
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

    return current_obs, phase_success, abort_reason


def _run_handle_phases(
    env,
    obs,
    candidate,
    action_mapping,
    phases,
    pre_approach_target,
    label,
    max_steps,
    position_gain,
    step_limit,
    render_sleep_sec,
    base_to_world_rot,
    abort_pos_error=None,
):
    current_obs = obs
    history = []
    phase_results = []
    reached = True

    for phase_name, phase_target, pos_tol, phase_rot, ori_tol in phases:
        print(f"[{label}] Starting phase: {phase_name} (pos_tol={pos_tol:.3f}, ori_tol={ori_tol:.3f})")
        sys.stdout.flush()
        current_obs, phase_ok, abort_reason = reach_phase(
            env,
            current_obs,
            phase_target,
            phase_rot,
            action_mapping,
            phase_name,
            max_steps,
            position_gain,
            step_limit,
            pos_tol,
            ori_tol,
            render_sleep_sec,
            history,
            base_to_world_rot=base_to_world_rot,
            abort_pos_error=abort_pos_error,
        )
        final_entry = history[-1] if history else {}
        print(
            f"[{label}] Phase {phase_name} done: ok={phase_ok}, "
            f"pos_err={final_entry.get('pos_error_norm', '?'):.4f}, "
            f"ori_err={final_entry.get('ori_error_norm', '?'):.4f}"
        )
        sys.stdout.flush()
        phase_results.append(
            {
                "phase": phase_name,
                "ok": bool(phase_ok),
                "steps_used": int(final_entry.get("step", -1)) + 1,
                "pos_error_norm": final_entry.get("pos_error_norm"),
                "ori_error_norm": final_entry.get("ori_error_norm"),
                "abort_reason": abort_reason,
            }
        )
        if not phase_ok:
            reached = False
        if abort_reason is not None:
            break

    candidate_pos = np.asarray(candidate["pos"], dtype=float)
    final_pos = get_robot0_eef_pos(current_obs)
    final_error_to_candidate = float(np.linalg.norm(candidate_pos - final_pos))
    validation = summarize_handle_reach_validation(phase_results, final_error_to_candidate)
    return current_obs, {
        "reach_only": True,
        "reach_success": bool(reached),
        "candidate_id": int(candidate["id"]),
        "candidate_type": candidate["grasp_type"],
        "candidate_pos": candidate_pos.tolist(),
        "pre_approach_target": pre_approach_target.tolist(),
        "robot0_eef_final": final_pos.tolist(),
        "final_error_to_candidate": final_error_to_candidate,
        "handle_reach_validation": validation,
        "phase_results": phase_results,
        "history_tail": history[-10:],
    }


def execute_reach_handle(
    env,
    obs,
    candidate,
    action_mapping,
    standoff=HANDLE_STANDOFF,
    max_steps=900,
    position_gain=15.0,
    step_limit=0.10,
    reach_tolerance=0.008,
    render_sleep_sec=0.005,
    include_contact_approach=True,
    abort_pos_error=None,
):
    candidate_pos = np.asarray(candidate["pos"], dtype=float)
    approach_axis = np.asarray(candidate["orientation"][2], dtype=float)
    approach_dir = approach_axis / max(np.linalg.norm(approach_axis), 1e-8)
    current_quat = get_robot0_eef_quat(obs)
    target_rot = build_target_rotation_for_handle(candidate, current_quat=current_quat)
    base_to_world_rot = extract_base_to_world_rotation(action_mapping)
    phases, pre_approach_target = build_handle_reach_phases(
        candidate_pos,
        approach_dir,
        target_rot,
        reach_tolerance,
        standoff=standoff,
        include_contact_approach=include_contact_approach,
    )

    current_rot = T.quat2mat(current_quat)
    print(
        json.dumps(
            {
                "handle_grasp_debug": {
                    "target_rot": target_rot.tolist(),
                    "current_eef_rot": current_rot.tolist(),
                    "current_eef_quat": current_quat.tolist(),
                    "candidate_pos": candidate_pos.tolist(),
                    "approach_dir": approach_dir.tolist(),
                    "pre_approach_target": pre_approach_target.tolist(),
                    "contact_approach_enabled": bool(include_contact_approach),
                    "target_rot_det": float(np.linalg.det(target_rot)),
                    "base_to_world_rot": base_to_world_rot.tolist() if base_to_world_rot is not None else None,
                }
            },
            indent=2,
        )
    )
    sys.stdout.flush()

    return _run_handle_phases(
        env,
        obs,
        candidate,
        action_mapping,
        phases,
        pre_approach_target,
        "handle_grasp",
        max_steps,
        position_gain,
        step_limit,
        render_sleep_sec,
        base_to_world_rot,
        abort_pos_error=abort_pos_error,
    )


def execute_reach_oblique_handle(
    env,
    obs,
    candidate,
    action_mapping,
    max_steps=450,
    position_gain=15.0,
    step_limit=0.10,
    reach_tolerance=0.008,
    render_sleep_sec=0.005,
    include_contact_approach=True,
    abort_pos_error=None,
):
    candidate_pos = np.asarray(candidate["pos"], dtype=float)
    approach_dir = normalize_vector(np.asarray(candidate["orientation"][2], dtype=float))
    if approach_dir is None:
        raise RuntimeError("handle_oblique_grasp candidate has invalid approach direction.")

    current_quat = get_robot0_eef_quat(obs)
    target_rot = build_target_rotation_for_handle(candidate, current_quat=current_quat)
    base_to_world_rot = extract_base_to_world_rotation(action_mapping)
    phases, pre_approach_target = build_oblique_reach_phases(
        candidate_pos,
        approach_dir,
        target_rot,
        reach_tolerance,
        include_contact_approach=include_contact_approach,
    )

    print(
        json.dumps(
            {
                "handle_oblique_debug": {
                    "candidate_pos": candidate_pos.tolist(),
                    "approach_dir": approach_dir.tolist(),
                    "pre_approach_target": pre_approach_target.tolist(),
                    "contact_approach_enabled": bool(include_contact_approach),
                    "target_rot": target_rot.tolist(),
                    "base_to_world_rot": base_to_world_rot.tolist() if base_to_world_rot is not None else None,
                    "source_handle_width": candidate.get("source_handle_width"),
                }
            },
            indent=2,
        )
    )
    sys.stdout.flush()

    return _run_handle_phases(
        env,
        obs,
        candidate,
        action_mapping,
        phases,
        pre_approach_target,
        "handle_oblique",
        max_steps,
        position_gain,
        step_limit,
        render_sleep_sec,
        base_to_world_rot,
        abort_pos_error=abort_pos_error,
    )
