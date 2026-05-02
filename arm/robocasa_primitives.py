"""RoboCasa single-arm reach, gripper, and lift primitives."""

import time

import numpy as np
import robosuite.utils.transform_utils as T

from arm.robocasa_execution import (
    build_flat_reach_action,
    get_robot0_eef_pos,
    get_robot0_eef_quat,
    read_gripper_width,
)

GRIPPER_CLOSED = 1.0
GRIPPER_OPEN = -1.0
GRIPPER_ACTUATION_STEPS = 50
POST_GRIPPER_SETTLE_STEPS = 12
LIFT_HEIGHT = 0.20
SINGLE_GRASP_CONTACT_Z_OFFSET = 0.028
HANDLE_TOP_DOWN_EEF_Z_OFFSET = -0.025
HANDLE_TOP_DOWN_REACH_TOLERANCE = 0.008
REACH_POSITION_ACTION_LIMIT = 1.0
TOP_DOWN_ORI_GAIN = 0.8
TOP_DOWN_ORI_MAX_ACTION = 0.18
TOP_DOWN_MAINTAIN_ORI_MAX_ACTION = 0.06
TOP_DOWN_YAW_GAIN = 0.65
TOP_DOWN_YAW_MAX_ACTION = 0.08
HANDLE_TOP_DOWN_YAW_GAIN = 0.90
HANDLE_TOP_DOWN_YAW_MAX_ACTION = 0.12
TOP_DOWN_SETTLE_ACTION_SCALE = 0.25
HANDLE_TOP_DOWN_SETTLE_ACTION_SCALE = 0.45


def build_target_rotation_for_top_down(candidate, current_quat=None):
    closing = np.asarray(candidate["orientation"][0], dtype=float)
    approach = np.asarray(candidate["orientation"][2], dtype=float)

    def normalize(vector):
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return vector / norm

    closing = normalize(closing)
    approach = normalize(approach)
    if closing is None or approach is None:
        raise RuntimeError("top-down candidate has invalid orientation axes.")

    def make_rotation(closing_axis, approach_axis):
        x_axis = normalize(np.cross(closing_axis, approach_axis))
        if x_axis is None:
            raise RuntimeError("top-down candidate closing and approach axes are degenerate.")
        rotation = np.column_stack([x_axis, closing_axis, approach_axis])
        return rotation

    rotation_a = make_rotation(closing, approach)
    rotation_b = make_rotation(-closing, approach)
    if current_quat is None:
        return rotation_a

    current_rot = T.quat2mat(current_quat)

    def angle_between(target_rot):
        error_rot = target_rot @ current_rot.T
        trace_val = np.clip((np.trace(error_rot) - 1.0) / 2.0, -1.0, 1.0)
        return np.arccos(trace_val)

    return rotation_a if angle_between(rotation_a) <= angle_between(rotation_b) else rotation_b


def compute_orientation_action(current_quat, target_rot, gain=TOP_DOWN_ORI_GAIN, max_action=TOP_DOWN_ORI_MAX_ACTION):
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
        ],
        dtype=float,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=np.float32), float(angle)
    action = np.clip((axis / axis_norm) * angle * gain, -max_action, max_action)
    return np.asarray(action, dtype=np.float32), float(angle)


def compute_top_down_yaw_action(current_quat, candidate, gain=TOP_DOWN_YAW_GAIN, max_action=TOP_DOWN_YAW_MAX_ACTION):
    current_rot = T.quat2mat(current_quat)
    current_closing = np.asarray(current_rot[:, 1], dtype=float)
    desired_closing = np.asarray(candidate["orientation"][0], dtype=float)
    current_xy = current_closing[:2]
    desired_xy = desired_closing[:2]
    current_norm = float(np.linalg.norm(current_xy))
    desired_norm = float(np.linalg.norm(desired_xy))
    if current_norm < 1e-8 or desired_norm < 1e-8:
        return np.zeros(3, dtype=np.float32), 0.0
    current_xy = current_xy / current_norm
    desired_xy = desired_xy / desired_norm
    cross_z = current_xy[0] * desired_xy[1] - current_xy[1] * desired_xy[0]
    dot = float(np.clip(np.dot(current_xy, desired_xy), -1.0, 1.0))
    yaw_error = float(np.arctan2(cross_z, dot))
    if abs(yaw_error) > np.pi / 2.0:
        yaw_error -= float(np.sign(yaw_error) * np.pi)
    action = np.array([0.0, 0.0, np.clip(yaw_error * gain, -max_action, max_action)], dtype=np.float32)
    return action, abs(yaw_error)


def compute_position_delta(pos_error, action_mapping=None, position_gain=15.0, step_limit=0.10):
    desired_world_delta = np.clip(
        np.asarray(pos_error, dtype=float) * float(position_gain),
        -float(step_limit),
        float(step_limit),
    )
    if action_mapping is None:
        position_delta = desired_world_delta
    else:
        position_delta = np.linalg.pinv(np.asarray(action_mapping, dtype=float)) @ desired_world_delta
        position_delta = np.clip(
            position_delta,
            -REACH_POSITION_ACTION_LIMIT,
            REACH_POSITION_ACTION_LIMIT,
        )
    return desired_world_delta, position_delta


def execute_reach(
    env,
    obs,
    candidate,
    action_mapping,
    hover_offset=0.08,
    settle_offset=0.03,
    max_steps=300,
    position_gain=10.0,
    step_limit=0.06,
    reach_tolerance=0.015,
    render_sleep_sec=0.005,
):
    return execute_oriented_top_down_reach(
        env,
        obs=obs,
        candidate=candidate,
        action_mapping=action_mapping,
        hover_offset=hover_offset,
        settle_offset=settle_offset,
        max_steps=max_steps,
        position_gain=position_gain,
        step_limit=step_limit,
        reach_tolerance=reach_tolerance,
        render_sleep_sec=render_sleep_sec,
        align_orientation=False,
    )


def execute_oriented_top_down_reach(
    env,
    obs,
    candidate,
    action_mapping,
    hover_offset=0.12,
    settle_offset=0.018,
    max_steps=300,
    align_steps=220,
    position_gain=4.5,
    step_limit=0.02,
    reach_tolerance=0.018,
    orientation_tolerance=0.08,
    render_sleep_sec=0.005,
    align_orientation=True,
):
    candidate_pos = np.asarray(candidate["pos"], dtype=float)
    pre_hover_target = candidate_pos + np.array([0.0, 0.0, float(hover_offset) + 0.08], dtype=float)
    hover_target = candidate_pos + np.array([0.0, 0.0, float(hover_offset)], dtype=float)
    if candidate.get("grasp_type") == "handle_top_down":
        settle_target = candidate_pos + np.array([0.0, 0.0, HANDLE_TOP_DOWN_EEF_Z_OFFSET], dtype=float)
        reach_tolerance = min(float(reach_tolerance), HANDLE_TOP_DOWN_REACH_TOLERANCE)
    else:
        settle_target = candidate_pos + np.array(
            [0.0, 0.0, float(settle_offset) - SINGLE_GRASP_CONTACT_Z_OFFSET],
            dtype=float,
        )

    current_obs = obs
    history = []
    reached = True
    orientation_success = not align_orientation
    target_rot = None

    if align_orientation:
        target_rot = build_target_rotation_for_top_down(candidate, current_quat=get_robot0_eef_quat(current_obs))

    phase_sequence = [("pre_hover", pre_hover_target)]
    if align_orientation:
        phase_sequence.append(("align_orientation", pre_hover_target))
    phase_sequence.extend([("hover", hover_target), ("settle", settle_target)])

    for phase_name, phase_target in phase_sequence:
        phase_success = False
        phase_max_steps = int(align_steps) if phase_name == "align_orientation" else int(max_steps)
        for step_idx in range(phase_max_steps):
            current_pos = get_robot0_eef_pos(current_obs)
            error = phase_target - current_pos
            error_norm = float(np.linalg.norm(error))
            orientation_delta = None
            ori_error_norm = None
            if align_orientation and target_rot is not None and phase_name != "pre_hover":
                yaw_gain = (
                    HANDLE_TOP_DOWN_YAW_GAIN
                    if candidate.get("grasp_type") == "handle_top_down"
                    else TOP_DOWN_YAW_GAIN
                )
                yaw_max_action = (
                    HANDLE_TOP_DOWN_YAW_MAX_ACTION
                    if candidate.get("grasp_type") == "handle_top_down"
                    else TOP_DOWN_YAW_MAX_ACTION
                )
                orientation_delta, ori_error_norm = compute_top_down_yaw_action(
                    get_robot0_eef_quat(current_obs),
                    candidate,
                    gain=yaw_gain,
                    max_action=yaw_max_action,
                )
                if ori_error_norm <= float(orientation_tolerance):
                    orientation_success = True
            history.append(
                {
                    "phase": phase_name,
                    "step": step_idx,
                    "eef_pos": current_pos.tolist(),
                    "target_pos": phase_target.tolist(),
                    "error_norm": error_norm,
                    "ori_error_norm": ori_error_norm,
                    "orientation_delta_action": None
                    if orientation_delta is None
                    else np.asarray(orientation_delta, dtype=float).tolist(),
                }
            )

            if phase_name == "align_orientation":
                if orientation_success:
                    phase_success = True
                    break
                position_delta = np.zeros(3, dtype=float)
            else:
                if error_norm <= float(reach_tolerance):
                    phase_success = True
                    break
                _, position_delta = compute_position_delta(
                    error,
                    action_mapping=action_mapping,
                    position_gain=position_gain,
                    step_limit=step_limit,
                )
                if phase_name == "settle":
                    settle_scale = (
                        HANDLE_TOP_DOWN_SETTLE_ACTION_SCALE
                        if candidate.get("grasp_type") == "handle_top_down"
                        else TOP_DOWN_SETTLE_ACTION_SCALE
                    )
                    position_delta = np.asarray(position_delta, dtype=float) * settle_scale

            current_obs, _, _, _ = env.step(
                build_flat_reach_action(
                    env,
                    position_delta,
                    orientation_delta=orientation_delta,
                    gripper_close=GRIPPER_OPEN,
                )
            )
            if step_idx % 3 == 0:
                env.render()
                if render_sleep_sec > 0.0:
                    time.sleep(render_sleep_sec)

        if not phase_success:
            reached = False
            if phase_name == "align_orientation":
                break

    final_pos = get_robot0_eef_pos(current_obs)
    final_rot = T.quat2mat(get_robot0_eef_quat(current_obs))
    target_closing = np.asarray(candidate["orientation"][0], dtype=float)
    target_lateral = np.asarray(candidate["orientation"][1], dtype=float)
    final_x_axis = np.asarray(final_rot[:, 0], dtype=float)
    final_y_axis = np.asarray(final_rot[:, 1], dtype=float)

    def normalized_dot(axis_a, axis_b):
        axis_a = np.asarray(axis_a, dtype=float)
        axis_b = np.asarray(axis_b, dtype=float)
        norm_a = float(np.linalg.norm(axis_a))
        norm_b = float(np.linalg.norm(axis_b))
        if norm_a < 1e-8 or norm_b < 1e-8:
            return None
        return float(np.dot(axis_a, axis_b) / (norm_a * norm_b))

    return current_obs, {
        "reach_success": bool(reached),
        "candidate_id": int(candidate["id"]),
        "candidate_type": candidate["grasp_type"],
        "candidate_pos": candidate_pos.tolist(),
        "pre_hover_target": pre_hover_target.tolist(),
        "hover_target": hover_target.tolist(),
        "settle_target": settle_target.tolist(),
        "orientation_aligned": bool(orientation_success),
        "target_rotation": None if target_rot is None else target_rot.tolist(),
        "final_eef_x_axis": final_x_axis.tolist(),
        "final_eef_y_axis": final_y_axis.tolist(),
        "final_closing_axis": final_y_axis.tolist(),
        "final_x_dot_target_closing": normalized_dot(final_x_axis, target_closing),
        "final_y_dot_target_closing": normalized_dot(final_y_axis, target_closing),
        "final_closing_dot_target_closing": normalized_dot(final_y_axis, target_closing),
        "final_x_dot_target_lateral": normalized_dot(final_x_axis, target_lateral),
        "final_y_dot_target_lateral": normalized_dot(final_y_axis, target_lateral),
        "robot0_eef_final": final_pos.tolist(),
        "final_error_to_settle": float(np.linalg.norm(settle_target - final_pos)),
        "history_tail": history[-10:],
    }


def execute_close(
    env,
    obs,
    actuation_steps=GRIPPER_ACTUATION_STEPS,
    settle_steps=POST_GRIPPER_SETTLE_STEPS,
    render_sleep_sec=0.005,
):
    eef_before_close = get_robot0_eef_pos(obs)
    width_before = read_gripper_width(obs)
    current_obs = obs

    close_action = build_flat_reach_action(env, [0.0, 0.0, 0.0], gripper_close=GRIPPER_CLOSED)
    for step in range(int(actuation_steps)):
        current_obs, _, _, _ = env.step(close_action)
        if step % 5 == 0:
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

    for step in range(int(settle_steps)):
        current_obs, _, _, _ = env.step(close_action)
        if step % 5 == 0:
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

    eef_after_close = get_robot0_eef_pos(current_obs)
    width_after = read_gripper_width(current_obs)

    return current_obs, {
        "close_success": True,
        "eef_before_close": eef_before_close.tolist(),
        "eef_after_close": eef_after_close.tolist(),
        "gripper_width_before": width_before,
        "gripper_width_after": width_after,
        "actuation_steps": int(actuation_steps),
        "settle_steps": int(settle_steps),
    }


def get_object_pos(env, obj_name="obj"):
    obj_body_id = getattr(env, "obj_body_id", None)
    sim = getattr(env, "sim", None)
    if obj_body_id is None or sim is None or obj_name not in obj_body_id:
        return None
    return np.asarray(sim.data.body_xpos[obj_body_id[obj_name]], dtype=float)


def object_has_gripper_contact(env, obj_name="obj"):
    objects = getattr(env, "objects", None)
    check_contact = getattr(env, "check_contact", None)
    if objects is None or obj_name not in objects or check_contact is None:
        return None
    for robot in getattr(env, "robots", []):
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and check_contact(objects[obj_name], gripper):
            return True
    return False


def execute_lift(
    env,
    obs,
    action_mapping=None,
    lift_height=LIFT_HEIGHT,
    max_steps=300,
    position_gain=15.0,
    step_limit=0.10,
    lift_tolerance=0.008,
    render_sleep_sec=0.005,
):
    eef_before_lift = get_robot0_eef_pos(obs)
    obj_before_lift = get_object_pos(env)
    lift_target = eef_before_lift + np.array([0.0, 0.0, float(lift_height)], dtype=float)

    current_obs = obs
    history = []
    lift_success = False
    steps_used = 0

    for step_idx in range(int(max_steps)):
        current_pos = get_robot0_eef_pos(current_obs)
        error = lift_target - current_pos
        error_norm = float(np.linalg.norm(error))
        history.append(
            {
                "step": step_idx,
                "eef_pos": current_pos.tolist(),
                "target_pos": lift_target.tolist(),
                "error_norm": error_norm,
            }
        )
        steps_used = step_idx + 1
        if error_norm <= float(lift_tolerance):
            lift_success = True
            break

        _, position_delta = compute_position_delta(
            error,
            action_mapping=action_mapping,
            position_gain=position_gain,
            step_limit=step_limit,
        )
        current_obs, _, _, _ = env.step(
            build_flat_reach_action(env, position_delta, gripper_close=GRIPPER_CLOSED)
        )
        if step_idx % 5 == 0:
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

    eef_after_lift = get_robot0_eef_pos(current_obs)
    obj_after_lift = get_object_pos(env)
    object_lift_delta_z = None
    if obj_before_lift is not None and obj_after_lift is not None:
        object_lift_delta_z = float(obj_after_lift[2] - obj_before_lift[2])
    lift_delta_z = float(eef_after_lift[2] - eef_before_lift[2])
    object_contact = object_has_gripper_contact(env)
    eef_lift_success = bool(lift_success)
    object_lift_success = object_lift_delta_z is not None and object_lift_delta_z >= min(0.05, float(lift_height) * 0.35)
    if object_lift_delta_z is not None:
        lift_success = bool(eef_lift_success and object_lift_success)

    return current_obs, {
        "lift_success": bool(lift_success),
        "eef_lift_success": bool(eef_lift_success),
        "object_lift_success": None if object_lift_delta_z is None else bool(object_lift_success),
        "object_gripper_contact": object_contact,
        "eef_before_lift": eef_before_lift.tolist(),
        "eef_after_lift": eef_after_lift.tolist(),
        "object_before_lift": None if obj_before_lift is None else obj_before_lift.tolist(),
        "object_after_lift": None if obj_after_lift is None else obj_after_lift.tolist(),
        "lift_target": lift_target.tolist(),
        "lift_delta_z": lift_delta_z,
        "object_lift_delta_z": object_lift_delta_z,
        "final_error_to_lift_target": float(np.linalg.norm(lift_target - eef_after_lift)),
        "steps_used": steps_used,
        "history_tail": history[-10:],
    }


def execute_place(
    env,
    obs,
    place_target,
    action_mapping=None,
    max_steps_per_waypoint=220,
    position_gain=8.0,
    step_limit=0.05,
    place_tolerance=0.018,
    release_steps=40,
    retract_height=0.10,
    render_sleep_sec=0.005,
):
    target = np.asarray(place_target, dtype=float)
    start_pos = get_robot0_eef_pos(obs)
    place_high = np.asarray([target[0], target[1], max(start_pos[2], target[2] + 0.12)], dtype=float)
    place_low = np.asarray([target[0], target[1], target[2]], dtype=float)
    retract_target = np.asarray([target[0], target[1], target[2] + float(retract_height)], dtype=float)
    waypoints = [("place_high", place_high), ("place_low", place_low)]

    current_obs = obs
    history = []
    reached = True

    for phase_name, waypoint in waypoints:
        phase_success = False
        for step_idx in range(int(max_steps_per_waypoint)):
            current_pos = get_robot0_eef_pos(current_obs)
            error = waypoint - current_pos
            error_norm = float(np.linalg.norm(error))
            history.append(
                {
                    "phase": phase_name,
                    "step": step_idx,
                    "eef_pos": current_pos.tolist(),
                    "target_pos": waypoint.tolist(),
                    "error_norm": error_norm,
                }
            )
            if error_norm <= float(place_tolerance):
                phase_success = True
                break
            _, position_delta = compute_position_delta(
                error,
                action_mapping=action_mapping,
                position_gain=position_gain,
                step_limit=step_limit,
            )
            if phase_name == "place_low":
                position_delta = np.asarray(position_delta, dtype=float) * 0.35
            current_obs, _, _, _ = env.step(
                build_flat_reach_action(env, position_delta, gripper_close=GRIPPER_CLOSED)
            )
            if step_idx % 5 == 0:
                env.render()
                if render_sleep_sec > 0.0:
                    time.sleep(render_sleep_sec)
        if not phase_success:
            reached = False
            break

    release_action = build_flat_reach_action(env, [0.0, 0.0, 0.0], gripper_close=GRIPPER_OPEN)
    for step_idx in range(int(release_steps)):
        current_obs, _, _, _ = env.step(release_action)
        if step_idx % 5 == 0:
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

    retract_success = False
    for step_idx in range(int(max_steps_per_waypoint)):
        current_pos = get_robot0_eef_pos(current_obs)
        error = retract_target - current_pos
        error_norm = float(np.linalg.norm(error))
        history.append(
            {
                "phase": "retract",
                "step": step_idx,
                "eef_pos": current_pos.tolist(),
                "target_pos": retract_target.tolist(),
                "error_norm": error_norm,
            }
        )
        if error_norm <= float(place_tolerance):
            retract_success = True
            break
        _, position_delta = compute_position_delta(
            error,
            action_mapping=action_mapping,
            position_gain=position_gain,
            step_limit=step_limit,
        )
        current_obs, _, _, _ = env.step(
            build_flat_reach_action(env, position_delta, gripper_close=GRIPPER_OPEN)
        )
        if step_idx % 5 == 0:
            env.render()
            if render_sleep_sec > 0.0:
                time.sleep(render_sleep_sec)

    final_pos = get_robot0_eef_pos(current_obs)
    return current_obs, {
        "place_success": bool(reached and retract_success),
        "place_target": target.tolist(),
        "place_high": place_high.tolist(),
        "place_low": place_low.tolist(),
        "retract_target": retract_target.tolist(),
        "robot0_eef_final": final_pos.tolist(),
        "final_error_to_retract": float(np.linalg.norm(retract_target - final_pos)),
        "release_steps": int(release_steps),
        "history_tail": history[-10:],
    }
