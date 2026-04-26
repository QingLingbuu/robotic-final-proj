"""Arm controller for grasping and pushing operations."""

import numpy as np


GRASP_HEIGHT_OFFSET = 0.05
DUAL_PRE_GRASP_HEIGHT_OFFSET = 0.09
DUAL_TRANSIT_HEIGHT_OFFSET = 0.16
DUAL_ALIGN_HEIGHT_OFFSET = 0.045
HANDLE_GRASP_Z_OFFSET = -0.04
PUSH_HEIGHT = 0.08
MOVE_STEP = 1.0
PRE_GRASP_SINGLE_ARM_STEP = 0.25
NEAR_MOVE_STEP = 0.35
MOVE_GAIN = 8.0
WAYPOINT_TOLERANCE = 0.025
DUAL_TRANSFER_TOLERANCE = 0.035
NEAR_WAYPOINT_DISTANCE = 0.06
DUAL_FINAL_DESCENT_OFFSET = 0.012
DUAL_FINAL_DESCENT_XY_TOLERANCE = 0.015
DUAL_FINAL_DESCENT_MAX_Z_ABOVE_TARGET = 0.06
DUAL_TRANSIT_MAX_STEPS = 240
DUAL_TRANSFER_SEGMENT_LENGTH = 0.06
DUAL_TRANSFER_SEGMENT_STEPS = 90
DUAL_TRANSFER_MIN_HEIGHT = 0.65
DUAL_CARRY_YAW_BIAS = 0.12
DUAL_UNSNAG_YAW_BIAS = 0.28
DUAL_UNSNAG_STEPS = 24
DUAL_WRIST_YAW_GAIN = 0.9
DUAL_WRIST_YAW_MAX_ACTION = 0.75
DUAL_WRIST_HANDLE_YAW_OFFSET = np.pi / 2.0
POST_GRIPPER_SETTLE_STEPS = 12
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0
GRIPPER_RELEASE_PARTIAL = -0.35
GRIPPER_ACTUATION_STEPS = 50
DEBUG_GRIPPER = True
GRASP_WIDTH_THRESHOLD = 0.02
OBJECT_MOVE_THRESHOLD = 0.02
OBJECT_LIFT_THRESHOLD = 0.02
OBJECT_PLACE_TOLERANCE = 0.08
SINGLE_LIFT_HEIGHT = 0.20
DUAL_LIFT_HEIGHT = 0.16
DUAL_RELEASE_RETRACT_DISTANCE = 0.055
DUAL_RELEASE_LIFT_CLEARANCE = 0.045
DUAL_RELEASE_SURFACE_CLEARANCE = 0.018
DUAL_RELEASE_PARTIAL_STEPS = 18
DUAL_RELEASE_RETRACT_STEPS = 90
MAX_GRASP_ATTEMPTS = 2
MAX_PLACE_ATTEMPTS = 2
PLACE_SETTLE_STEPS = 40
PUSH_APPROACH_DISTANCE = 0.12
PUSH_THROUGH_DISTANCE = 0.20
PUSH_SIDE_OFFSET = 0.13
PUSH_CONTACT_HEIGHT_OFFSET = -0.035
PUSH_PRE_HEIGHT_OFFSET = 0.08
PUSH_SETTLE_STEPS = 12
VISUAL_DUAL_GRASP_MIN_SPAN = 0.06


def compute_grasp_waypoints(target_pos):
    pre_grasp = [
        target_pos[0],
        target_pos[1],
        target_pos[2] + GRASP_HEIGHT_OFFSET,
    ]
    grasp = [
        target_pos[0],
        target_pos[1],
        target_pos[2],
    ]
    lift = [
        target_pos[0],
        target_pos[1],
        target_pos[2] + SINGLE_LIFT_HEIGHT,
    ]
    return pre_grasp, grasp, lift


def compute_push_waypoints(obstacle_pos, direction):
    """Build waypoints that contact the obstacle side, not its top."""
    direction = np.array(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("Push direction must be non-zero.")
    direction = direction / direction_norm

    obstacle_pos = np.array(obstacle_pos, dtype=float)
    contact_height = obstacle_pos[2] + PUSH_CONTACT_HEIGHT_OFFSET
    contact_pos = np.array(
        [
            obstacle_pos[0] - direction[0] * PUSH_SIDE_OFFSET,
            obstacle_pos[1] - direction[1] * PUSH_SIDE_OFFSET,
            contact_height,
        ],
        dtype=float,
    )

    push_pre = [
        contact_pos[0] - direction[0] * PUSH_APPROACH_DISTANCE,
        contact_pos[1] - direction[1] * PUSH_APPROACH_DISTANCE,
        contact_height + PUSH_PRE_HEIGHT_OFFSET,
    ]
    push_contact = [
        contact_pos[0],
        contact_pos[1],
        contact_height,
    ]
    push_end = [
        contact_pos[0] + direction[0] * PUSH_THROUGH_DISTANCE,
        contact_pos[1] + direction[1] * PUSH_THROUGH_DISTANCE,
        contact_height,
    ]
    return push_pre, push_contact, push_end


def compute_visual_dual_grasp_targets(
    target_pos,
    robot0_eef_pos,
    robot1_eef_pos,
    lateral_offset,
    vertical_offset=HANDLE_GRASP_Z_OFFSET,
    axis_mode="robots",
):
    """Infer two symmetric grasp points around a target from current arm layout."""
    target_pos = np.array(target_pos, dtype=float)
    robot0_eef_pos = np.array(robot0_eef_pos, dtype=float)
    robot1_eef_pos = np.array(robot1_eef_pos, dtype=float)

    axis_mode = str(axis_mode).strip().lower()
    if axis_mode == "x":
        span_axis = np.array([1.0, 0.0], dtype=float)
    elif axis_mode == "y":
        span_axis = np.array([0.0, 1.0], dtype=float)
    else:
        span_axis = robot1_eef_pos[:2] - robot0_eef_pos[:2]
        span_norm = np.linalg.norm(span_axis)
        if span_norm < 1e-6:
            span_axis = np.array([0.0, 1.0], dtype=float)
        else:
            span_axis = span_axis / span_norm

    span = max(float(lateral_offset), VISUAL_DUAL_GRASP_MIN_SPAN)
    left_xy = target_pos[:2] - span_axis * (span / 2.0)
    right_xy = target_pos[:2] + span_axis * (span / 2.0)

    left_target = np.array(
        [left_xy[0], left_xy[1], target_pos[2] + float(vertical_offset)],
        dtype=float,
    )
    right_target = np.array(
        [right_xy[0], right_xy[1], target_pos[2] + float(vertical_offset)],
        dtype=float,
    )
    return left_target, right_target


def move_to_waypoint(env, target_eef_pos, arm_idx=0):
    current = env.obs[f"robot{arm_idx}_eef_pos"]
    error = np.array(target_eef_pos) - current
    step_limit = MOVE_STEP
    if np.linalg.norm(error) <= NEAR_WAYPOINT_DISTANCE:
        step_limit = NEAR_MOVE_STEP
    action_delta = np.clip(error * MOVE_GAIN, -step_limit, step_limit)
    return action_delta


def move_to_waypoint_limited(env, target_eef_pos, arm_idx=0, step_limit=None):
    """Move toward a waypoint with an explicit max step size."""
    current = env.obs[f"robot{arm_idx}_eef_pos"]
    error = np.array(target_eef_pos) - current
    applied_step_limit = MOVE_STEP if step_limit is None else float(step_limit)
    if np.linalg.norm(error) <= NEAR_WAYPOINT_DISTANCE:
        applied_step_limit = min(applied_step_limit, NEAR_MOVE_STEP)
    return np.clip(error * MOVE_GAIN, -applied_step_limit, applied_step_limit)


def _build_action(env, arm_delta, gripper_action, arm_idx):
    action = np.zeros(env.action_dim)
    if arm_idx == 0:
        action[0:3] = arm_delta
        action[6] = gripper_action
    else:
        action[7:10] = arm_delta
        action[13] = gripper_action
    return action


def get_gripper_width(env, arm_idx=0):
    qpos = env.obs.get(f"robot{arm_idx}_gripper_qpos", np.array([0.0, 0.0]))
    return float(np.abs(qpos[0]) + np.abs(qpos[1]))


def actuate_gripper(env, arm_idx, gripper_action, steps=GRIPPER_ACTUATION_STEPS):
    width_before = get_gripper_width(env, arm_idx)
    action = _build_action(env, np.zeros(3), gripper_action, arm_idx)
    for _ in range(steps):
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False
    if DEBUG_GRIPPER:
        print(
            f"  gripper robot{arm_idx}: action={gripper_action:+.1f}, "
            f"width {width_before:.4f} -> {get_gripper_width(env, arm_idx):.4f}"
        )
    return True


def actuate_both_grippers(
    env,
    robot0_gripper,
    robot1_gripper,
    steps=GRIPPER_ACTUATION_STEPS,
):
    robot0_width_before = get_gripper_width(env, 0)
    robot1_width_before = get_gripper_width(env, 1)
    action = np.zeros(env.action_dim)
    action[6] = robot0_gripper
    action[13] = robot1_gripper
    for _ in range(steps):
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False
    if DEBUG_GRIPPER:
        print(
            f"  gripper robot0: action={robot0_gripper:+.1f}, "
            f"width {robot0_width_before:.4f} -> {get_gripper_width(env, 0):.4f}"
        )
        print(
            f"  gripper robot1: action={robot1_gripper:+.1f}, "
            f"width {robot1_width_before:.4f} -> {get_gripper_width(env, 1):.4f}"
        )
    return True


def get_primary_object_pos(obs):
    """Return the primary object position from robosuite observations."""
    for key in ("pot_pos", "cube_pos", "object_pos"):
        if key in obs:
            return np.array(obs[key][:3], dtype=float)

    object_state = obs.get("object-state")
    if object_state is not None and len(object_state) >= 3:
        return np.array(object_state[:3], dtype=float)

    return None


def get_default_grasp_target(obs, arm_idx=0):
    """Prefer a visible pot handle for grasp tests, then fall back to object center."""
    handle_keys = (
        ("handle0_xpos", "handle_0_xpos", "left_handle_xpos"),
        ("handle1_xpos", "handle_1_xpos", "right_handle_xpos"),
    )
    for key in handle_keys[min(arm_idx, 1)]:
        if key in obs:
            return np.array(obs[key][:3], dtype=float)

    return get_primary_object_pos(obs)


def get_handle_targets(obs):
    """Return handle targets for a two-arm lift test when available."""
    left = None
    right = None
    for key in ("handle0_xpos", "handle_0_xpos", "left_handle_xpos"):
        if key in obs:
            left = np.array(obs[key][:3], dtype=float)
            break
    for key in ("handle1_xpos", "handle_1_xpos", "right_handle_xpos"):
        if key in obs:
            right = np.array(obs[key][:3], dtype=float)
            break

    return left, right


def _wrap_angle(angle):
    """Map an angle to [-pi, pi] for stable yaw error control."""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _quat_xyzw_to_yaw(quat):
    """Return world yaw from a robosuite xyzw quaternion."""
    x, y, z, w = np.array(quat, dtype=float)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _get_eef_yaw(env, arm_idx):
    key = f"robot{arm_idx}_eef_quat"
    if key not in env.obs:
        return None
    return _quat_xyzw_to_yaw(env.obs[key])


def get_handle_axis_yaw(obs, left_handle_pos=None, right_handle_pos=None):
    """Estimate the current pot handle bar yaw in the world XY plane."""
    if left_handle_pos is None or right_handle_pos is None:
        left_handle_pos, right_handle_pos = get_handle_targets(obs)
    if left_handle_pos is None or right_handle_pos is None:
        if "pot_quat" in obs:
            return _quat_xyzw_to_yaw(obs["pot_quat"])
        return None

    span = np.array(right_handle_pos[:2], dtype=float) - np.array(left_handle_pos[:2], dtype=float)
    if np.linalg.norm(span) < 1e-6:
        return None
    return _wrap_angle(np.arctan2(span[1], span[0]) + DUAL_WRIST_HANDLE_YAW_OFFSET)


def _build_dual_wrist_yaw_deltas(env, handle_yaw):
    """Compute per-step wrist yaw deltas that align grippers with the pot handles."""
    if handle_yaw is None:
        return None, None

    deltas = []
    for arm_idx in (0, 1):
        current_yaw = _get_eef_yaw(env, arm_idx)
        if current_yaw is None:
            deltas.append(None)
            continue
        yaw_error = _wrap_angle(handle_yaw - current_yaw)
        yaw_action = np.clip(
            yaw_error * DUAL_WRIST_YAW_GAIN,
            -DUAL_WRIST_YAW_MAX_ACTION,
            DUAL_WRIST_YAW_MAX_ACTION,
        )
        deltas.append([0.0, 0.0, float(yaw_action)])
    return deltas[0], deltas[1]


def _hold_dual_wrist_yaw(env, handle_yaw, robot0_gripper, robot1_gripper, steps=12):
    """Rotate wrists in place after Cartesian convergence so yaw commands are not skipped."""
    if handle_yaw is None or getattr(env, "is_episode_terminated", lambda: False)():
        return not getattr(env, "is_episode_terminated", lambda: False)()

    for _ in range(steps):
        live_handle_yaw = get_handle_axis_yaw(env.obs)
        if live_handle_yaw is not None:
            handle_yaw = live_handle_yaw
        robot0_rot_delta, robot1_rot_delta = _build_dual_wrist_yaw_deltas(env, handle_yaw)
        action = np.zeros(env.action_dim)
        if robot0_rot_delta is not None:
            action[3:6] = np.array(robot0_rot_delta, dtype=float)
        action[6] = robot0_gripper
        if robot1_rot_delta is not None:
            action[10:13] = np.array(robot1_rot_delta, dtype=float)
        action[13] = robot1_gripper
        env.obs, _, done, _ = env.step(action)
        if done:
            return False
    return True


def get_dual_arm_alignment_errors(env, robot0_waypoint, robot1_waypoint):
    """Return per-arm Cartesian errors for dual-arm waypoint tracking diagnostics."""
    robot0_error = np.array(robot0_waypoint, dtype=float) - np.array(
        env.obs["robot0_eef_pos"], dtype=float
    )
    robot1_error = np.array(robot1_waypoint, dtype=float) - np.array(
        env.obs["robot1_eef_pos"], dtype=float
    )
    return {
        "robot0": [float(value) for value in robot0_error.tolist()],
        "robot1": [float(value) for value in robot1_error.tolist()],
        "robot0_norm": float(np.linalg.norm(robot0_error)),
        "robot1_norm": float(np.linalg.norm(robot1_error)),
    }


def _start_dual_arm_attempt(diagnostics):
    """Create one attempt record inside a dual-arm execution diagnostics payload."""
    if diagnostics is None:
        return None

    attempts = diagnostics.setdefault("attempts", [])
    attempt = {
        "attempt_index": len(attempts) + 1,
        "stages": [],
        "failed_stage": None,
        "gripper_width_after_close": None,
        "lift_delta_z": None,
        "success": False,
    }
    attempts.append(attempt)
    diagnostics["latest_attempt"] = attempt
    return attempt


def _record_dual_arm_stage(attempt, stage_name, env, robot0_waypoint, robot1_waypoint, success):
    """Append one stage result with final Cartesian waypoint error diagnostics."""
    if attempt is None:
        return

    stage_result = {
        "stage": stage_name,
        "success": bool(success),
        "alignment_errors": get_dual_arm_alignment_errors(
            env,
            robot0_waypoint,
            robot1_waypoint,
        ),
    }
    attempt["stages"].append(stage_result)
    if not success and attempt["failed_stage"] is None:
        attempt["failed_stage"] = stage_name


def _step_to_waypoint(env, waypoint, arm_idx, gripper_action, max_steps):
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    for _ in range(max_steps):
        if getattr(env, "is_episode_terminated", lambda: False)():
            return False
        current = env.obs[f"robot{arm_idx}_eef_pos"]
        if np.linalg.norm(np.array(waypoint) - current) <= WAYPOINT_TOLERANCE:
            return True

        delta = move_to_waypoint(env, waypoint, arm_idx)
        action = _build_action(env, delta, gripper_action, arm_idx)
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False

    current = env.obs[f"robot{arm_idx}_eef_pos"]
    return np.linalg.norm(np.array(waypoint) - current) <= WAYPOINT_TOLERANCE


def _step_two_arm_waypoints(
    env,
    robot0_waypoint,
    robot1_waypoint,
    robot0_gripper,
    robot1_gripper,
    max_steps,
    robot0_rot_delta=None,
    robot1_rot_delta=None,
    handle_yaw=None,
    follow_handle_yaw=False,
):
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    for _ in range(max_steps):
        if getattr(env, "is_episode_terminated", lambda: False)():
            return False
        robot0_current = env.obs["robot0_eef_pos"]
        robot1_current = env.obs["robot1_eef_pos"]
        robot0_error = np.linalg.norm(np.array(robot0_waypoint) - robot0_current)
        robot1_error = np.linalg.norm(np.array(robot1_waypoint) - robot1_current)
        if robot0_error <= WAYPOINT_TOLERANCE and robot1_error <= WAYPOINT_TOLERANCE:
            if handle_yaw is not None:
                return _hold_dual_wrist_yaw(env, handle_yaw, robot0_gripper, robot1_gripper)
            return True

        action = np.zeros(env.action_dim)
        action[0:3] = move_to_waypoint(env, robot0_waypoint, arm_idx=0)
        if follow_handle_yaw:
            live_handle_yaw = get_handle_axis_yaw(env.obs)
            if live_handle_yaw is not None:
                handle_yaw = live_handle_yaw
        if handle_yaw is not None:
            robot0_rot_delta, robot1_rot_delta = _build_dual_wrist_yaw_deltas(env, handle_yaw)
        if robot0_rot_delta is not None:
            action[3:6] = np.array(robot0_rot_delta, dtype=float)
        action[6] = robot0_gripper
        action[7:10] = move_to_waypoint(env, robot1_waypoint, arm_idx=1)
        if robot1_rot_delta is not None:
            action[10:13] = np.array(robot1_rot_delta, dtype=float)
        action[13] = robot1_gripper
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False

    robot0_error = np.linalg.norm(np.array(robot0_waypoint) - env.obs["robot0_eef_pos"])
    robot1_error = np.linalg.norm(np.array(robot1_waypoint) - env.obs["robot1_eef_pos"])
    return robot0_error <= WAYPOINT_TOLERANCE and robot1_error <= WAYPOINT_TOLERANCE


def _step_dual_arm_waypoint_with_compensation(
    env,
    robot0_waypoint,
    robot1_waypoint,
    robot0_gripper,
    robot1_gripper,
    max_steps,
    compensation_step=PRE_GRASP_SINGLE_ARM_STEP,
    robot0_rot_delta=None,
    robot1_rot_delta=None,
    handle_yaw=None,
    follow_handle_yaw=False,
):
    """Advance both arms together, then bias the slower arm if one side lags."""
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    for _ in range(max_steps):
        if getattr(env, "is_episode_terminated", lambda: False)():
            return False
        robot0_current = env.obs["robot0_eef_pos"]
        robot1_current = env.obs["robot1_eef_pos"]
        robot0_error_vec = np.array(robot0_waypoint) - robot0_current
        robot1_error_vec = np.array(robot1_waypoint) - robot1_current
        robot0_error = np.linalg.norm(robot0_error_vec)
        robot1_error = np.linalg.norm(robot1_error_vec)
        if robot0_error <= WAYPOINT_TOLERANCE and robot1_error <= WAYPOINT_TOLERANCE:
            if handle_yaw is not None:
                return _hold_dual_wrist_yaw(env, handle_yaw, robot0_gripper, robot1_gripper)
            return True

        action = np.zeros(env.action_dim)
        action[0:3] = move_to_waypoint_limited(env, robot0_waypoint, arm_idx=0)
        if follow_handle_yaw:
            live_handle_yaw = get_handle_axis_yaw(env.obs)
            if live_handle_yaw is not None:
                handle_yaw = live_handle_yaw
        if handle_yaw is not None:
            robot0_rot_delta, robot1_rot_delta = _build_dual_wrist_yaw_deltas(env, handle_yaw)
        if robot0_rot_delta is not None:
            action[3:6] = np.array(robot0_rot_delta, dtype=float)
        action[6] = robot0_gripper
        action[7:10] = move_to_waypoint_limited(env, robot1_waypoint, arm_idx=1)
        if robot1_rot_delta is not None:
            action[10:13] = np.array(robot1_rot_delta, dtype=float)
        action[13] = robot1_gripper

        if robot0_error > robot1_error + WAYPOINT_TOLERANCE:
            action[7:10] *= 0.5
        elif robot1_error > robot0_error + WAYPOINT_TOLERANCE:
            action[0:3] *= 0.5

        if robot0_error > NEAR_WAYPOINT_DISTANCE:
            action[0:3] = np.clip(action[0:3], -compensation_step, compensation_step)
        if robot1_error > NEAR_WAYPOINT_DISTANCE:
            action[7:10] = np.clip(action[7:10], -compensation_step, compensation_step)

        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False

    robot0_error = np.linalg.norm(np.array(robot0_waypoint) - env.obs["robot0_eef_pos"])
    robot1_error = np.linalg.norm(np.array(robot1_waypoint) - env.obs["robot1_eef_pos"])
    return robot0_error <= WAYPOINT_TOLERANCE and robot1_error <= WAYPOINT_TOLERANCE


def _wait_with_grippers(env, robot0_gripper, robot1_gripper, steps=POST_GRIPPER_SETTLE_STEPS):
    """Let the scene settle while maintaining a dual-arm gripper command."""
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    action = np.zeros(env.action_dim)
    action[6] = robot0_gripper
    action[13] = robot1_gripper
    for _ in range(steps):
        if getattr(env, "is_episode_terminated", lambda: False)():
            return False
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False
    return True


def _compute_dual_grasp_stages(
    left_handle_pos,
    right_handle_pos,
    targets_are_grasp_points=False,
):
    """Build a staged dual-arm approach path to reduce direct descents onto the target."""
    if targets_are_grasp_points:
        left_grasp = np.array(left_handle_pos, dtype=float)
        right_grasp = np.array(right_handle_pos, dtype=float)
    else:
        left_grasp = np.array(left_handle_pos, dtype=float) + np.array(
            [0.0, 0.0, HANDLE_GRASP_Z_OFFSET]
        )
        right_grasp = np.array(right_handle_pos, dtype=float) + np.array(
            [0.0, 0.0, HANDLE_GRASP_Z_OFFSET]
        )

    left_pre = left_grasp + np.array([0.0, 0.0, DUAL_PRE_GRASP_HEIGHT_OFFSET])
    right_pre = right_grasp + np.array([0.0, 0.0, DUAL_PRE_GRASP_HEIGHT_OFFSET])
    left_transit = left_grasp + np.array([0.0, 0.0, DUAL_TRANSIT_HEIGHT_OFFSET])
    right_transit = right_grasp + np.array([0.0, 0.0, DUAL_TRANSIT_HEIGHT_OFFSET])
    left_align = left_grasp + np.array([0.0, 0.0, DUAL_ALIGN_HEIGHT_OFFSET])
    right_align = right_grasp + np.array([0.0, 0.0, DUAL_ALIGN_HEIGHT_OFFSET])
    left_final = left_grasp + np.array([0.0, 0.0, DUAL_FINAL_DESCENT_OFFSET])
    right_final = right_grasp + np.array([0.0, 0.0, DUAL_FINAL_DESCENT_OFFSET])
    left_lift = left_grasp + np.array([0.0, 0.0, DUAL_LIFT_HEIGHT])
    right_lift = right_grasp + np.array([0.0, 0.0, DUAL_LIFT_HEIGHT])

    return {
        "left_grasp": left_grasp,
        "right_grasp": right_grasp,
        "left_pre": left_pre,
        "right_pre": right_pre,
        "left_transit": left_transit,
        "right_transit": right_transit,
        "left_align": left_align,
        "right_align": right_align,
        "left_final": left_final,
        "right_final": right_final,
        "left_lift": left_lift,
        "right_lift": right_lift,
    }


def _dual_final_descent_close_enough(env, left_grasp, right_grasp):
    """Accept final descent when XY is aligned and contact prevents lower motion."""
    robot0_pos = np.array(env.obs["robot0_eef_pos"], dtype=float)
    robot1_pos = np.array(env.obs["robot1_eef_pos"], dtype=float)
    left_grasp = np.array(left_grasp, dtype=float)
    right_grasp = np.array(right_grasp, dtype=float)

    robot0_xy_error = np.linalg.norm(robot0_pos[:2] - left_grasp[:2])
    robot1_xy_error = np.linalg.norm(robot1_pos[:2] - right_grasp[:2])
    robot0_z_above = robot0_pos[2] - left_grasp[2]
    robot1_z_above = robot1_pos[2] - right_grasp[2]

    return (
        robot0_xy_error <= DUAL_FINAL_DESCENT_XY_TOLERANCE
        and robot1_xy_error <= DUAL_FINAL_DESCENT_XY_TOLERANCE
        and 0.0 <= robot0_z_above <= DUAL_FINAL_DESCENT_MAX_Z_ABOVE_TARGET
        and 0.0 <= robot1_z_above <= DUAL_FINAL_DESCENT_MAX_Z_ABOVE_TARGET
    )


def _step_dual_arm_transfer_segments(
    env,
    left_start,
    right_start,
    left_end,
    right_end,
    attempt,
    stage_prefix,
    handle_yaw=None,
):
    """Move a grasped object in short synchronized segments to reduce tearing."""
    left_start = np.array(left_start, dtype=float)
    right_start = np.array(right_start, dtype=float)
    left_end = np.array(left_end, dtype=float)
    right_end = np.array(right_end, dtype=float)
    max_distance = max(
        np.linalg.norm(left_end - left_start),
        np.linalg.norm(right_end - right_start),
    )
    segment_count = max(1, int(np.ceil(max_distance / DUAL_TRANSFER_SEGMENT_LENGTH)))

    for segment_index in range(1, segment_count + 1):
        ratio = segment_index / segment_count
        left_waypoint = left_start + (left_end - left_start) * ratio
        right_waypoint = right_start + (right_end - right_start) * ratio
        stage_ok = _step_dual_arm_waypoint_with_compensation(
            env,
            left_waypoint,
            right_waypoint,
            GRIPPER_CLOSED,
            GRIPPER_CLOSED,
            DUAL_TRANSFER_SEGMENT_STEPS,
            handle_yaw=handle_yaw,
            follow_handle_yaw=True,
        )
        if not stage_ok:
            left_error = np.linalg.norm(left_waypoint - np.array(env.obs["robot0_eef_pos"], dtype=float))
            right_error = np.linalg.norm(right_waypoint - np.array(env.obs["robot1_eef_pos"], dtype=float))
            stage_ok = left_error <= DUAL_TRANSFER_TOLERANCE and right_error <= DUAL_TRANSFER_TOLERANCE
        if not stage_ok and _dual_arm_unsnag_wiggle(env):
            stage_ok = _step_dual_arm_waypoint_with_compensation(
                env,
                left_waypoint,
                right_waypoint,
                GRIPPER_CLOSED,
                GRIPPER_CLOSED,
                DUAL_TRANSFER_SEGMENT_STEPS,
                handle_yaw=handle_yaw,
                follow_handle_yaw=True,
            )
            if not stage_ok:
                left_error = np.linalg.norm(left_waypoint - np.array(env.obs["robot0_eef_pos"], dtype=float))
                right_error = np.linalg.norm(right_waypoint - np.array(env.obs["robot1_eef_pos"], dtype=float))
                stage_ok = left_error <= DUAL_TRANSFER_TOLERANCE and right_error <= DUAL_TRANSFER_TOLERANCE
        stage_name = f"{stage_prefix}_{segment_index:02d}"
        _record_dual_arm_stage(attempt, stage_name, env, left_waypoint, right_waypoint, stage_ok)
        object_pos = get_primary_object_pos(env.obs)
        if object_pos is None or object_pos[2] < DUAL_TRANSFER_MIN_HEIGHT:
            if attempt is not None and attempt["failed_stage"] is None:
                attempt["failed_stage"] = "transfer_slip"
            return False
        if not stage_ok:
            return False
    return True


def _dual_arm_unsnag_wiggle(env, robot0_gripper=GRIPPER_CLOSED, robot1_gripper=GRIPPER_CLOSED):
    """Apply small opposite wrist yaw motions to release handle torsion."""
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    for sign in (1.0, -1.0):
        action = np.zeros(env.action_dim)
        action[5] = sign * DUAL_UNSNAG_YAW_BIAS
        action[6] = robot0_gripper
        action[12] = -sign * DUAL_UNSNAG_YAW_BIAS
        action[13] = robot1_gripper
        for _ in range(DUAL_UNSNAG_STEPS):
            env.obs, _, done, _ = env.step(action)
            if done:
                return False
    return True


def release_dual_grasp_with_clearance(
    env,
    left_place_pos,
    right_place_pos,
    object_pos=None,
    handle_yaw=None,
):
    """Set the object near the table, then open and retract to avoid handle snagging."""
    if not _step_two_arm_waypoints(
        env,
        left_place_pos,
        right_place_pos,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        DUAL_RELEASE_RETRACT_STEPS,
        handle_yaw=handle_yaw,
        follow_handle_yaw=True,
    ):
        return False

    if not _wait_with_grippers(
        env,
        GRIPPER_RELEASE_PARTIAL,
        GRIPPER_RELEASE_PARTIAL,
        steps=DUAL_RELEASE_PARTIAL_STEPS,
    ):
        return False

    left_place_pos = np.array(left_place_pos, dtype=float)
    right_place_pos = np.array(right_place_pos, dtype=float)
    if object_pos is None:
        object_pos = get_primary_object_pos(env.obs)
    if object_pos is None:
        object_xy = (left_place_pos[:2] + right_place_pos[:2]) / 2.0
    else:
        object_xy = np.array(object_pos[:2], dtype=float)

    def retract_waypoint(place_pos):
        direction = np.array(place_pos[:2], dtype=float) - object_xy
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = direction / norm
        waypoint = np.array(place_pos, dtype=float)
        waypoint[:2] += direction * DUAL_RELEASE_RETRACT_DISTANCE
        waypoint[2] += DUAL_RELEASE_LIFT_CLEARANCE
        return waypoint

    left_clear = retract_waypoint(left_place_pos)
    right_clear = retract_waypoint(right_place_pos)
    if not _step_two_arm_waypoints(
        env,
        left_clear,
        right_clear,
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        DUAL_RELEASE_RETRACT_STEPS,
        handle_yaw=handle_yaw,
        follow_handle_yaw=True,
    ):
        if not _dual_arm_unsnag_wiggle(env, GRIPPER_OPEN, GRIPPER_OPEN):
            return False
        if not _step_dual_arm_waypoint_with_compensation(
            env,
            left_clear,
            right_clear,
            GRIPPER_OPEN,
            GRIPPER_OPEN,
            DUAL_RELEASE_RETRACT_STEPS,
            handle_yaw=handle_yaw,
            follow_handle_yaw=True,
        ):
            return False

    return actuate_both_grippers(env, GRIPPER_OPEN, GRIPPER_OPEN)


def is_object_near_place(env, place_pos):
    object_pos = get_primary_object_pos(env.obs)
    if object_pos is None:
        return False

    place_pos = np.array(place_pos, dtype=float)
    return np.linalg.norm(object_pos[:2] - place_pos[:2]) <= OBJECT_PLACE_TOLERANCE


def wait_for_object_settle(env, steps=PLACE_SETTLE_STEPS):
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    action = np.zeros(env.action_dim)
    for _ in range(steps):
        if getattr(env, "is_episode_terminated", lambda: False)():
            return False
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False
    return True


def check_object_at_place(env, place_pos):
    if not wait_for_object_settle(env):
        return False
    return is_object_near_place(env, place_pos)


def home_arms(env):
    home_positions = getattr(env, "home_eef_positions", None)
    if not home_positions:
        return False

    reached_home = _step_two_arm_waypoints(
        env,
        home_positions["robot0"],
        home_positions["robot1"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        180,
    )
    return reached_home and actuate_both_grippers(env, GRIPPER_OPEN, GRIPPER_OPEN)


def execute_grasp(env, target_pos, arm_idx=0):
    pre_grasp, grasp, lift = compute_grasp_waypoints(target_pos)
    object_pos_before = get_primary_object_pos(env.obs)

    if not _step_to_waypoint(env, pre_grasp, arm_idx, GRIPPER_OPEN, 120):
        return False
    if not _step_to_waypoint(env, grasp, arm_idx, GRIPPER_OPEN, 120):
        return False

    if not actuate_gripper(env, arm_idx, GRIPPER_CLOSED):
        return False

    if not _step_to_waypoint(env, lift, arm_idx, GRIPPER_CLOSED, 120):
        return False

    return verify_grasp(env, arm_idx, object_pos_before)


def execute_single_grasp_transfer(env, target_pos, place_pos, arm_idx=0):
    for _ in range(MAX_GRASP_ATTEMPTS):
        if check_object_at_place(env, place_pos):
            return True

        object_pos_before = get_primary_object_pos(env.obs)
        if object_pos_before is None:
            return False

        pre_grasp, grasp, lift = compute_grasp_waypoints(target_pos)
        if not _step_to_waypoint(env, pre_grasp, arm_idx, GRIPPER_OPEN, 120):
            return False
        if not _step_to_waypoint(env, grasp, arm_idx, GRIPPER_OPEN, 120):
            return False

        if not actuate_gripper(env, arm_idx, GRIPPER_CLOSED):
            return False

        if not _step_to_waypoint(env, lift, arm_idx, GRIPPER_CLOSED, 140):
            return False

        object_pos_lifted = get_primary_object_pos(env.obs)
        if object_pos_lifted is None:
            return False
        if object_pos_lifted[2] - object_pos_before[2] < OBJECT_LIFT_THRESHOLD:
            if not actuate_gripper(env, arm_idx, GRIPPER_OPEN):
                return False
            continue

        place_pos = np.array(place_pos, dtype=float)
        transfer_delta = np.array(
            [
                place_pos[0] - object_pos_lifted[0],
                place_pos[1] - object_pos_lifted[1],
                0.0,
            ]
        )
        place_high = np.array(env.obs[f"robot{arm_idx}_eef_pos"]) + transfer_delta
        place_low = place_high - np.array([0.0, 0.0, 0.08])

        if not _step_to_waypoint(env, place_high, arm_idx, GRIPPER_CLOSED, 180):
            return False
        if not _step_to_waypoint(env, place_low, arm_idx, GRIPPER_CLOSED, 100):
            return False

        if not actuate_gripper(env, arm_idx, GRIPPER_OPEN):
            return False

        object_pos_after = get_primary_object_pos(env.obs)
        if object_pos_after is None:
            return False
        if check_object_at_place(env, place_pos):
            return True

        target_pos = np.array(target_pos, dtype=float)
        target_pos[:2] += np.array(place_pos[:2]) - object_pos_after[:2]

    return False


def execute_dual_handle_lift(
    env,
    left_handle_pos,
    right_handle_pos,
    diagnostics=None,
    targets_are_grasp_points=False,
):
    object_pos_before = get_primary_object_pos(env.obs)
    stages = _compute_dual_grasp_stages(
        left_handle_pos,
        right_handle_pos,
        targets_are_grasp_points=targets_are_grasp_points,
    )
    attempt = _start_dual_arm_attempt(diagnostics)
    handle_yaw = get_handle_axis_yaw(env.obs, left_handle_pos, right_handle_pos)
    if attempt is not None and handle_yaw is not None:
        attempt["handle_yaw_target_rad"] = float(handle_yaw)
    stage_ok = _step_dual_arm_waypoint_with_compensation(
        env,
        stages["left_transit"],
        stages["right_transit"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        DUAL_TRANSIT_MAX_STEPS,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "transit", env, stages["left_transit"], stages["right_transit"], stage_ok
    )
    if not stage_ok:
        return False

    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_pre"],
        stages["right_pre"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        180,
        handle_yaw=handle_yaw,
        follow_handle_yaw=True,
    )
    _record_dual_arm_stage(
        attempt, "pre_grasp", env, stages["left_pre"], stages["right_pre"], stage_ok
    )
    if not stage_ok:
        return False
    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_align"],
        stages["right_align"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        180,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "align", env, stages["left_align"], stages["right_align"], stage_ok
    )
    if not stage_ok:
        return False
    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_final"],
        stages["right_final"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        140,
        handle_yaw=handle_yaw,
    )
    final_descent_contact_ready = False
    if not stage_ok and _dual_final_descent_close_enough(
        env,
        stages["left_grasp"],
        stages["right_grasp"],
    ):
        stage_ok = True
        final_descent_contact_ready = True
    _record_dual_arm_stage(
        attempt, "final_descent", env, stages["left_final"], stages["right_final"], stage_ok
    )
    if not stage_ok:
        return False
    stage_ok = final_descent_contact_ready or _step_two_arm_waypoints(
        env,
        stages["left_grasp"],
        stages["right_grasp"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        140,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "grasp_pose", env, stages["left_grasp"], stages["right_grasp"], stage_ok
    )
    if not stage_ok:
        return False

    if not actuate_both_grippers(env, GRIPPER_CLOSED, GRIPPER_CLOSED):
        if attempt is not None and attempt["failed_stage"] is None:
            attempt["failed_stage"] = "close_grippers"
        return False
    if attempt is not None:
        attempt["gripper_width_after_close"] = {
            "robot0": get_gripper_width(env, 0),
            "robot1": get_gripper_width(env, 1),
        }
    if not _wait_with_grippers(env, GRIPPER_CLOSED, GRIPPER_CLOSED):
        if attempt is not None and attempt["failed_stage"] is None:
            attempt["failed_stage"] = "post_close_settle"
        return False

    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_lift"],
        stages["right_lift"],
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        180,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "lift", env, stages["left_lift"], stages["right_lift"], stage_ok
    )
    if not stage_ok:
        return False

    if attempt is not None and object_pos_before is not None:
        object_pos_after = get_primary_object_pos(env.obs)
        if object_pos_after is not None:
            attempt["lift_delta_z"] = float(object_pos_after[2] - object_pos_before[2])

    grasp_ok = verify_grasp(env, object_pos_before=object_pos_before)
    if attempt is not None:
        attempt["success"] = bool(grasp_ok)
        if not grasp_ok and attempt["failed_stage"] is None:
            attempt["failed_stage"] = "verify_grasp"
    return grasp_ok


def execute_dual_handle_transfer(
    env,
    left_handle_pos,
    right_handle_pos,
    place_pos,
    remaining_attempts=MAX_GRASP_ATTEMPTS,
    diagnostics=None,
    targets_are_grasp_points=False,
):
    if getattr(env, "is_episode_terminated", lambda: False)():
        return False

    if check_object_at_place(env, place_pos):
        if diagnostics is not None:
            diagnostics["already_at_place"] = True
        return True

    object_pos_before = get_primary_object_pos(env.obs)
    if object_pos_before is None:
        return False

    stages = _compute_dual_grasp_stages(
        left_handle_pos,
        right_handle_pos,
        targets_are_grasp_points=targets_are_grasp_points,
    )
    attempt = _start_dual_arm_attempt(diagnostics)
    handle_yaw = get_handle_axis_yaw(env.obs, left_handle_pos, right_handle_pos)
    if attempt is not None and handle_yaw is not None:
        attempt["handle_yaw_target_rad"] = float(handle_yaw)

    stage_ok = _step_dual_arm_waypoint_with_compensation(
        env,
        stages["left_transit"],
        stages["right_transit"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        DUAL_TRANSIT_MAX_STEPS,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "transit", env, stages["left_transit"], stages["right_transit"], stage_ok
    )
    if not stage_ok:
        return False

    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_pre"],
        stages["right_pre"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        180,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "pre_grasp", env, stages["left_pre"], stages["right_pre"], stage_ok
    )
    if not stage_ok:
        return False
    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_align"],
        stages["right_align"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        180,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "align", env, stages["left_align"], stages["right_align"], stage_ok
    )
    if not stage_ok:
        if remaining_attempts <= 1:
            return False

        next_left_handle, next_right_handle = get_handle_targets(env.obs)
        if next_left_handle is None or next_right_handle is None:
            return False
        return execute_dual_handle_transfer(
            env,
            next_left_handle,
            next_right_handle,
            place_pos,
            remaining_attempts=remaining_attempts - 1,
            diagnostics=diagnostics,
            targets_are_grasp_points=False,
        )
    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_final"],
        stages["right_final"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        140,
        handle_yaw=handle_yaw,
    )
    final_descent_contact_ready = False
    if not stage_ok and _dual_final_descent_close_enough(
        env,
        stages["left_grasp"],
        stages["right_grasp"],
    ):
        stage_ok = True
        final_descent_contact_ready = True
    _record_dual_arm_stage(
        attempt, "final_descent", env, stages["left_final"], stages["right_final"], stage_ok
    )
    if not stage_ok:
        if remaining_attempts <= 1:
            return False

        next_left_handle, next_right_handle = get_handle_targets(env.obs)
        if next_left_handle is None or next_right_handle is None:
            return False
        return execute_dual_handle_transfer(
            env,
            next_left_handle,
            next_right_handle,
            place_pos,
            remaining_attempts=remaining_attempts - 1,
            diagnostics=diagnostics,
            targets_are_grasp_points=False,
        )
    stage_ok = final_descent_contact_ready or _step_two_arm_waypoints(
        env,
        stages["left_grasp"],
        stages["right_grasp"],
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        140,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "grasp_pose", env, stages["left_grasp"], stages["right_grasp"], stage_ok
    )
    if not stage_ok:
        if remaining_attempts <= 1:
            return False

        next_left_handle, next_right_handle = get_handle_targets(env.obs)
        if next_left_handle is None or next_right_handle is None:
            return False
        return execute_dual_handle_transfer(
            env,
            next_left_handle,
            next_right_handle,
            place_pos,
            remaining_attempts=remaining_attempts - 1,
            diagnostics=diagnostics,
            targets_are_grasp_points=False,
        )

    if not actuate_both_grippers(env, GRIPPER_CLOSED, GRIPPER_CLOSED):
        if attempt is not None and attempt["failed_stage"] is None:
            attempt["failed_stage"] = "close_grippers"
        return False
    if attempt is not None:
        attempt["gripper_width_after_close"] = {
            "robot0": get_gripper_width(env, 0),
            "robot1": get_gripper_width(env, 1),
        }
    if not _wait_with_grippers(env, GRIPPER_CLOSED, GRIPPER_CLOSED):
        if attempt is not None and attempt["failed_stage"] is None:
            attempt["failed_stage"] = "post_close_settle"
        return False

    stage_ok = _step_two_arm_waypoints(
        env,
        stages["left_lift"],
        stages["right_lift"],
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        180,
        handle_yaw=handle_yaw,
    )
    _record_dual_arm_stage(
        attempt, "lift", env, stages["left_lift"], stages["right_lift"], stage_ok
    )
    if not stage_ok:
        return False

    object_pos_lifted = get_primary_object_pos(env.obs)
    if object_pos_lifted is None:
        return False
    if attempt is not None:
        attempt["lift_delta_z"] = float(object_pos_lifted[2] - object_pos_before[2])
    if object_pos_lifted[2] - object_pos_before[2] < OBJECT_LIFT_THRESHOLD:
        if not actuate_both_grippers(env, GRIPPER_OPEN, GRIPPER_OPEN):
            return False
        if attempt is not None:
            attempt["failed_stage"] = attempt["failed_stage"] or "verify_lift"
            attempt["success"] = False
        if remaining_attempts <= 1:
            return False

        next_left_handle, next_right_handle = get_handle_targets(env.obs)
        if next_left_handle is None or next_right_handle is None:
            return False
        return execute_dual_handle_transfer(
            env,
            next_left_handle,
            next_right_handle,
            place_pos,
            remaining_attempts=remaining_attempts - 1,
            diagnostics=diagnostics,
            targets_are_grasp_points=False,
        )

    place_pos = np.array(place_pos, dtype=float)
    transfer_delta = np.array(
        [
            place_pos[0] - object_pos_lifted[0],
            place_pos[1] - object_pos_lifted[1],
            0.0,
        ]
    )
    left_place_high = stages["left_lift"] + transfer_delta
    right_place_high = stages["right_lift"] + transfer_delta

    stage_ok = _step_dual_arm_transfer_segments(
        env,
        stages["left_lift"],
        stages["right_lift"],
        left_place_high,
        right_place_high,
        attempt,
        "transfer_high",
        handle_yaw=handle_yaw,
    )
    if not stage_ok:
        return False

    object_pos_carried = get_primary_object_pos(env.obs)
    if object_pos_carried is None:
        return False
    placement_drop = max(
        0.0,
        object_pos_carried[2] - place_pos[2] - DUAL_RELEASE_SURFACE_CLEARANCE,
    )
    if placement_drop > 0.0:
        left_place_low = left_place_high - np.array([0.0, 0.0, placement_drop])
        right_place_low = right_place_high - np.array([0.0, 0.0, placement_drop])
    else:
        left_place_low = left_place_high.copy()
        right_place_low = right_place_high.copy()
    stage_ok = _step_two_arm_waypoints(
        env,
        left_place_low,
        right_place_low,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        160,
        handle_yaw=handle_yaw,
        follow_handle_yaw=True,
    )
    _record_dual_arm_stage(attempt, "transfer_low", env, left_place_low, right_place_low, stage_ok)
    if not stage_ok:
        return False

    if not release_dual_grasp_with_clearance(
        env,
        left_place_low,
        right_place_low,
        object_pos=object_pos_lifted,
        handle_yaw=handle_yaw,
    ):
        if attempt is not None and attempt["failed_stage"] is None:
            attempt["failed_stage"] = "release_clearance"
        return False

    object_pos_after = get_primary_object_pos(env.obs)
    if object_pos_after is None:
        return False

    if check_object_at_place(env, place_pos):
        if attempt is not None:
            attempt["success"] = True
        return True
    if attempt is not None and attempt["failed_stage"] is None:
        attempt["failed_stage"] = "verify_place"
    if remaining_attempts <= 1:
        return False

    next_left_handle, next_right_handle = get_handle_targets(env.obs)
    if next_left_handle is None or next_right_handle is None:
        return False
    return execute_dual_handle_transfer(
        env,
        next_left_handle,
        next_right_handle,
        place_pos,
        remaining_attempts=remaining_attempts - 1,
        diagnostics=diagnostics,
        targets_are_grasp_points=False,
    )


def execute_push(env, obstacle_pos, direction, arm_idx=1):
    object_pos_before = get_primary_object_pos(env.obs)
    push_pre, push_contact, push_end = compute_push_waypoints(obstacle_pos, direction)
    print(f"  Push waypoints: pre={np.round(push_pre, 4).tolist()}")
    print(f"  Push waypoints: contact={np.round(push_contact, 4).tolist()}")
    print(f"  Push waypoints: end={np.round(push_end, 4).tolist()}")

    def pushed_enough():
        object_pos_after = get_primary_object_pos(env.obs)
        if object_pos_before is None or object_pos_after is None:
            return False
        displacement_xy = np.linalg.norm(object_pos_after[:2] - object_pos_before[:2])
        return displacement_xy >= OBJECT_MOVE_THRESHOLD

    if not actuate_gripper(env, arm_idx, GRIPPER_OPEN):
        return pushed_enough()

    if not _step_to_waypoint(env, push_pre, arm_idx, GRIPPER_OPEN, 100):
        return pushed_enough()

    if not _step_to_waypoint(env, push_contact, arm_idx, GRIPPER_OPEN, 100):
        return pushed_enough()

    if not actuate_gripper(env, arm_idx, GRIPPER_CLOSED, steps=PUSH_SETTLE_STEPS):
        return pushed_enough()

    reached_push_end = _step_to_waypoint(env, push_end, arm_idx, GRIPPER_CLOSED, 160)
    return reached_push_end or pushed_enough()


def verify_grasp(env, arm_idx=0, object_pos_before=None):
    object_pos_after = get_primary_object_pos(env.obs)
    if object_pos_before is not None and object_pos_after is not None:
        return object_pos_after[2] - object_pos_before[2] >= OBJECT_LIFT_THRESHOLD

    width = get_gripper_width(env, arm_idx)
    return width > GRASP_WIDTH_THRESHOLD
