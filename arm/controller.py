"""Arm controller for grasping and pushing operations."""

import numpy as np


GRASP_HEIGHT_OFFSET = 0.05
HANDLE_GRASP_Z_OFFSET = -0.04
PUSH_HEIGHT = 0.08
MOVE_STEP = 1.0
MOVE_GAIN = 8.0
WAYPOINT_TOLERANCE = 0.025
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0
GRIPPER_ACTUATION_STEPS = 50
DEBUG_GRIPPER = True
GRASP_WIDTH_THRESHOLD = 0.02
OBJECT_MOVE_THRESHOLD = 0.02
OBJECT_LIFT_THRESHOLD = 0.02
OBJECT_PLACE_TOLERANCE = 0.08
SINGLE_LIFT_HEIGHT = 0.20
DUAL_LIFT_HEIGHT = 0.16
MAX_GRASP_ATTEMPTS = 2
MAX_PLACE_ATTEMPTS = 2
PLACE_SETTLE_STEPS = 40
PUSH_APPROACH_DISTANCE = 0.12
PUSH_THROUGH_DISTANCE = 0.20
PUSH_SURFACE_MARGIN = 0.04
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
    direction = np.array(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("Push direction must be non-zero.")
    direction = direction / direction_norm

    contact_pos = np.array(
        [
            obstacle_pos[0] - direction[0] * PUSH_SURFACE_MARGIN,
            obstacle_pos[1] - direction[1] * PUSH_SURFACE_MARGIN,
            obstacle_pos[2],
        ],
        dtype=float,
    )

    push_start = [
        contact_pos[0] - direction[0] * PUSH_APPROACH_DISTANCE,
        contact_pos[1] - direction[1] * PUSH_APPROACH_DISTANCE,
        obstacle_pos[2],
    ]
    push_end = [
        contact_pos[0] + direction[0] * PUSH_THROUGH_DISTANCE,
        contact_pos[1] + direction[1] * PUSH_THROUGH_DISTANCE,
        obstacle_pos[2],
    ]
    return push_start, push_end


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
    action_delta = np.clip(error * MOVE_GAIN, -MOVE_STEP, MOVE_STEP)
    return action_delta


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


def _step_to_waypoint(env, waypoint, arm_idx, gripper_action, max_steps):
    for _ in range(max_steps):
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
):
    for _ in range(max_steps):
        robot0_current = env.obs["robot0_eef_pos"]
        robot1_current = env.obs["robot1_eef_pos"]
        robot0_error = np.linalg.norm(np.array(robot0_waypoint) - robot0_current)
        robot1_error = np.linalg.norm(np.array(robot1_waypoint) - robot1_current)
        if robot0_error <= WAYPOINT_TOLERANCE and robot1_error <= WAYPOINT_TOLERANCE:
            return True

        action = np.zeros(env.action_dim)
        action[0:3] = move_to_waypoint(env, robot0_waypoint, arm_idx=0)
        action[6] = robot0_gripper
        action[7:10] = move_to_waypoint(env, robot1_waypoint, arm_idx=1)
        action[13] = robot1_gripper
        env.obs, _, done, _ = env.step(action)
        if done:
            env.arm_safe_retract()
            return False

    robot0_error = np.linalg.norm(np.array(robot0_waypoint) - env.obs["robot0_eef_pos"])
    robot1_error = np.linalg.norm(np.array(robot1_waypoint) - env.obs["robot1_eef_pos"])
    return robot0_error <= WAYPOINT_TOLERANCE and robot1_error <= WAYPOINT_TOLERANCE


def is_object_near_place(env, place_pos):
    object_pos = get_primary_object_pos(env.obs)
    if object_pos is None:
        return False

    place_pos = np.array(place_pos, dtype=float)
    return np.linalg.norm(object_pos[:2] - place_pos[:2]) <= OBJECT_PLACE_TOLERANCE


def wait_for_object_settle(env, steps=PLACE_SETTLE_STEPS):
    action = np.zeros(env.action_dim)
    for _ in range(steps):
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


def execute_dual_handle_lift(env, left_handle_pos, right_handle_pos):
    object_pos_before = get_primary_object_pos(env.obs)
    left_grasp = np.array(left_handle_pos, dtype=float) + np.array(
        [0.0, 0.0, HANDLE_GRASP_Z_OFFSET]
    )
    right_grasp = np.array(right_handle_pos, dtype=float) + np.array(
        [0.0, 0.0, HANDLE_GRASP_Z_OFFSET]
    )
    left_pre = left_grasp + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    right_pre = right_grasp + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    left_lift = left_grasp + np.array([0.0, 0.0, DUAL_LIFT_HEIGHT])
    right_lift = right_grasp + np.array([0.0, 0.0, DUAL_LIFT_HEIGHT])

    if not _step_two_arm_waypoints(
        env, left_pre, right_pre, GRIPPER_OPEN, GRIPPER_OPEN, 140
    ):
        return False
    if not _step_two_arm_waypoints(
        env, left_grasp, right_grasp, GRIPPER_OPEN, GRIPPER_OPEN, 180
    ):
        return False

    if not actuate_both_grippers(env, GRIPPER_CLOSED, GRIPPER_CLOSED):
        return False

    if not _step_two_arm_waypoints(
        env, left_lift, right_lift, GRIPPER_CLOSED, GRIPPER_CLOSED, 160
    ):
        return False

    return verify_grasp(env, object_pos_before=object_pos_before)


def execute_dual_handle_transfer(
    env,
    left_handle_pos,
    right_handle_pos,
    place_pos,
    remaining_attempts=MAX_GRASP_ATTEMPTS,
):
    if check_object_at_place(env, place_pos):
        return True

    object_pos_before = get_primary_object_pos(env.obs)
    if object_pos_before is None:
        return False

    left_grasp = np.array(left_handle_pos, dtype=float) + np.array(
        [0.0, 0.0, HANDLE_GRASP_Z_OFFSET]
    )
    right_grasp = np.array(right_handle_pos, dtype=float) + np.array(
        [0.0, 0.0, HANDLE_GRASP_Z_OFFSET]
    )
    left_pre = left_grasp + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    right_pre = right_grasp + np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET])
    left_lift = left_grasp + np.array([0.0, 0.0, DUAL_LIFT_HEIGHT])
    right_lift = right_grasp + np.array([0.0, 0.0, DUAL_LIFT_HEIGHT])

    if not _step_two_arm_waypoints(
        env, left_pre, right_pre, GRIPPER_OPEN, GRIPPER_OPEN, 140
    ):
        return False
    if not _step_two_arm_waypoints(
        env, left_grasp, right_grasp, GRIPPER_OPEN, GRIPPER_OPEN, 180
    ):
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
        )

    if not actuate_both_grippers(env, GRIPPER_CLOSED, GRIPPER_CLOSED):
        return False

    if not _step_two_arm_waypoints(
        env, left_lift, right_lift, GRIPPER_CLOSED, GRIPPER_CLOSED, 160
    ):
        return False

    object_pos_lifted = get_primary_object_pos(env.obs)
    if object_pos_lifted is None:
        return False
    if object_pos_lifted[2] - object_pos_before[2] < OBJECT_LIFT_THRESHOLD:
        if not actuate_both_grippers(env, GRIPPER_OPEN, GRIPPER_OPEN):
            return False
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
        )

    place_pos = np.array(place_pos, dtype=float)
    transfer_delta = np.array(
        [
            place_pos[0] - object_pos_lifted[0],
            place_pos[1] - object_pos_lifted[1],
            0.0,
        ]
    )
    left_place_high = left_lift + transfer_delta
    right_place_high = right_lift + transfer_delta
    left_place_low = left_place_high - np.array([0.0, 0.0, DUAL_LIFT_HEIGHT * 0.6])
    right_place_low = right_place_high - np.array([0.0, 0.0, DUAL_LIFT_HEIGHT * 0.6])

    if not _step_two_arm_waypoints(
        env, left_place_high, right_place_high, GRIPPER_CLOSED, GRIPPER_CLOSED, 180
    ):
        return False
    if not _step_two_arm_waypoints(
        env, left_place_low, right_place_low, GRIPPER_CLOSED, GRIPPER_CLOSED, 120
    ):
        return False

    if not actuate_both_grippers(env, GRIPPER_OPEN, GRIPPER_OPEN):
        return False

    object_pos_after = get_primary_object_pos(env.obs)
    if object_pos_after is None:
        return False

    if check_object_at_place(env, place_pos):
        return True
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
    )


def execute_push(env, obstacle_pos, direction, arm_idx=1):
    object_pos_before = get_primary_object_pos(env.obs)
    push_start, push_end = compute_push_waypoints(obstacle_pos, direction)

    if not actuate_gripper(env, arm_idx, GRIPPER_CLOSED):
        return False

    if not _step_to_waypoint(env, push_start, arm_idx, GRIPPER_CLOSED, 120):
        object_pos_after = get_primary_object_pos(env.obs)
        if object_pos_before is None or object_pos_after is None:
            return False
        displacement_xy = np.linalg.norm(object_pos_after[:2] - object_pos_before[:2])
        return displacement_xy >= OBJECT_MOVE_THRESHOLD

    reached_push_end = _step_to_waypoint(env, push_end, arm_idx, GRIPPER_CLOSED, 160)

    object_pos_after = get_primary_object_pos(env.obs)
    if object_pos_before is None or object_pos_after is None:
        return False

    displacement_xy = np.linalg.norm(object_pos_after[:2] - object_pos_before[:2])
    return reached_push_end or displacement_xy >= OBJECT_MOVE_THRESHOLD


def verify_grasp(env, arm_idx=0, object_pos_before=None):
    object_pos_after = get_primary_object_pos(env.obs)
    if object_pos_before is not None and object_pos_after is not None:
        return object_pos_after[2] - object_pos_before[2] >= OBJECT_LIFT_THRESHOLD

    width = get_gripper_width(env, arm_idx)
    return width > GRASP_WIDTH_THRESHOLD
