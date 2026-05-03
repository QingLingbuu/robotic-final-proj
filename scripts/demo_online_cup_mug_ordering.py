"""Run a continuous online CupMugSorting ordering demo in the MuJoCo viewer."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import robocasa  # noqa: F401 - import registers RoboCasa envs into robosuite.make
import robosuite
from termcolor import colored

from arm.base_torso import DEFAULT_BASE_ACTION_SLICE, DEFAULT_TORSO_ACTION_INDEX, calibrate_base_action_mapping
from arm.calibration import calibrate_position_action_mapping
from arm.cup_mug_live_execution import LiveExecutionOptions, run_live_target_execution
from planner.candidates import choose_reachable_candidate
from scripts.demo_robocasa_reach_onscreen import (
    CUP_MUG_SORTING_TASKS,
    DEFAULT_LEFT_SINK_LAYOUT_ID,
    build_env_config,
    cup_mug_sorting_place_anchors,
    cup_mug_sorting_sim_objects,
    get_rgbd,
    load_yaml,
    resolve_camera_config,
)
from rl.cup_mug_live_mapping import build_live_slot_mapping, resolve_live_target_from_slot, resolve_live_target_with_identity_preference
from rl.cup_mug_policies import select_action
from rl.harness import classify_cup_ordering_failure, inspect_cup_ordering_checkpoint_artifact, select_cup_ordering_rl_action
from vision.perception_loop import VisionPerceptionLoop
from vision.sorting_policy import (
    assign_sim_metadata_to_targets,
    build_sim_metadata_fallback_target,
    classify_drinkware_targets,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _append_attempt_log(step_summary, attempt_index, policy_debug, execution_summary, selected_object_id):
    attempt_log = {
        "attempt_index": int(attempt_index),
        "selected_object_id": selected_object_id,
        "policy_debug": policy_debug,
        "execution_summary": execution_summary,
    }
    step_summary.setdefault("attempts", []).append(attempt_log)
    step_summary["execution_summary"] = execution_summary
    step_summary["selected_object_id"] = selected_object_id
    return step_summary


def _ensure_handle_base_mapping(env, current_obs, base_action_mapping, args):
    if base_action_mapping is not None:
        return current_obs, base_action_mapping, None
    current_obs, base_mapping_summary = calibrate_base_action_mapping(
        env,
        obs=current_obs,
        base_slice=(int(args.base_action_start), int(args.base_action_end)),
        pulse_magnitude=0.1,
        pulse_steps=4,
        render_sleep_sec=float(args.render_sleep_sec),
    )
    base_action_mapping = np.asarray(base_mapping_summary["eef_xy_delta_from_base_action"], dtype=float)
    return current_obs, base_action_mapping, base_mapping_summary


def _live_execution_options_for_slot(args, resolved_slot, base_action_mapping):
    has_handle = bool((resolved_slot or {}).get("slot", {}).get("has_handle"))
    use_base_mapping = bool(args.use_base_action_mapping or has_handle)
    enable_preposition = bool(args.enable_base_torso_preposition or has_handle)
    auto_retry = bool(args.auto_preposition_retry or has_handle)
    desired_xy_standoff = 0.0 if has_handle else float(args.base_desired_xy_standoff)
    base_xy_deadband = 0.0 if has_handle else float(args.base_xy_deadband)
    base_action_limit = min(float(args.base_action_limit), 0.2) if has_handle else float(args.base_action_limit)
    return LiveExecutionOptions(
        render_sleep_sec=float(args.render_sleep_sec),
        enable_base_torso_preposition=enable_preposition,
        auto_preposition_retry=auto_retry,
        skip_vision_refresh_after_base_preposition=bool(args.skip_vision_refresh_after_base_preposition),
        retry_final_error_threshold=float(args.retry_final_error_threshold),
        base_torso_steps=int(args.base_torso_steps),
        base_action_start=int(args.base_action_start),
        base_action_end=int(args.base_action_end),
        torso_action_index=None if args.torso_action_index is None else int(args.torso_action_index),
        base_desired_xy_standoff=desired_xy_standoff,
        base_xy_deadband=base_xy_deadband,
        base_preposition_gain=float(args.base_preposition_gain),
        base_action_limit=base_action_limit,
        use_base_action_mapping=use_base_mapping and base_action_mapping is not None,
        base_mapping_trust=float(args.base_mapping_trust),
        base_max_world_delta=float(args.base_max_world_delta),
        use_drinkware_classification=True,
        retry_place_once=True,
    )


def _sorting_metadata_from_sim_objects(sim_objects):
    metadata = {}
    for obj in sim_objects:
        name = obj.get("name")
        has_handle = bool(obj.get("has_handle"))
        if not name:
            continue
        metadata[str(name)] = {
            "has_handle": has_handle,
            "target_zone": "handled" if has_handle else "plain",
            "recommended_grasp": "handle_top_down" if has_handle else "top_down",
        }
    return metadata


def _build_fallback_target_for_sim_object(sim_object):
    has_handle = bool(sim_object.get("has_handle"))
    label = "mug" if has_handle else "cup"
    grasp_type = "handle_top_down" if has_handle else "top_down"
    pos = [float(value) for value in sim_object.get("pos", [0.0, 0.0, 0.0])]
    target = {
        "label": label,
        "conf": 1.0,
        "pos": pos,
        "grasp_candidates": [
            {
                "id": 1,
                "pos": pos,
                "orientation": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                "gripper_width": 0.03 if has_handle else 0.08,
                "score": 1.0,
                "grasp_type": grasp_type,
            }
        ],
        "diagnostics": {"source": "online_sim_metadata_fallback"},
        "sim_object_name": sim_object.get("name"),
        "sim_has_handle": has_handle,
        "sim_metadata_distance_xy": 0.0,
    }
    assignment = {
        "object_index": None,
        "label": label,
        "conf": 1.0,
        "pos": pos,
        "has_handle": has_handle,
        "strategy": grasp_type,
        "arm": "right" if has_handle else "left",
        "sim_object_name": sim_object.get("name"),
        "sim_metadata_distance_xy": 0.0,
        "candidate_id": 1,
        "candidate_score": 1.0,
        "fallback_source": "online_sim_metadata",
    }
    return target, assignment


def _build_fallback_target_for_assignment(sim_objects, assignment):
    selected_name = assignment.get("sim_object_name")
    if selected_name is not None:
        for sim_object in sim_objects:
            if str(sim_object.get("name")) == str(selected_name):
                return _build_fallback_target_for_sim_object(sim_object)
    return build_sim_metadata_fallback_target(sim_objects, requested_label=assignment.get("label"))


def _augment_live_scene_with_sim_fallback(live_scene, finished):
    targets = list(live_scene["classified_targets"])
    assignments = list(live_scene["assignments"])
    present_names = {str(target.get("sim_object_name")) for target in targets if target.get("sim_object_name") is not None}
    added_names = []
    for sim_object in live_scene["sim_objects"]:
        name = sim_object.get("name")
        if name is None:
            continue
        name = str(name)
        if name in present_names or finished.get(name):
            continue
        target, assignment = _build_fallback_target_for_sim_object(sim_object)
        targets.append(target)
        assignments.append(assignment)
        added_names.append(name)
    return {
        **live_scene,
        "classified_targets": targets,
        "assignments": assignments,
        "fallback_added_names": added_names,
    }


def _classify_live_scene(env, obs, args, thresholds_config, vision_config):
    rgb, depth = get_rgbd(obs, camera_name=args.camera_name, sim=env.sim)
    camera_config = resolve_camera_config(env, camera_name=args.camera_name, width=args.width, height=args.height)
    loop = VisionPerceptionLoop(
        vision_config=vision_config,
        camera_config=camera_config,
        conf_thresh=float(thresholds_config["CONF_THRESH"]),
    )
    perception_summary = loop.infer_all_targets_with_diagnostics(rgb, depth)
    sim_objects = cup_mug_sorting_sim_objects(env)
    classified_targets = assign_sim_metadata_to_targets(
        perception_summary.get("targets", []),
        sim_objects,
        unmatched_fallback_max_xy_distance=0.35,
    )
    assignments = classify_drinkware_targets(classified_targets)
    return {
        "perception_summary": perception_summary,
        "sim_objects": sim_objects,
        "classified_targets": classified_targets,
        "assignments": assignments,
        "sorting_metadata": _sorting_metadata_from_sim_objects(sim_objects),
    }


def _build_live_execution_candidate(
    env,
    obs_for_vision,
    assignment,
    args,
    thresholds_config,
    vision_config,
    policy_name="greedy",
    step_index=0,
    attempt_index=0,
):
    rgb, depth = get_rgbd(obs_for_vision, camera_name=args.camera_name, sim=env.sim)
    camera_config = resolve_camera_config(env, camera_name=args.camera_name, width=args.width, height=args.height)
    loop = VisionPerceptionLoop(
        vision_config=vision_config,
        camera_config=camera_config,
        conf_thresh=float(thresholds_config["CONF_THRESH"]),
    )
    all_targets_summary = loop.infer_all_targets_with_diagnostics(rgb, depth)
    sim_objects = cup_mug_sorting_sim_objects(env)
    classified_targets = assign_sim_metadata_to_targets(
        all_targets_summary.get("targets", []),
        sim_objects,
        unmatched_fallback_max_xy_distance=0.35,
    )

    selected_target = None
    selected_assignment = None
    fresh_assignments = classify_drinkware_targets(classified_targets)
    for target in classified_targets:
        if str(target.get("sim_object_name")) == str(assignment.get("sim_object_name")):
            selected_target = target
            break
    if selected_target is not None:
        for fresh_assignment in fresh_assignments:
            if str(fresh_assignment.get("sim_object_name")) == str(assignment.get("sim_object_name")):
                selected_assignment = fresh_assignment
                break
    fallback_used = False
    if selected_target is None:
        selected_target, fallback_assignment = _build_fallback_target_for_assignment(sim_objects, assignment)
        fallback_used = selected_target is not None
        if fallback_used:
            assignment = fallback_assignment
    elif selected_assignment is not None:
        assignment = selected_assignment
    if selected_target is None:
        raise RuntimeError(f"Could not resolve live target for assignment: {assignment}")
    if fallback_used and bool(assignment.get("has_handle")) and str(policy_name) != "rl":
        raise RuntimeError(
            f"Handle target lost live perception for assignment {assignment}; refusing synthetic fallback execution."
        )

    payload = {
        "label": selected_target["label"],
        "conf": selected_target["conf"],
        "pos": selected_target["pos"],
        "grasp_candidates": selected_target.get("grasp_candidates", []),
        "status": "ready",
    }
    diagnostics = {"grasp_candidates": selected_target.get("diagnostics", {})}
    selected_candidate, selected_rejection = choose_reachable_candidate(
        payload,
        candidate_id=None,
        grasp_type=assignment.get("strategy"),
    )
    if selected_candidate is None:
        raise RuntimeError(f"No reachable candidate for assignment {assignment}: {selected_rejection}")

    candidate_score = float(selected_candidate.get("score", 0.0) or 0.0)
    candidate_width = float(selected_candidate.get("gripper_width", 0.0) or 0.0)
    requested_strategy = str(assignment.get("strategy") or "")
    handle_threshold = 0.24 if str(policy_name) == "rl" else 0.30
    if str(policy_name) == "rl" and int(step_index) > 0:
        handle_threshold = 0.20
    top_down_threshold = 0.15 if str(policy_name) == "rl" else 0.18
    later_step_margin_accept = False
    greedy_margin_accept = False
    selected_diag = diagnostics.get("grasp_candidates", {}).get(str(selected_candidate.get("id")), {})
    protrusion_quality = float(selected_diag.get("protrusion_quality", 0.0) or 0.0)
    handle_point_count = int(selected_diag.get("handle_point_count", 0) or 0)
    if (
        str(policy_name) == "rl"
        and requested_strategy == "handle_top_down"
        and int(step_index) > 0
        and not bool(fallback_used)
        and assignment.get("sim_object_name") is not None
        and 0.18 <= candidate_score < float(handle_threshold)
    ):
        later_step_margin_accept = True
    if (
        str(policy_name) == "greedy"
        and requested_strategy == "handle_top_down"
        and not bool(fallback_used)
        and assignment.get("sim_object_name") is not None
        and 0.25 <= candidate_score < float(handle_threshold)
        and handle_point_count >= 24
        and protrusion_quality >= 0.50
        and candidate_width <= 0.04
    ):
        greedy_margin_accept = True
    if requested_strategy == "handle_top_down" and candidate_score < float(handle_threshold) and not later_step_margin_accept and not greedy_margin_accept:
        raise RuntimeError(
            f"Handle candidate below online execution threshold for assignment {assignment}: score={candidate_score:.4f} threshold={handle_threshold:.2f}"
        )
    if requested_strategy == "top_down" and candidate_score < float(top_down_threshold):
        raise RuntimeError(
            f"Top-down candidate below online execution threshold for assignment {assignment}: score={candidate_score:.4f} threshold={top_down_threshold:.2f}"
        )
    if requested_strategy == "handle_top_down" and candidate_width > 0.06:
        raise RuntimeError(
            f"Handle candidate width too large for stable online execution for assignment {assignment}: width={candidate_width:.4f}"
        )
    summary = {
        "reason": "online_ordering_selected_slot",
        "candidate": selected_candidate,
        "payload_status": payload.get("status"),
        "drinkware_classification": {
            "requested_label": assignment.get("label"),
            "sim_objects": sim_objects,
            "assignments": fresh_assignments,
            "selected_assignment": assignment,
            "fallback_used": fallback_used,
                "online_candidate_thresholds": {
                    "handle_threshold": float(handle_threshold),
                    "top_down_threshold": float(top_down_threshold),
                    "later_step_margin_accept": bool(later_step_margin_accept),
                    "greedy_margin_accept": bool(greedy_margin_accept),
                    "protrusion_quality": protrusion_quality,
                    "handle_point_count": handle_point_count,
                    "step_index": int(step_index),
                    "attempt_index": int(attempt_index),
                },
        },
        "candidate_rejection": selected_rejection,
        "perception_candidates": [
            {
                "id": candidate.get("id"),
                "grasp_type": candidate.get("grasp_type"),
                "score": candidate.get("score"),
                "pos": candidate.get("pos"),
            }
            for candidate in payload.get("grasp_candidates", [])
        ],
        "grasp_candidate_diagnostics": diagnostics.get("grasp_candidates", {}),
    }
    return selected_candidate, selected_rejection, summary


def build_parser():
    parser = argparse.ArgumentParser(description="Run a continuous online CupMugSorting ordering demo.")
    parser.add_argument("--task", default="robocasa/CupMugSorting")
    parser.add_argument("--policy", default="greedy", choices=["greedy", "rl"])
    parser.add_argument("--rl-policy-mode", default="stub", choices=["stub", "checkpoint"])
    parser.add_argument("--rl-checkpoint", default=None)
    parser.add_argument("--camera-name", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render-sleep-sec", type=float, default=0.005)
    parser.add_argument("--keep-open-sec", type=float, default=0.0)
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--retry-on-failure-once", action="store_true")
    parser.add_argument("--skip-axis-calibration", action="store_true")
    parser.add_argument("--enable-base-torso-preposition", action="store_true")
    parser.add_argument("--calibrate-base-action-mapping", action="store_true")
    parser.add_argument("--use-base-action-mapping", action="store_true")
    parser.add_argument("--auto-preposition-retry", action="store_true")
    parser.add_argument("--skip-vision-refresh-after-base-preposition", action="store_true")
    parser.add_argument("--retry-final-error-threshold", type=float, default=0.03)
    parser.add_argument("--base-torso-steps", type=int, default=30)
    parser.add_argument("--base-action-start", type=int, default=DEFAULT_BASE_ACTION_SLICE[0])
    parser.add_argument("--base-action-end", type=int, default=DEFAULT_BASE_ACTION_SLICE[1])
    parser.add_argument("--torso-action-index", type=int, default=DEFAULT_TORSO_ACTION_INDEX)
    parser.add_argument("--base-desired-xy-standoff", type=float, default=0.18)
    parser.add_argument("--base-xy-deadband", type=float, default=0.06)
    parser.add_argument("--base-preposition-gain", type=float, default=4.0)
    parser.add_argument("--base-action-limit", type=float, default=0.35)
    parser.add_argument("--base-mapping-trust", type=float, default=0.25)
    parser.add_argument("--base-max-world-delta", type=float, default=0.03)
    return parser


def main():
    args = build_parser().parse_args()

    checkpoint_interface = None
    if args.policy == "rl" and args.rl_policy_mode == "checkpoint":
        if not args.rl_checkpoint:
            raise RuntimeError("policy=rl with --rl-policy-mode checkpoint requires --rl-checkpoint")
        checkpoint_interface = inspect_cup_ordering_checkpoint_artifact(args.rl_checkpoint)
        if not checkpoint_interface.get("supports_inference"):
            final_summary = {
                "policy": args.policy,
                "selected_order": [],
                "finished": {},
                "retry_counts": {},
                "last_success": None,
                "last_failure_type": "checkpoint_policy",
                "failure_reason": "rl_checkpoint_inference_unsupported",
                "failure_category": classify_cup_ordering_failure(
                    failure_reason="rl_checkpoint_inference_unsupported",
                    failure_phase="checkpoint_policy",
                ),
                "checkpoint_interface": checkpoint_interface,
                "step_logs": [],
            }
            print(json.dumps({"online_demo_summary": final_summary}, indent=2))
            if args.save_trace:
                trace_path = Path(args.trace_path) if args.trace_path else PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "online-greedy-demo-trace.json"
                written_trace = _write_json(
                    trace_path,
                    {
                        "task": args.task,
                        "policy": args.policy,
                        "layout": args.layout,
                        "style": args.style,
                        "seed": args.seed,
                        "rl_policy_mode": args.rl_policy_mode,
                        "rl_checkpoint": args.rl_checkpoint,
                        "checkpoint_interface": checkpoint_interface,
                        **final_summary,
                    },
                )
                print(json.dumps({"online_demo_trace_path": str(written_trace)}, indent=2))
            return

    env_name = args.task.replace("robocasa/", "", 1)
    layout_id = args.layout
    if layout_id is None and env_name in CUP_MUG_SORTING_TASKS:
        layout_id = DEFAULT_LEFT_SINK_LAYOUT_ID
    style_id = args.style
    if style_id is None and env_name in CUP_MUG_SORTING_TASKS:
        style_id = 1
    seed = args.seed
    layout_and_style_ids = None
    if env_name in CUP_MUG_SORTING_TASKS and layout_id is not None and style_id is not None:
        layout_and_style_ids = [[int(layout_id), int(style_id)]]

    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    vision_config["target_labels"] = ["mug", "cup", "glass cup"]
    vision_config["obstacle_labels"] = []
    vision_config["candidate_types"] = ["top_down", "handle_top_down"]
    vision_config["handle_labels"] = ["cup", "mug", "glass cup"]

    config = build_env_config(
        task_name=args.task,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        layout=layout_id,
        style=style_id,
        seed=seed,
        layout_and_style_ids=layout_and_style_ids,
    )
    print(colored("Initializing online CupMugSorting ordering environment...", "yellow"))
    print(json.dumps(config, indent=2))
    sys.stdout.flush()
    env = robosuite.make(**config)

    try:
        obs = env.reset()
        env.render()
        time.sleep(max(args.render_sleep_sec, 0.05))

        action_mapping = None
        current_obs = obs
        if not args.skip_axis_calibration:
            action_mapping, current_obs, calibration_summary = calibrate_position_action_mapping(
                env,
                obs=current_obs,
                render_sleep_sec=float(args.render_sleep_sec),
            )
            print(json.dumps({"axis_calibration": calibration_summary}, indent=2))

        base_action_mapping = None
        if args.calibrate_base_action_mapping or args.use_base_action_mapping:
            current_obs, base_mapping_summary = calibrate_base_action_mapping(
                env,
                obs=current_obs,
                base_slice=(int(args.base_action_start), int(args.base_action_end)),
                pulse_magnitude=0.1,
                pulse_steps=4,
                render_sleep_sec=float(args.render_sleep_sec),
            )
            base_action_mapping = np.asarray(base_mapping_summary["eef_xy_delta_from_base_action"], dtype=float)
            print(json.dumps({"base_action_mapping": base_mapping_summary}, indent=2))

        finished = {}
        retry_counts = {}
        step_logs = []
        selected_order = []
        last_action = None
        last_success = None
        last_failure_type = None
        failure_reason = None

        for step_index in range(5):
            live_scene = _classify_live_scene(env, current_obs, args, thresholds_config, vision_config)
            mapping_payload = build_live_slot_mapping(
                targets=live_scene["classified_targets"],
                assignments=live_scene["assignments"],
                sorting_metadata=live_scene["sorting_metadata"],
                finished=finished,
                retry_counts=retry_counts,
                conf_threshold=float(thresholds_config["CONF_THRESH"]),
                step_index=step_index,
                last_action=last_action,
                last_success=last_success,
                last_failure_type=last_failure_type,
            )
            observation = mapping_payload["observation"]
            if not any(observation["action_mask"]):
                unfinished_sim_objects = [
                    sim_object for sim_object in live_scene["sim_objects"] if not finished.get(str(sim_object.get("name")))
                ]
                if unfinished_sim_objects:
                    live_scene = _augment_live_scene_with_sim_fallback(live_scene, finished)
                    mapping_payload = build_live_slot_mapping(
                        targets=live_scene["classified_targets"],
                        assignments=live_scene["assignments"],
                        sorting_metadata=live_scene["sorting_metadata"],
                        finished=finished,
                        retry_counts=retry_counts,
                        conf_threshold=float(thresholds_config["CONF_THRESH"]),
                        step_index=step_index,
                        last_action=last_action,
                        last_success=last_success,
                        last_failure_type=last_failure_type,
                    )
                    observation = mapping_payload["observation"]
            if args.policy == "rl":
                action, policy_debug = select_cup_ordering_rl_action(
                    observation,
                    seed=0 if args.seed is None else int(args.seed),
                    step_index=step_index,
                    checkpoint_path=args.rl_checkpoint if args.rl_policy_mode == "checkpoint" else None,
                    checkpoint_metadata=checkpoint_interface,
                )
                policy_debug["rl_policy_mode"] = args.rl_policy_mode
                policy_debug["rl_checkpoint"] = args.rl_checkpoint
            else:
                action, policy_debug = select_action(observation, args.policy)
            if action is None:
                summary = {
                    "step_index": step_index,
                    "selected_slot": None,
                    "failure_reason": "all_actions_invalid",
                    "failure_category": classify_cup_ordering_failure(failure_reason="all_actions_invalid"),
                    "observation": observation,
                    "fallback_added_names": live_scene.get("fallback_added_names", []),
                }
                print(json.dumps({"online_demo_step": summary}, indent=2))
                step_logs.append(summary)
                failure_reason = "all_actions_invalid"
                break

            selected_order.append(int(action))
            step_summary = {
                "step_index": step_index,
                "selected_slot": int(action),
                "policy_debug": policy_debug,
                "attempts": [],
            }

            resolved = resolve_live_target_from_slot(mapping_payload, action)
            if not resolved["is_valid"]:
                summary = {
                    "step_index": step_index,
                    "selected_slot": int(action),
                    "failure_reason": resolved["failure_reason"],
                    "failure_category": classify_cup_ordering_failure(failure_reason=resolved["failure_reason"]),
                    "observation": observation,
                }
                print(json.dumps({"online_demo_step": summary}, indent=2))
                step_logs.append(summary)
                failure_reason = resolved["failure_reason"]
                break

            attempt_success = False
            resolved_for_attempt = resolved
            preferred_object_id = resolved.get("slot", {}).get("object_id")
            for attempt_index in range(2 if args.retry_on_failure_once else 1):
                try:
                    selected_candidate, candidate_rejection, active_candidate_summary = _build_live_execution_candidate(
                        env,
                        current_obs,
                        resolved_for_attempt["assignment"],
                        args,
                        thresholds_config,
                        vision_config,
                        policy_name=args.policy,
                        step_index=step_index,
                        attempt_index=attempt_index,
                    )
                except RuntimeError as exc:
                    synthetic_failure_summary = {
                        "candidate": None,
                        "base_torso_preposition": None,
                        "candidate_after_base_preposition": None,
                        "base_action_mapping": None,
                        "auto_preposition_retry": None,
                        "reach": None,
                        "reachability_diagnosis": None,
                        "close": None,
                        "lift": None,
                        "place_plan": None,
                        "place": None,
                        "overall_success": False,
                        "failure_phase": "candidate_build",
                        "failure_reason": str(exc),
                        "failure_category": classify_cup_ordering_failure(
                            failure_reason=str(exc),
                            failure_phase="candidate_build",
                        ),
                    }
                    _append_attempt_log(
                        step_summary,
                        attempt_index=attempt_index,
                        policy_debug=policy_debug,
                        execution_summary=synthetic_failure_summary,
                        selected_object_id=resolved_for_attempt["slot"]["object_id"],
                    )
                    if attempt_index == 0 and args.retry_on_failure_once:
                        live_scene_retry = _classify_live_scene(env, current_obs, args, thresholds_config, vision_config)
                        mapping_retry = build_live_slot_mapping(
                            targets=live_scene_retry["classified_targets"],
                            assignments=live_scene_retry["assignments"],
                            sorting_metadata=live_scene_retry["sorting_metadata"],
                            finished=finished,
                            retry_counts=retry_counts,
                            conf_threshold=float(thresholds_config["CONF_THRESH"]),
                            step_index=step_index,
                            last_action=int(action),
                            last_success=False,
                            last_failure_type="candidate_build",
                        )
                        resolved_retry = resolve_live_target_with_identity_preference(
                            mapping_retry,
                            action,
                            preferred_object_id=preferred_object_id,
                        )
                        step_summary["retry_refresh_slot_valid"] = resolved_retry["is_valid"]
                        step_summary["retry_refresh_failure_reason"] = resolved_retry["failure_reason"]
                        step_summary["retry_refresh_identity_preserved"] = resolved_retry.get("identity_preserved")
                        if not resolved_retry["is_valid"]:
                            resolved_for_attempt = None
                            break
                        resolved_for_attempt = resolved_retry
                        continue
                    break

                def refresh_candidate(next_obs, reason):
                        return _build_live_execution_candidate(
                            env,
                            next_obs,
                            resolved_for_attempt["assignment"],
                            args,
                            thresholds_config,
                            vision_config,
                            policy_name=args.policy,
                            step_index=step_index,
                            attempt_index=attempt_index,
                        )

                base_mapping_summary = None
                if bool(resolved_for_attempt["slot"].get("has_handle")):
                    current_obs, base_action_mapping, base_mapping_summary = _ensure_handle_base_mapping(
                        env,
                        current_obs,
                        base_action_mapping,
                        args,
                    )
                live_options = _live_execution_options_for_slot(args, resolved_for_attempt, base_action_mapping)

                execution_result = run_live_target_execution(
                    env=env,
                    current_obs=current_obs,
                    candidate=selected_candidate,
                    active_candidate_summary=active_candidate_summary,
                    action_mapping=action_mapping,
                    base_action_mapping=base_action_mapping,
                    infer_candidate_from_obs=refresh_candidate,
                    cup_mug_sorting_place_anchors=cup_mug_sorting_place_anchors,
                    options=live_options,
                )
                current_obs = execution_result["obs"]
                execution_summary = execution_result["execution_summary"]
                success = bool(execution_summary.get("overall_success"))
                _append_attempt_log(
                    step_summary,
                    attempt_index=attempt_index,
                    policy_debug=policy_debug,
                    execution_summary=execution_summary,
                    selected_object_id=resolved_for_attempt["slot"]["object_id"],
                )
                step_summary.setdefault("attempt_options", []).append(
                    {
                        "attempt_index": attempt_index,
                        "has_handle": bool(resolved_for_attempt["slot"].get("has_handle")),
                        "enable_base_torso_preposition": bool(live_options.enable_base_torso_preposition),
                        "auto_preposition_retry": bool(live_options.auto_preposition_retry),
                        "use_base_action_mapping": bool(live_options.use_base_action_mapping),
                        "retry_place_once": bool(live_options.retry_place_once),
                        "base_desired_xy_standoff": float(live_options.base_desired_xy_standoff),
                        "base_xy_deadband": float(live_options.base_xy_deadband),
                        "base_action_limit": float(live_options.base_action_limit),
                    }
                )
                if base_mapping_summary is not None:
                    step_summary.setdefault("base_mapping_summaries", []).append(base_mapping_summary)
                if success:
                    finished[str(resolved_for_attempt["slot"]["object_id"])] = True
                    attempt_success = True
                    break

                retry_counts[str(resolved_for_attempt["slot"]["object_id"])] = int(retry_counts.get(str(resolved_for_attempt["slot"]["object_id"]), 0)) + 1
                if attempt_index == 0 and args.retry_on_failure_once:
                    live_scene_retry = _classify_live_scene(env, current_obs, args, thresholds_config, vision_config)
                    mapping_retry = build_live_slot_mapping(
                        targets=live_scene_retry["classified_targets"],
                        assignments=live_scene_retry["assignments"],
                        sorting_metadata=live_scene_retry["sorting_metadata"],
                        finished=finished,
                        retry_counts=retry_counts,
                        conf_threshold=float(thresholds_config["CONF_THRESH"]),
                        step_index=step_index,
                        last_action=int(action),
                        last_success=False,
                        last_failure_type=execution_summary.get("failure_phase"),
                    )
                    resolved_retry = resolve_live_target_with_identity_preference(
                        mapping_retry,
                        action,
                        preferred_object_id=preferred_object_id,
                    )
                    step_summary["retry_refresh_slot_valid"] = resolved_retry["is_valid"]
                    step_summary["retry_refresh_failure_reason"] = resolved_retry["failure_reason"]
                    step_summary["retry_refresh_identity_preserved"] = resolved_retry.get("identity_preserved")
                    if not resolved_retry["is_valid"]:
                        resolved_for_attempt = None
                        break
                    resolved_for_attempt = resolved_retry
                else:
                    break

            last_action = int(action)
            last_success = attempt_success
            last_failure_type = step_summary["execution_summary"].get("failure_phase") if step_summary.get("execution_summary") else None
            step_summary["failure_category"] = classify_cup_ordering_failure(
                failure_reason=(step_summary.get("execution_summary") or {}).get("failure_reason") or step_summary.get("retry_refresh_failure_reason"),
                failure_phase=(step_summary.get("execution_summary") or {}).get("failure_phase"),
                retry_refresh_failure_reason=step_summary.get("retry_refresh_failure_reason"),
            )
            step_logs.append(step_summary)
            print(json.dumps({"online_demo_step": step_summary}, indent=2))
            if not attempt_success:
                failure_reason = step_summary["execution_summary"].get("failure_phase") if step_summary.get("execution_summary") else step_summary.get("retry_refresh_failure_reason")
                break

        final_summary = {
            "policy": args.policy,
            "selected_order": selected_order,
            "finished": finished,
            "retry_counts": retry_counts,
            "last_success": last_success,
            "last_failure_type": last_failure_type,
            "failure_reason": failure_reason,
            "failure_category": classify_cup_ordering_failure(
                failure_reason=failure_reason,
                failure_phase=last_failure_type,
            ),
            "checkpoint_interface": checkpoint_interface,
            "step_logs": step_logs,
        }
        print(json.dumps({"online_demo_summary": final_summary}, indent=2))

        if args.save_trace:
            trace_path = Path(args.trace_path) if args.trace_path else PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "online-greedy-demo-trace.json"
            written_trace = _write_json(
                trace_path,
                {
                    "task": args.task,
                    "policy": args.policy,
                    "layout": layout_id,
                    "style": style_id,
                    "seed": seed,
                    "rl_policy_mode": args.rl_policy_mode if args.policy == "rl" else None,
                    "rl_checkpoint": args.rl_checkpoint if args.policy == "rl" else None,
                    "selected_order": selected_order,
                    "finished": finished,
                    "retry_counts": retry_counts,
                    "last_success": last_success,
                    "last_failure_type": last_failure_type,
                    "failure_reason": failure_reason,
                    "failure_category": final_summary["failure_category"],
                    "checkpoint_interface": checkpoint_interface,
                    "step_logs": step_logs,
                },
            )
            print(json.dumps({"online_demo_trace_path": str(written_trace)}, indent=2))
            sys.stdout.flush()

        keep_open_sec = float(args.keep_open_sec)
        if keep_open_sec > 0.0:
            print(colored(f"Keeping viewer open for {keep_open_sec:.1f}s — close window or Ctrl+C to exit", "yellow"))
            end_time = time.time() + keep_open_sec
            while time.time() < end_time:
                env.render()
                time.sleep(0.02)
        else:
            print(colored("Viewer open — close window or Ctrl+C to exit", "yellow"))
            try:
                while True:
                    env.render()
                    time.sleep(0.02)
            except KeyboardInterrupt:
                pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
