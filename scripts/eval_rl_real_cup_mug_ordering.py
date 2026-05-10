"""Batch evaluation for CupMugSorting using real RoboCasa online execution."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import robocasa  # noqa: F401 - import registers RoboCasa envs into robosuite.make
import robosuite

from arm.base_torso import DEFAULT_BASE_ACTION_SLICE, DEFAULT_TORSO_ACTION_INDEX, calibrate_base_action_mapping
from arm.calibration import calibrate_position_action_mapping
from arm.cup_mug_live_execution import run_live_target_execution
from rl.cup_mug_live_mapping import build_live_slot_mapping, resolve_live_target_from_slot, resolve_live_target_with_identity_preference
from rl.cup_mug_metrics import aggregate_episode_records, build_episode_record, write_ordering_metrics
from rl.cup_mug_policies import select_action
from rl.harness import classify_cup_ordering_failure, inspect_cup_ordering_checkpoint_artifact, load_rl_config, select_cup_ordering_rl_action, validate_rl_config
from scripts.demo_online_cup_mug_ordering import (
    _append_attempt_log,
    _augment_live_scene_with_sim_fallback,
    _build_live_execution_candidate,
    _classify_live_scene,
    _ensure_handle_base_mapping,
    _live_execution_options_for_slot,
)
from scripts.demo_robocasa_reach_onscreen import CUP_MUG_SORTING_TASKS, DEFAULT_LEFT_SINK_LAYOUT_ID, build_env_config, cup_mug_sorting_place_anchors, load_yaml


def build_parser():
    parser = argparse.ArgumentParser(description="Batch real-environment evaluation for CupMugSorting policies.")
    parser.add_argument("--config", required=True, help="Path to cup-ordering RL YAML config")
    parser.add_argument("--policy", required=True, choices=["random", "greedy", "rl"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--checkpoint", default=None, help="Required when --policy rl --rl-policy-mode checkpoint")
    parser.add_argument("--rl-policy-mode", default="checkpoint", choices=["stub", "checkpoint"])
    parser.add_argument("--output-name", default=None, help="Optional metrics JSON file name")
    parser.add_argument("--save-report", action="store_true", help="Write per-episode trace report JSON")
    parser.add_argument("--camera-name", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--render-sleep-sec", type=float, default=0.0)
    parser.add_argument("--skip-axis-calibration", action="store_true")
    parser.add_argument("--enable-base-torso-preposition", action="store_true")
    parser.add_argument("--calibrate-base-action-mapping", action="store_true")
    parser.add_argument("--use-base-action-mapping", action="store_true")
    parser.add_argument("--auto-preposition-retry", action="store_true")
    parser.add_argument("--skip-vision-refresh-after-base-preposition", action="store_true")
    parser.add_argument("--retry-on-failure-once", action="store_true")
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


def _resolved_layout_style(task_name, layout, style):
    env_name = task_name.replace("robocasa/", "", 1)
    layout_id = layout
    style_id = style
    if layout_id is None and env_name in CUP_MUG_SORTING_TASKS:
        layout_id = DEFAULT_LEFT_SINK_LAYOUT_ID
    if style_id is None and env_name in CUP_MUG_SORTING_TASKS:
        style_id = 1
    layout_and_style_ids = None
    if env_name in CUP_MUG_SORTING_TASKS and layout_id is not None and style_id is not None:
        layout_and_style_ids = [[int(layout_id), int(style_id)]]
    return layout_id, style_id, layout_and_style_ids


def _count_failures(step_logs):
    failure_counts = Counter()
    invalid_action_count = 0
    for entry in step_logs:
        if entry.get("failure_reason") == "all_actions_invalid":
            failure_counts["all_actions_invalid"] += 1
            invalid_action_count += 1
        execution_summary = entry.get("execution_summary") or {}
        failure_phase = execution_summary.get("failure_phase")
        if failure_phase:
            failure_counts[str(failure_phase)] += 1
        retry_reason = entry.get("retry_refresh_failure_reason")
        if retry_reason:
            failure_counts[str(retry_reason)] += 1
    return dict(failure_counts), int(invalid_action_count)


def _episode_record_from_summary(summary, policy):
    finished_lookup = dict(summary.get("finished", {}))
    completed_object_ids = [object_id for object_id, done in finished_lookup.items() if done]
    failure_counts, invalid_action_count = _count_failures(summary.get("step_logs", []))
    completed_count = len(completed_object_ids)
    success = bool(completed_count >= 5 and summary.get("failure_reason") is None)
    return build_episode_record(
        {
            "episode_id": summary.get("episode_id"),
            "seed": summary.get("seed"),
            "selected_order": list(summary.get("selected_order", [])),
            "completed_slots": completed_object_ids,
            "completed_count": completed_count,
            "success": success,
            "invalid_action_count": invalid_action_count,
            "failure_counts": failure_counts,
            "reward_total": 0.0,
            "reward_breakdown": {},
            "step_logs": list(summary.get("step_logs", [])),
            "max_targets": 5,
        },
        policy=policy,
        duration_sec=0.0,
    )


def _run_one_episode(runtime_args, config, checkpoint_interface, episode_index, seed):
    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    vision_config["target_labels"] = ["mug", "cup", "glass cup"]
    vision_config["obstacle_labels"] = []
    vision_config["candidate_types"] = ["top_down", "handle_top_down"]
    vision_config["handle_labels"] = ["cup", "mug", "glass cup"]

    layout_id, style_id, layout_and_style_ids = _resolved_layout_style(
        config["task_name"],
        runtime_args.layout,
        runtime_args.style,
    )
    env_config = build_env_config(
        task_name=config["task_name"],
        camera_name=runtime_args.camera_name,
        width=runtime_args.width,
        height=runtime_args.height,
        layout=layout_id,
        style=style_id,
        seed=seed,
        layout_and_style_ids=layout_and_style_ids,
    )
    env_config["has_renderer"] = False
    env_config["has_offscreen_renderer"] = True
    env = robosuite.make(**env_config)
    env.render = lambda *args, **kwargs: None

    try:
        obs = env.reset()
        action_mapping = None
        current_obs = obs
        if not runtime_args.skip_axis_calibration:
            action_mapping, current_obs, _ = calibrate_position_action_mapping(
                env,
                obs=current_obs,
                render_sleep_sec=float(runtime_args.render_sleep_sec),
            )

        base_action_mapping = None
        if runtime_args.calibrate_base_action_mapping or runtime_args.use_base_action_mapping:
            current_obs, base_mapping_summary = calibrate_base_action_mapping(
                env,
                obs=current_obs,
                base_slice=(int(runtime_args.base_action_start), int(runtime_args.base_action_end)),
                pulse_magnitude=0.1,
                pulse_steps=4,
                render_sleep_sec=float(runtime_args.render_sleep_sec),
            )
            base_action_mapping = np.asarray(base_mapping_summary["eef_xy_delta_from_base_action"], dtype=float)

        finished = {}
        retry_counts = {}
        step_logs = []
        selected_order = []
        last_action = None
        last_success = None
        last_failure_type = None
        failure_reason = None

        for step_index in range(int(config["max_targets"])):
            live_scene = _classify_live_scene(env, current_obs, runtime_args, thresholds_config, vision_config)
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

            if runtime_args.policy == "rl":
                action, policy_debug = select_cup_ordering_rl_action(
                    observation,
                    seed=int(seed),
                    step_index=step_index,
                    checkpoint_path=runtime_args.checkpoint if runtime_args.rl_policy_mode == "checkpoint" else None,
                    checkpoint_metadata=checkpoint_interface,
                )
                policy_debug["rl_policy_mode"] = runtime_args.rl_policy_mode
                policy_debug["rl_checkpoint"] = runtime_args.checkpoint
            else:
                action, policy_debug = select_action(observation, runtime_args.policy)

            if action is None:
                summary = {
                    "step_index": step_index,
                    "selected_slot": None,
                    "failure_reason": "all_actions_invalid",
                    "failure_category": classify_cup_ordering_failure(failure_reason="all_actions_invalid"),
                    "observation": observation,
                    "policy_debug": policy_debug,
                    "fallback_added_names": live_scene.get("fallback_added_names", []),
                }
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
                    "policy_debug": policy_debug,
                }
                step_logs.append(summary)
                failure_reason = resolved["failure_reason"]
                break

            attempt_success = False
            resolved_for_attempt = resolved
            preferred_object_id = resolved.get("slot", {}).get("object_id")
            for attempt_index in range(2 if runtime_args.retry_on_failure_once else 1):
                try:
                    selected_candidate, _, active_candidate_summary = _build_live_execution_candidate(
                        env,
                        current_obs,
                        resolved_for_attempt["assignment"],
                        runtime_args,
                        thresholds_config,
                        vision_config,
                        policy_name=runtime_args.policy,
                        step_index=step_index,
                        attempt_index=attempt_index,
                    )
                except RuntimeError as exc:
                    synthetic_failure_summary = {
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
                    break

                def refresh_candidate(next_obs, reason):
                    return _build_live_execution_candidate(
                        env,
                        next_obs,
                        resolved_for_attempt["assignment"],
                        runtime_args,
                        thresholds_config,
                        vision_config,
                        policy_name=runtime_args.policy,
                        step_index=step_index,
                        attempt_index=attempt_index,
                    )

                base_mapping_summary = None
                if bool(resolved_for_attempt["slot"].get("has_handle")):
                    current_obs, base_action_mapping, base_mapping_summary = _ensure_handle_base_mapping(
                        env,
                        current_obs,
                        base_action_mapping,
                        runtime_args,
                    )
                live_options = _live_execution_options_for_slot(runtime_args, resolved_for_attempt, base_action_mapping)
                try:
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
                except RuntimeError as exc:
                    execution_summary = {
                        "candidate": {
                            "id": int(selected_candidate["id"]),
                            "grasp_type": selected_candidate["grasp_type"],
                            "pos": selected_candidate["pos"],
                            "score": float(selected_candidate.get("score", 0.0)),
                        },
                        "overall_success": False,
                        "failure_phase": "candidate_refresh",
                        "failure_reason": str(exc),
                        "failure_category": classify_cup_ordering_failure(
                            failure_reason=str(exc),
                            failure_phase="candidate_refresh",
                        ),
                    }
                success = bool(execution_summary.get("overall_success"))
                _append_attempt_log(
                    step_summary,
                    attempt_index=attempt_index,
                    policy_debug=policy_debug,
                    execution_summary=execution_summary,
                    selected_object_id=resolved_for_attempt["slot"]["object_id"],
                )
                if base_mapping_summary is not None:
                    step_summary.setdefault("base_mapping_summaries", []).append(base_mapping_summary)
                if success:
                    finished[str(resolved_for_attempt["slot"]["object_id"])] = True
                    attempt_success = True
                    break

                retry_counts[str(resolved_for_attempt["slot"]["object_id"])] = int(retry_counts.get(str(resolved_for_attempt["slot"]["object_id"]), 0)) + 1
                if attempt_index == 0 and runtime_args.retry_on_failure_once:
                    live_scene_retry = _classify_live_scene(env, current_obs, runtime_args, thresholds_config, vision_config)
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
            if not attempt_success:
                failure_reason = step_summary["execution_summary"].get("failure_phase") if step_summary.get("execution_summary") else step_summary.get("retry_refresh_failure_reason")
                break

        return {
            "episode_id": f"episode-{int(episode_index):03d}",
            "seed": int(seed),
            "policy": runtime_args.policy,
            "selected_order": selected_order,
            "finished": finished,
            "retry_counts": retry_counts,
            "last_success": last_success,
            "last_failure_type": last_failure_type,
            "failure_reason": failure_reason,
            "failure_category": classify_cup_ordering_failure(failure_reason=failure_reason, failure_phase=last_failure_type),
            "checkpoint_interface": checkpoint_interface,
            "step_logs": step_logs,
        }
    finally:
        env.close()


def main():
    args = build_parser().parse_args()
    config = validate_rl_config(load_rl_config(args.config))
    if config.get("mode") != "cup_ordering":
        raise SystemExit("This script only supports cup_ordering configs.")
    if args.policy == "rl" and args.rl_policy_mode == "checkpoint" and not args.checkpoint:
        raise SystemExit("--checkpoint is required when --policy rl --rl-policy-mode checkpoint")

    checkpoint_interface = None
    if args.policy == "rl":
        checkpoint_interface = inspect_cup_ordering_checkpoint_artifact(
            args.checkpoint if args.rl_policy_mode == "checkpoint" else None
        )

    records = []
    traces = []
    for episode_offset in range(int(args.episodes)):
        seed = int(args.seed_start) + episode_offset
        summary = _run_one_episode(args, config, checkpoint_interface, episode_offset + 1, seed)
        traces.append(summary)
        records.append(_episode_record_from_summary(summary, args.policy))
        print(json.dumps({"real_eval_episode_summary": summary}, indent=2))

    metrics_payload = aggregate_episode_records(
        records,
        policy=args.policy,
        episodes=int(args.episodes),
        seed_start=int(args.seed_start),
        max_targets=int(config["max_targets"]),
        scene_config=dict(config.get("scene", {})),
    )
    output_dir = Path(config["artifacts"]["root_dir"]) / config["artifacts"]["metrics_dir"]
    output_name = args.output_name or f"{args.policy}-real-episodes-{int(args.episodes)}.json"
    metrics_path = write_ordering_metrics(metrics_payload, output_dir=output_dir, file_name=output_name)
    print(metrics_path)

    if args.save_report:
        report_dir = Path(config["artifacts"]["root_dir"]) / config["artifacts"].get("report_dir", "reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{args.policy}-real-episodes-{int(args.episodes)}-trace.json"
        report_path.write_text(
            json.dumps(
                {
                    "mode": "cup_ordering_real_env",
                    "policy": args.policy,
                    "episodes": int(args.episodes),
                    "seed_start": int(args.seed_start),
                    "checkpoint": args.checkpoint,
                    "rl_policy_mode": args.rl_policy_mode if args.policy == "rl" else None,
                    "scene_config": dict(config.get("scene", {})),
                    "episode_traces": traces,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(report_path)


if __name__ == "__main__":
    raise SystemExit(main())
