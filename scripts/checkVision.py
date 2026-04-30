"""Compare vision grasp targets against robosuite ground truth and executed poses."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import numpy as np
import yaml

from arm.controller import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    _follow_single_arm_trajectory,
    actuate_gripper,
    compute_grasp_waypoints,
    get_primary_object_pos,
    home_arms,
    verify_grasp,
)
from arm.env_wrapper import RobosuiteEnvWrapper
from fsm.perception_cycle import consume_perception_queue
from fsm.state_machine import State, TaskStateMachine
from main import build_single_grasp_pos, get_corrected_vision_target
from runtime.perception_queue import create_perception_queue
from scripts.grasp_demo_common import build_gt_summary
from scripts.demo_vision_grasp import build_demo_env_and_labels, summarize_payload
from vision.perception_loop import VisionPerceptionLoop


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def _to_list(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def _delta(a, b):
    if a is None or b is None:
        return None
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).tolist()


def _scalar_delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _build_summary_row(report):
    if report.get("error") is not None:
        return {
            "scenario": report.get("scenario"),
            "grasp_source": "error",
            "target_label": "-",
            "target_conf": None,
            "candidate_count": None,
            "gt_center_pos": None,
            "vision_target_pos": None,
            "vision_target_z_minus_gt_top_surface": None,
            "candidate_z_minus_gt_top_surface": None,
            "final_grasp_z_minus_gt_top_surface": None,
            "eef_before_close_z_minus_planned_grasp_z": None,
            "eef_before_close_z_minus_gt_top_surface": None,
            "approach_ok": None,
            "close_ok": None,
            "lift_ok": None,
            "grasp_verified": None,
            "error": report.get("error"),
        }

    deltas = report.get("deltas", {})
    execution = report.get("execution") or {}
    planning = report.get("planning", {})
    payload_summary = report.get("payload_summary", {})
    target = payload_summary.get("target", {})
    ground_truth = report.get("ground_truth", {})
    return {
        "scenario": report.get("scenario"),
        "grasp_source": planning.get("grasp_source"),
        "target_label": target.get("label"),
        "target_conf": target.get("conf"),
        "candidate_count": payload_summary.get("candidate_count"),
        "gt_center_pos": ground_truth.get("center_pos"),
        "vision_target_pos": target.get("pos"),
        "vision_target_z_minus_gt_top_surface": deltas.get(
            "vision_target_z_minus_gt_top_surface"
        ),
        "candidate_z_minus_gt_top_surface": deltas.get(
            "candidate_z_minus_gt_top_surface"
        ),
        "final_grasp_z_minus_gt_top_surface": deltas.get(
            "final_grasp_z_minus_gt_top_surface"
        ),
        "eef_before_close_z_minus_planned_grasp_z": deltas.get(
            "eef_before_close_z_minus_planned_grasp_z"
        ),
        "eef_before_close_z_minus_gt_top_surface": deltas.get(
            "eef_before_close_z_minus_gt_top_surface"
        ),
        "approach_ok": execution.get("approach_ok"),
        "close_ok": execution.get("close_ok"),
        "lift_ok": execution.get("lift_ok"),
        "grasp_verified": execution.get("grasp_verified"),
        "error": None,
    }


def _format_xyz(value):
    if value is None:
        return "-"
    coords = np.asarray(value, dtype=float).reshape(-1)
    if coords.size < 3:
        return "-"
    return f"({coords[0]:+.3f},{coords[1]:+.3f},{coords[2]:+.3f})"


def _format_cell(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def _render_summary_table(summary_rows):
    columns = [
        ("scenario", "scenario"),
        ("grasp_source", "source"),
        ("target_label", "label"),
        ("target_conf", "conf"),
        ("candidate_count", "cand"),
        ("gt_center_pos", "gt_xyz"),
        ("vision_target_pos", "vision_xyz"),
        ("candidate_z_minus_gt_top_surface", "cand_z-gt_top"),
        ("final_grasp_z_minus_gt_top_surface", "final_z-gt_top"),
        ("eef_before_close_z_minus_planned_grasp_z", "eef_z-plan_z"),
        ("eef_before_close_z_minus_gt_top_surface", "eef_z-gt_top"),
        ("approach_ok", "approach"),
        ("close_ok", "close"),
        ("lift_ok", "lift"),
        ("grasp_verified", "verified"),
        ("error", "error"),
    ]

    rendered_rows = []
    widths = []
    for key, header in columns:
        if key in {"gt_center_pos", "vision_target_pos"}:
            rendered_column = [_format_xyz(row.get(key)) for row in summary_rows]
        else:
            rendered_column = [_format_cell(row.get(key)) for row in summary_rows]
        rendered_rows.append(rendered_column)
        widths.append(max(len(header), *(len(cell) for cell in rendered_column)))

    header_line = " | ".join(
        header.ljust(width) for (_, header), width in zip(columns, widths)
    )
    separator_line = "-+-".join("-" * width for width in widths)

    lines = [header_line, separator_line]
    for row_index in range(len(summary_rows)):
        lines.append(
            " | ".join(
                rendered_rows[col_index][row_index].ljust(widths[col_index])
                for col_index in range(len(columns))
            )
        )
    return "\n".join(lines)


def _run_execution_diagnostic(env, final_grasp_pos, arm_idx=0):
    pre_grasp, grasp, lift = compute_grasp_waypoints(final_grasp_pos)
    object_pos_before = get_primary_object_pos(env.obs)
    eef_start = np.array(env.obs[f"robot{arm_idx}_eef_pos"], dtype=float)

    approach_ok = _follow_single_arm_trajectory(
        env,
        [eef_start, pre_grasp, grasp],
        arm_idx,
        GRIPPER_OPEN,
        stage_name="approach",
        max_steps_per_segment=120,
    )
    eef_before_close = np.array(env.obs[f"robot{arm_idx}_eef_pos"], dtype=float)
    close_ok = False
    lift_ok = False
    grasp_verified = False
    eef_after_lift = None
    object_pos_after = None

    if approach_ok:
        close_ok = actuate_gripper(env, arm_idx, GRIPPER_CLOSED)
        if close_ok:
            lift_ok = _follow_single_arm_trajectory(
                env,
                [np.array(env.obs[f"robot{arm_idx}_eef_pos"], dtype=float), lift],
                arm_idx,
                GRIPPER_CLOSED,
                stage_name="lift",
                max_steps_per_segment=120,
            )
            eef_after_lift = np.array(env.obs[f"robot{arm_idx}_eef_pos"], dtype=float)
            if lift_ok:
                grasp_verified = bool(verify_grasp(env, arm_idx, object_pos_before))
                object_pos_after = get_primary_object_pos(env.obs)

    return {
        "approach_ok": bool(approach_ok),
        "close_ok": bool(close_ok),
        "lift_ok": bool(lift_ok),
        "grasp_verified": bool(grasp_verified),
        "eef_start": _to_list(eef_start),
        "planned_pre_grasp": _to_list(pre_grasp),
        "planned_grasp": _to_list(grasp),
        "planned_lift": _to_list(lift),
        "eef_before_close": _to_list(eef_before_close),
        "eef_after_lift": _to_list(eef_after_lift),
        "object_pos_before": _to_list(object_pos_before),
        "object_pos_after": _to_list(object_pos_after),
    }


def run_check(
    seed=None,
    scenario="cube",
    allow_download=False,
    execute=True,
):
    static_camera_config = load_yaml(PROJECT_ROOT / "configs" / "camera.yaml")
    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    conf_thresh = float(thresholds_config["CONF_THRESH"])
    env_config, target_labels, obstacle_labels = build_demo_env_and_labels(scenario)

    if allow_download:
        vision_config["local_files_only"] = False

    vision_config["target_labels"] = target_labels
    vision_config["obstacle_labels"] = obstacle_labels

    env = RobosuiteEnvWrapper(config=env_config, camera_config=static_camera_config)
    if seed is not None:
        env.reset(seed=seed)

    fx, fy, cx, cy = env.get_camera_intrinsics()
    rotation, translation = env.get_camera_extrinsics()
    camera_to_world_transform = env.get_camera_to_world_transform()
    camera_config = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_to_world_transform": camera_to_world_transform.tolist(),
        "T_world_cam": {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
    }

    perception_loop = VisionPerceptionLoop(
        vision_config=vision_config,
        camera_config=camera_config,
        conf_thresh=conf_thresh,
    )
    perception_queue = create_perception_queue()
    fsm = TaskStateMachine()

    rgb_image, depth_image, proprio = env.get_observation()
    payload = perception_loop.publish_from_observation(
        perception_queue=perception_queue,
        rgb_image=rgb_image,
        depth_image=depth_image,
    )

    fsm.transition_to(State.PLANNING)
    payload, cycle_ok = consume_perception_queue(fsm, perception_queue)
    if not cycle_ok:
        raise RuntimeError("Vision payload did not pass planning validation.")

    selected_candidate = fsm.get_selected_candidate()
    corrected_target = get_corrected_vision_target(
        env=env,
        latest_detection=payload,
        vision_config=vision_config,
    )

    if selected_candidate is not None:
        grasp_source = "candidate"
        grasp_target_pos = np.array(selected_candidate["pos"], dtype=float)
    elif corrected_target is not None:
        grasp_source = "vision"
        grasp_target_pos = np.array(corrected_target, dtype=float)
    else:
        raise RuntimeError("No usable visual grasp target was produced.")

    final_grasp_pos = build_single_grasp_pos(
        env=env,
        grasp_target_pos=grasp_target_pos,
        source=grasp_source,
        vision_config=vision_config,
    )
    pre_grasp, grasp_waypoint, lift_waypoint = compute_grasp_waypoints(final_grasp_pos)

    gt_summary = build_gt_summary(env, scenario)
    gt_center = gt_summary["center_pos"]
    gt_top_surface_z = gt_summary["top_surface_z"]

    execution_summary = None
    if execute:
        execution_summary = _run_execution_diagnostic(env, final_grasp_pos, arm_idx=0)
        home_arms(env)

    report = {
        "scenario": scenario,
        "camera": {
            "camera_name": env.camera_name,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        },
        "payload_summary": summarize_payload(payload),
        "planning": {
            "grasp_source": grasp_source,
            "selected_candidate_index": fsm.get_selected_candidate_index(),
            "selected_candidate": selected_candidate,
            "target_pos": payload.get("target", {}).get("pos"),
            "corrected_target_pos": _to_list(corrected_target),
            "raw_grasp_target_pos": _to_list(grasp_target_pos),
            "final_grasp_pos": _to_list(final_grasp_pos),
            "planned_pre_grasp": _to_list(pre_grasp),
            "planned_grasp": _to_list(grasp_waypoint),
            "planned_lift": _to_list(lift_waypoint),
            "robot0_eef_pos_initial": _to_list(proprio["robot0_eef_pos"]),
        },
        "ground_truth": gt_summary,
        "deltas": {
            "vision_target_minus_gt_center": _delta(payload.get("target", {}).get("pos"), gt_center),
            "candidate_minus_gt_center": _delta(
                None if selected_candidate is None else selected_candidate.get("pos"),
                gt_center,
            ),
            "final_grasp_minus_gt_center": _delta(final_grasp_pos, gt_center),
            "vision_target_z_minus_gt_top_surface": _scalar_delta(
                None if payload.get("target") is None else payload.get("target", {}).get("pos", [None, None, None])[2],
                gt_top_surface_z,
            ),
            "candidate_z_minus_gt_top_surface": _scalar_delta(
                None if selected_candidate is None else selected_candidate.get("pos", [None, None, None])[2],
                gt_top_surface_z,
            ),
            "final_grasp_z_minus_gt_top_surface": _scalar_delta(final_grasp_pos[2], gt_top_surface_z),
        },
        "execution": execution_summary,
    }

    if execution_summary is not None:
        report["deltas"].update(
            {
                "eef_before_close_minus_planned_grasp": _delta(
                    execution_summary["eef_before_close"],
                    execution_summary["planned_grasp"],
                ),
                "eef_before_close_z_minus_planned_grasp_z": _scalar_delta(
                    execution_summary["eef_before_close"][2],
                    execution_summary["planned_grasp"][2],
                ),
                "eef_before_close_z_minus_gt_top_surface": _scalar_delta(
                    execution_summary["eef_before_close"][2],
                    gt_top_surface_z,
                ),
                "eef_after_lift_minus_planned_lift": _delta(
                    execution_summary["eef_after_lift"],
                    execution_summary["planned_lift"],
                )
                if execution_summary.get("eef_after_lift") is not None
                else None,
            }
        )

    return report


def main():
    supported_scenarios = ["cube", "can", "milk", "bread", "cereal"]
    parser = argparse.ArgumentParser(
        description="Compare vision grasp targets vs GT and executed EEF poses for robosuite demos."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic reset seed.")
    parser.add_argument(
        "--scenario",
        default="cube",
        choices=supported_scenarios + ["all"],
        help="Demo object scenario: cube, can, milk, bread, cereal, or all",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face download if the GroundingDINO cache is missing.",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Only compare vision/planning vs GT without executing the grasp trajectory.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact per-scenario summary instead of the full report.",
    )
    parser.add_argument(
        "--summary-table",
        action="store_true",
        help="Print the summary as a human-readable text table.",
    )
    args = parser.parse_args()
    scenarios = supported_scenarios if args.scenario == "all" else [args.scenario]
    reports = []
    for scenario in scenarios:
        try:
            reports.append(
                run_check(
                    seed=args.seed,
                    scenario=scenario,
                    allow_download=args.allow_download,
                    execute=not args.skip_execution,
                )
            )
        except Exception as exc:
            if args.scenario != "all":
                raise
            reports.append(
                {
                    "scenario": scenario,
                    "error": str(exc),
                }
            )

    if args.summary:
        summary = [_build_summary_row(report) for report in reports]
        if len(summary) == 1:
            print(json.dumps(summary[0], indent=2))
        else:
            print(json.dumps({"summary": summary}, indent=2))
        return

    if args.summary_table:
        summary = [_build_summary_row(report) for report in reports]
        print(_render_summary_table(summary))
        return

    if len(reports) == 1:
        print(json.dumps(reports[0], indent=2))
    else:
        print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
