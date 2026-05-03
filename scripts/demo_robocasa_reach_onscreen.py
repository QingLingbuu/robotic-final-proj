"""Run a vision-driven RoboCasa reach while showing the live onscreen viewer."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import robocasa  # noqa: F401 - import registers RoboCasa envs into robosuite.make
import robosuite
import robosuite.utils.camera_utils as robosuite_camera_utils
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from termcolor import colored

from arm.action_space_diagnostics import diagnose_action_space
from arm.base_torso import (
    DEFAULT_BASE_ACTION_SLICE,
    DEFAULT_TORSO_ACTION_INDEX,
    calibrate_base_action_mapping,
    preposition_base_torso,
)
from arm.calibration import calibrate_position_action_mapping
from arm.cup_mug_live_execution import LiveExecutionOptions, run_live_target_execution
from arm.reachability import diagnose_reachability
from arm.reach_retry import should_retry_with_preposition
from arm.robocasa_execution import get_robot0_eef_quat
from arm.robocasa_primitives import execute_close, execute_lift, execute_oriented_top_down_reach, execute_place, execute_reach
from planner.sorting_zones import choose_cup_mug_place_target
from vision.detected_objects import build_detected_objects
from vision.perception_loop import VisionPerceptionLoop
from vision.sorting_policy import (
    assign_sim_metadata_to_targets,
    build_sim_metadata_fallback_target,
    classify_drinkware_targets,
    select_drinkware_target,
)
from planner.candidates import choose_reachable_candidate


DEFAULT_LEFT_SINK_LAYOUT_ID = 1
CUP_MUG_SORTING_TASKS = {"CupMugSorting", "CupMugSortingClean"}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def normalize_rgb_uint8(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in {1, 3, 4} and frame.shape[-1] not in {1, 3, 4}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype.kind == "f":
        max_value = float(np.nanmax(frame)) if frame.size else 0.0
        if max_value <= 1.0:
            frame = np.clip(frame * 255.0, 0.0, 255.0)
        else:
            frame = np.clip(frame, 0.0, 255.0)
    return np.asarray(frame, dtype=np.uint8)


def normalize_depth(depth, sim):
    depth = np.asarray(depth, dtype=float)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if np.all(np.isfinite(depth)) and np.all(depth >= 0.0) and np.all(depth <= 1.0):
        depth = robosuite_camera_utils.get_real_depth_map(sim=sim, depth_map=depth)
    return np.asarray(depth, dtype=float)


def build_env_config(
    task_name,
    camera_name,
    width,
    height,
    layout=None,
    style=None,
    seed=None,
    layout_and_style_ids=None,
):
    if layout_and_style_ids is not None:
        layout = None
        style = None
    return {
        "env_name": task_name.replace("robocasa/", "", 1),
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "camera_names": [camera_name],
        "camera_widths": [int(width)],
        "camera_heights": [int(height)],
        "camera_depths": [True],
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "render_camera": "robot0_frontview",
        "ignore_done": True,
        "use_camera_obs": True,
        "use_object_obs": True,
        "control_freq": 20,
        "renderer": "mjviewer",
        "layout_ids": layout,
        "style_ids": style,
        "seed": seed,
        "obj_instance_split": None,
        "layout_and_style_ids": layout_and_style_ids,
        "translucent_robot": True,
    }


def resolve_camera_config(env, camera_name, width, height):
    intrinsic = robosuite_camera_utils.get_camera_intrinsic_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    camera_to_world = robosuite_camera_utils.get_camera_extrinsic_matrix(
        sim=env.sim,
        camera_name=camera_name,
    )
    world_to_camera = np.linalg.inv(camera_to_world)
    transform = robosuite_camera_utils.get_camera_transform_matrix(
        sim=env.sim,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width,
    )
    return {
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "camera_to_world_transform": np.linalg.inv(np.asarray(transform, dtype=float)).tolist(),
        "T_world_cam": {
            "rotation": np.asarray(world_to_camera[:3, :3], dtype=float).tolist(),
            "translation": np.asarray(world_to_camera[:3, 3], dtype=float).tolist(),
        },
    }


def get_rgbd(obs, camera_name, sim):
    rgb_key = f"{camera_name}_image"
    depth_key = f"{camera_name}_depth"
    if rgb_key not in obs:
        raise RuntimeError(f"Observation missing RGB key {rgb_key!r}. Available keys: {sorted(obs.keys())}")
    if depth_key not in obs:
        raise RuntimeError(f"Observation missing depth key {depth_key!r}. Available keys: {sorted(obs.keys())}")
    rgb = normalize_rgb_uint8(obs[rgb_key][::-1].copy())
    depth = normalize_depth(obs[depth_key][::-1].copy(), sim=sim)
    return rgb, depth

def normalize_angle_deg(angle_deg):
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def axis_yaw_deg(axis):
    axis = np.asarray(axis, dtype=float)
    xy = axis[:2]
    norm = float(np.linalg.norm(xy))
    if norm < 1e-8:
        return None
    xy = xy / norm
    return float(np.degrees(np.arctan2(xy[1], xy[0])))


def build_yaw_debug(obs, candidate, diagnostics):
    current_rot = T.quat2mat(get_robot0_eef_quat(obs))
    current_closing = np.asarray(current_rot[:, 1], dtype=float)
    target_closing = np.asarray(candidate["orientation"][0], dtype=float)
    current_yaw = axis_yaw_deg(current_closing)
    target_yaw = axis_yaw_deg(target_closing)
    yaw_error = None
    if current_yaw is not None and target_yaw is not None:
        yaw_error = normalize_angle_deg(target_yaw - current_yaw)
        if abs(yaw_error) > 90.0:
            yaw_error = normalize_angle_deg(yaw_error - np.sign(yaw_error) * 180.0)

    candidate_diagnostics = diagnostics.get(str(candidate.get("id")), {})
    handle_center = candidate_diagnostics.get("handle_center") or candidate.get("source_handle_pos")
    grasp_xy = candidate_diagnostics.get("grasp_xy")
    outward_axis_xy = candidate_diagnostics.get("outward_axis_xy")
    outward_axis = None if outward_axis_xy is None else np.asarray([outward_axis_xy[0], outward_axis_xy[1], 0.0], dtype=float)
    target_dot_outward = None
    current_dot_outward = None
    if outward_axis is not None:
        outward_norm = float(np.linalg.norm(outward_axis))
        target_norm = float(np.linalg.norm(target_closing))
        current_norm = float(np.linalg.norm(current_closing))
        if outward_norm > 1e-8 and target_norm > 1e-8:
            target_dot_outward = float(np.dot(target_closing, outward_axis) / (target_norm * outward_norm))
        if outward_norm > 1e-8 and current_norm > 1e-8:
            current_dot_outward = float(np.dot(current_closing, outward_axis) / (current_norm * outward_norm))
    return {
        "candidate_id": candidate.get("id"),
        "grasp_type": candidate.get("grasp_type"),
        "candidate_pos": candidate.get("pos"),
        "handle_center": handle_center,
        "grasp_xy": grasp_xy,
        "outward_axis_xy": outward_axis_xy,
        "target_closing_axis": target_closing.tolist(),
        "current_closing_axis": current_closing.tolist(),
        "current_closing_axis_source": "eef_y_axis",
        "target_closing_dot_outward": target_dot_outward,
        "current_closing_dot_outward": current_dot_outward,
        "current_eef_x_axis": np.asarray(current_rot[:, 0], dtype=float).tolist(),
        "current_eef_y_axis": np.asarray(current_rot[:, 1], dtype=float).tolist(),
        "current_eef_z_axis": np.asarray(current_rot[:, 2], dtype=float).tolist(),
        "target_yaw_deg": target_yaw,
        "current_yaw_deg": current_yaw,
        "expected_yaw_rotation_deg": yaw_error,
        "handle_point_source": candidate_diagnostics.get("handle_point_source"),
        "bbox_side_band": candidate_diagnostics.get("bbox_side_band"),
        "estimated_width": candidate_diagnostics.get("estimated_width") or candidate.get("source_handle_width"),
        "selected_width": candidate_diagnostics.get("selected_width") or candidate.get("gripper_width"),
    }


def _normalize_xy(vector):
    vector = np.asarray(vector, dtype=float)[:2]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _angle_deg_between_xy(axis_a, axis_b):
    axis_a = _normalize_xy(axis_a)
    axis_b = _normalize_xy(axis_b)
    if axis_a is None or axis_b is None:
        return None
    dot = float(np.clip(np.dot(axis_a, axis_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _is_descendant_body(model, body_id, root_body_id):
    current = int(body_id)
    root_body_id = int(root_body_id)
    while current >= 0:
        if current == root_body_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _mesh_vertices_for_geom(sim, geom_id):
    model = sim.model
    data_id = int(model.geom_dataid[geom_id])
    if data_id < 0:
        return None
    vert_num = int(model.mesh_vertnum[data_id])
    if vert_num <= 0:
        return None
    vert_adr = int(model.mesh_vertadr[data_id])
    local_vertices = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=float)
    geom_pos = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
    geom_rot = np.asarray(sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    return geom_pos + local_vertices @ geom_rot.T


def _estimate_handle_from_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 16:
        return None
    xy = points[:, :2]
    z = points[:, 2]
    center_xy = np.median(xy, axis=0)
    rel_xy = xy - center_xy
    radial = np.linalg.norm(rel_xy, axis=1)
    z_low = float(np.quantile(z, 0.20))
    z_high = float(np.quantile(z, 0.95))
    z_mask = (z >= z_low) & (z <= z_high)
    angles = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])
    sector_count = 72
    sector_width = 2.0 * np.pi / float(sector_count)
    min_points = max(8, int(0.005 * points.shape[0]))
    best_sector = None
    for sector_idx in range(sector_count):
        sector_center = -np.pi + (sector_idx + 0.5) * sector_width
        angular_delta = np.arctan2(np.sin(angles - sector_center), np.cos(angles - sector_center))
        sector_mask = z_mask & (np.abs(angular_delta) <= sector_width)
        if np.count_nonzero(sector_mask) < min_points:
            continue
        sector_radial = radial[sector_mask]
        score = float(np.quantile(sector_radial, 0.95))
        if best_sector is None or score > best_sector["score"]:
            best_sector = {
                "center": float(sector_center),
                "score": score,
                "mask": sector_mask,
            }
    if best_sector is None:
        return None
    sector_radial = radial[best_sector["mask"]]
    radial_threshold = float(np.quantile(sector_radial, 0.60))
    handle_mask = best_sector["mask"] & (radial >= radial_threshold)
    handle_points = points[handle_mask]
    if handle_points.shape[0] < min_points:
        return None
    handle_center = np.mean(handle_points, axis=0)
    tail_distance = np.linalg.norm(handle_points[:, :2] - center_xy, axis=1)
    tail_points = handle_points[tail_distance >= float(np.quantile(tail_distance, 0.75))]
    direction_center_xy = np.median(tail_points[:, :2], axis=0) if tail_points.shape[0] else handle_center[:2]
    outward_axis_xy = _normalize_xy(direction_center_xy - center_xy)
    if outward_axis_xy is None:
        return None
    return {
        "center_xy": center_xy.tolist(),
        "handle_center": handle_center.tolist(),
        "outward_axis_xy": outward_axis_xy.tolist(),
        "sector_center_deg": float(np.degrees(best_sector["center"])),
        "sector_score": float(best_sector["score"]),
        "handle_point_count": int(handle_points.shape[0]),
        "point_count": int(points.shape[0]),
        "z_low": z_low,
        "z_high": z_high,
    }


def build_sim_handle_mesh_debug(env, candidate, candidate_diagnostics, obj_name="obj"):
    sim = getattr(env, "sim", None)
    obj_body_id = getattr(env, "obj_body_id", None)
    if sim is None or obj_body_id is None or obj_name not in obj_body_id:
        return {"available": False, "reason": "missing_sim_or_obj_body"}

    model = sim.model
    root_body_id = int(obj_body_id[obj_name])
    mesh_points = []
    geom_names = []
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if not _is_descendant_body(model, body_id, root_body_id):
            continue
        vertices = _mesh_vertices_for_geom(sim, geom_id)
        if vertices is None:
            continue
        mesh_points.append(vertices)
        try:
            geom_names.append(model.geom_id2name(geom_id))
        except Exception:
            geom_names.append(str(geom_id))
    if not mesh_points:
        return {"available": False, "reason": "no_object_mesh_vertices", "obj_body_id": root_body_id}

    points = np.vstack(mesh_points)
    estimate = _estimate_handle_from_points(points)
    if estimate is None:
        return {
            "available": False,
            "reason": "mesh_handle_estimate_failed",
            "obj_body_id": root_body_id,
            "mesh_point_count": int(points.shape[0]),
            "geom_names": geom_names[:20],
        }

    pc_handle_center = candidate_diagnostics.get("handle_center")
    pc_outward = candidate_diagnostics.get("outward_axis_xy")
    sim_handle_center = np.asarray(estimate["handle_center"], dtype=float)
    sim_outward = np.asarray(estimate["outward_axis_xy"], dtype=float)
    center_delta_xy = None
    outward_dot = None
    outward_angle_deg = None
    if pc_handle_center is not None:
        center_delta_xy = (
            np.asarray(pc_handle_center, dtype=float)[:2] - sim_handle_center[:2]
        ).tolist()
    if pc_outward is not None:
        pc_outward_xy = _normalize_xy(pc_outward)
        if pc_outward_xy is not None:
            outward_dot = float(np.dot(pc_outward_xy, sim_outward))
            outward_angle_deg = _angle_deg_between_xy(pc_outward_xy, sim_outward)

    return {
        "available": True,
        "method": "compiled_mesh_outer_sector",
        "obj_body_id": root_body_id,
        "geom_names": geom_names[:20],
        "sim_handle_center": estimate["handle_center"],
        "sim_outward_axis_xy": estimate["outward_axis_xy"],
        "sim_sector_center_deg": estimate["sector_center_deg"],
        "sim_handle_point_count": estimate["handle_point_count"],
        "sim_mesh_point_count": estimate["point_count"],
        "pointcloud_handle_center": pc_handle_center,
        "pointcloud_outward_axis_xy": pc_outward,
        "pointcloud_minus_sim_handle_center_xy": center_delta_xy,
        "pointcloud_sim_handle_center_distance_xy": None
        if center_delta_xy is None
        else float(np.linalg.norm(center_delta_xy)),
        "pointcloud_vs_sim_outward_dot": outward_dot,
        "pointcloud_vs_sim_outward_angle_deg": outward_angle_deg,
        "candidate_pos": candidate.get("pos"),
    }


def cup_mug_sorting_sim_objects(env):
    metadata = getattr(env, "sorting_metadata", {}) or {}
    obj_body_id = getattr(env, "obj_body_id", {}) or {}
    sim = getattr(env, "sim", None)
    if sim is None:
        return []
    objects = []
    for name, info in metadata.items():
        if name not in obj_body_id:
            continue
        objects.append(
            {
                "name": name,
                "pos": np.asarray(sim.data.body_xpos[obj_body_id[name]], dtype=float).tolist(),
                "has_handle": bool(info.get("has_handle")),
            }
        )
    return objects


def _fixture_anchor(fixture):
    if fixture is None:
        return None
    size = getattr(fixture, "size", None)
    return {
        "pos": np.asarray(getattr(fixture, "pos"), dtype=float).tolist(),
        "rot": float(getattr(fixture, "rot", 0.0) or 0.0),
        "size": None if size is None else np.asarray(size, dtype=float).tolist(),
        "width": None if getattr(fixture, "width", None) is None else float(fixture.width),
        "depth": None if getattr(fixture, "depth", None) is None else float(fixture.depth),
        "height": None if getattr(fixture, "height", None) is None else float(fixture.height),
    }


def cup_mug_sorting_place_anchors(env):
    return {
        "sink": _fixture_anchor(getattr(env, "sink", None)),
        "counter": _fixture_anchor(getattr(env, "counter", None)),
    }



def main():
    parser = argparse.ArgumentParser(description="Run a RoboCasa reach in the live onscreen viewer.")
    parser.add_argument("--task", default="robocasa/CoffeeSetupMug")
    parser.add_argument("--target-label", default="mug")
    parser.add_argument("--candidate-id", type=int, default=None)
    parser.add_argument("--grasp-type", default="any", choices=["top_down", "handle_top_down", "any"],
                        help="Filter candidates by grasp type (default: any)")
    parser.add_argument("--camera-name", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--skip-axis-calibration", action="store_true")
    parser.add_argument("--diagnose-action-space", action="store_true",
                        help="Probe all action indices and exit before perception/grasp execution.")
    parser.add_argument("--diagnostic-pulse", type=float, default=0.1)
    parser.add_argument("--diagnostic-steps", type=int, default=4)
    parser.add_argument("--enable-base-torso-preposition", action="store_true",
                        help="Experimental: move base/torso toward the selected candidate before arm reach.")
    parser.add_argument("--calibrate-base-action-mapping", action="store_true",
                        help="Probe base action indices and print a base-specific XY action mapping.")
    parser.add_argument("--use-base-action-mapping", action="store_true",
                        help="Use calibrated base XY mapping for preposition commands when available.")
    parser.add_argument("--auto-preposition-retry", action="store_true",
                        help="Retry top-down reach once after base preposition if the first reach fails.")
    parser.add_argument("--skip-vision-refresh-after-base-preposition", action="store_true",
                        help="Debug only: keep using the old candidate after base movement.")
    parser.add_argument("--retry-final-error-threshold", type=float, default=0.03)
    parser.add_argument("--base-torso-steps", type=int, default=30)
    parser.add_argument("--base-action-start", type=int, default=DEFAULT_BASE_ACTION_SLICE[0])
    parser.add_argument("--base-action-end", type=int, default=DEFAULT_BASE_ACTION_SLICE[1])
    parser.add_argument("--torso-action-index", type=int, default=DEFAULT_TORSO_ACTION_INDEX,
                        help="Optional torso action index. Default is disabled until action-space diagnostics confirms it.")
    parser.add_argument("--base-desired-xy-standoff", type=float, default=0.18)
    parser.add_argument("--base-xy-deadband", type=float, default=0.06)
    parser.add_argument("--base-preposition-gain", type=float, default=4.0)
    parser.add_argument("--base-action-limit", type=float, default=0.35)
    parser.add_argument("--base-mapping-trust", type=float, default=0.25)
    parser.add_argument("--base-max-world-delta", type=float, default=0.03)
    parser.add_argument("--base-mapping-pulse", type=float, default=0.1)
    parser.add_argument("--base-mapping-steps", type=int, default=4)
    parser.add_argument("--render-sleep-sec", type=float, default=0.005)
    parser.add_argument("--keep-open-sec", type=float, default=0.0)
    args = parser.parse_args()

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
    vision_config["target_labels"] = [args.target_label]
    vision_config["obstacle_labels"] = []
    use_drinkware_classification = (
        env_name in CUP_MUG_SORTING_TASKS
        and args.target_label.strip().lower() in {"cup", "glass cup", "mug"}
    )
    if use_drinkware_classification:
        vision_config["target_labels"] = ["mug", "cup", "glass cup"]
    if "mug" in args.target_label.lower() or "cup" in args.target_label.lower():
        vision_config["candidate_types"] = ["top_down", "handle_top_down"]
        vision_config["handle_labels"] = ["cup", "mug", "glass cup"]
    if args.allow_download:
        vision_config["local_files_only"] = False

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

    print(colored("Initializing onscreen RoboCasa environment...", "yellow"))
    print(json.dumps(config, indent=2))
    sys.stdout.flush()
    env = robosuite.make(**config)
    print(colored("Environment created. Resetting...", "yellow"))
    sys.stdout.flush()

    obs = env.reset()
    print(colored("Reset done. Rendering first frame...", "yellow"))
    sys.stdout.flush()
    env.render()
    time.sleep(max(args.render_sleep_sec, 0.05))

    if args.diagnose_action_space:
        _, diagnostic_summary = diagnose_action_space(
            env,
            obs,
            pulse_magnitude=float(args.diagnostic_pulse),
            pulse_steps=int(args.diagnostic_steps),
            render_sleep_sec=float(args.render_sleep_sec),
        )
        print(json.dumps({"action_space_diagnostics": diagnostic_summary}, indent=2))
        env.close()
        return

    grasp_type_filter = None if args.grasp_type == "any" else args.grasp_type

    def infer_candidate_from_obs(obs_for_vision, reason):
        rgb, depth = get_rgbd(obs_for_vision, camera_name=args.camera_name, sim=env.sim)
        camera_config = resolve_camera_config(env, camera_name=args.camera_name, width=args.width, height=args.height)
        loop = VisionPerceptionLoop(
            vision_config=vision_config,
            camera_config=camera_config,
            conf_thresh=float(thresholds_config["CONF_THRESH"]),
        )
        classification_summary = None
        if use_drinkware_classification:
            all_targets_summary = loop.infer_all_targets_with_diagnostics(rgb, depth)
            sim_objects = cup_mug_sorting_sim_objects(env)
            classified_targets = assign_sim_metadata_to_targets(
                all_targets_summary.get("targets", []),
                sim_objects,
            )
            selected_target, selected_assignment = select_drinkware_target(
                classified_targets,
                requested_label=args.target_label,
            )
            fallback_used = False
            if selected_target is None:
                selected_target, selected_assignment = build_sim_metadata_fallback_target(
                    sim_objects,
                    requested_label=args.target_label,
                )
                fallback_used = selected_target is not None
            classification_summary = {
                "requested_label": args.target_label,
                "sim_objects": sim_objects,
                "assignments": classify_drinkware_targets(classified_targets),
                "selected_assignment": selected_assignment,
                "fallback_used": fallback_used,
            }
            if selected_target is None:
                raise RuntimeError(
                    "Drinkware classification did not find a matching target for "
                    f"{args.target_label!r}. Summary: {classification_summary}"
                )
            payload = build_detected_objects(
                target_label=selected_target["label"],
                target_pos=selected_target["pos"],
                target_conf=selected_target["conf"],
                obstacles=[],
                status="ready",
                conf_thresh=float(thresholds_config["CONF_THRESH"]),
                grasp_candidates=selected_target.get("grasp_candidates", []),
            )
            diagnostics = {"grasp_candidates": selected_target.get("diagnostics", {})}
        else:
            payload, diagnostics = loop.infer_detected_objects_with_diagnostics(rgb, depth)
        candidate_summaries = [
            {
                "id": candidate.get("id"),
                "grasp_type": candidate.get("grasp_type"),
                "score": candidate.get("score"),
                "pos": candidate.get("pos"),
                "closing_axis": candidate.get("orientation", [None])[0],
                "approach_axis": candidate.get("orientation", [None, None, None])[2],
            }
            for candidate in payload.get("grasp_candidates", [])
        ]
        selected_candidate, selected_rejection = choose_reachable_candidate(
            payload,
            candidate_id=args.candidate_id,
            grasp_type=grasp_type_filter,
        )
        if selected_candidate is None:
            raise RuntimeError(
                "No reachable grasp candidate was produced for the current scene. "
                f"Rejected candidate: {selected_rejection}"
            )
        selected_candidate_diagnostics = diagnostics.get("grasp_candidates", {}).get(
            str(selected_candidate.get("id")),
            {},
        )
        summary = {
            "reason": reason,
            "candidate": selected_candidate,
            "payload_status": payload.get("status"),
            "drinkware_classification": classification_summary,
            "candidate_rejection": selected_rejection,
            "perception_candidates": candidate_summaries,
            "grasp_candidate_diagnostics": diagnostics.get("grasp_candidates", {}),
            "selected_candidate_yaw_debug": build_yaw_debug(
                obs_for_vision,
                selected_candidate,
                diagnostics.get("grasp_candidates", {}),
            ),
            "sim_handle_mesh_debug": build_sim_handle_mesh_debug(
                env,
                selected_candidate,
                selected_candidate_diagnostics,
            )
            if selected_candidate.get("grasp_type") == "handle_top_down"
            else None,
        }
        del loop, payload, rgb, depth
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return selected_candidate, selected_rejection, summary

    candidate, candidate_rejection, initial_candidate_summary = infer_candidate_from_obs(obs, reason="initial")
    active_candidate_summary = initial_candidate_summary

    print(json.dumps(initial_candidate_summary, indent=2))

    action_mapping = None
    calibration_summary = None
    current_obs = obs
    if not args.skip_axis_calibration:
        action_mapping, current_obs, calibration_summary = calibrate_position_action_mapping(
            env,
            obs=current_obs,
            render_sleep_sec=float(args.render_sleep_sec),
        )
        print(json.dumps({"axis_calibration": calibration_summary}, indent=2))

    base_action_mapping = None
    base_mapping_summary = None
    if args.calibrate_base_action_mapping or args.use_base_action_mapping:
        current_obs, base_mapping_summary = calibrate_base_action_mapping(
            env,
            obs=current_obs,
            base_slice=(int(args.base_action_start), int(args.base_action_end)),
            pulse_magnitude=float(args.base_mapping_pulse),
            pulse_steps=int(args.base_mapping_steps),
            render_sleep_sec=float(args.render_sleep_sec),
        )
        base_action_mapping = np.asarray(base_mapping_summary["eef_xy_delta_from_base_action"], dtype=float)
        print(json.dumps({"base_action_mapping": base_mapping_summary}, indent=2))

    execution_result = run_live_target_execution(
        env=env,
        current_obs=current_obs,
        candidate=candidate,
        active_candidate_summary=active_candidate_summary,
        action_mapping=action_mapping,
        base_action_mapping=base_action_mapping,
        infer_candidate_from_obs=infer_candidate_from_obs,
        cup_mug_sorting_place_anchors=cup_mug_sorting_place_anchors,
        options=LiveExecutionOptions(
            render_sleep_sec=float(args.render_sleep_sec),
            enable_base_torso_preposition=bool(args.enable_base_torso_preposition),
            auto_preposition_retry=bool(args.auto_preposition_retry),
            skip_vision_refresh_after_base_preposition=bool(args.skip_vision_refresh_after_base_preposition),
            retry_final_error_threshold=float(args.retry_final_error_threshold),
            base_torso_steps=int(args.base_torso_steps),
            base_action_start=int(args.base_action_start),
            base_action_end=int(args.base_action_end),
            torso_action_index=None if args.torso_action_index is None else int(args.torso_action_index),
            base_desired_xy_standoff=float(args.base_desired_xy_standoff),
            base_xy_deadband=float(args.base_xy_deadband),
            base_preposition_gain=float(args.base_preposition_gain),
            base_action_limit=float(args.base_action_limit),
            use_base_action_mapping=bool(args.use_base_action_mapping),
            base_mapping_trust=float(args.base_mapping_trust),
            base_max_world_delta=float(args.base_max_world_delta),
            use_drinkware_classification=bool(use_drinkware_classification),
        ),
    )
    current_obs = execution_result["obs"]
    candidate = execution_result["candidate"]
    candidate_rejection = execution_result["candidate_rejection"]
    active_candidate_summary = execution_result["active_candidate_summary"]
    execution_summary = execution_result["execution_summary"]
    print(json.dumps({"execution_summary": execution_summary}, indent=2))

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

    env.close()


if __name__ == "__main__":
    main()
