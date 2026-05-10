"""Smoke test for the CupMugSorting RoboCasa environment."""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import robocasa  # noqa: F401 - import registers RoboCasa envs into robosuite.make
import robosuite
from termcolor import colored

from scripts.demo_robocasa_reach_onscreen import (
    build_env_config,
    get_rgbd,
    load_yaml,
    resolve_camera_config,
)
from vision.perception_loop import VisionPerceptionLoop
from vision.frame_io import save_rgb_frame
from vision.sorting_policy import classify_drinkware_targets

DEFAULT_LEFT_SINK_LAYOUT_ID = 1
CUP_MUG_SORTING_TASKS = {"CupMugSorting", "CupMugSortingClean"}
RANDOM_CUP_MUG_SORTING_TASKS = {"CupMugSortingRandom"}


def _load_task_config(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def main():
    parser = argparse.ArgumentParser(
        description="Create CupMugSorting and classify visible cups / mugs."
    )
    parser.add_argument("--task", default="robocasa/CupMugSortingClean")
    parser.add_argument("--task-config", default=str(PROJECT_ROOT / "configs" / "tasks" / "cup_mug_sorting.yaml"))
    parser.add_argument("--camera-name", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-frame", default=None, help="Optional path to save the current RGB camera frame.")
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

    task_config = _load_task_config(args.task_config)
    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    vision_config["target_labels"] = list(task_config.get("labels", ["mug", "cup", "glass cup"]))
    vision_config["obstacle_labels"] = []
    vision_config["candidate_types"] = ["top_down", "handle_top_down"]
    vision_config["handle_labels"] = ["cup", "mug", "glass cup"]
    if args.allow_download:
        vision_config["local_files_only"] = False

    env_config = build_env_config(
        task_name=args.task,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        layout=layout_id,
        style=style_id,
        seed=seed,
        layout_and_style_ids=layout_and_style_ids,
    )

    print(colored("Initializing CupMugSorting environment...", "yellow"))
    print(json.dumps(env_config, indent=2))
    sys.stdout.flush()
    env = robosuite.make(**env_config)

    try:
        obs = env.reset()
        env.render()
        time.sleep(max(args.render_sleep_sec, 0.05))

        rgb, depth = get_rgbd(obs, camera_name=args.camera_name, sim=env.sim)
        saved_frame = None
        if args.save_frame:
            saved_frame = save_rgb_frame(rgb, args.save_frame)
        camera_config = resolve_camera_config(
            env,
            camera_name=args.camera_name,
            width=args.width,
            height=args.height,
        )
        loop = VisionPerceptionLoop(
            vision_config=vision_config,
            camera_config=camera_config,
            conf_thresh=float(thresholds_config["CONF_THRESH"]),
        )
        perception_summary = loop.infer_all_targets_with_diagnostics(rgb, depth)
        assignments = classify_drinkware_targets(perception_summary["targets"])
        summary = {
            "task": args.task,
            "scene_id": task_config.get("scene_id"),
            "status": perception_summary["status"],
            "saved_frame": saved_frame,
            "target_count": len(perception_summary["targets"]),
            "assignments": assignments,
            "targets": perception_summary["targets"],
        }
        print(json.dumps({"cup_mug_sorting_summary": summary}, indent=2))

        keep_open_sec = float(args.keep_open_sec)
        if keep_open_sec > 0.0:
            end_time = time.time() + keep_open_sec
            while time.time() < end_time:
                env.render()
                time.sleep(0.02)
    finally:
        env.close()


if __name__ == "__main__":
    main()
