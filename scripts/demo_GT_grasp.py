"""Demo a GT-driven grasp pipeline with the same scenario interface as demo_vision_grasp."""

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
    execute_grasp,
    execute_single_grasp_transfer,
    get_primary_object_pos,
    home_arms,
)
from arm.env_wrapper import RobosuiteEnvWrapper
from main import build_single_grasp_pos
from scripts.demo_vision_grasp import build_demo_env_and_labels
from scripts.grasp_demo_common import build_gt_summary, resolve_gt_grasp_target


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def run_demo(
    seed=None,
    place_test=False,
    allow_download=False,
    render=False,
    scenario="cube",
):
    del allow_download  # CLI parity with demo_vision_grasp.py; unused for GT mode.

    static_camera_config = load_yaml(PROJECT_ROOT / "configs" / "camera.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    env_config, _, _ = build_demo_env_and_labels(scenario)

    print("Stage 1/4: create robosuite environment")
    print(json.dumps({"scenario": scenario, "env_config": env_config}, indent=2))
    env = RobosuiteEnvWrapper(config=env_config, camera_config=static_camera_config)
    if seed is not None:
        env.reset(seed=seed)

    print("Stage 2/4: resolve GT object pose and derive GT grasp target")
    gt_summary = build_gt_summary(env, scenario)
    gt_grasp_target = resolve_gt_grasp_target(
        gt_summary=gt_summary,
        penetration_offset=float(vision_config.get("top_down_penetration_offset", 0.01)),
    )
    print(
        json.dumps(
            {
                "ground_truth": gt_summary,
                "gt_grasp_target": gt_grasp_target.tolist(),
            },
            indent=2,
        )
    )

    print("Stage 3/4: convert GT target into a grasp command")
    rgb_image, depth_image, proprio = env.get_observation()
    del rgb_image, depth_image
    final_grasp_pos = build_single_grasp_pos(
        env=env,
        grasp_target_pos=gt_grasp_target,
        source="ground_truth",
        vision_config=vision_config,
    )
    print(
        json.dumps(
            {
                "grasp_source": "ground_truth",
                "raw_target_pos": gt_grasp_target.tolist(),
                "final_grasp_pos": final_grasp_pos.tolist(),
                "robot0_eef_pos": np.asarray(proprio["robot0_eef_pos"], dtype=float).tolist(),
            },
            indent=2,
        )
    )

    print("Stage 4/4: execute grasp")
    object_pos_before = get_primary_object_pos(env.obs)
    if place_test:
        place_target = (
            np.asarray(object_pos_before, dtype=float)
            if object_pos_before is not None
            else final_grasp_pos
        )
        grasp_success = execute_single_grasp_transfer(
            env,
            final_grasp_pos,
            place_target,
            arm_idx=0,
        )
    else:
        grasp_success = execute_grasp(env, final_grasp_pos, arm_idx=0)
    object_pos_after = get_primary_object_pos(env.obs)

    print(
        json.dumps(
            {
                "grasp_success": bool(grasp_success),
                "object_pos_before": None
                if object_pos_before is None
                else np.asarray(object_pos_before, dtype=float).tolist(),
                "object_pos_after": None
                if object_pos_after is None
                else np.asarray(object_pos_after, dtype=float).tolist(),
            },
            indent=2,
        )
    )

    if render:
        env.render()

    home_success = home_arms(env)
    print(json.dumps({"home_arms_success": bool(home_success)}, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Run the GT-driven grasp pipeline once using robosuite ground-truth target coordinates."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic reset seed.")
    parser.add_argument(
        "--place-test",
        action="store_true",
        help="Use grasp-then-place instead of grasp-only execution.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Ignored in GT mode; kept for CLI parity with demo_vision_grasp.py.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render one frame at the end of the demo.",
    )
    parser.add_argument(
        "--scenario",
        default="cube",
        help="Demo object scenario: cube, can, milk, bread, cereal",
    )
    args = parser.parse_args()
    run_demo(
        seed=args.seed,
        place_test=args.place_test,
        allow_download=args.allow_download,
        render=args.render,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
