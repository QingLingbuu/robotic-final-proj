"""Demo the full vision-driven grasp pipeline with explicit stage logging."""

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
from fsm.perception_cycle import consume_perception_queue
from fsm.state_machine import State, TaskStateMachine
from main import build_single_grasp_pos, get_corrected_vision_target
from runtime.perception_queue import create_perception_queue
from vision.perception_loop import VisionPerceptionLoop


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def summarize_payload(payload):
    """Return a compact JSON-serializable summary for console output."""
    target = payload.get("target", {})
    candidates = payload.get("grasp_candidates", [])
    return {
        "status": payload.get("status"),
        "target": {
            "label": target.get("label"),
            "conf": target.get("conf"),
            "pos": target.get("pos"),
        },
        "obstacle_count": len(payload.get("obstacles", [])),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "id": candidate["id"],
                "grasp_type": candidate["grasp_type"],
                "score": candidate["score"],
                "gripper_width": candidate["gripper_width"],
                "pos": candidate["pos"],
            }
            for candidate in candidates
        ],
    }


def build_demo_env_and_labels(scenario):
    normalized = str(scenario).strip().lower()
    if normalized == "cube":
        return (
            {
                "env_name": "Lift",
                "robots": ["Panda", "Panda"],
                "env_configuration": "parallel",
            },
            ["cube"],
            [],
        )

    if normalized in {"can", "milk", "bread", "cereal"}:
        env_name_by_object = {
            "milk": "PickPlaceMilk",
            "bread": "PickPlaceBread",
            "cereal": "PickPlaceCereal",
            "can": "PickPlaceCan",
        }
        return (
            {
                "env_name": env_name_by_object[normalized],
                "robots": "Panda",
                "env_configuration": "default",
            },
            [normalized],
            [],
        )

    raise ValueError(
        "Unsupported demo scenario. Use one of: cube, can, milk, bread, cereal."
    )


def run_demo(
    seed=None,
    place_test=False,
    allow_download=False,
    render=False,
    scenario="cube",
):
    static_camera_config = load_yaml(PROJECT_ROOT / "configs" / "camera.yaml")
    thresholds_config = load_yaml(PROJECT_ROOT / "configs" / "thresholds.yaml")
    vision_config = load_yaml(PROJECT_ROOT / "configs" / "vision.yaml")
    conf_thresh = float(thresholds_config["CONF_THRESH"])
    env_config, target_labels, obstacle_labels = build_demo_env_and_labels(scenario)

    if allow_download:
        vision_config["local_files_only"] = False

    # Keep labels scene-specific. Generic labels such as "object" attract large
    # background boxes that poison 3D candidate geometry.
    vision_config["target_labels"] = target_labels
    vision_config["obstacle_labels"] = obstacle_labels

    print("Stage 1/6: create robosuite environment")
    print(json.dumps({"scenario": scenario, "env_config": env_config}, indent=2))
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
    print(
        json.dumps(
            {
                "camera_name": env.camera_name,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            },
            indent=2,
        )
    )

    print("Stage 2/6: initialize GroundingDINO vision pipeline")
    try:
        perception_loop = VisionPerceptionLoop(
            vision_config=vision_config,
            camera_config=camera_config,
            conf_thresh=conf_thresh,
        )
    except Exception as exc:
        print("Vision initialization failed.")
        print(
            "If the model cache is missing, run "
            "`python scripts/setup/cache_grounding_dino.py` first."
        )
        raise RuntimeError("GroundingDINO is not ready in the active environment.") from exc

    perception_queue = create_perception_queue()
    fsm = TaskStateMachine()

    print("Stage 3/6: capture one RGB-D observation and run detection")
    rgb_image, depth_image, proprio = env.get_observation()
    raw_detections = perception_loop.detector.detect(
        rgb_image,
        perception_loop.target_labels + perception_loop.obstacle_labels,
    )
    print(
        json.dumps(
            {
                "raw_detections": [
                    {
                        "label": detection.label,
                        "score": float(detection.score),
                        "box_xyxy": [float(value) for value in detection.box_xyxy],
                    }
                    for detection in raw_detections
                ]
            },
            indent=2,
        )
    )
    payload = perception_loop.publish_from_observation(
        perception_queue=perception_queue,
        rgb_image=rgb_image,
        depth_image=depth_image,
    )
    print(json.dumps(summarize_payload(payload), indent=2))

    print("Stage 4/6: feed payload into FSM planning")
    fsm.transition_to(State.PLANNING)
    payload, cycle_ok = consume_perception_queue(fsm, perception_queue)
    print(
        json.dumps(
            {
                "cycle_ok": bool(cycle_ok),
                "fsm_state": fsm.get_current_state().value,
                "selected_candidate_index": fsm.get_selected_candidate_index(),
            },
            indent=2,
        )
    )
    if not cycle_ok:
        raise RuntimeError(
            "Vision payload did not pass planning validation. "
            "Check confidence threshold, detections, and grasp candidate generation."
        )

    print("Stage 5/6: convert selected visual candidate into a grasp command")
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
    print(
        json.dumps(
            {
                "grasp_source": grasp_source,
                "raw_target_pos": grasp_target_pos.tolist(),
                "final_grasp_pos": final_grasp_pos.tolist(),
                "robot0_eef_pos": np.asarray(proprio["robot0_eef_pos"], dtype=float).tolist(),
            },
            indent=2,
        )
    )

    print("Stage 6/6: execute grasp")
    object_pos_before = get_primary_object_pos(env.obs)
    if place_test:
        place_target = np.asarray(object_pos_before, dtype=float) if object_pos_before is not None else final_grasp_pos
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
                "final_fsm_state": fsm.get_current_state().value,
            },
            indent=2,
        )
    )

    home_success = home_arms(env)
    print(json.dumps({"home_arms_success": bool(home_success)}, indent=2))

    if render:
        env.render()


def main():
    parser = argparse.ArgumentParser(
        description="Run the full vision-driven grasp pipeline once using GroundingDINO."
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
        help="Allow Hugging Face download if the GroundingDINO cache is missing.",
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
