import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from runtime.bootstrap import project_root
from rl.contracts import (
    RLConfigError,
    build_selector_contract,
    load_rl_config,
    summarize_rl_config,
    validate_rl_config,
)
from rl.harness import build_selector_observation, ensure_artifact_dirs, run_eval_smoke, run_train_smoke


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUAL_ARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "dual_arm_robocasa.yaml"
SINGLE_ARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "single_arm_robocasa_ppo.yaml"
ROBOSUITE_SINGLE_ARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "single_arm_robosuite_ppo.yaml"


class RLContractTests(unittest.TestCase):
    def test_rl_config_validates_and_summarizes(self):
        config = validate_rl_config(load_rl_config(DUAL_ARM_CONFIG_PATH))
        summary = summarize_rl_config(config)
        self.assertEqual(summary["backend"], "robocasa")
        self.assertEqual(summary["policy_topology"], "shared_dual_arm_policy")
        self.assertEqual(summary["action_dimensions"], 8)
        self.assertIn("robot0_eef_pos", summary["observation_fields"])
        self.assertIn("successful_lift", summary["reward_terms"])

    def test_missing_required_field_fails_validation(self):
        config = load_rl_config(DUAL_ARM_CONFIG_PATH)
        del config["termination"]["max_steps"]
        with self.assertRaises(RLConfigError):
            validate_rl_config(config)

    def test_single_arm_config_validates_and_summarizes(self):
        config = validate_rl_config(load_rl_config(SINGLE_ARM_CONFIG_PATH))
        summary = summarize_rl_config(config)
        self.assertEqual(summary["backend"], "robocasa")
        self.assertEqual(summary["policy_topology"], "single_arm_ppo_policy")
        self.assertEqual(summary["action_dimensions"], 4)
        self.assertIn("robot0_eef_pos", summary["observation_fields"])
        self.assertIn("successful_lift", summary["reward_terms"])

    def test_single_arm_robosuite_config_validates(self):
        config = validate_rl_config(load_rl_config(ROBOSUITE_SINGLE_ARM_CONFIG_PATH))
        summary = summarize_rl_config(config)
        self.assertEqual(summary["backend"], "robosuite")
        self.assertEqual(summary["action_dimensions"], 7)
        self.assertIn("cube_pos", summary["observation_fields"])

    def test_selector_contract_summary_is_available_when_configured(self):
        config = validate_rl_config(load_rl_config(SINGLE_ARM_CONFIG_PATH))
        config["selector"] = {
            "enabled": True,
            "max_candidates": 4,
            "fallback_actions": ["clear", "resense"],
        }

        selector_contract = build_selector_contract(config)
        summary = summarize_rl_config(config)

        self.assertTrue(selector_contract["enabled"])
        self.assertEqual(selector_contract["max_candidates"], 4)
        self.assertEqual(selector_contract["action_meanings"], ["select_candidate", "clear", "resense"])
        self.assertTrue(summary["selector_enabled"])
        self.assertEqual(summary["selector_max_candidates"], 4)

    def test_selector_observation_pads_candidates_and_encodes_fsm_state(self):
        payload = {
            "target": {"label": "cube", "pos": [0.1, 0.2, 0.3], "conf": 0.9, "timestamp": 1.0},
            "grasp_candidates": [
                {
                    "id": 1,
                    "pos": [0.1, 0.2, 0.3],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.04,
                    "score": 0.91,
                    "grasp_type": "top_down",
                },
                {
                    "id": 2,
                    "pos": [0.2, 0.1, 0.35],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.05,
                    "score": 0.72,
                    "grasp_type": "side_grasp",
                },
            ],
            "obstacles": [],
            "status": "ready",
        }

        observation = build_selector_observation(
            payload=payload,
            fsm_state="GRASPING",
            max_candidates=4,
        )

        self.assertEqual(observation["candidate_count"], 2)
        self.assertEqual(observation["fsm_state"], "GRASPING")
        self.assertEqual(len(observation["candidate_features"]), 4)
        self.assertEqual(observation["candidate_features"][0]["id"], 1)
        self.assertEqual(observation["candidate_features"][1]["id"], 2)
        self.assertEqual(observation["candidate_features"][2]["mask"], 0.0)
        self.assertEqual(observation["candidate_features"][3]["mask"], 0.0)

    def test_train_dry_run_writes_checkpoint(self):
        checkpoint_path = run_train_smoke(SINGLE_ARM_CONFIG_PATH, dry_run=True, steps=3)
        self.assertTrue(checkpoint_path.exists())
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["step_count"], 3)
        self.assertEqual(payload["backend"], "robocasa")

    def test_eval_dry_run_requires_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            run_eval_smoke(SINGLE_ARM_CONFIG_PATH, PROJECT_ROOT / "missing.ckpt.json", dry_run=True)

    def test_eval_dry_run_writes_metrics(self):
        checkpoint_path = run_train_smoke(SINGLE_ARM_CONFIG_PATH, dry_run=True, steps=2)
        metrics_path = run_eval_smoke(SINGLE_ARM_CONFIG_PATH, checkpoint_path, dry_run=True)
        self.assertTrue(metrics_path.exists())
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("success_rate", payload)
        self.assertIn("episode_return", payload)
        self.assertIn("episode_length", payload)
        self.assertIn("failure_modes_triggered", payload)

    def test_artifact_dirs_resolve_from_project_root_even_if_cwd_changes(self):
        config = validate_rl_config(load_rl_config(SINGLE_ARM_CONFIG_PATH))
        original_cwd = Path.cwd()
        try:
            os.chdir(PROJECT_ROOT / "scripts")
            dirs = ensure_artifact_dirs(config)
        finally:
            os.chdir(original_cwd)
        self.assertEqual(dirs["root_dir"], project_root() / "outputs" / "rl")
        self.assertEqual(dirs["video_dir"], project_root() / "outputs" / "rl" / "videos")

    def test_train_script_print_summary(self):
        result = subprocess.run(
            [sys.executable, "scripts/train_rl.py", "--config", str(SINGLE_ARM_CONFIG_PATH), "--print-summary"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("single_arm_ppo_policy", result.stdout)


if __name__ == "__main__":
    unittest.main()
