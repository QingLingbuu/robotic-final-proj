import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from runtime.bootstrap import project_root
from rl.contracts import (
    RLConfigError,
    build_selector_contract,
    load_rl_config,
    summarize_rl_config,
    validate_rl_config,
)
import rl.harness as rl_harness
from rl.harness import build_selector_observation, ensure_artifact_dirs, run_eval_smoke, run_train_smoke
from rl.harness import (
    flatten_cup_ordering_observation,
    inspect_cup_ordering_checkpoint_artifact,
    select_cup_ordering_rl_action,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUAL_ARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "dual_arm_robocasa.yaml"
SINGLE_ARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "single_arm_robocasa_ppo.yaml"
ROBOSUITE_SINGLE_ARM_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "single_arm_robosuite_ppo.yaml"
CUP_MUG_ORDERING_CONFIG_PATH = PROJECT_ROOT / "configs" / "rl" / "cup_mug_ordering_robocasa.yaml"


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

    def test_cup_mug_ordering_config_validates_and_summarizes(self):
        config = validate_rl_config(load_rl_config(CUP_MUG_ORDERING_CONFIG_PATH))
        summary = summarize_rl_config(config)
        self.assertEqual(summary["mode"], "cup_ordering")
        self.assertEqual(summary["task_name"], "robocasa/CupMugSorting")
        self.assertEqual(summary["backend"], "robocasa")
        self.assertEqual(summary["max_targets"], 5)
        self.assertEqual(summary["action_space"], "Discrete(5)")
        self.assertEqual(summary["action_space_n"], 5)
        self.assertEqual(summary["scene"]["layout_and_style_ids"], [[1, 1]])
        self.assertEqual(summary["scene"]["num_mugs"], 2)
        self.assertEqual(summary["scene"]["num_cups"], 3)
        self.assertEqual(summary["ordering"]["handled_strategy"], "handle_top_down")
        self.assertEqual(summary["ordering"]["plain_strategy"], "top_down")
        self.assertEqual(summary["ordering"]["handled_zone"], "sink")
        self.assertEqual(summary["ordering"]["plain_zone"], "right_counter")
        self.assertIn("random", summary["policies"]["baselines"])
        self.assertIn("greedy", summary["policies"]["baselines"])
        self.assertEqual(summary["checkpoint_interface"]["artifact_kind"], "cup_ordering_policy_checkpoint")
        self.assertTrue(summary["checkpoint_interface"]["supports_inference"])
        self.assertIn("correct_zone_place", summary["reward_terms"])
        self.assertEqual(summary["episodes_per_policy"], 30)

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

    def test_train_script_prints_cup_ordering_summary(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--dry-run",
                "--print-summary",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("cup_ordering", result.stdout)
        self.assertIn("max_targets", result.stdout)
        self.assertIn("5", result.stdout)
        self.assertIn("Discrete(5)", result.stdout)
        self.assertIn("sanity", result.stdout)

    def test_cup_ordering_eval_dry_run_needs_no_checkpoint(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/eval_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--policy",
                "random",
                "--episodes",
                "2",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("outputs", result.stdout)
        metrics_path = Path(result.stdout.strip())
        self.assertTrue(metrics_path.exists())
        trace_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "random-episodes-2-trace.json"
        self.assertTrue(trace_path.exists())

    def test_cup_ordering_eval_dry_run_trace_report_contains_step_logs(self):
        subprocess.run(
            [
                sys.executable,
                "scripts/eval_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--policy",
                "greedy",
                "--episodes",
                "2",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        trace_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "greedy-episodes-2-trace.json"
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["policy"], "greedy")
        self.assertEqual(payload["episodes"], 2)
        self.assertIn("episode_traces", payload)
        self.assertTrue(payload["episode_traces"])
        self.assertIn("step_logs", payload["episode_traces"][0])

    def test_cup_ordering_eval_print_summary_supports_policy_fields(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/eval_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--policy",
                "greedy",
                "--episodes",
                "2",
                "--seed",
                "7",
                "--dry-run",
                "--print-summary",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("greedy", result.stdout)
        self.assertIn("episodes", result.stdout)
        self.assertIn("trace_report", result.stdout)

    def test_cup_ordering_rl_eval_dry_run_writes_metrics_and_trace(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/eval_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--policy",
                "rl",
                "--episodes",
                "1",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        metrics_path = Path(result.stdout.strip())
        self.assertTrue(metrics_path.exists())
        trace_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "rl-episodes-1-trace.json"
        self.assertTrue(trace_path.exists())
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["policy"], "rl")
        self.assertTrue(payload["episode_traces"])

    def test_cup_ordering_train_dry_run_artifact_reserves_checkpoint_interface(self):
        checkpoint_path = run_train_smoke(CUP_MUG_ORDERING_CONFIG_PATH, dry_run=True, steps=4)
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["artifact_kind"], "cup_ordering_policy_checkpoint")
        self.assertEqual(payload["schema_version"], 1)
        self.assertFalse(payload["supports_inference"])
        self.assertIn("unsupported_reason", payload)

    def test_cup_ordering_checkpoint_artifact_is_inspectable_but_not_inference_capable(self):
        checkpoint_path = run_train_smoke(CUP_MUG_ORDERING_CONFIG_PATH, dry_run=True, steps=4)
        metadata = inspect_cup_ordering_checkpoint_artifact(checkpoint_path)
        self.assertEqual(metadata["policy_mode"], "checkpoint")
        self.assertEqual(metadata["artifact_kind"], "cup_ordering_policy_checkpoint")
        self.assertFalse(metadata["supports_inference"])

    def test_cup_ordering_checkpoint_mode_fails_cleanly_in_eval(self):
        checkpoint_path = run_train_smoke(CUP_MUG_ORDERING_CONFIG_PATH, dry_run=True, steps=4)
        with self.assertRaisesRegex(RuntimeError, "reserved but dry-run eval cannot execute real inference"):
            run_eval_smoke(
                CUP_MUG_ORDERING_CONFIG_PATH,
                checkpoint_path,
                dry_run=True,
                policy_name="rl",
                episodes=1,
            )

    def test_cup_ordering_observation_flattens_to_stable_vector(self):
        observation = {
            "action_mask": [True, False, True, False, False],
            "slots": [
                {
                    "slot_index": 0,
                    "visible": True,
                    "finished": False,
                    "has_handle": True,
                    "type_id": 1,
                    "conf": 0.8,
                    "pos": [0.1, 0.2, 0.3],
                    "candidate_score": 0.4,
                    "reachability": 0.5,
                    "place_zone_id": 1,
                    "retry_count": 0,
                    "valid_action": True,
                }
            ],
            "global": {
                "remaining_count": 1,
                "completed_count": 0,
                "step_index": 2,
                "last_action": 3,
                "last_success": False,
                "last_failure_type": "reach",
            },
        }
        vector = flatten_cup_ordering_observation(observation)
        self.assertEqual(vector.shape, (81,))
        self.assertEqual(vector.dtype.name, "float32")
        self.assertEqual(float(vector[0]), 0.0)
        self.assertEqual(float(vector[-5]), 1.0)
        self.assertEqual(float(vector[-3]), 1.0)

    def test_cup_ordering_zip_checkpoint_uses_model_predict(self):
        checkpoint_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "checkpoints" / "fake-policy.zip"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"fake")

        class FakeModel:
            def __init__(self):
                self.observations = []

            def predict(self, observation, deterministic=True):
                self.observations.append((observation, deterministic))
                return 2, None

        class FakePPO:
            loaded_paths = []
            model = FakeModel()

            @classmethod
            def load(cls, path):
                cls.loaded_paths.append(path)
                return cls.model

        original_ppo = rl_harness.PPO
        original_cache = dict(rl_harness.CUP_ORDERING_CHECKPOINT_MODEL_CACHE)
        try:
            rl_harness.PPO = FakePPO
            rl_harness.CUP_ORDERING_CHECKPOINT_MODEL_CACHE.clear()
            observation = {
                "action_mask": [True, False, True, False, False],
                "slots": [
                    {"slot_index": 0, "candidate_score": 0.4, "conf": 0.7, "reachability": 0.8, "retry_count": 0, "valid_action": True},
                    {"slot_index": 2, "candidate_score": 0.3, "conf": 0.6, "reachability": 0.7, "retry_count": 1, "valid_action": True},
                ],
                "global": {"remaining_count": 2, "completed_count": 0, "step_index": 0},
            }
            action, debug = select_cup_ordering_rl_action(
                observation,
                seed=7,
                step_index=0,
                checkpoint_path=checkpoint_path,
            )
        finally:
            rl_harness.PPO = original_ppo
            rl_harness.CUP_ORDERING_CHECKPOINT_MODEL_CACHE.clear()
            rl_harness.CUP_ORDERING_CHECKPOINT_MODEL_CACHE.update(original_cache)
            checkpoint_path.unlink(missing_ok=True)

        self.assertEqual(action, 2)
        self.assertEqual(debug["selection_mode"], "checkpoint_inference")
        self.assertTrue(debug["predicted_action_valid"])
        self.assertEqual(debug["observation_vector_length"], 81)
        self.assertEqual(FakePPO.loaded_paths, [str(checkpoint_path)])
        self.assertTrue(FakePPO.model.observations[0][1])

    def test_cup_ordering_stub_rl_action_still_works(self):
        observation = {
            "action_mask": [True, False, True, False, False],
            "slots": [
                {"slot_index": 0, "has_handle": True, "candidate_score": 0.4, "conf": 0.7, "reachability": 0.8, "retry_count": 0},
                {"slot_index": 1, "has_handle": True, "candidate_score": 0.9, "conf": 0.9, "reachability": 0.9, "retry_count": 0},
                {"slot_index": 2, "has_handle": False, "candidate_score": 0.3, "conf": 0.6, "reachability": 0.7, "retry_count": 1},
            ],
        }
        action, debug = select_cup_ordering_rl_action(observation, seed=7, step_index=0)
        self.assertEqual(action, 0)
        self.assertEqual(debug["rl_policy_mode"], "stub")
        self.assertEqual(debug["selection_mode"], "handle_first_quality_biased_stub")
        self.assertEqual(debug["checkpoint_interface"]["artifact_kind"], "cup_ordering_stub_policy")

    def test_cup_ordering_stub_prioritizes_handle_targets_before_plain_cups(self):
        observation = {
            "action_mask": [True, True, True, True, False],
            "slots": [
                {"slot_index": 0, "has_handle": True, "candidate_score": 0.35, "conf": 0.6, "reachability": 0.4, "retry_count": 0},
                {"slot_index": 1, "has_handle": True, "candidate_score": 0.30, "conf": 0.6, "reachability": 0.4, "retry_count": 0},
                {"slot_index": 2, "has_handle": False, "candidate_score": 0.95, "conf": 0.9, "reachability": 0.9, "retry_count": 0},
                {"slot_index": 3, "has_handle": False, "candidate_score": 0.90, "conf": 0.9, "reachability": 0.9, "retry_count": 0},
            ],
        }

        action, debug = select_cup_ordering_rl_action(observation, seed=7, step_index=1)

        self.assertEqual(action, 0)
        self.assertEqual(debug["ranked_actions"], [0, 1, 2, 3])
        self.assertEqual(debug["selection_mode"], "handle_first_quality_biased_stub")

    def test_cup_ordering_stub_uses_quality_after_handles_are_finished(self):
        observation = {
            "action_mask": [False, False, True, True, True],
            "slots": [
                {"slot_index": 0, "has_handle": True, "finished": True, "candidate_score": 0.99, "conf": 0.99, "reachability": 0.99, "retry_count": 0},
                {"slot_index": 1, "has_handle": True, "finished": True, "candidate_score": 0.98, "conf": 0.98, "reachability": 0.98, "retry_count": 0},
                {"slot_index": 2, "has_handle": False, "candidate_score": 0.30, "conf": 0.7, "reachability": 0.7, "retry_count": 0},
                {"slot_index": 3, "has_handle": False, "candidate_score": 0.60, "conf": 0.7, "reachability": 0.7, "retry_count": 0},
                {"slot_index": 4, "has_handle": False, "candidate_score": 0.50, "conf": 0.7, "reachability": 0.7, "retry_count": 0},
            ],
        }

        action, debug = select_cup_ordering_rl_action(observation, seed=7, step_index=2)

        self.assertEqual(action, 3)
        self.assertEqual(debug["ranked_actions"], [3, 4, 2])

    def test_cup_ordering_render_dry_run_writes_frame_report(self):
        subprocess.run(
            [
                sys.executable,
                "scripts/eval_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--policy",
                "greedy",
                "--episodes",
                "1",
                "--dry-run",
                "--render",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        trace_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "greedy-episodes-1-trace.json"
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        frame_paths = payload["episode_traces"][0]["frame_paths"]
        self.assertTrue(frame_paths)
        self.assertTrue(Path(frame_paths[0]).exists())
        self.assertIn(payload["episode_traces"][0]["frame_source"], {"real_robocasa_camera", "synthetic_fallback"})
        sidecar_path = Path(payload["episode_traces"][0]["sidecar_path"])
        self.assertTrue(sidecar_path.exists())
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["policy"], "greedy")
        self.assertIn("selected_order", sidecar)
        self.assertIn("object_ids", sidecar)
        self.assertIn("grasp_strategy", sidecar)
        self.assertIn("place_zone", sidecar)
        self.assertIn("success", sidecar)

    def test_cup_ordering_render_video_dry_run_writes_mp4(self):
        video_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "videos" / "greedy-episodes-1-test.mp4"
        if video_path.exists():
            video_path.unlink()
        subprocess.run(
            [
                sys.executable,
                "scripts/eval_rl.py",
                "--config",
                str(CUP_MUG_ORDERING_CONFIG_PATH),
                "--policy",
                "greedy",
                "--episodes",
                "1",
                "--dry-run",
                "--render",
                "--save-video",
                "--video-path",
                str(video_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        trace_path = PROJECT_ROOT / "outputs" / "rl_cup_mug_ordering" / "reports" / "greedy-episodes-1-trace.json"
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        if video_path.exists():
            self.assertEqual(payload["video_output_path"], str(video_path))
        else:
            self.assertEqual(payload["video_skip_reason"], "imageio_not_installed")


if __name__ == "__main__":
    unittest.main()
