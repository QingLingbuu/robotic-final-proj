import time
import unittest
from queue import Empty

from fsm.state_machine import State, TaskStateMachine
from runtime.perception_queue import (
    PERCEPTION_QUEUE_NAME,
    PERCEPTION_QUEUE_TIMEOUT_SEC,
    create_perception_queue,
    publish_detected_objects,
    read_detected_objects,
)
from runtime.run_logger import build_run_log, infer_failure_mode, infer_failure_modes_triggered
from vision.detected_objects import MAX_OBSTACLES, build_detected_objects, validate_detected_objects


class DetectedObjectsContractTests(unittest.TestCase):
    def test_build_detected_objects_matches_schema(self):
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.1, 0.2, 0.3],
            target_conf=0.95,
            obstacles=[{"label": "cup", "pos": [1, 2, 3], "id": 1, "conf": 0.4}],
        )

        self.assertTrue(validate_detected_objects(payload))
        self.assertEqual(set(payload.keys()), {"target", "obstacles", "status"})
        self.assertEqual(set(payload["target"].keys()), {"label", "pos", "conf", "timestamp"})
        self.assertEqual(payload["status"], "ready")

    def test_detected_objects_uses_monotonic_timestamp_and_caps_obstacles(self):
        before = time.monotonic()
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.1, 0.2, 0.3],
            target_conf=0.99,
            obstacles=[
                {"label": f"obs-{idx}", "pos": [idx, idx, idx], "id": idx, "conf": float(idx)}
                for idx in range(MAX_OBSTACLES + 5)
            ],
        )
        after = time.monotonic()

        self.assertGreaterEqual(payload["target"]["timestamp"], before)
        self.assertLessEqual(payload["target"]["timestamp"], after)
        self.assertEqual(len(payload["obstacles"]), MAX_OBSTACLES)
        kept_ids = [entry["id"] for entry in payload["obstacles"]]
        self.assertEqual(kept_ids, list(range(MAX_OBSTACLES + 4, 4, -1)))

    def test_low_confidence_target_forces_error_status(self):
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.1, 0.2, 0.3],
            target_conf=0.2,
            conf_thresh=0.5,
        )
        self.assertEqual(payload["status"], "error")

    def test_invalid_payload_fails_validation(self):
        invalid_payload = {
            "target": {"label": "cube", "pos": [0, 0, 0], "conf": 0.9},
            "obstacles": [],
            "status": "ready",
        }
        self.assertFalse(validate_detected_objects(invalid_payload))


class PerceptionQueueContractTests(unittest.TestCase):
    def test_queue_name_and_roundtrip(self):
        self.assertEqual(PERCEPTION_QUEUE_NAME, "perception_queue")

        queue = create_perception_queue()
        payload = build_detected_objects("cube", [0.0, 0.0, 0.0], 0.9)
        publish_detected_objects(queue, payload)

        roundtrip = read_detected_objects(queue)
        self.assertEqual(roundtrip, payload)

    def test_stale_read_raises_empty_with_timeout_contract(self):
        self.assertEqual(PERCEPTION_QUEUE_TIMEOUT_SEC, 0.2)
        queue = create_perception_queue()
        start = time.monotonic()
        with self.assertRaises(Empty):
            read_detected_objects(queue)
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.15)


class FSMContractTests(unittest.TestCase):
    def test_state_enum_matches_required_contract(self):
        expected = {
            "IDLE",
            "PLANNING",
            "CLEARING",
            "GRASPING",
            "VERIFYING",
            "RETRY_SENSING",
            "RETRY_PUSH",
            "RETRY_GRASP",
            "RESET",
            "EMERGENCY",
            "SUCCESS",
            "FAILED",
        }
        self.assertEqual({state.value for state in State}, expected)

    def test_failure_counters_map_to_n1_n2_n3(self):
        fsm = TaskStateMachine(max_planning_retries=3, max_push_retries=2, max_grasp_retries=2)
        fsm.handle_perception_timeout()
        fsm.handle_push_blocked()
        fsm.handle_empty_grasp()

        self.assertEqual(fsm.get_failure_counts(), {"n1": 1, "n2": 1, "n3": 1})

    def test_retry_budget_exhaustion_transitions_to_failed(self):
        fsm = TaskStateMachine(max_planning_retries=1, max_push_retries=1, max_grasp_retries=1)
        self.assertEqual(fsm.handle_perception_timeout(), State.FAILED)
        fsm.reset()
        self.assertEqual(fsm.handle_push_blocked(), State.FAILED)
        fsm.reset()
        self.assertEqual(fsm.handle_empty_grasp(), State.FAILED)

    def test_collision_goes_to_emergency(self):
        fsm = TaskStateMachine()
        self.assertEqual(fsm.handle_collision(), State.EMERGENCY)


class RunLogContractTests(unittest.TestCase):
    def test_build_run_log_preserves_required_fields(self):
        counts = {"n1": 1, "n2": 0, "n3": 0}
        run_log = build_run_log(
            run_id="run-001",
            config_version="v1",
            scene_config={"seed": 42, "object_count": 1, "scene_id": "scene-01"},
            success=False,
            counts=counts,
            duration_sec=1.25,
            context="contract-test",
            execution_summary={"push_failure_mode": "perception_error"},
        )

        required = {
            "run_id",
            "commit_hash",
            "config_version",
            "scene_config",
            "timestamp",
            "success",
            "failure_mode",
            "failure_modes_triggered",
            "counts",
            "duration_sec",
            "context",
            "execution_summary",
        }
        self.assertEqual(set(run_log.keys()), required)
        self.assertEqual(run_log["failure_mode"], "perception_error")
        self.assertIn("perception_error", run_log["failure_modes_triggered"])

    def test_failure_mode_priority_and_success_behavior(self):
        self.assertEqual(infer_failure_mode({"n1": 0, "n2": 1, "n3": 1}), "execution_drift")
        self.assertEqual(infer_failure_mode({"n1": 0, "n2": 0, "n3": 1}), "physical_slip")
        self.assertEqual(infer_failure_mode({"n1": 1, "n2": 0, "n3": 0}), "perception_error")

        run_log = build_run_log(
            run_id="run-002",
            config_version="v1",
            scene_config={"seed": 0, "object_count": 1, "scene_id": "scene-02"},
            success=True,
            counts={"n1": 0, "n2": 0, "n3": 0},
            duration_sec=0.5,
            context="success-case",
        )
        self.assertEqual(run_log["failure_mode"], "success")
        self.assertEqual(run_log["failure_modes_triggered"], [])

    def test_failure_modes_triggered_collects_all_unique_modes(self):
        modes = infer_failure_modes_triggered(
            {"n1": 1, "n2": 1, "n3": 0},
            execution_summary={"grasp_failure_mode": "physical_slip"},
        )
        self.assertEqual(modes, ["perception_error", "execution_drift", "physical_slip"])


if __name__ == "__main__":
    unittest.main()
