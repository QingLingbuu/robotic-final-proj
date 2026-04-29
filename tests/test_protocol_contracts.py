import time
import numpy as np
import unittest
from unittest.mock import patch
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
from fsm.perception_cycle import consume_perception_queue
from vision.detected_objects import MAX_OBSTACLES, build_detected_objects, validate_detected_objects
from vision.detector import Detection
from vision.perception_loop import VisionPerceptionLoop


class DetectedObjectsContractTests(unittest.TestCase):
    def test_build_detected_objects_matches_schema(self):
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.1, 0.2, 0.3],
            target_conf=0.95,
            obstacles=[{"label": "cup", "pos": [1, 2, 3], "id": 1, "conf": 0.4}],
        )

        self.assertTrue(validate_detected_objects(payload))
        self.assertEqual(set(payload.keys()), {"target", "grasp_candidates", "obstacles", "status"})
        self.assertEqual(set(payload["target"].keys()), {"label", "pos", "conf", "timestamp"})
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["grasp_candidates"], [])

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

    def test_build_detected_objects_preserves_grasp_candidates(self):
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.1, 0.2, 0.3],
            target_conf=0.95,
            grasp_candidates=[
                {
                    "id": 1,
                    "pos": [0.1, 0.2, 0.35],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.04,
                    "score": 0.9,
                    "grasp_type": "top_down",
                }
            ],
        )

        self.assertTrue(validate_detected_objects(payload))
        self.assertIn("grasp_candidates", payload)
        self.assertEqual(len(payload["grasp_candidates"]), 1)
        self.assertEqual(payload["grasp_candidates"][0]["grasp_type"], "top_down")

    def test_missing_grasp_candidates_fails_validation(self):
        payload = {
            "target": {"label": "cube", "pos": [0, 0, 0], "conf": 0.9, "timestamp": time.monotonic()},
            "obstacles": [],
            "status": "ready",
        }
        self.assertFalse(validate_detected_objects(payload))

    def test_invalid_payload_fails_validation(self):
        invalid_payload = {
            "target": {"label": "cube", "pos": [0, 0, 0], "conf": 0.9},
            "obstacles": [],
            "status": "ready",
        }
        self.assertFalse(validate_detected_objects(invalid_payload))


class VisionPerceptionContractTests(unittest.TestCase):
    def test_infer_detected_objects_adds_handle_grasp_for_cup_like_target(self):
        class _Detector:
            def detect(self, rgb_image, labels):
                return [Detection(label="cup", score=0.95, box_xyxy=[0.0, 0.0, 4.0, 4.0])]

        with patch("vision.perception_loop.build_detector", return_value=_Detector()):
            loop = VisionPerceptionLoop(
                vision_config={
                    "target_labels": ["cup"],
                    "obstacle_labels": [],
                    "candidate_types": ["top_down", "handle_grasp"],
                    "depth_sample_radius": 0,
                    "max_obstacles": MAX_OBSTACLES,
                    "default_gripper_width": 0.04,
                    "handle_gripper_width": 0.03,
                    "handle_labels": ["cup", "mug"],
                    "handle_min_points": 2,
                    "handle_radial_ratio": 0.2,
                    "handle_radial_offset": 0.01,
                    "handle_height_quantiles": [0.2, 0.9],
                },
                camera_config={
                    "fx": 10.0,
                    "fy": 10.0,
                    "cx": 2.0,
                    "cy": 2.0,
                    "T_world_cam": {
                        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "translation": [0.0, 0.0, 0.0],
                    },
                },
                conf_thresh=0.1,
            )

        depth = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.6, 1.6],
                [1.0, 1.0, 1.0, 1.6, 1.6],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        )

        payload = loop.infer_detected_objects(
            rgb_image=np.zeros((5, 5, 3), dtype=np.uint8),
            depth_image=depth,
        )

        grasp_types = [candidate["grasp_type"] for candidate in payload["grasp_candidates"]]
        self.assertIn("top_down", grasp_types)
        self.assertIn("handle_grasp", grasp_types)
        handle_candidate = next(candidate for candidate in payload["grasp_candidates"] if candidate["grasp_type"] == "handle_grasp")
        self.assertGreater(handle_candidate["pos"][0], payload["target"]["pos"][0])
        self.assertGreater(handle_candidate["gripper_width"], 0.1)
        self.assertLess(handle_candidate["orientation"][2][0], -0.7)
        self.assertGreater(abs(handle_candidate["orientation"][1][2]), 0.7)

    def test_handle_grasp_score_increases_with_stronger_protrusion(self):
        loop = VisionPerceptionLoop.__new__(VisionPerceptionLoop)
        loop.handle_min_points = 2
        loop.handle_radial_ratio = 0.15
        loop.handle_radial_offset = 0.01
        loop.handle_height_quantiles = (0.0, 1.0)
        loop.handle_gripper_width = 0.03
        loop.handle_width_margin = 0.01
        loop.handle_width_min = 0.02

        target = {"label": "cup", "pos": [0.0, 0.0, 0.0], "conf": 0.9}
        weak_point_cloud = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.00, 0.02, 0.1],
                [0.00, -0.02, 0.2],
                [0.00, 0.01, 0.3],
                [0.12, 0.00, 0.1],
                [0.12, 0.01, 0.2],
            ],
            dtype=float,
        )
        strong_point_cloud = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.00, 0.02, 0.1],
                [0.00, -0.02, 0.2],
                [0.00, 0.01, 0.3],
                [0.24, 0.00, 0.1],
                [0.24, 0.01, 0.2],
            ],
            dtype=float,
        )

        weak_candidate = loop._estimate_handle_candidate(target, weak_point_cloud, 1)
        strong_candidate = loop._estimate_handle_candidate(target, strong_point_cloud, 1)

        if weak_candidate is None or strong_candidate is None:
            self.fail("expected both heuristic handle candidates to be generated")
        self.assertGreater(strong_candidate["score"], weak_candidate["score"])

    def test_infer_detected_objects_normalizes_detector_label_case(self):
        class _Detector:
            def detect(self, rgb_image, labels):
                return [Detection(label="Cube", score=0.95, box_xyxy=[0.0, 0.0, 0.0, 0.0])]

        with patch("vision.perception_loop.build_detector", return_value=_Detector()):
            loop = VisionPerceptionLoop(
                vision_config={
                    "target_labels": ["cube"],
                    "obstacle_labels": [],
                    "candidate_types": ["top_down"],
                    "depth_sample_radius": 0,
                    "max_obstacles": MAX_OBSTACLES,
                    "default_gripper_width": 0.04,
                },
                camera_config={
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "T_world_cam": {
                        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "translation": [0.0, 0.0, 0.0],
                    },
                },
                conf_thresh=0.1,
            )

        payload = loop.infer_detected_objects(
            rgb_image=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_image=np.ones((2, 2), dtype=float),
        )

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["target"]["label"], "cube")
        self.assertGreaterEqual(len(payload["grasp_candidates"]), 1)

    def test_top_down_candidate_uses_top_surface_geometry_not_raw_target_pos(self):
        class _Detector:
            def detect(self, rgb_image, labels):
                return [Detection(label="cube", score=0.95, box_xyxy=[0.0, 0.0, 4.0, 4.0])]

        with patch("vision.perception_loop.build_detector", return_value=_Detector()):
            loop = VisionPerceptionLoop(
                vision_config={
                    "target_labels": ["cube"],
                    "obstacle_labels": [],
                    "candidate_types": ["top_down"],
                    "depth_sample_radius": 0,
                    "max_obstacles": MAX_OBSTACLES,
                    "default_gripper_width": 0.04,
                    "top_down_surface_quantile": 0.8,
                    "top_down_penetration_offset": 0.02,
                },
                camera_config={
                    "fx": 10.0,
                    "fy": 10.0,
                    "cx": 2.0,
                    "cy": 2.0,
                    "T_world_cam": {
                        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "translation": [0.0, 0.0, 0.0],
                    },
                },
                conf_thresh=0.1,
            )

        depth = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.2, 1.2, 1.0],
                [1.0, 1.2, 1.4, 1.4, 1.0],
                [1.0, 1.2, 1.4, 1.4, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        )

        payload = loop.infer_detected_objects(
            rgb_image=np.zeros((5, 5, 3), dtype=np.uint8),
            depth_image=depth,
        )

        top_candidate = next(candidate for candidate in payload["grasp_candidates"] if candidate["grasp_type"] == "top_down")
        self.assertNotEqual(top_candidate["pos"], payload["target"]["pos"])
        self.assertGreater(top_candidate["pos"][2], 1.2)
        self.assertLess(top_candidate["pos"][2], 1.4)
        self.assertAlmostEqual(top_candidate["orientation"][2][2], -1.0, places=6)
        self.assertGreater(top_candidate["gripper_width"], 0.04)
        self.assertLess(top_candidate["score"], payload["target"]["conf"])

    def test_infer_detected_objects_emits_grasp_candidates_for_target(self):
        class _Detector:
            def detect(self, rgb_image, labels):
                return [Detection(label="cube", score=0.95, box_xyxy=[0.0, 0.0, 0.0, 0.0])]

        with patch("vision.perception_loop.build_detector", return_value=_Detector()):
            loop = VisionPerceptionLoop(
                vision_config={
                    "target_labels": ["cube"],
                    "obstacle_labels": [],
                    "candidate_types": ["top_down"],
                    "depth_sample_radius": 0,
                    "max_obstacles": MAX_OBSTACLES,
                    "default_gripper_width": 0.04,
                },
                camera_config={
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "T_world_cam": {
                        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "translation": [0.0, 0.0, 0.0],
                    },
                },
                conf_thresh=0.1,
            )

        payload = loop.infer_detected_objects(
            rgb_image=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_image=np.ones((2, 2), dtype=float),
        )

        self.assertEqual(payload["status"], "ready")
        self.assertGreaterEqual(len(payload["grasp_candidates"]), 1)
        candidate = payload["grasp_candidates"][0]
        self.assertEqual(candidate["grasp_type"], "top_down")
        self.assertLess(candidate["pos"][2], payload["target"]["pos"][2])
        self.assertAlmostEqual(candidate["orientation"][2][2], -1.0, places=6)



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

    def test_ready_payload_selects_first_grasp_candidate(self):
        queue = create_perception_queue()
        fsm = TaskStateMachine()
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.0, 0.0, 0.0],
            target_conf=0.9,
            grasp_candidates=[
                {
                    "id": 1,
                    "pos": [0.1, 0.0, 0.2],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.04,
                    "score": 0.9,
                    "grasp_type": "top_down",
                },
                {
                    "id": 2,
                    "pos": [0.2, 0.0, 0.2],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.05,
                    "score": 0.8,
                    "grasp_type": "side_grasp",
                },
            ],
        )
        publish_detected_objects(queue, payload)

        latest_detection, cycle_ok = consume_perception_queue(fsm, queue)

        self.assertTrue(cycle_ok)
        self.assertEqual(latest_detection, payload)
        self.assertEqual(fsm.get_current_state(), State.GRASPING)
        self.assertEqual(fsm.get_selected_candidate_index(), 0)
        selected_candidate = fsm.get_selected_candidate()
        if selected_candidate is None:
            self.fail("selected candidate should not be None after planning")
        self.assertEqual(selected_candidate["id"], 1)

    def test_empty_grasp_advances_to_next_candidate_before_failing(self):
        fsm = TaskStateMachine(max_grasp_retries=2)
        fsm.set_planning_candidates(
            [
                {
                    "id": 1,
                    "pos": [0.1, 0.0, 0.2],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.04,
                    "score": 0.9,
                    "grasp_type": "top_down",
                },
                {
                    "id": 2,
                    "pos": [0.2, 0.0, 0.2],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gripper_width": 0.05,
                    "score": 0.8,
                    "grasp_type": "side_grasp",
                },
            ]
        )

        next_state = fsm.handle_empty_grasp()

        self.assertEqual(next_state, State.RETRY_GRASP)
        self.assertEqual(fsm.get_current_state(), State.RETRY_GRASP)
        self.assertEqual(fsm.get_selected_candidate_index(), 1)
        selected_candidate = fsm.get_selected_candidate()
        if selected_candidate is None:
            self.fail("selected candidate should advance to the next option")
        self.assertEqual(selected_candidate["id"], 2)

    def test_ready_payload_without_candidates_routes_to_retry_sensing(self):
        queue = create_perception_queue()
        fsm = TaskStateMachine()
        payload = build_detected_objects(
            target_label="cube",
            target_pos=[0.0, 0.0, 0.0],
            target_conf=0.9,
            grasp_candidates=[],
        )
        publish_detected_objects(queue, payload)

        latest_detection, cycle_ok = consume_perception_queue(fsm, queue)

        self.assertFalse(cycle_ok)
        self.assertEqual(latest_detection, payload)
        self.assertEqual(fsm.get_current_state(), State.RETRY_SENSING)

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
            "candidate_count",
            "selected_candidate_id",
            "selected_candidate_score",
            "failure_stage",
            "execution_summary",
        }
        self.assertEqual(set(run_log.keys()), required)
        self.assertEqual(run_log["failure_mode"], "perception_error")
        self.assertIn("perception_error", run_log["failure_modes_triggered"])
        self.assertEqual(run_log["candidate_count"], 0)
        self.assertIsNone(run_log["selected_candidate_id"])
        self.assertIsNone(run_log["selected_candidate_score"])
        self.assertIsNone(run_log["failure_stage"])

    def test_build_run_log_extracts_candidate_decision_metadata(self):
        counts = {"n1": 0, "n2": 1, "n3": 0}
        execution_summary = {
            "selected_candidate": {"id": 7, "score": 0.83, "grasp_type": "top_down"},
            "dual_arm_execution_diagnostics": {
                "latest_attempt": {"failed_stage": "final_descent"}
            },
        }

        run_log = build_run_log(
            run_id="run-003",
            config_version="v1",
            scene_config={"seed": 1, "object_count": 2, "scene_id": "scene-03"},
            success=False,
            counts=counts,
            duration_sec=2.5,
            context="candidate-metadata",
            execution_summary=execution_summary,
        )

        self.assertEqual(run_log["candidate_count"], 1)
        self.assertEqual(run_log["selected_candidate_id"], 7)
        self.assertEqual(run_log["selected_candidate_score"], 0.83)
        self.assertEqual(run_log["failure_stage"], "final_descent")

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
