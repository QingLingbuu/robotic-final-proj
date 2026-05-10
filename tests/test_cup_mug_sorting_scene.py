import unittest
from unittest.mock import patch

import numpy as np

from vision.detector import Detection
from vision.perception_loop import VisionPerceptionLoop
from vision.sorting_policy import classify_drinkware_targets


def _loop_with_detector(detector):
    with patch("vision.perception_loop.build_detector", return_value=detector):
        return VisionPerceptionLoop(
            vision_config={
                "target_labels": ["mug", "cup"],
                "obstacle_labels": [],
                "candidate_types": ["top_down"],
                "depth_sample_radius": 0,
                "max_obstacles": 16,
                "default_gripper_width": 0.04,
            },
            camera_config={
                "fx": 10.0,
                "fy": 10.0,
                "cx": 0.0,
                "cy": 0.0,
                "T_world_cam": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [0.0, 0.0, 0.0],
                },
            },
            conf_thresh=0.1,
        )


class CupMugSortingSceneTests(unittest.TestCase):
    def test_cup_mug_sorting_uses_graspable_drinkware_scale(self):
        from pathlib import Path

        source = Path(
            "third_party/robocasa/robocasa/environments/kitchen/composite/"
            "organizing_dishes_and_containers/cup_mug_sorting.py"
        ).read_text(encoding="utf-8")

        self.assertIn('self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))', source)
        self.assertIn('"counter", dict(id=FixtureType.COUNTER, ref=self.sink)', source)
        self.assertIn("PLAIN_CUP_OBJECT_SCALE = [0.75, 0.75, 1.0]", source)
        self.assertIn("PLAIN_CUP_MAX_SIZE = (0.072, 0.072, None)", source)
        self.assertIn("def __init__(self, num_mugs=2, num_cups=3", source)
        self.assertIn("sample_region_kwargs=dict", source)
        self.assertIn("loc=\"right\"", source)
        self.assertIn("front_edge_y = -0.92", source)
        self.assertIn("np.linspace(-0.55, 0.55, total)", source)
        self.assertIn("class CupMugSortingRandom(CupMugSorting):", source)
        self.assertIn('"random_scene"] = True', source)

    def test_infer_all_targets_keeps_each_detected_drinkware_object(self):
        class _Detector:
            def detect(self, rgb_image, labels):
                return [
                    Detection(label="mug", score=0.91, box_xyxy=[0.0, 0.0, 1.0, 1.0]),
                    Detection(label="cup", score=0.82, box_xyxy=[2.0, 2.0, 3.0, 3.0]),
                ]

        loop = _loop_with_detector(_Detector())
        depth = np.ones((4, 4), dtype=float)

        result = loop.infer_all_targets_with_diagnostics(
            rgb_image=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_image=depth,
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual([target["label"] for target in result["targets"]], ["mug", "cup"])
        self.assertTrue(all(target["grasp_candidates"] for target in result["targets"]))
        single_payload = loop.infer_detected_objects(
            rgb_image=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_image=depth,
        )
        self.assertEqual(single_payload["target"]["label"], "mug")

    def test_classify_drinkware_targets_assigns_handle_and_plain_strategies(self):
        targets = [
            {
                "label": "mug",
                "conf": 0.9,
                "pos": [0.1, 0.2, 0.3],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.4},
                    {"id": 2, "grasp_type": "handle_top_down", "score": 0.7},
                ],
                "diagnostics": {"2": {"handle_point_count": 24}},
            },
            {
                "label": "cup",
                "conf": 0.8,
                "pos": [0.4, 0.5, 0.6],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.5},
                ],
                "diagnostics": {},
            },
        ]

        assignments = classify_drinkware_targets(targets)

        self.assertEqual(assignments[0]["strategy"], "handle_top_down")
        self.assertEqual(assignments[0]["arm"], "right")
        self.assertTrue(assignments[0]["has_handle"])
        self.assertEqual(assignments[0]["candidate_id"], 2)
        self.assertEqual(assignments[1]["strategy"], "top_down")
        self.assertEqual(assignments[1]["arm"], "left")
        self.assertFalse(assignments[1]["has_handle"])
        self.assertEqual(assignments[1]["candidate_id"], 1)


if __name__ == "__main__":
    unittest.main()
