import unittest

from vision.sorting_policy import (
    assign_sim_metadata_to_targets,
    build_sim_metadata_fallback_target,
    select_drinkware_target,
)


class SortingPolicyTests(unittest.TestCase):
    def test_select_cup_prefers_plain_object_even_if_mug_has_higher_confidence(self):
        targets = [
            {
                "label": "mug",
                "conf": 0.95,
                "pos": [0.0, 0.0, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.4},
                    {"id": 2, "grasp_type": "handle_top_down", "score": 0.7},
                ],
            },
            {
                "label": "cup",
                "conf": 0.80,
                "pos": [1.0, 0.0, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.6},
                ],
            },
        ]

        target, assignment = select_drinkware_target(targets, requested_label="cup")

        self.assertEqual(target["pos"], [1.0, 0.0, 1.0])
        self.assertFalse(assignment["has_handle"])
        self.assertEqual(assignment["strategy"], "top_down")

    def test_select_mug_prefers_handle_object(self):
        targets = [
            {
                "label": "cup",
                "conf": 0.95,
                "pos": [0.0, 0.0, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.6},
                ],
            },
            {
                "label": "mug",
                "conf": 0.70,
                "pos": [1.0, 0.0, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.4},
                    {"id": 2, "grasp_type": "handle_top_down", "score": 0.8},
                ],
            },
        ]

        target, assignment = select_drinkware_target(targets, requested_label="mug")

        self.assertEqual(target["pos"], [1.0, 0.0, 1.0])
        self.assertTrue(assignment["has_handle"])
        self.assertEqual(assignment["strategy"], "handle_top_down")

    def test_low_score_handle_candidate_is_treated_as_plain_cup(self):
        targets = [
            {
                "label": "glass cup",
                "conf": 0.36,
                "pos": [0.0, 0.0, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.30},
                    {"id": 2, "grasp_type": "handle_top_down", "score": 0.22},
                ],
                "diagnostics": {
                    "2": {
                        "handle_point_count": 8,
                        "protrusion_quality": 0.20,
                    }
                },
            }
        ]

        target, assignment = select_drinkware_target(targets, requested_label="cup")

        self.assertIsNotNone(target)
        self.assertFalse(assignment["has_handle"])
        self.assertEqual(assignment["strategy"], "top_down")

    def test_strong_handle_geometry_overrides_borderline_handle_score(self):
        targets = [
            {
                "label": "glass cup",
                "conf": 0.46,
                "pos": [0.0, 0.0, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.23},
                    {"id": 2, "grasp_type": "handle_top_down", "score": 0.297},
                ],
                "diagnostics": {
                    "2": {
                        "handle_point_count": 39,
                        "protrusion_quality": 0.57,
                    }
                },
            }
        ]

        target, assignment = select_drinkware_target(targets, requested_label="cup")

        self.assertIsNone(target)
        self.assertIsNone(assignment)

    def test_sim_metadata_overrides_visual_handle_false_positive(self):
        targets = [
            {
                "label": "glass cup",
                "conf": 0.47,
                "pos": [1.02, 0.01, 1.0],
                "grasp_candidates": [
                    {"id": 1, "grasp_type": "top_down", "score": 0.23},
                    {"id": 2, "grasp_type": "handle_top_down", "score": 0.31},
                ],
            }
        ]
        sim_objects = [
            {"name": "cup_1", "pos": [1.0, 0.0, 1.0], "has_handle": False},
            {"name": "mug_1", "pos": [2.0, 0.0, 1.0], "has_handle": True},
        ]

        corrected = assign_sim_metadata_to_targets(targets, sim_objects)
        target, assignment = select_drinkware_target(corrected, requested_label="cup")

        self.assertIsNotNone(target)
        self.assertEqual(target["sim_object_name"], "cup_1")
        self.assertFalse(assignment["has_handle"])
        self.assertEqual(assignment["strategy"], "top_down")

    def test_builds_plain_cup_fallback_when_vision_only_sees_mugs(self):
        sim_objects = [
            {"name": "mug_1", "pos": [0.5, -0.5, 0.97], "has_handle": True},
            {"name": "cup_1", "pos": [2.1, -0.48, 0.97], "has_handle": False},
        ]

        target, assignment = build_sim_metadata_fallback_target(sim_objects, requested_label="cup")

        self.assertEqual(target["label"], "cup")
        self.assertEqual(target["sim_object_name"], "cup_1")
        self.assertFalse(assignment["has_handle"])
        self.assertEqual(assignment["strategy"], "top_down")
        self.assertEqual(target["grasp_candidates"][0]["grasp_type"], "top_down")


if __name__ == "__main__":
    unittest.main()
