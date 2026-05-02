import unittest

from planner.candidates import choose_reachable_candidate


class CandidatePlannerTests(unittest.TestCase):
    def _payload(self):
        return {
            "target": {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9, "timestamp": 1.0},
            "grasp_candidates": [
                {
                    "id": 1,
                    "pos": [0.0, 0.0, 1.2],
                    "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                    "gripper_width": 0.095,
                    "score": 0.4,
                    "grasp_type": "top_down",
                },
                {
                    "id": 2,
                    "pos": [0.10, 0.00, 1.0],
                    "orientation": [[0, 1, 0], [0, 0, 1], [-1, 0, 0]],
                    "gripper_width": 0.14,
                    "score": 0.7,
                    "grasp_type": "handle_grasp",
                },
            ],
            "status": "ready",
        }

    def test_default_handle_mode_falls_back_to_top_down(self):
        candidate, reason = choose_reachable_candidate(
            self._payload(), grasp_type="handle_grasp"
        )

        self.assertEqual(candidate["grasp_type"], "top_down")
        self.assertEqual(reason["strategy"], "top_down_default_fallback")

    def test_missing_handle_top_down_builds_from_handle_candidate(self):
        candidate, reason = choose_reachable_candidate(
            self._payload(), grasp_type="handle_top_down"
        )

        self.assertEqual(candidate["grasp_type"], "handle_top_down")
        self.assertEqual(candidate["source_handle_candidate_id"], 2)
        self.assertEqual(candidate["orientation"][2], [0.0, 0.0, -1.0])
        self.assertAlmostEqual(candidate["pos"][0], 0.10, places=7)
        self.assertLess(candidate["pos"][2], 1.02)
        self.assertEqual(candidate["gripper_width"], 0.08)
        self.assertEqual(reason["strategy"], "handle_top_down_from_handle_grasp")

    def test_direct_handle_top_down_is_preferred_over_fallback(self):
        payload = self._payload()
        payload["grasp_candidates"].append(
            {
                "id": 3,
                "pos": [0.11, 0.01, 1.01],
                "orientation": [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
                "gripper_width": 0.04,
                "score": 0.6,
                "grasp_type": "handle_top_down",
            }
        )

        candidate, reason = choose_reachable_candidate(payload, grasp_type="handle_top_down")

        self.assertEqual(candidate["grasp_type"], "handle_top_down")
        self.assertEqual(candidate["id"], 3)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
