import unittest

import numpy as np

from vision.perception_loop import VisionPerceptionLoop


class _Detector:
    def detect(self, rgb_image, candidate_labels):
        return []


class HandleTopDownCandidateTests(unittest.TestCase):
    def _loop(self):
        loop = VisionPerceptionLoop.__new__(VisionPerceptionLoop)
        loop.vision_config = {}
        loop.detector = _Detector()
        loop.target_labels = ["mug"]
        loop.obstacle_labels = []
        loop.candidate_types = ["handle_top_down"]
        loop.default_gripper_width = 0.04
        loop.top_down_surface_quantile = 0.85
        loop.top_down_penetration_offset = 0.01
        loop.top_down_width_margin = 0.01
        loop.top_down_width_min = 0.04
        loop.handle_width_margin = 0.01
        loop.handle_width_min = 0.02
        loop.handle_top_down_inset = 0.0
        loop.handle_top_down_surface_quantile = 0.60
        loop.handle_top_down_penetration_offset = 0.008
        loop.handle_top_down_width_margin = 0.008
        loop.handle_top_down_width_min = 0.015
        loop.handle_top_down_width_max = 0.08
        loop.handle_labels = {"mug", "cup"}
        loop.handle_min_points = 4
        loop.handle_radial_ratio = 0.10
        loop.handle_radial_offset = 0.005
        loop.handle_side_band_fraction = 0.35
        loop.handle_height_quantiles = (0.20, 0.95)
        return loop

    def test_diagnostics_include_handle_top_down_geometry(self):
        body_points = np.array(
            [
                [-0.02, -0.02, 1.00],
                [-0.02, 0.02, 1.00],
                [0.02, -0.02, 1.00],
                [0.02, 0.02, 1.00],
                [0.00, 0.00, 1.08],
            ],
            dtype=float,
        )
        handle_points = np.array(
            [
                [0.12, -0.010, 1.02],
                [0.12, 0.010, 1.02],
                [0.14, -0.010, 1.04],
                [0.14, 0.010, 1.04],
                [0.16, -0.010, 1.06],
                [0.16, 0.010, 1.06],
            ],
            dtype=float,
        )
        loop = self._loop()
        loop._grasp_candidate_diagnostics = {}
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}

        candidate = loop._estimate_handle_top_down_candidate(target, np.vstack([body_points, handle_points]), 3)

        self.assertIsNotNone(candidate)
        diagnostics = loop._grasp_candidate_diagnostics["3"]
        self.assertEqual(diagnostics["grasp_type"], "handle_top_down")
        self.assertGreaterEqual(diagnostics["handle_point_count"], loop.handle_min_points)
        self.assertIn("outward_axis_xy", diagnostics)
        self.assertIn("thickness", diagnostics)
        self.assertIn("estimated_width", diagnostics)

    def test_choose_handle_side_point_cloud_uses_more_protruding_bbox_side(self):
        loop = self._loop()
        left_cloud = np.array(
            [[-0.02, 0.0, 1.0], [-0.01, 0.0, 1.0], [-0.02, 0.01, 1.0], [-0.01, 0.01, 1.0]],
            dtype=float,
        )
        right_cloud = np.array(
            [[0.12, 0.0, 1.0], [0.13, 0.0, 1.0], [0.12, 0.01, 1.0], [0.13, 0.01, 1.0]],
            dtype=float,
        )
        calls = []

        def extract(detection, depth_image, side_band=None):
            calls.append(side_band)
            if side_band == "left":
                return left_cloud
            if side_band == "right":
                return right_cloud
            return np.vstack([left_cloud, right_cloud])

        loop._extract_bbox_point_cloud = extract
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}

        cloud, side_band = loop._choose_handle_side_point_cloud(object(), np.zeros((2, 2)), target)

        self.assertEqual(side_band, "right")
        np.testing.assert_allclose(cloud, right_cloud)
        self.assertEqual(calls, ["left", "right"])

    def test_handle_top_down_uses_radial_closing_axis_and_thickness_width(self):
        body_points = np.array(
            [
                [-0.02, -0.02, 1.00],
                [-0.02, 0.02, 1.00],
                [0.02, -0.02, 1.00],
                [0.02, 0.02, 1.00],
                [0.00, 0.00, 1.08],
            ],
            dtype=float,
        )
        handle_points = np.array(
            [
                [0.12, -0.010, 1.02],
                [0.12, 0.010, 1.02],
                [0.14, -0.010, 1.04],
                [0.14, 0.010, 1.04],
                [0.16, -0.010, 1.06],
                [0.16, 0.010, 1.06],
            ],
            dtype=float,
        )
        point_cloud = np.vstack([body_points, handle_points])
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}

        candidate = self._loop()._estimate_handle_top_down_candidate(target, point_cloud, 3)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["grasp_type"], "handle_top_down")
        closing = np.asarray(candidate["orientation"][0], dtype=float)
        approach = np.asarray(candidate["orientation"][2], dtype=float)
        outward = np.array([1.0, 0.0, 0.0], dtype=float)
        self.assertGreater(float(np.dot(closing, outward)), 0.95)
        np.testing.assert_allclose(approach, [0.0, 0.0, -1.0])
        self.assertGreater(candidate["pos"][0], 0.13)
        self.assertLess(candidate["pos"][0], 0.18)
        self.assertLess(candidate["gripper_width"], 0.05)
        self.assertGreater(candidate["gripper_width"], 0.02)

    def test_handle_top_down_uses_target_center_for_cup_orientation(self):
        handle_points = np.array(
            [
                [0.10, 0.10, 1.02],
                [0.09, 0.11, 1.02],
                [0.11, 0.09, 1.03],
                [0.10, 0.12, 1.04],
                [0.12, 0.10, 1.05],
                [0.11, 0.11, 1.06],
            ],
            dtype=float,
        )
        body_points = np.array(
            [
                [-0.01, -0.01, 1.00],
                [0.01, -0.01, 1.00],
                [-0.01, 0.01, 1.00],
                [0.01, 0.01, 1.00],
            ],
            dtype=float,
        )
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}

        candidate = self._loop()._estimate_handle_top_down_candidate(target, np.vstack([body_points, handle_points]), 3)

        self.assertIsNotNone(candidate)
        closing = np.asarray(candidate["orientation"][0], dtype=float)
        outward = np.array([1.0, 1.0, 0.0], dtype=float) / np.sqrt(2.0)
        self.assertGreater(float(np.dot(closing, outward)), 0.95)
        self.assertGreater(abs(float(closing[0])), 0.5)
        self.assertGreater(abs(float(closing[1])), 0.5)

        body_points = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0], [-0.01, 0.0, 1.0]])
        handle_points = np.array(
            [
                [0.12, -0.05, 1.02],
                [0.12, 0.05, 1.02],
                [0.14, -0.05, 1.04],
                [0.14, 0.05, 1.04],
            ],
            dtype=float,
        )
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}

        candidate = self._loop()._estimate_handle_top_down_candidate(target, np.vstack([body_points, handle_points]), 1)

        self.assertIsNone(candidate)
    def test_build_candidates_emits_direct_handle_top_down(self):
        loop = self._loop()
        loop.candidate_types = ["top_down", "handle_top_down", "handle_grasp"]
        body_points = np.array(
            [
                [-0.02, -0.02, 1.00],
                [-0.02, 0.02, 1.00],
                [0.02, -0.02, 1.00],
                [0.02, 0.02, 1.00],
                [0.00, 0.00, 1.08],
            ],
            dtype=float,
        )
        handle_points = np.array(
            [
                [0.12, -0.010, 1.02],
                [0.12, 0.010, 1.02],
                [0.14, -0.010, 1.04],
                [0.14, 0.010, 1.04],
                [0.16, -0.010, 1.06],
                [0.16, 0.010, 1.06],
            ],
            dtype=float,
        )
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}

        candidates = loop._build_grasp_candidates(
            target,
            np.vstack([body_points, handle_points]),
            handle_point_cloud=handle_points,
            handle_side_band="right",
        )

        self.assertIn("handle_top_down", [candidate["grasp_type"] for candidate in candidates])

    def test_handle_top_down_uses_full_cloud_when_target_pos_is_biased(self):
        loop = self._loop()
        body_points = np.array(
            [
                [-0.03, -0.03, 1.00],
                [-0.03, 0.03, 1.00],
                [0.03, -0.03, 1.00],
                [0.03, 0.03, 1.00],
                [0.00, 0.00, 1.08],
                [0.02, 0.00, 1.04],
                [-0.02, 0.00, 1.04],
            ],
            dtype=float,
        )
        handle_points = np.array(
            [
                [0.12, -0.010, 1.02],
                [0.12, 0.010, 1.02],
                [0.14, -0.010, 1.04],
                [0.14, 0.010, 1.04],
                [0.16, -0.010, 1.06],
                [0.16, 0.010, 1.06],
            ],
            dtype=float,
        )
        target = {"label": "mug", "pos": [0.11, 0.0, 1.0], "conf": 0.9}

        candidate = loop._estimate_handle_top_down_candidate(
            target,
            np.vstack([body_points, handle_points]),
            3,
            handle_point_cloud=handle_points,
        )

        self.assertIsNotNone(candidate)
        closing = np.asarray(candidate["orientation"][0], dtype=float)
        outward = np.array([1.0, 0.0, 0.0], dtype=float)
        self.assertGreater(float(np.dot(closing, outward)), 0.95)
        self.assertLess(abs(float(closing[1])), 0.2)
    def test_handle_top_down_uses_outer_tail_for_outward_direction(self):
        loop = self._loop()
        body_points = np.array(
            [
                [-0.03, -0.03, 1.00],
                [-0.03, 0.03, 1.00],
                [0.03, -0.03, 1.00],
                [0.03, 0.03, 1.00],
                [0.00, 0.00, 1.08],
            ],
            dtype=float,
        )
        handle_points = np.array(
            [
                [0.03, 0.08, 1.02],
                [0.04, 0.08, 1.03],
                [0.02, 0.09, 1.03],
                [0.05, 0.09, 1.04],
                [0.06, 0.11, 1.04],
                [0.07, 0.13, 1.05],
                [0.08, 0.15, 1.05],
                [0.09, 0.17, 1.06],
            ],
            dtype=float,
        )
        target = {"label": "mug", "pos": [0.0, 0.0, 1.0], "conf": 0.9}
        loop.handle_min_points = 2
        loop._grasp_candidate_diagnostics = {}

        candidate = loop._estimate_handle_top_down_candidate(
            target,
            np.vstack([body_points, handle_points]),
            3,
            handle_point_cloud=handle_points,
        )

        self.assertIsNotNone(candidate)
        diagnostics = loop._grasp_candidate_diagnostics["3"]
        self.assertEqual(diagnostics["handle_direction_source"], "handle_tail_outer_points")
        outward = np.asarray(diagnostics["outward_axis_xy"], dtype=float)
        expected_tail_direction = np.array([0.09, 0.17], dtype=float)
        expected_tail_direction /= np.linalg.norm(expected_tail_direction)
        self.assertGreater(float(np.dot(outward, expected_tail_direction)), 0.95)
        closing = np.asarray(candidate["orientation"][0], dtype=float)
        self.assertGreater(float(np.dot(closing[:2], outward)), 0.95)


if __name__ == "__main__":
    unittest.main()
