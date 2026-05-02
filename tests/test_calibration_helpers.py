import unittest

import numpy as np

from arm.calibration import extract_base_to_world_rotation


class CalibrationHelperTests(unittest.TestCase):
    def test_extract_base_to_world_rotation_returns_none_without_mapping(self):
        self.assertIsNone(extract_base_to_world_rotation(None))

    def test_extract_base_to_world_rotation_orthonormalizes_mapping(self):
        theta = np.pi / 4.0
        expected = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        noisy_mapping = expected @ np.diag([0.9, 1.1, 1.0])

        rotation = extract_base_to_world_rotation(noisy_mapping)

        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        self.assertGreater(np.linalg.det(rotation), 0.0)
        np.testing.assert_allclose(rotation, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
