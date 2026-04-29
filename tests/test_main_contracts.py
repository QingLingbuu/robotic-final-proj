import unittest

import numpy as np

from main import build_single_grasp_pos


class MainPlanningContractTests(unittest.TestCase):
    def test_candidate_source_uses_candidate_position_without_extra_offset(self):
        target_pos = np.array([0.1, 0.2, 0.3], dtype=float)

        grasp_pos = build_single_grasp_pos(
            env=None,
            grasp_target_pos=target_pos,
            source="candidate",
            vision_config={"single_grasp_z_offset": -0.04},
        )

        np.testing.assert_allclose(grasp_pos, target_pos)


if __name__ == "__main__":
    unittest.main()
