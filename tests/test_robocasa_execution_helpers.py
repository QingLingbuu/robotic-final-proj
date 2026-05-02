import unittest

import numpy as np

from arm.robocasa_primitives import (
    HANDLE_TOP_DOWN_EEF_Z_OFFSET,
    HANDLE_TOP_DOWN_YAW_GAIN,
    HANDLE_TOP_DOWN_YAW_MAX_ACTION,
    TOP_DOWN_SETTLE_ACTION_SCALE,
    TOP_DOWN_YAW_GAIN,
    TOP_DOWN_YAW_MAX_ACTION,
    build_target_rotation_for_top_down,
    compute_top_down_yaw_action,
)
from arm.robocasa_execution import (
    build_flat_reach_action,
    get_robot0_eef_pos,
    get_robot0_eef_quat,
    read_gripper_width,
)


class _Env:
    action_dim = 14


class RobocasaExecutionHelperTests(unittest.TestCase):
    def test_build_flat_reach_action_sets_right_arm_and_gripper(self):
        action = build_flat_reach_action(
            _Env(),
            position_delta=[0.1, -0.2, 0.3],
            orientation_delta=[0.4, -0.5, 0.6],
            gripper_close=-1.0,
        )

        self.assertEqual(action.shape, (14,))
        np.testing.assert_allclose(action[0:3], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(action[3:6], [0.4, -0.5, 0.6])
        self.assertEqual(float(action[6]), -1.0)
        np.testing.assert_allclose(action[7:], np.zeros(7))

    def test_get_eef_pos_and_quat_read_expected_keys(self):
        obs = {
            "robot0_eef_pos": [1.0, 2.0, 3.0],
            "robot0_eef_quat_site": [1.0, 0.0, 0.0, 0.0],
        }

        np.testing.assert_allclose(get_robot0_eef_pos(obs), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(get_robot0_eef_quat(obs), [1.0, 0.0, 0.0, 0.0])

    def test_get_eef_quat_falls_back_to_robot_key(self):
        obs = {"robot0_eef_quat": [0.0, 1.0, 0.0, 0.0]}

        np.testing.assert_allclose(get_robot0_eef_quat(obs), [0.0, 1.0, 0.0, 0.0])

    def test_read_gripper_width_sums_absolute_finger_positions(self):
        self.assertAlmostEqual(read_gripper_width({"robot0_gripper_qpos": [-0.02, 0.03]}), 0.05)
        self.assertIsNone(read_gripper_width({}))

    def test_build_top_down_rotation_maps_candidate_closing_to_gripper_y_axis(self):
        candidate = {
            "orientation": [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        }

        rotation = build_target_rotation_for_top_down(candidate)

        np.testing.assert_allclose(rotation[:, 1], [0.0, 1.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)

    def test_top_down_yaw_action_uses_gripper_y_axis_as_current_closing_axis(self):
        candidate = {
            "orientation": [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        }

        action, error = compute_top_down_yaw_action([1.0, 0.0, 0.0, 0.0], candidate)

        self.assertAlmostEqual(error, 0.0, places=7)
        np.testing.assert_allclose(action, [0.0, 0.0, 0.0], atol=1e-7)

    def test_handle_top_down_settle_offset_targets_contact_height(self):
        self.assertLess(HANDLE_TOP_DOWN_EEF_Z_OFFSET, 0.0)
        self.assertAlmostEqual(HANDLE_TOP_DOWN_EEF_Z_OFFSET, -0.025)

    def test_top_down_settle_action_scale_keeps_final_descent_conservative(self):
        self.assertLessEqual(TOP_DOWN_SETTLE_ACTION_SCALE, 0.30)
        self.assertGreater(TOP_DOWN_SETTLE_ACTION_SCALE, 0.0)

    def test_handle_top_down_yaw_alignment_is_slightly_faster(self):
        self.assertGreater(HANDLE_TOP_DOWN_YAW_GAIN, TOP_DOWN_YAW_GAIN)
        self.assertGreater(HANDLE_TOP_DOWN_YAW_MAX_ACTION, TOP_DOWN_YAW_MAX_ACTION)
        self.assertLessEqual(HANDLE_TOP_DOWN_YAW_MAX_ACTION, 0.12)


if __name__ == "__main__":
    unittest.main()
