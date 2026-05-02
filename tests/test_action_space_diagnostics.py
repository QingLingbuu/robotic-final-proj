import unittest

import numpy as np

from arm.action_space_diagnostics import (
    build_axis_pulse_action,
    changed_numeric_obs_keys,
    infer_action_slices,
    summarize_action_effect,
)


class _Env:
    action_dim = 8


class ActionSpaceDiagnosticsTests(unittest.TestCase):
    def test_build_axis_pulse_action_sets_one_index(self):
        action = build_axis_pulse_action(_Env(), action_index=3, magnitude=-0.25)

        self.assertEqual(action.shape, (8,))
        self.assertEqual(float(action[3]), -0.25)
        np.testing.assert_allclose(np.delete(action, 3), np.zeros(7))

    def test_summarize_action_effect_reports_eef_and_gripper_delta(self):
        summary = summarize_action_effect(
            {"robot0_eef_pos": [1.0, 2.0, 3.0], "robot0_gripper_qpos": [0.02, 0.02]},
            {"robot0_eef_pos": [1.1, 2.0, 2.9], "robot0_gripper_qpos": [0.01, 0.01]},
        )

        np.testing.assert_allclose(summary["eef_delta"], [0.1, 0.0, -0.1])
        self.assertAlmostEqual(summary["gripper_width_delta"], -0.02)

    def test_changed_numeric_obs_keys_reports_numeric_deltas(self):
        changed = changed_numeric_obs_keys(
            {"robot0_base_qpos": [0.0, 0.0], "name": "start"},
            {"robot0_base_qpos": [0.1, 0.0], "name": "end"},
        )

        self.assertEqual(changed[0]["key"], "robot0_base_qpos")
        self.assertAlmostEqual(changed[0]["delta_norm"], 0.1)

    def test_infer_action_slices_marks_effective_indices(self):
        summary = infer_action_slices([
            {"action_index": 0, "positive": {"eef_delta_norm": 0.01, "gripper_width_delta": 0.0}},
            {"action_index": 6, "positive": {"eef_delta_norm": 0.0, "gripper_width_delta": -0.02}},
        ])

        self.assertEqual(summary["eef_effect_indices"], [0])
        self.assertEqual(summary["gripper_effect_indices"], [6])


if __name__ == "__main__":
    unittest.main()
