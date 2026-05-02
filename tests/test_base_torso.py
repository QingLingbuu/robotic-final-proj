import unittest

import numpy as np

from arm.base_torso import (
    build_base_torso_action,
    calibrate_base_action_mapping,
    compute_preposition_command,
    preposition_base_torso,
)


class _Env:
    action_dim = 12

    def __init__(self):
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        return {"robot0_eef_pos": [0.0, 0.0, 1.1]}, 0.0, False, {}

    def render(self):
        return None


class _BaseCalibrationEnv:
    action_dim = 12

    def __init__(self):
        self.eef_pos = np.array([0.0, 0.0, 1.0], dtype=float)
        self.base_pos = np.array([0.0, 0.0, 0.0], dtype=float)

    def step(self, action):
        delta = np.array([action[7], action[8], 0.0], dtype=float)
        self.base_pos += delta
        self.eef_pos[:2] += delta[:2]
        return {
            "robot0_eef_pos": self.eef_pos.tolist(),
            "robot0_base_pos": self.base_pos.tolist(),
        }, 0.0, False, {}

    def render(self):
        return None


class BaseTorsoTests(unittest.TestCase):
    def test_build_base_torso_action_sets_configured_indices(self):
        action = build_base_torso_action(
            _Env(),
            base_xy=[0.2, -0.1, 0.05],
            torso=0.05,
            base_slice=(7, 10),
            torso_index=None,
        )

        self.assertEqual(action.shape, (12,))
        np.testing.assert_allclose(action[7:10], [0.2, -0.1, 0.05])
        np.testing.assert_allclose(action[:7], np.zeros(7))
        np.testing.assert_allclose(action[10:], np.zeros(2))

    def test_compute_preposition_command_moves_when_target_is_far(self):
        command = compute_preposition_command(
            eef_pos=[0.0, 0.0, 1.1],
            target_pos=[0.6, 0.0, 1.1],
            action_mapping=np.eye(3),
            desired_xy_standoff=0.18,
        )

        self.assertTrue(command["should_move"])
        self.assertGreater(command["base_xy"][0], 0.0)
        self.assertAlmostEqual(command["torso"], 0.0)
        self.assertGreater(command["base_command_norm"], 0.0)

    def test_compute_preposition_command_uses_base_action_mapping(self):
        command = compute_preposition_command(
            eef_pos=[0.0, 0.0, 1.1],
            target_pos=[0.4, 0.0, 1.1],
            base_action_mapping=np.array([[2.0, 0.0], [0.0, 1.0]]),
            desired_xy_standoff=0.0,
            xy_deadband=0.0,
            base_gain=1.0,
            base_action_limit=1.0,
            base_mapping_trust=1.0,
            max_base_world_delta=1.0,
            action_steps=1,
        )

        self.assertTrue(command["base_mapping_used"])
        np.testing.assert_allclose(command["base_xy"], [0.2, 0.0])

    def test_compute_preposition_command_scales_mapping_by_action_steps(self):
        command = compute_preposition_command(
            eef_pos=[0.0, 0.0, 1.1],
            target_pos=[0.4, 0.0, 1.1],
            base_action_mapping=np.array([[2.0, 0.0], [0.0, 1.0]]),
            desired_xy_standoff=0.0,
            xy_deadband=0.0,
            base_gain=1.0,
            base_action_limit=1.0,
            action_steps=10,
            base_mapping_trust=1.0,
            max_base_world_delta=1.0,
        )

        np.testing.assert_allclose(command["base_xy"], [0.02, 0.0])

    def test_compute_preposition_command_limits_world_delta_and_trust(self):
        command = compute_preposition_command(
            eef_pos=[0.0, 0.0, 1.1],
            target_pos=[1.0, 0.0, 1.1],
            base_action_mapping=np.eye(2),
            desired_xy_standoff=0.0,
            xy_deadband=0.0,
            base_gain=1.0,
            base_action_limit=1.0,
            action_steps=1,
            base_mapping_trust=0.25,
            max_base_world_delta=0.2,
        )

        np.testing.assert_allclose(command["desired_base_world_delta"], [0.2, 0.0])
        np.testing.assert_allclose(command["base_xy"], [0.05, 0.0])

    def test_compute_preposition_command_stays_idle_inside_deadband(self):
        command = compute_preposition_command(
            eef_pos=[0.0, 0.0, 1.1],
            target_pos=[0.2, 0.0, 1.12],
            action_mapping=np.eye(3),
            desired_xy_standoff=0.18,
            xy_deadband=0.06,
            z_deadband=0.04,
        )

        self.assertFalse(command["should_move"])
        np.testing.assert_allclose(command["base_xy"], [0.0, 0.0])

    def test_preposition_ignores_torso_when_torso_index_is_disabled(self):
        env = _Env()
        _, summary = preposition_base_torso(
            env,
            obs={"robot0_eef_pos": [0.0, 0.0, 1.1]},
            target_pos=[0.2, 0.0, 0.9],
            action_mapping=np.eye(3),
            torso_index=None,
            render_sleep_sec=0.0,
        )

        self.assertTrue(summary["command"]["should_move"])
        self.assertTrue(summary["torso_command_ignored"])
        self.assertFalse(summary["effective_should_move"])
        self.assertEqual(env.step_count, 0)

    def test_preposition_reports_xy_distance_improvement(self):
        class MovingEnv(_Env):
            def step(self, action):
                self.step_count += 1
                return {"robot0_eef_pos": [0.4, 0.0, 1.1]}, 0.0, False, {}

        env = MovingEnv()
        _, summary = preposition_base_torso(
            env,
            obs={"robot0_eef_pos": [0.0, 0.0, 1.1]},
            target_pos=[0.6, 0.0, 1.1],
            action_mapping=np.eye(3),
            desired_xy_standoff=0.0,
            xy_deadband=0.0,
            base_gain=1.0,
            render_sleep_sec=0.0,
        )

        self.assertTrue(summary["effective_should_move"])
        self.assertGreater(summary["action_norm"], 0.0)
        self.assertLess(summary["xy_distance_after"], summary["xy_distance_before"])
        self.assertTrue(summary["xy_distance_improved"])

    def test_calibrate_base_action_mapping_estimates_eef_xy_columns(self):
        env = _BaseCalibrationEnv()
        obs = {"robot0_eef_pos": env.eef_pos.tolist(), "robot0_base_pos": env.base_pos.tolist()}

        _, summary = calibrate_base_action_mapping(
            env,
            obs=obs,
            base_slice=(7, 9),
            pulse_magnitude=0.1,
            pulse_steps=2,
            render_sleep_sec=0.0,
        )

        np.testing.assert_allclose(summary["eef_xy_delta_from_base_action"], [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
