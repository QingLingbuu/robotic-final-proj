import unittest

from arm.reachability import diagnose_reachability


class ReachabilityDiagnosisTests(unittest.TestCase):
    def test_successful_reach_is_marked_reachable(self):
        diagnosis = diagnose_reachability({"reach_success": True, "candidate_type": "top_down"})

        self.assertEqual(diagnosis["likely_cause"], "reachable")
        self.assertTrue(diagnosis["reach_success"])

    def test_position_saturation_suggests_base_or_workspace_issue(self):
        diagnosis = diagnose_reachability({
            "reach_success": False,
            "candidate_type": "top_down",
            "final_error_to_settle": 0.14,
            "history_tail": [{"position_delta_action": [1.0, 0.2, -1.0]}],
        })

        self.assertTrue(diagnosis["position_saturated"])
        self.assertEqual(diagnosis["likely_cause"], "position_control_saturated_or_base_torso_needed")

    def test_wide_handle_flags_gripper_width_risk_first(self):
        diagnosis = diagnose_reachability(
            {"reach_success": False, "candidate_type": "handle_grasp"},
            candidate={"gripper_width": 0.14},
        )

        self.assertTrue(diagnosis["gripper_width_risk"])
        self.assertEqual(diagnosis["likely_cause"], "candidate_gripper_width_exceeds_panda_limit")


if __name__ == "__main__":
    unittest.main()
