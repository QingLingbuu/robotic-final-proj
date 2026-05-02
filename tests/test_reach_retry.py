import unittest

from arm.reach_retry import should_retry_with_preposition


class ReachRetryTests(unittest.TestCase):
    def test_no_retry_when_reach_already_succeeds(self):
        self.assertFalse(should_retry_with_preposition(
            {"reach_success": True},
            {"final_error": 0.1},
            {"grasp_type": "top_down"},
        ))

    def test_retry_top_down_when_position_saturated(self):
        self.assertTrue(should_retry_with_preposition(
            {"reach_success": False},
            {"position_saturated": True, "final_error": 0.01},
            {"grasp_type": "top_down"},
        ))

    def test_retry_top_down_when_final_error_is_large(self):
        self.assertTrue(should_retry_with_preposition(
            {"reach_success": False},
            {"position_saturated": False, "final_error": 0.05},
            {"grasp_type": "top_down"},
            final_error_threshold=0.03,
        ))

    def test_no_retry_for_handle_reach_only_experiment(self):
        self.assertFalse(should_retry_with_preposition(
            {"reach_success": False, "reach_only": True},
            {"position_saturated": True, "final_error": 0.2},
            {"grasp_type": "handle_oblique_grasp"},
        ))


if __name__ == "__main__":
    unittest.main()
