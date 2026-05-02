import unittest

import numpy as np

from arm.handle_phases import build_oblique_reach_phases
from arm.handle_validation import summarize_handle_reach_validation


class HandleExperimentsTests(unittest.TestCase):
    def test_stable_handle_reach_requires_all_phases_and_final_tolerances(self):
        validation = summarize_handle_reach_validation(
            [
                {"phase": "pre_approach", "ok": True, "pos_error_norm": 0.02, "ori_error_norm": 0.0},
                {"phase": "approach", "ok": True, "pos_error_norm": 0.01, "ori_error_norm": 0.2},
            ],
            final_error_to_candidate=0.01,
        )

        self.assertEqual(validation["mode"], "handle_reach_only")
        self.assertTrue(validation["stable_reach"])
        self.assertEqual(validation["failed_phases"], [])

    def test_failed_phase_marks_handle_reach_unstable(self):
        validation = summarize_handle_reach_validation(
            [
                {"phase": "pre_approach", "ok": True, "pos_error_norm": 0.02, "ori_error_norm": 0.0},
                {"phase": "approach", "ok": False, "pos_error_norm": 0.04, "ori_error_norm": 0.2},
            ],
            final_error_to_candidate=0.04,
        )

        self.assertFalse(validation["stable_reach"])
        self.assertEqual(validation["failed_phases"], ["approach"])

    def test_oblique_reach_defaults_to_contact_limitation_demo(self):
        phases, pre_approach = build_oblique_reach_phases(
            np.array([1.0, 2.0, 0.9]),
            np.array([1.0, 0.0, 0.0]),
            np.eye(3),
            reach_tolerance=0.01,
            include_contact_approach=True,
        )

        self.assertEqual([phase[0] for phase in phases], [
            "oblique_pre_approach",
            "oblique_align_orientation",
            "oblique_approach",
        ])
        self.assertLess(pre_approach[0], 1.0)

    def test_oblique_noncontact_validation_is_explicit(self):
        phases, _ = build_oblique_reach_phases(
            np.array([1.0, 2.0, 0.9]),
            np.array([1.0, 0.0, 0.0]),
            np.eye(3),
            reach_tolerance=0.01,
            include_contact_approach=False,
        )

        self.assertEqual([phase[0] for phase in phases], ["oblique_pre_approach", "oblique_align_orientation"])

if __name__ == "__main__":
    unittest.main()
