import unittest
import random
import json
import tempfile
from pathlib import Path

from rl.cup_mug_observation import (
    MAX_TARGETS,
    OPPOSITE_COUNTER_ZONE_ID,
    SINK_ZONE_ID,
    UNKNOWN_ZONE_ID,
    build_cup_mug_ordering_observation,
)
from rl.cup_mug_policies import select_action
from rl.cup_mug_runner import CupMugOrderingEpisodeRunner
from rl.cup_mug_metrics import (
    aggregate_episode_records,
    build_episode_record,
    build_rl_ordering_run_payload,
    write_ordering_metrics,
)
from rl.cup_mug_live_mapping import build_live_slot_mapping, resolve_live_target_from_slot, resolve_live_target_with_identity_preference


def _target(name, label, has_handle, conf=0.9, score=0.8, pos=None, visible=True):
    grasp_type = "handle_top_down" if has_handle else "top_down"
    return {
        "label": label,
        "conf": conf,
        "pos": pos or [0.0, 0.0, 1.0],
        "visible": visible,
        "sim_object_name": name,
        "sim_has_handle": has_handle,
        "grasp_candidates": [
            {"id": 1, "grasp_type": grasp_type, "score": score},
        ],
    }


def _assignment(name, label, has_handle, score=0.8, reachability=0.7):
    return {
        "label": label,
        "sim_object_name": name,
        "has_handle": has_handle,
        "strategy": "handle_top_down" if has_handle else "top_down",
        "candidate_score": score,
        "reachability": reachability,
    }


SORTING_METADATA = {
    "mug_1": {"has_handle": True, "target_zone": "handled", "recommended_grasp": "handle_top_down"},
    "mug_2": {"has_handle": True, "target_zone": "handled", "recommended_grasp": "handle_top_down"},
    "cup_1": {"has_handle": False, "target_zone": "plain", "recommended_grasp": "top_down"},
    "cup_2": {"has_handle": False, "target_zone": "plain", "recommended_grasp": "top_down"},
    "cup_3": {"has_handle": False, "target_zone": "plain", "recommended_grasp": "top_down"},
}


class CupMugOrderingObservationTests(unittest.TestCase):
    def test_five_valid_objects_produce_five_valid_actions_in_slot_order(self):
        names = ["cup_2", "mug_2", "cup_1", "mug_1", "cup_3"]
        targets = [_target(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]
        assignments = [_assignment(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]

        observation = build_cup_mug_ordering_observation(targets, assignments, SORTING_METADATA)

        self.assertEqual(observation["max_targets"], MAX_TARGETS)
        self.assertEqual([slot["object_id"] for slot in observation["slots"]], ["mug_1", "mug_2", "cup_1", "cup_2", "cup_3"])
        self.assertEqual(observation["action_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(observation["finished_mask"], [0, 0, 0, 0, 0])
        self.assertEqual(observation["slots"][0]["place_zone_id"], SINK_ZONE_ID)
        self.assertEqual(observation["slots"][2]["place_zone_id"], OPPOSITE_COUNTER_ZONE_ID)

    def test_zero_targets_builds_all_padding_all_invalid(self):
        observation = build_cup_mug_ordering_observation([])

        self.assertEqual(len(observation["slots"]), 5)
        self.assertEqual(observation["action_mask"], [0, 0, 0, 0, 0])
        self.assertTrue(all(slot["object_id"] is None for slot in observation["slots"]))
        self.assertEqual(observation["global"]["remaining_count"], 0)

    def test_two_targets_are_padded_to_five_slots(self):
        targets = [_target("mug_1", "mug", True), _target("cup_1", "cup", False)]
        assignments = [_assignment("mug_1", "mug", True), _assignment("cup_1", "cup", False)]

        observation = build_cup_mug_ordering_observation(targets, assignments, SORTING_METADATA)

        self.assertEqual([slot["object_id"] for slot in observation["slots"]], ["mug_1", "cup_1", None, None, None])
        self.assertEqual(observation["action_mask"], [1, 1, 0, 0, 0])

    def test_more_than_five_candidates_are_truncated_after_stable_ordering(self):
        names = ["cup_4", "mug_1", "cup_1", "cup_2", "mug_2", "cup_3"]
        targets = [_target(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]
        assignments = [_assignment(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]

        observation = build_cup_mug_ordering_observation(targets, assignments, SORTING_METADATA)

        self.assertEqual([slot["object_id"] for slot in observation["slots"]], ["mug_1", "mug_2", "cup_1", "cup_2", "cup_3"])

    def test_finished_low_confidence_unknown_duplicate_and_invisible_slots_are_masked(self):
        targets = [
            _target("mug_1", "mug", True),
            _target("mug_1", "mug", True),
            _target("cup_1", "cup", False, conf=0.2),
            {"label": "unknown", "conf": 0.9, "pos": [0.3, 0.0, 1.0], "grasp_candidates": [{"id": 1, "score": 0.7}]},
            _target("cup_2", "cup", False, visible=False),
        ]
        assignments = [
            _assignment("mug_1", "mug", True),
            _assignment("mug_1", "mug", True),
            _assignment("cup_1", "cup", False),
            {"label": "unknown", "candidate_score": 0.7},
            _assignment("cup_2", "cup", False),
        ]

        observation = build_cup_mug_ordering_observation(
            targets,
            assignments,
            SORTING_METADATA,
            finished={"cup_2": True},
        )

        self.assertEqual(observation["action_mask"], [0, 0, 0, 0, 0])
        self.assertEqual(observation["slots"][0]["object_id"], "mug_1")
        self.assertFalse(observation["slots"][0]["valid_action"])
        self.assertEqual(observation["slots"][3]["place_zone_id"], UNKNOWN_ZONE_ID)
        cup_2_slot = next(slot for slot in observation["slots"] if slot["object_id"] == "cup_2")
        self.assertTrue(cup_2_slot["finished"])

    def test_fallback_ordering_without_sim_names_is_deterministic(self):
        targets = [
            {"label": "cup", "conf": 0.9, "pos": [0.2, 0.0, 1.0], "sim_has_handle": False, "grasp_candidates": [{"id": 1, "score": 0.5}]},
            {"label": "mug", "conf": 0.9, "pos": [0.4, 0.0, 1.0], "sim_has_handle": True, "grasp_candidates": [{"id": 1, "score": 0.5}]},
            {"label": "cup", "conf": 0.9, "pos": [0.1, 0.0, 1.0], "sim_has_handle": False, "grasp_candidates": [{"id": 1, "score": 0.5}]},
        ]
        assignments = [
            {"label": "cup", "has_handle": False, "candidate_score": 0.5},
            {"label": "mug", "has_handle": True, "candidate_score": 0.5},
            {"label": "cup", "has_handle": False, "candidate_score": 0.5},
        ]

        observation = build_cup_mug_ordering_observation(targets, assignments)

        self.assertEqual([slot["label"] for slot in observation["slots"][:3]], ["mug", "cup", "cup"])
        self.assertEqual([slot["pos"][0] for slot in observation["slots"][:3]], [0.4, 0.1, 0.2])

    def test_global_fields_capture_step_context(self):
        observation = build_cup_mug_ordering_observation(
            [_target("mug_1", "mug", True)],
            [_assignment("mug_1", "mug", True)],
            SORTING_METADATA,
            step_index=3,
            last_action=1,
            last_success=False,
            last_failure_type="placement_failure",
        )

        self.assertEqual(observation["global"]["remaining_count"], 1)
        self.assertEqual(observation["global"]["completed_count"], 0)
        self.assertEqual(observation["global"]["step_index"], 3)
        self.assertEqual(observation["global"]["last_action"], 1)
        self.assertFalse(observation["global"]["last_success"])
        self.assertEqual(observation["global"]["last_failure_type"], "placement_failure")

    def test_random_policy_samples_only_valid_actions(self):
        observation = build_cup_mug_ordering_observation(
            [_target("mug_1", "mug", True), _target("cup_1", "cup", False), _target("cup_2", "cup", False)],
            [_assignment("mug_1", "mug", True), _assignment("cup_1", "cup", False), _assignment("cup_2", "cup", False)],
            SORTING_METADATA,
            finished={"cup_1": True},
        )

        selected = [select_action(observation, "random", rng=random.Random(7))[0] for _ in range(10)]

        self.assertTrue(all(action in {0, 2} for action in selected))
        self.assertEqual(select_action(observation, "random", rng=random.Random(7))[1]["valid_actions"], [0, 2])

    def test_greedy_policy_tie_break_is_deterministic(self):
        observation = build_cup_mug_ordering_observation(
            [_target("mug_1", "mug", True), _target("cup_1", "cup", False), _target("cup_2", "cup", False)],
            [
                _assignment("mug_1", "mug", True, score=0.5, reachability=0.5),
                _assignment("cup_1", "cup", False, score=0.9, reachability=0.6),
                _assignment("cup_2", "cup", False, score=0.9, reachability=0.6),
            ],
            SORTING_METADATA,
        )

        action, debug = select_action(observation, "greedy")
        repeated_action, repeated_debug = select_action(observation, "greedy")

        self.assertEqual(action, 1)
        self.assertEqual(debug["ranked_actions"], [1, 2, 0])
        self.assertEqual(debug, repeated_debug)
        self.assertEqual(action, repeated_action)

    def test_policies_return_noop_for_all_invalid_input(self):
        observation = build_cup_mug_ordering_observation([])

        random_action, random_debug = select_action(observation, "random", rng=random.Random(3))
        greedy_action, greedy_debug = select_action(observation, "greedy")

        self.assertIsNone(random_action)
        self.assertIsNone(greedy_action)
        self.assertEqual(random_debug["reason"], "all_actions_invalid")
        self.assertEqual(greedy_debug["reason"], "all_actions_invalid")


class CupMugOrderingRunnerTests(unittest.TestCase):
    def test_mocked_five_object_episode_completes(self):
        call_slots = []

        def executor(slot, observation, step_index):
            call_slots.append((step_index, slot["slot_index"], slot["object_id"]))
            return {"low_level_success": True, "correct_zone": True, "failure_type": None, "trace": {"executor": "ok"}}

        runner = CupMugOrderingEpisodeRunner(build_cup_mug_ordering_observation, executor)
        names = ["mug_1", "mug_2", "cup_1", "cup_2", "cup_3"]
        targets = [_target(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]
        assignments = [_assignment(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]
        runner.reset(targets, assignments, SORTING_METADATA, episode_id="ep-1", seed=7)

        last_result = None
        for action in [0, 1, 2, 3, 4]:
            last_result = runner.step(action)

        self.assertIsNotNone(last_result)
        self.assertTrue(last_result["done"])
        self.assertTrue(last_result["success"])
        self.assertEqual(last_result["summary"]["completed_count"], 5)
        self.assertEqual(last_result["summary"]["selected_order"], [0, 1, 2, 3, 4])
        self.assertEqual(last_result["summary"]["invalid_action_count"], 0)
        self.assertEqual(len(call_slots), 5)

    def test_invalid_action_does_not_call_executor(self):
        executor_calls = []

        def executor(slot, observation, step_index):
            executor_calls.append(slot)
            return {"low_level_success": True, "correct_zone": True}

        runner = CupMugOrderingEpisodeRunner(build_cup_mug_ordering_observation, executor)
        runner.reset(
            [_target("mug_1", "mug", True), _target("cup_1", "cup", False)],
            [_assignment("mug_1", "mug", True), _assignment("cup_1", "cup", False)],
            SORTING_METADATA,
            episode_id="ep-2",
            seed=11,
        )
        runner._episode_state["finished"]["mug_1"] = True

        result = runner.step(0)

        self.assertEqual(executor_calls, [])
        self.assertFalse(result["step_log"]["valid_action"])
        self.assertEqual(result["step_log"]["failure_type"], "invalid_action")
        self.assertEqual(result["step_log"]["reward_terms"]["invalid_action"], -3.0)


class LiveSlotMappingTests(unittest.TestCase):
    def test_retry_resolution_preserves_preferred_object_identity(self):
        payload = {
            "observation": {
                "slots": [
                    {"slot_index": 0, "object_id": "mug_2", "valid_action": True},
                    {"slot_index": 1, "object_id": "mug_1", "valid_action": True},
                ]
            },
            "slot_mapping": {
                0: {
                    "slot": {"slot_index": 0, "object_id": "mug_2", "valid_action": True},
                    "target": {"sim_object_name": "mug_2"},
                    "assignment": {"sim_object_name": "mug_2"},
                    "is_missing": False,
                    "is_valid": True,
                },
                1: {
                    "slot": {"slot_index": 1, "object_id": "mug_1", "valid_action": True},
                    "target": {"sim_object_name": "mug_1"},
                    "assignment": {"sim_object_name": "mug_1"},
                    "is_missing": False,
                    "is_valid": True,
                },
            },
        }

        resolved = resolve_live_target_with_identity_preference(payload, 0, preferred_object_id="mug_1")

        self.assertTrue(resolved["is_valid"])
        self.assertEqual(resolved["slot"]["object_id"], "mug_1")
        self.assertTrue(resolved.get("identity_preserved"))

    def test_retry_resolution_falls_back_to_selected_slot_when_preferred_missing(self):
        targets = [
            _target("mug_2", "mug", True, score=0.9),
            _target("cup_1", "cup", False, score=0.6),
        ]
        assignments = [
            _assignment("mug_2", "mug", True, score=0.9),
            _assignment("cup_1", "cup", False, score=0.6),
        ]
        payload = build_live_slot_mapping(targets, assignments, SORTING_METADATA)

        resolved = resolve_live_target_with_identity_preference(payload, 0, preferred_object_id="mug_1")

        self.assertTrue(resolved["is_valid"])
        self.assertEqual(resolved["slot"]["object_id"], "mug_2")
        self.assertFalse(resolved.get("identity_preserved"))
        self.assertEqual(resolved.get("preferred_object_id"), "mug_1")

    def test_low_level_failure_leaves_slot_unfinished_and_records_failure(self):
        def executor(slot, observation, step_index):
            return {"low_level_success": False, "correct_zone": False, "failure_type": "grasp_failure"}

        runner = CupMugOrderingEpisodeRunner(build_cup_mug_ordering_observation, executor)
        runner.reset([_target("mug_1", "mug", True)], [_assignment("mug_1", "mug", True)], SORTING_METADATA)

        result = runner.step(0)
        summary = runner.summary()

        self.assertFalse(result["done"])
        self.assertEqual(result["failure_type"], None)
        self.assertEqual(summary["completed_count"], 0)
        self.assertEqual(summary["failure_counts"]["grasp_failure"], 1)
        self.assertEqual(summary["step_logs"][0]["failure_type"], "grasp_failure")

    def test_all_invalid_observation_finishes_episode(self):
        def executor(slot, observation, step_index):
            self.fail("executor should not be called for all-invalid observations")

        runner = CupMugOrderingEpisodeRunner(build_cup_mug_ordering_observation, executor)
        runner.reset([], [], SORTING_METADATA)

        result = runner.step(0)

        self.assertTrue(result["done"])
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_type"], "all_actions_invalid")


class CupMugOrderingMetricsTests(unittest.TestCase):
    def test_metrics_json_is_valid_and_contains_required_fields(self):
        record_a = build_episode_record(
            {
                "episode_id": "ep-1",
                "seed": 7,
                "selected_order": [0, 1, 2],
                "completed_slots": [0, 1, 2],
                "completed_count": 3,
                "success": True,
                "invalid_action_count": 0,
                "failure_counts": {},
                "reward_total": 10.0,
                "reward_breakdown": {"selected_target_success": 6.0, "correct_zone_place": 4.0},
                "step_logs": [],
                "max_targets": 5,
            },
            policy="greedy",
            duration_sec=1.2,
        )
        record_b = build_episode_record(
            {
                "episode_id": "ep-2",
                "seed": 8,
                "selected_order": [0, 4],
                "completed_slots": [0],
                "completed_count": 1,
                "success": False,
                "invalid_action_count": 1,
                "failure_counts": {"invalid_action": 1},
                "reward_total": -1.0,
                "reward_breakdown": {"invalid_action": -3.0, "selected_target_success": 2.0},
                "step_logs": [],
                "max_targets": 5,
            },
            policy="greedy",
            duration_sec=0.8,
        )
        metrics = aggregate_episode_records(
            [record_a, record_b],
            policy="greedy",
            episodes=2,
            seed_start=7,
            max_targets=5,
            scene_config={"layout_and_style_ids": [[1, 1]], "num_mugs": 2, "num_cups": 3},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = write_ordering_metrics(metrics, output_dir=temp_dir)
            payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

        self.assertEqual(payload["policy"], "greedy")
        self.assertEqual(payload["episodes"], 2)
        self.assertIn("success_rate", payload)
        self.assertIn("completed_count_mean", payload)
        self.assertIn("invalid_action_count", payload)
        self.assertIn("failure_counts", payload)
        self.assertIn("reward_breakdown", payload)
        self.assertEqual(payload["scene_config"]["num_mugs"], 2)
        self.assertEqual(payload["scene_config"]["num_cups"], 3)
        self.assertTrue(all(all(0 <= int(action) <= 4 for action in order) for order in payload["selected_orders"]))

    def test_rl_ordering_run_payload_is_nested(self):
        metrics = aggregate_episode_records(
            [],
            policy="random",
            episodes=0,
            seed_start=7,
            max_targets=5,
            scene_config={"layout_and_style_ids": [[1, 1]], "num_mugs": 2, "num_cups": 3},
        )
        run_payload = build_rl_ordering_run_payload(metrics)

        self.assertIn("rl_ordering", run_payload)
        self.assertEqual(run_payload["rl_ordering"]["policy"], "random")
        self.assertEqual(run_payload["rl_ordering"]["max_targets"], 5)


class CupMugLiveMappingTests(unittest.TestCase):
    def test_build_live_slot_mapping_preserves_stable_slot_resolution(self):
        names = ["cup_2", "mug_2", "cup_1", "mug_1", "cup_3"]
        targets = [_target(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]
        assignments = [_assignment(name, "mug" if name.startswith("mug") else "cup", name.startswith("mug")) for name in names]

        payload = build_live_slot_mapping(targets, assignments, SORTING_METADATA)

        self.assertEqual(payload["observation"]["action_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(payload["slot_mapping"][0]["slot"]["object_id"], "mug_1")
        self.assertEqual(payload["slot_mapping"][0]["assignment"]["sim_object_name"], "mug_1")
        self.assertEqual(payload["slot_mapping"][4]["target"]["sim_object_name"], "cup_3")

    def test_resolve_live_target_from_slot_handles_invalid_or_missing_slot(self):
        payload = build_live_slot_mapping(
            [_target("mug_1", "mug", True), _target("cup_1", "cup", False)],
            [_assignment("mug_1", "mug", True), _assignment("cup_1", "cup", False)],
            SORTING_METADATA,
            finished={"mug_1": True},
        )

        invalid_result = resolve_live_target_from_slot(payload, 0)
        missing_result = resolve_live_target_from_slot(payload, 4)
        valid_result = resolve_live_target_from_slot(payload, 1)

        self.assertFalse(invalid_result["is_valid"])
        self.assertEqual(invalid_result["failure_reason"], "slot_invalid")
        self.assertFalse(missing_result["is_valid"])
        self.assertEqual(missing_result["failure_reason"], "slot_invalid")
        self.assertTrue(valid_result["is_valid"])
        self.assertEqual(valid_result["target"]["sim_object_name"], "cup_1")


if __name__ == "__main__":
    unittest.main()
