"""Deterministic high-level episode runner for CupMugSorting ordering."""

from copy import deepcopy


DEFAULT_REWARD_WEIGHTS = {
    "correct_zone_place": 5.0,
    "selected_target_success": 2.0,
    "grasp_failure": -2.0,
    "placement_failure": -2.0,
    "invalid_action": -3.0,
    "retry_penalty": -0.5,
    "timeout": -3.0,
    "collision": -5.0,
    "all_invalid": -4.0,
}


def _reward_value(reward_weights, key):
    return float((reward_weights or {}).get(key, DEFAULT_REWARD_WEIGHTS.get(key, 0.0)))


class CupMugOrderingEpisodeRunner:
    """Pure/mockable runner for five-slot cup/mug ordering episodes."""

    def __init__(self, observation_builder, low_level_executor, reward_weights=None, allowed_failed_attempts=0, max_targets=5):
        self.observation_builder = observation_builder
        self.low_level_executor = low_level_executor
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        self.reward_weights.update(reward_weights or {})
        self.allowed_failed_attempts = int(allowed_failed_attempts)
        self.max_targets = int(max_targets)
        self.max_decisions = self.max_targets + self.allowed_failed_attempts
        self._episode_state = None

    def reset(self, targets, assignments=None, sorting_metadata=None, episode_id=None, seed=None):
        self._episode_state = {
            "targets": list(targets or []),
            "assignments": list(assignments or []),
            "sorting_metadata": dict(sorting_metadata or {}),
            "finished": {},
            "retry_counts": {},
            "step_logs": [],
            "selected_order": [],
            "reward_total": 0.0,
            "reward_breakdown": {key: 0.0 for key in self.reward_weights},
            "invalid_action_count": 0,
            "failure_counts": {},
            "step_index": 0,
            "last_action": None,
            "last_success": None,
            "last_failure_type": None,
            "episode_id": episode_id,
            "seed": seed,
            "done": False,
            "success": False,
        }
        return self.get_observation()

    def get_observation(self):
        state = self._require_state()
        return self.observation_builder(
            state["targets"],
            state["assignments"],
            state["sorting_metadata"],
            finished=state["finished"],
            retry_counts=state["retry_counts"],
            step_index=state["step_index"],
            last_action=state["last_action"],
            last_success=state["last_success"],
            last_failure_type=state["last_failure_type"],
        )

    def step(self, action):
        state = self._require_state()
        if state["done"]:
            raise RuntimeError("Episode already finished. Call reset() before stepping again.")

        observation = self.get_observation()
        if not any(observation["action_mask"]):
            return self._finish_all_invalid(observation)

        step_index = int(state["step_index"])
        slot = observation["slots"][int(action)] if 0 <= int(action) < len(observation["slots"]) else None
        if slot is None or not bool(observation["action_mask"][int(action)]):
            return self._record_invalid_action(observation, action)

        state["selected_order"].append(int(action))
        executor_result = dict(self.low_level_executor(slot=deepcopy(slot), observation=deepcopy(observation), step_index=step_index) or {})
        low_level_success = bool(executor_result.get("low_level_success", False))
        correct_zone = bool(executor_result.get("correct_zone", False))
        failure_type = executor_result.get("failure_type")
        if not low_level_success and not failure_type:
            failure_type = "placement_failure"

        reward = 0.0
        reward_terms = {}
        if low_level_success:
            reward_terms["selected_target_success"] = _reward_value(self.reward_weights, "selected_target_success")
            reward += reward_terms["selected_target_success"]
            if correct_zone:
                reward_terms["correct_zone_place"] = _reward_value(self.reward_weights, "correct_zone_place")
                reward += reward_terms["correct_zone_place"]
                state["finished"][slot["object_id"]] = True
        else:
            failure_key = failure_type if failure_type in self.reward_weights else "placement_failure"
            reward_terms[failure_key] = _reward_value(self.reward_weights, failure_key)
            reward += reward_terms[failure_key]
            reward_terms["retry_penalty"] = _reward_value(self.reward_weights, "retry_penalty")
            reward += reward_terms["retry_penalty"]
            state["retry_counts"][slot["object_id"]] = int(state["retry_counts"].get(slot["object_id"], 0)) + 1
            state["failure_counts"][failure_type] = int(state["failure_counts"].get(failure_type, 0)) + 1

        state["reward_total"] += float(reward)
        for key, value in reward_terms.items():
            state["reward_breakdown"][key] = float(state["reward_breakdown"].get(key, 0.0)) + float(value)

        log_entry = {
            "step_index": step_index,
            "selected_slot": int(action),
            "object_id": slot.get("object_id"),
            "policy": executor_result.get("policy"),
            "valid_action": True,
            "has_handle": slot.get("has_handle"),
            "grasp_strategy": slot.get("recommended_grasp"),
            "place_zone": slot.get("place_zone_id"),
            "low_level_success": low_level_success,
            "correct_zone": correct_zone,
            "failure_type": failure_type,
            "reward": float(reward),
            "reward_terms": reward_terms,
            "trace": executor_result.get("trace"),
        }
        state["step_logs"].append(log_entry)
        state["step_index"] += 1
        state["last_action"] = int(action)
        state["last_success"] = low_level_success
        state["last_failure_type"] = failure_type

        next_observation = self.get_observation()
        done, success, terminal_failure_type = self._terminal_status(next_observation)
        if not done and state["step_index"] >= self.max_decisions:
            done = True
            success = False
            terminal_failure_type = "timeout"
            timeout_reward = _reward_value(self.reward_weights, "timeout")
            state["reward_total"] += timeout_reward
            state["reward_breakdown"]["timeout"] = float(state["reward_breakdown"].get("timeout", 0.0)) + timeout_reward
            state["failure_counts"]["timeout"] = int(state["failure_counts"].get("timeout", 0)) + 1
            state["step_logs"][-1]["reward"] = float(state["step_logs"][-1]["reward"]) + timeout_reward
            state["step_logs"][-1]["reward_terms"]["timeout"] = timeout_reward

        if done:
            state["done"] = True
            state["success"] = bool(success)
            state["last_failure_type"] = terminal_failure_type

        return {
            "observation": next_observation,
            "done": bool(done),
            "success": bool(success),
            "failure_type": terminal_failure_type,
            "reward": float(state["step_logs"][-1]["reward"]),
            "step_log": deepcopy(state["step_logs"][-1]),
            "summary": self.summary() if done else None,
        }

    def summary(self):
        state = self._require_state()
        observation = self.get_observation()
        completed_slots = [slot["slot_index"] for slot in observation["slots"] if slot.get("finished")]
        return {
            "episode_id": state.get("episode_id"),
            "seed": state.get("seed"),
            "selected_order": list(state["selected_order"]),
            "completed_slots": completed_slots,
            "completed_count": len(completed_slots),
            "success": bool(state["success"]),
            "invalid_action_count": int(state["invalid_action_count"]),
            "failure_counts": deepcopy(state["failure_counts"]),
            "reward_total": float(state["reward_total"]),
            "reward_breakdown": deepcopy(state["reward_breakdown"]),
            "step_logs": deepcopy(state["step_logs"]),
            "max_targets": self.max_targets,
        }

    def _record_invalid_action(self, observation, action):
        state = self._require_state()
        reward = _reward_value(self.reward_weights, "invalid_action")
        state["invalid_action_count"] += 1
        state["reward_total"] += reward
        state["reward_breakdown"]["invalid_action"] = float(state["reward_breakdown"].get("invalid_action", 0.0)) + reward
        state["failure_counts"]["invalid_action"] = int(state["failure_counts"].get("invalid_action", 0)) + 1
        step_log = {
            "step_index": int(state["step_index"]),
            "selected_slot": int(action),
            "object_id": None,
            "policy": None,
            "valid_action": False,
            "has_handle": None,
            "grasp_strategy": None,
            "place_zone": None,
            "low_level_success": False,
            "correct_zone": False,
            "failure_type": "invalid_action",
            "reward": reward,
            "reward_terms": {"invalid_action": reward},
            "trace": None,
        }
        state["step_logs"].append(step_log)
        state["step_index"] += 1
        state["last_action"] = int(action)
        state["last_success"] = False
        state["last_failure_type"] = "invalid_action"
        next_observation = self.get_observation()
        done, success, terminal_failure_type = self._terminal_status(next_observation)
        if done:
            state["done"] = True
            state["success"] = bool(success)
            state["last_failure_type"] = terminal_failure_type
        return {
            "observation": next_observation,
            "done": bool(done),
            "success": bool(success),
            "failure_type": terminal_failure_type,
            "reward": reward,
            "step_log": deepcopy(step_log),
            "summary": self.summary() if done else None,
        }

    def _finish_all_invalid(self, observation):
        state = self._require_state()
        reward = _reward_value(self.reward_weights, "all_invalid")
        state["done"] = True
        state["success"] = False
        state["last_failure_type"] = "all_actions_invalid"
        state["reward_total"] += reward
        state["reward_breakdown"]["all_invalid"] = float(state["reward_breakdown"].get("all_invalid", 0.0)) + reward
        state["failure_counts"]["all_actions_invalid"] = int(state["failure_counts"].get("all_actions_invalid", 0)) + 1
        return {
            "observation": deepcopy(observation),
            "done": True,
            "success": False,
            "failure_type": "all_actions_invalid",
            "reward": reward,
            "step_log": None,
            "summary": self.summary(),
        }

    def _terminal_status(self, observation):
        completed_count = int(observation["global"]["completed_count"])
        if completed_count >= self.max_targets:
            return True, True, None
        if not any(observation["action_mask"]):
            return True, False, "all_actions_invalid"
        return False, False, None

    def _require_state(self):
        if self._episode_state is None:
            raise RuntimeError("Runner has no active episode. Call reset() first.")
        return self._episode_state
