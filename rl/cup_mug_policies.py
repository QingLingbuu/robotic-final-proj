"""Baseline ordering policies for CupMugSorting high-level RL."""

import random as random_module


SUPPORTED_POLICIES = {"random", "greedy"}


def _valid_action_indices(observation):
    return [index for index, value in enumerate(observation.get("action_mask", [])) if bool(value)]


def _slot_by_index(observation, slot_index):
    for slot in observation.get("slots", []):
        if int(slot.get("slot_index", -1)) == int(slot_index):
            return slot
    return {}


def _greedy_sort_key(slot):
    distance_to_place = float(slot.get("distance_to_place", 0.0) or 0.0)
    return (
        -float(slot.get("candidate_score", 0.0) or 0.0),
        -float(slot.get("conf", 0.0) or 0.0),
        -float(slot.get("reachability", 0.0) or 0.0),
        int(slot.get("retry_count", 0) or 0),
        distance_to_place,
        int(slot.get("slot_index", 0) or 0),
    )


def select_random_action(observation, rng=None):
    """Select one valid slot uniformly using a seedable RNG."""
    valid_actions = _valid_action_indices(observation)
    if not valid_actions:
        return None, {"policy": "random", "valid_actions": [], "reason": "all_actions_invalid"}
    resolved_rng = rng if rng is not None else random_module
    action = int(resolved_rng.choice(valid_actions))
    return action, {"policy": "random", "valid_actions": list(valid_actions)}


def select_greedy_action(observation):
    """Select the deterministic risk-aware greedy slot from valid actions."""
    valid_actions = _valid_action_indices(observation)
    if not valid_actions:
        return None, {"policy": "greedy", "valid_actions": [], "reason": "all_actions_invalid"}
    ranked_slots = sorted((_slot_by_index(observation, action) for action in valid_actions), key=_greedy_sort_key)
    selected = ranked_slots[0]
    action = int(selected.get("slot_index"))
    return action, {
        "policy": "greedy",
        "valid_actions": list(valid_actions),
        "selected_score": {
            "candidate_score": float(selected.get("candidate_score", 0.0) or 0.0),
            "conf": float(selected.get("conf", 0.0) or 0.0),
            "reachability": float(selected.get("reachability", 0.0) or 0.0),
            "retry_count": int(selected.get("retry_count", 0) or 0),
            "distance_to_place": float(selected.get("distance_to_place", 0.0) or 0.0),
            "slot_index": action,
        },
        "ranked_actions": [int(slot.get("slot_index")) for slot in ranked_slots],
    }


def select_action(observation, policy_name, rng=None):
    """Select an ordering action and return debug metadata for logging."""
    policy = str(policy_name).strip().lower()
    if policy == "random":
        return select_random_action(observation, rng=rng)
    if policy == "greedy":
        return select_greedy_action(observation)
    raise ValueError(f"Unsupported cup/mug ordering policy: {policy_name}")
