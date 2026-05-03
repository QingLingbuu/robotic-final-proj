"""Live slot-to-target mapping helpers for online CupMugSorting demos."""

from copy import deepcopy

from rl.cup_mug_observation import build_cup_mug_ordering_observation


def build_live_slot_mapping(
    targets,
    assignments,
    sorting_metadata,
    finished=None,
    retry_counts=None,
    conf_threshold=0.35,
    step_index=0,
    last_action=None,
    last_success=None,
    last_failure_type=None,
):
    """Build current live observation plus slot-index -> target/assignment mapping."""
    target_list = list(targets or [])
    assignment_list = list(assignments or [])
    observation = build_cup_mug_ordering_observation(
        target_list,
        assignment_list,
        sorting_metadata,
        finished=finished,
        retry_counts=retry_counts,
        conf_threshold=conf_threshold,
        step_index=step_index,
        last_action=last_action,
        last_success=last_success,
        last_failure_type=last_failure_type,
    )

    mapping = {}
    by_object_id = {}
    for index, target in enumerate(target_list):
        assignment = dict(assignment_list[index]) if index < len(assignment_list) else {}
        object_id = target.get("sim_object_name") or assignment.get("sim_object_name") or target.get("object_id") or assignment.get("object_id")
        if object_id is None:
            continue
        by_object_id[str(object_id)] = {
            "target": deepcopy(target),
            "assignment": deepcopy(assignment),
        }

    for slot in observation["slots"]:
        slot_index = int(slot["slot_index"])
        object_id = slot.get("object_id")
        entry = {
            "slot": deepcopy(slot),
            "target": None,
            "assignment": None,
            "is_missing": object_id is None,
            "is_valid": bool(slot.get("valid_action")),
        }
        if object_id is not None and str(object_id) in by_object_id:
            entry["target"] = by_object_id[str(object_id)]["target"]
            entry["assignment"] = by_object_id[str(object_id)]["assignment"]
        mapping[slot_index] = entry

    return {
        "observation": observation,
        "slot_mapping": mapping,
    }


def resolve_live_target_from_slot(slot_mapping_payload, selected_slot):
    """Resolve the current live target/assignment for one selected slot."""
    mapping = dict(slot_mapping_payload.get("slot_mapping", {}))
    slot_index = int(selected_slot)
    if slot_index not in mapping:
        return {
            "slot": None,
            "target": None,
            "assignment": None,
            "failure_reason": "slot_out_of_range",
            "is_valid": False,
        }

    entry = mapping[slot_index]
    if not entry.get("is_valid"):
        return {
            "slot": deepcopy(entry.get("slot")),
            "target": None,
            "assignment": None,
            "failure_reason": "slot_invalid",
            "is_valid": False,
        }
    if entry.get("target") is None or entry.get("assignment") is None:
        return {
            "slot": deepcopy(entry.get("slot")),
            "target": None,
            "assignment": None,
            "failure_reason": "target_missing_for_slot",
            "is_valid": False,
        }
    return {
        "slot": deepcopy(entry.get("slot")),
        "target": deepcopy(entry.get("target")),
        "assignment": deepcopy(entry.get("assignment")),
        "failure_reason": None,
        "is_valid": True,
    }


def resolve_live_target_with_identity_preference(slot_mapping_payload, selected_slot, preferred_object_id=None):
    """Resolve a slot, preferring the same object identity across retries when possible."""

    resolved = resolve_live_target_from_slot(slot_mapping_payload, selected_slot)
    if not resolved.get("is_valid"):
        return resolved

    if preferred_object_id is None:
        return resolved

    preferred = str(preferred_object_id)
    current_object_id = resolved.get("slot", {}).get("object_id")
    if current_object_id is not None and str(current_object_id) == preferred:
        return resolved

    mapping = dict(slot_mapping_payload.get("slot_mapping", {}))
    for entry in mapping.values():
        slot = dict(entry.get("slot") or {})
        if str(slot.get("object_id")) != preferred:
            continue
        if not entry.get("is_valid"):
            break
        if entry.get("target") is None or entry.get("assignment") is None:
            break
        return {
            "slot": deepcopy(slot),
            "target": deepcopy(entry.get("target")),
            "assignment": deepcopy(entry.get("assignment")),
            "failure_reason": None,
            "is_valid": True,
            "identity_preserved": True,
        }

    resolved["identity_preserved"] = False
    resolved["preferred_object_id"] = preferred
    return resolved
