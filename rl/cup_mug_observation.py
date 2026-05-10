"""Observation and action-mask helpers for CupMugSorting ordering RL."""

import math


MAX_TARGETS = 5
DEFAULT_SLOT_OBJECT_IDS = ("mug_1", "mug_2", "cup_1", "cup_2", "cup_3")
DEFAULT_CONF_THRESHOLD = 0.35
HANDLED_TYPE_ID = 1
PLAIN_TYPE_ID = 0
UNKNOWN_TYPE_ID = -1
SINK_ZONE_ID = 1
OPPOSITE_COUNTER_ZONE_ID = 0
UNKNOWN_ZONE_ID = -1


def _float_or_default(value, default=0.0):
    if value is None:
        return float(default)
    return float(value)


def _position_or_default(value):
    if value is None:
        return [0.0, 0.0, 0.0]
    values = list(value)
    padded = values[:3] + [0.0] * max(0, 3 - len(values))
    return [float(item) for item in padded[:3]]


def _best_candidate_score(target):
    scores = [
        float(candidate.get("score", 0.0))
        for candidate in target.get("grasp_candidates", [])
        if candidate.get("score") is not None
    ]
    if not scores:
        return 0.0
    return max(scores)


def _target_object_id(target, assignment):
    for key in ("object_id", "sim_object_name", "name"):
        value = target.get(key)
        if value:
            return str(value)
    for key in ("object_id", "sim_object_name", "name"):
        value = assignment.get(key)
        if value:
            return str(value)
    return None


def _metadata_for_object(object_id, sorting_metadata):
    if object_id is None:
        return {}
    return dict((sorting_metadata or {}).get(object_id, {}) or {})


def _resolve_has_handle(target, assignment, metadata):
    if "has_handle" in metadata:
        return bool(metadata["has_handle"])
    if assignment.get("has_handle") is not None:
        return bool(assignment["has_handle"])
    if target.get("sim_has_handle") is not None:
        return bool(target["sim_has_handle"])
    return None


def _type_id(has_handle):
    if has_handle is None:
        return UNKNOWN_TYPE_ID
    return HANDLED_TYPE_ID if has_handle else PLAIN_TYPE_ID


def _zone_id(has_handle, metadata, assignment):
    target_zone = metadata.get("target_zone") or assignment.get("place_zone") or assignment.get("target_zone")
    if target_zone in {"sink", "handled"}:
        return SINK_ZONE_ID
    if target_zone in {"opposite_counter", "plain"}:
        return OPPOSITE_COUNTER_ZONE_ID
    if has_handle is True:
        return SINK_ZONE_ID
    if has_handle is False:
        return OPPOSITE_COUNTER_ZONE_ID
    return UNKNOWN_ZONE_ID


def _fallback_sort_key(item):
    original_index, target, assignment = item
    has_handle = _resolve_has_handle(target, assignment, {})
    handle_rank = 0 if has_handle else 1
    label = str(target.get("label") or assignment.get("label") or "unknown")
    pos = _position_or_default(target.get("pos") or assignment.get("pos"))
    return (handle_rank, label, pos[0], pos[1], pos[2], original_index)


def _ordered_items(targets, assignments):
    raw_items = _raw_items(targets, assignments)

    by_object_id = {
        _target_object_id(target, assignment): (index, target, assignment)
        for index, target, assignment in raw_items
        if _target_object_id(target, assignment) is not None
    }
    if by_object_id:
        ordered = []
        used_ids = set()
        for object_id in DEFAULT_SLOT_OBJECT_IDS:
            item = by_object_id.get(object_id)
            if item is not None:
                ordered.append(item)
                used_ids.add(object_id)
        leftovers = [item for item in raw_items if _target_object_id(item[1], item[2]) not in used_ids]
        ordered.extend(sorted(leftovers, key=_fallback_sort_key))
        return ordered
    return sorted(raw_items, key=_fallback_sort_key)


def _raw_items(targets, assignments):
    raw_items = []
    for index, target in enumerate(targets):
        assignment = dict(assignments[index]) if index < len(assignments) else {}
        raw_items.append((index, dict(target), assignment))
    return raw_items


def _duplicate_object_ids(items):
    seen = set()
    duplicates = set()
    for _, target, assignment in items:
        object_id = _target_object_id(target, assignment)
        if object_id is None:
            continue
        if object_id in seen:
            duplicates.add(object_id)
        seen.add(object_id)
    return duplicates


def _slot_from_item(slot_index, item, sorting_metadata, finished_lookup, retry_lookup, conf_threshold, duplicate_ids):
    _, target, assignment = item
    object_id = _target_object_id(target, assignment)
    metadata = _metadata_for_object(object_id, sorting_metadata)
    has_handle = _resolve_has_handle(target, assignment, metadata)
    conf = _float_or_default(target.get("conf", assignment.get("conf", 0.0)))
    candidate_score = _float_or_default(assignment.get("candidate_score"), _best_candidate_score(target))
    finished = bool(finished_lookup.get(object_id, finished_lookup.get(slot_index, False)))
    retry_count = int(retry_lookup.get(object_id, retry_lookup.get(slot_index, 0)))
    visible = bool(target.get("visible", True))
    place_zone_id = _zone_id(has_handle, metadata, assignment)
    valid_action = bool(
        object_id is not None
        and object_id not in duplicate_ids
        and visible
        and not finished
        and conf >= float(conf_threshold)
        and has_handle is not None
        and place_zone_id != UNKNOWN_ZONE_ID
        and candidate_score > 0.0
    )
    return {
        "slot_index": int(slot_index),
        "object_id": object_id,
        "label": str(target.get("label") or assignment.get("label") or "unknown"),
        "visible": visible,
        "finished": finished,
        "has_handle": has_handle,
        "type_id": _type_id(has_handle),
        "conf": conf,
        "pos": _position_or_default(target.get("pos") or assignment.get("pos")),
        "candidate_score": candidate_score,
        "reachability": _float_or_default(assignment.get("reachability", target.get("reachability", 0.0))),
        "place_zone_id": place_zone_id,
        "retry_count": retry_count,
        "valid_action": valid_action,
        "recommended_grasp": metadata.get("recommended_grasp") or assignment.get("strategy"),
    }


def _padding_slot(slot_index):
    return {
        "slot_index": int(slot_index),
        "object_id": None,
        "label": "unknown",
        "visible": False,
        "finished": False,
        "has_handle": None,
        "type_id": UNKNOWN_TYPE_ID,
        "conf": 0.0,
        "pos": [0.0, 0.0, 0.0],
        "candidate_score": 0.0,
        "reachability": 0.0,
        "place_zone_id": UNKNOWN_ZONE_ID,
        "retry_count": 0,
        "valid_action": False,
        "recommended_grasp": None,
    }


def build_cup_mug_ordering_observation(
    targets,
    assignments=None,
    sorting_metadata=None,
    finished=None,
    retry_counts=None,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    step_index=0,
    last_action=None,
    last_success=None,
    last_failure_type=None,
):
    """Build the fixed five-slot observation and action mask for cup/mug ordering."""
    target_list = list(targets or [])
    assignment_list = list(assignments or [])
    duplicate_ids = _duplicate_object_ids(_raw_items(target_list, assignment_list))
    ordered = _ordered_items(target_list, assignment_list)
    finished_lookup = dict(finished or {})
    retry_lookup = dict(retry_counts or {})

    slots = []
    for slot_index, item in enumerate(ordered[:MAX_TARGETS]):
        slots.append(
            _slot_from_item(
                slot_index=slot_index,
                item=item,
                sorting_metadata=sorting_metadata or {},
                finished_lookup=finished_lookup,
                retry_lookup=retry_lookup,
                conf_threshold=conf_threshold,
                duplicate_ids=duplicate_ids,
            )
        )
    while len(slots) < MAX_TARGETS:
        slots.append(_padding_slot(len(slots)))

    action_mask = [1 if slot["valid_action"] else 0 for slot in slots]
    finished_mask = [1 if slot["finished"] else 0 for slot in slots]
    completed_count = sum(finished_mask)
    remaining_count = sum(1 for slot in slots if slot["object_id"] is not None and not slot["finished"])
    return {
        "max_targets": MAX_TARGETS,
        "slots": slots,
        "action_mask": action_mask,
        "finished_mask": finished_mask,
        "global": {
            "remaining_count": int(remaining_count),
            "completed_count": int(completed_count),
            "step_index": int(step_index),
            "last_action": last_action,
            "last_success": last_success,
            "last_failure_type": last_failure_type,
        },
    }


def distance_to_zone(slot, zone_pos):
    """Small utility for later greedy ranking; kept pure for tests."""
    if zone_pos is None:
        return 0.0
    pos = slot.get("pos") or [0.0, 0.0, 0.0]
    return float(math.dist(_position_or_default(pos), _position_or_default(zone_pos)))
