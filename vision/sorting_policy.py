"""Drinkware sorting policy helpers for cup / mug classification demos."""

import math


def _best_candidate(candidates, grasp_type):
    matches = [
        candidate
        for candidate in candidates
        if str(candidate.get("grasp_type", "")).strip().lower() == grasp_type
    ]
    if not matches:
        return None
    return max(matches, key=lambda candidate: float(candidate.get("score", 0.0)))


DEFAULT_HANDLE_TOP_DOWN_MIN_SCORE = 0.30
DEFAULT_HANDLE_POINT_MIN_COUNT = 24
DEFAULT_HANDLE_PROTRUSION_MIN_QUALITY = 0.45


def _candidate_diagnostics(target, candidate):
    diagnostics = target.get("diagnostics", {})
    candidate_id = candidate.get("id")
    if candidate_id is None:
        return {}
    return diagnostics.get(str(candidate_id), {}) or diagnostics.get(candidate_id, {}) or {}


def _has_handle_evidence(
    target,
    candidate,
    min_score,
    min_point_count=DEFAULT_HANDLE_POINT_MIN_COUNT,
    min_protrusion_quality=DEFAULT_HANDLE_PROTRUSION_MIN_QUALITY,
):
    if candidate is None:
        return False
    if float(candidate.get("score", 0.0)) >= float(min_score):
        return True
    diagnostics = _candidate_diagnostics(target, candidate)
    return (
        int(diagnostics.get("handle_point_count", 0)) >= int(min_point_count)
        and float(diagnostics.get("protrusion_quality", 0.0)) >= float(min_protrusion_quality)
    )


def assign_sim_metadata_to_targets(targets, sim_objects, max_xy_distance=0.18, unmatched_fallback_max_xy_distance=None):
    """Attach nearest CupMugSorting sim metadata to visual targets."""
    assigned = []
    used_names = set()
    for target in targets:
        target_pos = target.get("pos")
        best = None
        if target_pos is not None:
            for sim_object in sim_objects:
                sim_pos = sim_object.get("pos")
                if sim_pos is None:
                    continue
                distance = math.dist(
                    [float(target_pos[0]), float(target_pos[1])],
                    [float(sim_pos[0]), float(sim_pos[1])],
                )
                if best is None or distance < best[0]:
                    best = (distance, sim_object)
        corrected = dict(target)
        if best is not None and best[0] <= float(max_xy_distance):
            corrected["sim_object_name"] = best[1].get("name")
            corrected["sim_has_handle"] = bool(best[1].get("has_handle"))
            corrected["sim_metadata_distance_xy"] = float(best[0])
            if best[1].get("name") is not None:
                used_names.add(str(best[1].get("name")))
        assigned.append(corrected)

    fallback_limit = max_xy_distance if unmatched_fallback_max_xy_distance is None else float(unmatched_fallback_max_xy_distance)
    if fallback_limit > float(max_xy_distance):
        for corrected in assigned:
            if corrected.get("sim_object_name") is not None:
                continue
            target_pos = corrected.get("pos")
            if target_pos is None:
                continue
            best = None
            for sim_object in sim_objects:
                sim_name = sim_object.get("name")
                if sim_name is None or str(sim_name) in used_names:
                    continue
                sim_pos = sim_object.get("pos")
                if sim_pos is None:
                    continue
                distance = math.dist(
                    [float(target_pos[0]), float(target_pos[1])],
                    [float(sim_pos[0]), float(sim_pos[1])],
                )
                if best is None or distance < best[0]:
                    best = (distance, sim_object)
            if best is not None and best[0] <= fallback_limit:
                corrected["sim_object_name"] = best[1].get("name")
                corrected["sim_has_handle"] = bool(best[1].get("has_handle"))
                corrected["sim_metadata_distance_xy"] = float(best[0])
                if best[1].get("name") is not None:
                    used_names.add(str(best[1].get("name")))
    return assigned


def classify_drinkware_targets(targets, handle_top_down_min_score=DEFAULT_HANDLE_TOP_DOWN_MIN_SCORE):
    """Assign grasp strategy and arm suggestion for detected drinkware targets."""
    assignments = []
    for index, target in enumerate(targets, start=1):
        candidates = list(target.get("grasp_candidates", []))
        raw_handle_candidate = _best_candidate(candidates, "handle_top_down")
        sim_has_handle = target.get("sim_has_handle")
        if sim_has_handle is None:
            has_handle = _has_handle_evidence(
                target,
                raw_handle_candidate,
                min_score=handle_top_down_min_score,
            )
        else:
            has_handle = bool(sim_has_handle)
        handle_candidate = raw_handle_candidate if has_handle else None
        top_down_candidate = _best_candidate(candidates, "top_down")

        if handle_candidate is not None:
            selected = handle_candidate
            strategy = "handle_top_down"
            arm = "right"
            has_handle = True
        else:
            selected = top_down_candidate or (
                max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))
                if candidates
                else None
            )
            strategy = "top_down" if top_down_candidate is not None else None
            arm = "left" if top_down_candidate is not None else None
            has_handle = False

        assignments.append(
            {
                "object_index": index,
                "label": target.get("label"),
                "conf": target.get("conf"),
                "pos": target.get("pos"),
                "has_handle": has_handle,
                "strategy": strategy,
                "arm": arm,
                "sim_object_name": target.get("sim_object_name"),
                "sim_metadata_distance_xy": target.get("sim_metadata_distance_xy"),
                "candidate_id": None if selected is None else selected.get("id"),
                "candidate_score": None if selected is None else selected.get("score"),
                "raw_handle_top_down_score": None
                if raw_handle_candidate is None
                else raw_handle_candidate.get("score"),
                "handle_top_down_min_score": float(handle_top_down_min_score),
            }
        )
    return assignments


def _requested_label_matches_assignment(requested_label, assignment):
    requested = str(requested_label).strip().lower()
    if requested in {"cup", "glass cup", "plain cup"}:
        return not bool(assignment.get("has_handle"))
    if requested in {"mug", "handled cup", "handled mug"}:
        return bool(assignment.get("has_handle"))
    return str(assignment.get("label", "")).strip().lower() == requested


def select_drinkware_target(
    targets,
    requested_label,
    handle_top_down_min_score=DEFAULT_HANDLE_TOP_DOWN_MIN_SCORE,
):
    """Select a target after classifying drinkware by handle availability."""
    target_list = list(targets)
    assignments = classify_drinkware_targets(
        target_list,
        handle_top_down_min_score=handle_top_down_min_score,
    )
    matches = [
        (target_list[int(assignment["object_index"]) - 1], assignment)
        for assignment in assignments
        if _requested_label_matches_assignment(requested_label, assignment)
    ]
    if not matches:
        return None, None

    def score(pair):
        target, assignment = pair
        candidate_score = assignment.get("candidate_score")
        if candidate_score is None:
            candidate_score = 0.0
        return float(target.get("conf", 0.0)), float(candidate_score)

    return max(matches, key=score)


def _requested_wants_handle(requested_label):
    requested = str(requested_label).strip().lower()
    if requested in {"cup", "glass cup", "plain cup"}:
        return False
    if requested in {"mug", "handled cup", "handled mug"}:
        return True
    return None


def build_sim_metadata_fallback_target(sim_objects, requested_label):
    """Build a fallback target from CupMugSorting sim metadata when vision misses the class."""
    wants_handle = _requested_wants_handle(requested_label)
    if wants_handle is None:
        return None, None
    matches = [
        sim_object
        for sim_object in sim_objects
        if bool(sim_object.get("has_handle")) == bool(wants_handle)
        and sim_object.get("pos") is not None
    ]
    if not matches:
        return None, None
    selected = sorted(matches, key=lambda item: str(item.get("name", "")))[0]
    pos = [float(value) for value in selected["pos"]]
    label = "mug" if wants_handle else "cup"
    grasp_type = "handle_top_down" if wants_handle else "top_down"
    candidate = {
        "id": 1,
        "pos": pos,
        "orientation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        "gripper_width": 0.08 if not wants_handle else 0.03,
        "score": 1.0,
        "grasp_type": grasp_type,
    }
    target = {
        "label": label,
        "conf": 1.0,
        "pos": pos,
        "grasp_candidates": [candidate],
        "diagnostics": {"source": "sim_metadata_fallback"},
        "sim_object_name": selected.get("name"),
        "sim_has_handle": bool(wants_handle),
        "sim_metadata_distance_xy": 0.0,
    }
    assignment = {
        "object_index": None,
        "label": label,
        "conf": 1.0,
        "pos": pos,
        "has_handle": bool(wants_handle),
        "strategy": grasp_type,
        "arm": "right" if wants_handle else "left",
        "sim_object_name": selected.get("name"),
        "sim_metadata_distance_xy": 0.0,
        "candidate_id": 1,
        "candidate_score": 1.0,
        "fallback_source": "sim_metadata",
    }
    return target, assignment
