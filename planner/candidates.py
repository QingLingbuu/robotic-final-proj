"""Candidate selection and experimental handle-grasp transforms."""

import numpy as np

PANDA_GRIPPER_MAX_WIDTH = 0.08
HANDLE_ANCHOR_INSET = 0.03
HANDLE_TOP_DOWN_FALLBACK_INSET = 0.0
HANDLE_TOP_DOWN_FALLBACK_Z_BIAS = 0.012
HANDLE_TOP_DOWN_FALLBACK_WIDTH_MARGIN = 0.008
HANDLE_OBLIQUE_INSET = 0.02
HANDLE_OBLIQUE_APPROACH_Z_BIAS = -0.35
HANDLE_OBLIQUE_GRIPPER_WIDTH = 0.06


def choose_candidate(payload, candidate_id=None, grasp_type=None, max_gripper_width=None):
    candidates = payload.get("grasp_candidates", [])
    if not candidates:
        return None
    if grasp_type is not None:
        candidates = [candidate for candidate in candidates if candidate.get("grasp_type") == grasp_type]
        if not candidates:
            return None
    if max_gripper_width is not None:
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("gripper_width", 0.0)) <= float(max_gripper_width)
        ]
        if not candidates:
            return None
    if candidate_id is not None:
        for candidate in candidates:
            if int(candidate["id"]) == int(candidate_id):
                return candidate
        raise ValueError(f"Candidate id {candidate_id} was not present in the payload.")
    return max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def build_handle_anchor_top_down_candidate(payload, handle_candidate, inset=HANDLE_ANCHOR_INSET):
    top_down_candidate = choose_candidate(payload, grasp_type="top_down")
    if top_down_candidate is None:
        return None

    handle_pos = np.asarray(handle_candidate["pos"], dtype=float)
    top_down_pos = np.asarray(top_down_candidate["pos"], dtype=float)
    body_pos = np.asarray(payload.get("target", {}).get("pos", top_down_pos), dtype=float)

    outward_xy = handle_pos[:2] - body_pos[:2]
    outward_norm = float(np.linalg.norm(outward_xy))
    if outward_norm > 1e-8:
        anchor_xy = handle_pos[:2] - (outward_xy / outward_norm) * float(inset)
    else:
        anchor_xy = top_down_pos[:2]

    return {
        "id": int(handle_candidate["id"]),
        "pos": [float(anchor_xy[0]), float(anchor_xy[1]), float(top_down_pos[2])],
        "orientation": top_down_candidate["orientation"],
        "gripper_width": min(
            float(top_down_candidate.get("gripper_width", PANDA_GRIPPER_MAX_WIDTH)),
            PANDA_GRIPPER_MAX_WIDTH,
        ),
        "score": float(handle_candidate.get("score", 0.0)),
        "grasp_type": "handle_anchor_top_down",
        "source_handle_candidate_id": int(handle_candidate["id"]),
        "source_top_down_candidate_id": int(top_down_candidate["id"]),
    }


def build_handle_oblique_candidate(payload, handle_candidate, inset=HANDLE_OBLIQUE_INSET):
    handle_pos = np.asarray(handle_candidate["pos"], dtype=float)
    body_pos = np.asarray(payload.get("target", {}).get("pos", handle_pos), dtype=float)

    horizontal_approach = normalize_vector(np.asarray(handle_candidate["orientation"][2], dtype=float))
    if horizontal_approach is None:
        outward_xy = handle_pos[:2] - body_pos[:2]
        outward_norm = float(np.linalg.norm(outward_xy))
        if outward_norm < 1e-8:
            return None
        horizontal_approach = np.array([-outward_xy[0], -outward_xy[1], 0.0], dtype=float) / outward_norm

    approach = normalize_vector(
        horizontal_approach + np.array([0.0, 0.0, HANDLE_OBLIQUE_APPROACH_Z_BIAS], dtype=float)
    )
    closing = normalize_vector(np.asarray(handle_candidate["orientation"][0], dtype=float))
    if approach is None or closing is None:
        return None

    lateral = normalize_vector(np.cross(approach, closing))
    if lateral is None:
        return None
    closing = normalize_vector(np.cross(lateral, approach))
    if closing is None:
        return None

    grasp_pos = handle_pos + horizontal_approach * float(inset)
    return {
        "id": int(handle_candidate["id"]),
        "pos": [float(value) for value in grasp_pos.tolist()],
        "orientation": [
            [float(value) for value in closing.tolist()],
            [float(value) for value in lateral.tolist()],
            [float(value) for value in approach.tolist()],
        ],
        "gripper_width": min(HANDLE_OBLIQUE_GRIPPER_WIDTH, PANDA_GRIPPER_MAX_WIDTH),
        "score": float(handle_candidate.get("score", 0.0)),
        "grasp_type": "handle_oblique_grasp",
        "source_handle_candidate_id": int(handle_candidate["id"]),
        "source_handle_pos": [float(value) for value in handle_pos.tolist()],
        "source_handle_width": float(handle_candidate.get("gripper_width", 0.0)),
    }


def build_handle_top_down_from_side_candidate(payload, handle_candidate, inset=HANDLE_TOP_DOWN_FALLBACK_INSET):
    handle_pos = np.asarray(handle_candidate["pos"], dtype=float)
    body_pos = np.asarray(payload.get("target", {}).get("pos", handle_pos), dtype=float)
    outward_xy = handle_pos[:2] - body_pos[:2]
    outward_norm = float(np.linalg.norm(outward_xy))
    if outward_norm < 1e-8:
        return None

    outward_axis_xy = outward_xy / outward_norm
    closing = normalize_vector(np.array([outward_axis_xy[0], outward_axis_xy[1], 0.0], dtype=float))
    approach = np.array([0.0, 0.0, -1.0], dtype=float)
    lateral = normalize_vector(np.cross(approach, closing))
    if closing is None or lateral is None:
        return None

    top_down_candidate = choose_candidate(payload, grasp_type="top_down")
    grasp_z = float(handle_pos[2] + HANDLE_TOP_DOWN_FALLBACK_Z_BIAS)
    source_width = float(handle_candidate.get("gripper_width", PANDA_GRIPPER_MAX_WIDTH))
    gripper_width = min(source_width + HANDLE_TOP_DOWN_FALLBACK_WIDTH_MARGIN, PANDA_GRIPPER_MAX_WIDTH)
    grasp_xy = handle_pos[:2] - outward_axis_xy * float(inset)
    return {
        "id": int(handle_candidate["id"]),
        "pos": [float(grasp_xy[0]), float(grasp_xy[1]), grasp_z],
        "orientation": [
            [float(value) for value in closing.tolist()],
            [float(value) for value in lateral.tolist()],
            [float(value) for value in approach.tolist()],
        ],
        "gripper_width": gripper_width,
        "score": float(handle_candidate.get("score", 0.0)),
        "grasp_type": "handle_top_down",
        "source_handle_candidate_id": int(handle_candidate["id"]),
        "source_handle_pos": [float(value) for value in handle_pos.tolist()],
        "source_handle_width": source_width,
        "source_top_down_candidate_id": None if top_down_candidate is None else int(top_down_candidate["id"]),
    }


def choose_reachable_candidate(payload, candidate_id=None, grasp_type=None, handle_mode="off"):
    candidate = choose_candidate(payload, candidate_id=candidate_id, grasp_type=grasp_type)
    if candidate is None and grasp_type == "handle_top_down":
        handle_candidate = choose_candidate(payload, candidate_id=candidate_id, grasp_type="handle_grasp")
        if handle_candidate is not None:
            fallback = build_handle_top_down_from_side_candidate(payload, handle_candidate)
            reason = {
                "source_candidate_id": int(handle_candidate["id"]),
                "source_grasp_type": handle_candidate.get("grasp_type"),
                "source_gripper_width": float(handle_candidate.get("gripper_width", 0.0)),
                "requested_grasp_type": "handle_top_down",
                "strategy": "handle_top_down_from_handle_grasp",
                "fallback_grasp_type": None if fallback is None else fallback.get("grasp_type"),
            }
            return fallback, reason
    if candidate is None:
        return None, None

    if candidate.get("grasp_type") != "handle_grasp":
        return candidate, None

    if handle_mode == "side_reach":
        return candidate, {
            "source_candidate_id": int(candidate["id"]),
            "source_grasp_type": candidate.get("grasp_type"),
            "source_gripper_width": float(candidate.get("gripper_width", 0.0)),
            "strategy": "side_reach",
            "fallback_grasp_type": candidate.get("grasp_type"),
        }

    if handle_mode == "anchor_top_down":
        fallback = build_handle_anchor_top_down_candidate(payload, candidate)
        strategy = "handle_anchor_top_down"
    elif handle_mode == "oblique_reach":
        fallback = build_handle_oblique_candidate(payload, candidate)
        strategy = "handle_oblique_grasp"
    else:
        fallback = choose_candidate(payload, grasp_type="top_down")
        strategy = "top_down_default_fallback"

    reason = {
        "source_candidate_id": int(candidate["id"]),
        "source_grasp_type": candidate.get("grasp_type"),
        "source_gripper_width": float(candidate.get("gripper_width", 0.0)),
        "max_gripper_width": PANDA_GRIPPER_MAX_WIDTH,
        "handle_mode": handle_mode,
        "strategy": strategy,
        "fallback_grasp_type": None if fallback is None else fallback.get("grasp_type"),
    }
    return fallback, reason
