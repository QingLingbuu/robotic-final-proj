"""Lightweight handle reach phase planning helpers."""

import numpy as np

HANDLE_STANDOFF = 0.10
HANDLE_PRE_APPROACH_Z_OFFSET = 0.05
HANDLE_OBLIQUE_STANDOFF = 0.12
HANDLE_OBLIQUE_PRE_Z_OFFSET = 0.08


def build_handle_reach_phases(
    candidate_pos,
    approach_dir,
    target_rot,
    reach_tolerance,
    standoff=HANDLE_STANDOFF,
    include_contact_approach=False,
):
    pre_approach_target = candidate_pos - np.asarray(approach_dir, dtype=float) * float(standoff)
    pre_approach_target = np.asarray(pre_approach_target, dtype=float)
    pre_approach_target[2] = candidate_pos[2] + HANDLE_PRE_APPROACH_Z_OFFSET
    phases = [
        ("pre_approach", pre_approach_target, reach_tolerance * 3, None, 0.20),
        ("align_orientation", pre_approach_target.copy(), reach_tolerance * 4, target_rot, 0.20),
    ]
    if include_contact_approach:
        phases.append(("approach", candidate_pos.copy(), reach_tolerance, target_rot, 0.20))
    return phases, pre_approach_target


def build_oblique_reach_phases(candidate_pos, approach_dir, target_rot, reach_tolerance, include_contact_approach=False):
    pre_approach_target = candidate_pos - np.asarray(approach_dir, dtype=float) * HANDLE_OBLIQUE_STANDOFF
    pre_approach_target[2] = candidate_pos[2] + HANDLE_OBLIQUE_PRE_Z_OFFSET
    phases = [
        ("oblique_pre_approach", pre_approach_target, reach_tolerance * 4, None, 0.25),
        ("oblique_align_orientation", pre_approach_target.copy(), reach_tolerance * 5, target_rot, 0.25),
    ]
    if include_contact_approach:
        phases.append(("oblique_approach", candidate_pos.copy(), reach_tolerance * 3, target_rot, 0.25))
    return phases, pre_approach_target
