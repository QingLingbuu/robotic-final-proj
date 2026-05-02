"""Lightweight handle reach-only validation helpers."""

HANDLE_REACH_STABLE_POS_TOLERANCE = 0.025
HANDLE_REACH_STABLE_ORI_TOLERANCE = 0.35


def summarize_handle_reach_validation(phase_results, final_error_to_candidate):
    failed_phases = [result["phase"] for result in phase_results if not result.get("ok")]
    final_phase = phase_results[-1] if phase_results else {}
    final_pos_error = final_phase.get("pos_error_norm")
    final_ori_error = final_phase.get("ori_error_norm")
    stable_position = final_pos_error is not None and float(final_pos_error) <= HANDLE_REACH_STABLE_POS_TOLERANCE
    stable_orientation = final_ori_error is not None and float(final_ori_error) <= HANDLE_REACH_STABLE_ORI_TOLERANCE
    return {
        "mode": "handle_reach_only",
        "stable_reach": bool(not failed_phases and stable_position and stable_orientation),
        "failed_phases": failed_phases,
        "final_phase": final_phase.get("phase"),
        "final_pos_error_norm": None if final_pos_error is None else float(final_pos_error),
        "final_ori_error_norm": None if final_ori_error is None else float(final_ori_error),
        "final_error_to_candidate": float(final_error_to_candidate),
        "stable_pos_tolerance": HANDLE_REACH_STABLE_POS_TOLERANCE,
        "stable_ori_tolerance": HANDLE_REACH_STABLE_ORI_TOLERANCE,
    }
