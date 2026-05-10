"""Retry policy helpers for reach failures."""


def should_retry_with_preposition(
    reach_summary,
    reachability_diagnosis,
    candidate,
    final_error_threshold=0.03,
    allowed_grasp_types=("top_down", "handle_top_down"),
):
    if reach_summary.get("reach_success"):
        return False
    if reach_summary.get("reach_only"):
        return False
    if candidate.get("grasp_type") not in set(allowed_grasp_types):
        return False

    final_error = reachability_diagnosis.get("final_error")
    if reachability_diagnosis.get("position_saturated"):
        return True
    if reachability_diagnosis.get("likely_cause") == "position_control_saturated_or_base_torso_needed":
        return True
    if final_error is not None and float(final_error) >= float(final_error_threshold):
        return True
    return False
