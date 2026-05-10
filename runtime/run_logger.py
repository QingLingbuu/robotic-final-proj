"""Helpers for recording experiment run logs."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def get_commit_hash():
    """Return the short git commit hash when available."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output.strip() or "unknown"


def infer_failure_mode(counts, execution_summary=None):
    """Map retry counters to the required failure mode strings."""
    if counts["n2"] > 0:
        return "execution_drift"
    if counts["n3"] > 0:
        return "physical_slip"
    if counts["n1"] > 0:
        return "perception_error"
    if execution_summary:
        for key in ("push_failure_mode", "grasp_failure_mode"):
            mode = execution_summary.get(key)
            if mode in {"execution_drift", "physical_slip", "perception_error"}:
                return mode
    return "success"


def infer_failure_modes_triggered(counts, execution_summary=None):
    """Return every failure mode triggered during the run."""
    modes = []
    if counts["n1"] > 0:
        modes.append("perception_error")
    if counts["n2"] > 0:
        modes.append("execution_drift")
    if counts["n3"] > 0:
        modes.append("physical_slip")
    if execution_summary:
        for key in ("push_failure_mode", "grasp_failure_mode"):
            mode = execution_summary.get(key)
            if mode in {"execution_drift", "physical_slip", "perception_error"} and mode not in modes:
                modes.append(mode)
    return modes


def extract_candidate_metadata(execution_summary=None):
    """Extract stable candidate-selection metadata from execution summary."""
    metadata = {
        "candidate_count": 0,
        "selected_candidate_id": None,
        "selected_candidate_score": None,
        "failure_stage": None,
    }
    if not execution_summary:
        return metadata

    selected_candidate = execution_summary.get("selected_candidate")
    if isinstance(selected_candidate, dict):
        metadata["selected_candidate_id"] = selected_candidate.get("id")
        score = selected_candidate.get("score")
        metadata["selected_candidate_score"] = None if score is None else float(score)
        metadata["candidate_count"] = 1

    for key in ("planning_candidates", "grasp_candidates"):
        candidates = execution_summary.get(key)
        if isinstance(candidates, list):
            metadata["candidate_count"] = len(candidates)
            break

    diagnostics = execution_summary.get("dual_arm_execution_diagnostics")
    if isinstance(diagnostics, dict):
        latest_attempt = diagnostics.get("latest_attempt")
        if isinstance(latest_attempt, dict):
            metadata["failure_stage"] = latest_attempt.get("failed_stage")

    if metadata["failure_stage"] is None:
        metadata["failure_stage"] = execution_summary.get("failure_stage")

    return metadata


def build_run_log(
    run_id,
    config_version,
    scene_config,
    success,
    counts,
    duration_sec,
    context,
    execution_summary=None,
    extra_payload=None,
):
    """Build a run log matching the required JSON schema."""
    candidate_metadata = extract_candidate_metadata(execution_summary)
    run_log = {
        "run_id": run_id,
        "commit_hash": get_commit_hash(),
        "config_version": config_version,
        "scene_config": scene_config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": bool(success),
        "failure_mode": (
            infer_failure_mode(counts, execution_summary) if not success else "success"
        ),
        "failure_modes_triggered": (
            []
            if success
            else infer_failure_modes_triggered(counts, execution_summary)
        ),
        "counts": counts,
        "duration_sec": float(duration_sec),
        "context": context,
        **candidate_metadata,
    }
    if execution_summary is not None:
        run_log["execution_summary"] = execution_summary
    if extra_payload is not None:
        run_log.update(extra_payload)
    return run_log


def write_run_log(run_log, output_dir="logs"):
    """Persist a run log JSON file and return the written path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"{run_log['run_id']}.json"
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    return log_path
