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


def infer_failure_mode(counts):
    """Map retry counters to the required failure mode strings."""
    if counts["n2"] > 0:
        return "execution_drift"
    if counts["n3"] > 0:
        return "physical_slip"
    if counts["n1"] > 0:
        return "perception_error"
    return "success"


def infer_failure_modes_triggered(counts):
    """Return every failure mode triggered during the run."""
    modes = []
    if counts["n1"] > 0:
        modes.append("perception_error")
    if counts["n2"] > 0:
        modes.append("execution_drift")
    if counts["n3"] > 0:
        modes.append("physical_slip")
    return modes


def build_run_log(
    run_id,
    config_version,
    scene_config,
    success,
    counts,
    duration_sec,
    context,
):
    """Build a run log matching the required JSON schema."""
    return {
        "run_id": run_id,
        "commit_hash": get_commit_hash(),
        "config_version": config_version,
        "scene_config": scene_config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": bool(success),
        "failure_mode": infer_failure_mode(counts) if not success else "success",
        "failure_modes_triggered": (
            [] if success else infer_failure_modes_triggered(counts)
        ),
        "counts": counts,
        "duration_sec": float(duration_sec),
        "context": context,
    }


def write_run_log(run_log, output_dir="logs"):
    """Persist a run log JSON file and return the written path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"{run_log['run_id']}.json"
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    return log_path
