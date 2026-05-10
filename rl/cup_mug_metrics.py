"""Metrics and serialization helpers for cup/mug ordering RL dry-runs."""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def build_episode_record(summary, policy, duration_sec=0.0):
    """Build one JSON-serializable episode record from a runner summary."""
    payload = deepcopy(summary)
    payload["policy"] = policy
    payload["duration_sec"] = float(duration_sec)
    return payload


def aggregate_episode_records(records, policy, episodes, seed_start, max_targets, scene_config):
    """Aggregate episode records into one metrics payload."""
    episode_list = [deepcopy(record) for record in records]
    success_count = sum(1 for record in episode_list if bool(record.get("success")))
    completed_counts = [int(record.get("completed_count", 0)) for record in episode_list]
    invalid_action_count = sum(int(record.get("invalid_action_count", 0)) for record in episode_list)
    failure_counts = {}
    reward_breakdown = {}
    selected_orders = []
    for record in episode_list:
        selected_orders.append([int(action) for action in record.get("selected_order", [])])
        for key, value in dict(record.get("failure_counts", {})).items():
            failure_counts[key] = int(failure_counts.get(key, 0)) + int(value)
        for key, value in dict(record.get("reward_breakdown", {})).items():
            reward_breakdown[key] = float(reward_breakdown.get(key, 0.0)) + float(value)

    return {
        "policy": policy,
        "episodes": int(episodes),
        "success_rate": float(success_count / max(len(episode_list), 1)),
        "completed_count_mean": float(sum(completed_counts) / max(len(completed_counts), 1)),
        "invalid_action_count": int(invalid_action_count),
        "failure_counts": failure_counts,
        "selected_orders": selected_orders,
        "reward_breakdown": reward_breakdown,
        "seed_start": int(seed_start),
        "max_targets": int(max_targets),
        "scene_config": deepcopy(scene_config),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes_detail": episode_list,
    }


def write_ordering_metrics(metrics_payload, output_dir, file_name="cup_mug_ordering_metrics.json"):
    """Write cup/mug ordering metrics JSON to disk."""
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolved_dir / file_name
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_path


def build_rl_ordering_run_payload(metrics_payload):
    """Build a nested run-log-compatible RL ordering payload."""
    return {
        "rl_ordering": {
            "policy": metrics_payload["policy"],
            "episodes": metrics_payload["episodes"],
            "success_rate": metrics_payload["success_rate"],
            "completed_count_mean": metrics_payload["completed_count_mean"],
            "invalid_action_count": metrics_payload["invalid_action_count"],
            "failure_counts": deepcopy(metrics_payload["failure_counts"]),
            "selected_orders": deepcopy(metrics_payload["selected_orders"]),
            "reward_breakdown": deepcopy(metrics_payload["reward_breakdown"]),
            "max_targets": metrics_payload["max_targets"],
        }
    }
