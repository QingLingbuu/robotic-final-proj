"""Contracts for the milestone-1 dual-arm RoboCasa RL baseline."""

from copy import deepcopy
from pathlib import Path

import yaml


REQUIRED_TOP_LEVEL_KEYS = {
    "task_name",
    "backend",
    "policy",
    "observation",
    "action",
    "reward",
    "termination",
    "metrics",
    "train",
    "eval",
    "artifacts",
    "render",
}

REQUIRED_POLICY_KEYS = {"topology", "shared_policy"}
REQUIRED_OBSERVATION_KEYS = {"source", "fields"}
REQUIRED_ACTION_KEYS = {"type", "dimensions"}
REQUIRED_REWARD_KEYS = {"coefficients"}
REQUIRED_TERMINATION_KEYS = {"max_steps", "success_conditions", "failure_conditions"}
REQUIRED_METRICS_KEYS = {"primary", "report"}
REQUIRED_TRAIN_KEYS = {"smoke_steps", "checkpoint_every", "n_steps", "progress_print_freq"}
REQUIRED_EVAL_KEYS = {"seed_set", "episodes"}
REQUIRED_ARTIFACT_KEYS = {"root_dir", "checkpoint_dir", "metrics_dir", "video_dir"}
REQUIRED_RENDER_KEYS = {"enabled", "mode"}


class RLConfigError(ValueError):
    """Raised when the RL baseline config violates the milestone contract."""


def load_rl_config(config_path):
    """Load one RL config from YAML."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise RLConfigError(f"RL config not found: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RLConfigError("RL config must deserialize to a mapping.")
    return data


def _ensure_keys(section_name, payload, required_keys):
    if not isinstance(payload, dict):
        raise RLConfigError(f"{section_name} must be a mapping.")
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise RLConfigError(f"{section_name} missing required keys: {', '.join(missing)}")


def validate_rl_config(config):
    """Validate the milestone-1 dual-arm RoboCasa RL config contract."""
    _ensure_keys("config", config, REQUIRED_TOP_LEVEL_KEYS)
    _ensure_keys("policy", config["policy"], REQUIRED_POLICY_KEYS)
    _ensure_keys("observation", config["observation"], REQUIRED_OBSERVATION_KEYS)
    _ensure_keys("action", config["action"], REQUIRED_ACTION_KEYS)
    _ensure_keys("reward", config["reward"], REQUIRED_REWARD_KEYS)
    _ensure_keys("termination", config["termination"], REQUIRED_TERMINATION_KEYS)
    _ensure_keys("metrics", config["metrics"], REQUIRED_METRICS_KEYS)
    _ensure_keys("train", config["train"], REQUIRED_TRAIN_KEYS)
    _ensure_keys("eval", config["eval"], REQUIRED_EVAL_KEYS)
    _ensure_keys("artifacts", config["artifacts"], REQUIRED_ARTIFACT_KEYS)
    _ensure_keys("render", config["render"], REQUIRED_RENDER_KEYS)

    if config["backend"] not in {"robocasa", "robosuite"}:
        raise RLConfigError("RL baseline backend must be 'robocasa' or 'robosuite'.")
    if config["observation"]["source"] != "privileged_state":
        raise RLConfigError("Milestone-1 observation.source must be 'privileged_state'.")
    if not isinstance(config["observation"]["fields"], list) or not config["observation"]["fields"]:
        raise RLConfigError("observation.fields must be a non-empty list.")
    if int(config["action"]["dimensions"]) <= 0:
        raise RLConfigError("action.dimensions must be a positive integer.")
    if int(config["termination"]["max_steps"]) <= 0:
        raise RLConfigError("termination.max_steps must be a positive integer.")
    if not isinstance(config["reward"]["coefficients"], dict) or not config["reward"]["coefficients"]:
        raise RLConfigError("reward.coefficients must be a non-empty mapping.")
    if not isinstance(config["eval"]["seed_set"], list) or not config["eval"]["seed_set"]:
        raise RLConfigError("eval.seed_set must be a non-empty list.")
    if int(config["eval"]["episodes"]) <= 0:
        raise RLConfigError("eval.episodes must be a positive integer.")

    normalized = deepcopy(config)
    normalized["termination"]["max_steps"] = int(normalized["termination"]["max_steps"])
    normalized["action"]["dimensions"] = int(normalized["action"]["dimensions"])
    normalized["train"]["smoke_steps"] = int(normalized["train"]["smoke_steps"])
    normalized["train"]["checkpoint_every"] = int(normalized["train"]["checkpoint_every"])
    normalized["eval"]["episodes"] = int(normalized["eval"]["episodes"])
    return normalized


def summarize_rl_config(config):
    """Return a compact, test-friendly summary of the RL config."""
    return {
        "task_name": config["task_name"],
        "backend": config["backend"],
        "policy_topology": config["policy"]["topology"],
        "shared_policy": bool(config["policy"]["shared_policy"]),
        "observation_fields": list(config["observation"]["fields"]),
        "action_dimensions": int(config["action"]["dimensions"]),
        "reward_terms": sorted(config["reward"]["coefficients"].keys()),
        "max_steps": int(config["termination"]["max_steps"]),
        "seed_set": list(config["eval"]["seed_set"]),
        "render_enabled": bool(config["render"]["enabled"]),
    }
