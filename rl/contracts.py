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

CUP_ORDERING_MODE = "cup_ordering"
CUP_ORDERING_REQUIRED_TOP_LEVEL_KEYS = {
    "mode",
    "task_name",
    "backend",
    "max_targets",
    "action_space",
    "scene",
    "ordering",
    "policies",
    "reward",
    "eval",
    "artifacts",
    "render",
}
CUP_ORDERING_REQUIRED_ACTION_SPACE_KEYS = {"type", "n"}
CUP_ORDERING_REQUIRED_SCENE_KEYS = {"layout_and_style_ids", "num_mugs", "num_cups"}
CUP_ORDERING_REQUIRED_ORDERING_KEYS = {
    "handled_strategy",
    "plain_strategy",
    "handled_zone",
    "plain_zone",
}
CUP_ORDERING_REQUIRED_POLICIES_KEYS = {"baselines", "rl_sanity", "checkpoint_interface"}
CUP_ORDERING_REQUIRED_CHECKPOINT_INTERFACE_KEYS = {
    "artifact_kind",
    "schema_version",
    "supports_inference",
    "unsupported_reason",
}
CUP_ORDERING_REQUIRED_EVAL_KEYS = {"episodes_per_policy", "seed_set"}
CUP_ORDERING_REQUIRED_ARTIFACT_KEYS = {
    "root_dir",
    "checkpoint_dir",
    "metrics_dir",
    "report_dir",
    "video_dir",
}
CUP_ORDERING_REWARD_TERMS = {
    "correct_zone_place",
    "selected_target_success",
    "grasp_failure",
    "placement_failure",
    "invalid_action",
    "retry_penalty",
    "timeout",
    "collision",
    "all_invalid",
}


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


def build_selector_contract(config):
    """Build a normalized RL selector contract from optional config."""
    selector = deepcopy(config.get("selector") or {})
    enabled = bool(selector.get("enabled", False))
    max_candidates = int(selector.get("max_candidates", 0 if not enabled else 1))
    if enabled and max_candidates <= 0:
        raise RLConfigError("selector.max_candidates must be positive when selector is enabled.")

    fallback_actions = [
        str(action).strip().lower()
        for action in selector.get("fallback_actions", [])
        if str(action).strip()
    ]
    valid_fallbacks = {"clear", "resense"}
    invalid = [action for action in fallback_actions if action not in valid_fallbacks]
    if invalid:
        raise RLConfigError(
            "selector.fallback_actions contains unsupported actions: " + ", ".join(invalid)
        )

    action_meanings = ["select_candidate"] + fallback_actions if enabled else []
    return {
        "enabled": enabled,
        "max_candidates": max_candidates,
        "fallback_actions": fallback_actions,
        "action_meanings": action_meanings,
    }


def _validate_cup_ordering_config(config):
    _ensure_keys("config", config, CUP_ORDERING_REQUIRED_TOP_LEVEL_KEYS)
    _ensure_keys("action_space", config["action_space"], CUP_ORDERING_REQUIRED_ACTION_SPACE_KEYS)
    _ensure_keys("scene", config["scene"], CUP_ORDERING_REQUIRED_SCENE_KEYS)
    _ensure_keys("ordering", config["ordering"], CUP_ORDERING_REQUIRED_ORDERING_KEYS)
    _ensure_keys("policies", config["policies"], CUP_ORDERING_REQUIRED_POLICIES_KEYS)
    _ensure_keys(
        "policies.checkpoint_interface",
        config["policies"]["checkpoint_interface"],
        CUP_ORDERING_REQUIRED_CHECKPOINT_INTERFACE_KEYS,
    )
    _ensure_keys("reward", config["reward"], REQUIRED_REWARD_KEYS)
    _ensure_keys("eval", config["eval"], CUP_ORDERING_REQUIRED_EVAL_KEYS)
    _ensure_keys("artifacts", config["artifacts"], CUP_ORDERING_REQUIRED_ARTIFACT_KEYS)
    _ensure_keys("render", config["render"], REQUIRED_RENDER_KEYS)

    if config["mode"] != CUP_ORDERING_MODE:
        raise RLConfigError("cup ordering config mode must be 'cup_ordering'.")
    if config["backend"] != "robocasa":
        raise RLConfigError("cup ordering backend must be 'robocasa'.")
    if config["task_name"] != "robocasa/CupMugSorting":
        raise RLConfigError("cup ordering task_name must be 'robocasa/CupMugSorting'.")
    if int(config["max_targets"]) != 5:
        raise RLConfigError("cup ordering max_targets must be 5.")
    if str(config["action_space"]["type"]).lower() != "discrete":
        raise RLConfigError("cup ordering action_space.type must be 'discrete'.")
    if int(config["action_space"]["n"]) != 5:
        raise RLConfigError("cup ordering action_space.n must be 5.")
    if config["scene"]["layout_and_style_ids"] != [[1, 1]]:
        raise RLConfigError("cup ordering scene.layout_and_style_ids must be [[1, 1]].")
    if int(config["scene"]["num_mugs"]) != 2:
        raise RLConfigError("cup ordering scene.num_mugs must be 2.")
    if int(config["scene"]["num_cups"]) != 3:
        raise RLConfigError("cup ordering scene.num_cups must be 3.")
    if config["ordering"]["handled_strategy"] != "handle_top_down":
        raise RLConfigError("cup ordering handled_strategy must be 'handle_top_down'.")
    if config["ordering"]["plain_strategy"] != "top_down":
        raise RLConfigError("cup ordering plain_strategy must be 'top_down'.")
    if config["ordering"]["handled_zone"] != "sink":
        raise RLConfigError("cup ordering handled_zone must be 'sink'.")
    if config["ordering"]["plain_zone"] != "right_counter":
        raise RLConfigError("cup ordering plain_zone must be 'right_counter'.")
    if int(config["eval"]["episodes_per_policy"]) <= 0:
        raise RLConfigError("cup ordering eval.episodes_per_policy must be positive.")
    if not isinstance(config["eval"]["seed_set"], list) or not config["eval"]["seed_set"]:
        raise RLConfigError("cup ordering eval.seed_set must be a non-empty list.")
    baselines = config["policies"]["baselines"]
    if not isinstance(baselines, list) or not {"random", "greedy"}.issubset(set(baselines)):
        raise RLConfigError("cup ordering policies.baselines must include random and greedy.")
    if config["policies"]["rl_sanity"] != "rl":
        raise RLConfigError("cup ordering policies.rl_sanity must be 'rl'.")
    checkpoint_interface = config["policies"]["checkpoint_interface"]
    if checkpoint_interface["artifact_kind"] != "cup_ordering_policy_checkpoint":
        raise RLConfigError(
            "cup ordering checkpoint_interface.artifact_kind must be 'cup_ordering_policy_checkpoint'."
        )
    if int(checkpoint_interface["schema_version"]) <= 0:
        raise RLConfigError("cup ordering checkpoint_interface.schema_version must be positive.")
    if not isinstance(checkpoint_interface["supports_inference"], bool):
        raise RLConfigError("cup ordering checkpoint_interface.supports_inference must be boolean.")
    if not str(checkpoint_interface["unsupported_reason"]).strip():
        raise RLConfigError("cup ordering checkpoint_interface.unsupported_reason must be non-empty.")
    reward_terms = set(config["reward"]["coefficients"].keys())
    missing_reward_terms = sorted(CUP_ORDERING_REWARD_TERMS - reward_terms)
    if missing_reward_terms:
        raise RLConfigError("cup ordering reward coefficients missing required terms: " + ", ".join(missing_reward_terms))

    normalized = deepcopy(config)
    normalized["max_targets"] = int(normalized["max_targets"])
    normalized["action_space"]["type"] = str(normalized["action_space"]["type"]).lower()
    normalized["action_space"]["n"] = int(normalized["action_space"]["n"])
    normalized["scene"]["num_mugs"] = int(normalized["scene"]["num_mugs"])
    normalized["scene"]["num_cups"] = int(normalized["scene"]["num_cups"])
    normalized["eval"]["episodes_per_policy"] = int(normalized["eval"]["episodes_per_policy"])
    normalized["policies"]["checkpoint_interface"]["schema_version"] = int(
        normalized["policies"]["checkpoint_interface"]["schema_version"]
    )
    return normalized


def validate_rl_config(config):
    """Validate the milestone-1 dual-arm RoboCasa RL config contract."""
    if config.get("mode") == CUP_ORDERING_MODE:
        return _validate_cup_ordering_config(config)

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
    normalized["selector"] = build_selector_contract(normalized)
    return normalized


def summarize_rl_config(config):
    """Return a compact, test-friendly summary of the RL config."""
    if config.get("mode") == CUP_ORDERING_MODE:
        action_n = int(config["action_space"]["n"])
        return {
            "mode": CUP_ORDERING_MODE,
            "task_name": config["task_name"],
            "backend": config["backend"],
            "max_targets": int(config["max_targets"]),
            "action_space": f"Discrete({action_n})",
            "action_space_type": config["action_space"]["type"],
            "action_space_n": action_n,
            "scene": deepcopy(config["scene"]),
            "ordering": deepcopy(config["ordering"]),
            "policies": deepcopy(config["policies"]),
            "checkpoint_interface": deepcopy(config["policies"]["checkpoint_interface"]),
            "reward_terms": sorted(config["reward"]["coefficients"].keys()),
            "episodes_per_policy": int(config["eval"]["episodes_per_policy"]),
            "seed_set": list(config["eval"]["seed_set"]),
            "render_enabled": bool(config["render"]["enabled"]),
        }

    selector = build_selector_contract(config)
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
        "selector_enabled": selector["enabled"],
        "selector_max_candidates": selector["max_candidates"],
        "selector_actions": list(selector["action_meanings"]),
    }
