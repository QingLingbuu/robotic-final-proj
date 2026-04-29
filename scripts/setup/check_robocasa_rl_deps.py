"""Minimal dependency smoke checks for the RoboCasa + RL migration baseline."""

import importlib
import json


DEPENDENCIES = {
    "numpy": {"required": True},
    "torch": {"required": True},
    "torchvision": {"required": True},
    "gymnasium": {"required": True},
    "robosuite": {"required": True},
    "stable_baselines3": {"required": True},
    "mujoco": {"required": True},
    "mink": {"required": False},
    "robocasa": {"required": False},
}


def inspect_dependency(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - informational script
        return {
            "installed": False,
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    return {
        "installed": True,
        "version": getattr(module, "__version__", "unknown"),
    }


def main():
    results = {
        name: inspect_dependency(name)
        for name in DEPENDENCIES
    }
    missing_required = [
        name
        for name, meta in DEPENDENCIES.items()
        if meta["required"] and not results[name]["installed"]
    ]
    mink_requires_numpy_lt = True
    payload = {
        "environment_strategy": "separate_robocasa_rl_env",
        "shared_env_supported": False,
        "mink_numpy_conflict": (
            results["mink"]["installed"]
            and results["numpy"]["installed"]
            and results["numpy"].get("version", "").split(".")[0].isdigit()
            and int(results["numpy"]["version"].split(".")[0]) >= 2
        ),
        "known_mink_constraint": "mink 0.0.5 requires numpy<2.0.0",
        "mink_requires_numpy_lt_2_0_0": mink_requires_numpy_lt,
        "results": results,
        "missing_required": missing_required,
        "robocasa_ready": results["robocasa"]["installed"],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

