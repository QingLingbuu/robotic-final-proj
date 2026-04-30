"""Minimal dependency smoke checks for the RoboCasa + RL migration baseline."""

import json
from importlib import metadata


DEPENDENCIES = {
    "numpy": {"required": True, "distribution": "numpy"},
    "torch": {"required": True, "distribution": "torch"},
    "torchvision": {"required": True, "distribution": "torchvision"},
    "gymnasium": {"required": True, "distribution": "gymnasium"},
    "robosuite": {"required": True, "distribution": "robosuite"},
    "stable_baselines3": {"required": True, "distribution": "stable-baselines3"},
    "mujoco": {"required": True, "distribution": "mujoco"},
    "mink": {"required": False, "distribution": "mink"},
    "robocasa": {"required": False, "distribution": "robocasa"},
}


def inspect_dependency(module_name, distribution_name):
    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return {
            "installed": False,
            "error": f"PackageNotFoundError: {distribution_name}",
        }

    return {
        "installed": True,
        "version": version,
    }


def main():
    results = {
        name: inspect_dependency(name, meta["distribution"])
        for name, meta in DEPENDENCIES.items()
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

