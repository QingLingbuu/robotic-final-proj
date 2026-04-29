"""Minimal dependency smoke checks for the RoboCasa + RL migration baseline."""

import importlib
import json


PACKAGE_MATRIX = {
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


def check_package(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - smoke check wants surfaced import details
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}

    return {"installed": True, "version": getattr(module, "__version__", "unknown")}


def _major_version(version):
    try:
        return int(str(version).split(".", 1)[0])
    except Exception:  # noqa: BLE001 - smoke helper only
        return None


def main():
    results = {name: check_package(name) for name in PACKAGE_MATRIX}
    missing_required = [
        name for name, meta in PACKAGE_MATRIX.items() if meta["required"] and not results[name]["installed"]
    ]

    numpy_major = _major_version(results["numpy"].get("version"))
    mink_numpy_conflict = bool(results["mink"]["installed"] and numpy_major is not None and numpy_major >= 2)

    payload = {
        "environment_strategy": "separate_robocasa_rl_env",
        "shared_env_supported": False,
        "mink_numpy_conflict": mink_numpy_conflict,
        "known_mink_constraint": "mink 0.0.5 requires numpy<2.0.0",
        "mink_requires_numpy_lt_2_0_0": True,
        "results": results,
        "missing_required": missing_required,
        "robocasa_ready": results["robocasa"]["installed"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if missing_required:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
