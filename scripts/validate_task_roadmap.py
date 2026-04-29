"""Validate docs/task.md contains the expected current roadmap and contract markers."""

from pathlib import Path


REQUIRED_SNIPPETS = [
    "tests/test_protocol_contracts.py",
    "BaseEnvWrapper",
    "create_env_wrapper()",
    "configs/rl/dual_arm_robocasa.yaml",
    "python -m unittest tests.test_protocol_contracts tests.test_env_wrapper_contract tests.test_rl_contracts",
    "RoboCasa backend adapter",
    "dry-run harness",
    "独立于现有 robosuite/mink 基线",
    "由于 `mink 0.0.5` 明确要求 `numpy<2.0.0`",
]


def main():
    task_doc = Path("docs/task.md")
    if not task_doc.exists():
        raise SystemExit("docs/task.md not found")

    content = task_doc.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in content]
    if missing:
        raise SystemExit("Missing required roadmap snippets: " + ", ".join(missing))

    print("docs/task.md roadmap validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
