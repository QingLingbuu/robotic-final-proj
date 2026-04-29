"""Minimal RL training harness entrypoint for contract validation and dry runs."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

from rl.harness import run_train_smoke, validate_and_summarize_config


def build_parser():
    parser = argparse.ArgumentParser(description="Milestone-1 RL train harness")
    parser.add_argument("--config", required=True, help="Path to RL YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Run contract-only smoke training")
    parser.add_argument("--steps", type=int, default=None, help="Override smoke step count")
    parser.add_argument("--timesteps", type=int, default=None, help="Override real PPO training timesteps")
    parser.add_argument("--print-summary", action="store_true", help="Print resolved config summary and exit")
    return parser


def main():
    args = build_parser().parse_args()
    if args.print_summary:
        print(validate_and_summarize_config(args.config))
        return 0
    override_steps = args.timesteps if args.timesteps is not None else args.steps
    checkpoint_path = run_train_smoke(args.config, dry_run=args.dry_run, steps=override_steps)
    print(checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
