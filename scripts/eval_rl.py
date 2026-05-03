"""Minimal RL evaluation harness entrypoint for contract validation and dry runs."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

from rl.harness import get_latest_checkpoint, run_eval_smoke, validate_and_summarize_config
from rl.contracts import load_rl_config, validate_rl_config


def build_parser():
    parser = argparse.ArgumentParser(description="Milestone-1 RL eval harness")
    parser.add_argument("--config", required=True, help="Path to RL YAML config")
    parser.add_argument("--checkpoint", required=False, help="Path to checkpoint artifact")
    parser.add_argument("--policy", required=False, choices=["random", "greedy", "rl"], help="Cup-ordering evaluation policy")
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation episode count")
    parser.add_argument("--seed", type=int, default=None, help="Override starting seed for cup-ordering dry-runs")
    parser.add_argument("--dry-run", action="store_true", help="Run contract-only dry-run evaluation")
    parser.add_argument("--latest", action="store_true", help="Use the latest PPO checkpoint from the configured artifacts directory")
    parser.add_argument("--render", action="store_true", help="Force human render during evaluation rollout")
    parser.add_argument("--save-video", action="store_true", help="Capture evaluation frames and write an mp4 video")
    parser.add_argument("--video-path", required=False, help="Optional path for the evaluation mp4 output")
    parser.add_argument("--print-summary", action="store_true", help="Print resolved config summary and exit")
    return parser


def main():
    args = build_parser().parse_args()
    config = validate_rl_config(load_rl_config(args.config))
    if args.print_summary:
        summary = validate_and_summarize_config(args.config)
        if config.get("mode") == "cup_ordering":
            summary = dict(summary)
            if args.policy is not None:
                summary["policy"] = args.policy
            if args.episodes is not None:
                summary["episodes"] = int(args.episodes)
            if args.seed is not None:
                summary["seed"] = int(args.seed)
            if args.dry_run:
                summary["trace_report"] = True
        print(summary)
        return 0
    checkpoint_path = args.checkpoint
    if args.latest:
        checkpoint_path = str(get_latest_checkpoint(config))
    if config.get("mode") == "cup_ordering" and args.policy in {"random", "greedy", "rl"}:
        checkpoint_path = checkpoint_path or "cup-ordering-dry-run"
    if not checkpoint_path:
        raise SystemExit("--checkpoint is required unless --print-summary is used")
    try:
        metrics_path = run_eval_smoke(
            args.config,
            checkpoint_path,
            dry_run=args.dry_run,
            render_override=args.render,
            save_video=args.save_video,
            video_path=args.video_path,
            policy_name=args.policy,
            episodes=args.episodes,
            seed=args.seed,
        )
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
