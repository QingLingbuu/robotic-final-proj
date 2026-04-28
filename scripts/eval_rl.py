"""Minimal RL evaluation harness entrypoint for contract validation and dry runs."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.harness import get_latest_checkpoint, run_eval_smoke, validate_and_summarize_config
from rl.contracts import load_rl_config, validate_rl_config


def build_parser():
    parser = argparse.ArgumentParser(description="Milestone-1 RL eval harness")
    parser.add_argument("--config", required=True, help="Path to RL YAML config")
    parser.add_argument("--checkpoint", required=False, help="Path to checkpoint artifact")
    parser.add_argument("--dry-run", action="store_true", help="Run contract-only dry-run evaluation")
    parser.add_argument("--latest", action="store_true", help="Use the latest PPO checkpoint from the configured artifacts directory")
    parser.add_argument("--render", action="store_true", help="Force human render during evaluation rollout")
    parser.add_argument("--save-video", action="store_true", help="Capture evaluation frames and write an mp4 video")
    parser.add_argument("--video-path", required=False, help="Optional path for the evaluation mp4 output")
    parser.add_argument("--print-summary", action="store_true", help="Print resolved config summary and exit")
    return parser


def main():
    args = build_parser().parse_args()
    if args.print_summary:
        print(validate_and_summarize_config(args.config))
        return 0
    checkpoint_path = args.checkpoint
    if args.latest:
        checkpoint_path = str(get_latest_checkpoint(validate_rl_config(load_rl_config(args.config))))
    if not checkpoint_path:
        raise SystemExit("--checkpoint is required unless --print-summary is used")
    metrics_path = run_eval_smoke(
        args.config,
        checkpoint_path,
        dry_run=args.dry_run,
        render_override=args.render,
        save_video=args.save_video,
        video_path=args.video_path,
    )
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
