"""Generate English report figures for Cup/Mug ordering baselines and RL training."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.bootstrap import ensure_runtime_paths

ensure_runtime_paths()

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path):
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _episode_records(metrics_payload):
    return list(metrics_payload.get("episodes_detail", []))


def _completed_counts(metrics_payload):
    return [int(record.get("completed_count", 0)) for record in _episode_records(metrics_payload)]


def _completed_ratios(metrics_payload, max_targets=5):
    return [float(count) / float(max_targets) for count in _completed_counts(metrics_payload)]


def _full_success_flags(metrics_payload, max_targets=5):
    return [1.0 if int(count) >= int(max_targets) else 0.0 for count in _completed_counts(metrics_payload)]


def _cumulative_mean(values):
    running = []
    total = 0.0
    for index, value in enumerate(values, start=1):
        total += float(value)
        running.append(total / float(index))
    return running


def _training_series(history_payload, key):
    points = []
    for record in list(history_payload.get("records", [])):
        value = record.get(key)
        if value is None:
            continue
        points.append((int(record["timesteps"]), float(value)))
    return points


def _plot_completion_histogram(output_path, payloads_by_label, episodes, max_targets=5):
    fig, axes = plt.subplots(1, len(payloads_by_label), figsize=(15, 4.8), sharey=True)
    if len(payloads_by_label) == 1:
        axes = [axes]
    bins = np.arange(max_targets + 1)
    for axis, (label, payload) in zip(axes, payloads_by_label.items()):
        counts = _completed_counts(payload)
        frequencies = [counts.count(index) for index in bins]
        axis.bar(bins, frequencies, color="#4472C4", edgecolor="black", width=0.75)
        axis.set_title(f"{label}: Completed Objects per Episode")
        axis.set_xlabel("Completed Objects (out of 5)")
        axis.set_xticks(list(bins))
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Number of Episodes")
    fig.suptitle(f"Completion Count Distribution over {int(episodes)} Episodes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_curve(output_path, payloads_by_label, series_builder, title, ylabel):
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    color_map = {
        "Random": "#A5A5A5",
        "Greedy": "#ED7D31",
        "RL": "#4472C4",
    }
    for label, payload in payloads_by_label.items():
        series = list(series_builder(payload))
        x_values = np.arange(1, len(series) + 1)
        axis.plot(x_values, series, label=label, linewidth=2.2, color=color_map.get(label))
    axis.set_title(title)
    axis.set_xlabel("Episode Index")
    axis.set_ylabel(ylabel)
    axis.set_xlim(left=1)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_training_reward(output_path, history_payload):
    points = _training_series(history_payload, "rollout/ep_rew_mean")
    if not points:
        return False
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(x_values, y_values, linewidth=2.4, color="#4472C4")
    axis.set_title("RL Training Reward Curve")
    axis.set_xlabel("Training Timesteps")
    axis.set_ylabel("Mean Episode Reward")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_training_loss(output_path, history_payload):
    policy_points = _training_series(history_payload, "train/policy_gradient_loss")
    value_points = _training_series(history_payload, "train/value_loss")
    if not policy_points and not value_points:
        raise ValueError("Training history does not contain policy or value loss records.")

    fig, axis_left = plt.subplots(figsize=(8.5, 5.2))
    if value_points:
        axis_left.plot(
            [point[0] for point in value_points],
            [point[1] for point in value_points],
            linewidth=2.2,
            color="#ED7D31",
            label="Value Loss",
        )
    axis_left.set_xlabel("Training Timesteps")
    axis_left.set_ylabel("Value Loss", color="#ED7D31")
    axis_left.tick_params(axis="y", labelcolor="#ED7D31")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    if policy_points:
        axis_right.plot(
            [point[0] for point in policy_points],
            [point[1] for point in policy_points],
            linewidth=2.2,
            color="#4472C4",
            label="Policy Gradient Loss",
        )
    axis_right.set_ylabel("Policy Gradient Loss", color="#4472C4")
    axis_right.tick_params(axis="y", labelcolor="#4472C4")

    axis_left.set_title("RL Training Loss Curves")
    handles = axis_left.get_lines() + axis_right.get_lines()
    labels = [line.get_label() for line in handles]
    axis_left.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate report figures for cup/mug ordering.")
    parser.add_argument("--random-metrics", required=True, help="Path to random policy metrics JSON.")
    parser.add_argument("--greedy-metrics", required=True, help="Path to greedy policy metrics JSON.")
    parser.add_argument("--rl-metrics", required=True, help="Path to RL policy metrics JSON.")
    parser.add_argument("--train-history", required=True, help="Path to RL training history JSON.")
    parser.add_argument(
        "--output-dir",
        default="outputs/rl_cup_mug_ordering/figures",
        help="Directory for generated report figures.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads_by_label = {
        "Random": _load_json(args.random_metrics),
        "Greedy": _load_json(args.greedy_metrics),
        "RL": _load_json(args.rl_metrics),
    }
    history_payload = _load_json(args.train_history)

    episode_count = int(next(iter(payloads_by_label.values())).get("episodes", 0))
    _plot_completion_histogram(output_dir / "completion_count_histogram.png", payloads_by_label, episodes=episode_count)
    _plot_cumulative_curve(
        output_dir / "cumulative_mean_completion_ratio.png",
        payloads_by_label,
        series_builder=lambda payload: _cumulative_mean(_completed_ratios(payload, max_targets=5)),
        title="Cumulative Mean Completion Ratio",
        ylabel="Cumulative Mean of Completed Objects / 5",
    )
    _plot_cumulative_curve(
        output_dir / "cumulative_full_success_rate.png",
        payloads_by_label,
        series_builder=lambda payload: _cumulative_mean(_full_success_flags(payload, max_targets=5)),
        title="Cumulative Full-Episode Success Rate",
        ylabel="Cumulative Mean of 5/5 Success Indicator",
    )
    reward_curve_written = _plot_training_reward(output_dir / "rl_training_reward_curve.png", history_payload)
    _plot_training_loss(output_dir / "rl_training_loss_curves.png", history_payload)

    manifest = {
        "completion_histogram": str(output_dir / "completion_count_histogram.png"),
        "cumulative_completion_ratio": str(output_dir / "cumulative_mean_completion_ratio.png"),
        "cumulative_full_success_rate": str(output_dir / "cumulative_full_success_rate.png"),
        "training_loss_curves": str(output_dir / "rl_training_loss_curves.png"),
    }
    if reward_curve_written:
        manifest["training_reward_curve"] = str(output_dir / "rl_training_reward_curve.png")
    else:
        manifest["training_reward_curve"] = None
        manifest["training_reward_curve_skip_reason"] = "rollout_ep_rew_mean_missing_in_training_history"
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
