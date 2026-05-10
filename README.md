# Semantic-Driven Cup/Mug Ordering Demo

This repository contains our RoboCasa-based cup and mug ordering pipeline for perception, policy selection, live execution, and report figure generation.

This README is written as an **application-facing walkthrough** for instructors, TAs, and project reviewers. It no longer serves as a development-rules document.

## What This Project Demonstrates

- Multi-object cup/mug perception in RoboCasa
- Policy-based ordering decisions over five drinkware targets
- Live execution in a RoboCasa environment
- RL training, evaluation, and report figure generation

## Recommended Environment

Use the project environment that already contains the RoboCasa and RL dependencies.

Example:

```cmd
conda activate robotic-robocasa-rl
```

If your local machine can already run the existing RoboCasa demos in this repository, you can use the same environment for the commands below.

## Quick Demo Commands

### 1. Online Greedy Demo in RoboCasa

This opens the live RoboCasa environment and runs the online cup/mug ordering loop with the greedy policy.

```cmd
python scripts/demo_online_cup_mug_ordering.py --policy greedy --layout 1 --style 1 --save-trace --keep-open-sec 1
```

### 2. Online RL Demo in RoboCasa with a Trained Checkpoint

Replace the checkpoint path if you trained a different model.

```cmd
python scripts/demo_online_cup_mug_ordering.py --policy rl --rl-policy-mode checkpoint --rl-checkpoint outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-ppo-102400.zip --layout 1 --style 1 --save-trace --keep-open-sec 1
```

## End-to-End RL Workflow

The full workflow is:

1. Train an RL checkpoint
2. Run real RoboCasa evaluation
3. Generate report figures

### Step 1. Train the RL Policy

Example with `102400` training timesteps:

```cmd
python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --timesteps 102400
```

Main outputs:

- Checkpoint: `outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-ppo-102400.zip`
- Training history: `outputs/rl_cup_mug_ordering/metrics/cup_ordering-train-history-102400.json`

### Step 2. Smoke Test Real RoboCasa Evaluation

Before batch evaluation, run one real-environment episode:

```cmd
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --rl-policy-mode checkpoint --checkpoint outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-ppo-102400.zip --episodes 1 --seed-start 7 --save-report
```

If this succeeds, continue with multi-episode evaluation.

### Step 3. Run Real RoboCasa Evaluation

For a fast report pass, we recommend `10` episodes per policy.

#### Random baseline

```cmd
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 10 --seed-start 7 --save-report
```

#### Greedy baseline

```cmd
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 10 --seed-start 7 --save-report
```

#### RL checkpoint policy

```cmd
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --rl-policy-mode checkpoint --checkpoint outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-ppo-102400.zip --episodes 10 --seed-start 7 --save-report
```

Main outputs:

- `outputs/rl_cup_mug_ordering/metrics/random-real-episodes-10.json`
- `outputs/rl_cup_mug_ordering/metrics/greedy-real-episodes-10.json`
- `outputs/rl_cup_mug_ordering/metrics/rl-real-episodes-10.json`

And matching trace reports under:

- `outputs/rl_cup_mug_ordering/reports/`

## Report Figure Generation

The plotting script consumes:

- real-environment evaluation metrics for `random`, `greedy`, and `rl`
- the RL training history JSON

Example:

```cmd
python scripts/plot_cup_mug_ordering_report.py --random-metrics outputs/rl_cup_mug_ordering/metrics/random-real-episodes-10.json --greedy-metrics outputs/rl_cup_mug_ordering/metrics/greedy-real-episodes-10.json --rl-metrics outputs/rl_cup_mug_ordering/metrics/rl-real-episodes-10.json --train-history outputs/rl_cup_mug_ordering/metrics/cup_ordering-train-history-102400.json
```

Generated figures are written to:

- `outputs/rl_cup_mug_ordering/figures/`

Current figure set:

- `completion_count_histogram.png`
- `cumulative_mean_completion_ratio.png`
- `cumulative_full_success_rate.png`
- `rl_training_loss_curves.png`

If the training history contains reward statistics in the future, the plotting pipeline can also include an RL reward curve automatically.

## Suggested Minimal Demonstration for Review

If time is limited, use this compact sequence:

1. Train a checkpoint

```cmd
python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --timesteps 102400
```

2. Verify one real RL episode

```cmd
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --rl-policy-mode checkpoint --checkpoint outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-ppo-102400.zip --episodes 1 --seed-start 7 --save-report
```

3. Run `10` episodes for each policy

```cmd
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 10 --seed-start 7 --save-report
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 10 --seed-start 7 --save-report
python scripts/eval_rl_real_cup_mug_ordering.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --rl-policy-mode checkpoint --checkpoint outputs/rl_cup_mug_ordering/checkpoints/cup-ordering-ppo-102400.zip --episodes 10 --seed-start 7 --save-report
```

4. Generate the final figures

```cmd
python scripts/plot_cup_mug_ordering_report.py --random-metrics outputs/rl_cup_mug_ordering/metrics/random-real-episodes-10.json --greedy-metrics outputs/rl_cup_mug_ordering/metrics/greedy-real-episodes-10.json --rl-metrics outputs/rl_cup_mug_ordering/metrics/rl-real-episodes-10.json --train-history outputs/rl_cup_mug_ordering/metrics/cup_ordering-train-history-102400.json
```

## Notes for Reviewers

- The root `README.md` now focuses on **how to run the project workflow** rather than internal development conventions.
- Runtime outputs, traces, figures, and evaluation artifacts are written under `outputs/rl_cup_mug_ordering/`.
- The most important assets for grading or inspection are:
  - trained checkpoint
  - real-environment metrics JSON
  - trace reports
  - generated report figures
