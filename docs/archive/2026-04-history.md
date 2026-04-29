# 2026-04 History Summary

This document keeps historical stage notes that were previously mixed into `docs/task.md`.

It is not the active task board. Current status and next actions belong in [task.md](/D:/Code/MyRepositories/robotic-final-proj/docs/task.md).

## Historical Highlights

### 1. Baseline Integration

- The project established a `robosuite`-first manipulation baseline for validating perception, FSM, execution, and logging interfaces.
- Real GroundingDINO perception was wired into the manipulation loop instead of staying as a standalone demo.
- FSM transitions and retry accounting were aligned more closely with actual grasp outcomes.

### 2. Environment and RL Bring-Up

- Separate dependency baselines were introduced for the legacy `robosuite/mink` path and the `RoboCasa/RL` path.
- RL contract, harness, train/eval scripts, and artifact layout were introduced and gradually moved from dry-run to real smoke execution.
- RoboCasa single-arm smoke training, evaluation, and video export were successfully brought up in the dedicated environment.

### 3. Execution and Reward Iteration

- Single-arm `robosuite` RL smoke and baseline runs were brought up to validate the training / checkpoint / evaluation loop independent of RoboCasa asset issues.
- Reward shaping was iterated from simple lift-oriented shaping toward more grasp-oriented behavior.
- RoboCasa single-arm runs were narrowed to a grasp-first baseline to reduce action-space complexity and make rollout behavior easier to inspect.

### 4. Repository Refactor

- Tool-private `.sisyphus` artifacts were removed from the project workflow.
- RL outputs were standardized under `outputs/rl/`.
- `robocasa` and `robosuite` were moved under `third_party/` and converted into vendored source trees.
- Runtime bootstrap, queue helpers, and run logging were consolidated into the `runtime/` package.

## Why This File Exists

The project accumulated long chronological notes while the environment, RL, and runtime chain were being stabilized. Those notes are useful as historical context, but they make the active task board too noisy.

This archive keeps the historical storyline without forcing every reader to scan old stage-by-stage logs before finding the current blockers.

