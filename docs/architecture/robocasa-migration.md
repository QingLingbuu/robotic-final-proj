# RoboCasa Migration Architecture

## 1. Scope

This document records the stable architectural decisions for migrating the project from the current `robosuite` baseline to a `RoboCasa`-backed RL workflow.

It replaces earlier temporary planning notes stored under tool-private directories. Only decisions that still match the repository state are kept here.

## 2. Current Baseline

- The repository currently validates the end-to-end RL runner on `robosuite` first.
- `RoboCasa` support exists as a migration target and is already wired into the RL harness path.
- The main swap boundary is the environment wrapper layer in `arm/env_wrapper.py`.
- Upper-layer interfaces in `vision/`, `runtime/`, and `fsm/` should remain as stable as possible during environment migration.

## 3. Migration Boundary

The project should treat the environment layer as the only required simulator-specific boundary.

Rules:

- `arm/env_wrapper.py` and related adapter code own backend selection.
- Training and evaluation entrypoints must not require manual `PYTHONPATH` edits.
- `vision/`, `runtime/`, and `fsm/` must not be rewritten just to accommodate `RoboCasa`.
- Simulator-specific compatibility fixes belong in adapter/bootstrap code, not in upper layers.

## 4. Runtime Bootstrap

The repository now uses `runtime/bootstrap.py` to normalize runtime startup:

- resolve repo-root-relative paths
- prepend local source roots when needed
- make `scripts/train_rl.py` and `scripts/eval_rl.py` runnable from different working directories

This bootstrap is a transition mechanism. It avoids fragile manual shell setup, but it does not replace a clean dependency layout.

## 5. Third-Party Source Layout

The repository now prefers `third_party/` as the stable home for editable simulator source trees.

Preferred steady-state layout:

```text
third_party/
  robocasa/
  robosuite/
```

Guidelines:

- External source trees should live under a dedicated `third_party/` directory.
- Project code should import them through the bootstrap layer, not by assuming they sit at repo root.
- Re-cloning the repository should not require copying these directories back to the root.
- This repository now uses the vendored-code approach for `third_party/robocasa` and `third_party/robosuite`.
- Inner `.git/` directories must not be kept inside these vendored trees, otherwise Git will treat them as separate repositories instead of normal project files.

## 6. Environment Strategy

For the `RoboCasa` RL path, the repository should prefer:

- a lightweight `conda` environment shell
- `python=3.10`
- `pip` installation from `requirements.txt`

Reason:

- the earlier fully pinned Windows `conda` solve path was brittle
- `mink==0.0.5` conflicts with `numpy 2.x`
- RL runtime and simulator dependencies need to be isolated from unrelated local environments

The repository should continue treating the RoboCasa RL environment as a dedicated environment, separate from any environment that requires incompatible `numpy` or `mink` combinations.

## 7. Output Layout

Project outputs must not be written into tool-private directories such as `.sisyphus/`.

Tracked convention:

```text
outputs/
  rl/
    checkpoints/
    metrics/
    videos/
```

Rules:

- artifact paths must resolve from the repository root
- current working directory must not affect output location
- generated checkpoints, metrics, and rollout videos belong under `outputs/rl/`
- tool-private folders such as `.sisyphus/` are not part of the project contract

## 8. Phase Order

The repository should keep the following order of work:

1. Validate runner, config, artifact, and environment bootstrap on `robosuite`
2. Keep the wrapper boundary stable while bringing up `RoboCasa`
3. Establish a reproducible single-arm RoboCasa RL smoke baseline
4. Expand training quality and evaluation coverage
5. Revisit higher-level integration only after the environment and harness path are stable

This keeps simulator migration from turning into an uncontrolled architecture rewrite.

## 9. Immediate Refactor Priorities

- move external source trees under `third_party/`
- formalize bootstrap assumptions in docs and setup scripts
- keep generated artifacts under `outputs/`
- remove obsolete tool-private project files from the repository workflow
- tighten environment setup docs around the dedicated RoboCasa RL environment

## 10. Non-Goals For This Refactor

- no algorithm rewrite
- no reward redesign as part of repository cleanup
- no broad rewrite of `vision/`, `runtime/`, or `fsm/`
- no simulator-specific code leaking upward into orchestration layers
