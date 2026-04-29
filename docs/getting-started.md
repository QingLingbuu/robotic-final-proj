# Getting Started

## 1. What This Repository Contains

This repository now vendors its simulator source dependencies directly under `third_party/`:

```text
third_party/
  robocasa/
  robosuite/
```

That means:

- cloning this repository once is enough
- you do not need to clone `robocasa` and `robosuite` separately again
- these vendored trees are part of the main repository history

## 2. Two-Environment Strategy

Do not try to force everything into one Python environment.

Current project reality:

- the `robosuite/mink` baseline and the `RoboCasa/RL` baseline are separate environments
- observed `RoboCasa` installs drift to `numpy 2.2.5`, `torch 2.7.1`, `torchvision 0.22.1`, `gymnasium 0.29.1`
- `mink 0.0.5` requires `numpy<2.0.0`

Conclusion:

- keep the legacy baseline environment separate
- keep the `RoboCasa/RL` environment separate

## 3. Recommended Repository Layout

The current expected layout is:

```text
robotic-final-proj/
  arm/
  configs/
  docs/
  fsm/
  runtime/
  rl/
  scripts/
  tests/
  third_party/
    robocasa/
    robosuite/
  vision/
```

Runtime bootstrap is handled by `runtime/bootstrap.py`, so project entrypoints do not need manual `PYTHONPATH` edits.

## 4. RoboCasa RL Environment Setup

From the repository root:

```powershell
conda create -n robotic-robocasa-rl python=3.10 pip
conda activate robotic-robocasa-rl
pip install -r requirements-robocasa-rl.txt
pip install -e ./third_party/robosuite
pip install -e ./third_party/robocasa
```

If RoboCasa macro setup expects a root-level template inside its vendored tree:

```powershell
Copy-Item .\third_party\robocasa\robocasa\macros.py .\third_party\robocasa\macros.py
cd .\third_party\robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
cd ..\..
```

Dependency smoke check:

```powershell
python scripts/setup/check_robocasa_rl_deps.py
```

## 5. Common Commands

Print the current RoboCasa setup guidance:

```powershell
python scripts/setup/print_robocasa_rl_setup.py
```

Print training config summary:

```powershell
python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --print-summary
```

Print evaluation config summary:

```powershell
python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --print-summary
```

Run a short RoboCasa training smoke:

```powershell
python scripts/train_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --timesteps 200
```

Run evaluation and save a rollout video:

```powershell
python scripts/eval_rl.py --config configs/rl/single_arm_robocasa_ppo.yaml --latest --save-video
```

## 6. Output Paths

Project artifacts must go under:

```text
outputs/
  rl/
    checkpoints/
    metrics/
    videos/
```

Do not treat `.sisyphus/` or `scripts/.sisyphus/` as project output locations.

## 7. What To Commit

Safe to commit:

- source code
- configs
- docs
- vendored third-party source files under `third_party/`

Do not commit:

- `__pycache__/`
- `*.pyc`
- `*.egg-info/`
- `build/`
- `dist/`
- runtime `outputs/`
- tool-private `.sisyphus/`

## 8. Current Caveats

- `third_party/robocasa` includes large simulator assets, so the repository will be much bigger than before
- `gymnasium` observation-space warnings on the RoboCasa path are still unresolved
- some Windows-specific tests still need cleanup
- if you want to sync with upstream `robocasa` or `robosuite`, you now need to do that manually because these are vendored trees
