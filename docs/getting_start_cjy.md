# Getting Started CJY

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
pip install -r requirements.txt
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

## 5. robosuite Baseline Environment Setup

The robosuite baseline uses the separate `robotic-final` environment defined in [environment.yml](/D:/Code/MyRepositories/robotic-final-proj/environment.yml:1).

From the repository root:

```powershell
conda env create -f environment.yml
conda activate robotic-final
```

This baseline environment is the one to use for:

- `main.py`
- `scripts/demo_vision_grasp.py`
- the robosuite-side perception / FSM / grasp pipeline

For the current round's machine-local diagnosis memo and ad-hoc script shortcuts, see [`../LHYstart.md`](../LHYstart.md). Treat that file as a convenience note, but treat **this document** as the project-level source of truth for environment strategy.

## 6. CupMugSorting Commands

The current RoboCasa demo task is `robocasa/CupMugSorting`.

Scene behavior:

- The kitchen scene is pinned to `layout_and_style_ids: [[1, 1]]`.
- The sink/counter layout stays fixed.
- Cup and mug instances may vary because `seed` is unset by default.
- The multi-object scene contains 5 drinkware objects by default: 2 mugs and 3 cups.
- Objects start in one row near the front counter edge.
- Mugs are handled drinkware and use `handle_top_down`.
- Cups are no-handle drinkware and use `top_down`.
- After lifting, mugs are placed into the nearest sink basin.
- After lifting, cups are placed on the right/front counter edge.

### 6.1 Onscreen Scene / Motion Viewer

Use this when you want to see the live `mjviewer` window:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --keep-open-sec 5
```

For the mug / handle path:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label mug --grasp-type handle_top_down --keep-open-sec 5
```

Expected env config at startup:

```json
"layout_ids": null,
"style_ids": null,
"layout_and_style_ids": [
  [
    1,
    1
  ]
]
```

If you need a fully reproducible reset, add a seed explicitly:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --seed 1 --keep-open-sec 5
```

### 6.2 Classification / Scene Smoke

Use this when you only need to initialize the task, run perception, and print the classification summary:

```powershell
python scripts/demo_multi_cup_mug_sort.py --task robocasa/CupMugSorting --keep-open-sec 5
```

Clean two-object validation scene:

```powershell
python scripts/demo_multi_cup_mug_sort.py --task robocasa/CupMugSortingClean --keep-open-sec 5
```

Save the current RGB frame:

```powershell
python scripts/demo_multi_cup_mug_sort.py --task robocasa/CupMugSorting --keep-open-sec 5 --save-frame outputs/cup_mug_sorting.png
```

### 6.3 Expected JSON Fields

For cup runs, check:

```text
drinkware_classification.selected_assignment.has_handle = false
drinkware_classification.selected_assignment.strategy = top_down
execution_summary.place_plan.zone = opposite_counter
```

For mug runs, check:

```text
drinkware_classification.selected_assignment.has_handle = true
drinkware_classification.selected_assignment.strategy = handle_top_down
execution_summary.place_plan.zone = sink
execution_summary.place_plan.placement_rule = nearest_sink_basin
```

### 6.4 Supported Grasp Paths

The active RoboCasa Cup/Mug path intentionally exposes only:

- `top_down`
- `handle_top_down`

The old `oblique_reach`, `side_reach`, and `anchor_top_down` experiment paths have been removed from the active demo.

## 7. Other Common Commands

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

Run the older RoboCasa onscreen automatic reach demo:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug
```

If you want the viewer to stay open longer after the reach:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug --keep-open-sec 20
```

If axis calibration is causing issues and you want a quicker smoke:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug --skip-axis-calibration
```

Current practical way to capture a real RoboCasa motion video:

- run `scripts/demo_robocasa_reach_onscreen.py`
- record the onscreen viewer with Windows Game Bar (`Win + G`) or OBS
- prefer this path over the older offscreen mp4 experiments, which were observed to freeze or go black after `env.step()`

Common robosuite baseline commands:

```powershell
python scripts/demo_vision_grasp.py --scenario cube --render
python scripts/demo_vision_grasp.py --scenario can --render
python scripts/demo_vision_grasp.py --scenario milk --render
python scripts/demo_vision_grasp.py --scenario bread --render
python scripts/demo_vision_grasp.py --scenario cereal --render
```

Use `scripts/demo_vision_grasp.py` when you want the explicit robosuite demo entrypoint with staged console output.

Additional diagnosis helpers used in the current round:

```powershell
python scripts/demo_GT_grasp.py --scenario cube
python scripts/checkVision.py --scenario cube
python scripts/checkVision.py --scenario all --summary-table
```

Run the main robosuite project flow:

```powershell
python main.py
```

## 8. Output Paths

Project artifacts must go under:

```text
outputs/
  rl/
    checkpoints/
    metrics/
    videos/
```

Do not treat `.sisyphus/` or `scripts/.sisyphus/` as project output locations.

## 9. What To Commit

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

## 10. Current Caveats

- `third_party/robocasa` includes large simulator assets, so the repository will be much bigger than before
- `gymnasium` observation-space warnings on the RoboCasa path are still unresolved
- some Windows-specific tests still need cleanup
- if you want to sync with upstream `robocasa` or `robosuite`, you now need to do that manually because these are vendored trees
