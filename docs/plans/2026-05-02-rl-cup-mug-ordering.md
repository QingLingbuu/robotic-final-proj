# RL Cup/Mug Ordering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a high-level RL ordering layer for `CupMugSorting` that chooses which of the 5 visible cup/mug objects to process next, while the existing deterministic perception, grasp selection, execution, and placement logic remain responsible for how to grasp and where to place each object.

**Architecture:** The RL layer is an episode-level target-ordering policy, not a low-level robot controller. It consumes a padded 5-slot scene observation, returns one target slot index, and delegates the selected target to the current `top_down` or `handle_top_down` pipeline based on existing cup/mug classification and sim metadata. Random, greedy, and RL policies must share the same observation builder, mask semantics, completion rules, execution runner, and logging schema.

**Tech Stack:** Python, RoboCasa / robosuite, existing `unittest` tests, YAML configs, current `vision/`, `planner/`, `arm/`, `runtime/`, and `rl/` modules.

---

## Current Project Context

- Current environment target is `robocasa/CupMugSorting`, implemented in `third_party/robocasa/robocasa/environments/kitchen/composite/organizing_dishes_and_containers/cup_mug_sorting.py`.
- The scene has 5 drinkware objects by default: 2 handled mugs and 3 plain cups.
- The fixed visual scene uses `layout_and_style_ids: [[1, 1]]`, while object instances and positions may vary.
- Active grasp strategies are only:
  - `top_down` for plain cups
  - `handle_top_down` for handled mugs
- Failed experimental paths have been removed from the active interface:
  - `oblique_reach`
  - `side_reach`
  - `anchor_top_down`
  - `handle_oblique_grasp`
  - `handle_anchor_top_down`
- `handle_grasp` may still exist internally only as a fallback geometric source for synthesizing `handle_top_down`; do not expose it as an RL action.
- Current single-target onscreen command already supports grasp + lift + place:
  - Cup: `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --keep-open-sec 5`
  - Mug: `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label mug --grasp-type handle_top_down --keep-open-sec 5`
- Current RL code in `rl/harness.py` is a legacy low-level continuous-action baseline. The cup/mug ordering work must add a new `cup_ordering` mode instead of reinterpreting the old continuous control contract.

## Corrected Design Decisions

- Replace the external plan's `MAX_CUPS=4` with `MAX_TARGETS=5`, because the current environment intentionally contains 5 objects.
- Replace the external plan's "single unified target area" with the current classification objective:
  - handled mugs go to the nearest sink basin
  - plain cups go to the right/front counter edge
- RL action is `Discrete(5)` and means "select the next unfinished drinkware slot".
- RL must not choose:
  - grasp type
  - candidate id
  - gripper width
  - target pose
  - trajectory
  - arm assignment
  - placement zone
- Completion means successful placement in the correct class-specific zone, not merely successful grasp or lift.
- Training should be treated as a small-budget sanity path first. The first reliable milestone is a fair random vs greedy vs RL-compatible evaluation harness, not guaranteed RL superiority.

## Success Criteria

- `CupMugSorting` can run as a multi-object episode with up to 5 target-selection decisions.
- Random, greedy, and RL policies all use the same observation builder and action mask.
- Invalid actions are logged and do not call low-level execution.
- Completed objects are masked out and cannot be selected again.
- Logs distinguish ordering failure from perception failure, grasp failure, placement failure, timeout, collision, and invalid action.
- Existing `detected_objects` schema and `perception_queue` semantics are unchanged.
- Existing single-target cup/mug onscreen commands continue to work.

---

### Task 1: Add Cup Ordering Config Contract

**Files:**
- Create: `configs/rl/cup_mug_ordering_robocasa.yaml`
- Modify: `rl/contracts.py`
- Test: `tests/test_rl_contracts.py`

**Step 1: Write failing config validation tests**

Add tests that load `configs/rl/cup_mug_ordering_robocasa.yaml` and assert:

```python
self.assertEqual(summary["mode"], "cup_ordering")
self.assertEqual(summary["max_targets"], 5)
self.assertEqual(summary["action_space"], "Discrete(5)")
self.assertIn("correct_zone_place", summary["reward_terms"])
self.assertIn("invalid_action", summary["reward_terms"])
```

Also assert existing RL configs still validate.

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_rl_contracts.py"
```

Expected: failure because `cup_mug_ordering_robocasa.yaml` and `cup_ordering` validation do not exist yet.

**Step 3: Add the config**

Create `configs/rl/cup_mug_ordering_robocasa.yaml` with this semantic shape:

```yaml
mode: cup_ordering
task_name: robocasa/CupMugSorting
backend: robocasa
max_targets: 5
action_space:
  type: discrete
  n: 5
scene:
  layout_and_style_ids:
    - [1, 1]
  num_mugs: 2
  num_cups: 3
  seed: null
ordering:
  valid_target_labels:
    - cup
    - glass cup
    - mug
  handled_zone: sink
  plain_zone: opposite_counter
  handled_strategy: handle_top_down
  plain_strategy: top_down
policies:
  - random
  - greedy
  - rl
reward:
  correct_zone_place: 5.0
  selected_target_success: 2.0
  grasp_failure: -1.0
  placement_failure: -2.0
  invalid_action: -2.0
  retry_penalty: -0.2
  timeout: -3.0
  collision: -5.0
tiers:
  low:
    object_count: 2
    num_mugs: 1
    num_cups: 1
  medium:
    object_count: 4
    num_mugs: 2
    num_cups: 2
  high:
    object_count: 5
    num_mugs: 2
    num_cups: 3
eval:
  episodes_per_policy: 30
  tiers:
    - low
    - medium
    - high
  episodes_per_tier: 10
  seed_start: 42
timeouts:
  episode_sec: 300
  action_sec: 120
artifacts:
  root_dir: outputs/rl_cup_mug_ordering
  logs_dir: logs
  reports_dir: reports
```

**Step 4: Extend contracts minimally**

In `rl/contracts.py`, add a separate validator path for `mode: cup_ordering`. Do not weaken existing milestone selector validation.

Required checks:

- `mode == "cup_ordering"`
- `max_targets == 5`
- `action_space.type == "discrete"`
- `action_space.n == max_targets`
- `scene.layout_and_style_ids == [[1, 1]]`
- `ordering.handled_strategy == "handle_top_down"`
- `ordering.plain_strategy == "top_down"`
- reward contains all required keys
- eval tiers contain `low`, `medium`, `high`

**Step 5: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_rl_contracts.py"
```

Expected: pass.

**Step 6: Commit when requested**

Do not commit unless the user explicitly asks. Suggested future commit:

```powershell
git add configs/rl/cup_mug_ordering_robocasa.yaml rl/contracts.py tests/test_rl_contracts.py
git commit -m "logic/rl: add cup mug ordering contract"
```

---

### Task 2: Build 5-Slot Scene Observation and Masks

**Files:**
- Create: `rl/cup_mug_observation.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write failing observation tests**

Create tests for:

- 0 visible objects
- 1 visible object
- 2 visible objects
- 5 visible objects
- more than 5 objects
- finished objects
- low-confidence objects
- unknown cup type

Example expected behavior:

```python
obs = build_cup_mug_ordering_observation(scene, max_targets=5)
self.assertEqual(len(obs["targets"]), 5)
self.assertEqual(obs["valid_action_mask"], [1, 1, 0, 0, 0])
self.assertEqual(obs["finished_mask"], [0, 0, 0, 0, 0])
```

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: fail because `rl/cup_mug_observation.py` does not exist.

**Step 3: Implement minimal observation builder**

Implement:

```python
MAX_TARGETS = 5

def build_cup_mug_ordering_observation(scene, max_targets=MAX_TARGETS):
    ...
```

Each target slot should include:

- `slot_index`
- `object_name`
- `label`
- `type_code`: `handled`, `plain`, or `unknown`
- `has_handle`
- `conf`
- `pos`
- `target_zone`
- `recommended_grasp`
- `candidate_count`
- `best_candidate_score`
- `reachability`
- `distance_to_zone`
- `local_clutter`
- `retry_count`
- `finished`
- `valid`

Global fields:

- `remaining_count`
- `completed_count`
- `decision_step`
- `last_action_success`
- `last_failure_type`
- `counts.n1`
- `counts.n2`
- `counts.n3`

Mask fields:

- `valid_action_mask`
- `finished_mask`

**Step 4: Deterministic ordering**

Use deterministic slot ordering:

1. Known sim object name order: `mug_1`, `cup_1`, `mug_2`, `cup_2`, `cup_3`
2. If object names are missing, sort by `(x, y, label, rounded position)`
3. If more than 5 objects are present, keep the top 5 by deterministic order and mark overflow in `truncated_count`

**Step 5: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: pass for observation and masks.

**Step 6: Commit when requested**

Suggested future commit:

```powershell
git add rl/cup_mug_observation.py tests/test_cup_mug_ordering_contracts.py
git commit -m "logic/rl: add cup mug scene observation"
```

---

### Task 3: Add Ordering Policy Interface and Baselines

**Files:**
- Create: `rl/cup_mug_policies.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write failing policy tests**

Tests should cover:

- random never selects invalid slots
- greedy uses the same observation input as random
- greedy tie-breaks deterministically by lower slot index
- unknown policy name raises a controlled error
- policy output is a target slot index, not a grasp candidate id

Example:

```python
action = select_cup_mug_target(obs, policy="greedy")
self.assertEqual(action["selected_slot"], 1)
self.assertNotIn("candidate_id", action)
```

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: fail because policy helpers do not exist.

**Step 3: Implement policy interface**

Implement:

```python
def select_cup_mug_target(observation, policy, rng=None, model=None):
    if policy == "random":
        return select_random_target(observation, rng=rng)
    if policy == "greedy":
        return select_greedy_target(observation)
    if policy == "rl":
        return select_rl_target(observation, model=model)
    raise ValueError(...)
```

Random:

- sample uniformly over `valid_action_mask == 1`
- if no valid actions, return controlled no-op with `invalid_reason: "no_valid_targets"`

Greedy score:

```text
score =
  1.0 * best_candidate_score
  + 0.7 * conf
  + 0.5 * reachability
  - 0.4 * local_clutter
  - 0.2 * distance_to_zone
  - 0.3 * retry_count
```

**Step 4: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: pass.

**Step 5: Commit when requested**

Suggested future commit:

```powershell
git add rl/cup_mug_policies.py tests/test_cup_mug_ordering_contracts.py
git commit -m "logic/rl: add cup mug ordering policies"
```

---

### Task 4: Define Completion and Episode State

**Files:**
- Create: `rl/cup_mug_episode.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write failing completion tests**

Tests should assert:

- successful mug placement in sink marks the selected mug finished
- successful cup placement in opposite counter marks the selected cup finished
- grasp-only success does not mark completion
- lift-only success does not mark completion
- placement into wrong zone fails completion
- completed object remains invalid even if still visible

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: fail because episode state helpers do not exist.

**Step 3: Implement episode state**

Implement a small state object:

```python
class CupMugOrderingEpisode:
    def __init__(self, max_targets=5):
        self.finished = {}
        self.retry_counts = {}
        self.cup_order = []
        self.invalid_action_count = 0
        self.decision_step = 0
```

Add methods:

- `mark_selected(slot)`
- `mark_finished(object_name, zone)`
- `record_failure(object_name, failure_phase)`
- `is_finished(object_name)`
- `build_masks(scene_targets)`

**Step 4: Completion rule**

Use:

- `assignment.has_handle == True` requires `place_plan.zone == "sink"` and `place.place_success == True`
- `assignment.has_handle == False` requires `place_plan.zone == "opposite_counter"` and `place.place_success == True`

Do not use only `lift_success`.

**Step 5: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: pass.

**Step 6: Commit when requested**

Suggested future commit:

```powershell
git add rl/cup_mug_episode.py tests/test_cup_mug_ordering_contracts.py
git commit -m "logic/rl: track cup mug ordering episode state"
```

---

### Task 5: Add Multi-Object Execution Runner

**Files:**
- Create: `rl/cup_mug_runner.py`
- Modify: `scripts/demo_robocasa_reach_onscreen.py` only if reusable helpers must be exposed cleanly
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write failing runner tests with fakes**

Do not start RoboCasa in unit tests. Use fake scene snapshots and fake execution results.

Tests should assert:

- runner calls policy once per unfinished object
- invalid policy action does not call execution
- selected cup routes to `top_down`
- selected mug routes to `handle_top_down`
- finished target is not selected again
- runner stops when all 5 objects are complete

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: fail because runner does not exist.

**Step 3: Implement fake-testable runner**

Implement a pure orchestration layer first:

```python
def run_cup_mug_ordering_episode(
    env,
    perception_fn,
    execute_target_fn,
    policy,
    config,
    rng=None,
):
    ...
```

`execute_target_fn` should receive only:

- selected object metadata
- requested target label
- deterministic grasp type inferred from `has_handle`

It must not receive an RL-selected grasp candidate.

**Step 4: Add bridge to current single-target logic**

Use the current logic from `scripts/demo_robocasa_reach_onscreen.py` as the reference path:

- classify drinkware
- choose matching target
- choose candidate
- reach
- close
- lift
- choose place target with `planner/sorting_zones.py`
- execute place

Keep the first implementation narrow. If extracting helpers from the large script is too risky, create a script-level runner that calls the same local functions without changing behavior.

**Step 5: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: pass with fake runner tests.

**Step 6: Commit when requested**

Suggested future commit:

```powershell
git add rl/cup_mug_runner.py scripts/demo_robocasa_reach_onscreen.py tests/test_cup_mug_ordering_contracts.py
git commit -m "logic/rl: add cup mug ordering runner"
```

---

### Task 6: Add Train/Eval CLI Dry Runs

**Files:**
- Modify: `scripts/train_rl.py`
- Modify: `scripts/eval_rl.py`
- Modify: `rl/harness.py` or create `rl/cup_mug_harness.py`
- Test: `tests/test_rl_contracts.py`

**Step 1: Write failing CLI tests**

Add dry-run tests that execute:

```powershell
python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --timesteps 10 --dry-run --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 2 --tier low --dry-run --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 2 --tier low --dry-run --print-summary
```

Expected output should include:

- `mode=cup_ordering`
- `max_targets=5`
- `action_space=Discrete(5)`
- `policy_type=random` or `policy_type=greedy`
- `tier=low`

**Step 2: Run tests to verify failure**

Run:

```powershell
python -m unittest discover -s tests -p "test_rl_contracts.py"
```

Expected: fail because the CLI does not accept ordering-specific args.

**Step 3: Add CLI arguments**

In `scripts/eval_rl.py`, add:

- `--policy`
- `--episodes`
- `--tier`
- `--tiers`
- `--seed`
- `--timeout-sec`

In `scripts/train_rl.py`, add:

- `--timeout-sec`
- keep `--timesteps`
- keep `--dry-run`
- keep `--print-summary`

**Step 4: Dispatch by config mode**

If config has `mode: cup_ordering`, route to new cup/mug ordering harness.
Otherwise keep existing RL harness behavior.

**Step 5: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_rl_contracts.py"
```

Expected: pass.

**Step 6: Manual dry-run commands**

Run:

```powershell
python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --timesteps 10 --dry-run --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 2 --tier low --dry-run --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 2 --tier low --dry-run --print-summary
```

Expected: all exit 0 and print ordering summaries.

**Step 7: Commit when requested**

Suggested future commit:

```powershell
git add scripts/train_rl.py scripts/eval_rl.py rl/harness.py rl/cup_mug_harness.py tests/test_rl_contracts.py
git commit -m "eval/rl: route cup mug ordering cli"
```

---

### Task 7: Add Structured Ordering Logs

**Files:**
- Modify: `runtime/run_logger.py` or create `runtime/cup_mug_ordering_logger.py`
- Test: `tests/test_protocol_contracts.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write failing log schema tests**

Assert logs include:

- `commit_hash`
- `scene_config`
- `policy_type`
- `selected_slot`
- `selected_object_name`
- `cup_order`
- `cup_features_snapshot`
- `valid_action_mask`
- `finished_mask`
- `invalid_action_count`
- `episode_tier`
- `seed`
- `reward_breakdown`
- `task_success`
- `cup_success_count`
- `elapsed_decision_steps`
- `collision_count`
- `emergency_count`
- `counts.n1`
- `counts.n2`
- `counts.n3`
- `failure_mode`

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_protocol_contracts.py"
```

Expected: fail because ordering log helper does not exist.

**Step 3: Implement small log builder**

Prefer a small helper:

```python
def build_cup_mug_ordering_log(...):
    return {...}
```

Keep it independent from generated runtime artifacts.

**Step 4: Add reward breakdown**

Use stable keys:

- `correct_zone_place`
- `selected_target_success`
- `grasp_failure`
- `placement_failure`
- `invalid_action`
- `retry_penalty`
- `timeout`
- `collision`

**Step 5: Run tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_protocol_contracts.py"
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: pass.

**Step 6: Commit when requested**

Suggested future commit:

```powershell
git add runtime/cup_mug_ordering_logger.py runtime/run_logger.py tests/test_protocol_contracts.py tests/test_cup_mug_ordering_contracts.py
git commit -m "eval/logging: record cup mug ordering metrics"
```

---

### Task 8: Add Onscreen Multi-Object Smoke Script

**Files:**
- Create: `scripts/demo_cup_mug_ordering_onscreen.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write CLI contract tests**

Use subprocess dry-run or parser tests to assert the script accepts:

- `--policy random`
- `--policy greedy`
- `--policy rl`
- `--episodes`
- `--tier`
- `--seed`
- `--keep-open-sec`
- `--print-summary`

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

Expected: fail because script does not exist.

**Step 3: Implement script**

The script should:

1. Create `CupMugSorting` with fixed `layout_and_style_ids: [[1, 1]]`
2. Reset the environment
3. For each decision step:
   - build current scene observation
   - select target with policy
   - execute deterministic target pipeline
   - update episode state
   - print compact JSON summary
4. Stop when all valid targets are finished or timeout/failure budget is reached

**Step 4: Keep visual smoke separate from unit tests**

Do not make unit tests import heavy RoboCasa startup. Unit tests should validate parser and pure runner behavior only.

**Step 5: Manual onscreen smoke commands**

Run low-clutter first:

```powershell
python scripts/demo_cup_mug_ordering_onscreen.py --policy greedy --tier low --seed 42 --keep-open-sec 5 --print-summary
```

Then full scene:

```powershell
python scripts/demo_cup_mug_ordering_onscreen.py --policy greedy --tier high --seed 42 --keep-open-sec 5 --print-summary
```

Expected:

- config prints `layout_and_style_ids: [[1, 1]]`
- selected target changes after completion
- completed target is masked out
- cup uses `top_down`
- mug uses `handle_top_down`
- `place_plan.zone` is `opposite_counter` for cup and `sink` for mug

**Step 6: Commit when requested**

Suggested future commit:

```powershell
git add scripts/demo_cup_mug_ordering_onscreen.py tests/test_cup_mug_ordering_contracts.py
git commit -m "eval/rl: add cup mug ordering onscreen smoke"
```

---

### Task 9: Add Real RoboCasa Smoke Gates

**Files:**
- Modify: `scripts/eval_rl.py`
- Modify: `rl/cup_mug_harness.py`
- Modify: `runtime/cup_mug_ordering_logger.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Add controlled failure tests**

Use fake env startup exceptions and assert the harness returns structured failure instead of an uncaught traceback.

Expected fields:

- `task_success: false`
- `failure_mode: "environment_startup"` or a more specific controlled label
- `scene_config`
- `policy_type`
- `seed`
- `counts`

**Step 2: Implement timeout/failure wrapper**

Wrap one-episode smoke so these become structured failures:

- RoboCasa import/startup failure
- reset failure
- perception target missing
- invalid action
- execution failure
- place failure
- timeout

Do not suppress collision or emergency handling.

**Step 3: Manual smoke command**

Run:

```powershell
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 1 --tier low --seed 42 --timeout-sec 120 --print-summary
```

Expected:

- exit 0 with success summary, or controlled failure summary
- no uncaught traceback
- summary contains `policy_type=random`, `tier=low`, `seed=42`, `scene_config`, and `counts`

**Step 4: Small-budget RL sanity**

Run:

```powershell
python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --timesteps 1000 --seed 42 --timeout-sec 300 --print-summary
```

Expected:

- exit 0 with training summary, or controlled timeout/failure
- no uncaught traceback
- collision / emergency handling is not suppressed

**Step 5: Commit when requested**

Suggested future commit:

```powershell
git add scripts/eval_rl.py scripts/train_rl.py rl/cup_mug_harness.py runtime/cup_mug_ordering_logger.py tests/test_cup_mug_ordering_contracts.py
git commit -m "eval/rl: add robocasa cup mug smoke gates"
```

---

### Task 10: Add Fair 30-Episode Policy Evaluation

**Files:**
- Modify: `scripts/eval_rl.py`
- Modify: `rl/cup_mug_harness.py`
- Create or modify: `rl/cup_mug_reports.py`
- Test: `tests/test_cup_mug_ordering_contracts.py`

**Step 1: Write failing schedule tests**

Assert:

- `episodes=30`
- tiers `low,medium,high`
- exactly 10 episodes per tier
- random, greedy, and rl share identical seed/tier schedule

**Step 2: Implement schedule builder**

Add:

```python
def build_tier_seed_schedule(episodes, tiers, seed_start):
    ...
```

For 30 and 3 tiers:

- 10 low
- 10 medium
- 10 high

**Step 3: Implement report aggregation**

Aggregate fields:

- `policy_type`
- `task_success_rate`
- `cup_success_rate`
- `avg_retries`
- `avg_decision_steps`
- `avg_sim_steps`
- `invalid_action_count`
- `collision_count`
- `emergency_count`
- `counts.n1`
- `counts.n2`
- `counts.n3`

**Step 4: Dry-run fairness commands**

Run:

```powershell
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 30 --tiers low,medium,high --seed 42 --dry-run --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 30 --tiers low,medium,high --seed 42 --dry-run --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --episodes 30 --tiers low,medium,high --seed 42 --dry-run --print-summary
```

Expected:

- each summary reports 30 total episodes
- each has 10 low, 10 medium, 10 high
- seed schedule is identical across policies

**Step 5: Real evaluation commands**

Run only after Task 9 smoke is acceptable:

```powershell
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 30 --tiers low,medium,high --seed 42 --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 30 --tiers low,medium,high --seed 42 --print-summary
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --episodes 30 --tiers low,medium,high --seed 42 --print-summary
```

Expected:

- failures are included in averages
- failed episodes are not dropped
- reports are comparable by policy and tier

**Step 6: Commit when requested**

Suggested future commit:

```powershell
git add scripts/eval_rl.py rl/cup_mug_harness.py rl/cup_mug_reports.py tests/test_cup_mug_ordering_contracts.py
git commit -m "eval/rl: compare cup mug ordering policies"
```

---

### Task 11: Update Documentation and Project Board

**Files:**
- Modify: `docs/task.md`
- Modify: `docs/getting_start_cjy.md`
- Optionally modify: `docs/index.md`

**Step 1: Update getting-started docs**

Add a short section to `docs/getting_start_cjy.md` with:

- cup/mug ordering dry-run commands
- onscreen ordering smoke command
- random / greedy / rl evaluation commands
- warning that RL selects target order only

**Step 2: Update task board**

Update `docs/task.md` only with git-relevant status:

- contract/config status
- observation/policy status
- smoke verification status
- recommended next step

Do not record `.venv`, caches, generated model checkpoints, videos, or local runtime paths.

**Step 3: Verify docs links**

Run:

```powershell
rg -n "cup_mug_ordering|demo_cup_mug_ordering|CupMugSorting" docs
```

Expected: docs mention the new commands and do not point to stale `MAX_CUPS=4`.

**Step 4: Commit when requested**

Suggested future commit:

```powershell
git add docs/task.md docs/getting_start_cjy.md docs/index.md
git commit -m "infra/docs: document cup mug ordering rl"
```

---

### Task 12: Final Health Checks and Scope Audit

**Files:**
- Inspect: all changed source/config/test/docs files

**Step 1: Run focused tests**

Run:

```powershell
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
python -m unittest discover -s tests -p "test_rl_contracts.py"
python -m unittest discover -s tests -p "test_protocol_contracts.py"
python -m unittest discover -s tests -p "test_candidate_planner.py"
python -m unittest discover -s tests -p "test_sorting_policy.py"
python -m unittest discover -s tests -p "test_sorting_zones.py"
```

Expected: pass. If RoboCasa import tests time out in this shell, record the exact timeout and keep pure ordering tests green.

**Step 2: Run syntax checks**

Run:

```powershell
python -c "from pathlib import Path; files=['rl/contracts.py','rl/cup_mug_observation.py','rl/cup_mug_policies.py','rl/cup_mug_episode.py','rl/cup_mug_runner.py','scripts/train_rl.py','scripts/eval_rl.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"
```

Expected: `syntax ok`.

**Step 3: Search for forbidden scope creep**

Run:

```powershell
rg -n "oblique_reach|side_reach|handle_oblique|handle_anchor|anchor_top_down|MAX_CUPS=4|MAX_CUPS: 4|unified target" rl scripts configs tests docs
```

Expected:

- no active references to removed grasp paths
- no `MAX_CUPS=4`
- no unified target-area completion for `CupMugSorting`

**Step 4: Check git status**

Run:

```powershell
git status --short
```

Expected:

- only intended source/config/test/docs changes
- no `.venv/`
- no `__pycache__/`
- no generated videos
- no model checkpoints
- no runtime output artifacts

**Step 5: Commit when requested**

Suggested final commit:

```powershell
git add rl scripts configs tests runtime docs
git commit -m "infra/tests: verify cup mug ordering integration"
```

---

## Recommended Implementation Order

1. Contract/config
2. Observation builder
3. Policy interface
4. Episode state
5. Fake-testable runner
6. CLI dry-runs
7. Logging schema
8. Onscreen ordering smoke
9. Real RoboCasa smoke gates
10. 30-episode fair evaluation
11. Docs/task board
12. Final health checks

## Practical First Milestone

The first useful milestone is not PPO training. The first useful milestone is:

```powershell
python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 2 --tier low --dry-run --print-summary
python scripts/demo_cup_mug_ordering_onscreen.py --policy greedy --tier low --seed 42 --keep-open-sec 5 --print-summary
```

Success means the system can repeatedly choose unfinished targets and route them through the existing deterministic cup/mug grasp and place pipeline.

## Risks

- RoboCasa startup and rendering can be slow or flaky in local shell tests. Keep heavy simulator checks out of unit tests.
- Placement may still fail if a selected target is too far from a reachable sink/counter target. Use structured failure logs instead of treating that as RL failure.
- Perception may not see all 5 objects from one camera angle. Use sim metadata fallback only for `CupMugSorting` and record whether fallback was used.
- PPO may not outperform greedy with a small budget. Treat RL as an experimental comparison path, not a guaranteed improvement.

## Non-Goals

- Do not train DINO or change semantic segmentation.
- Do not add new grasp modes.
- Do not let RL command robot actions directly.
- Do not change `detected_objects` schema.
- Do not change `perception_queue` IPC semantics.
- Do not implement multi-agent RL.
- Do not expand beyond the current 5 cup/mug task until the deterministic episode runner is stable.
