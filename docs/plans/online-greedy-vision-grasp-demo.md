# Online Greedy + Vision + Grasp Demo Runner Plan

## TL;DR
> **Summary**: Build a real-time Cup/Mug ordering demo runner for `robocasa/CupMugSorting` that keeps the current high-level ordering contract (`Discrete(5)` over stable drinkware slots) but executes each selected target through the existing live vision → candidate selection → reach / close / lift / place pipeline in one continuous MuJoCo viewer session. Start with `greedy` as the canonical online policy because it is deterministic and explainable; only after that path is stable should the same runner accept `policy=rl`.
> **Goal**: In a single RoboCasa environment reset, show one full episode of sequential target selection and physical execution over up to 5 drinkware objects, with viewer-visible execution, per-step summaries, and end-of-episode status.
> **Non-goal**: This plan does **not** introduce learned low-level control, real PPO policy loading, or multi-agent behavior. It also does **not** claim RL performance gains.

## Context
- Current implementation already has:
  - `rl/cup_mug_observation.py` fixed 5-slot observation/mask builder.
  - `rl/cup_mug_policies.py` with `random` / deterministic `greedy` and `policy=rl` sanity stub.
  - `rl/cup_mug_runner.py` deterministic high-level episode runner with per-step logs.
  - `scripts/demo_robocasa_reach_onscreen.py` single-target live execution path for cup/mug reach / close / lift / place.
  - `scripts/demo_multi_cup_mug_sort.py` live multi-object CupMugSorting scene + classification smoke path.
  - `rl/harness.py` dry-run render/video/trace path and real RoboCasa background frame support under `robotic-robocasa-rl`.
- Current gap: the ordering layer and the real live execution layer are still separate. We can inspect ordering decisions and render traces, and we can execute one real selected cup/mug, but we do not yet have one continuous online episode that does:
  1. observe scene
  2. choose next slot
  3. map slot to current visible target
  4. execute grasp/place in the live env
  5. update finished state
  6. continue until all valid targets are done or failure stops the run

## Core Objective
Implement a dedicated online demo runner that combines:
- high-level ordering policy (`greedy` first),
- current live vision/classification stack,
- current real RoboCasa grasp/place execution stack,
- and one continuous MuJoCo viewer session.

Success means the user can watch a full CupMugSorting episode unfold in the viewer and correlate each physical step with structured per-step decision logs.

## Scope Boundaries
### Must Have
- A dedicated script entrypoint, separate from `main.py`, for a continuous online ordering demo.
- Initial supported policy: `greedy`.
- Stable slot-to-target mapping using the same 5-slot semantics already used by dry-run ordering.
- Re-sense / re-classify between each executed target.
- Per-step viewer-visible execution in the same RoboCasa env/session.
- End-of-episode summary with `selected_order`, executed `object_ids`, success/failure, and final counts.
- Optional JSON trace artifact for the live run, reusing the existing artifact conventions where sensible.

### Must Not Have
- Do not let RL choose grasp type, candidate id, arm command, trajectory, retry fallback, or placement zone.
- Do not push substantial ordering orchestration into `main.py`.
- Do not silently continue after unrecoverable execution failures.
- Do not revive deprecated `handle_grasp` as a public online action.
- Do not replace the current dry-run evaluation path; this is an additional online demo capability.

## High-Level Design
### 1. Online episode controller
Add a small controller module / script that owns one continuous CupMugSorting env session.

Suggested home:
- `scripts/demo_online_cup_mug_ordering.py`

Responsibilities:
- create env with the same stable scene contract (`layout_and_style_ids=[[1,1]]`, `PandaOmron`, RGB-D camera)
- reset once
- loop until success / stop condition
- call vision/classification on current frame
- build current 5-slot observation
- choose action via `greedy`
- map selected slot back to the best current visible target/assignment
- execute that target through the existing live execution path
- update finished/retry/failure state
- render continuously and print/save structured step summaries

### 2. Slot-to-live-target mapping
The dry-run observation builder already works from detected targets + assignments. For the live online path, we need a deterministic adapter:
- read all visible targets via `VisionPerceptionLoop.infer_all_targets_with_diagnostics()`
- attach sim metadata using `assign_sim_metadata_to_targets()`
- classify with `classify_drinkware_targets()`
- build observation with finished/retry state carried across the episode
- after greedy selects slot `i`, find the live target/assignment whose `sim_object_name` matches that slot

Guardrails:
- if chosen slot is currently invalid/missing, do **not** blindly execute
- either terminate with structured failure, or perform one controlled re-sense and retry decision depending on the configured policy

### 3. Reuse of real execution path
Do **not** reimplement grasp execution.

Reuse from `scripts/demo_robocasa_reach_onscreen.py`:
- `infer_candidate_from_obs()` style logic for target → candidate
- `execute_reach()` / `execute_oriented_top_down_reach()`
- `execute_close()`
- `execute_lift()`
- `choose_cup_mug_place_target()`
- `execute_place()`

Recommended refactor:
- extract the single-target live execution sequence into a reusable helper module so both the old single-target demo and the new online ordering demo call the same execution helper

### 4. State update rules
After each attempted execution:
- success in correct zone → mark slot finished
- invalid action / missing target → record failure and stop or controlled retry according to explicit policy
- reach / close / lift / place failure → record phase, leave slot unfinished, and decide whether the episode stops or continues

Recommended Phase-1 policy:
- keep it conservative
- if a selected target fails execution, stop the online demo and emit structured failure

Rationale: easier to debug, safer, and aligns with "demo" quality before adding retry orchestration.

### 5. Viewer + artifact behavior
The online demo must prioritize the live MuJoCo viewer.

Optional artifacts:
- JSON trace file for the online episode
- saved RGB frames or mp4 only if the environment supports them without destabilizing the live viewer

The online viewer path should not depend on offline synthetic overlays. Those remain useful for dry-run evaluation, but the online demo's primary output is the live RoboCasa scene itself.

## Implementation Tasks

### Task A — Extract reusable live target execution helper
**Files**:
- create or extend helper under `arm/` or `scripts/` support modules
- modify `scripts/demo_robocasa_reach_onscreen.py`

**Goal**:
Refactor the single-target cup/mug execution sequence into a callable helper that accepts:
- env
- obs
- selected target/assignment
- camera / threshold / vision config
- render timing options

Returns:
- updated obs
- structured execution summary
- success/failure phase

**Acceptance**:
- existing single-target onscreen cup/mug commands still work
- no behavioral drift in current single-target demo path

**QA Scenarios**:
```
Scenario: Existing single-target cup path still works
  Tool: Bash
  Steps: Run `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --layout 1 --keep-open-sec 1` in the verified RoboCasa environment.
  Expected: Command reaches the existing cup path without import/runtime regression; stdout still contains candidate / execution summaries.

Scenario: Existing single-target mug path still works
  Tool: Bash
  Steps: Run `python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label mug --grasp-type handle_top_down --layout 1 --keep-open-sec 1` in the verified RoboCasa environment.
  Expected: Command reaches the existing mug path without import/runtime regression; stdout still contains candidate / execution summaries.
```

### Task B — Build live slot selection adapter
**Files**:
- new `rl/` or `scripts/` helper

**Goal**:
Given current scene observations, produce:
- live `targets`
- live `assignments`
- 5-slot observation
- mapping from slot index → current assignment

**Acceptance**:
- deterministic slot mapping for visible `mug_1`, `mug_2`, `cup_1`, `cup_2`, `cup_3`
- controlled behavior when a slot is missing or invalid

**QA Scenarios**:
```
Scenario: Stable slot mapping under current CupMugSorting scene
  Tool: Bash
  Steps: Run the new unit/helper test command or a dedicated debug script for the live slot adapter against a deterministic `CupMugSorting` scene.
  Expected: Output maps visible objects into the stable slot order `mug_1`, `mug_2`, `cup_1`, `cup_2`, `cup_3` when metadata is available.

Scenario: Missing/invalid selected slot fails cleanly
  Tool: Bash
  Steps: Run the same verification with a fixture or debug mode where one selected slot is currently invalid or absent.
  Expected: The adapter emits a structured invalid/missing-target result and does not attempt execution.
```

### Task C — Implement continuous online demo runner script
**Files**:
- create `scripts/demo_online_cup_mug_ordering.py`

**Suggested CLI**:
- `--task robocasa/CupMugSorting`
- `--policy greedy`
- `--seed`
- `--layout`
- `--style`
- `--camera-name`
- `--render-sleep-sec`
- `--keep-open-sec`
- `--save-trace`
- optional `--save-video`

**Loop**:
1. reset env
2. render first frame
3. sense + classify
4. build observation
5. choose next slot via `greedy`
6. resolve live target for that slot
7. execute single-target live helper
8. update finished/retry/failure state
9. continue or stop

**Acceptance**:
- one command runs one continuous multi-object online episode
- viewer remains open and shows sequential execution
- per-step stdout/JSON logs identify slot, object, strategy, zone, and result

**QA Scenarios**:
```
Scenario: Greedy online demo starts and emits per-step logs
  Tool: Bash
  Steps: Run `conda activate robotic-robocasa-rl` and then `python scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy greedy --layout 1 --style 1 --keep-open-sec 1`.
  Expected: One continuous viewer session starts; stdout or JSON logs include per-step `selected_slot`, `object_id`, `grasp_strategy`, `place_zone`, and success/failure fields.

Scenario: Online demo stops cleanly on first execution failure
  Tool: Bash
  Steps: Run the same command in a scene/seed that triggers a real execution failure, or use a controlled debug flag / injected failure mode if provided.
  Expected: The episode terminates with an explicit structured failure summary instead of silently continuing.
```

### Task D — Add structured online trace output
**Files**:
- `runtime/run_logger.py` or a dedicated helper
- tests if needed

**Goal**:
Record a compact online-demo artifact with:
- `policy`
- `seed`
- `selected_order`
- `executed_object_ids`
- `step_logs`
- `success`
- `failure_phase`
- optional video/frame paths

**Acceptance**:
- online run can be inspected after the viewer closes

**QA Scenarios**:
```
Scenario: Online demo writes trace artifact
  Tool: Bash
  Steps: Run the online demo with trace-saving enabled.
  Expected: A JSON trace artifact exists and includes `policy`, `seed`, `selected_order`, `executed_object_ids`, `step_logs`, and success/failure summary.

Scenario: Trace links to optional visual artifacts
  Tool: Bash
  Steps: Run the online demo with optional frame/video capture enabled if supported.
  Expected: Trace records frame/video paths when produced, or a structured skip/fallback reason when not produced.
```

### Task E — Verification and guardrails
**Goal**:
Verify:
- single-target path still works
- online greedy path works in `robotic-robocasa-rl`
- dry-run ordering path is unchanged
- no generated runtime artifacts are added to source-controlled changes

**QA Scenarios**:
```
Scenario: Dry-run ordering path remains green
  Tool: Bash
  Steps: Run `python -m unittest discover -s tests -p "test_rl_contracts.py"` and `python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"`.
  Expected: Existing dry-run ordering tests still pass with no regression.

Scenario: No generated artifacts are staged
  Tool: Bash
  Steps: Run `git status --short` after the online demo changes.
  Expected: Only intended source/config/test/docs files appear; `outputs/`, videos, checkpoints, and other runtime artifacts are not part of the staged source change set.
```

## Verification Plan
### Unit / contract level
- keep `test_cup_mug_ordering_contracts.py`, `test_rl_contracts.py`, `test_protocol_contracts.py`, `test_sorting_policy.py`, `test_sorting_zones.py` green
- add only narrow new tests for any newly extracted pure helpers

### Manual / executable verification
Recommended staged commands:

1. Existing dry-run baseline regression:
```powershell
python -m unittest discover -s tests -p "test_rl_contracts.py"
python -m unittest discover -s tests -p "test_cup_mug_ordering_contracts.py"
```

2. Existing single-target live sanity:
```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label cup --grasp-type top_down --layout 1 --keep-open-sec 5
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CupMugSorting --target-label mug --grasp-type handle_top_down --layout 1 --keep-open-sec 5
```

3. New online greedy full-episode demo:
```powershell
conda activate robotic-robocasa-rl
python scripts/demo_online_cup_mug_ordering.py --task robocasa/CupMugSorting --policy greedy --layout 1 --style 1 --keep-open-sec 5
```

### Success Criteria for the new online demo
- one viewer session handles multiple targets sequentially
- at least one full visible order is emitted in logs
- selected slot and executed object id match the current live classification output
- cup targets use `top_down`; mug targets use `handle_top_down`
- place zones remain class-consistent
- on failure, the run stops with explicit structured failure instead of silent continuation

## Risks
- target identity can drift after object movement
- visual re-detection can fail after one object is moved
- real execution failures may leave the scene in a partially modified state
- viewer + video capture can destabilize the live loop in some environments

## Recommended First Implementation Target
Start with:
- `policy=greedy`
- one continuous viewer session
- stop-on-first-failure behavior
- optional trace JSON

Do **not** start with real learned policy loading.

## Definition of Done for this plan
- a dedicated online demo script exists
- it reuses current vision and execution logic instead of duplicating them
- greedy can complete or fail a full multi-object online episode in one MuJoCo session with structured logs
- dry-run ordering artifacts remain intact and unchanged
