# RL Cup Ordering Strategy for RoboCasa Tabletop Tidying

## TL;DR
> **Summary**: Add a high-level RL object-ordering layer that only chooses the next drinkware target to process in a multi-object RoboCasa tidying episode. Low-level grasp candidate selection, arm control, DINO perception, IPC, FSM retry handling, and placement execution remain deterministic/current-system responsibilities.
> **Deliverables**:
> - Cup/mug ordering RL contract with padded `MAX_TARGETS=5` observation, masks, reward schema, and invalid-action behavior.
> - Shared cup scene observation builder used by Random, risk-aware greedy, and RL policies.
> - RoboCasa-compatible ordering train/eval path with low/medium/high clutter tiers.
> - Structured logging and report outputs for 30 episodes per policy total: 10 low + 10 medium + 10 high.
> - Contract tests, CLI dry-runs, one-episode RoboCasa smoke, small-budget direct RoboCasa training sanity, and final policy comparison.
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 6 → Task 8 → Final Verification

## Context
### Original Request
- 当前已完成 RoboCasa 环境中有把手杯子和无把手杯子的 DINO 语义分割点云 Vision 识别。
- 当前继续做分类识别 Vision。
- 目标是在复杂、杂乱桌面上识别两种杯子，并准确夹取、放到指定位置。
- 最终加入 RL 学习选择策略，对比随机选择顺序，先验证强化学习接口与评估链路是否完整；Phase 1 不宣称 RL 已优于 greedy。

### Interview Summary
- RL 只学“先处理哪个杯子”，不学习低层抓取控制、grasp candidate 选择、机械臂轨迹、collision recovery、完整任务规划。
- 目标优先级：成功率 > 稳定性 = 耗时 > 碰撞。碰撞虽然不是主优化指标，但作为安全硬惩罚处理。
- 每个 episode 连续整理多个杯子。
- 当前场景规模：5 个 drinkware 目标（2 个 mug + 3 个 cup）+ 若干杂物。
- handled mugs 放到 sink；plain cups 放到 opposite/right counter，不再使用统一目标区域。
- Baseline：Random + Risk-aware greedy + RL。
- Heuristic baseline：risk-aware greedy，使用 candidate score、detection confidence、reachability、clutter/risk、distance。
- 训练路线：直接 RoboCasa 中训练。
- Evaluation：low / medium / high clutter 三档。
- 主评估预算：每个 policy 总计 30 episodes，按 low/medium/high 各 10 episodes 分配。

### Metis Review (gaps addressed)
- Added explicit cup-ordering contract separate from existing grasp-candidate selector semantics.
- Added cup identity, padding, finished-mask, invalid-action, completion, and >4-cup edge-case requirements.
- Added direct RoboCasa training gates to avoid unbounded RL runs: contract tests → dry-run CLI → deterministic RoboCasa smoke → small-budget training → 30-episode evaluation.
- Added fairness guardrails: Random, greedy, and RL must share the same observation builder, seeds, clutter tiers, completion contract, low-level execution, and logging schema.
- Added guardrail against further bloating `main.py`; orchestration must be moved into small RL/FSM/runtime helpers.
- Added explicit logging requirements for policy type, selected cup, cup order, reward breakdown, clutter tier, seed, invalid actions, and N1/N2/N3.

## Work Objectives
### Core Objective
Implement and verify a high-level RoboCasa cup/mug-ordering RL layer where `Discrete(MAX_TARGETS=5)` selects the next unfinished drinkware target to process, while existing perception, candidate generation, FSM retries, and execution perform the actual grasp/place behavior.

### Deliverables
- `configs/rl/cup_ordering_robocasa.yaml` configuration for ordering RL, baselines, reward weights, clutter tiers, seeds, artifacts, and RoboCasa train/eval settings.
- Cup-ordering contract additions in `rl/contracts.py` or a narrow companion module, without breaking existing selector configs.
- Cup scene observation builder and flattening path in `rl/` that produces deterministic padded observations and masks.
- Policy interface supporting `random`, `greedy`, and `rl` backends.
- RoboCasa ordering train/eval harness path integrated into `scripts/train_rl.py` and `scripts/eval_rl.py`.
- Ordering-specific structured logging in `runtime/run_logger.py` or a dedicated logger helper consumed by existing run logging.
- Contract and integration tests for observation shape, masks, invalid actions, reward/log fields, policy fairness, and CLI dry-runs.
- Evaluation outputs for 30 episodes per policy: 10 low, 10 medium, 10 high.
- `docs/task.md` update describing project status, verification performed, and recommended next step if implementation changes roadmap/status.

### Definition of Done (verifiable conditions with commands)
- `python -m unittest discover -s tests -p "test_rl_contracts.py"` exits `0`.
- `python -m unittest discover -s tests -p "test_protocol_contracts.py"` exits `0`.
- `python -m unittest discover -s tests -p "test_main_contracts.py"` exits `0`.
- `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` exits `0` after the new ordering contract tests are added.
- `python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --dry-run --print-summary` exits `0` and prints `max_targets=5`, `action_space=Discrete(5)`, reward component names, and ordering mode.
- `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 2 --dry-run --print-summary` exits `0` and prints policy/episode summary.
- `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 2 --dry-run --print-summary` exits `0` and prints greedy score components.
- `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 30 --dry-run` writes metrics and trace artifacts.
- `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 30 --dry-run` writes metrics and trace artifacts using the same observation/runner path.
- `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy rl --episodes 1 --dry-run` writes RL sanity metrics and trace artifacts without claiming superiority.

### Must Have
- `MAX_TARGETS=5` ordering contract.
- Observation padding and `valid_action_mask`.
- `finished_mask` preventing completed cups from being reselected.
- Deterministic cup ordering/indexing policy for each decision step.
- Invalid action behavior: penalty, no low-level execution call for invalid slot, structured log entry, controlled fallback only if explicitly configured.
- Cup/mug completion contract based on successful placement into the class-specific target zone (`sink` for handled mugs, `opposite_counter` for plain cups).
- Fixed low-level grasp selection: RL cannot select `top_down`, `handle_grasp`, candidate id, gripper width, trajectory, or arm command.
- Three policies using the same observation builder: Random, risk-aware greedy, RL.
- Risk-aware greedy score formula/config recorded in summary outputs.
- Direct RoboCasa training path with small-budget sanity gate and timeout controls.
- Low/medium/high clutter tier definitions in config.
- Run logs containing policy comparison fields and reproducibility metadata.
- Agent-executable tests and QA scenarios for all new behavior.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Do not train or modify DINO/segmentation/classification models.
- Do not change camera intrinsics sourcing rules or hardcode camera intrinsics.
- Do not change `perception_queue` IPC name, queue semantics, or timeout behavior.
- Do not remove or rename `target`, `obstacles`, `status`, or `grasp_candidates` in `detected_objects`.
- Do not make RL choose grasp candidate, grasp pose, gripper command, trajectory, fallback action, or arm assignment in this plan.
- Do not introduce sockets, files, TCP, or SharedMemory for perception-to-logic IPC.
- Do not silently suppress collision or emergency handling.
- Do not implement multi-zone placement, arbitrary object categories beyond cups, >4 target support, or multi-agent RL.
- Do not put substantial ordering orchestration into `main.py`; use small helper modules and keep `main.py` as wiring only.
- Do not use vague acceptance criteria such as “works well”; every task needs commands and expected fields.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after + contract-first checks, using existing `unittest` style and CLI smoke paths.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`.
- Primary verification layers:
  1. Static/contract tests for config, observation, masks, policy outputs, reward/log schema.
  2. CLI dry-runs for train/eval scripts.
  3. One-episode deterministic RoboCasa smoke with timeout and structured failure logging.
  4. Small-budget direct RoboCasa training sanity.
  5. 30-episode policy comparison for Random, greedy, and RL over low/medium/high clutter.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 contract/config, Task 2 observation builder, Task 3 completion/identity semantics, Task 4 baseline policy interface.
Wave 2: Task 5 reward/env adapter, Task 6 train/eval CLI integration, Task 7 logging/reporting, Task 8 tests.
Wave 3: Task 9 RoboCasa smoke/training gates, Task 10 evaluation protocol and scripts, Task 11 docs/task update and handoff evidence.
Wave 4: Task 12 cleanup/refactor guardrails and final readiness.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 5, 6, 8, 9, 10.
- Task 2 blocks Tasks 4, 5, 6, 7, 8, 9, 10.
- Task 3 blocks Tasks 5, 7, 8, 9, 10.
- Task 4 blocks Tasks 6, 8, 10.
- Task 5 blocks Tasks 6, 8, 9, 10.
- Task 6 blocks Tasks 9, 10.
- Task 7 blocks Tasks 9, 10, 11.
- Task 8 blocks Tasks 9, 10, 12.
- Task 9 blocks Task 10.
- Task 10 blocks Task 11.
- Task 11 blocks Task 12.
- Task 12 blocks Final Verification Wave.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → deep, quick, unspecified-high.
- Wave 2 → 4 tasks → deep, quick, unspecified-high.
- Wave 3 → 3 tasks → unspecified-high, writing.
- Wave 4 → 1 task → quick.

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Define cup-ordering RL contract and config

  **What to do**: Add a new cup/mug-ordering configuration and contract path separate from the existing candidate-selector semantics. Create `configs/rl/cup_mug_ordering_robocasa.yaml` with `mode: cup_ordering`, `max_targets: 5`, `action_space: discrete`, reward weights, seed schedule, artifact paths, and policy names `random`, `greedy`, `rl`. Extend `rl/contracts.py` or add a narrow companion under `rl/` so cup/mug-ordering configs validate without weakening existing legacy RL validation.
  **Must NOT do**: Do not reinterpret the existing grasp-candidate selector as cup ordering by renaming fields only. Do not break existing selector, PPO, or dual-arm configs. Do not add candidate-grasp actions.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: Contract changes are foundational and must preserve existing RL config behavior.
  - Skills: [] - No extra skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: Tasks 2, 5, 6, 8, 9, 10 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `rl/contracts.py` - Existing RL config validation style and selector contract checks.
  - Pattern: `configs/rl/single_arm_selector_robocasa.yaml` - Existing selector config shape; use only as config style reference, not semantic inheritance.
  - Pattern: `configs/rl/single_arm_robocasa_ppo.yaml` - Existing artifact/metrics/eval config conventions.
  - Guardrail: `docs/task.md:81-90` - Existing selector smoke/interface history; new work must be clearly cup-ordering, not candidate-ranking.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest discover -s tests -p "test_rl_contracts.py"` exits `0`.
  - [ ] `python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --dry-run --print-summary` exits `0` and prints `mode=cup_ordering`, `max_targets=5`, and `action_space=Discrete(5)`.
  - [ ] Existing `configs/rl/single_arm_selector_robocasa.yaml` still validates through its current tests.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Cup-ordering config validates
    Tool: Bash
    Steps: Run `python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --dry-run --print-summary`.
    Expected: Exit code 0; stdout includes `cup_ordering`, `max_targets=5`, `Discrete(5)`, and reward weight names.
    Evidence: .sisyphus/evidence/task-1-contract-dry-run.txt

  Scenario: Existing selector config remains valid
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_rl_contracts.py"`.
    Expected: Exit code 0; no regression in existing selector tests.
    Evidence: .sisyphus/evidence/task-1-existing-rl-tests.txt
  ```

  **Commit**: YES | Message: `logic/rl: add cup ordering contract` | Files: [`rl/contracts.py`, `configs/rl/cup_ordering_robocasa.yaml`, `tests/test_rl_contracts.py`]

- [ ] 2. Build deterministic cup scene observation and mask builder

  **What to do**: Implement a cup/mug scene observation builder in `rl/` that converts perception/FSM scene data into a fixed `MAX_TARGETS=5` observation. Include per-target features such as handle/type encoding, detection confidence, normalized world position, reachability proxy, candidate score, finished flag, and retry count. Include global features: remaining count, completed count, elapsed decision steps, N1/N2/N3, last action success, and last failure type. Include `valid_action_mask` and `finished_mask`. Use deterministic ordering by stable target slot semantics rather than random scene order.
  **Must NOT do**: Do not feed raw RGB-D into RL. Do not require DINO classification to be perfect; support `unknown` cup type. Do not mark padded cups valid. Do not silently randomize cup order.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: Observation semantics determine the whole MDP and fairness of all policies.
  - Skills: [] - No extra skill required.
  - Omitted: [`playwright`] - No browser/UI verification.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: Tasks 4, 5, 6, 7, 8, 9, 10 | Blocked By: Task 1 for final config constants, but can start from draft contract.

  **References**:
  - Pattern: `rl/harness.py` - Existing selector observation and flattening helpers.
  - Pattern: `vision/detected_objects.py` - Payload schema, status/confidence constraints, candidate fields.
  - Pattern: `vision/perception_loop.py` - Candidate score/type generation source.
  - Pattern: `fsm/state_machine.py` - Retry counters and candidate-aware state.
  - Pattern: `runtime/run_logger.py` - Existing count/failure metadata extraction patterns.

  **Acceptance Criteria**:
  - [ ] New tests cover 0, 1, 2, 4, and >4 detected cups.
  - [ ] Observation shape is deterministic and matches config for all covered cases.
  - [ ] `valid_action_mask` excludes padded, finished, low-confidence invalid, and truncated-out cups.
  - [ ] `unknown` cup type is encoded without error.

  **QA Scenarios**:
  ```
  Scenario: Padded multi-cup observation
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` after adding cases for partial scenes and MAX_TARGETS=5.
    Expected: Exit code 0; test asserts two valid cup slots and two invalid padded slots.
    Evidence: .sisyphus/evidence/task-2-observation-padding.txt

  Scenario: More than four cups handled deterministically
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` with a >4-cup fixture.
    Expected: Exit code 0; either deterministic truncation by configured score/order or controlled contract error, exactly as config specifies.
    Evidence: .sisyphus/evidence/task-2-overflow-cups.txt
  ```

  **Commit**: YES | Message: `logic/rl: add cup scene observation builder` | Files: [`rl/*.py`, `tests/test_cup_ordering_contracts.py`]

- [ ] 3. Define cup identity, completion, and finished-mask semantics

  **What to do**: Implement/centralize rules for maintaining target identity within an episode, marking objects completed, and preventing reselection. Completion means the selected target is successfully placed into the correct class-specific target zone (`sink` for handled mugs, `opposite_counter` for plain cups). If a selected target disappears before execution, route through existing perception lost/stale handling. If a completed target remains visible near its target zone, keep it finished and invalid for future actions. If perception lacks stable instance ids, maintain episode-local identity through nearest-neighbor position matching with deterministic tie-breakers.
  **Must NOT do**: Do not require long-term object tracking beyond this episode. Do not reselect completed cups. Do not consider grasp-only success as completion unless placement also succeeds.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: Requires careful edge-case handling across perception/FSM/eval but is narrower than full architecture.
  - Skills: [] - No extra skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: Tasks 5, 7, 8, 9, 10 | Blocked By: none

  **References**:
  - Pattern: `fsm/state_machine.py` - Existing success/failed/retry states and failure counts.
  - Pattern: `fsm/perception_cycle.py` - Existing invalid/stale perception transitions.
  - Pattern: `runtime/run_logger.py` - Existing failure stage and execution summary logging.
  - Pattern: `arm/env_wrapper.py` - RoboCasa wrapper environment success/info access patterns.

  **Acceptance Criteria**:
  - [ ] Completed cups are masked invalid on subsequent decisions.
  - [ ] A cup that disappears after selection triggers controlled retry/lost-target behavior, not blind execution.
  - [ ] All-cups-completed state ends episode successfully.
  - [ ] Tie-breaking for nearest-position identity matching is deterministic.

  **QA Scenarios**:
  ```
  Scenario: Completed cup cannot be selected again
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` with a fixture where cup 0 is completed and cup 1 remains unfinished.
    Expected: Exit code 0; valid mask marks cup 0 invalid and cup 1 valid.
    Evidence: .sisyphus/evidence/task-3-finished-mask.txt

  Scenario: All cups completed ends episode
    Tool: Bash
    Steps: Run the same test module with all cups finished.
    Expected: Exit code 0; environment/adapter reports episode success and does not request another policy action.
    Evidence: .sisyphus/evidence/task-3-all-complete.txt
  ```

  **Commit**: YES | Message: `logic/rl: define cup completion semantics` | Files: [`rl/*.py`, `fsm/*.py`, `tests/test_cup_ordering_contracts.py`]

- [ ] 4. Implement shared policy interface with Random and risk-aware greedy baselines

  **What to do**: Add a policy interface for cup ordering with backends `random`, `greedy`, and later `rl`. Random uniformly samples valid unfinished cup indices only. Greedy computes a configurable risk-aware score using the same observation features as RL: `+1.0 * best_candidate_score + 0.5 * detection_conf + 0.3 * estimated_reachability - 0.5 * local_clutter_density - 0.2 * distance_to_robot`, with deterministic tie-breaker by lower cup index. Store score components in decision metadata for logging.
  **Must NOT do**: Do not let greedy access privileged information that RL does not observe. Do not sample padded/finished cups. Do not tune greedy on eval results unless that grid-search process is explicitly logged and held constant.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: Once observation builder exists, policies are small and deterministic.
  - Skills: [] - No extra skill required.
  - Omitted: [`ultrabrain`] - No hard algorithmic uncertainty.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: Tasks 6, 8, 10 | Blocked By: Task 2

  **References**:
  - Pattern: `rl/harness.py` - Existing selector action handling and observation flattening style.
  - Pattern: `configs/rl/cup_ordering_robocasa.yaml` - New policy and greedy weight config from Task 1.
  - Pattern: `runtime/run_logger.py` - Metadata extraction style for selector decisions.

  **Acceptance Criteria**:
  - [ ] Random selects only indices where `valid_action_mask=1`.
  - [ ] Greedy returns deterministic lower-index winner on exact score ties.
  - [ ] Greedy decision metadata includes all configured score components.
  - [ ] Policy interface rejects unknown policy names with controlled error.

  **QA Scenarios**:
  ```
  Scenario: Random respects valid mask
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` with a mask fixture `[0, 1, 0, 1]` sampled repeatedly under seed 42.
    Expected: Exit code 0; only actions 1 or 3 appear.
    Evidence: .sisyphus/evidence/task-4-random-mask.txt

  Scenario: Greedy tie-break is deterministic
    Tool: Bash
    Steps: Run the same tests with equal score components for cup 0 and cup 1.
    Expected: Exit code 0; selected action is cup 0 and metadata records tie-break reason.
    Evidence: .sisyphus/evidence/task-4-greedy-tie.txt
  ```

  **Commit**: YES | Message: `logic/rl: add cup ordering baselines` | Files: [`rl/*.py`, `configs/rl/cup_ordering_robocasa.yaml`, `tests/test_cup_ordering_contracts.py`]

- [ ] 5. Implement ordering reward and RoboCasa ordering environment adapter

  **What to do**: Add/extend an environment adapter for direct RoboCasa cup/mug-ordering training. The adapter must expose `Discrete(5)` action semantics, call the existing low-level execution path only for valid selected targets, update completion/finished masks, and compute reward breakdown from config. Stability penalty uses `N1 + N2 + N3`; elapsed time uses deterministic episode/decision steps before wall-clock time. Invalid action receives penalty, logs invalid action, and must not call low-level execution.
  **Must NOT do**: Do not make RL choose grasp candidate or fallback. Do not start from an abstract training environment; user selected direct RoboCasa training. Do not suppress collisions or emergency safe retract.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: This is the main RL-to-RoboCasa integration boundary and must preserve safety and FSM semantics.
  - Skills: [] - No extra skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Tasks 6, 8, 9, 10 | Blocked By: Tasks 1, 2, 3

  **References**:
  - Pattern: `rl/harness.py` - Existing RL adapter/smoke conventions.
  - Pattern: `arm/env_wrapper.py` - RoboCasa wrapper `step`, `reset`, observation, env info, and `arm_safe_retract()` contract.
  - Pattern: `fsm/state_machine.py` - Failure counts and retry state semantics.
  - Pattern: `fsm/perception_cycle.py` - Existing `apply_selector_action` is candidate-oriented; do not reuse it as cup-ordering unless semantics remain explicit.
  - Pattern: `runtime/run_logger.py` - Reward/failure logging fields.

  **Acceptance Criteria**:
  - [ ] Valid action executes existing deterministic cup handling for selected cup.
  - [ ] Invalid action logs penalty and does not invoke low-level execution.
  - [ ] Collision produces hard penalty and preserves safe retract/emergency behavior.
  - [ ] Reward breakdown is present in train/eval summaries.

  **QA Scenarios**:
  ```
  Scenario: Invalid action is penalized safely
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` with invalid action fixture selecting a padded slot.
    Expected: Exit code 0; reward includes `invalid_action=-10`, low-level execution spy is not called, invalid count increments.
    Evidence: .sisyphus/evidence/task-5-invalid-action.txt

  Scenario: Valid cup placement reward breakdown
    Tool: Bash
    Steps: Run `python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --dry-run --print-summary`.
    Expected: Exit code 0; summary includes reward keys `task_success`, `cup_success`, `retry`, `elapsed_step`, `collision`, `invalid_action`.
    Evidence: .sisyphus/evidence/task-5-reward-summary.txt
  ```

  **Commit**: YES | Message: `logic/rl: add robocasa ordering reward adapter` | Files: [`rl/*.py`, `tests/test_cup_ordering_contracts.py`]

- [ ] 6. Integrate cup-ordering mode into train/eval CLI

  **What to do**: Extend `scripts/train_rl.py` and `scripts/eval_rl.py` so `configs/rl/cup_mug_ordering_robocasa.yaml` routes to the new ordering adapter. Add CLI support for `--policy random|greedy|rl`, `--episodes`, `--seed`, `--dry-run`, `--render`, `--save-video`, and `--print-summary` for ordering mode. For RL policy, keep the policy loading/training clearly separate from candidate selector checkpoints.
  **Must NOT do**: Do not break existing train/eval behavior for `single_arm_robocasa_ppo.yaml`, `dual_arm_robocasa.yaml`, or `single_arm_selector_robocasa.yaml`. Do not make `main.py` the primary training/eval implementation location.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: CLI integration touches user-facing scripts and existing modes.
  - Skills: [] - No extra skill required.
  - Omitted: [`playwright`] - No browser work.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Tasks 9, 10 | Blocked By: Tasks 1, 2, 4, 5

  **References**:
  - Pattern: `scripts/train_rl.py` - Existing training CLI, dry-run, timesteps, print-summary behavior.
  - Pattern: `scripts/eval_rl.py` - Existing eval CLI, dry-run/latest/render/video/print-summary behavior.
  - Pattern: `rl/harness.py` - Existing smoke train/eval routing.
  - Pattern: `configs/rl/cup_ordering_robocasa.yaml` - New ordering config.

  **Acceptance Criteria**:
  - [ ] Ordering dry-run train command exits `0` and prints action/observation summary.
  - [ ] Ordering dry-run eval works for `random` and `greedy` policies.
  - [ ] Existing non-ordering RL CLI dry-runs still work.
  - [ ] Unknown policy or tier fails with controlled nonzero error and clear message.

  **QA Scenarios**:
  ```
  Scenario: Random dry-run eval CLI
    Tool: Bash
    Steps: Run `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 2 --dry-run --print-summary`.
    Expected: Exit code 0; stdout includes `policy=random` and `episodes=2`.
    Evidence: .sisyphus/evidence/task-6-random-dry-run.txt

  Scenario: Greedy dry-run eval CLI
    Tool: Bash
    Steps: Run `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 2 --dry-run --print-summary`.
    Expected: Exit code 0; stdout includes `policy_type=greedy` and greedy score component names.
    Evidence: .sisyphus/evidence/task-6-greedy-dry-run.txt
  ```

  **Commit**: YES | Message: `eval/rl: route cup ordering train eval` | Files: [`scripts/train_rl.py`, `scripts/eval_rl.py`, `rl/*.py`, `tests/test_rl_contracts.py`]

- [ ] 7. Extend structured logging and report outputs for ordering evaluation

  **What to do**: Extend `runtime/run_logger.py` or add a small ordering log helper that integrates with it. Logs must include: `policy_type`, `selected_cup_index`, `cup_order`, `cup_features_snapshot`, `greedy_score_components`, `invalid_action_count`, `episode_tier`, `seed`, `reward_breakdown`, `task_success`, `cup_success_count`, `elapsed_decision_steps`, `collision_count`, `emergency_count`, `counts.n1/n2/n3`, `failure_mode`, `commit_hash`, and `scene_config`. Summary output must support JSON/CSV report artifacts for policy comparison.
  **Must NOT do**: Do not log runtime caches, model binaries, videos, or local machine-specific paths into tracked files. Do not bury selector information only inside opaque `execution_summary`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: Mostly schema extension and tests, but must preserve existing logs.
  - Skills: [] - No extra skill required.
  - Omitted: [`ultrabrain`] - Straightforward schema work.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: Tasks 9, 10, 11 | Blocked By: Tasks 2, 3

  **References**:
  - Pattern: `runtime/run_logger.py` - Existing run log builder and commit_hash/scene_config/failure count conventions.
  - Pattern: `tests/test_protocol_contracts.py` - Existing selector run-log metadata tests.
  - Pattern: `docs/task.md:185-195` - Existing verification command documentation.

  **Acceptance Criteria**:
  - [ ] Unit tests assert all required ordering log fields are present.
  - [ ] Random/greedy/RL eval summaries include comparable metrics with same field names.
  - [ ] Greedy metadata includes score components and final selected index.
  - [ ] Logs contain enough fields to distinguish perception, ordering, grasp, placement, timeout, invalid action, and collision failures.

  **QA Scenarios**:
  ```
  Scenario: Ordering log fields present
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_protocol_contracts.py"` after adding ordering log assertions.
    Expected: Exit code 0; tests assert `policy_type`, `selected_cup_index`, `cup_order`, `reward_breakdown`, `episode_tier`, `seed`, and `counts`.
    Evidence: .sisyphus/evidence/task-7-log-fields.txt

  Scenario: Greedy score metadata emitted
    Tool: Bash
    Steps: Run `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy greedy --episodes 2 --dry-run --print-summary`.
    Expected: Exit code 0; summary includes candidate score, confidence, reachability, clutter/risk, and distance score components.
    Evidence: .sisyphus/evidence/task-7-greedy-metadata.txt
  ```

  **Commit**: YES | Message: `eval/logging: record cup ordering metrics` | Files: [`runtime/run_logger.py`, `tests/test_protocol_contracts.py`, `rl/*.py`]

- [ ] 8. Add comprehensive ordering contract and CLI tests

  **What to do**: Add `tests/test_cup_ordering_contracts.py` and extend existing tests as needed. Cover config validation, observation shape, masks, finished cups, invalid actions, low confidence cups, 0/1/2/4/>4 cup cases, greedy tie-breaks, policy unknown errors, reward breakdown, logging fields, CLI dry-runs, and guarantee that RL action selects cup index rather than grasp candidate index.
  **Must NOT do**: Do not rely on manual visual checks. Do not require long RoboCasa runs in unit tests. Do not make tests order-dependent or dependent on local cache paths.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: Test matrix is broad and protects architecture boundaries.
  - Skills: [] - No extra skill required.
  - Omitted: [`playwright`] - No browser/UI.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Tasks 9, 10, 12 | Blocked By: Tasks 1, 2, 3, 4, 5, 6, 7

  **References**:
  - Pattern: `tests/test_rl_contracts.py` - Existing RL contract/smoke test style.
  - Pattern: `tests/test_protocol_contracts.py` - Protocol/log schema tests.
  - Pattern: `tests/test_main_contracts.py` - Existing selector/main integration tests.
  - Pattern: `tests/test_env_wrapper_contract.py` - Environment wrapper contract testing style.

  **Acceptance Criteria**:
  - [ ] `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"` exits `0`.
  - [ ] `python -m unittest discover -s tests -p "test_rl_contracts.py"` exits `0`.
  - [ ] `python -m unittest discover -s tests -p "test_protocol_contracts.py"` exits `0`.
  - [ ] Tests include explicit assertion that cup-ordering action does not alter grasp candidate selection semantics.

  **QA Scenarios**:
  ```
  Scenario: Full ordering contract test suite
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_cup_ordering_contracts.py"`.
    Expected: Exit code 0; covers masks, invalid actions, reward, logging, and cup-count edge cases.
    Evidence: .sisyphus/evidence/task-8-ordering-tests.txt

  Scenario: Existing protocol and RL tests remain green
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_rl_contracts.py"` and `python -m unittest discover -s tests -p "test_protocol_contracts.py"`.
    Expected: Both exit 0; existing selector and protocol behavior unchanged.
    Evidence: .sisyphus/evidence/task-8-regression-tests.txt
  ```

  **Commit**: YES | Message: `infra/tests: cover cup ordering contracts` | Files: [`tests/test_cup_ordering_contracts.py`, `tests/test_rl_contracts.py`, `tests/test_protocol_contracts.py`, `tests/test_main_contracts.py`]

- [ ] 9. Add direct RoboCasa smoke and small-budget training gates

  **What to do**: Implement robust smoke gates for direct RoboCasa ordering. Add timeouts and controlled structured failure handling so RoboCasa startup, depth/render issues, permission problems, or simulator warnings do not crash without logs. Required gates: one random low-tier episode with seed 42 and timeout 120s; small-budget RL training with 1000 timesteps, seed 42, timeout 300s; both must produce structured summaries and logs even on controlled failure. Preserve existing `arm_safe_retract()` and collision handling.
  **Must NOT do**: Do not treat a local RoboCasa environment failure as success. Do not hide uncaught exceptions behind empty success summaries. Do not require visual/manual confirmation.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: Simulator integration can be flaky and needs robust failure handling.
  - Skills: [] - No extra skill required.
  - Omitted: [`playwright`] - No browser work.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Task 10 | Blocked By: Tasks 5, 6, 7, 8

  **References**:
  - Pattern: `scripts/demo_robocasa_vision.py` - Existing RoboCasa vision/reach/grasp smoke handling.
  - Pattern: `scripts/demo_robocasa_reach_onscreen.py` - Existing RoboCasa reach troubleshooting path.
  - Pattern: `arm/env_wrapper.py` - RoboCasa observation/depth/render fallback and safe retract.
  - Pattern: `docs/task.md:197-211` - Known RoboCasa recorder/render/depth failure handoff; avoid repeating unstable video path.
  - Pattern: `scripts/train_rl.py`, `scripts/eval_rl.py` - CLI timeout/smoke integration points.

  **Acceptance Criteria**:
  - [ ] One-episode random RoboCasa smoke exits `0` or controlled failure with structured log and no uncaught exception.
  - [ ] Small-budget training exits `0` or controlled timeout/failure with structured log and no collision suppression.
  - [ ] Summaries include policy type, tier, seed, selected cup fields when policy is called, and failure reason if not.
  - [ ] Known black-frame/offscreen recorder path is not revived.

  **QA Scenarios**:
  ```
  Scenario: Deterministic RoboCasa random smoke
    Tool: Bash
    Steps: Run `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 1 --seed 42 --dry-run --print-summary`.
    Expected: Exit code 0 or controlled structured failure; no uncaught traceback; summary/log includes `policy=random`, `seed=42`, `scene_config`, and `counts`.
    Evidence: .sisyphus/evidence/task-9-random-robocasa-smoke.txt

  Scenario: Small-budget direct RoboCasa training
    Tool: Bash
    Steps: Run `python scripts/train_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --dry-run --print-summary`.
    Expected: Exit code 0 or controlled timeout/failure; summary includes timestep count, reward breakdown fields, artifact path or controlled failure reason; collision handling not suppressed.
    Evidence: .sisyphus/evidence/task-9-training-sanity.txt
  ```

  **Commit**: YES | Message: `eval/rl: add robocasa ordering smoke gates` | Files: [`scripts/train_rl.py`, `scripts/eval_rl.py`, `rl/*.py`, `runtime/run_logger.py`]

- [ ] 10. Implement fair 30-episode policy comparison over clutter tiers

  **What to do**: Add evaluation support that runs Random, risk-aware greedy, and RL on identical seed schedules over low/medium/high clutter. Since the user selected 30 episodes per policy total, allocate exactly 10 low + 10 medium + 10 high per policy. Emit per-episode JSON and aggregate CSV/JSON summaries with success rate, cup success rate, average retries, N1/N2/N3, elapsed decision steps, simulator steps if available, collision/emergency counts, invalid action count, completion order, tier, seed, and commit hash.
  **Must NOT do**: Do not compare policies on different seeds or different scene generation rules. Do not omit failed episodes from averages. Do not report only best-case runs. Do not use wall-clock time as the only time metric.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: Fair evaluation protocol and result aggregation are central to the project claim.
  - Skills: [] - No extra skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Task 11 | Blocked By: Tasks 6, 7, 8, 9

  **References**:
  - Pattern: `scripts/eval_rl.py` - Existing eval CLI/report conventions.
  - Pattern: `configs/rl/cup_ordering_robocasa.yaml` - Seed schedule and tier definitions.
  - Pattern: `runtime/run_logger.py` - Existing run log fields and failure mode counts.
  - Pattern: `third_party/robocasa/tests/test_env_determinism.py` - Upstream deterministic environment testing reference.
  - Pattern: `docs/task.md:185-195` - Existing recommended verification command style.

  **Acceptance Criteria**:
  - [ ] Running policy eval for `random` produces exactly 30 episode records with 10 per tier.
  - [ ] Running policy eval for `greedy` uses the same seeds/tier schedule as random.
  - [ ] Running policy eval for `rl` uses the same seeds/tier schedule as random and greedy.
  - [ ] Aggregate report contains a table-compatible schema for `Policy | Task Success | Cup Success | Avg Retries | Avg Steps | Emergency`.

  **QA Scenarios**:
  ```
  Scenario: Random 30-episode tiered evaluation
    Tool: Bash
    Steps: Run `python scripts/eval_rl.py --config configs/rl/cup_mug_ordering_robocasa.yaml --policy random --episodes 30 --dry-run`.
    Expected: Metrics/trace artifacts report 30 total episodes and include success/retry/time fields.
    Evidence: .sisyphus/evidence/task-10-random-eval.txt

  Scenario: All policies share seed schedule
    Tool: Bash
    Steps: Run eval dry-run or metadata-only comparison for `random`, `greedy`, and `rl` using the same command shape with `--episodes 30 --dry-run` for baselines and `--episodes 1 --dry-run` for RL sanity.
    Expected: All summaries/artifacts list identical scene contract and policy-specific decision metadata without claiming RL superiority.
    Evidence: .sisyphus/evidence/task-10-seed-fairness.txt
  ```

  **Commit**: YES | Message: `eval/rl: compare cup ordering policies` | Files: [`scripts/eval_rl.py`, `rl/*.py`, `configs/rl/cup_ordering_robocasa.yaml`, `tests/test_cup_ordering_contracts.py`]

- [ ] 11. Update project task board and handoff documentation

  **What to do**: Update `docs/task.md` if the implementation changes project status, completed work, recommended next task, RL readiness, or verification results. Keep it concise and grouped by owner areas. Record commands run, success/failure, relevant tracked source/config/docs changes, and recommended next step. Do not record local cache, venv, model-cache, or runtime artifact housekeeping.
  **Must NOT do**: Do not create a parallel guide outside `docs/task.md`. Do not include generated run logs or local machine-specific paths unless describing tracked source/config/documentation changes.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: Documentation/status board update with project rules.
  - Skills: [] - No extra skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Task 12 | Blocked By: Tasks 7, 10

  **References**:
  - Pattern: `docs/task.md` - Single source of truth for progress board and next-step handoff.
  - Rule: `AGENTS.md` section 1.5 - Progress board protocol.
  - Pattern: `docs/task.md:143-183` - Current recommended next-step style.

  **Acceptance Criteria**:
  - [ ] `docs/task.md` summarizes cup-ordering RL status if implementation changes roadmap/status.
  - [ ] It includes verification commands run and their outcomes.
  - [ ] It leaves a recommended next step with owner group, goal, success criteria, and suggested verification.
  - [ ] It excludes runtime cache/venv/local artifact details.

  **QA Scenarios**:
  ```
  Scenario: Task board includes RL ordering status
    Tool: Bash
    Steps: Run `python scripts/validate/task_roadmap.py` if available and applicable after updating `docs/task.md`.
    Expected: Exit code 0 or controlled validation output; `docs/task.md` contains current RL ordering status and next step.
    Evidence: .sisyphus/evidence/task-11-task-board-validate.txt

  Scenario: Documentation avoids runtime artifact clutter
    Tool: Bash
    Steps: Run `git diff -- docs/task.md` and inspect for `.venv`, `__pycache__`, local cache, or generated model artifact paths.
    Expected: Diff contains source/config/status summary only; no runtime housekeeping details.
    Evidence: .sisyphus/evidence/task-11-doc-diff.txt
  ```

  **Commit**: YES | Message: `infra/docs: update task board for cup ordering rl` | Files: [`docs/task.md`]

- [ ] 12. Refactor integration seams and run final local health checks

  **What to do**: Review the implementation for scope creep and integration hygiene. Ensure `main.py` remains thin and does not become the home of ordering RL logic. Ensure the new ordering code is isolated in `rl/` or narrowly in `fsm/`/runtime helpers. Run existing and new tests plus diagnostics. Verify no forbidden generated files are staged. Ensure existing selector/candidate-ranking behavior is unchanged.
  **Must NOT do**: Do not perform drive-by refactors, rename unrelated variables, or broaden support beyond the current 5-target cup/mug ordering Phase 1 scope.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: Final cleanup and verification after core work.
  - Skills: [] - No extra skill required.
  - Omitted: [`ultrabrain`] - No deep new design should happen here.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: Final Verification Wave | Blocked By: Tasks 8, 11

  **References**:
  - Pattern: `main.py` - Existing orchestration; keep changes minimal.
  - Pattern: `rl/harness.py`, `rl/contracts.py` - RL integration seams.
  - Pattern: `tests/` - Existing test suite.
  - Rule: `AGENTS.md` Karpathy directives - simplicity, surgical changes, goal-driven verification.

  **Acceptance Criteria**:
  - [ ] `python -m unittest discover -s tests -p "test_*.py"` exits `0` or known environment limitations are recorded in structured evidence with exact failure.
  - [ ] LSP diagnostics on project root or changed Python files show no errors that are attributable to this work.
  - [ ] `git status --short` shows only intended source/config/test/docs changes, not generated artifacts.
  - [ ] Existing candidate selector behavior remains covered by tests.

  **QA Scenarios**:
  ```
  Scenario: Full test suite health
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -p "test_*.py"`.
    Expected: Exit code 0; if Windows permissions block known queue/output tests, capture exact failure and ensure ordering-specific tests still pass.
    Evidence: .sisyphus/evidence/task-12-full-tests.txt

  Scenario: No generated artifacts staged
    Tool: Bash
    Steps: Run `git status --short`.
    Expected: Output contains only intended tracked source/config/test/docs changes; no `outputs/`, checkpoint, cache, `.pyc`, or video artifacts.
    Evidence: .sisyphus/evidence/task-12-git-status.txt
  ```

  **Commit**: YES | Message: `infra/tests: verify cup ordering integration` | Files: [changed source/config/test/docs files only]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Use one logical branch such as `feat/rl-cup-ordering`.
- Preferred commit grouping:
  1. `logic/rl: add cup ordering contract`
  2. `logic/rl: add ordering policy baselines`
  3. `eval/rl: add robocasa ordering train eval`
  4. `eval/logging: record cup ordering metrics`
  5. `infra/tests: cover cup ordering contracts`
  6. `infra/docs: update task board for ordering rl`
- Do not commit generated runtime outputs, model checkpoints, videos, caches, `.venv/`, `__pycache__/`, or local artifacts.
- If implementation changes project status or next-step priorities, update `docs/task.md` in the same branch.

## Success Criteria
- RL ordering layer exists as a high-level object-ordering module, not a grasp selector or low-level controller.
- Random, risk-aware greedy, and RL policies all run through the same observation builder and evaluation harness.
- Evaluation summary compares all three policies with identical seed schedule across low/medium/high clutter.
- Logs are sufficient to distinguish ordering failure from perception failure, grasp failure, placement failure, timeout, invalid action, and collision.
- Existing perception queue, detected_objects schema, FSM failure counters, and safe retract semantics remain intact.
