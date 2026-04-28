# RoboCasa Migration + Dual-Arm RL Roadmap

## TL;DR
> **Summary**: Migrate the current robosuite-first manipulation stack to a RoboCasa-backed execution environment without changing the frozen `vision/`, `ipc/`, `fsm/`, or `logs/` protocols, and make the first milestone a directly trainable dual-arm RL baseline for approach-and-grasp using privileged environment observations plus a minimal training/evaluation harness.
> **Deliverables**:
> - RoboCasa-compatible environment adapter behind the existing wrapper boundary
> - Frozen protocol contract checks for `vision/`, `ipc/`, `fsm/`, `logs/`
> - Dual-arm RL task definition for approach-and-grasp baseline
> - Minimal train/eval/checkpoint harness with reproducible metrics
> - Phase-gated extension path for obstacle clearing, semantic vision reintegration, and final cluttered collaboration
> **Effort**: XL
> **Parallel**: YES - 2 waves
> **Critical Path**: 1 → 2 → 3 → 4 → 5 → 8 → 9 → 10

## Context
### Original Request
- 整体阅读当前完整项目进度。
- 开始向 RoboCasa 环境迁移，并添加 RL。
- 先冻结视觉和 FSM。
- 第一阶段先完成基础接近目标并抓取。
- 第二阶段增加遮挡障碍，需要先抓取清理障碍，再抓取目标。
- 第三阶段由视觉模型确定障碍和目标。
- 最终实现复杂环境中的双臂协作清障并抓取目标。

### Interview Summary
- 当前仓库权威进度来源是 `docs/task.md`，当前运行基线是 `robosuite`，而不是 RoboCasa。
- 用户确认第一阶段采用“先环境后视觉”：先在 RoboCasa 中建立 baseline，不依赖真实视觉闭环。
- 用户确认冻结范围是协议层：冻结 `vision/`、`ipc/`、`fsm/`、`logs/` 的协议，允许 `arm/` 与环境适配层为迁移做必要修改。
- 用户确认 RL 不后置，而是第一阶段就进入主线；并进一步确认第一阶段 baseline 本身就是 RL 主驱动。
- 用户确认第一阶段直接做双臂 RL，而不是先单臂；同时把最小训练/评估框架纳入第一版计划。

### Metis Review (gaps addressed)
- 已补足的关键 guardrails：必须先定义第一阶段 RoboCasa 双臂 RL 任务契约、训练/评估闭环、冻结协议契约，否则后续障碍清理和视觉回接会漂移。
- 已纳入的风险限制：第一阶段禁止把障碍清理、真实视觉、最终复杂协作混入同一交付；`arm/env_wrapper.py` 维持为唯一环境交换边界。
- 已纳入的验收要求：所有阶段都必须提供 agent-executable 命令、失败路径、checkpoint/metrics/log 证据，不允许“人工看起来可以”式验收。
- 已记录的反向证据：`docs/task.md` 当前官方建议 RL 应后置且先做局部子任务；本计划按用户要求提前 RL，但通过 privileged-state baseline、严格 phase gate、稳定日志契约来控制风险。

## Work Objectives
### Core Objective
在不破坏 `vision/`、`ipc/`、`fsm/`、`logs/` 现有协议的前提下，把当前项目的环境层从 `robosuite` 抽象为可切换的 RoboCasa 后端，并建立一个可训练、可评估、可复现的 **RoboCasa 双臂 RL 接近+抓取 baseline**。后续所有障碍清理、视觉语义重接和复杂协作都建立在该 baseline 的可量化结果之上。

### Deliverables
- `arm/env_wrapper.py` 重构为后端无关入口，并新增 RoboCasa 后端适配实现
- 冻结协议契约文档化与自动化校验：`detected_objects`、`perception_queue`、FSM 状态/重试、run log schema
- 双臂 RL baseline 任务规范：观测、动作、奖励、终止、成功定义、种子策略
- 训练 CLI、评估 CLI、checkpoint、metrics、failure logs、对比脚本
- `docs/task.md` 更新为 RoboCasa + RL 路线后的新阶段基线与 recommended next step

### Definition of Done (verifiable conditions with commands)
- RoboCasa backend smoke test passes with deterministic reset/step/render metadata checks.
- Frozen protocol contract tests pass without changing required payload keys or timeout semantics.
- Dual-arm RL baseline can train for a short smoke budget, save a checkpoint, and run evaluation from that checkpoint.
- Evaluation writes reproducible metrics and run logs with fixed seed set.
- Phase-2/3/4 follow-up stubs are specified in docs/task.md without being prematurely implemented in milestone 1.

### Must Have
- Freeze-by-contract for `vision/`, `ipc/`, `fsm/`, `logs/`
- Single migration boundary centered on `arm/env_wrapper.py`
- Milestone-1 observation source is privileged environment state, not real vision
- Milestone-1 task is direct dual-arm RL approach-and-grasp only
- Minimal but real train/eval/checkpoint/logging harness
- Explicit failure-mode accounting compatible with existing `logs/run_logger.py`

### Must NOT Have
- Must NOT redesign `detected_objects` schema defined in `vision/detected_objects.py:15-55`
- Must NOT rename or replace `perception_queue` semantics defined in `ipc/perception_queue.py:6-24`
- Must NOT change FSM state names or retry categories defined in `fsm/state_machine.py:6-106`
- Must NOT break run log top-level fields defined in `logs/run_logger.py:54-95`
- Must NOT reintroduce real vision into milestone 1
- Must NOT fold obstacle clearing into milestone 1
- Must NOT replace high-level FSM policy with RL in milestone 1
- Must NOT broad-refactor `main.py` beyond extracting orchestration needed by the new adapter/harness boundaries

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after + existing script-style validation upgraded with targeted automated checks
- QA policy: Every task includes a happy-path and a failure/edge-path scenario
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: environment contract + protocol freeze + RL task specification + logging/eval foundations (Tasks 1-5)

Wave 2: RoboCasa adapter implementation + training/eval harness + orchestration integration + roadmap documentation (Tasks 6-10)

### Dependency Matrix (full, all tasks)
- 1 blocks 2, 6, 8
- 2 blocks 6, 7, 8
- 3 blocks 7, 8
- 4 blocks 7, 9
- 5 blocks 8, 9
- 6 blocks 8, 9
- 7 blocks 8, 9
- 8 blocks 9, 10
- 9 blocks 10
- 10 blocks Final Verification Wave

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 5 tasks → deep / ultrabrain / unspecified-high / quick
- Wave 2 → 5 tasks → deep / ultrabrain / unspecified-high / writing

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Freeze protocol contracts before any migration work

  **What to do**: Define and enforce milestone-1 compatibility contracts for `vision/`, `ipc/`, `fsm/`, and `logs/` before adding any RoboCasa or RL code. Convert the current implicit contracts into explicit assertions / schema tests anchored to the existing implementations. Preserve the `detected_objects` payload structure, `perception_queue` name + timeout semantics, FSM state names + retry counters, and run log top-level schema.
  **Must NOT do**: Do not introduce “temporary” protocol key changes; do not rename FSM states; do not widen queue semantics beyond the current `Queue.get(timeout=0.2)` contract.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: multi-file contract work across protocol boundaries with test assertions and no architecture redesign.
  - Skills: `[]` - no extra skill required.
  - Omitted: `review-work` - not needed during implementation planning phase.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 6, 8 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\vision\detected_objects.py:15-55` - canonical `detected_objects` builder with `target/obstacles/status` schema and `time.monotonic()` timestamp.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\vision\detected_objects.py:58-82` - existing validator logic; extend this rather than inventing a new payload validator style.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\ipc\perception_queue.py:6-24` - fixed `PERCEPTION_QUEUE_NAME`, timeout constant, publish/read semantics.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\fsm\state_machine.py:6-106` - required state enum and `n1/n2/n3` retry accounting.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\logs\run_logger.py:54-95` - required run-log top-level schema and failure-mode fields.
  - Test: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:423-425` - project already calls for minimal tests covering `detected_objects`, FSM transitions, and run log behavior.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A contract test command exits 0 and verifies the required `detected_objects` keys, obstacle cap, and timestamp source behavior.
  - [ ] A contract test command exits 0 and verifies `PERCEPTION_QUEUE_NAME == "perception_queue"` and stale-read timeout behavior remains compatible.
  - [ ] A contract test command exits 0 and verifies all FSM state enum values and `n1/n2/n3` counters remain unchanged.
  - [ ] A contract test command exits 0 and verifies the run log still emits `run_id`, `commit_hash`, `config_version`, `scene_config`, `timestamp`, `success`, `failure_mode`, `failure_modes_triggered`, `counts`, `duration_sec`, `context`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Protocol contract happy path
    Tool: Bash
    Steps: Run the protocol contract test command for schema, queue, FSM, and run-log compatibility.
    Expected: Exit code 0; generated evidence records all checks as pass.
    Evidence: .sisyphus/evidence/task-1-freeze-protocol-contracts.txt

  Scenario: Invalid payload failure path
    Tool: Bash
    Steps: Run the contract tests with a deliberately malformed detected_objects fixture or invalid run-log counts fixture.
    Expected: Non-zero exit or explicit assertion failure naming the incompatible field.
    Evidence: .sisyphus/evidence/task-1-freeze-protocol-contracts-error.txt
  ```

  **Commit**: YES | Message: `infra/contracts: freeze protocol layer interfaces` | Files: `vision/*`, `ipc/*`, `fsm/*`, `logs/*`, test files, `docs/task.md`

- [ ] 2. Define the backend-agnostic environment wrapper contract

  **What to do**: Refactor the current single-class robosuite wrapper shape into an explicit backend-agnostic contract that can support both robosuite and RoboCasa without leaking backend-specific semantics upward. Document and implement the exact methods milestone 1 depends on: reset, step, observation retrieval, camera intrinsics/extrinsics access, object/task metadata access, safe retract behavior, and deterministic seed/reset entrypoints for evaluation.
  **Must NOT do**: Do not duplicate environment-selection logic inside controllers, FSM, or training harnesses; do not expose backend-specific raw simulator objects outside the adapter layer unless hidden behind an internal implementation-only module.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this is the main migration boundary and mistakes here cause architecture bleed.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - too risky for core architecture boundary work.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 6, 7, 8 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\env_wrapper.py:7-196` - current wrapper surface and behavior; preserve externally useful methods while generalizing backend internals.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:122-136` - current perception refresh path relies on `get_observation()` returning `(rgb, depth, proprioception)`.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:249-260` - current execution path expects environment access to observation-backed grasp target selection.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\controller.py:170-258` - controllers assume `env.obs`, `env.action_dim`, and robot-indexed observation keys.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:158-167` - environment layer must become the sole swap boundary for `robosuite -> RoboCasa` migration.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A backend contract smoke test exits 0 for the robosuite backend and verifies reset, step, action_dim, observation access, and safe retract methods.
  - [ ] The wrapper interface is documented in code/docstrings or a dedicated developer-facing module so milestone-1 RL code can instantiate by backend name without importing backend-specific modules directly.
  - [ ] A deterministic reset/seed smoke test exits 0 and records whether the wrapper can produce repeatable initial states for evaluation.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Wrapper contract happy path
    Tool: Bash
    Steps: Run the wrapper smoke test against the existing robosuite backend using a fixed seed and one reset-step cycle.
    Expected: Exit code 0; evidence includes action_dim, observation keys, and reset determinism summary.
    Evidence: .sisyphus/evidence/task-2-wrapper-contract.txt

  Scenario: Unsupported backend failure path
    Tool: Bash
    Steps: Run the same wrapper smoke test with an unknown backend identifier.
    Expected: Non-zero exit with a clear unsupported-backend error message.
    Evidence: .sisyphus/evidence/task-2-wrapper-contract-error.txt
  ```

  **Commit**: YES | Message: `execution/env: define backend-agnostic wrapper contract` | Files: `arm/env_wrapper.py`, new adapter modules/tests, `docs/task.md`

- [ ] 3. Specify the milestone-1 dual-arm RoboCasa RL task contract

  **What to do**: Write the exact task specification for milestone 1 as a direct dual-arm RoboCasa RL baseline: target object class/domain, reset distribution, privileged observation set, action parameterization, reward terms, termination conditions, collision handling, success criteria, episode length, seed policy, and output metrics. This specification must deliberately exclude obstacle clearing and semantic vision. Decide whether the baseline uses one shared policy, two policies, or centralized training with shared action head; make the choice explicit in code and docs instead of leaving it to the implementer.
  **Must NOT do**: Do not leave reward design, observation fields, or success thresholds as TODO comments; do not make milestone 1 depend on camera calibration or GroundingDINO outputs.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` - Reason: RL task formulation is the highest-leverage decision point and must be decision-complete.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - unsuitable for reward/observation/action design.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7, 8 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:263-345` - existing repository guidance on RL role, observation/action/reward patterns, and success criteria; adapt deliberately because user requested more aggressive RL timing.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\controller.py:6-71` - current tuned constants reveal what the hand-written controller currently optimizes around: approach heights, descent offsets, tolerances, gripper thresholds.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:44-61` - stage taxonomy already distinguishes drift-like vs slip-like failure stages and should inform RL metric reporting.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\logs\run_logger.py:21-51` - failure-mode vocabulary must remain compatible with existing logs.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A checked-in milestone-1 RL config/spec file exists and explicitly enumerates observation fields, action dimensions, reward coefficients, termination rules, and metrics.
  - [ ] A validation command exits 0 and proves the RL config/spec can be parsed into a task object without hidden defaults.
  - [ ] A negative test exits non-zero when a required field such as reward coefficients or episode horizon is omitted.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: RL task spec happy path
    Tool: Bash
    Steps: Run the task-spec validation command against the milestone-1 dual-arm RoboCasa RL config.
    Expected: Exit code 0; evidence prints resolved observation keys, action dimension, reward terms, and horizon.
    Evidence: .sisyphus/evidence/task-3-rl-task-contract.txt

  Scenario: Missing field failure path
    Tool: Bash
    Steps: Run the same validation command against a malformed config missing a required reward or termination field.
    Expected: Non-zero exit with explicit missing-field error.
    Evidence: .sisyphus/evidence/task-3-rl-task-contract-error.txt
  ```

  **Commit**: YES | Message: `logic/rl: define dual-arm robocasa baseline task contract` | Files: RL config/spec files, validation utilities, `docs/task.md`

- [ ] 4. Standardize RL-compatible logging and evaluation schema

  **What to do**: Extend the existing logging path so RL training/evaluation runs can emit metrics without breaking the current run-log schema. Keep the top-level fields stable, and add RL-specific diagnostics in nested fields only: seed, episode return, success rate, checkpoint path, collision count, timeout count, grasp-stage diagnostics, and evaluation cohort metadata. Ensure the schema supports later A/B comparison against heuristic baselines.
  **Must NOT do**: Do not replace the current `build_run_log()` field names; do not log ephemeral local-machine state that violates the repository’s documentation policy.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: cross-cutting metrics/logging work with schema-compatibility constraints.
  - Skills: `[]` - no extra skill required.
  - Omitted: `writing` - code/schema work dominates over prose.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7, 9 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\logs\run_logger.py:54-95` - top-level run-log schema must stay stable.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:194-236` - current execution summary and failure-count conventions are the bridge to RL-compatible diagnostics.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\AGENTS.md:97-110` - `docs/task.md` and tracked logs/docs should reflect git-relevant progress only.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:241-249` - benchmark comparability and report traceability are required end goals.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A logging schema test exits 0 and verifies old run logs still validate while new RL fields appear only in nested structures.
  - [ ] An evaluation dry-run command exits 0 and writes a metrics artifact containing at least `success_rate`, `episode_return`, `episode_length`, `seed`, and `failure_modes_triggered`.
  - [ ] A malformed logging input test exits non-zero with a clear schema or missing-field error.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: RL logging happy path
    Tool: Bash
    Steps: Run the evaluation dry-run command for one fixed seed and inspect the produced metrics artifact.
    Expected: Exit code 0; output contains legacy run-log fields plus nested RL diagnostics.
    Evidence: .sisyphus/evidence/task-4-rl-logging.txt

  Scenario: RL logging failure path
    Tool: Bash
    Steps: Run the logging validation command with a malformed metrics payload missing required top-level fields.
    Expected: Non-zero exit with explicit validation error.
    Evidence: .sisyphus/evidence/task-4-rl-logging-error.txt
  ```

  **Commit**: YES | Message: `eval/logs: add rl-compatible metrics without schema drift` | Files: `logs/*`, eval utilities/tests, `docs/task.md`

- [ ] 5. Define the minimal training and evaluation harness contract

  **What to do**: Create the CLI-level contract for milestone-1 training and evaluation so the project gains a reproducible RL baseline rather than ad hoc scripts. Define the exact commands, config loading rules, checkpoint paths, resume semantics, fixed seed-set evaluation behavior, and failure exit codes. The harness must work even before real vision is reintroduced and must be designed for later obstacle-clearing and semantic-vision extensions without changing milestone-1 CLI semantics.
  **Must NOT do**: Do not hide critical defaults inside notebook-only or script-local variables; do not make evaluation depend on training-side mutable global state.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this determines experiment reproducibility and future extension compatibility.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - too much cross-cutting reproducibility risk.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 8, 9 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\check_env.py`, `D:\pyCode\robot\final\robotic-final-proj\check_action.py`, `D:\pyCode\robot\final\robotic-final-proj\test_move.py`, `D:\pyCode\robot\final\robotic-final-proj\test_render.py` - current repo uses script-style validation; harness should preserve CLI simplicity while becoming reproducible.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:467-470` - baseline engineering acceptance already expects reproducible startup, log writing, and minimal test execution.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\logs\run_logger.py:89-95` - existing log writing style should inform output-path conventions.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A training smoke command exits 0 for a tiny fixed-step budget and writes a checkpoint file to a deterministic path.
  - [ ] An evaluation command exits 0 when pointed at that checkpoint and writes metrics for a fixed seed set.
  - [ ] A resume or invalid-checkpoint command either resumes correctly or exits non-zero with a clear checkpoint-not-found error.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Harness happy path
    Tool: Bash
    Steps: Run a tiny training smoke command, then run the paired evaluation command against the produced checkpoint.
    Expected: Both commands exit 0; checkpoint and metrics artifacts exist in deterministic locations.
    Evidence: .sisyphus/evidence/task-5-harness-contract.txt

  Scenario: Missing checkpoint failure path
    Tool: Bash
    Steps: Run the evaluation command with a nonexistent checkpoint path.
    Expected: Non-zero exit with a clear checkpoint-not-found message.
    Evidence: .sisyphus/evidence/task-5-harness-contract-error.txt
  ```

  **Commit**: YES | Message: `infra/rl: define reproducible train eval harness contract` | Files: train/eval entrypoints, configs, tests, `docs/task.md`

- [ ] 6. Implement the RoboCasa backend adapter behind the wrapper contract

  **What to do**: Add a RoboCasa-backed adapter that satisfies the backend-agnostic wrapper contract from Task 2, while preserving the observation shape milestones 1 and later frozen interfaces rely on. Map RoboCasa reset/step/task metadata/camera access into the contract, expose deterministic seeding for evaluation, and keep any backend-specific imports/config contained inside adapter modules. If RoboCasa cannot natively provide a field that robosuite provided, define a compatibility shim inside the adapter rather than leaking differences upward.
  **Must NOT do**: Do not rewrite controller/FSM code to import RoboCasa directly; do not silently degrade missing observation fields without explicit compatibility handling.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: environment migration is the primary technical boundary and likely source of hidden integration failures.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - unsuitable for simulator-adapter integration.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8, 9 | Blocked By: 1, 2

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\env_wrapper.py:10-35` - current constructor shape and initial environment bootstrapping behavior.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\env_wrapper.py:50-115` - required observation + camera access shape for later compatibility.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\env_wrapper.py:169-192` - safe retract and reset lifecycle expectations.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\README.md:9-10` - migration must prioritize interface reuse and keep frozen protocols stable.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\AGENTS.md:9-10` - environment wrapper is the main swap boundary during migration.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A RoboCasa adapter smoke test exits 0 and verifies backend selection, reset, step, observation access, and deterministic seed support.
  - [ ] A compatibility test exits 0 and verifies the adapter returns the wrapper contract shape expected by milestone-1 RL code.
  - [ ] An invalid RoboCasa config test exits non-zero with a clear backend initialization or scene-creation error.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: RoboCasa adapter happy path
    Tool: Bash
    Steps: Run the backend smoke test against the RoboCasa backend for one fixed-seed reset-step cycle.
    Expected: Exit code 0; evidence records resolved backend name, action_dim, key observation fields, and seed behavior.
    Evidence: .sisyphus/evidence/task-6-robocasa-adapter.txt

  Scenario: RoboCasa config failure path
    Tool: Bash
    Steps: Run the backend smoke test with an invalid RoboCasa scene/task config.
    Expected: Non-zero exit with clear initialization failure details.
    Evidence: .sisyphus/evidence/task-6-robocasa-adapter-error.txt
  ```

  **Commit**: YES | Message: `execution/env: add robocasa backend adapter` | Files: `arm/*`, adapter tests/configs, `docs/task.md`

- [ ] 7. Build the direct dual-arm RL baseline package for RoboCasa

  **What to do**: Implement the milestone-1 RL environment/task package using the task contract from Task 3 and the backend wrapper from Task 6. The baseline must use privileged environment observations, direct dual-arm control, explicit reward shaping, deterministic evaluation seeds, and run through the CLI contract from Task 5. Choose and codify the exact policy/control topology required by the plan—e.g. one shared dual-arm policy with a single action head unless a stronger alternative is justified in the code and docs. The baseline package must be measurable before any real-vision reintegration.
  **Must NOT do**: Do not route policy observations through `detected_objects`; do not make obstacle logic or semantic label discrimination part of the milestone-1 action/reward path.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` - Reason: this combines RL formulation, dual-arm control assumptions, and integration with simulator contracts.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - unsuitable for a high-coupling RL baseline.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8, 9 | Blocked By: 3, 4

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\arm\controller.py:6-71` - current heuristic constants define the spatial scales, tolerances, and gripper thresholds that the RL baseline must at least understand diagnostically.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:149-175` - existing failure classification should inform RL evaluation labeling.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:286-345` - existing RL observation/action/reward guidance provides repository-consistent vocabulary even though user requested an earlier RL phase.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\configs\vision.yaml:1-29` - real-vision config exists but milestone 1 must remain independent from it.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A minimal training smoke run exits 0 and produces a checkpoint for the dual-arm RoboCasa baseline.
  - [ ] A deterministic evaluation run exits 0 on a fixed seed set and emits reproducible metrics artifacts.
  - [ ] A task diagnostic command exits 0 and prints the resolved observation dimension, action dimension, reward components, and episode horizon.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: RL baseline happy path
    Tool: Bash
    Steps: Run the tiny train command for the dual-arm RoboCasa baseline, then evaluate the produced checkpoint over the fixed seed set.
    Expected: Exit code 0 for both; checkpoint and metrics artifacts are created, and the metrics include success rate and failure-mode summaries.
    Evidence: .sisyphus/evidence/task-7-dual-arm-rl-baseline.txt

  Scenario: Invalid RL config failure path
    Tool: Bash
    Steps: Run the training command with an invalid observation or reward config.
    Expected: Non-zero exit with an explicit config validation error before training begins.
    Evidence: .sisyphus/evidence/task-7-dual-arm-rl-baseline-error.txt
  ```

  **Commit**: YES | Message: `logic/rl: add direct dual-arm robocasa baseline` | Files: RL env/task/policy package, configs, tests, `docs/task.md`

- [ ] 8. Integrate the training/evaluation harness with adapter and baseline

  **What to do**: Connect the CLI-level harness from Task 5 to the actual RoboCasa adapter from Task 6 and the dual-arm RL baseline from Task 7. Implement reproducible checkpoint paths, resume/evaluate workflows, metrics export, and failure exits. Ensure the harness can run short smoke training and fixed-seed evaluation without requiring manual simulator setup beyond documented configs.
  **Must NOT do**: Do not make train/eval paths depend on hidden notebook state or environment-specific local paths; do not bypass the wrapper contract when launching environments.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: integration-heavy task joining contracts from multiple prior tasks.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - too many cross-module dependencies.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9, 10 | Blocked By: 1, 2, 3, 5, 6, 7

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:64-101` - config loading and target-source utility style; keep CLI/config ergonomics similarly simple.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\logs\run_logger.py:89-95` - deterministic artifact writing style.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:467-470` - baseline engineering acceptance requires reproducible startup, logs, and minimal tests.

  **Acceptance Criteria** (agent-executable only):
  - [ ] End-to-end smoke train/eval command chain exits 0 using RoboCasa backend, writes checkpoint + metrics + run log artifacts, and can be re-run with the same seed set.
  - [ ] A resume-from-checkpoint command exits 0 and continues from a previous smoke checkpoint or clearly indicates unsupported resume with explicit error if that choice was formalized.
  - [ ] An invalid-backend or invalid-checkpoint invocation exits non-zero with explicit, user-actionable error output.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: End-to-end harness happy path
    Tool: Bash
    Steps: Run the documented smoke training command, then the matching evaluation command, then the resume or second evaluation command against the same checkpoint.
    Expected: Exit code 0; deterministic artifacts exist and can be compared across reruns.
    Evidence: .sisyphus/evidence/task-8-harness-integration.txt

  Scenario: Invalid launch failure path
    Tool: Bash
    Steps: Run the same CLI with an invalid backend name or checkpoint path.
    Expected: Non-zero exit with explicit configuration or path error.
    Evidence: .sisyphus/evidence/task-8-harness-integration-error.txt
  ```

  **Commit**: YES | Message: `infra/rl: wire robocasa baseline into train eval harness` | Files: entrypoints, adapter integration, configs, tests, `docs/task.md`

- [ ] 9. Reconcile orchestration and logging with the frozen upper-layer stack

  **What to do**: Make the new RL baseline coexist cleanly with the current project orchestration so future phases can reintroduce frozen `vision/`, `ipc/`, `fsm/`, and `logs/` without rewrite. Extract only the minimum orchestration seams required from `main.py` and ensure milestone-1 RL artifacts can be compared against the existing failure-mode and execution-summary vocabulary. Update or add integration utilities so the project can later switch between heuristic/legacy and RL/RoboCasa baselines without protocol drift.
  **Must NOT do**: Do not perform a broad “clean architecture” rewrite of the whole repo; do not change top-level logging names to fit RL convenience.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this is controlled architectural reconciliation with strict anti-scope-creep constraints.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - dangerous given the orchestration boundary sensitivity.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 10 | Blocked By: 4, 5, 6, 7, 8

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:194-236` - current execution summary/failure count vocabulary that later A/B comparisons should preserve.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\main.py:239-260` - current detection/selection summary style; future reintegration should be able to plug back into a known orchestration shape.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:156-167` - main flow should shrink over time while wrapper/runner boundaries get clearer.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\AGENTS.md:31-46` - task summary sync and git-visible project status must remain aligned.

  **Acceptance Criteria** (agent-executable only):
  - [ ] An integration smoke test exits 0 and proves legacy protocol validators still pass after RL/RoboCasa baseline integration.
  - [ ] A comparison/evaluation command exits 0 and produces metrics/logs that retain the existing failure-mode vocabulary.
  - [ ] A failure-path integration command exits non-zero when protocol drift is intentionally introduced in a test fixture.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Upper-layer compatibility happy path
    Tool: Bash
    Steps: Run the integration smoke suite after wiring the RL/RoboCasa baseline into the project, including protocol validators and one evaluation run.
    Expected: Exit code 0; evidence shows protocol contracts still pass and logs retain legacy-compatible fields.
    Evidence: .sisyphus/evidence/task-9-upper-layer-compatibility.txt

  Scenario: Protocol drift failure path
    Tool: Bash
    Steps: Run the same integration suite against a fixture or branch-local test that intentionally changes a frozen field name.
    Expected: Non-zero exit with explicit compatibility failure.
    Evidence: .sisyphus/evidence/task-9-upper-layer-compatibility-error.txt
  ```

  **Commit**: YES | Message: `integration/orchestration: preserve frozen protocol compatibility for rl baseline` | Files: `main.py` seams, integration utilities/tests, `docs/task.md`

- [ ] 10. Update project progress and phase gates for the new roadmap

  **What to do**: Update `docs/task.md` so the project’s authoritative progress board reflects the new roadmap chosen in this planning session: milestone 1 is a direct dual-arm RL baseline in RoboCasa with privileged observations and a minimal train/eval harness; milestone 2 adds obstacle clearing before target grasp; milestone 3 reintroduces semantic vision for obstacle/target discrimination; milestone 4 targets cluttered dual-arm collaborative clearing-and-grasp. Preserve a clear owner group, success criteria, verification commands, and recommended next step.
  **Must NOT do**: Do not leave `docs/task.md` describing RL as purely future/postponed once milestone-1 implementation begins; do not document local-machine-only runtime notes.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: this is authoritative project-status and roadmap documentation with technical precision requirements.
  - Skills: `[]` - no extra skill required.
  - Omitted: `quick` - documentation drift here would mislead all later agents.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Final Verification Wave | Blocked By: 8, 9

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:5-79` - current authoritative status summary and bottleneck framing.
  - Pattern: `D:\pyCode\robot\final\robotic-final-proj\docs\task.md:251-362` - current RL guidance that must be updated carefully to reflect the new chosen roadmap without losing risk framing.
  - Constraint: `D:\pyCode\robot\final\robotic-final-proj\AGENTS.md:31-48` - `docs/task.md` is the single source of truth and must include recommended next step, owner group, success criteria, and verification.

  **Acceptance Criteria** (agent-executable only):
  - [ ] A documentation validation/readback step confirms `docs/task.md` now names the RoboCasa dual-arm RL baseline as the active milestone and describes the later phase gates accurately.
  - [ ] The updated document explicitly records scope boundaries for phase 1 vs phase 2/3/4 and a concrete recommended next step.
  - [ ] A drift check flags failure if the roadmap still states RL is deferred until after RoboCasa migration despite the implemented milestone.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: docs/task.md happy path
    Tool: Bash
    Steps: Run a documentation validation/readback command or script that checks required roadmap headings and milestone language in docs/task.md.
    Expected: Exit code 0; evidence shows the new active milestone, owner group, verification, and next step are present.
    Evidence: .sisyphus/evidence/task-10-roadmap-docs.txt

  Scenario: Roadmap drift failure path
    Tool: Bash
    Steps: Run the same validation against a stale docs/task.md fixture missing the new roadmap language.
    Expected: Non-zero exit with explicit missing-section or stale-roadmap error.
    Evidence: .sisyphus/evidence/task-10-roadmap-docs-error.txt
  ```

  **Commit**: YES | Message: `docs/task: update roadmap for robocasa dual-arm rl baseline` | Files: `docs/task.md`, validation scripts if added

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer atomic commits aligned to wave boundaries.
- Expected style: `<group>/<scope>: <verb> <what>` per `AGENTS.md`.
- Do not commit generated training artifacts, caches, `.venv/`, or local logs beyond tracked schema fixtures/tests.

## Success Criteria
- RoboCasa migration is isolated behind the environment wrapper boundary.
- Frozen protocol tests prove `vision/`, `ipc/`, `fsm/`, and `logs/` remain compatible.
- Dual-arm RL baseline is trainable, checkpointable, and evaluable on a fixed seed set.
- The milestone-1 baseline is explicitly limited to approach-and-grasp and does not absorb later-phase clutter semantics.
- `docs/task.md` clearly reflects completed milestone scope, remaining roadmap, owner group, and recommended next step.
