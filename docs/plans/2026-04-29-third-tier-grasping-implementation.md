# Third-Tier Grasping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a candidate-driven grasping pipeline where vision publishes `grasp_candidates`, FSM selects candidates, and control executes candidate geometry.

**Architecture:** Extend the existing `detected_objects` payload into a candidate-aware perception payload, then refactor planning and execution to consume candidates instead of heuristic offsets. Roll the change out in vertical slices so the pipeline stays debuggable.

**Tech Stack:** Python, robosuite baseline, multiprocessing queue IPC, YAML config, existing FSM and control modules, optional downstream RL policy.

---

### Task 1: Define candidate schema and validation

**Files:**
- Modify: `vision/detected_objects.py`
- Modify: relevant schema tests or add `tests/test_detected_objects_candidates.py`

**Step 1: Write the failing test**
- Add tests that validate a payload containing `grasp_candidates` with `id`, `pos`, `orientation`, `gripper_width`, `score`, and `grasp_type`.

**Step 2: Run test to verify it fails**
- Run: `python -m pytest tests/test_detected_objects_candidates.py -v`
- Expected: schema validation fails because candidates are unsupported.

**Step 3: Write minimal implementation**
- Extend schema builder and validator to support candidate lists while preserving existing `target`, `obstacles`, and `status` behavior.

**Step 4: Run test to verify it passes**
- Run the same pytest command and confirm PASS.

### Task 2: Generate placeholder candidates in vision

**Files:**
- Modify: `vision/perception_loop.py`
- Modify: `configs/vision.yaml`
- Add or modify tests around perception payload generation

**Step 1: Write the failing test**
- Add tests for perception output to require a non-empty `grasp_candidates` list for detectable tabletop objects.

**Step 2: Run test to verify it fails**
- Run the targeted perception tests.

**Step 3: Write minimal implementation**
- Generate one or more baseline candidates from existing target geometry using deterministic heuristics as a placeholder third-tier backend.
- Add config toggles for candidate count and candidate types.

**Step 4: Run test to verify it passes**
- Re-run perception tests and confirm PASS.

### Task 3: Teach the queue consumer to read candidate payloads

**Files:**
- Modify: `fsm/perception_cycle.py`
- Modify: any queue consumer tests

**Step 1: Write the failing test**
- Add tests asserting the perception cycle treats `status=ready` with valid candidates as planning-ready.

**Step 2: Run test to verify it fails**
- Run the targeted FSM perception tests.

**Step 3: Write minimal implementation**
- Update payload consumption helpers to check candidate availability and propagate candidate lists upstream.

**Step 4: Run test to verify it passes**
- Re-run the same tests and confirm PASS.

### Task 4: Refactor FSM planning around selected candidates

**Files:**
- Modify: `fsm/state_machine.py`
- Modify: `main.py` planning path
- Add tests for candidate retry sequencing

**Step 1: Write the failing test**
- Add tests covering candidate selection, candidate exhaustion, and retry transitions.

**Step 2: Run test to verify it fails**
- Run targeted FSM tests.

**Step 3: Write minimal implementation**
- Introduce selected-candidate state, candidate index tracking, and retry semantics that advance across candidates before declaring failure.

**Step 4: Run test to verify it passes**
- Re-run targeted FSM tests and confirm PASS.

### Task 5: Move control to candidate execution

**Files:**
- Modify: `arm/controller.py`
- Modify: `main.py` grasp execution path
- Add or modify tests for candidate execution helpers

**Step 1: Write the failing test**
- Add tests requiring execution helpers to accept candidate geometry rather than heuristic offsets only.

**Step 2: Run test to verify it fails**
- Run targeted control tests.

**Step 3: Write minimal implementation**
- Add candidate execution adapters that consume `pos`, `orientation`, and `gripper_width`; keep a compatibility fallback only where unavoidable.

**Step 4: Run test to verify it passes**
- Re-run control tests and confirm PASS.

### Task 6: Log candidate decisions for RL and debugging

**Files:**
- Modify: `runtime/run_logger.py`
- Modify: `docs/task.md` once implementation work starts affecting roadmap
- Add tests for log payload contents

**Step 1: Write the failing test**
- Add tests requiring logs to capture candidate count, selected candidate ID, selected candidate score, and failure stage attribution.

**Step 2: Run test to verify it fails**
- Run logger tests.

**Step 3: Write minimal implementation**
- Extend run logs with candidate metadata and failure attribution fields.

**Step 4: Run test to verify it passes**
- Re-run logger tests and confirm PASS.

### Task 7: Add an RL-ready candidate selector interface

**Files:**
- Modify: `rl/contracts.py`
- Modify: `rl/harness.py`
- Add tests for selector observation and action format

**Step 1: Write the failing test**
- Add tests describing an observation structure containing candidate lists and an action that selects candidate index or requests fallback.

**Step 2: Run test to verify it fails**
- Run targeted RL contract tests.

**Step 3: Write minimal implementation**
- Define a selector-facing contract without yet requiring a full learned policy.

**Step 4: Run test to verify it passes**
- Re-run RL contract tests and confirm PASS.

### Task 8: End-to-end verification slice

**Files:**
- Modify as needed based on prior tasks
- Update `docs/task.md` with completed work and next step once implementation starts

**Step 1: Run focused tests**
- Run all targeted tests added above.

**Step 2: Run type and diagnostics checks**
- Use `lsp_diagnostics` and any existing validation commands.

**Step 3: Run a smoke flow**
- Exercise the planning-to-execution path with candidate payloads in the current robosuite baseline.

**Step 4: Record outcomes**
- Capture what worked, remaining blockers, and what the next implementation slice should be.
