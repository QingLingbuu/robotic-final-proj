# RoboCasa Single-Arm Execution Loop Design

**Date:** 2026-05-01

## Goal
Extend the current RoboCasa onscreen reach smoke into a minimal single-arm execution loop smoke: `hover reach -> settle reach -> close gripper -> vertical lift`, without yet integrating FSM, run logger, or the main `robosuite` orchestration path.

## Scope
This design is intentionally narrow. It only upgrades `scripts/demo_robocasa_reach_onscreen.py` from a reach-only visual smoke into a minimal execution smoke that can prove the RoboCasa path is capable of running one complete manipulation attempt after candidate selection.

In scope:
- keep `scripts/demo_robocasa_reach_onscreen.py` as the only CLI entrypoint
- preserve the existing visual candidate generation and candidate selection path
- add post-reach stages for gripper close and lift
- emit a structured stage summary for manual inspection

Out of scope:
- no FSM integration
- no `runtime/run_logger.py` integration
- no migration into `main.py`
- no retry policy or candidate reselection loop
- no broad execution framework for every RoboCasa task

## Recommended Architecture
Use a thin execution helper that the onscreen script calls after candidate selection. The script should remain responsible for environment boot, camera config, viewer lifecycle, and vision inference. The helper should own only the minimal execution stages and their summaries.

This keeps the current smoke workflow intact while avoiding a larger architecture jump. It also preserves the migration boundary documented in `docs/architecture/robocasa-migration.md`: simulator-specific control details stay in the lower execution path instead of leaking upward into the broader orchestration layer.

## Runtime Flow
1. Build RoboCasa env and show the live viewer.
2. Capture RGB-D and run `VisionPerceptionLoop`.
3. Select one candidate exactly as today.
4. Optionally run axis calibration exactly as today.
5. Execute `hover -> settle` reach exactly as today.
6. If reach succeeds, execute gripper close.
7. If close succeeds, execute a fixed world-frame lift.
8. Print a structured execution summary and keep the viewer open.

## Module Boundary
### Script responsibilities
`scripts/demo_robocasa_reach_onscreen.py` should continue to own:
- argument parsing
- environment creation
- camera config resolution
- vision inference and candidate selection
- optional axis calibration
- viewer keep-open behavior
- final top-level console summary

### New helper responsibilities
A small RoboCasa execution helper should own:
- close-gripper stage
- lift stage
- per-stage success flags
- final execution summary payload
- compact phase history needed for debugging

The helper should be intentionally local in purpose. It should not pretend to be a reusable full project execution framework yet.

## Failure Handling
Failure handling stays stage-based and non-recovering for this smoke.

- If `hover` or `settle` does not finish within the allowed steps, stop and report a reach failure.
- If gripper close cannot be executed or finishes with obviously invalid output, stop and report a close failure.
- If lift cannot complete, stop and report a lift failure.
- No candidate reselection.
- No perception refresh.
- No retry loop.

This is deliberate. The purpose of this milestone is to prove that RoboCasa has a minimum viable execution loop beyond pure reach, not to design the final recovery policy.

## Success Criteria
The smoke is successful when all of the following are true:
- one visual candidate is selected
- reach finishes its `hover` and `settle` phases successfully
- gripper close stage runs successfully
- lift stage runs successfully
- final summary includes stage outcomes and post-lift EEF pose

For this stage, “success” means the execution pipeline completed coherently. It does **not** yet require proving robust object pickup across all scenes.

## Summary Payload
The final output should include at least:
- selected candidate metadata
- `reach_success`
- `close_success`
- `lift_success`
- `eef_before_close`
- `eef_after_lift`
- `lift_delta_z`
- final error to settle target
- phase failure point, if any

This keeps the output directly useful for manual debugging and gives a clean future bridge to the project run logger.

## Testing and Verification
### Static verification
- ensure the modified script still imports cleanly
- ensure the helper imports cleanly
- ensure no syntax or obvious path/bootstrap regressions are introduced

### Manual smoke verification
Run the real command in the RoboCasa environment:

```powershell
python scripts/demo_robocasa_reach_onscreen.py --task robocasa/CoffeeSetupMug --target-label mug
```

Expected evidence:
- viewer shows the arm approach the candidate
- gripper close stage visibly occurs
- post-close lift visibly occurs
- terminal prints a structured summary with stage booleans and lift delta

## Known Control Limitation: Extremely Slow Reach Motion
The current `scripts/demo_robocasa_reach_onscreen.py` reach controller already shows a known behavioral limitation: the arm moves toward the object, but very slowly. This is not just a viewer illusion; it follows from the current control design in `execute_reach(...)`.

Contributing factors already visible in code:
- world-frame motion is clipped by a conservative `step_limit=0.03`
- motion is split into separate `hover` and `settle` phases instead of one assertive approach
- when axis calibration is enabled, the desired world delta is passed through `pinv(action_mapping)`, which can further shrink effective action-space motion
- each control step also pays the cost of `env.render()` plus `time.sleep(render_sleep_sec)`

Design implication for this milestone:
- the first execution-loop implementation should treat slow reach as an explicit known constraint, not as proof that candidate selection is wrong
- the new `close -> lift` stages must be added in a way that preserves visibility into stage timing and per-stage motion
- the design should leave room to expose reach tuning parameters later, but parameterization itself is not required for this first minimal execution-loop milestone

## Risks
- current action-to-world mapping may still be too conservative, causing extremely slow reach and lift motion
- gripper close semantics for the PandaOmron composite controller may need empirical tuning
- a visually plausible reach may still fail to produce a stable grasp because candidate geometry is only approximate

## Immediate Next Step After This Design
Implement the minimal execution helper and wire it into `scripts/demo_robocasa_reach_onscreen.py` without touching `main.py`, FSM, or the run logger.
