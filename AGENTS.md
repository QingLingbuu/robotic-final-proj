# 🤖 AGENTS.md — robotic-final-proj AI Agent Rules

**Project:** Semantic-Driven Dual-Arm Collab Desktop Tidy (RoboCasa)
**Version:** V3.4 (2026-04)
**Status:** Interface Controlled. Breaking changes require explicit `[BREAKING CHANGE]` tags and full-group review.

## 0. Karpathy Directives (The Four Rules)
All agents must internalize these rules across *every* action: coding, reviewing, committing, commenting, and collaborating.

1.  **Think Before Coding:** Never assume an API contract. State your assumptions explicitly. If uncertain, ask in the Issue comment before pushing code.
2.  **Simplicity First:** Minimum code to solve the problem. No speculative generalization. No abstractions for single-use logic.
3.  **Surgical Changes:** Touch *only* what is necessary. If a PR is meant to fix FSM state 3, do not "clean up" the perception dictionary keys, even if you think they are ugly.
4.  **Goal-Driven Execution:** Turn requirements into verifiable goals. Define success criteria before implementation. Loop until verified.

---

## 1. GitHub Collaboration Protocols
How agents interact with the repository, issues, and pull requests.

### 1.1 Multi-Agent Routing & Ownership
To prevent multiple agents from stepping on each other, follow strict routing rules:

| Agent Group | Primary Directory | Responsibility |
|---|---|---|
| **`perception`** | `vision/`, `calibration/`, `ipc/` | Camera intrinsics, DINO inference, IPC writer, coordinate math |
| **`logic`** | `fsm/`, `planner/`, `counters/` | State machine, retry/fallback logic, failure mode counters |
| **`execution`** | `arm/`, `osc/`, `safety/` | Robot control, collision safety, `arm_safe_retract()` |
| **`eval`** | `logs/`, `metrics/`, `report/` | Experiment logging, statistical analysis, report generation |
| **`infra`** | `configs/`, `docs/`, `scripts/` | Configuration, setup, CI/testing pipelines |

*   **Collision Prevention:** Before starting work, an agent **must** check Issues labeled `👤 assigned`. If multiple agents claim the same issue, the one with the earliest comment claiming ownership takes precedence.
*   **Cross-Boundary PRs:** If a change touches multiple groups (e.g., modifying both `ipc/` and `fsm/`), the PR description *must* explicitly tag affected owners and request review from all involved groups.

### 1.2 Issue Handling
*   **Claim First, Code Second:** Reply with "Claiming this. Plan: [brief plan]. Verification: [method]." *Then* apply the `👤 assigned` label.
*   **Clarification Loop:** If the Issue description is ambiguous, *ask* in a comment. Do not hallucinate the intended scope.
*   **Kill Feature Creep:** If a request implies 1000 lines but 100 will suffice, push back in the comments. Propose the minimal path.

### 1.3 Pull Requests (PR)
*   **Atomic Changes:** One PR = One logical change. Never mix a bugfix with a refactoring or a new feature.
*   **PR Body Template:** Every PR must include Goal and Verification sections:
    ```markdown
    ## Goal
    [One sentence. e.g., "Fix FSM RETRY_SENSING counter overflow"]
    ## Verification
    - [ ] All existing tests pass
    - [ ] Manual test: [Describe simple run to verify]
    - [ ] `detected_objects` schema intact
    - [ ] No hardcoded intrinsics introduced
    ```
*   **No Drive-by Refactors:** Do not rename variables or fix whitespace outside the exact files needed for the PR goal.

### 1.4 Code Reviews
*   **State Your Reason:** Approve or reject with a brief reason. "LGTM" alone is not enough — explain what you verified.
*   **Surface Tradeoffs:** If a PR introduces a hardcoded value (e.g., camera intrinsics, magic numbers), comment immediately.
*   **Run Diff Locally** if the change touches FSM logic, safety code, or IPC interfaces.

---

## 2. Technical Architecture (V3.4 Strict)
The following constraints are **hard blocks**. Never violate them.

### 2.1 3D Coordinate & Camera
*   **Math Alignment:** `Px_cam = [(u - cx)*d/fx, (v - cy)*d/fy, d]^T`. World: `P_world = R_wc * P_cam + t_wc`.
*   **DO:** Use `T_world_cam` from config/RoboCasa object.
*   **DO NOT:** Hardcode intrinsics `(fx, fy, cx, cy)`. Intrinsics must be sourced from `configs/camera.yaml` or the RoboCamera API at runtime.

### 2.2 IPC (`perception_queue`)
Percept-to-logic communication uses `multiprocessing.Queue` — Python's built-in, zero-boilerplate choice.

*   **Protocol:**
    *   Writer (perception): `queue.put(detected_objects)`
    *   Reader (logic): `detected_objects = queue.get(timeout=0.2)`
    *   `queue.Empty` → stale data → FSM transitions to **RETRY_SENSING**
*   **Hard Block:** All IPC between perception and logic **must** use this queue name. Sockets, files, TCP, or manual SharedMemory are forbidden — they add complexity without measurable benefit in a RoboCasa course project.

### 2.3 `detected_objects` Schema Constraints
Strict format. Do not alter structure or key names.

```python
detected_objects = {
    "target": {"label": "str", "pos": [x, y, z], "conf": float, "timestamp": float},
    "obstacles": [{"label": "str", "pos": [x, y, z], "id": int}],
    "status": "ready" | "processing" | "error"
}
```

*   **Obstacle Limit:** Maximum 16 entries. If DINO detects more, sort by `conf` and truncate. Exceeding 16 bloats IPC latency and FSM processing overhead.
*   **Clock Source:** `timestamp` **must** use `time.monotonic()`, NOT `time.time()`, to prevent logic skew from system clock jumps.
*   **Confidence Threshold (`CONF_THRESHOLD`):** Must be centralized in `configs/thresholds.yaml`. If `conf < CONF_THRESHOLD`, `perception` must set `status = "error"` and the FSM must NOT use the target position.

### 2.4 FSM State Machine (Transitions & Fallbacks)
| State | Normal | Exception | Fallback |
|---|---|---|---|
| **IDLE** | Cmd → **PLANNING** | - | - |
| **PLANNING** | Locked → **CLEARING** | Lost → **RETRY_SENSING** | 3x Lost → **FAILED → IDLE** |
| **CLEARING** | Done → **PLANNING** | Blocked → **RETRY_PUSH** | 2x Push → **FAILED → RESET** |
| **GRASPING** | Closed → **VERIFYING** | Col → **EMERGENCY** | **3s Timeout → FAILED → IDLE** |
| **VERIFYING** | Success → **SUCCESS** | Empty → **RETRY_GRASP** | 2x Empty → **FAILED → IDLE** |

---

## 3. Experiment & Reporting
### 3.1 Failure Mode Categorization
Every run must be logged. Do not use "unknown" or "other".
*   **Perception Error ($N_1$):** Count every `RETRY_SENSING` trigger.
*   **Execution Drift ($N_2$):** Count every `RETRY_PUSH` trigger.
*   **Physical Slip ($N_3$):** Count every `RETRY_GRASP` trigger.

### 3.2 Reproducibility & Run Log Format
Logs must capture the exact environment to allow exact experiment replication.
```json
{
  "run_id": "20260423_01",
  "commit_hash": "a1b2c3d",
  "config_version": "v1.2",
  "scene_config": {
    "seed": 42,
    "object_count": 8,
    "scene_id": "kitchen_table_01"
  },
  "timestamp": "...",
  "success": false,
  "failure_mode": "execution_drift",
  "counts": {"n1": 1, "n2": 2, "n3": 0},
  "duration_sec": 12.5,
  "context": "Blocker failed to move past x_threshold"
}
```

---

## 4. Safety & Degradation
*   **Emergency Halt:** `arm_safe_retract()` must *always* exist. It returns arms to idle instantly.
*   **Collision:** Triggers **EMERGENCY**. Physically stops motors. Do not wrap in `try-except` to silently suppress.
*   **Thresholds:** Do not lower friction or collision thresholds to force success. This invalidates the experiment.
*   **Safe Defaults:** If `perception_queue.get()` times out or returns invalid data, the execution group must assume the environment is unpredictable and enter **RETRY_SENSING** or **IDLE**. Do not attempt to move blindly.

---

## 5. Development Milestones
*   **T+3:** Interface Demo (Calibration + RGB/Depth rendering). *Check: XYZ error < 5mm.*
*   **T+7:** Vision Loop (Queue IPC + Single-arm grasp). *Check: Latency < 200ms.*
*   **T+12:** Final Eval (Dual-arm RoboCasa run). *Check: N1+N2+N3 < 10.*

---

## 6. Git Hygiene & Release Flow
*   **Message Format:** `<group>/<scope>: <verb> <what>`. 
    *   Example: `logic/fsm: fix RETRY_SENSING counter overflow`
*   **Source-Only Commits (Hard Rule):** Commits and pushes must include source and project files only. Never commit runtime artifacts such as `.venv/`, `__pycache__/`, `*.pyc`, local caches, or other generated files.
*   **Branch Strategy:**
    *   `main`: Stable, runnable.
    *   `feat/<group>-<feature>`: Feature dev.
    *   `fix/<group>-<bug>`: Bug fix. (covers normal fixes and urgent hotfixes)
    *   **Never** push untested code to `main`.
*   **Verification Checklist — Every Merge:**
    1. `lsp_diagnostics` clean? ✅
    2. Tests passed? ✅
    3. No hardcoded intrinsics? ✅
    4. `perception_queue` interface unchanged? ✅
    5. `detected_objects` schema intact? ✅
*   **Verification Checklist — Before Evaluation Run:**
    - [ ] `eval/` run log template includes `scene_config` + `commit_hash` ✅
    - [ ] `CONF_THRESHOLD` sourced from `configs/thresholds.yaml`, not hardcoded ✅
    - [ ] Queue `get(timeout=0.2)` stale handling in place ✅
