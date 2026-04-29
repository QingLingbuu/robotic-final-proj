# Third-Tier Grasping Design

**Date:** 2026-04-29

## Goal
Upgrade the manipulation pipeline from object-position-driven grasping to candidate-driven grasping, where vision produces structured `grasp_candidates` and downstream FSM / control / RL consume those candidates for selection and execution.

## Scope
This design targets tabletop grasping for dispersed household objects. It assumes the current robosuite baseline remains the integration environment and preserves the existing queue-based perception handoff.

## Recommended Architecture
The system should move from `target.pos -> heuristic grasp point -> execute` to `grasp candidate generation -> candidate filtering -> candidate selection -> execution`. Vision becomes responsible for proposing graspable options, control becomes responsible for feasibility checking and execution, and RL becomes responsible primarily for selecting among structured candidates.

## Candidate Schema
Each candidate should include at minimum:
- `id`: stable candidate identifier within one perception pass
- `pos`: 3D world-frame grasp anchor
- `orientation`: grasp orientation or approach frame
- `gripper_width`: suggested opening width
- `score`: vision-side confidence or prior quality
- `grasp_type`: semantic tag such as `top_down`, `side_grasp`, `handle_grasp`

Recommended queue payload shape:

```python
{
  "target": {
    "label": "cup",
    "pos": [x, y, z],
    "conf": 0.93,
    "timestamp": 123.45,
  },
  "grasp_candidates": [
    {
      "id": 1,
      "pos": [x, y, z],
      "orientation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
      "gripper_width": 0.042,
      "score": 0.91,
      "grasp_type": "top_down",
      "reachable_hint": true,
    }
  ],
  "obstacles": [...],
  "status": "ready",
}
```

## Layer Responsibilities

### Vision
- Detect target objects and produce one or more executable grasp candidates per frame
- Preserve `target` and `obstacles` for semantics, logging, and clearing decisions
- Rank candidates with a lightweight prior score before downstream selection

### IPC
- Keep `multiprocessing.Queue` and the existing perception queue naming
- Extend payload schema rather than replacing the queue mechanism
- Treat `status` as candidate availability, not only target availability

### FSM
- Replace target-centric planning with candidate-centric planning
- `PLANNING` selects a candidate, `RETRY_GRASP` advances to the next candidate or triggers re-sensing
- `CLEARING` should consider candidate occlusion or infeasibility, not only object presence

### Control / Execution
- Consume selected candidate geometry rather than invent grasp offsets
- Validate reachability, IK, collision safety, and execution preconditions
- Return explicit failure reasons so the FSM or RL policy can adapt

### RL
- Preferred role: candidate selector, not end-to-end low-level actor
- Input should include candidate geometry, candidate scores, obstacle relations, current FSM state, and retry history
- Actions should include selecting candidate `i`, requesting clearing, or requesting re-sensing

## Why This Direction
This architecture has the highest ceiling for random object poses and later RL integration while avoiding the sample inefficiency and debugging difficulty of full end-to-end visual motor control. It keeps perception and control modular while shifting the system core object from `target position` to `grasp candidate`.

## Key Risks
- Candidate geometry may still be noisy if derived from coarse 2D detections and sparse depth
- Orientation conventions must be explicit and consistent across vision and control
- Candidate failure attribution must be logged clearly or RL training will receive ambiguous signals

## Success Criteria
- Vision can produce multiple structured candidates for a dispersed tabletop scene
- FSM can choose, retry, and fall back across candidates
- Control can execute a selected candidate without needing heuristic offset generation
- RL can be introduced first as a candidate-ranking policy without redesigning the interface
