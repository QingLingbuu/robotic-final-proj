## Vision Artifacts

This directory stores selected, versioned visual references derived from runtime exports.

Rules:
- Keep raw runtime outputs under `outputs/`.
- Copy only representative artifacts here when they are useful for review, demos, or merge-time traceability.
- Prefer small, curated snapshots over large run dumps.

Current snapshot:
- `cup_mug_sorting_all/`: representative multi-target Cup/Mug point cloud export, including RGB/depth overlays, camera reprojections, summary JSON, and point cloud `.ply` files.
