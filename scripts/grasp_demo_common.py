"""Shared helpers for robosuite grasp demo scripts."""

import numpy as np


def resolve_gt_object(env_wrapper, scenario):
    env = env_wrapper.env
    normalized = str(scenario).strip().lower()

    if normalized == "cube":
        obj = getattr(env, "cube", None)
        body_name = getattr(obj, "root_body", None)
        body_pos = None
        if body_name is not None:
            body_pos = np.array(
                env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)], dtype=float
            )
        bbox_half = (
            None if obj is None else np.asarray(obj.get_bounding_box_half_size(), dtype=float)
        )
        return {
            "label": "cube",
            "body_name": body_name,
            "center_pos": None if body_pos is None else body_pos,
            "bbox_half_size": None if bbox_half is None else bbox_half,
        }

    obj_name = getattr(env, "obj_to_use", None)
    objects = list(getattr(env, "objects", []))
    obj = None
    for candidate in objects:
        candidate_name = getattr(candidate, "name", None)
        if obj_name is not None and candidate_name == obj_name:
            obj = candidate
            break
    if obj is None and len(objects) == 1:
        obj = objects[0]
    if obj is None and getattr(env, "object_id", None) is not None:
        index = int(env.object_id)
        if 0 <= index < len(objects):
            obj = objects[index]

    body_name = None if obj is None else getattr(obj, "root_body", None)
    body_pos = None
    if body_name is not None:
        body_pos = np.array(
            env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)], dtype=float
        )
    bbox_half = (
        None if obj is None else np.asarray(obj.get_bounding_box_half_size(), dtype=float)
    )

    return {
        "label": normalized,
        "body_name": body_name,
        "center_pos": None if body_pos is None else body_pos,
        "bbox_half_size": None if bbox_half is None else bbox_half,
    }


def build_gt_summary(env_wrapper, scenario):
    gt = resolve_gt_object(env_wrapper, scenario)
    center_pos = gt["center_pos"]
    bbox_half = gt["bbox_half_size"]
    top_surface_z = None
    if center_pos is not None and bbox_half is not None and bbox_half.shape[0] >= 3:
        top_surface_z = float(center_pos[2] + bbox_half[2])
    return {
        "label": gt["label"],
        "body_name": gt["body_name"],
        "center_pos": None if center_pos is None else np.asarray(center_pos, dtype=float).tolist(),
        "bbox_half_size": None if bbox_half is None else np.asarray(bbox_half, dtype=float).tolist(),
        "top_surface_z": top_surface_z,
    }


def resolve_gt_grasp_target(gt_summary, penetration_offset):
    center_pos = gt_summary.get("center_pos")
    if center_pos is None:
        raise RuntimeError("GT object center is unavailable for this scenario.")

    target = np.asarray(center_pos, dtype=float).copy()
    top_surface_z = gt_summary.get("top_surface_z")
    if top_surface_z is not None:
        target[2] = float(top_surface_z) - float(penetration_offset)
    return target
