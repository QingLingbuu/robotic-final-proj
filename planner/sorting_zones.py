"""Placement target helpers for cup / mug sorting."""

import math


def _fixture_pos(anchor):
    if not anchor:
        return None
    pos = anchor.get("pos")
    if pos is None:
        return None
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _world_offset(anchor, offset):
    pos = _fixture_pos(anchor)
    if pos is None:
        return None
    rot = float(anchor.get("rot", 0.0) or 0.0)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    dx, dy, dz = [float(v) for v in offset]
    return [
        pos[0] + cos_r * dx - sin_r * dy,
        pos[1] + sin_r * dx + cos_r * dy,
        pos[2] + dz,
    ]


def _local_offset(anchor, point):
    pos = _fixture_pos(anchor)
    if pos is None or point is None:
        return None
    rot = float(anchor.get("rot", 0.0) or 0.0)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    dx = float(point[0]) - pos[0]
    dy = float(point[1]) - pos[1]
    return [
        cos_r * dx + sin_r * dy,
        -sin_r * dx + cos_r * dy,
        float(point[2]) - pos[2] if len(point) > 2 else 0.0,
    ]


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _suffix_index(name):
    if not name:
        return None
    parts = str(name).rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _category_y_offset(name, step_size=0.1):
    index = _suffix_index(name)
    if index is None:
        return 0.0
    return float(index - 1) * float(step_size)


def _nearest_sink_target(sink_anchor, source_pos, place_z, basin_sign=None, basin_y_offset=0.0):
    sink_pos = _fixture_pos(sink_anchor)
    if sink_pos is None:
        return None

    sink_width = float(sink_anchor.get("width") or sink_anchor.get("size", [0.5, 0.4, 0.2])[0])
    sink_depth = float(sink_anchor.get("depth") or sink_anchor.get("size", [0.5, 0.4, 0.2])[1])
    local_source = _local_offset(sink_anchor, source_pos) or [0.0, 0.0, 0.0]

    # For double-basin sinks, choose an explicit basin side when requested.
    # Fallback to the side nearest the current object when no preference exists.
    if basin_sign is None:
        basin_sign = -1.0 if local_source[0] < 0.0 else 1.0
    basin_sign = -1.0 if float(basin_sign) < 0.0 else 1.0
    target_local_x = basin_sign * sink_width * 0.22
    target_local_y = _clamp(local_source[1] + float(basin_y_offset), -sink_depth * 0.20, sink_depth * 0.20)
    target = _world_offset(
        sink_anchor,
        [target_local_x, target_local_y, float(place_z) - sink_pos[2]],
    )
    if target is None:
        return None
    return [float(target[0]), float(target[1]), float(target[2])]


def choose_cup_mug_place_target(assignment, sim_objects, zone_margin=0.10, place_anchors=None):
    """Choose a world-space place target for the selected cup/mug assignment."""
    positions = [obj.get("pos") for obj in sim_objects if obj.get("pos") is not None]
    if not positions:
        source_pos = assignment.get("pos", [0.0, 0.0, 0.9])
        positions = [source_pos]

    xs = [float(pos[0]) for pos in positions]
    ys = [float(pos[1]) for pos in positions]
    zs = [float(pos[2]) for pos in positions]
    is_handled = bool(assignment.get("has_handle"))
    place_z = max(zs) + 0.04

    anchors = place_anchors or {}
    sink_anchor = anchors.get("sink")
    counter_anchor = anchors.get("counter")

    if is_handled and sink_anchor:
        assignment_name = str(assignment.get("sim_object_name") or "")
        basin_y_offset = 0.0
        if assignment_name.endswith("_1"):
            basin_y_offset = -0.04
        elif assignment_name.endswith("_2"):
            basin_y_offset = 0.04
        sink_target = _nearest_sink_target(
            sink_anchor,
            assignment.get("pos"),
            place_z,
            basin_sign=1.0,
            basin_y_offset=basin_y_offset,
        )
        if sink_target is not None:
            return {
                "zone": "sink",
                "target_pos": sink_target,
                "source_assignment": assignment,
                "anchor": "sink",
                "placement_rule": "spaced_sink_basin",
            }

    if (not is_handled) and counter_anchor:
        counter_pos = _fixture_pos(counter_anchor)
        if counter_pos is not None:
            sink_pos = _fixture_pos(sink_anchor)
            counter_width = float(counter_anchor.get("width") or counter_anchor.get("size", [0.8, 0.6, 0.9])[0])
            counter_depth = float(counter_anchor.get("depth") or counter_anchor.get("size", [0.8, 0.6, 0.9])[1])
            side_sign = 1.0
            if sink_pos is not None and sink_pos[0] > counter_pos[0]:
                side_sign = -1.0
            assignment_name = str(assignment.get("sim_object_name") or "")
            counter_y_offset = _category_y_offset(assignment_name, step_size=0.1)
            target = _world_offset(
                counter_anchor,
                [
                    side_sign * counter_width * 0.44,
                    _clamp(-counter_depth * 0.42 + counter_y_offset, -counter_depth * 0.42, counter_depth * 0.25),
                    place_z - counter_pos[2],
                ],
            )
            if target is not None:
                return {
                    "zone": "right_counter",
                    "target_pos": [float(target[0]), float(target[1]), float(target[2])],
                    "source_assignment": assignment,
                    "anchor": "counter",
                    "placement_rule": "spread_counter_edge",
                }

    zone = "handled" if is_handled else "plain"
    target_x = min(xs) - float(zone_margin) if is_handled else max(xs) + float(zone_margin)
    target_y = sum(ys) / float(len(ys))

    return {
        "zone": zone,
        "target_pos": [float(target_x), float(target_y), float(place_z)],
        "source_assignment": assignment,
    }
