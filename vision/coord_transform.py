"""Coordinate transformation functions for camera<->world conversions.

Provides pixel to 3D and world to camera coordinate transformations.
"""

import numpy as np

try:
    from robosuite.utils.camera_utils import transform_from_pixels_to_world as robosuite_pixels_to_world
except ModuleNotFoundError:
    robosuite_pixels_to_world = None


def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Convert pixel coordinates to 3D camera coordinates.
    
    Args:
        u: Pixel x coordinate (column)
        v: Pixel y coordinate (row)
        depth: Depth value at this pixel in meters
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate
        cy: Principal point y coordinate
    
    Returns:
        np.ndarray: 3D point in camera coordinates (x, y, z) in meters
    """
    # Convert pixel to normalized camera coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.array([x, y, z])


def world_to_camera(point_world, rotation, translation):
    """
    Transform 3D point from world coordinates to camera coordinates.
    
    Args:
        point_world: 3D point in world coordinates (x, y, z)
        rotation: 3x3 rotation matrix from world to camera
        translation: 3D translation vector from world to camera
    
    Returns:
        np.ndarray: 3D point in camera coordinates (x, y, z)
    """
    point_world = np.asarray(point_world)
    # Apply rotation and translation: P_cam = R * P_world + t
    point_cam = rotation @ point_world + translation
    
    return point_cam


def camera_to_world(point_cam, rotation, translation):
    """
    Transform 3D point from camera coordinates to world coordinates.
    
    Args:
        point_cam: 3D point in camera coordinates (x, y, z)
        rotation: 3x3 rotation matrix from world to camera
        translation: 3D translation vector from world to camera
    
    Returns:
        np.ndarray: 3D point in world coordinates (x, y, z)
    """
    # Inverse transformation: P_world = R^T * (P_cam - t)
    point_cam = np.asarray(point_cam)
    rot_world = rotation.T
    trans_world = -rot_world @ translation
    
    return rot_world @ point_cam + trans_world


def pixels_to_world(pixels_rc, depth_image, camera_to_world_transform):
    """
    Convert pixel coordinates directly to world coordinates using a camera-to-world transform.

    Args:
        pixels_rc: Pixel coordinates shaped (..., 2) in (row, col) order.
        depth_image: Real-valued depth map in meters shaped (H, W) or (H, W, 1).
        camera_to_world_transform: 4x4 homogeneous camera-to-world matrix.

    Returns:
        np.ndarray: World points shaped (..., 3).
    """
    pixels_rc = np.asarray(pixels_rc, dtype=float)
    depth_map = np.asarray(depth_image, dtype=float)
    if depth_map.ndim == 2:
        depth_map = depth_map[..., None]
    camera_to_world_transform = np.asarray(camera_to_world_transform, dtype=float)

    if pixels_rc.ndim == 1:
        pixels_rc = pixels_rc[None, :]

    rows = np.clip(np.round(pixels_rc[:, 0]).astype(int), 0, depth_map.shape[0] - 1)
    cols = np.clip(np.round(pixels_rc[:, 1]).astype(int), 0, depth_map.shape[1] - 1)

    if robosuite_pixels_to_world is not None and pixels_rc.shape[0] == 1:
        return np.asarray(
            robosuite_pixels_to_world(
                pixels=pixels_rc[0],
                depth_map=depth_map,
                camera_to_world_transform=camera_to_world_transform,
            ),
            dtype=float,
        )[None, :]

    z = depth_map[rows, cols, 0]
    cam_points = np.stack(
        [
            pixels_rc[:, 1] * z,
            pixels_rc[:, 0] * z,
            z,
            np.ones_like(z),
        ],
        axis=-1,
    )
    world_points = (camera_to_world_transform @ cam_points.T).T
    return world_points[:, :3]
