"""Coordinate transformation functions for camera<->world conversions.

Provides pixel to 3D and world to camera coordinate transformations.
"""

import numpy as np


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
