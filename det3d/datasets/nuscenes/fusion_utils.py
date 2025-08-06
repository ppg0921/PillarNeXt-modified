import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import PointCloud, LidarPointCloud, Box
from numba import njit
from numba import types




@njit
def project_points(points, camera_intrinsics: np.ndarray, image_size,
                   min_dist=0.0):
    """
    Project points on to an image plane using camera intrinsics, and return
    mask of the ones that are within the image size.
    The code is taken from parts of "map_pointcloud" function in nuscenes
    devkit: https://github.com/nutonomy/nuscenes-devkit/blob/20ab9c7df6b9fb731\
    9f32ffb5758dd5005a4d2ea/python-sdk/nuscenes/nuscenes.py#L532
    Args:
        points: <np.float32: d, n> Matrix of points, where each point
         (x, y, z, ...) is along a column.
        camera_intrinsics: <np.float32: 3, 3> camera intrinsics matrix.
        image_size: (int, int) image width and height in pixels.
        min_dist: minimum distance (z) below which the mask will be false.

    Returns: <np.float32: d, m> Matrix of points, transformed and masked.
             <np.bool , n> boolean mask the size of number of input points,
             indicating whether each point was within camera FOV.

    """
    # Save the depths for filtering
    depths = points[2, :]

    viewpad = np.eye(4)
    viewpad[:camera_intrinsics.shape[0], :camera_intrinsics.shape[1]] = \
        camera_intrinsics

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    p_points = np.concatenate((points, np.ones((1, nbr_points))))
    p_points = np.dot(viewpad, p_points)
    p_points = p_points[:3, :]

    # p_points = p_points / p_points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    denom = np.empty((3, nbr_points), dtype=p_points.dtype)
    denom[0, :] = p_points[2, :]
    denom[1, :] = p_points[2, :]
    denom[2, :] = p_points[2, :]
    # denom has the same shape as p_points and contains the repeated third row

    p_points = p_points / denom

    # Indices for 2D pixel dimensions
    u = 0  # along the width axis
    v = 1  # along the height axis

    # Create a mask for choosing points that are at least a certain distance
    # away, and within the camera FOV.
    # mask = np.ones(points.shape[1], dtype=bool)
    mask = np.ones(points.shape[1], dtype=np.bool_)

    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, p_points[u, :] >= 0)
    mask = np.logical_and(mask, p_points[u, :] < image_size[0])
    mask = np.logical_and(mask, p_points[v, :] >= 0)
    mask = np.logical_and(mask, p_points[v, :] < image_size[1])

    return p_points, mask


