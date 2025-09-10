import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import PointCloud, LidarPointCloud, Box
from numba import njit, prange
from numba import types
import os
import tempfile

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

def atomic_save_npz(path, **arrays):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), prefix='.tmp_', delete=False) as f:
        tmp_name = f.name 
        np.savez_compressed(f, **arrays)
    os.replace(tmp_name, path)


@njit(parallel=True, fastmath=True)
def fill_missing_with_local_median(depth_base, missing_idx_y, missing_idx_x, window):
    H, W = depth_base.shape
    half = window // 2
    out = np.full(missing_idx_y.shape[0], np.nan, np.float32)
    buf = np.empty(window * window, np.float32)  # reusable local buffer

    for i in prange(missing_idx_y.shape[0]):
        py = missing_idx_y[i]
        px = missing_idx_x[i]

        y0 = 0 if py - half < 0 else py - half
        y1 = H if py + half + 1 > H else py + half + 1
        x0 = 0 if px - half < 0 else px - half
        x1 = W if px + half + 1 > W else px + half + 1

        cnt = 0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                v = depth_base[yy, xx]
                if np.isfinite(v):
                    buf[cnt] = v
                    cnt += 1

        if cnt > 0:
            # median of buf[:cnt]
            tmp = np.sort(buf[:cnt])
            if (cnt & 1) == 1:
                out[i] = tmp[cnt // 2]
            else:
                out[i] = 0.5 * (tmp[cnt // 2 - 1] + tmp[cnt // 2])

    return out

def box_count(mask: np.ndarray, window: int) -> np.ndarray:
    """Fast neighborhood counts via integral image (O(HW))."""
    h = window // 2
    # pad with zeros
    padded = np.pad(mask.astype(np.int32), ((h, h), (h, h)), mode="constant")
    I = padded.cumsum(0).cumsum(1)
    # sum over each window centered at original pixels:
    # counts[y,x] = sum of padded[y:y+2h+1, x:x+2h+1]
    counts = I[2*h:, 2*h:] - I[:-2*h, 2*h:] - I[2*h:, :-2*h] + I[:-2*h, :-2*h]
    return counts

def build_depth_map(
    p_points: np.ndarray,
    depths_cam: np.ndarray,
    H_img: int,
    W_img: int,
    inst_map: np.ndarray,
    window: int = 11,
    far_depth: float = 100.0
) -> np.ndarray:
    """
    Create a per-pixel depth map for an image using projected LiDAR points.

    Args:
        p_points: (2, N) projected pixel coords (float); p_points[0]=x, p_points[1]=y.
        depths_cam: (N,) LiDAR depths along camera Z (positive forward).
        H_img, W_img: image height/width.
        inst_map: (H_img, W_img) integer instance IDs (0 = background).
        window: odd integer kernel size for neighbor search (default 11).
        far_depth: depth to use for background with no info (default 100.0).

    Returns:
        depth_map: (H_img, W_img) float32 depth map.
    """
    assert p_points.shape[0] == 2, "p_points must be shape (2, N)"
    assert depths_cam.ndim == 1 and depths_cam.shape[0] == p_points.shape[1], \
        "depths_cam must be length N (matching p_points)"
    assert inst_map.shape == (H_img, W_img), "inst_map must be H×W"
    assert window % 2 == 1 and window >= 1, "window must be an odd >=1"

    # ------------- Step 0: prepare inputs & masks -------------
    # Clip to image bounds; keep only finite, positive depths.
    xs = np.floor(p_points[0]).astype(np.int32)
    ys = np.floor(p_points[1]).astype(np.int32)

    valid = (
        (xs >= 0) & (xs < W_img) &
        (ys >= 0) & (ys < H_img) &
        np.isfinite(depths_cam) & (depths_cam > 0)
    )
    if not np.any(valid):   # no valid points, fill every pixel with far_depth
        return np.full((H_img, W_img), far_depth, dtype=np.float32)

    xs = xs[valid]
    ys = ys[valid]
    d  = depths_cam[valid]

    # ------------- Step 1: median per pixel from direct LiDAR hits -------------
    depth_map = np.full((H_img, W_img), np.nan, dtype=np.float32)

    # Group depths by pixel using sorted indices for efficient median computation
    pix_ids = ys.astype(np.int64) * W_img + xs.astype(np.int64)
    order = np.argsort(pix_ids, kind="mergesort")  # stable
    pix_sorted = pix_ids[order] # sorted pixel IDs
    depth_sorted = d[order] # depths sorted by pixel ID

    # Find segment boundaries
    boundaries = np.flatnonzero(np.diff(pix_sorted)) + 1      # the first index of a new pixel ID
    starts = np.r_[0, boundaries]
    ends   = np.r_[boundaries, pix_sorted.size]
    lengths = ends - starts

    groups = pix_sorted[starts]
    py = (groups // W_img).astype(np.int64)
    px = (groups %  W_img).astype(np.int64)
    
    mid = lengths // 2
    odd = (lengths & 1) == 1
    even = ~odd
    idx_odd = starts[odd] + mid[odd]
    depth_map[py[odd], px[odd]] = depth_sorted[idx_odd]
    if np.any(even):
        idx1 = starts[even] + mid[even] - 1
        idx2 = starts[even] + mid[even]
        med_even = (depth_sorted[idx1] + depth_sorted[idx2]) * 0.5
        depth_map[py[even], px[even]] = med_even
    # Preserve a mask of pixels that got depth from step 1
    step1_mask = np.isfinite(depth_map)

    # ------------- Step 2: per-instance medians from step-1 pixels only -------------
    inst_median = {0: float(far_depth)}  # background -> far
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        msk = (inst_map == iid) & step1_mask
        inst_median[int(iid)] = float(np.median(depth_map[msk])) if np.any(msk) else np.nan

    # ------------- Step 3: neighbor median (11×11 by default) using ONLY step-1 depths -------------
    # We'll compute for pixels still missing depth after step 1.
    missing = ~step1_mask
    if np.any(missing):
        # Count valid neighbors in the window using integral images (fast)
        counts = box_count(step1_mask, window)  # per-pixel counts of valid step-1 neighbors
        need_neighbors = missing & (counts > 0)

        if np.any(need_neighbors):
            base = depth_map.copy()  # only step-1 values are finite
            my, mx = np.where(need_neighbors)
            fills = fill_missing_with_local_median(base, my.astype(np.int64), mx.astype(np.int64), window)
            depth_map[my, mx] = fills

    # ------------- Step 4: final fill by background/instance fallback -------------
    missing = ~np.isfinite(depth_map)
    if np.any(missing):
        inst_ids_missing = inst_map[missing].astype(int)
        inst_fallback = np.array([inst_median.get(iid, np.nan) for iid in inst_ids_missing], dtype=np.float32)
        depth_map[missing] = np.where(np.isfinite(inst_fallback), inst_fallback, far_depth).astype(np.float32)

    return depth_map.astype(np.float32)