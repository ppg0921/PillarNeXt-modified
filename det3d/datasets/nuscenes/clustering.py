import numpy as np
from cuml.cluster import DBSCAN as cuDBSCAN
# import cupy as cp

def _to_numpy_labels(labels):
    """Convert cuML outputs (CuPy/cudf) or sklearn outputs to a NumPy ndarray."""
    # CuPy ndarray -> NumPy
    try:
        import cupy as cp
        if isinstance(labels, cp.ndarray):
            return cp.asnumpy(labels)
    except Exception:
        pass

    # cuDF Series/DataFrame -> NumPy
    try:
        import cudf
        if isinstance(labels, (cudf.Series, cudf.DataFrame)):
            return labels.to_numpy()
    except Exception:
        pass

    # Already NumPy-like
    return np.asarray(labels)

def filter_paint_feats_by_dbscan_per_instance(
    pc_lidar: np.ndarray,
    inst_ids: np.ndarray,
    paint_feats: np.ndarray,
    inst_to_indices: dict[int, np.ndarray],
    eps: float = 0.4,
    min_samples: int = 5,
    metric: str = "euclidean",
    include_background: bool = False,
    treat_noise_as_cluster: bool = False,
):
    """
    For each instance id, run DBSCAN on the instance's LiDAR xyz, keep ONLY the cluster
    whose closest point to the origin is nearest, and zero out paint_feats for all
    other clusters (and optionally noise).

    Parameters
    ----------
    pc_lidar : (Nf, >=3) float
        LiDAR points filtered to camera FOV. Columns 0:3 must be xyz.
    inst_ids : (Nf,) int
        Per-point instance ids (0 = background).
    paint_feats : (Nf, K) float
        Class scores/features per point. This array is modified in-place.
    inst_to_indices : dict[int, np.ndarray]
        Mapping: instance id -> 1D array of indices into pc_lidar/paint_feats/inst_ids.
    eps : float
        DBSCAN neighborhood radius (meters).
    min_samples : int
        DBSCAN min_samples.
    metric : str
        Distance metric for DBSCAN (e.g., "euclidean").
    include_background : bool
        If True, also process iid==0; otherwise, skip background.
    treat_noise_as_cluster : bool
        If True, consider DBSCAN noise label (-1) as its own "cluster" when
        selecting the nearest cluster; if False, all noise points are zeroed.

    Returns
    -------
    cluster_labels_out : np.ndarray, shape (Nf,), dtype=int32
        Per-point cluster label within its instance. -1 for noise or unprocessed.
        Labels are NOT global across instances; they restart at 0 per instance.
    """
    # Try to import scikit-learn here (local import to avoid global dependency)
    try:
        from sklearn.cluster import DBSCAN
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for DBSCAN. Please install it, e.g.:\n"
            "  pip install scikit-learn\n\n"
            f"Import error: {e}"
        )

    Nf = pc_lidar.shape[0]    # number of lidar point clouds
    cluster_labels_out = np.full((Nf,), -1, dtype=np.int32)

    for iid, idxs in inst_to_indices.items():
        if idxs.size == 0:
            continue
        if iid == 0 and not include_background:
            # skip background unless requested
            continue

        pts_xyz = pc_lidar[idxs, :3]
        if pts_xyz.shape[0] < max(1, min_samples):
            continue
            # Too few points: keep the single closest point's "cluster" (degenerate),
            # zero out paint_feats for the rest.
            dists = np.linalg.norm(pts_xyz, axis=1)
            keep_idx_local = int(np.argmin(dists))
            # mark the kept one as cluster 0, rest stay -1
            cluster_labels_out[idxs[keep_idx_local]] = 0
            # zero others
            if pts_xyz.shape[0] > 1:
                zero_idxs = np.delete(idxs, keep_idx_local)
                paint_feats[zero_idxs, :] = 0.0
            continue

        # DBSCAN
        # db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        # labels = cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_xyz.astype(np.float32)).get()
        try:
            # Prefer giving cuML a CuPy array for speed (if CuPy is present)
            try:
                import cupy as cp
                pts_dev = cp.asarray(pts_xyz, dtype=cp.float32)
                use_cupy = True
            except Exception:
                pts_dev = pts_xyz.astype(np.float32, copy=False)
                use_cupy = False

            from cuml.cluster import DBSCAN as cuDBSCAN
            labels = cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_dev)
            labels = _to_numpy_labels(labels).astype(np.int32, copy=False)
            return labels
        except Exception:
            # Fallback: sklearn (CPU)
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(
                pts_xyz.astype(np.float32, copy=False)
            )
            return labels.astype(np.int32, copy=False)
        # Record raw labels (per instance)
        cluster_labels_out[idxs] = labels

        # Determine which cluster to keep
        unique_labels = np.unique(labels)
        # Optionally drop noise from consideration
        label_pool = unique_labels if treat_noise_as_cluster else unique_labels[unique_labels != -1]
        if label_pool.size == 0:
            # All noise: keep the single closest point to origin, zero the rest
            dists = np.linalg.norm(pts_xyz, axis=1)
            keep_idx_local = int(np.argmin(dists))
            # mark the kept one as cluster 0 (normalize label)
            cluster_labels_out[idxs] = -1  # reset
            cluster_labels_out[idxs[keep_idx_local]] = 0
            zero_idxs = np.delete(idxs, keep_idx_local)
            paint_feats[zero_idxs, :] = 0.0
            continue

        # For each candidate cluster, compute the minimum distance-to-origin among its points
        # and keep the cluster with the smallest such minimum (closest to origin).
        dists = np.linalg.norm(pts_xyz, axis=1)
        best_label = None
        best_min_dist = np.inf
        for lbl in label_pool:
            members = (labels == lbl)
            if not np.any(members):
                continue
            min_dist = dists[members].min()
            if min_dist < best_min_dist:
                best_min_dist = min_dist
                best_label = lbl

        # Keep best_label cluster; zero out others (and noise if not treated as a cluster)
        if best_label is None:
            # Should not happen (guard), but be safe
            zero_idxs = idxs
        else:
            keep_mask = (labels == best_label)
            zero_mask = ~keep_mask
            if not treat_noise_as_cluster:
                zero_mask |= (labels == -1)
            zero_idxs = idxs[zero_mask]

            # Normalize labels inside this instance (optional):
            # Set the kept cluster label to 0, others remain as-is or -1
            cluster_labels_out[idxs[keep_mask]] = 0

        # Zero out paint for all points not in the kept cluster
        if zero_idxs.size > 0:
            paint_feats[zero_idxs, :] = 0.0

    return cluster_labels_out
