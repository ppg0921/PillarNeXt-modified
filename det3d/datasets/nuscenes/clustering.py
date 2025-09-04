import numpy as np
from cuml.cluster import DBSCAN as cuDBSCAN
from sklearn.cluster import DBSCAN
import cupy as cp

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
    selection_mode: str = "closest",
    cluster_dims: str = "xyz"
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
    selection_mode : str
        "closest": select the cluster whose closest point to origin is nearest.
        "largest": select the largest cluster; if tie, select the one whose closest
    Returns
    -------
    cluster_labels_out : np.ndarray, shape (Nf,), dtype=int32
        Per-point cluster label within its instance. -1 for noise or unprocessed.
        Labels are NOT global across instances; they restart at 0 per instance.
    """

    Nf = pc_lidar.shape[0]    # number of lidar point clouds
    cluster_labels_out = np.full((Nf,), -1, dtype=np.int32)
    proc_mask = (inst_ids > 0)

    if not proc_mask.any():
        return cluster_labels_out
    
    if cluster_dims == "xy":
        coords = pc_lidar[proc_mask, :2].astype(np.float32, copy=False)
    else:
        coords = pc_lidar[proc_mask, :3].astype(np.float32, copy=False)
    
    inst_sel = inst_ids[proc_mask].astype(np.int32, copy=False)     # selected instance ids
    
    alpha = 10*eps
    inst_feat = (inst_sel.astype(np.float32)*alpha).reshape(-1, 1)  # (Np, 1)
    coords_feat = np.concatenate((coords, inst_feat), axis=1) .astype(np.float32, order='C')
    
    feats_dev = cp.asarray(coords_feat, dtype=cp.float32)
    if not feats_dev.flags.c_contiguous:
        feats_dev = cp.ascontiguousarray(feats_dev)
    
    labels_dev = cuDBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(feats_dev)
    labels_np = cp.asnumpy(labels_dev).astype(np.int32, copy=False)   # (Np,)
    
    proc_idx_global = np.nonzero(proc_mask)[0]      # indices of the chosen points in the full set
    uniq_iids, inv = np.unique(inst_sel, return_inverse=True)
    inst_to_local = {iid: np.nonzero(inv == i)[0] for i, iid in enumerate(uniq_iids)}       # construct lists of local indices per instance id
    
    dists_proc = np.linalg.norm(coords, axis=1)
    

    for iid, loc in inst_to_local.items():      # each instance's index list
        if loc.size == 0:
            continue

        lbls = labels_np[loc]
        
        cand = np.unique(lbls)
        cand = cand[cand != -1] if not treat_noise_as_cluster else cand
        
        if cand.size == 0:
            # all noise
            continue
        
        best_label = None
        if selection_mode == "largest":
            best_size = -1
            best_min = np.inf
            for c in cand:
                members = (lbls == c)
                if not np.any(members):
                    continue
                cluster_size = np.sum(members)
                min_dist = float(dists_proc[loc][members].min())
                if cluster_size > best_size or (cluster_size == best_size and min_dist < best_min):
                    best_size = cluster_size
                    best_min = min_dist
                    best_label = int(c)

        else: # "closest" (default)
            best_min_dist = np.inf
            for c in cand:
                members = (lbls == c)
                if not np.any(members):
                    continue
                min_dist = float(dists_proc[loc][members].min())
                if min_dist < best_min_dist:
                    best_min_dist = min_dist
                    best_label = int(c)

        keep_mask = (lbls == best_label)
        zero_mask = ~keep_mask
        zero_mask |= (lbls == -1)
        
        global_idxs = proc_idx_global[loc]
        zero_idxs = global_idxs[zero_mask]
        
        if zero_idxs.size:
            paint_feats[zero_idxs, :] = 0.0
        
        lbls_norm = np.where(keep_mask, 0, lbls)
        labels_np[loc] = lbls_norm
    
    cluster_labels_out[proc_idx_global] = labels_np
    return cluster_labels_out

