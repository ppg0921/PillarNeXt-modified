#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes

def load_depth(npz_path: str) -> np.ndarray:
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Depth npz not found: {npz_path}")
    data = np.load(npz_path)
    # support either named 'depth' or default 'arr_0'
    return (data['depth'] if 'depth' in data else data['arr_0']).astype(np.float32)

def colorize_depth_near_red_far_blue(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    Map depth to BGR colors (near=red, far=blue) using JET with an inversion.
    Returns uint8 BGR image same HxW.
    """
    mask = np.isfinite(depth)
    if not np.any(mask):
        # no valid pixels: return neutral grayscale
        out = np.zeros((*depth.shape, 3), dtype=np.uint8)
        return out

    finite = depth[mask]
    if vmin is None or vmax is None:
        # robust range
        vmin = np.percentile(finite, 1.0) if vmin is None else vmin
        vmax = np.percentile(finite, 99.0) if vmax is None else vmax
        if vmax <= vmin:
            vmax = vmin + 1e-6

    depth_clamped = np.clip(depth, vmin, vmax)
    norm = (depth_clamped - vmin) / (vmax - vmin)  # 0=near .. 1=far
    inv = 1.0 - norm                               # invert so near->1, far->0 for JET
    depth_u8 = (inv * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)  # BGR
    # make invalid pixels transparent-ish by setting them to black (we'll blend with original)
    colored[~mask] = 0
    return colored

def overlay_depth_on_image(img_bgr: np.ndarray, depth_colored_bgr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    # Blend only where depth has nonzero color (valid)
    valid = depth_colored_bgr.sum(axis=2) > 0
    out = img_bgr.copy()
    if np.any(valid):
        blended = (alpha * depth_colored_bgr + (1.0 - alpha) * img_bgr).astype(np.uint8)
        out[valid] = blended[valid]
    return out

def main():
    ap = argparse.ArgumentParser(description="Overlay saved depth map on a nuScenes camera image (near=red, far=blue).")
    ap.add_argument("--nusc_root", required=True, help="nuScenes dataroot")
    ap.add_argument("--version", default="v1.0-trainval", help="nuScenes version (e.g., v1.0-mini)")
    ap.add_argument("--depth_path", required=True, help="Directory containing <token>.npz (with 'depth')")
    ap.add_argument("--alpha", type=float, default=0.6, help="Overlay opacity (0..1)")
    ap.add_argument("--vmin", type=float, default=None, help="Depth min for color scaling (meters)")
    ap.add_argument("--vmax", type=float, default=None, help="Depth max for color scaling (meters)")
    ap.add_argument("--out", default=None, help="Output image path (defaults to <depth_path>/<token>_overlay.jpg)")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    cam_token = "0a6129c27f5643cc8c4ae011de763a1c"
    sd = nusc.get("sample_data", cam_token)
    img_path = os.path.join(nusc.dataroot, sd["filename"])
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    H_img, W_img = img_bgr.shape[:2]

    npz_path = os.path.join(args.depth_path, f"{cam_token}.npz")
    depth_map = load_depth(npz_path)

    # If depth shape mismatches, resize to image size (nearest to preserve medians)
    if depth_map.shape != (H_img, W_img):
        depth_map = cv2.resize(depth_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    depth_col_bgr = colorize_depth_near_red_far_blue(depth_map, vmin=args.vmin, vmax=args.vmax)
    overlay = overlay_depth_on_image(img_bgr, depth_col_bgr, alpha=args.alpha)

    # annotate scale
    txt = "near=red  far=blue"
    cv2.putText(overlay, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    out_path = args.out or os.path.join(f"./upsampling/visualization", f"{cam_token}_overlay.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)
    print(f"Saved overlay: {out_path}")

if __name__ == "__main__":
    main()
