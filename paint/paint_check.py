#!/usr/bin/env python3
# viz_paint_random_nommcv.py
import os, random, argparse
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes

# Class list (order must match how you painted)
NUIM_CLASSES = (
    'car','truck','trailer','bus','construction_vehicle',
    'pedestrian','motorcycle','bicycle','traffic_cone','barrier'
)

# A simple distinct palette (BGR)
PALETTE = np.array([
    [  0,  0,255],  # car (red)
    [  0,165,255],  # truck (orange)
    [  0,255,255],  # trailer (yellow)
    [ 60, 20,220],  # bus (purple-ish)
    [255,191,  0],  # construction_vehicle (blue-ish)
    [144,238,144],  # pedestrian (light green)
    [203,192,255],  # motorcycle (pink-ish)
    [128,128,  0],  # bicycle (olive)
    [255,  0,255],  # traffic_cone (magenta)
    [128,  0,128],  # barrier (purple)
], dtype=np.uint8)

def load_candidates(nusc, nusc_root, paint_dir, cams, only_keyframes=True):
    items = []
    for sd in nusc.sample_data:
        if sd['channel'] not in cams:
            continue
        if only_keyframes and not sd['is_key_frame']:
            continue
        token = sd['token']
        img_path = os.path.join(nusc_root, sd['filename'])
        npz_path = os.path.join(paint_dir, f'{token}.npz')
        if os.path.exists(img_path) and os.path.exists(npz_path):
            items.append((token, img_path, npz_path))
    return items

def overlay_paint(img_bgr, scores, alpha=0.5, conf_thresh=0.3):
    H, W, K = scores.shape
    label = scores.argmax(axis=-1).astype(np.int32)
    conf  = scores.max(axis=-1).astype(np.float32)

    colors = PALETTE[label % len(PALETTE)]
    mask = conf >= conf_thresh
    overlay = img_bgr.copy()
    overlay[mask] = (
        (1.0 - alpha) * img_bgr[mask].astype(np.float32) +
        alpha * colors[mask].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return overlay

def draw_legend(img_bgr, classes=NUIM_CLASSES, swatch=18, gap=6, margin=10):
    out = img_bgr.copy()
    x, y = margin, margin
    for i, name in enumerate(classes):
        color = tuple(int(c) for c in PALETTE[i % len(PALETTE)])
        cv2.rectangle(out, (x, y), (x+swatch, y+swatch), color=color, thickness=-1)
        cv2.putText(out, name, (x+swatch+gap, y+swatch-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += swatch + gap
    return out

def inst_edges_from_id(inst_id):
    H, W = inst_id.shape
    e = np.zeros((H, W), dtype=np.uint8)
    
    #right and down neighbor
    e[:, :-1] |= (inst_id[:, 1:] != inst_id[:, :-1]).astype(np.uint8)
    e[:-1, :] |= (inst_id[1:, :] != inst_id[:-1, :]).astype(np.uint8)
    bg = (inst_id == 0).astype(np.uint8)
    return e

def color_for_instance(inst_id_number):
    """
    Deterministic pseudo-random BGR color for an instance id (>=1).
    """
    # Map id -> hue in HSV, then convert to BGR
    hue = (inst_id_number * 37) % 180  # 0..179
    hsv = np.uint8([[[hue, 200, 255]]])  # S,V fixed
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return tuple(int(c) for c in bgr.tolist())

def draw_instance_edges(img_bgr, inst_id, thickness=1, per_instance_colors=False):

    out = img_bgr.copy()
    if inst_id is None:
        return out

    if not per_instance_colors:
        edges = inst_edges_from_id(inst_id)  # {0,1}
        if thickness > 1:
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges*255, kernel, iterations=1)
        else:
            edges = edges * 255
        # White edges for visibility
        out[edges > 0] = (255, 255, 255)
        return out

    # Per-instance colored contours (skip background=0)
    ids = np.unique(inst_id)
    ids = ids[ids != 0]
    for iid in ids:
        mask = (inst_id == iid).astype(np.uint8) * 255
        # Find contours on a slightly eroded mask to avoid noisy borders
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = color_for_instance(int(iid))
        for cnt in contours:
            if cnt.shape[0] < 5:
                continue
            cv2.drawContours(out, [cnt], -1, col, thickness)
    return out

def label_instance_ids(img_bgr, inst_id, top_n=10, min_area=50):
    """
    Put instance id numbers near centroids for the largest instances.
    """
    if top_n <= 0 or inst_id is None:
        return img_bgr
    out = img_bgr.copy()
    ids, counts = np.unique(inst_id, return_counts=True)
    # Remove background
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    if len(ids) == 0:
        return out
    # Sort by area desc and take top_n
    order = np.argsort(-counts)
    ids = ids[order][:top_n]
    for iid in ids:
        if (inst_id == iid).sum() < min_area:
            continue
        ys, xs = np.nonzero(inst_id == iid)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(out, f'{int(iid)}', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, f'{int(iid)}', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def main():
    parser = argparse.ArgumentParser(description='Randomly visualize painted nuScenes frames (no mmcv)')
    parser.add_argument('--nusc_root', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--paint_dir', required=True, help='Dir containing <token>.npz')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--k', type=int, default=5, help='number of samples to visualize')
    parser.add_argument('--cams', nargs='+',
        default=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'])
    parser.add_argument('--only_keyframes', action='store_true', default=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--conf_thresh', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show_instances', action='store_true', help='Overlay instance boundaries if inst_id is present')
    parser.add_argument('--per_instance_colors', action='store_true', help='Color edges by instance id (slower)')
    parser.add_argument('--inst_edge_thickness', type=int, default=1, help='Edge thickness in pixels')
    parser.add_argument('--label_instances_top', type=int, default=0, help='Label top-N largest instances with their id (0=off)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    items = load_candidates(nusc, args.nusc_root, args.paint_dir, args.cams, args.only_keyframes)
    if not items:
        print("No matching (image, npz) pairs found.")
        return

    random.seed(args.seed)
    random.shuffle(items)
    pick = items[:min(args.k, len(items))]

    print(f"Visualizing {len(pick)} samples...")
    for token, img_path, npz_path in pick:
        data = np.load(npz_path)
        S = data['scores'].astype(np.float32)
        inst_id = data['inst_id'].astype(np.int32) if 'inst_id' in data else None
        img = cv2.imread(img_path)  # BGR

        H_img, W_img = img.shape[:2]
        H_s, W_s, K = S.shape
        if (H_img, W_img) != (H_s, W_s):
            # resize each class channel
            S_resized = np.stack([
                cv2.resize(S[..., c], (W_img, H_img), interpolation=cv2.INTER_LINEAR)
                for c in range(K)
            ], axis=-1)
            S = S_resized

        if inst_id is not None and inst_id.shape[:2] != (H_img, W_img):
            inst_id = cv2.resize(inst_id, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        blend = overlay_paint(img, S, alpha=args.alpha, conf_thresh=args.conf_thresh)
        if args.show_instances and inst_id is not None:
            blend = draw_instance_edges(blend, inst_id,
                        thickness=args.inst_edge_thickness,
                        per_instance_colors=args.per_instance_colors)
            if args.label_instance_top > 0:
                blend = label_instance_ids(blend, inst_id, top_n=args.label_instances_top)
        
        blend = draw_legend(blend, classes=NUIM_CLASSES)

        out_file = os.path.join(args.out_dir, f'vis_{token}.jpg')
        cv2.imwrite(out_file, blend)
        print(f"Saved -> {out_file}")

if __name__ == '__main__':
    main()
