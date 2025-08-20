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

        blend = overlay_paint(img, S, alpha=args.alpha, conf_thresh=args.conf_thresh)
        blend = draw_legend(blend, classes=NUIM_CLASSES)

        out_file = os.path.join(args.out_dir, f'vis_{token}.jpg')
        cv2.imwrite(out_file, blend)
        print(f"Saved -> {out_file}")

if __name__ == '__main__':
    main()
