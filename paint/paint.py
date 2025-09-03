#!/usr/bin/env python3
# paint_nuscenes_mmdet.py
import os
import argparse
from tqdm import tqdm
import numpy as np
import mmcv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

# nuImages / nuScenes 10 detection classes (order matters!)
NUIM_CLASSES = (
    'car','truck','trailer','bus','construction_vehicle',
    'pedestrian','motorcycle','bicycle','traffic_cone','barrier'
)
K = len(NUIM_CLASSES)

def convert_pretrained_to_init_cfg(node):
    if isinstance(node, dict):
        if 'pretrained' in node:
            node['init_cfg'] = dict(type='Pretrained', checkpoint=node.pop('pretrained'))
        for v in node.values():
            convert_pretrained_to_init_cfg(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            convert_pretrained_to_init_cfg(v)

def build_model(cfg_path, ckpt, device='cuda:0'):
    cfg = Config.fromfile(cfg_path)
    cfg.default_scope = 'mmdet'
    convert_pretrained_to_init_cfg(cfg)

    # Remove legacy train/test_cfg if present
    for k in ('train_cfg', 'test_cfg'):
        if hasattr(cfg, k):
            setattr(cfg, k, None)

    # Ensure 3.x data preprocessor
    cfg.model.setdefault('data_preprocessor', dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ))

    # Minimal test dataloader/pipeline so inference_detector can prepare inputs
    cfg.test_dataloader = dict(
        batch_size=1, num_workers=2, persistent_workers=False,
        dataset=dict(
            type='CocoDataset',
            metainfo=dict(classes=NUIM_CLASSES),
            data_root='',
            ann_file='',
            data_prefix=dict(img=''),
            lazy_init=True,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                dict(type='Pad', size_divisor=32),
                dict(type='PackDetInputs',
                     meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
            ]
        )
    )

    model = init_detector(cfg, checkpoint=ckpt, device=device)
    if not hasattr(model, 'dataset_meta') or 'classes' not in model.dataset_meta:
        model.dataset_meta = dict(classes=NUIM_CLASSES)
    return model

@torch.no_grad()
def gpu_instances_to_semantic(result, H, W, K, score_thresh=0.5, mask_thresh=0.5, device='cuda'):
    """
    GPU pipeline:
      - labels/scores/masks from result.pred_instances
      - upsample masks to (H,W) on GPU
      - per-class max of (mask * score)
    Returns: (H, W, K) float32 on CPU
    """
    S = torch.zeros((K, H, W), device=device, dtype=torch.float32)
    
    inst_id = torch.zeros((H, W), dtype=torch.int32, device=device)  # per pixel instance id map

    if not hasattr(result, 'pred_instances'):
        print("[WARNING] No pred_instances found")
        return S.permute(1,2,0).cpu(), inst_id.cpu()

    inst = result.pred_instances
    labels = inst.get('labels', None)
    scores = inst.get('scores', None)
    masks  = inst.get('masks', None)

    if labels is None or scores is None or masks is None or len(labels) == 0:
        return S.permute(1,2,0).cpu(), inst_id.cpu()

    labels = labels.to(device)
    scores = scores.to(device)

    # BitmapMasks -> dense tensor
    if hasattr(masks, 'to_tensor'):
        masks_t = masks.to_tensor(dtype=torch.float32, device=device)  # (N, h, w)
    else:
        m = masks
        if torch.is_tensor(m):
            m = m.to(device=device, dtype=torch.float32)
            if m.ndim == 4 and m.shape[1] == 1:
                m = m[:, 0]
            masks_t = m
        else:
            masks_t = torch.from_numpy(np.array(masks)).to(device=device, dtype=torch.float32)
            if masks_t.ndim == 4 and masks_t.shape[1] == 1:
                masks_t = masks_t[:, 0]

    if masks_t.numel() == 0:
        return S.permute(1,2,0).cpu(), inst_id.cpu()

    keep = scores >= score_thresh
    if keep.sum() == 0:
        return S.permute(1,2,0).cpu(), inst_id.cpu()

    labels = labels[keep]
    scores = scores[keep]
    masks_t = masks_t[keep]  # (Nf, h, w)
    Nf = masks_t.shape[0]   # number of instances

    # Upsample to (H, W) in one go
    masks_up = F.interpolate(
        masks_t.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(1)  # (Nf, H, W)

    # Classwise max pooling
    for c in range(K):
        idx = (labels == c)
        if idx.any():
            sc = scores[idx].view(-1, 1, 1)
            S[c] = torch.max(masks_up[idx] * sc, dim=0).values

    
    # instance id map
    if mask_thresh is not None:
        masks_bin = (masks_up >= mask_thresh).float()
    else:
        masks_bin = (masks_up > 0).float()
    
    w = scores.view(Nf, 1, 1).to(masks_bin.dtype)
    inst_weighted = masks_bin * w

    max_vals, max_idx = torch.max(inst_weighted, dim=0)
    inst_id = torch.where(max_vals > 0, max_idx+1, torch.tensor(0, device=device))
    inst_id = inst_id.to(torch.int32)

    return S.permute(1,2,0).cpu(), inst_id.cpu()

def atomic_save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)

class NuscCamKeyframeDataset(Dataset):
    def __init__(self, items):
        # items: list of (cam_token, img_path)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cam_token, img_path = self.items[idx]
        img = mmcv.imread(img_path)
        H, W = int(img.shape[0]), int(img.shape[1])
        return cam_token, img_path, H, W

def build_items(nusc, dataroot, cams, only_keyframes, out_dir, skip_existing=True, check_files=True, limit=None):
    """Return list of (cam_token, img_path) filtered to existing files and (optionally) not-yet-painted."""
    all_sds = [sd for sd in nusc.sample_data if sd['channel'] in cams and (sd['is_key_frame'] or not only_keyframes)]
    total = len(all_sds)
    items, missing, skipped = [], 0, 0
    for sd in all_sds:
        cam_token = sd['token']
        img_path = os.path.join(dataroot, sd['filename'])
        if check_files and not os.path.exists(img_path):
            missing += 1
            continue
        out_path = os.path.join(out_dir, f'{cam_token}.npz')
        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        items.append((cam_token, img_path))
        if limit is not None and len(items) >= limit:
            break
    return items, total, missing, skipped

def main():
    parser = argparse.ArgumentParser(description="Paint nuScenes cameras with MMDetection Mask R-CNN (batched, GPU merge, resumable)")
    parser.add_argument('--nusc_root', required=True, help='nuScenes dataroot (contains samples/ sweeps/ etc.)')
    parser.add_argument('--version', default='v1.0-trainval', help='nuScenes version tag')
    parser.add_argument('--cfg', default='configs/nuimages/mask-rcnn_r50_fpn_coco-2x_1x_nuim.py', help='MMDet config path for nuImages Mask R-CNN')
    parser.add_argument('--ckpt', default='mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth', help='Checkpoint path (.pth)')
    parser.add_argument('--out_dir', required=True, help='Output dir for .npz paint files')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score_thresh', type=float, default=0.5)
    parser.add_argument('--mask_thresh', type=float, default=0.5)
    parser.add_argument('--dtype', default='fp16', choices=['fp16','fp32'])
    parser.add_argument('--cams', nargs='+',
                        default=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'])
    parser.add_argument('--only_keyframes', action='store_true', default=True,
                        help='Process only keyframe camera images (default True)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp', action='store_true', help='Use torch.cuda.amp.autocast for faster inference')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip if output npz already exists')
    parser.add_argument('--no_check_files', action='store_true', help='Do not check image path existence')
    parser.add_argument('--limit', type=int, default=None, help='Process at most N images (after filtering)')
    parser.add_argument('--dry_run', action='store_true', help='Only report counts; do not run inference')
    args = parser.parse_args()

    # Lazy import NuScenes
    from nuscenes.nuscenes import NuScenes

    torch.backends.cudnn.benchmark = True

    model = build_model(args.cfg, args.ckpt, device=args.device)
    device = args.device if isinstance(args.device, str) else 'cuda'

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)

    # Build filtered item list (existing files + not-yet-painted)
    items, total, missing, skipped = build_items(
        nusc, args.nusc_root, args.cams, args.only_keyframes,
        args.out_dir, skip_existing=args.skip_existing,
        check_files=(not args.no_check_files), limit=args.limit
    )
    print(f"Total cam sample_data: {total}")
    print(f"Missing image files  : {missing}")
    print(f"Already painted (skipped): {skipped}")
    print(f"To process now       : {len(items)}")

    if args.dry_run:
        print("Dry run — exiting before inference.")
        return

    ds = NuscCamKeyframeDataset(items)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda x: x  # keep as list of tuples
    )

    use_fp16 = (args.dtype == 'fp16')

    for batch in tqdm(loader, desc='Painting (batched)', dynamic_ncols=True):
        cam_tokens = [b[0] for b in batch]
        img_paths  = [b[1] for b in batch]
        Hs         = [b[2] for b in batch]
        Ws         = [b[3] for b in batch]

        # Try true batch inference (list input). If unsupported, fallback per-image.
        try:
            if args.amp:
                with torch.cuda.amp.autocast(enabled=True):
                    results = inference_detector(model, img_paths)
            else:
                results = inference_detector(model, img_paths)
            if not isinstance(results, (list, tuple)):
                results = [results]
            assert len(results) == len(img_paths)
        except Exception:
            results = []
            for p in img_paths:
                if args.amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        r = inference_detector(model, p)
                else:
                    r = inference_detector(model, p)
                results.append(r)

        # Save each result if its npz doesn't already exist (re-check in case of concurrent runs)
        for cam_token, img_path, H, W, res in zip(cam_tokens, img_paths, Hs, Ws, results):
            out_path = os.path.join(args.out_dir, f'{cam_token}.npz')
            if args.skip_existing and os.path.exists(out_path):
                continue

            try:
                S, inst_id = gpu_instances_to_semantic(
                    res, H, W, K=K, score_thresh=args.score_thresh, mask_thresh=args.mask_thresh, device=device if 'cuda' in device else 'cpu'
                )
                S = S.numpy()
                inst_id = inst_id.numpy()
                if use_fp16:
                    S = S.astype(np.float16)
                atomic_save_npz(out_path, scores=S, inst_id=inst_id, height=H, width=W, class_names=np.array(NUIM_CLASSES))
            except Exception as e:
                # Log and continue; you can also write a .err file next to the image if you want
                print(f"[WARN] Failed on token {cam_token}: {e}")
        
    print("Painting Finished")

if __name__ == '__main__':
    main()
