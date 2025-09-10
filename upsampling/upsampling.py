import argparse
import os
import torch
from det3d.datasets.nuscenes.nusc import NuScenesDataset

def main():
    parser = argparse.ArgumentParser(description="Paint nuScenes cameras with MMDetection Mask R-CNN (batched, GPU merge, resumable)")
    parser.add_argument('--nusc_root', required=True, help='nuScenes dataroot (contains samples/ sweeps/ etc.)')
    parser.add_argument('--painted_path', required=True, help='Directory for .npz paint files')
    parser.add_argument('--depth_path', required=True, help='Output directory for .npz depth files')
    parser.add_argument('--version', default='v1.0-trainval', help='nuScenes version tag')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp', action='store_true', help='Use torch.cuda.amp.autocast for faster inference')
    parser.add_argument('--nsweeps', type=int, default=10, help='Number of sweeps to fuse, max 10 for nuScenes')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='Skip if output npz already exists')

    args = parser.parse_args()

    pipeline = ['load_pointcloud', 'load_box3d']
    trainset = NuScenesDataset(
      loading_pipelines=pipeline,
      nsweeps=args.nsweeps,
      root_path=args.nusc_root,
      info_path="infos_train_10sweeps_withvelo_filterZero.pkl",
      version=args.version,
      fuse_camera=True,
      cam_name='CAM_FRONT',
      padding=False,
      painted_path=args.painted_path,
      depth_path=args.depth_path
    )
    valset = NuScenesDataset(
      loading_pipelines=pipeline,
      nsweeps=args.nsweeps,
      root_path=args.nusc_root,
      info_path="infos_val_10sweeps_withvelo_filterZero.pkl",
      version=args.version,
      fuse_camera=True,
      cam_name='CAM_FRONT',
      padding=False,
      painted_path=args.painted_path,
      depth_path=args.depth_path
    )
    trainset.upsampling_all()
    valset.upsampling_all()

if __name__ == "__main__":
    main()