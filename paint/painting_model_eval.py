# eval_nuimages.py
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector  # not strictly required for runner.test()
import os

# === Paths you must set ===
cfg_path = 'configs/nuimages/mask-rcnn_r50_fpn_coco-2x_1x_nuim.py'
ckpt     = 'mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'
img_root = '../nuImages_data_root'                   # e.g., /data/nuimages
ann_val  = '../nuImages_data_root/annotations/nuimages_v1.0-val.json'
img_dir  = '../nuImages_data_root/samples'           # or the folder your images live in

nuim_classes = (
    'car','truck','trailer','bus','construction_vehicle',
    'pedestrian','motorcycle','bicycle','traffic_cone','barrier'
)

def convert_pretrained_to_init_cfg(node):
    if isinstance(node, dict):
        if 'pretrained' in node:
            node['init_cfg'] = dict(type='Pretrained', checkpoint=node.pop('pretrained'))
        for v in node.values():
            convert_pretrained_to_init_cfg(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            convert_pretrained_to_init_cfg(v)

cfg = Config.fromfile(cfg_path)
cfg.default_scope = 'mmdet'
convert_pretrained_to_init_cfg(cfg)

# Remove legacy top-level train/test cfgs if any
for k in ('train_cfg', 'test_cfg'):
    if hasattr(cfg, k):
        setattr(cfg, k, None)

# Ensure a 3.x data_preprocessor
cfg.model.setdefault('data_preprocessor', dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
))

# ----- VAL/TEST DATALOADER (point to real ann_file + images) -----
common_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

dataset_cfg = dict(
    type='CocoDataset',
    metainfo=dict(classes=nuim_classes),
    data_root='',
    ann_file=ann_val,
    data_prefix=dict(img=img_dir),
    pipeline=common_pipeline
)

cfg.test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    dataset=dataset_cfg
)

# You can reuse the same for val (some runners call .val; here we call .test())
cfg.val_dataloader = cfg.test_dataloader

# ----- EVALUATOR: COCO mAP for bbox + segm -----
evaluator_cfg = dict(
    type='CocoMetric',
    ann_file=ann_val,
    metric=['bbox', 'segm'],
    classwise=True    # optional: per-class AP table
)
cfg.test_evaluator = evaluator_cfg
cfg.val_evaluator  = evaluator_cfg

# (Optional but recommended) Make sure score_thr doesn't filter predictions prematurely
# If your config has model.test_cfg.rcnn.score_thr, set it to 0.0 for fair mAP computation.
rcnn = cfg.model.get('roi_head', cfg.model.get('roi_heads', None))
if rcnn and 'bbox_head' in rcnn:
    # newer configs store this under test_cfg; if present, lower it
    test_cfg = rcnn.get('test_cfg', None)
    if isinstance(test_cfg, dict):
        test_cfg['score_thr'] = 0.0

# Where results + logs go
cfg.load_from = ckpt
cfg.work_dir = './nuim_eval'
os.makedirs(cfg.work_dir, exist_ok=True)

# (Optional) Visualizer for sanity checks
cfg.visualizer = cfg.get('visualizer', dict(type='DetLocalVisualizer', name='vis'))

# Build runner and run test
runner = Runner.from_cfg(cfg)
# runner.test() will:
#   - build model, load ckpt
#   - iterate over cfg.test_dataloader
#   - compute CocoMetric (bbox + segm) and print a summary table
runner.test()
