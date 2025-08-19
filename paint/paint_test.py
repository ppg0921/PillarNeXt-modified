# paint_test_mmdet3_final.py
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

cfg_path = 'configs/nuimages/mask-rcnn_r50_fpn_coco-2x_1x_nuim.py'
ckpt     = 'mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'
img_path = 'test_img.jpg'

# nuImages 10 classes
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

# Remove old top-level train/test cfgs if present (avoid duplication)
for k in ('train_cfg', 'test_cfg'):
    if hasattr(cfg, k):
        setattr(cfg, k, None)

# ✅ Add a 3.x data preprocessor so inputs become a Tensor (not a list)
cfg.model.setdefault('data_preprocessor', dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
))

# Minimal 3.x test dataloader + pipeline (so inference_detector can prepare data)
cfg.test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=nuim_classes),
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

# Visualizer
cfg.visualizer = cfg.get('visualizer', dict(type='DetLocalVisualizer', name='vis'))

# Build + run
model = init_detector(cfg, checkpoint=ckpt, device='cuda:0')
result = inference_detector(model, img_path)

# Visualize
vis = VISUALIZERS.build(model.cfg.visualizer)
vis.dataset_meta = getattr(model, 'dataset_meta', dict(classes=nuim_classes))
img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')
vis.add_datasample('nuimages', img, data_sample=result, draw_gt=False, out_file='result_inst.jpg')
print('Saved -> result_inst.jpg')
