from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

cfg = 'configs/nuimages/mask-rcnn_r50_fpn_coco-2x_1x_nuim.py'
ckpt = 'mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'
model = init_detector(cfg, checkpoint=ckpt, device='cuda:0')  # uses load_from in cfg

res = inference_detector(model, 'test_img.jpg')

vis = VISUALIZERS.build(model.cfg.visualizer)
vis.dataset_meta = model.dataset_meta
img = mmcv.imread('test_img.jpg')
img = mmcv.imconvert(img, 'bgr', 'rgb')
vis.add_datasample('vis', img, data_sample=res, draw_gt=False, out_file='result_inst.jpg')
