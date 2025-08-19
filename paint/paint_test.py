# paint_test_v2.py  (MMDet 2.x)
from mmdet.apis import init_detector, inference_detector
import mmcv

CONFIG = 'configs/nuimages/mask-rcnn_r50_fpn_coco-2x_1x_nuim.py'
CKPT   = 'mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'
IMG    = 'test_img.jpg'

# Build model (MMDet 2.x)
model = init_detector(CONFIG, CKPT, device='cuda:0')

# Inference
result = inference_detector(model, IMG)

# Visualize & save (MMDet 2.x)
model.show_result(IMG, result, out_file='result_inst.jpg', score_thr=0.3)
print('saved -> result_inst.jpg')
