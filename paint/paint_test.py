from mmdet.apis import init_detector, inference_detector
import mmcv

# Path to your config
config = 'mask-rcnn_r50_fpn_coco-2x_1x_nuim.py'

# You can use the load_from field in config, or explicitly specify the checkpoint here
checkpoint = None  # will use load_from in config
# checkpoint = 'checkpoints/mask_rcnn_r50_fpn_2x_coco.pth'  # if you downloaded manually

# Instantiate model
model = init_detector(config, checkpoint, device='cuda:0')

# Run inference on an image
result = inference_detector(model, 'test_img.jpg')

# Show and save results
model.show_result('test_img.jpg', result, out_file='result.jpg')
