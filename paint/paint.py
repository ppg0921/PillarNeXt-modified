# painter.py
import os, cv2, numpy as np, torch
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

NUIM_DET_CLASSES = [
    'car','truck','trailer','bus','construction_vehicle',
    'bicycle','motorcycle','pedestrian','traffic_cone','barrier'
]
K = len(NUIM_DET_CLASSES)

def instances_to_semantic_map(out, H, W):
    # out: torchvision maskrcnn single-image output dict
    S = np.zeros((H, W, K), dtype=np.float32)   # final semantic map
    labels = out['labels'].cpu().numpy()        # already 0..9 if you trained that way
    scores = out['scores'].cpu().numpy()
    masks  = out['masks'].cpu().numpy()[:,0]    # (N_inst, h', w')

    for lab, sc, mk in zip(labels, scores, masks):
        if sc < 0.4:  # tune
            continue
        mk = cv2.resize(mk.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        np.maximum(S[..., lab], sc * mk, out=S[..., lab])   # keep the max if overlapping
    return S


def image_transform(img_rgb_numpy):
    return F.to_tensor(img_rgb_numpy) 

@torch.no_grad()
def paint_all(nusc: NuScenes, model, device, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    for sd in tqdm(nusc.sample_data, desc="Painting cams"):
        if not sd['is_key_frame']: 
            continue
        if not sd['channel'].startswith('CAM_'):
            continue
        cam_token = sd['token']
        img_path = os.path.join(nusc.dataroot, sd['filename'])
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB (not sure)
        H, W = img.shape[:2]

        # your transforms -> tensor on device
        inp = image_transform(img).unsqueeze(0).to(device)
        out = model(inp)[0]  # Mask R-CNN output

        S = instances_to_semantic_map(out, H, W).astype(np.float16)
        np.savez_compressed(
            os.path.join(cache_dir, f"{cam_token}.npz"),
            scores=S, height=H, width=W, class_names=np.array(NUIM_DET_CLASSES)
        )


def get_model_instance_segmentation(num_classes=11, pretrained=True, weight_path=None, device="cuda"):
    # Load base Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    # Load weights if provided
    if weight_path is not None:
        checkpoint = torch.load(weight_path, map_location=device)
        if "model" in checkpoint:   # if you saved like {'model': state_dict}
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("=> Loaded weights from", weight_path)
        print("   Missing keys:", missing)
        print("   Unexpected keys:", unexpected)

    model.to(device)
    return model

if __name__ == "__main__":
    from det3d.datasets.nuscenes import NuScenesDataset
    import argparse

    parser = argparse.ArgumentParser(description="Paint the nuScenes dataset using Mask R-CNN and finetuned weights")
    parser.add_argument('--dataroot', type=str, required=True, help="Path to nuScenes data root")
    parser.add_argument('--version', type=str, default="v1.0-trainval", help="nuScenes dataset version")
    parser.add_argument('--weight_path', type=str, default=None, help="Path to the model weights file")
    parser.add_argument('--output_dir', type=str, default="../nuScene_data_root/painted_cache/", help="Directory to save painted maps")
    args = parser.parse_args()

    # Load NuScenes dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    dataset = NuScenesDataset(nusc, split='train')

    # Build your model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_instance_segmentation(num_classes=11, pretrained=True, weight_path=args.weight_path, device=device)
    model.eval()
    model.to(device)

    # Paint all images
    paint_all(nusc, model, device, cache_dir=args.output_dir)