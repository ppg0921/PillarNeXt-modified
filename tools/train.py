from trainer.trainer import Trainer
from det3d.datasets.loader.build_loader import build_dataloader
from torch.nn.parallel import DistributedDataParallel
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import logging
import os

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path='../configs/experiments')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # distributed training
    distributed = False
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://",
                                             world_size=world_size, rank=global_rank)

    # init logger before other steps
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    if distributed and local_rank != 0:
        logger.setLevel("ERROR")
    logger.info("Distributed training: {}".format(distributed))
    logger.info(
        f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    train_dataset = instantiate(cfg.data.train_dataset)
    print(f"[DEBUG] Train dataset size: {len(train_dataset)}")

# Debug print: check a few samples
    # for idx in range(3):
    #     sample = train_dataset[idx]
    #     points = sample.get('points', None)
    #     gt_boxes = sample.get('gt_boxes', None)
    #     print(f"[DEBUG] Sample {idx}:")
    #     if points is not None:
    #         print(f" - Points shape: {points.shape}")
    #     else:
    #         print(" - No points found")

    #     if gt_boxes is not None:
    #         print(f" - GT boxes: {len(gt_boxes)} items")
    #         if len(gt_boxes) > 0:
    #             print(f"   - First box: {gt_boxes[0]}")
    #     else:
    #         print(" - No GT boxes")
    train_dataloader = build_dataloader(train_dataset, **cfg.dataloader.train)
    # —– DEBUG: dump one batch and exit —–
    batch = next(iter(train_dataloader))
    print("=== BATCH DEBUG ===")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k}: tensor, shape={v.shape}")
        elif isinstance(v, list):
            # usually lists of varying-length Tensors
            print(f"{k}: list, len={len(v)}, first elem shape={v[0].shape if len(v)>0 and hasattr(v[0],'shape') else 'N/A'}")
        else:
            print(f"{k}: {type(v)}")
    # import sys; sys.exit(0)
    if 'val_dataset' in cfg.data:
        val_dataset = instantiate(cfg.data.val_dataset)
        val_dataloader = build_dataloader(val_dataset, **cfg.dataloader.val)
    else:
        val_dataloader = None

    # build model
    model = instantiate(cfg.model)
    print("[CONFIG CHECK] voxel_size passed to PillarFeatureNet:", cfg.model.reader.voxel_size)

    if distributed:
        if cfg.model.sync_batchnorm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DistributedDataParallel(
            model.cuda(local_rank), device_ids=[local_rank],
            output_device=local_rank, find_unused_parameters=False,)
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.scheduler, optimizer=optimizer, steps_per_epoch=len(
        train_dataloader), _recursive_=False)

    trainer = Trainer(
        model, train_dataloader, val_dataloader, optimizer, lr_scheduler, logger=logger, **cfg.trainer)

    if 'resume_from' in cfg:
        trainer.resume(cfg.resume_from)

    if 'load_from' in cfg:
        trainer.load_checkpoint(cfg.load_from)

    trainer.fit()


if __name__ == "__main__":
    main()
