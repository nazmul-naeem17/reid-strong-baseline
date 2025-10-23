# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # Check if DATASETS.NAMES is a tuple or list and is not empty
    if isinstance(cfg.DATASETS.NAMES, (tuple, list)) and len(cfg.DATASETS.NAMES) >= 1:
        # --- MODIFIED LINE ---
        # Pass the first dataset name string (e.g., 'market1501') from the tuple/list
        dataset = init_dataset(cfg.DATASETS.NAMES[0], root=cfg.DATASETS.ROOT_DIR)
        # ---------------------
    else:
        # Handle cases where NAMES might be unexpectedly empty or not a sequence
        # Or potentially handle multi-dataset logic if cfg.DATASETS.NAMES has multiple elements
        # For now, we'll raise an error or default if appropriate
        raise ValueError("cfg.DATASETS.NAMES is not a valid sequence or is empty.")
        # If supporting multi-dataset:
        # dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR) # Keep original multi-dataset logic if needed

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes
