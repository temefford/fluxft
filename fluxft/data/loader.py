# fluxft/data/loader.py
from __future__ import annotations
from pathlib import Path
from typing import List
import torch, datasets
from PIL import Image
from torchvision import transforms
from ..config import DataConfig
from ..utils import seed_everything
from diffusers.utils import PIL_INTERPOLATION

def default_transforms(res: int = 1200):
    return transforms.Compose(
        [
            transforms.Resize((res, res), interpolation=PIL_INTERPOLATION["lanczos"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Correct for RGB
        ]
    )

def build_dataset(cfg: DataConfig, img_size: int):
    if cfg.dataset_type == "imagefolder":
        ds = datasets.load_dataset("imagefolder", data_dir=str(cfg.data_dir), split="train")
    else:  # hf_metadata
        meta = cfg.data_dir / "metadata.json"
        ds = datasets.load_dataset("json", data_files=str(meta), split="train")
    seed_everything(42)
    if cfg.validation_split:
        ds = ds.shuffle(seed=42)
    return ds

import logging

def preprocess_fn(example, cfg: DataConfig, processor, img_size: int):
    image_root = cfg.data_dir
    img_name = str(example[cfg.image_column])
    # Try multiple extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_path = None
    for ext in extensions:
        candidate = Path(image_root) / f"{img_name}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        logging.warning(f"Image not found for {img_name} with extensions {extensions}")
        return None  # Optionally: handle missing image more gracefully
    try:
        img = Image.open(img_path).convert("RGB")
        example["pixel_values"] = processor(img)
        example["input_ids_2"] = example.get(cfg.caption_column, "")
    except Exception as e:
        logging.warning(f"Failed to load/process image {img_path}: {e}")
        return None
    return example

def build_dataloaders(cfg: DataConfig, batch_size: int, img_size: int):
    processor = default_transforms(img_size)
    ds_all = build_dataset(cfg, img_size)
    # split
    if cfg.validation_split:
        ds = ds_all.train_test_split(test_size=cfg.validation_split, seed=42)
        train_ds, val_ds = ds["train"], ds["test"]
    else:
        train_ds, val_ds = ds_all, None
    train_ds = train_ds.map(
        preprocess_fn,
        fn_kwargs=dict(cfg=cfg, processor=processor, img_size=img_size),
        remove_columns=train_ds.column_names,
    )
    # Remove None entries from train_ds
    train_ds = train_ds.filter(lambda x: x is not None)
    if val_ds:
        val_ds = val_ds.map(
            preprocess_fn,
            fn_kwargs=dict(cfg=cfg, processor=processor, img_size=img_size),
            remove_columns=val_ds.column_names,
        )
        val_ds = val_ds.filter(lambda x: x is not None)
    # fluxft/data/loader.py
    train_loader = torch.utils.data.DataLoader(
        train_ds.with_format("torch"),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    # fluxft/data/loader.py
    val_loader = None
    if val_ds:
        val_loader = torch.utils.data.DataLoader(
            val_ds.with_format("torch"),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
    return train_loader, val_loader