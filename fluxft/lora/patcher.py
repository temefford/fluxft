# fluxft/lora/patcher.py
from __future__ import annotations
import torch
from peft import LoraConfig

import logging

def add_lora_to_unet(unet, lora_cfg: LoraConfig):
    """
    Patch all attention processors in the UNet with LoRA, ensuring trainable parameters are created.
    Uses diffusers >=0.27 add_lora_attn_processor API if available, otherwise falls back to PEFT.
    """
    logging.info(f"Available attn_processors: {list(unet.attn_processors.keys())}")
    logging.info(f"LoRA target modules: {lora_cfg.target_modules}")

    # Try diffusers built-in LoRA support first (preferred for 0.33.1+)
    if hasattr(unet, "add_lora_attn_processor"):
        logging.info("Using diffusers UNet.add_lora_attn_processor API for LoRA patching.")
        unet.add_lora_attn_processor(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
        )
        patched = lora_cfg.target_modules
    else:
        # Fallback: Use PEFT
        from peft import get_peft_model
        logging.info("Falling back to PEFT get_peft_model for LoRA patching.")
        peft_cfg = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias="none",
        )
        unet = get_peft_model(unet, peft_cfg)
        patched = lora_cfg.target_modules

    # Freeze all base parameters
    for p in unet.parameters():
        p.requires_grad_(False)
    # Unfreeze LoRA weights
    for n, p in unet.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    # Debug: Log all trainable parameters
    trainable = [n for n, p in unet.named_parameters() if p.requires_grad]
    if not trainable:
        logging.warning("No trainable LoRA parameters found after patching!")
    logging.info(f"Patched LoRA modules: {patched}")
    logging.info(f"Trainable parameters after LoRA patching: {trainable}")
    return unet