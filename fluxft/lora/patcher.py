# fluxft/lora/patcher.py
from __future__ import annotations
import torch
from peft import LoraConfig
from diffusers.models.attention_processor import LoRAAttnProcessor

import logging

def add_lora_to_unet(unet, lora_cfg: LoraConfig):
    """
    Patch all attention processors in the UNet. For each processor, if its name matches a LoRA target,
    replace it with a LoRAAttnProcessor; otherwise, keep the original processor. This guarantees the number
    of processors matches the number of layers, preventing ValueError.
    Adds debug logging to show all processor names and which are patched.
    """
    lora_procs = {}
    patched = []
    logging.info(f"Available attn_processors: {list(unet.attn_processors.keys())}")
    logging.info(f"LoRA target modules: {lora_cfg.target_modules}")
    for name, module in unet.attn_processors.items():
        if any(t in name for t in lora_cfg.target_modules):
            lora_procs[name] = LoRAAttnProcessor(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.lora_alpha,
                dropout=lora_cfg.lora_dropout,
                train_kv=True,
            )
            patched.append(name)
        else:
            lora_procs[name] = module  # keep original
    if not patched:
        logging.warning(f"No LoRA modules patched in UNet! Target modules: {lora_cfg.target_modules}")
    else:
        logging.info(f"Patched LoRA modules: {patched}")
    unet.set_attn_processor(lora_procs)
    # freeze base parameters
    for p in unet.parameters():
        p.requires_grad_(False)
    # make LoRA trainable
    for n, p in unet.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    # Debug: Log all trainable parameters
    trainable = [n for n, p in unet.named_parameters() if p.requires_grad]
    logging.info(f"Trainable parameters after LoRA patching: {trainable}")
    return unet