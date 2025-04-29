import torch

import logging

def collate_fn(batch):
    # Remove None entries
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # Stack pixel_values
    pixel_values = [b["pixel_values"] for b in batch]
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    # Always include captions
    captions = []
    for b in batch:
        cap = b.get("caption", "")
        if cap == "":
            logging.warning("[collate_fn] Missing caption in batch entry, using empty string.")
        captions.append(cap)
    out = {
        "pixel_values": torch.stack(pixel_values),
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "captions": captions,
    }
    return out

