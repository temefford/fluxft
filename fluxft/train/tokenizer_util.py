from transformers import CLIPTokenizer, T5Tokenizer

def get_clip_tokenizer():
    return CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", use_fast=True
    )

def get_t5_tokenizer():
    return T5Tokenizer.from_pretrained(
        "google/t5-v1_1-xxl", use_fast=True
    )
