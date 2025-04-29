from transformers import AutoTokenizer

def get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)
