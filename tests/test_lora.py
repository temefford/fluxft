# tests/test_lora.py
import torch, types
from fluxft.lora.patcher import add_lora_to_unet
from peft import LoraConfig
class DummyUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_processors = {"to_q": torch.nn.Linear(4,4)}
    def set_attn_processor(self, d): self.attn_processors.update(d)
def test_lora_injection():
    u = DummyUnet()
    cfg = LoraConfig(r=4, lora_alpha=4)
    add_lora_to_unet(u, cfg)
    assert any("lora_" in n for n, _ in u.named_parameters())