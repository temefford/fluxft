# fluxft/config.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, validator

_DEFAULT_DATA_DIR = Path("data")  # overridable

class LoRAConfig(BaseModel):
    rank: int = Field(16, ge=1, le=128)
    dropout: float = Field(0.1, ge=0.0, le=0.5)
    target_modules: List[str] = ["to_q", "to_k", "to_v"]

class TrainConfig(BaseModel):
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    revision: Optional[str] = None
    learning_rate: float = Field(1e-4, ge=1e-6, le=5e-4)
    batch_size: int = Field(2, ge=1)
    gradient_accum_steps: int = Field(4, ge=1)
    max_steps: int = 1000
    mixed_precision: str = Field("fp16", pattern="^(fp16|bf16|no)$")
    epochs: int = 1
    seed: int = 42
    scheduler: str = "cosine"
    warmup_steps: int = 50
    checkpoint_every: int = 200

class DataConfig(BaseModel):
    dataset_type: str = Field("imagefolder", pattern="^(imagefolder|hf_metadata)$")
    data_dir: Path = _DEFAULT_DATA_DIR
    image_column: str = "hash"
    caption_column: str = "caption"
    validation_split: float = Field(0.1, ge=0.0, le=0.5)

class SearchSpace(BaseModel):
    lr: List[float] = [5e-5, 1e-4, 2e-4]
    rank: List[int] = [4, 8, 16, 32]
    dropout: List[float] = [0.0, 0.05, 0.1]
    batch_size: List[int] = [1, 2, 4]

class GlobalConfig(BaseModel):
    output_dir: Path = Path("outputs")
    log_level: str = "INFO"
    lora: LoRAConfig = LoRAConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    search: SearchSpace = SearchSpace()

    @validator("output_dir", pre=True)
    def _mkdir(cls, p: Path) -> Path:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p