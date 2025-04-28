# fluxft/__init__.py
"""LoRA fine-tuning toolkit for FLUX 1-schnell."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("fluxft")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .config import GlobalConfig
from .train.trainer import LoRATrainer
from .eval.evaluator import MetricComputer