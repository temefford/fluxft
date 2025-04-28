# fluxft/search/hp_search.py
from __future__ import annotations
import itertools, random, json, time, logging
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy
from ..config import GlobalConfig
from ..train.trainer import LoRATrainer
from ..eval.evaluator import MetricComputer
from ..utils import set_logging

log = logging.getLogger(__name__)

def random_grid(cfg: GlobalConfig, n_trials: int = 20):
    space = cfg.search
    combos = list(
        itertools.product(
            space.lr, space.rank, space.dropout, space.batch_size
        )
    )
    random.shuffle(combos)
    return combos[:n_trials]

def objective(cfg: GlobalConfig, params) -> Dict[str, Any]:
    lr, rank, drop, bs = params
    cfg_trial = deepcopy(cfg)
    cfg_trial.train.learning_rate = lr
    cfg_trial.lora.rank = rank
    cfg_trial.lora.dropout = drop
    cfg_trial.train.batch_size = bs
    tag = f"lr{lr}_r{rank}_d{drop}_bs{bs}"
    cfg_trial.output_dir = Path(cfg.output_dir) / tag
    trainer = LoRATrainer(cfg_trial)
    res = trainer.train()  # returns step & seconds
    evaluator = MetricComputer(cfg_trial, cfg_trial.output_dir / "ckpt-final")
    metrics = evaluator.run(
        prompts=["A red car", "A blue cat"], ref_dir=cfg.data.data_dir
    )
    score = metrics["CLIPScore"] - 0.05 * metrics["FID"]
    # lower sec is better; combine cost
    composite = score / (1 + res["seconds"] / 3600)
    run_info = dict(
        tag=tag, score=score, seconds=res["seconds"], composite=composite, **metrics
    )
    log.info(f"[{tag}] composite={composite:.4f}")
    # persist
    with open(Path(cfg.output_dir) / "search_log.jsonl", "a") as fp:
        fp.write(json.dumps(run_info) + "\n")
    return run_info

def run_search(cfg: GlobalConfig, n_trials: int = 10):
    set_logging(cfg.log_level)
    best = None
    for params in random_grid(cfg, n_trials):
        try:
            res = objective(cfg, params)
            if best is None or res["composite"] > best["composite"]:
                best = res
        except Exception as e:
            log.error(f"Trial {params} failed: {e}")
    log.info(f"BEST RUN ⇒ {best}")