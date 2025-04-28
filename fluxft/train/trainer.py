# fluxft/train/trainer.py
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
try:
    from accelerate import ProjectConfiguration
except ImportError:
    ProjectConfiguration = None
from diffusers import FluxPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel

from ..config import GlobalConfig
from ..data.loader import build_dataloaders
from ..lora.patcher import add_lora_to_unet
from ..utils import set_logging, seed_everything

# — shape debug logger ————————————————————————————————
shape_debug_logger = logging.getLogger("shape_debug")
shape_debug_logger.setLevel(logging.WARNING)
logs_dir = Path(__file__).parents[1] / "logs"
logs_dir.mkdir(exist_ok=True)
fh = logging.FileHandler(logs_dir / "last_train.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
shape_debug_logger.addHandler(fh)
shape_debug_logger.propagate = False
# ——————————————————————————————————————————————————————

log = logging.getLogger(__name__)


class LoRATrainer:
    """Wraps accelerator, data, and the training loop for Flux-1 with LoRA."""

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        set_logging(cfg.log_level)
        seed_everything(cfg.train.seed)
        torch.backends.cudnn.benchmark = True

        self._init_accelerator()
        self._load_pipeline()
        self._prepare_data()

    def _init_accelerator(self):
        pcfg = (
            ProjectConfiguration(
                project_dir=str(self.cfg.output_dir),
                logging_dir=str(self.cfg.output_dir / "logs"),
            )
            if ProjectConfiguration
            else None
        )
        self.accel = Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accum_steps,
            mixed_precision=self.cfg.train.mixed_precision,
            project_config=pcfg,
        )

    def _load_pipeline(self):
        # Choose dtype
        mp = self.cfg.train.mixed_precision
        self.dtype = torch.float16 if mp == "fp16" else torch.bfloat16 if mp == "bf16" else torch.float32

        log.info("Loading FluxPipeline …")
        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.train.model_id,
            revision=self.cfg.train.revision,
            torch_dtype=self.dtype,
        )

        # Replace inference scheduler with a training-friendly DDPMScheduler
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config, prediction_type="epsilon"
        )

        # **IMPORTANT**: FluxPipeline exposes the denoiser as `pipe.transformer`, not `pipe.unet`
        lora_cfg = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.rank,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
        )
        self.transformer = add_lora_to_unet(self.pipe.transformer, lora_cfg)

        # Freeze everything else
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data, self.cfg.train.batch_size, img_size=img_size
        )

        # Optimizer only over LoRA adapter params
        trainable = [p for p in self.transformer.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable, lr=self.cfg.train.learning_rate, weight_decay=1e-2)

        # LR scheduler
        total_steps = (
            self.cfg.train.max_steps
            if self.cfg.train.max_steps > 0
            else len(self.train_dl) * self.cfg.train.epochs
        )
        self.lr_sched = get_scheduler(
            self.cfg.train.scheduler,
            self.opt,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=total_steps,
        )

        # Move VAE and transformer to device and dtype before wrapping
        device = self.accel.device
        self.pipe.vae = self.pipe.vae.to(device, dtype=self.dtype)
        self.transformer = self.transformer.to(device, dtype=self.dtype)

        # Prepare with accelerate
        components = [self.pipe.vae, self.transformer, self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None:
            components.append(self.val_dl)
        prepared = self.accel.prepare(*components)

        # Unpack
        self.pipe.vae, self.transformer, self.opt, self.lr_sched, self.train_dl, *rest = prepared
        if self.val_dl is not None:
            self.val_dl = rest[0]
        # Re-apply device and dtype to ensure modules are on GPU
        device = self.accel.device
        self.pipe.vae = self.pipe.vae.to(device, dtype=self.dtype)
        self.transformer = self.transformer.to(device, dtype=self.dtype)

    def train(self) -> Dict[str, Any]:
        cfg, acc = self.cfg, self.accel
        total_steps = (
            cfg.train.max_steps
            if cfg.train.max_steps > 0
            else len(self.train_dl) * cfg.train.epochs
        )
        log.info(f"Starting training for {total_steps} steps…")

        step, t0 = 0, time.time()
        scaler = self.pipe.vae.config.scaling_factor

        try:
            for batch in self.train_dl:
                try:
                    with acc.accumulate(self.transformer):
                        imgs = batch["pixel_values"].to(acc.device, dtype=self.dtype)

                        # Encode → latents
                        latents = self.pipe.vae.encode(imgs).latent_dist.sample()
                        latents = latents * scaler

                        shape_debug_logger.warning(f"[SHAPE] latents={latents.shape}")

                        # Add noise
                        noise = torch.randn_like(latents)
                        ts = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (latents.shape[0],),
                            device=latents.device,
                        ).long()
                        lat_noisy = self.noise_scheduler.add_noise(latents, noise, ts)

                        # Forward through Flux’s transformer
                        out = self.transformer(
                            hidden_states=lat_noisy,
                            timestep=ts,
                            encoder_hidden_states=None,
                            pooled_projections=None,
                            img_ids=torch.arange(latents.shape[2] * latents.shape[3], device=latents.device).repeat(latents.shape[0],1),
                            txt_ids=None,
                        )
                        # The model returns a ModelOutput; the noise prediction is in `.sample`
                        preds = out.sample

                        loss = F.mse_loss(preds.float(), noise.float())
                        acc.backward(loss)
                        self.opt.step()
                        self.lr_sched.step()
                        self.opt.zero_grad()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        log.warning("CUDA OOM, skipping batch…")
                        torch.cuda.empty_cache()
                        continue
                    raise

                # After each gradient accumulation
                if acc.sync_gradients:
                    step += 1
                    if step % cfg.train.checkpoint_every == 0 and acc.is_main_process:
                        self._save_checkpoint(step)
                    if step >= total_steps:
                        break

        except KeyboardInterrupt:
            log.warning("Interrupted by user — saving checkpoint…")
            if acc.is_main_process:
                self._save_checkpoint(f"interrupt_{step}")

        acc.wait_for_everyone()
        if acc.is_main_process:
            self._save_checkpoint("final")

        return {"step": step, "seconds": time.time() - t0}

    def validate(self):
        log.info("Validation not implemented.")
        pass

    def _save_checkpoint(self, tag: str):
        out = self.cfg.output_dir / f"ckpt-{tag}"
        out.mkdir(exist_ok=True, parents=True)
        unwrapped = self.accel.unwrap_model(self.transformer)
        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(str(out))
        else:
            torch.save(unwrapped.state_dict(), out / "pytorch_model.bin")
        log.info(f"Saved checkpoint → {out}")