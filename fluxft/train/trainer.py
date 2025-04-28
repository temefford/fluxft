# fluxft/train/trainer.py
from __future__ import annotations

import math, time, logging, os
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

log = logging.getLogger(__name__)


class LoRATrainer:
    """Wraps accelerator, data, and training loop."""

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        set_logging(cfg.log_level)
        seed_everything(cfg.train.seed)
        import torch.backends.cudnn
        torch.backends.cudnn.benchmark = True

        # Latent projection layer: 64 (VAE channels) -> 3072 (transformer input)
        self.latent_proj = torch.nn.Linear(64, 3072)

        self._init_accelerator()
        self._load_pipeline()
        self._prepare_data()

    # ---------- internal helpers ----------
    def _init_accelerator(self):
        pcfg = None
        if ProjectConfiguration is not None:
            pcfg = ProjectConfiguration(
                project_dir=str(self.cfg.output_dir),
                logging_dir=str(self.cfg.output_dir / "logs"),
            )
        self.accel = Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accum_steps,
            mixed_precision=self.cfg.train.mixed_precision,
            project_config=pcfg,
        )

    def _load_pipeline(self):
        self.dtype = (
            torch.float16 if self.cfg.train.mixed_precision == "fp16"
            else torch.bfloat16 if self.cfg.train.mixed_precision == "bf16"
            else torch.float32
        )
        log.info("Loading FLUX pipeline …")
        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.train.model_id,
            revision=self.cfg.train.revision,
            torch_dtype=self.dtype,
        )
        # Replace scheduler with DDPMScheduler for training
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config, prediction_type="epsilon"
        )
        # Apply LoRA to transformer
        lora_cfg = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.rank,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
        )
        self.unet = add_lora_to_unet(self.pipe.transformer, lora_cfg)

        # Freeze non-trainable modules
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data, self.cfg.train.batch_size, img_size=img_size
        )

        param_groups = [p for p in self.unet.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(
            param_groups, lr=self.cfg.train.learning_rate, weight_decay=1e-2
        )

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

        components = (
            self.unet,
            self.pipe.vae,
            self.latent_proj,
            self.opt,
            self.lr_sched,
            self.train_dl,
            *( [self.val_dl] if self.val_dl else [] ),
        )
        prepared = self.accel.prepare(*components)
        (
            self.unet,
            self.pipe.vae,
            self.latent_proj,
            self.opt,
            self.lr_sched,
            self.train_dl,
            *rest,
        ) = prepared
        if self.val_dl:
            self.val_dl = rest[0]

    # ---------- public API ----------
    def train(self) -> Dict[str, Any]:
        cfg, acc = self.cfg, self.accel
        total_steps = (
            cfg.train.max_steps
            if cfg.train.max_steps > 0
            else len(self.train_dl) * cfg.train.epochs
        )
        log.info(f"Start training for {total_steps} steps")

        step, t0 = 0, time.time()
        img_scaler = self.pipe.vae.config.scaling_factor

        try:
            for batch in self.train_dl:
                try:
                    with acc.accumulate(self.unet):
                        imgs = batch["pixel_values"].to(
                            acc.device, dtype=self.dtype
                        )

                        # VAE encoding
                        latents = self.pipe.vae.encode(imgs).latent_dist.sample()
                        latents *= img_scaler

                        # Prepare latents
                        b, c, h, w = latents.shape
                        lat = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)
                        lat = self.latent_proj(lat)
                        log.info(f"latents.shape={latents.shape}, lat.shape (after proj)={lat.shape}")

                        # Add noise
                        noise = torch.randn_like(lat, device=lat.device)
                        ts = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (b,),
                            device=lat.device,
                        ).long()
                        lat_noisy = self.noise_scheduler.add_noise(lat, noise, ts)

                        # Forward
                        preds = self.unet(
                            hidden_states=lat_noisy,
                            timestep=ts,
                            encoder_hidden_states=None,
                            pooled_projections=None,
                            img_ids=torch.arange(h * w, device=lat.device).repeat(b, 1),
                            txt_ids=None,
                        ).sample

                        loss = F.mse_loss(preds.float(), noise.float())

                        # Backward
                        acc.backward(loss)
                        self.opt.step()
                        self.lr_sched.step()
                        self.opt.zero_grad()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        log.warning('CUDA OOM on batch, skipping batch.')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

                # Only after accumulation
                if acc.sync_gradients:
                    step += 1

                    if step % cfg.train.checkpoint_every == 0 and acc.is_main_process:
                        self._save_checkpoint(step)

                    if step >= total_steps:
                        break

        except KeyboardInterrupt:
            log.warning('Training interrupted by user. Saving checkpoint...')
            if acc.is_main_process:
                self._save_checkpoint(f"interrupt_{step}")

        acc.wait_for_everyone()
        if acc.is_main_process:
            self._save_checkpoint("final")

        duration = time.time() - t0
        return dict(step=step, seconds=duration)

    def validate(self):
        log.info('Validation not yet implemented.')
        pass

    # ---------- saving ----------
    def _save_checkpoint(self, tag: str):
        path = self.cfg.output_dir / f"ckpt-{tag}"
        path.mkdir(exist_ok=True, parents=True)

        unwrapped = self.accel.unwrap_model(self.unet)
        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(str(path))
        else:
            torch.save(unwrapped.state_dict(), path / "pytorch_model.bin")

        log.info(f"Saved checkpoint to {path}")