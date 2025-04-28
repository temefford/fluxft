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

# set up a file logger for shape debugging
shape_debug_logger = logging.getLogger("shape_debug")
shape_debug_logger.setLevel(logging.WARNING)
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(logs_dir / "last_train.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
shape_debug_logger.addHandler(fh)
shape_debug_logger.propagate = False

log = logging.getLogger(__name__)


class LoRATrainer:
    """Wraps accelerator, data, and the training loop."""

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        set_logging(cfg.log_level)
        seed_everything(cfg.train.seed)
        torch.backends.cudnn.benchmark = True

        self._init_accelerator()
        self._load_pipeline()
        self._prepare_data()

    # internal helpers

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
        # choose dtype based on config
        self.dtype = (
            torch.float16
            if self.cfg.train.mixed_precision == "fp16"
            else torch.bfloat16
            if self.cfg.train.mixed_precision == "bf16"
            else torch.float32
        )

        log.info("Loading FLUX pipeline …")
        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.train.model_id,
            revision=self.cfg.train.revision,
            torch_dtype=self.dtype,
        )

        # swap in a training scheduler
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config, prediction_type="epsilon"
        )

        # inject LoRA into the model
        lora_cfg = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.rank,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
        )
        self.unet = add_lora_to_unet(self.pipe.transformer, lora_cfg)

        # freeze everything except LoRA and projection
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)

        # --- NEW: build a single projection layer up front ---
        # maps VAE latent channels -> transformer input channels
        latent_c = self.pipe.vae.config.latent_channels
        trans_c = self.unet.config.in_channels
        self.latent_proj = torch.nn.Linear(latent_c, trans_c)

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data, self.cfg.train.batch_size, img_size=img_size
        )

        # optimizer over only LoRA + projection parameters
        params = [
            p for p in self.unet.parameters() if p.requires_grad
        ] + list(self.latent_proj.parameters())
        self.opt = torch.optim.AdamW(params, lr=self.cfg.train.learning_rate, weight_decay=1e-2)

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

        # prepare everything with accelerate
        components = [
            self.unet,
            self.pipe.vae,
            self.latent_proj,
            self.opt,
            self.lr_sched,
            self.train_dl,
        ]
        if self.val_dl is not None:
            components.append(self.val_dl)
        prepared = self.accel.prepare(*components)
        self.unet = prepared[0]
        self.pipe.vae = prepared[1]
        self.latent_proj = prepared[2]
        self.opt = prepared[3]
        self.lr_sched = prepared[4]
        self.train_dl = prepared[5]
        if self.val_dl is not None:
            self.val_dl = prepared[6]
        # Debug log to identify NoneType issues
        log.warning(f"[DEBUG] post-prepare types: unet={type(self.unet)}, vae={type(self.pipe.vae)}, latent_proj={type(self.latent_proj)}, opt={type(self.opt)}, lr_sched={type(self.lr_sched)}, train_dl={type(self.train_dl)}, val_dl={type(self.val_dl) if self.val_dl is not None else 'None'}")
        log.warning(f"[DEBUG] post-prepare values: unet={self.unet}, vae={self.pipe.vae}, latent_proj={self.latent_proj}, opt={self.opt}, lr_sched={self.lr_sched}, train_dl={self.train_dl}, val_dl={self.val_dl if self.val_dl is not None else 'None'}")
        # Assert all components are non-None
        assert self.unet is not None, "self.unet is None after accelerate.prepare"
        assert self.pipe.vae is not None, "self.pipe.vae is None after accelerate.prepare"
        assert self.latent_proj is not None, "self.latent_proj is None after accelerate.prepare"


    # public API

    def train(self) -> Dict[str, Any]:
        cfg, acc = self.cfg, self.accel
        total_steps = (
            cfg.train.max_steps
            if cfg.train.max_steps > 0
            else len(self.train_dl) * cfg.train.epochs
        )
        log.info(f"Starting training for {total_steps} steps…")

        step, t0 = 0, time.time()
        scaling = self.pipe.vae.config.scaling_factor

        try:
            for batch in self.train_dl:
                try:
                    with acc.accumulate(self.unet):
                        imgs = batch["pixel_values"].to(acc.device, dtype=self.dtype)

                        # encode to latents
                        latents = self.pipe.vae.encode(imgs).latent_dist.sample()
                        latents = latents * scaling

                        # flatten spatial dims
                        b, c, h, w = latents.shape
                        lat_flat = latents.permute(0, 2, 3, 1).reshape(b, h*w, c)

                        # debug shapes
                        shape_debug_logger.warning(f"[SHAPE] lat_flat={lat_flat.shape}")

                        # project into transformer channel space
                        lat_proj = self.latent_proj(lat_flat)

                        # add noise
                        noise = torch.randn_like(lat_proj)

                        # --- DEBUG: log and check scheduler timesteps ---
                        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
                        log.warning(f"num_train_timesteps={num_train_timesteps}")
                        assert isinstance(num_train_timesteps, int) and num_train_timesteps > 0 and num_train_timesteps < 100_000, "num_train_timesteps must be a finite positive integer"

                        ts = torch.randint(
                            0,
                            num_train_timesteps,
                            (b,),
                            device=lat_proj.device,
                        ).long()
                        log.warning(f"Sampled ts: {ts}")
                        assert torch.isfinite(ts).all(), "Non-finite timestep detected!"

                        lat_noisy = self.noise_scheduler.add_noise(lat_proj, noise, ts)

                        # predict
                        preds = self.unet(
                            hidden_states=lat_noisy,
                            timestep=ts,
                            encoder_hidden_states=None,
                            pooled_projections=None,
                            img_ids=torch.arange(h*w, device=lat_proj.device).repeat(b,1),
                            txt_ids=None,
                        ).sample

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
                    else:
                        raise

                # after accumulation
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

        duration = time.time() - t0
        return {"step": step, "seconds": duration}

    def validate(self):
        log.info("Validation loop not implemented yet.")
        pass

    # saving

    def _save_checkpoint(self, tag: str):
        out = self.cfg.output_dir / f"ckpt-{tag}"
        out.mkdir(exist_ok=True, parents=True)
        unwrapped = self.accel.unwrap_model(self.unet)
        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(str(out))
        else:
            torch.save(unwrapped.state_dict(), out / "pytorch_model.bin")
        log.info(f"Saved checkpoint → {out}")