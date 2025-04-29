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
from .tokenizer_util import get_clip_tokenizer, get_t5_tokenizer

# — shape-debug logger setup ————————————————————————————————
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

# Monkey-patch FluxPipeline to produce integer image‐IDs for rotary embeddings
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline as _FluxPipeline
def _patched_prepare_latent_image_ids(batch_size, height, width, device, dtype):
    # create (H, W, 3) ID tensor then flatten to (H*W, 3)
    ids = torch.zeros(height, width, 3, device=device)
    ids[..., 1] = torch.arange(height, device=device)[:, None]
    ids[..., 2] = torch.arange(width, device=device)[None, :]
    ids = ids.reshape(height * width, 3).long()
    return ids
_FluxPipeline._prepare_latent_image_ids = staticmethod(_patched_prepare_latent_image_ids)


class LoRATrainer:
    """Encapsulates Accelerator, data, LoRA-patching, and the training loop."""

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
        # Choose compute dtype
        mp = self.cfg.train.mixed_precision
        self.dtype = (
            torch.float16 if mp == "fp16"
            else torch.bfloat16 if mp == "bf16"
            else torch.float32
        )

        log.info("Loading FluxPipeline …")
        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.train.model_id,
            revision=self.cfg.train.revision,
            torch_dtype=self.dtype,
        )

        # Swap in training scheduler
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config, prediction_type="epsilon"
        )

        # Move core modules to device
        self.pipe.vae.to(self.accel.device)
        self.pipe.text_encoder.to(self.accel.device)
        self.pipe.text_encoder_2.to(self.accel.device)
        self.pipe.transformer.to(self.accel.device)

        # Inject LoRA into the transformer (denoiser)
        lora_cfg = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.rank,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
        )
        self.transformer = add_lora_to_unet(self.pipe.transformer, lora_cfg)

        # Freeze all non-LoRA parameters
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.text_encoder_2.requires_grad_(False)

        # Build projection: VAE latent C -> transformer input C
        latent_c = self.pipe.vae.config.latent_channels
        trans_c = self.transformer.config.in_channels
        self.latent_proj = torch.nn.Linear(latent_c, trans_c).to(self.accel.device, dtype=self.dtype)

        # Initialize tokenizers for text conditioning
        self.clip_tokenizer = get_clip_tokenizer().to(self.accel.device)
        self.t5_tokenizer = get_t5_tokenizer().to(self.accel.device)

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data,
            batch_size=self.cfg.train.batch_size,
            img_size=img_size,
            tokenizer=self.clip_tokenizer,
        )

        # Optimizer over LoRA + projection parameters
        trainable = [p for p in self.transformer.parameters() if p.requires_grad] + list(self.latent_proj.parameters())
        self.opt = torch.optim.AdamW(
            trainable,
            lr=self.cfg.train.learning_rate,
            weight_decay=1e-2,
        )

        # Learning‐rate scheduler
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

        # Prepare with accelerate (include val_dl if present)
        components = [self.transformer, self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None:
            components.append(self.val_dl)

        prepared = self.accel.prepare(*components)
        self.transformer, self.opt, self.lr_sched, self.train_dl, *rest = prepared
        if self.val_dl is not None:
            self.val_dl = rest[0]

        log.info(f"Prepared: transformer={type(self.transformer)}, optimizer={type(self.opt)}")

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
                with acc.accumulate(self.transformer):
                    # Move and cast batch
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
                        (latents.size(0),),
                        device=latents.device,
                    ).long()
                    lat_noisy = self.noise_scheduler.add_noise(latents, noise, ts)

                    # Project latents to transformer channels
                    b, c, h, w = lat_noisy.shape
                    lat_flat = lat_noisy.permute(0, 2, 3, 1).reshape(b, h*w, c)
                    lat_proj = self.latent_proj(lat_flat)
                    shape_debug_logger.warning(f"[SHAPE] lat_proj={lat_proj.shape}")

                    # Text conditioning (if captions present)
                    # Using CLIP tokenizer + text encoder
                    input_ids = batch["input_ids"].to(acc.device)
                    attention_mask = batch["attention_mask"].to(acc.device)
                    clip_out = self.pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_proj = clip_out.pooler_output  # shape [b, hidden]

                    # Flux transformer forward
                    out = self.transformer(
                        hidden_states=lat_proj,
                        timestep=ts,
                        encoder_hidden_states=None,
                        pooled_projections=pooled_proj,
                        img_ids=None,
                        txt_ids=None,
                    )
                    preds = out.sample

                    # Compute loss
                    loss = F.mse_loss(preds.float(), noise.float())
                    acc.backward(loss)
                    self.opt.step()
                    self.lr_sched.step()
                    self.opt.zero_grad()

                if acc.sync_gradients:
                    step += 1
                    if step % cfg.train.checkpoint_every == 0 and acc.is_main_process:
                        self._save_checkpoint(step)
                    if step >= total_steps:
                        break

        except KeyboardInterrupt:
            log.warning("Interrupted by user—saving checkpoint…")
            if acc.is_main_process:
                self._save_checkpoint(f"interrupt_{step}")

        acc.wait_for_everyone()
        if acc.is_main_process:
            self._save_checkpoint("final")

        return {"step": step, "seconds": time.time() - t0}

    def validate(self):
        log.info("Validation loop not implemented.")
        pass

    def _save_checkpoint(self, tag: str):
        out = self.cfg.output_dir / f"ckpt-{tag}"
        out.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accel.unwrap_model(self.transformer)
        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(str(out))
        else:
            torch.save(unwrapped.state_dict(), out / "pytorch_model.bin")
        log.info(f"Saved checkpoint → {out}")