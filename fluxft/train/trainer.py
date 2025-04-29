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

        # Move all relevant modules to acc.device
        self.pipe.text_encoder.to(self.accel.device)
        self.pipe.text_encoder_2.to(self.accel.device)
        self.pipe.vae.to(self.accel.device)
        self.pipe.transformer.to(self.accel.device)

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

        # Build projection layer mapping VAE latents -> transformer channel space
        latent_c = self.pipe.vae.config.latent_channels
        trans_c = self.transformer.config.in_channels
        self.latent_proj = torch.nn.Linear(latent_c, trans_c)

        # Initialize CLIP and T5 tokenizers for text conditioning
        self.clip_tokenizer = get_clip_tokenizer()
        self.t5_tokenizer = get_t5_tokenizer()

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        # Pass CLIP tokenizer to dataloader
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data, self.cfg.train.batch_size, img_size=img_size, tokenizer=self.clip_tokenizer
        )

        # Optimizer only over LoRA adapter params + projection
        trainable = [p for p in self.transformer.parameters() if p.requires_grad] + list(self.latent_proj.parameters())
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

        # Prepare with accelerate (include projection layer)
        components = [self.pipe.vae, self.transformer, self.latent_proj, self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None:
            components.append(self.val_dl)
        prepared = self.accel.prepare(*components)

        # Unpack with projection
        self.pipe.vae, self.transformer, self.latent_proj, self.opt, self.lr_sched, self.train_dl, *rest = prepared
        # Debug logging for prepared components
        debug_components = [self.pipe.vae, self.transformer, self.latent_proj, self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None and rest:
            self.val_dl = rest[0]
            debug_components.append(self.val_dl)
        for idx, comp in enumerate(debug_components):
            log.warning(f"[DEBUG] Component {idx}: type={type(comp)}, is None={comp is None}")
        # Ensure projection layer on correct device & dtype
        self.latent_proj = self.latent_proj.to(self.accel.device, dtype=self.dtype)
        # No need to manually .to() VAE/transformer after accelerator.prepare


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
                        # Move all tensors in batch to acc.device
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                # Only cast float tensors (images/latents) to self.dtype, keep ids/masks as int
                                if k == "pixel_values":
                                    batch[k] = v.to(acc.device, dtype=self.dtype)
                                else:
                                    batch[k] = v.to(acc.device)

                        imgs = batch["pixel_values"]

                        # Encode → latents
                        latents = self.pipe.vae.encode(imgs).latent_dist.sample()
                        latents = latents * scaler

                        # Add noise in latent space (before projection)
                        noise = torch.randn_like(latents)
                        ts = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (latents.shape[0],),
                            device=latents.device,
                        ).long()
                        lat_noisy = self.noise_scheduler.add_noise(latents, noise, ts)

                        # Flatten spatial dims and project to transformer channels
                        b, c, h, w = lat_noisy.shape
                        lat_flat = lat_noisy.permute(0, 2, 3, 1).reshape(b, h * w, c)
                        shape_debug_logger.warning(f"[SHAPE] lat_flat={lat_flat.shape}")
                        lat_proj = self.latent_proj(lat_flat)
                        shape_debug_logger.warning(f"[SHAPE] lat_proj={lat_proj.shape}")

                        # Text encoding & pooled projections
                        # Ensure token indices are proper dtype and device for embedding
                        input_ids = batch["input_ids"].long().to(acc.device)
                        attention_mask = batch["attention_mask"].long().to(acc.device)
                        text_outputs = self.pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                        txt_embeds = text_outputs[0]
                        # Use embedding inputs directly to avoid embedding layer index errors
                        pooled_proj = self.pipe.text_encoder_2(inputs_embeds=txt_embeds).last_hidden_state
                        shape_debug_logger.warning(f"[SHAPE] pooled_proj={pooled_proj.shape}")

                        # Forward through Flux’s transformer
                        out = self.transformer(
                            hidden_states=lat_proj,
                            timestep=ts,
                            encoder_hidden_states=None,
                            pooled_projections=pooled_proj,
                            txt_ids=None,
                        )
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