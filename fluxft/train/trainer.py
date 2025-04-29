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
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline as _FluxPipeline
from peft import LoraConfig, PeftModel

from ..config import GlobalConfig
from ..data.loader import build_dataloaders
from ..lora.patcher import add_lora_to_unet
from ..utils import set_logging, seed_everything
from .tokenizer_util import get_clip_tokenizer, get_t5_tokenizer

# — shape debug logger ——————————————————————————————————
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

def _patched_prepare_latent_image_ids(batch_size, height, width, device, dtype):
    # Keep original semantics, ensure indices are Long for embeddings
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] += torch.arange(height)[:, None]
    latent_image_ids[..., 2] += torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=torch.long)

# Patch the static method on the HF FluxPipeline
_FluxPipeline._prepare_latent_image_ids = staticmethod(_patched_prepare_latent_image_ids)


class LoRATrainer:
    """Trainer for FLUX.1-Schnell with LoRA and correct shape handling."""

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
        mp = self.cfg.train.mixed_precision
        self.dtype = torch.float16 if mp == "fp16" else torch.bfloat16 if mp == "bf16" else torch.float32

        log.info("Loading FluxPipeline …")
        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.train.model_id,
            revision=self.cfg.train.revision,
            torch_dtype=self.dtype,
        )
        # Scheduler for training
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config, prediction_type="epsilon"
        )

        # Move core modules to device
        for module in [self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.transformer]:
            module.to(self.accel.device)

        # Apply LoRA to transformer
        lora_cfg = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.rank,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
        )
        self.transformer = add_lora_to_unet(self.pipe.transformer, lora_cfg)

        # Freeze non-LoRA modules
        for mod in [self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2]:
            mod.requires_grad_(False)

        # — Correct latent projection with 1×1 conv —
        latent_c = self.pipe.vae.config.latent_channels
        trans_c = self.transformer.config.in_channels
        self.latent_proj = torch.nn.Conv2d(
            in_channels=latent_c,
            out_channels=trans_c,
            kernel_size=1,
            bias=False
        ).to(self.accel.device, dtype=self.dtype)

        # Tokenizers for text conditioning
        self.clip_tokenizer = get_clip_tokenizer()
        self.t5_tokenizer = get_t5_tokenizer()

        # Text projection layers
        clip_dim = self.pipe.text_encoder.config.hidden_size
        pooled_dim = self.transformer.config.pooled_projection_dim
        self.clip_proj = torch.nn.Linear(clip_dim, pooled_dim).to(self.accel.device, dtype=self.dtype)

        t5_dim = self.pipe.text_encoder_2.config.hidden_size
        cross_dim = self.transformer.config.cross_attention_dim or self.transformer.config.joint_attention_dim
        self.t5_proj = torch.nn.Linear(t5_dim, cross_dim).to(self.accel.device, dtype=self.dtype)

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data,
            self.cfg.train.batch_size,
            img_size=img_size,
            tokenizer=self.clip_tokenizer
        )

        # Optimizer over LoRA adapters + projection layers
        trainable = (
            [p for p in self.transformer.parameters() if p.requires_grad] +
            list(self.latent_proj.parameters()) +
            list(self.clip_proj.parameters()) +
            list(self.t5_proj.parameters())
        )
        self.opt = torch.optim.AdamW(trainable, lr=self.cfg.train.learning_rate, weight_decay=1e-2)

        total_steps = (
            self.cfg.train.max_steps if self.cfg.train.max_steps > 0
            else len(self.train_dl) * self.cfg.train.epochs
        )
        self.lr_sched = get_scheduler(
            self.cfg.train.scheduler,
            self.opt,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=total_steps,
        )

        components = [self.pipe.vae, self.transformer, self.latent_proj, self.clip_proj, self.t5_proj,
                      self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None:
            components.append(self.val_dl)

        # Prepare with accelerate
        prepared = self.accel.prepare(*components)
        # Unpack
        (self.pipe.vae, self.transformer, self.latent_proj, self.clip_proj, self.t5_proj,
         self.opt, self.lr_sched, self.train_dl, *rest) = prepared
        if self.val_dl is not None:
            self.val_dl = rest[0]

        # Ensure all on correct device & dtype
        self.latent_proj.to(self.accel.device, dtype=self.dtype)
        self.clip_proj.to(self.accel.device, dtype=self.dtype)
        self.t5_proj.to(self.accel.device, dtype=self.dtype)

    def train(self) -> Dict[str, Any]:
        cfg, acc = self.cfg, self.accel
        total_steps = (
            cfg.train.max_steps if cfg.train.max_steps > 0
            else len(self.train_dl) * cfg.train.epochs
        )
        log.info(f"Starting training for {total_steps} steps…")

        step, start_time = 0, time.time()
        scaling = self.pipe.vae.config.scaling_factor

        for batch in self.train_dl:
            try:
                with acc.accumulate(self.transformer):
                    # Move and cast batch
                    pixel_values = batch["pixel_values"].to(acc.device, dtype=self.dtype)
                    captions = batch["captions"]  # list[str]

                    # — Encode images to latents —
                    latents = self.pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * scaling
                    # Project via 1×1 conv and flatten
                    latents = self.latent_proj(latents)
                    b, c, h, w = latents.shape
                    lat_flat = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)

                    # — Add noise —
                    noise = torch.randn_like(lat_flat)
                    ts = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (b,),
                        device=lat_flat.device
                    ).long()
                    lat_noisy = self.noise_scheduler.add_noise(lat_flat, noise, ts)

                    # — Text conditioning: CLIP —
                    clip_inputs = self.clip_tokenizer(
                        captions, padding="longest", return_tensors="pt"
                    ).to(acc.device)
                    clip_out = self.pipe.text_encoder(**clip_inputs)
                    clip_embeds = clip_out.pooler_output  # [b, clip_dim]
                    clip_proj = self.clip_proj(clip_embeds)  # [b, pooled_dim]

                    # — Text conditioning: T5 —
                    t5_inputs = self.t5_tokenizer(
                        captions,
                        padding="max_length",
                        truncation=True,
                        max_length=self.t5_tokenizer.model_max_length,
                        return_tensors="pt"
                    ).to(acc.device)
                    t5_out = self.pipe.text_encoder_2(**t5_inputs)
                    t5_embeds = t5_out.last_hidden_state  # [b, seq_len, t5_dim]
                    prompt_embeds_2 = self.t5_proj(t5_embeds)  # [b, seq_len, cross_dim]

                    # — Forward through Flux transformer —
                    out = self.transformer(
                        hidden_states=lat_noisy,
                        timestep=ts,
                        encoder_hidden_states=prompt_embeds_2,
                        pooled_projections=clip_proj,
                        txt_ids=None
                    )
                    preds = out.sample

                    # — Compute loss & backward —
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

            if acc.sync_gradients:
                step += 1
                if step % cfg.train.checkpoint_every == 0 and acc.is_main_process:
                    self._save_checkpoint(step)
                if step >= total_steps:
                    break

        acc.wait_for_everyone()
        if acc.is_main_process:
            self._save_checkpoint("final")

        duration = time.time() - start_time
        return {"step": step, "seconds": duration}

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