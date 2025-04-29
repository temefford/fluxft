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

def _patched_prepare_latent_image_ids(batch_size, height, width, device, dtype):
    # replicate original logic but cast to Long at the end
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] += torch.arange(height)[:, None]
    latent_image_ids[..., 2] += torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    # force integer type for embedding lookup
    return latent_image_ids.to(device=device, dtype=torch.long)

_FluxPipeline._prepare_latent_image_ids = staticmethod(_patched_prepare_latent_image_ids)

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
        # Set transformer channel count to match x_embedder's expected input (64)
        trans_c = 64
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

        # Unpack with projection, matching the number of components
        if self.val_dl is not None:
            (self.pipe.vae, self.transformer, self.latent_proj, self.opt, self.lr_sched, self.train_dl, self.val_dl) = prepared
        else:
            (self.pipe.vae, self.transformer, self.latent_proj, self.opt, self.lr_sched, self.train_dl) = prepared
        # Debug logging for prepared components
        debug_components = [self.pipe.vae, self.transformer, self.latent_proj, self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None:
            debug_components.append(self.val_dl)
        for idx, comp in enumerate(debug_components):
            log.warning(f"[DEBUG] Component {idx}: type={type(comp)}, is None={comp is None}, value={comp}")
        log.warning(f"[DEBUG] self.dtype: {self.dtype}")
        log.warning(f"[DEBUG] self.accel.device: {self.accel.device}")
        # Ensure projection layer on correct device & dtype
        assert self.latent_proj is not None, "latent_proj is None after accelerate.prepare!"
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

                        pix = batch["pixel_values"].to(acc.device, dtype=self.dtype)
                        log.info(f"pixel_values shape: {pix.shape}")
                        lat = self.pipe.vae.encode(pix).latent_dist.sample() * scaler
                        log.info(f"vae latents shape: {lat.shape}")
                        lat = self.latent_proj(lat)  # [B, C, H, W]
                        log.info(f"latent_proj output shape: {lat.shape}")
                        b,c,h,w = lat.shape
                        lat = lat.permute(0,2,3,1).reshape(b, h*w, c)
                        log.info(f"lat after flatten shape: {lat.shape}")

                        # Noise & scheduler
                        noise = torch.randn_like(lat)
                        ts    = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (b,), device=lat.device
                        ).long()
                        lat_noisy = self.noise_scheduler.add_noise(lat, noise, ts)
                        log.info(f"lat_noisy shape: {lat_noisy.shape}")

                        # CLIP conditioning
                        captions = batch["captions"]
                        log.info(f"captions: {captions}")
                        clip_in = self.clip_tokenizer(
                            captions, padding="longest", return_tensors="pt"
                        ).to(acc.device)
                        log.info(f"clip_in.input_ids shape: {clip_in['input_ids'].shape}")
                        clip_out = self.pipe.text_encoder(**clip_in)
                        clip_emb = clip_out.pooler_output
                        log.info(f"clip_emb (pooler_output) shape: {clip_emb.shape}")
                        clip_emb_proj = clip_emb  # [B, pooled_dim]
                        log.info(f"clip_emb after clip_proj shape: {clip_emb_proj.shape}")

                        # T5 conditioning
                        t5_in = self.t5_tokenizer(
                            captions, padding="max_length", truncation=True,
                            max_length=self.t5_tokenizer.model_max_length,
                            return_tensors="pt"
                        ).to(acc.device)
                        log.info(f"t5_in.input_ids shape: {t5_in['input_ids'].shape}")
                        t5_out = self.pipe.text_encoder_2(**t5_in)
                        t5_emb = t5_out.last_hidden_state  # [B, seq, cross_dim]
                        log.info(f"t5_emb after t5_proj shape: {t5_emb.shape}")

                        # Forward & loss
                        log.info(f"Passing to transformer: lat_noisy shape {lat_noisy.shape}, ts shape {ts.shape}, encoder_hidden_states shape {t5_emb.shape}, pooled_projections shape {clip_emb_proj.shape}")

                        # Use raw CLIP embeddings as encoder_hidden_states (no projection)
                        enc_proj = clip_embeds
                        shape_debug_logger.warning(f"[SHAPE] encoder_hidden_states={enc_proj.shape}")

                        # Pass raw clip_embeds (no projection) to any text projection block, only project for transformer context
                        out = self.transformer(
                            hidden_states=lat_proj,
                            timestep=ts,
                            encoder_hidden_states=enc_proj,
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