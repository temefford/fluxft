# fluxft/train/trainer.py
from __future__ import annotations
import time, logging
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

# Patch to ensure latent image IDs are Long for embeddings
def _patched_prepare_latent_image_ids(batch_size, height, width, device, dtype):
    ids = torch.zeros(height, width, 3)
    ids[...,1] += torch.arange(height)[:,None]
    ids[...,2] += torch.arange(width)[None,:]
    ids = ids.reshape(height*width, 3).to(device, dtype=torch.long)
    return ids
_FluxPipeline._prepare_latent_image_ids = staticmethod(_patched_prepare_latent_image_ids)

log = logging.getLogger(__name__)
shape_debug_logger = logging.getLogger("shape_debug")
shape_debug_logger.setLevel(logging.WARNING)
logs_dir = Path(__file__).parents[1] / "logs"
logs_dir.mkdir(exist_ok=True)
fh = logging.FileHandler(logs_dir/"last_train.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
shape_debug_logger.addHandler(fh)
shape_debug_logger.propagate = False

class LoRATrainer:
    """Trainer for FLUX.1-Schnell with correct dimension handling."""
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        set_logging(cfg.log_level)
        seed_everything(cfg.train.seed)
        torch.backends.cudnn.benchmark = True

        self._init_accelerator()
        self._load_pipeline()
        self._prepare_data()

    def _init_accelerator(self):
        pcfg = (ProjectConfiguration(
            project_dir=str(self.cfg.output_dir),
            logging_dir=str(self.cfg.output_dir/"logs"),
        ) if ProjectConfiguration else None)
        self.accel = Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accum_steps,
            mixed_precision=self.cfg.train.mixed_precision,
            project_config=pcfg,
        )

    def _load_pipeline(self):
        mp = self.cfg.train.mixed_precision
        self.dtype = (torch.float16 if mp=="fp16"
                      else torch.bfloat16 if mp=="bf16"
                      else torch.float32)

        log.info("Loading FluxPipeline …")
        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.train.model_id,
            revision=self.cfg.train.revision,
            torch_dtype=self.dtype,
        )
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config, prediction_type="epsilon"
        )

        # Move modules to device
        for module in [self.pipe.vae, self.pipe.text_encoder,
                       self.pipe.text_encoder_2, self.pipe.transformer]:
            module.to(self.accel.device)

        # Inject LoRA into transformer
        lora_cfg = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.rank,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
        )
        self.transformer = add_lora_to_unet(self.pipe.transformer, lora_cfg)

        # Freeze non‐LoRA modules
        for mod in [self.pipe.vae, self.pipe.text_encoder, self.pipe.text_encoder_2]:
            mod.requires_grad_(False)

        # — Latent projection via 1×1 conv —
        latent_c = self.pipe.vae.config.latent_channels
        trans_c  = self.transformer.config["in_channels"]  # must match transformer in_channels  [oai_citation:12‡Hugging Face](https://huggingface.co/docs/diffusers/v0.17.0/en/api/models?utm_source=chatgpt.com)
        self.latent_proj = torch.nn.Conv2d(
            in_channels=latent_c,
            out_channels=trans_c,
            kernel_size=1,
            bias=False
        ).to(self.accel.device, dtype=self.dtype)

        # Tokenizers & text-projection layers
        self.clip_tokenizer = get_clip_tokenizer()
        clip_dim   = self.pipe.text_encoder.config.hidden_size        # 768  [oai_citation:13‡Hugging Face](https://huggingface.co/docs/diffusers/v0.17.0/en/api/models?utm_source=chatgpt.com)
        pooled_dim = self.transformer.config["pooled_projection_dim"] # 768  [oai_citation:14‡Hugging Face](https://huggingface.co/docs/diffusers/en/api/models/flux_transformer?utm_source=chatgpt.com)
        self.clip_proj = torch.nn.Linear(clip_dim, pooled_dim).to(self.accel.device, dtype=self.dtype)

        self.t5_tokenizer = get_t5_tokenizer()
        t5_dim    = self.pipe.text_encoder_2.config.hidden_size      # e.g. 1024  [oai_citation:15‡Hugging Face](https://huggingface.co/docs/diffusers/v0.17.0/en/api/models?utm_source=chatgpt.com)
        cross_dim = self.transformer.config["joint_attention_dim"]   # 4096  [oai_citation:16‡GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py?utm_source=chatgpt.com)
        self.t5_proj = torch.nn.Linear(t5_dim, cross_dim).to(self.accel.device, dtype=self.dtype)

    def _prepare_data(self):
        img_size = getattr(self.cfg.data, "img_size", 1200)
        self.train_dl, self.val_dl = build_dataloaders(
            self.cfg.data, self.cfg.train.batch_size,
            img_size=img_size, tokenizer=self.clip_tokenizer
        )
        # Optimizer over LoRA + all projection layers
        trainable = (
            [p for p in self.transformer.parameters() if p.requires_grad] +
            list(self.latent_proj.parameters()) +
            list(self.clip_proj.parameters()) +
            list(self.t5_proj.parameters())
        )
        self.opt = torch.optim.AdamW(
            trainable, lr=self.cfg.train.learning_rate, weight_decay=1e-2
        )
        total_steps = (self.cfg.train.max_steps if self.cfg.train.max_steps>0
                       else len(self.train_dl)*self.cfg.train.epochs)
        self.lr_sched = get_scheduler(
            self.cfg.train.scheduler, self.opt,
            num_warmup_steps=self.cfg.train.warmup_steps,
            num_training_steps=total_steps,
        )
        components = [self.pipe.vae, self.transformer,
                      self.latent_proj, self.clip_proj, self.t5_proj,
                      self.opt, self.lr_sched, self.train_dl]
        if self.val_dl is not None:
            components.append(self.val_dl)
        prepared = self.accel.prepare(*components)
        (self.pipe.vae, self.transformer,
         self.latent_proj, self.clip_proj, self.t5_proj,
         self.opt, self.lr_sched, self.train_dl, *rest) = prepared
        if self.val_dl is not None:
            self.val_dl = rest[0]

    def train(self) -> Dict[str, Any]:
        cfg, acc = self.cfg, self.accel
        total_steps = (cfg.train.max_steps if cfg.train.max_steps>0
                       else len(self.train_dl)*cfg.train.epochs)
        log.info(f"Training for {total_steps} steps…")

        step, start = 0, time.time()
        scale   = self.pipe.vae.config.scaling_factor

        for batch in self.train_dl:
            try:
                with acc.accumulate(self.transformer):
                    # Image→latents
                    pix = batch["pixel_values"].to(acc.device, dtype=self.dtype)
                    lat = self.pipe.vae.encode(pix).latent_dist.sample() * scale
                    lat = self.latent_proj(lat)  # [B, C, H, W]
                    b,c,h,w = lat.shape
                    lat = lat.permute(0,2,3,1).reshape(b, h*w, c)

                    # Noise & scheduler
                    noise = torch.randn_like(lat)
                    ts    = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (b,), device=lat.device
                    ).long()
                    lat_noisy = self.noise_scheduler.add_noise(lat, noise, ts)

                    # CLIP conditioning
                    captions = batch["captions"]
                    clip_in = self.clip_tokenizer(
                        captions, padding="longest", return_tensors="pt"
                    ).to(acc.device)
                    clip_out = self.pipe.text_encoder(**clip_in)
                    clip_emb = clip_out.pooler_output
                    clip_emb = self.clip_proj(clip_emb)  # [B, pooled_dim]

                    # T5 conditioning
                    t5_in = self.t5_tokenizer(
                        captions, padding="max_length", truncation=True,
                        max_length=self.t5_tokenizer.model_max_length,
                        return_tensors="pt"
                    ).to(acc.device)
                    t5_out = self.pipe.text_encoder_2(**t5_in)
                    t5_emb = self.t5_proj(t5_out.last_hidden_state)  # [B, seq, cross_dim]

                    # Forward & loss
                    out = self.transformer(
                        hidden_states=lat_noisy,
                        timestep=ts,
                        encoder_hidden_states=t5_emb,
                        pooled_projections=clip_emb,
                        txt_ids=None
                    )
                    loss = F.mse_loss(out.sample.float(), noise.float())
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

            if acc.sync_gradients:
                step += 1
                if step % cfg.train.checkpoint_every == 0 and acc.is_main_process:
                    self._save_checkpoint(step)
                if step >= total_steps:
                    break

        acc.wait_for_everyone()
        if acc.is_main_process:
            self._save_checkpoint("final")

        return {"step": step, "seconds": time.time() - start}

    def _save_checkpoint(self, tag: str):
        out = self.cfg.output_dir / f"ckpt-{tag}"
        out.mkdir(exist_ok=True, parents=True)
        unwrapped = self.accel.unwrap_model(self.transformer)
        if isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(str(out))
        else:
            torch.save(unwrapped.state_dict(), out/"pytorch_model.bin")
        log.info(f"Saved checkpoint → {out}")