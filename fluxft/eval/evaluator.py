# fluxft/eval/evaluator.py
from __future__ import annotations
import logging, torch, json
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict
import torchmetrics
from diffusers import FluxPipeline
from torchvision.utils import save_image
from transformers import CLIPProcessor, CLIPModel
from torch_fidelity import calculate_metrics
from ..config import GlobalConfig
from ..utils import set_logging

log = logging.getLogger(__name__)

class MetricComputer:
    """Compute CLIPScore, FID, AestheticScore (PickScore optional)."""

    def __init__(self, cfg: GlobalConfig, lora_path: Path):
        set_logging(cfg.log_level)
        dtype = (
            torch.float16
            if cfg.train.mixed_precision == "fp16"
            else torch.float32
        )
        self.pipe = FluxPipeline.from_pretrained(
            cfg.train.model_id, torch_dtype=dtype
        )
        self.pipe.load_lora_weights(str(lora_path))
        self.pipe.to("cuda")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.img_temp_dir = Path(cfg.output_dir) / "eval_imgs"
        self.img_temp_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def _generate(self, prompts: List[str], n_per_prompt: int = 2) -> List[Path]:
        paths = []
        for p in tqdm(prompts, desc="Generating"):
            imgs = self.pipe(p, num_inference_steps=40).images  # type: ignore
            for i, im in enumerate(imgs[:n_per_prompt]):
                f = self.img_temp_dir / f"{hash(p)}_{i}.png"
                im.save(f)
                paths.append(f)
        return paths

    @torch.no_grad()
    def clip_score(self, imgs: List[Path], prompts: List[str]) -> float:
        scores = []
        for img_path, text in zip(imgs, prompts):
            clip_inputs = self.clip_proc(
                text=[text], images=[open(img_path, "rb").read()], return_tensors="pt"
            ).to("cuda")
            out = self.clip_model(**clip_inputs)
            s = torch.cosine_similarity(
                out.image_embeds, out.text_embeds
            )[0].item()
            scores.append(s)
        return float(torch.tensor(scores).mean())

    def fid(self, imgs: List[Path], ref_dir: Path) -> float:
        metrics = calculate_metrics(
            input1=ref_dir, input2=imgs, cuda=True, isc=False, fid=True, kid=False
        )
        return float(metrics["frechet_inception_distance"])

    def run(
        self, prompts: List[str], ref_dir: Path, n_per_prompt: int = 2
    ) -> Dict[str, float]:
        imgs = self._generate(prompts, n_per_prompt)
        clip = self.clip_score(imgs, prompts)
        fid_val = self.fid(imgs, ref_dir)
        result = dict(CLIPScore=clip, FID=fid_val)
        out = Path(ref_dir).parent / "metrics.json"
        with open(out, "w") as fp:
            json.dump(result, fp, indent=2)
        return result