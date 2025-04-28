# fluxft/cli.py
import typer, json
from pathlib import Path
import yaml
from .config import GlobalConfig
from .train.trainer import LoRATrainer
from .search.hp_search import run_search
from .eval.evaluator import MetricComputer

app = typer.Typer(help="LoRA fine-tuning CLI")

@app.command()
def finetune(cfg_path: Path = typer.Option(..., exists=True)):
    with open(cfg_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    cfg = GlobalConfig.model_validate(yaml_data)
    trainer = LoRATrainer(cfg)
    res = trainer.train()
    typer.echo(json.dumps(res, indent=2))

@app.command()
def evaluate(cfg_path: Path, lora_path: Path, prompts_file: Path):
    with open(cfg_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    cfg = GlobalConfig.model_validate(yaml_data)
    prompts = [p.strip() for p in prompts_file.read_text().splitlines()]
    evaluator = MetricComputer(cfg, lora_path)
    res = evaluator.run(prompts, cfg.data.data_dir)
    typer.echo(res)

@app.command()
def search(cfg_path: Path, trials: int = 10):
    with open(cfg_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    cfg = GlobalConfig.model_validate(yaml_data)
    run_search(cfg, trials)

if __name__ == "__main__":
    app()