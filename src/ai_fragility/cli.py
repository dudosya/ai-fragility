"""Typer CLI for the AI fragility demo."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import torch
import typer
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rich.console import Console
from rich.table import Table
from torchvision.transforms import functional as TF

from ai_fragility.defenses import DefenseConfig
from ai_fragility.pipeline import AttackConfig, AttackDefensePipeline
from ai_fragility.tracking import RunLogConfig, RunLogger
from ai_fragility.viz import gradient_to_heatmap

console = Console()
app = typer.Typer(help="Adversarial fragility playground")


def _load_config(config_dir: Path, config_name: str = "config"):
    """Load Hydra config from disk."""

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.3"):
        cfg = compose(config_name=config_name)
    return cfg


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.cpu().clamp(0.0, 1.0)).save(path)


def _print_topk(stage: str, logits: torch.Tensor, categories: list[str], k: int = 3) -> None:
    probs = torch.softmax(logits, dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    table = Table(title=f"{stage} top-{k}")
    table.add_column("Rank")
    table.add_column("Class")
    table.add_column("Confidence", justify="right")
    for rank, idx, val in zip(range(1, k + 1), indices.tolist(), values.tolist()):
        label = categories[idx] if categories else str(idx)
        table.add_row(str(rank), label, f"{val:.4f}")
    console.print(table)


@app.command()
def demo(
    image_path: Path = typer.Argument(..., help="Path to an input image"),
    method: str = typer.Option("fgsm", help="Attack method: fgsm or pgd"),
    epsilon: float = typer.Option(0.02, help="Perturbation magnitude"),
    alpha: float = typer.Option(0.005, help="PGD step size"),
    steps: int = typer.Option(20, help="PGD steps"),
    defense: str = typer.Option("gaussian", help="Defense kind: gaussian|median|jpeg|identity"),
    kernel_size: int = typer.Option(3, help="Median kernel size"),
    sigma: float = typer.Option(1.0, help="Gaussian blur sigma"),
    jpeg_quality: int = typer.Option(75, help="JPEG quality for defense"),
    save_dir: Optional[Path] = typer.Option(None, help="Where to save visualizations"),
    config_dir: Optional[Path] = typer.Option(None, help="Hydra config directory override"),
    config_name: str = typer.Option("config", help="Hydra config name"),
    log: bool = typer.Option(False, "--log", help="Log run to DuckDB/wandb"),
    duckdb_path: Path = typer.Option(Path("artifacts/runs.duckdb"), help="DuckDB file for run logs"),
    wandb: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
    wandb_project: str = typer.Option("ai-fragility", help="wandb project name"),
    wandb_entity: Optional[str] = typer.Option(None, help="wandb entity"),
) -> None:
    """Run a single attack/defense demo on an image."""

    # Default configs live at repository_root/configs (one level above src)
    cfg_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "configs"
    cfg = _load_config(config_dir=cfg_dir, config_name=config_name)

    attack_config = AttackConfig(
        method=method,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
    )
    defense_config = DefenseConfig(
        kind=defense,
        kernel_size=kernel_size,
        sigma=sigma,
        jpeg_quality=jpeg_quality,
    )

    pipeline = AttackDefensePipeline(
        attack_config=attack_config,
        defense_config=defense_config,
        model_name=cfg.model.name,
        seed=cfg.seed,
    )

    start = perf_counter()
    result = pipeline.run(image_path)
    latency_s = perf_counter() - start

    console.rule("Predictions")
    _print_topk("Clean", result.clean_logits, pipeline.bundle.categories)
    _print_topk("Attacked", result.attacked_logits, pipeline.bundle.categories)
    _print_topk("Defended", result.defended_logits, pipeline.bundle.categories)

    console.rule("Security Gap")
    console.print(
        f"Clean: {result.security_gap.clean_confidence:.4f} | "
        f"Attacked: {result.security_gap.attacked_confidence:.4f} | "
        f"Defended: {result.security_gap.defended_confidence:.4f} | "
        f"Delta: {result.security_gap.delta:+.4f}"
    )

    if save_dir:
        _save_tensor_image(result.clean_image[0], save_dir / "clean.png")
        _save_tensor_image(result.attacked_image[0], save_dir / "attacked.png")
        _save_tensor_image(result.defended_image[0], save_dir / "defended.png")
        perturbation_heatmap = gradient_to_heatmap(result.perturbation[0])
        gradient_heatmap = gradient_to_heatmap(result.gradient)
        save_dir.mkdir(parents=True, exist_ok=True)
        perturbation_heatmap.save(save_dir / "noise.png")
        gradient_heatmap.save(save_dir / "gradient.png")
        console.print(f"Saved outputs to {save_dir}")

    if log:
        logger = RunLogger(
            RunLogConfig(
                enable_wandb=wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                duckdb_path=duckdb_path,
            )
        )
        logger.log(
            result=result,
            attack_config=attack_config,
            defense_config=defense_config,
            latency_s=latency_s,
            categories=pipeline.bundle.categories,
        )
        if duckdb_path:
            console.print(f"Logged run to {duckdb_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
