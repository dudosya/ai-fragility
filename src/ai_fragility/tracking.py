"""Run logging utilities for metrics and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import polars as pl
import torch
from torchvision.transforms import functional as TF

from ai_fragility.defenses import DefenseConfig
from ai_fragility.pipeline import AttackConfig, NoiseConfig, PipelineResult


@dataclass
class RunLogConfig:
    """Configuration for logging runs."""

    enable_wandb: bool = False
    wandb_project: str = "ai-fragility"
    wandb_entity: str | None = None
    duckdb_path: Path | None = Path("artifacts/runs.duckdb")
    run_name: str | None = None
    tags: list[str] | None = None


class RunLogger:
    """Logger that records runs to DuckDB and optionally Weights & Biases."""

    def __init__(self, config: RunLogConfig) -> None:
        self.config = config
        self._wandb_run = None
        if self.config.enable_wandb:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                tags=self.config.tags,
                settings=wandb.Settings(start_method="thread"),
            )

    def log(
        self,
        *,
        result: PipelineResult,
        attack_config: AttackConfig,
        defense_config: DefenseConfig,
        noise_config: NoiseConfig,
        latency_s: float,
        categories: list[str],
    ) -> None:
        """Log a single pipeline run."""

        record = self._build_record(
            result=result,
            attack_config=attack_config,
            defense_config=defense_config,
            noise_config=noise_config,
            latency_s=latency_s,
            categories=categories,
        )

        if self.config.duckdb_path is not None:
            self._append_duckdb(record)

        if self._wandb_run is not None:
            self._log_wandb(record, result)

    def _build_record(
        self,
        *,
        result: PipelineResult,
        attack_config: AttackConfig,
        defense_config: DefenseConfig,
        noise_config: NoiseConfig,
        latency_s: float,
        categories: list[str],
    ) -> dict[str, Any]:
        clean_top = _topk_labels(result.clean_logits, categories, k=3)
        attacked_top = _topk_labels(result.attacked_logits, categories, k=3)
        defended_top = _topk_labels(result.defended_logits, categories, k=3)

        record: dict[str, Any] = {
            "run_id": str(uuid4()),
            "ts": datetime.now(timezone.utc),
            "method": attack_config.method,
            "epsilon": float(attack_config.epsilon),
            "alpha": float(attack_config.alpha),
            "steps": int(attack_config.steps),
            "defense": defense_config.kind,
            "kernel_size": int(defense_config.kernel_size),
            "sigma": float(defense_config.sigma),
            "jpeg_quality": int(defense_config.jpeg_quality),
            "noise_sigma": float(noise_config.sigma),
            "noise_enabled": bool(noise_config.enabled),
            "clean_conf": float(result.security_gap.clean_confidence),
            "attacked_conf": float(result.security_gap.attacked_confidence),
            "defended_conf": float(result.security_gap.defended_confidence),
            "delta": float(result.security_gap.delta),
            "latency_ms": float(latency_s * 1000.0),
            "clean_top1": clean_top[0][0] if clean_top else "",
            "attacked_top1": attacked_top[0][0] if attacked_top else "",
            "defended_top1": defended_top[0][0] if defended_top else "",
        }

        return record

    def _append_duckdb(self, record: dict[str, Any]) -> None:
        path = self.config.duckdb_path
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(path))
        df = pl.DataFrame([record])
        con.register("log_df", df)
        con.execute("CREATE TABLE IF NOT EXISTS runs AS SELECT * FROM log_df WHERE FALSE")
        con.execute("INSERT INTO runs SELECT * FROM log_df")
        con.unregister("log_df")
        con.close()

    def _log_wandb(self, record: dict[str, Any], result: PipelineResult) -> None:
        import wandb

        metrics = {
            "clean_conf": record["clean_conf"],
            "attacked_conf": record["attacked_conf"],
            "defended_conf": record["defended_conf"],
            "delta": record["delta"],
            "latency_ms": record["latency_ms"],
        }

        images = {
            "clean": wandb.Image(TF.to_pil_image(result.clean_image[0].cpu().clamp(0.0, 1.0)), caption="clean"),
            "attacked": wandb.Image(TF.to_pil_image(result.attacked_image[0].cpu().clamp(0.0, 1.0)), caption="attacked"),
            "defended": wandb.Image(TF.to_pil_image(result.defended_image[0].cpu().clamp(0.0, 1.0)), caption="defended"),
        }

        self._wandb_run.log(metrics | images)


def _topk_labels(logits: torch.Tensor, categories: list[str], k: int = 3) -> list[tuple[str, float]]:
    probs = torch.softmax(logits, dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    labels: list[tuple[str, float]] = []
    for idx, val in zip(indices.tolist(), values.tolist()):
        label = categories[idx] if categories else str(idx)
        labels.append((label, float(val)))
    return labels


__all__ = ["RunLogConfig", "RunLogger"]
