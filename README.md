# AI Fragility: Adversarial Assurance Demo

Interactive framework to visualize adversarial fragility of vision models, apply classic signal-processing defenses, and quantify recovery.

## Stack

- Python 3.11+, managed with uv
- PyTorch + torchvision + Lightning
- Typer CLI, FastAPI service, Hydra configs
- Polars + DuckDB for tabular logging
- wandb for experiment tracking (optional)

## Quickstart

1. Install dependencies and the local package: `uv sync && uv pip install -e .`
2. Run a demo attack/defense (use a JPEG/PNG image path): `uv run python -m ai_fragility.cli demo path/to/image.jpg --save-dir artifacts/demo`
   - Optional logging: add `--log --duckdb-path artifacts/runs.duckdb` (and `--wandb --wandb-project ai-fragility` if you want wandb)
3. Start the API + UI: `uv run uvicorn ai_fragility.api:app --reload --host 0.0.0.0 --port 8000`
   - Open http://localhost:8000/, upload an image, tweak attack/defense sliders, and optionally tick "Log run" to append to DuckDB.

Outputs include clean/attacked/defended images, perturbation heatmap, gradient map, and a security gap metric (defended vs attacked confidence on the clean top-1 class).

## Project Layout

- `src/ai_fragility/` core code (pipeline, attacks, defenses, CLI, API)
- `configs/` Hydra configuration (model/attack/defense defaults)
- `Justfile` common tasks (`just lint`, `just test`, `just serve`)

## Logging

- DuckDB: written to `artifacts/runs.duckdb` via CLI `--log` or UI checkbox.
- wandb: enable with CLI flags `--wandb --wandb-project <project> [--wandb-entity <entity>]` (UI logging currently DuckDB-only).

## Web UI

- Start server: `uv run uvicorn ai_fragility.api:app --reload --host 0.0.0.0 --port 8000`
- Visit http://localhost:8000/ and adjust epsilon/steps/defense; outputs render inline (clean/attacked/defended, noise, gradient, top-3, security gap).

## Next Steps

- Expand defenses (denoise, spectral filters, wavelet)
- Add download buttons for rendered outputs and latency display
- Harden logging (batch writes, richer metadata) and add more explainability views
