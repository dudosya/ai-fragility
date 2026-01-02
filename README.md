# AI Fragility

Minimal demo to expose adversarial attacks, simple defenses, and robustness metrics for vision models.

## Requirements

- Python 3.11+
- `uv` for dependency management
- PyTorch downloads ResNet18 weights on first run (expect a one-time wait)

## Install

```bash
uv sync --extra dev
```

## CLI

- Demo on a local image: `uv run python -m ai_fragility.cli demo path/to/image.jpg --method fgsm --epsilon 0.02 --save-dir artifacts/demo`
- Enable logging: add `--log --duckdb-path artifacts/runs.duckdb`
- Optional wandb: add `--wandb --wandb-project ai-fragility [--wandb-entity <entity>]`

Outputs: clean/attacked/defended images, perturbation heatmap, gradient map, security gap.

## API / Web UI

- Start: `uv run uvicorn ai_fragility.api:app --reload --host 0.0.0.0 --port 8000`
- Use: open http://localhost:8000/, upload an image, adjust attack/defense/noise controls. Tick "Log run" to write to DuckDB.
- Logs page: http://localhost:8000/logs shows recent DuckDB rows.

## Logging

- DuckDB: `artifacts/runs.duckdb` (CLI flag `--log`, UI checkbox). `/logs` lists recent runs.
- wandb: CLI-only (flags above). UI logging is DuckDB-only.

## Tasks (Justfile)

- `just` (lists tasks) — requires the `just` binary on PATH (e.g., `choco install just` or `scoop install just`).
- `just lint` → `ruff check`
- `just fmt` → `ruff format`
- `just test` → `pytest`
- `just serve` → run API with reload
- `just demo` → run CLI demo

## Tests

`uv run pytest`

## Troubleshooting

- Install deps with `uv sync --extra dev` to get `ruff`, `pytest`, `pyarrow`, `pytz`.
- First model run downloads weights; wait for completion.
- Upload limit is 10 MB; invalid images return 400.
