set shell := ["cmd.exe", "/C"]

default:
	@just -l

install:
	uv sync
	uv pip install -e .

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

test:
	uv run pytest

serve:
	uv run uvicorn ai_fragility.api:app --reload --host 0.0.0.0 --port 8000

demo image="examples/cat.jpg" method="fgsm" epsilon="0.02":
	uv run python -m ai_fragility.cli demo {{image}} --method {{method}} --epsilon {{epsilon}} --save-dir artifacts/demo
