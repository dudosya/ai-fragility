"""FastAPI service exposing the attack/defense pipeline with a lightweight web UI."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import List

import torch
import duckdb
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image

from ai_fragility.defenses import DefenseConfig
from ai_fragility.pipeline import AttackConfig, AttackDefensePipeline, NoiseConfig
from ai_fragility.tracking import RunLogConfig, RunLogger
from ai_fragility.viz import gradient_to_heatmap


class Prediction(BaseModel):
    label: str
    confidence: float


class SecurityGapModel(BaseModel):
    clean_confidence: float
    attacked_confidence: float
    defended_confidence: float
    delta: float


class InferenceResponse(BaseModel):
    clean: List[Prediction]
    noisy: List[Prediction] | None
    attacked: List[Prediction]
    defended: List[Prediction]
    security_gap: SecurityGapModel
    clean_image: str
    noisy_image: str | None
    attacked_image: str
    defended_image: str
    noise_map: str
    gradient_map: str


attack_config = AttackConfig()
defense_config = DefenseConfig()
pipeline = AttackDefensePipeline(
    attack_config=attack_config,
    defense_config=defense_config,
    model_name="resnet18",
    seed=42,
)

app = FastAPI(title="AI Fragility API", version="0.1.0")
LOGGER = RunLogger(RunLogConfig(enable_wandb=False, duckdb_path=Path("artifacts/runs.duckdb")))


def _to_pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if len(data) > 10 * 1024 * 1024:
        from fastapi import HTTPException

        raise HTTPException(status_code=413, detail="File too large; limit 10MB")
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Invalid image upload") from exc


def _topk(logits: torch.Tensor, categories: list[str], k: int = 3) -> list[Prediction]:
    probs = torch.softmax(logits, dim=-1)[0]
    values, indices = torch.topk(probs, k=k)
    preds: list[Prediction] = []
    for idx, val in zip(indices.tolist(), values.tolist()):
        label = categories[idx] if categories else str(idx)
        preds.append(Prediction(label=label, confidence=val))
    return preds


def _tensor_to_data_url(tensor_or_image: torch.Tensor | Image.Image) -> str:
    if isinstance(tensor_or_image, Image.Image):
        pil_img = tensor_or_image
    else:
        clamped = tensor_or_image.detach().cpu().clamp(0.0, 1.0)
        pil_img = Image.fromarray((clamped.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_table(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    header = "".join(f"<th>{col}</th>" for col in columns)
    body = "".join("<tr>" + "".join(f"<td>{val}</td>" for val in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


UI_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>AI Fragility Demo</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
        h1 { margin-bottom: 8px; }
        .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
        label { display: block; margin-top: 12px; font-weight: 600; }
        input[type="file"], select, input[type="number"], input[type="range"] { width: 100%; margin-top: 6px; }
        button { margin-top: 16px; padding: 12px 16px; background: #22d3ee; color: #0f172a; border: none; border-radius: 8px; font-weight: 700; cursor: pointer; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
        .img-box { background: #0b1220; border: 1px solid #1f2937; border-radius: 10px; padding: 8px; text-align: center; }
        img { max-width: 100%; border-radius: 6px; }
        table { width: 100%; border-collapse: collapse; margin-top: 8px; }
        th, td { padding: 6px; text-align: left; border-bottom: 1px solid #1f2937; }
        .pill { display: inline-block; padding: 4px 8px; background: #1f2937; border-radius: 999px; font-size: 12px; }
        .row { display: flex; gap: 12px; flex-wrap: wrap; }
        .row > div { flex: 1 1 160px; }
        .muted { color: #94a3b8; font-size: 12px; }
    </style>
</head>
<body>
    <h1>AI Fragility: Attack & Defense</h1>
    <div class=\"card\">
        <form id=\"form\">
            <label>Input image (jpg/png)
                <input type=\"file\" name=\"file\" accept=\"image/*\" required />
            </label>
            <div class=\"row\">
                <div>
                    <label>Attack method
                        <select name=\"method\">
                            <option value=\"fgsm\">FGSM</option>
                            <option value=\"pgd\">PGD</option>
                        </select>
                    </label>
                </div>
                <div>
                    <label>Epsilon
                        <input type=\"range\" name=\"epsilon\" min=\"0.0\" max=\"0.2\" step=\"0.005\" value=\"0.02\" oninput=\"epsilonVal.textContent=this.value\" />
                        <div class=\"pill\">ε = <span id=\"epsilonVal\">0.02</span></div>
                    </label>
                </div>
                <div>
                    <label>Steps (PGD)
                        <input type=\"number\" name=\"steps\" min=\"1\" max=\"100\" value=\"20\" />
                    </label>
                </div>
                <div>
                    <label>Alpha (PGD)
                        <input type=\"number\" name=\"alpha\" step=\"0.001\" value=\"0.005\" />
                    </label>
                </div>
            </div>
            <div class=\"row\">
                <div>
                    <label>Defense
                        <select name=\"defense\">
                            <option value=\"gaussian\">Gaussian</option>
                            <option value=\"median\">Median</option>
                            <option value=\"jpeg\">JPEG</option>
                            <option value=\"identity\">Identity (off)</option>
                        </select>
                    </label>
                </div>
                <div>
                    <label>Sigma / kernel
                        <input type=\"number\" name=\"sigma\" step=\"0.1\" value=\"1.0\" />
                    </label>
                </div>
                <div>
                    <label>Kernel size (median)
                        <input type=\"number\" name=\"kernel_size\" min=\"1\" step=\"2\" value=\"3\" />
                    </label>
                </div>
                <div>
                    <label>JPEG quality
                        <input type=\"number\" name=\"jpeg_quality\" min=\"30\" max=\"95\" value=\"75\" />
                    </label>
                </div>
            </div>
            <div class="row">
                <div>
                    <label>Noise sigma (random)
                        <input type="range" name="noise_sigma" min="0.0" max="0.2" step="0.005" value="0.02" oninput="noiseVal.textContent=this.value" />
                        <div class="pill">σ = <span id="noiseVal">0.02</span></div>
                    </label>
                </div>
                <div class="muted">Noisy view helps contrast random vs adversarial perturbations.</div>
            </div>
            <button type="submit">Run attack & defense</button>
            <span id="status" class="pill"></span>
            <label style="margin-top:8px;"><input type="checkbox" name="log" /> Log run (DuckDB)</label>
    </div>

    <div id=\"results\" class=\"grid\" style=\"display:none;\">
        <div class=\"img-box\"><h3>Clean</h3><img id=\"imgClean\" alt=\"Clean\" /></div>
        <div class=\"img-box\"><h3>Attacked</h3><img id=\"imgAttack\" alt=\"Attacked\" /></div>
        <div class="img-box"><h3>Noisy</h3><img id="imgNoisy" alt="Noisy" /></div>
        <div class=\"img-box\"><h3>Defended</h3><img id=\"imgDefended\" alt=\"Defended\" /></div>
        <div class=\"img-box\"><h3>Noise</h3><img id=\"imgNoise\" alt=\"Noise\" /></div>
        <div class=\"img-box\"><h3>Gradient</h3><img id=\"imgGrad\" alt=\"Gradient\" /></div>
    </div>

    <div id=\"tables\" class=\"grid\" style=\"display:none;\">
        <div class=\"card\"><h3>Clean top-3</h3><table id=\"tableClean\"></table></div>
        <div class=\"card\"><h3>Attacked top-3</h3><table id=\"tableAttack\"></table></div>
        <div class="card"><h3>Noisy top-3</h3><table id="tableNoisy"></table></div>
        <div class=\"card\"><h3>Defended top-3</h3><table id=\"tableDefended\"></table></div>
        <div class=\"card\"><h3>Security Gap</h3><div id=\"gap\"></div></div>
        <div class=\"card\"><h3>Logs</h3><a href=\"/logs\" style=\"color:#22d3ee;\">View latest DuckDB runs</a></div>
    </div>

    <script>
        const form = document.getElementById('form');
        const statusEl = document.getElementById('status');
        const resultsEl = document.getElementById('results');
        const tablesEl = document.getElementById('tables');

        function renderTable(el, preds) {
            el.innerHTML = '<tr><th>Rank</th><th>Class</th><th>Conf</th></tr>' +
                preds.map((p,i)=>`<tr><td>${i+1}</td><td>${p.label}</td><td>${p.confidence.toFixed(4)}</td></tr>`).join('');
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = new FormData(form);
            statusEl.textContent = 'Running...';
            statusEl.style.background = '#22d3ee';
            try {
                const res = await fetch('/infer', { method: 'POST', body: data });
                if (!res.ok) throw new Error(await res.text());
                const payload = await res.json();
                document.getElementById('imgClean').src = payload.clean_image;
                if (payload.noisy_image) document.getElementById('imgNoisy').src = payload.noisy_image;
                document.getElementById('imgAttack').src = payload.attacked_image;
                document.getElementById('imgDefended').src = payload.defended_image;
                document.getElementById('imgNoise').src = payload.noise_map;
                document.getElementById('imgGrad').src = payload.gradient_map;
                renderTable(document.getElementById('tableClean'), payload.clean);
                renderTable(document.getElementById('tableNoisy'), payload.noisy || []);
                renderTable(document.getElementById('tableAttack'), payload.attacked);
                renderTable(document.getElementById('tableDefended'), payload.defended);
                const g = payload.security_gap;
                document.getElementById('gap').innerHTML = `Clean: ${g.clean_confidence.toFixed(4)}<br/>Attacked: ${g.attacked_confidence.toFixed(4)}<br/>Defended: ${g.defended_confidence.toFixed(4)}<br/>Delta: ${g.delta.toFixed(4)}`;
                resultsEl.style.display = 'grid';
                tablesEl.style.display = 'grid';
                statusEl.textContent = 'Done';
                statusEl.style.background = '#16a34a';
            } catch(err) {
                statusEl.textContent = 'Error';
                statusEl.style.background = '#dc2626';
                alert(err);
            }
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content=UI_HTML)


@app.get("/logs", response_class=HTMLResponse)
def logs() -> HTMLResponse:
    path = LOGGER.config.duckdb_path
    if path is None:
        return HTMLResponse("<h2>DuckDB logging is disabled.</h2>")
    if not path.exists():
        return HTMLResponse(f"<h2>No DuckDB file found at {path}</h2>")

    con = duckdb.connect(str(path))
    try:
        query = (
            "SELECT ts, method, epsilon, steps, defense, noise_sigma, delta, latency_ms "
            "FROM runs ORDER BY ts DESC LIMIT 50"
        )
        res = con.execute(query)
        rows = res.fetchall()
        if not rows:
            return HTMLResponse("<h2>No runs logged yet.</h2>")
        columns = [col[0] for col in res.description]
    except duckdb.CatalogException:
        return HTMLResponse("<h2>No runs table yet.</h2>")
    finally:
        con.close()

    table_html = _render_table(columns, rows)
    page = f"""
    <html><head><title>Recent runs</title>
    <style>body{{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;}} table{{width:100%;border-collapse:collapse;}} th,td{{padding:8px;border-bottom:1px solid #1f2937;text-align:left;}}</style>
    </head><body>
    <h2>Recent runs (latest 50)</h2>
    {table_html}
    </body></html>
    """
    return HTMLResponse(content=page)


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),
    method: str = Form("fgsm"),
    epsilon: float = Form(0.02),
    alpha: float = Form(0.005),
    steps: int = Form(20),
    defense: str = Form("gaussian"),
    kernel_size: int = Form(3),
    sigma: float = Form(1.0),
    jpeg_quality: int = Form(75),
    noise_sigma: float = Form(0.02),
    log: bool = Form(False),
) -> InferenceResponse:
    image = _to_pil(file)
    start = perf_counter()

    attack_cfg = AttackConfig(method=method, epsilon=epsilon, alpha=alpha, steps=steps)
    defense_cfg = DefenseConfig(kind=defense, kernel_size=kernel_size, sigma=sigma, jpeg_quality=jpeg_quality)
    noise_cfg = NoiseConfig(sigma=noise_sigma, enabled=noise_sigma > 0)

    result = pipeline.run(image, attack_config=attack_cfg, defense_config=defense_cfg, noise_config=noise_cfg)
    latency_s = perf_counter() - start

    clean_pred = _topk(result.clean_logits, pipeline.bundle.categories)
    noisy_pred = _topk(result.noisy_logits, pipeline.bundle.categories) if result.noisy_logits is not None else []
    attacked_pred = _topk(result.attacked_logits, pipeline.bundle.categories)
    defended_pred = _topk(result.defended_logits, pipeline.bundle.categories)

    clean_img = _tensor_to_data_url(result.clean_image[0])
    noisy_img = _tensor_to_data_url(result.noisy_image[0]) if result.noisy_image is not None else None
    attacked_img = _tensor_to_data_url(result.attacked_image[0])
    defended_img = _tensor_to_data_url(result.defended_image[0])
    noise_map = _tensor_to_data_url(gradient_to_heatmap(result.perturbation[0]).convert("RGB"))
    gradient_map = _tensor_to_data_url(gradient_to_heatmap(result.gradient).convert("RGB"))

    if log:
        LOGGER.log(
            result=result,
            attack_config=attack_cfg,
            defense_config=defense_cfg,
            noise_config=noise_cfg,
            latency_s=latency_s,
            categories=pipeline.bundle.categories,
        )

    return InferenceResponse(
        clean=clean_pred,
        noisy=noisy_pred if noisy_pred else None,
        attacked=attacked_pred,
        defended=defended_pred,
        security_gap=SecurityGapModel(**result.security_gap.__dict__),
        clean_image=clean_img,
        noisy_image=noisy_img,
        attacked_image=attacked_img,
        defended_image=defended_img,
        noise_map=noise_map,
        gradient_map=gradient_map,
    )
