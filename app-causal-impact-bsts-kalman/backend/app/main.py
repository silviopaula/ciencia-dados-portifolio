"""API REST FastAPI do CausalImpact Studio — Inferência Causal em Séries Temporais (BSTS)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .causal_engine import CausalImpactEngine
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DatasetSummary,
    DemoDatasetMeta,
    HealthResponse,
)

APP_NAME = "CausalImpact Studio API"
APP_VERSION = "2.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

engine = CausalImpactEngine(data_dir=DATA_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Carrega automaticamente o dataset demo principal na inicialização
    if engine.current_df is None and engine.list_demos():
        engine.load_demo("marketing_campanha")
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Backend Python (FastAPI) para Inferência Causal em Séries Temporais (CausalImpact / BSTS), "
        "com modelos estruturais em espaço de estados (Statsmodels UnobservedComponents), "
        "simulação posterior conjunta Monte Carlo, auditoria No-Spillover, Teste Placebo no Tempo e exportação Excel."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["Sistema"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=APP_NAME,
        version=APP_VERSION,
        dataset_loaded=engine.current_df is not None,
        current_dataset_name=engine.current_meta.name if engine.current_meta else None,
        demos_available=len(engine.list_demos()),
    )


@app.get("/api/demos", response_model=List[DemoDatasetMeta], tags=["Dados"])
def list_demos() -> List[DemoDatasetMeta]:
    return engine.list_demos()


@app.get("/api/dataset/current", response_model=DatasetSummary, tags=["Dados"])
def get_current_dataset() -> DatasetSummary:
    if engine.current_meta is None:
        raise HTTPException(status_code=404, detail="Nenhum dataset carregado no momento.")
    return engine.current_meta


@app.post("/api/dataset/demo/{demo_id}", response_model=DatasetSummary, tags=["Dados"])
def load_demo_dataset(demo_id: str) -> DatasetSummary:
    try:
        return engine.load_demo(demo_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/dataset/upload", response_model=DatasetSummary, tags=["Dados"])
async def upload_dataset(file: UploadFile = File(...)) -> DatasetSummary:
    fname = file.filename or "dados.csv"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="O arquivo excede o limite de 25 MB.")
    try:
        return engine.load_uploaded_file(fname, raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Inferência Causal"])
def run_analysis(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return engine.analyze(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao estimar modelo BSTS: {exc}") from exc


@app.get("/api/export/excel", tags=["Exportação"])
def export_excel() -> Response:
    try:
        xlsx_bytes = engine.build_excel_report()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="relatorio_causal_impact.xlsx"'},
    )


@app.get("/api/export/csv", tags=["Exportação"])
def export_csv(freq: str = Query(default="original", pattern="^(original|month|quarter|semester|year)$")) -> Response:
    if engine.last_analysis is None:
        raise HTTPException(status_code=400, detail="Execute uma análise antes de exportar CSV.")
    rows = engine.last_analysis.aggregated_tables.get(freq, [])
    lines = ["periodo,n_obs,realizado,contrafactual,impacto_abs,impacto_pct,impacto_acumulado,ic_inf,ic_sup"]
    for r in rows:
        lines.append(
            f"{r.period_label},{r.n_obs},{r.actual_sum},{r.predicted_sum},"
            f"{r.abs_effect_sum},{r.rel_effect_pct},{r.cum_effect},{r.lower_effect_sum},{r.upper_effect_sum}"
        )
    return Response(
        content="\n".join(lines).encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="impacto_causal_{freq}.csv"'},
    )


@app.get("/api/export/report", tags=["Exportação"])
def export_report_md() -> Response:
    if engine.last_analysis is None:
        raise HTTPException(status_code=400, detail="Execute uma análise antes de exportar o relatório.")
    return Response(
        content=engine.last_analysis.detailed_report_md.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="relatorio_causal_impact.md"'},
    )


@app.get("/api/export/template", tags=["Exportação"])
def export_template_excel() -> Response:
    xlsx_bytes = engine.build_template_excel()
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="modelo_dados_causalimpact.xlsx"'},
    )
