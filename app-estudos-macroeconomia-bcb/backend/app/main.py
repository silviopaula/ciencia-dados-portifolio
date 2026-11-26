"""API REST FastAPI do Observatório Macro Brasil Studio (porta dedicada 8004)."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .catalog import STUDIES_BY_CODE, STUDIES_CATALOG, THEMES
from .data_store import get_store_stats
from .macro_models import (
    execute_study,
    get_conjuntura_dashboard,
    get_study_export_csv,
    sync_all_core_bcb,
)

APP_NAME = "Observatório Macro Brasil API"
APP_VERSION = "2.0.0"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Plataforma full-stack de Macroeconomia Aplicada & Econometria do Brasil: "
        "Painel de Conjuntura BCB (SGS + Olinda) com cache DuckDB/Parquet e 60 estudos autorais interativos "
        "(VAR/Granger/IRF, Engle-Granger/ECM, Filtro HP, SARIMA, Regra de Taylor HAC, Inércia Rolling AR(1) e EWMA)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunStudyPayload(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health", tags=["Sistema"])
def health_check() -> Dict[str, Any]:
    stats = get_store_stats()
    return {
        "status": "ok",
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "studies_count": len(STUDIES_CATALOG),
        "db_tables": stats["tables_count"],
        "bcb_series": stats["series_count"],
    }


@app.get("/api/conjuntura", tags=["Conjuntura Brasil"])
def get_conjuntura() -> Dict[str, Any]:
    return get_conjuntura_dashboard(force_live=False)


@app.post("/api/conjuntura/sync", tags=["Conjuntura Brasil"])
def sync_conjuntura() -> Dict[str, Any]:
    return sync_all_core_bcb()


@app.get("/api/studies", tags=["Catálogo de Estudos"])
def list_studies() -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for s in STUDIES_CATALOG:
        counts[s["theme"]] = counts.get(s["theme"], 0) + 1
    themes_out = [{**t, "count": counts.get(t["key"], 0)} for t in THEMES]
    return {
        "total": len(STUDIES_CATALOG),
        "specialized_count": sum(1 for s in STUDIES_CATALOG if s["specialized"]),
        "themes": themes_out,
        "db_stats": get_store_stats(),
        "items": STUDIES_CATALOG,
    }


@app.get("/api/studies/{code}", tags=["Catálogo de Estudos"])
def get_study(code: str) -> Dict[str, Any]:
    study = STUDIES_BY_CODE.get(code.upper())
    if not study:
        raise HTTPException(status_code=404, detail=f"Estudo '{code}' não encontrado.")
    return study


@app.post("/api/studies/{code}/run", tags=["Execução Econométrica"])
def run_study(code: str, body: RunStudyPayload) -> Dict[str, Any]:
    try:
        return execute_study(code, body.params)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao executar estudo {code}: {exc}") from exc


@app.get("/api/studies/{code}/export-csv", tags=["Exportação"])
def export_study_csv(code: str) -> Response:
    try:
        csv_bytes = get_study_export_csv(code)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="observatorio_macro_{code.lower()}.csv"'},
    )
