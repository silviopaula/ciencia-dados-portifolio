"""Aplicação FastAPI — Studio de Séries Temporais SARIMA / SARIMAX.

Para rodar o backend localmente:
    cd app_forecast_ARIMA/backend
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd

from app.forecasting import build_excel_report, run_diagnostics, run_sarimax_forecast
from app.schemas import (
    DatasetSummary,
    DiagnosticsRequest,
    DiagnosticsResponse,
    ForecastRequest,
    ForecastResponse,
)

DEMO_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dados.xlsx"

app = FastAPI(
    title="SARIMAX Forecast Studio API",
    description="Backend analítico em Python (FastAPI + Statsmodels) para diagnóstico Box-Jenkins, modelagem SARIMA/SARIMAX e exportação de relatórios.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Armazenamento em memória por sessão (dataset_id -> (filename, DataFrame))
DATASETS: Dict[str, Tuple[str, pd.DataFrame]] = {}
LAST_REPORTS: Dict[str, Tuple[pd.DataFrame, ForecastResponse]] = {}


def _clean_json_value(val):
    """Garante compatibilidade estrita com JSON (converte NaN/Inf/Timestamp/NumPy com segurança)."""
    if val is None:
        return None
    if isinstance(val, (pd.Timestamp,)):
        return val.strftime("%Y-%m-%d")
    if hasattr(val, "isoformat"):
        try:
            return val.isoformat()[:10]
        except Exception:
            pass
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, (float,)):
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return round(val, 4)
    if hasattr(val, "item"):
        try:
            item = val.item()
            if isinstance(item, float) and (item != item or item in (float("inf"), float("-inf"))):
                return None
            return item
        except Exception:
            pass
    return val


def _to_numeric_flexible(series: pd.Series) -> pd.Series:
    """Converte série numérica aceitando tanto formato US (1234.56) quanto BR (1.234,56)."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() >= 0.3 or pd.api.types.is_numeric_dtype(series):
        return num
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    num_br = pd.to_numeric(cleaned, errors="coerce")
    return num_br if num_br.notna().mean() > num.notna().mean() else num


def _summarize_dataset(dataset_id: str, filename: str, df: pd.DataFrame) -> DatasetSummary:
    cols = [str(c) for c in df.columns]
    df.columns = cols
    date_candidates = []
    numeric_cols = []

    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in c.lower() for k in ["data", "date", "dt", "mes", "ano"]):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_candidates.append(c)
                continue
        num = _to_numeric_flexible(df[c])
        if num.notna().mean() > 0.3:
            df[c] = num
            numeric_cols.append(c)

    if not date_candidates:
        date_candidates = [cols[0]]
    default_date_col = date_candidates[0]
    default_target_col = numeric_cols[0] if numeric_cols else cols[-1]

    dates_parsed = pd.to_datetime(df[default_date_col], errors="coerce").dropna()
    obs_count = int(_to_numeric_flexible(df[default_target_col]).notna().sum())
    fut_count = int(len(df) - obs_count)

    preview_df = df.head(8).copy()
    preview_records = [
        {str(k): _clean_json_value(v) for k, v in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]

    return DatasetSummary(
        dataset_id=dataset_id,
        filename=filename,
        total_rows=int(len(df)),
        columns=cols,
        date_candidates=date_candidates,
        numeric_columns=numeric_cols,
        default_date_col=default_date_col,
        default_target_col=default_target_col,
        observed_rows=obs_count,
        future_exog_rows=fut_count,
        date_min=dates_parsed.min().strftime("%Y-%m-%d") if len(dates_parsed) else "",
        date_max=dates_parsed.max().strftime("%Y-%m-%d") if len(dates_parsed) else "",
        preview=preview_records,
    )


def _get_dataset(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset não encontrado na sessão. Carregue o demo ou envie um arquivo.")
    return DATASETS[dataset_id][1]


@app.get("/api/health")
def health():
    return {"status": "online", "service": "SARIMAX Forecast Studio API", "version": app.version}


@app.post("/api/dataset/demo", response_model=DatasetSummary)
def load_demo_dataset() -> DatasetSummary:
    """Carrega o dataset demonstrativo oficial (Consumo de Energia Elétrica + Variáveis Exógenas)."""
    if not DEMO_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Arquivo demo não encontrado: {DEMO_DATA_PATH}")
    df = pd.read_excel(DEMO_DATA_PATH)
    dataset_id = "demo-energia"
    DATASETS[dataset_id] = ("dados.xlsx (Consumo de Energia)", df)
    return _summarize_dataset(dataset_id, "dados.xlsx (Consumo de Energia)", df)


@app.post("/api/dataset/upload", response_model=DatasetSummary)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetSummary:
    """Recebe upload de planilha Excel (.xlsx/.xls) ou CSV e retorna metadados detectados."""
    content = await file.read()
    fname = file.filename or "dataset.csv"
    lower = fname.lower()
    try:
        if lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        elif lower.endswith(".csv"):
            df = None
            last_exc = None
            for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=enc)
                    break
                except Exception as exc_enc:
                    last_exc = exc_enc
            if df is None:
                raise last_exc or ValueError("Falha ao decodificar CSV.")
        else:
            raise HTTPException(status_code=400, detail="Formato não suportado. Envie .xlsx, .xls ou .csv.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {exc}") from exc

    if df.empty or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="O arquivo precisa conter ao menos 2 colunas (Data + Série Alvo).")

    dataset_id = str(uuid.uuid4())[:8]
    DATASETS[dataset_id] = (fname, df)
    return _summarize_dataset(dataset_id, fname, df)


@app.post("/api/diagnostics", response_model=DiagnosticsResponse)
def diagnostics_endpoint(req: DiagnosticsRequest) -> DiagnosticsResponse:
    """Executa diagnóstico Box-Jenkins (ADF, KPSS, ACF, PACF e recomendação de d/D)."""
    df = _get_dataset(req.dataset_id)
    try:
        return run_diagnostics(df, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/forecast", response_model=ForecastResponse)
def forecast_endpoint(req: ForecastRequest) -> ForecastResponse:
    """Ajusta modelo SARIMA/SARIMAX (Auto ou Manual), executa backtest e projeta o horizonte futuro."""
    df = _get_dataset(req.dataset_id)
    try:
        resp, df_export = run_sarimax_forecast(df, req)
        LAST_REPORTS[req.dataset_id] = (df_export, resp)
        return resp
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao estimar SARIMAX: {exc}") from exc


@app.get("/api/export/{dataset_id}")
def export_excel(dataset_id: str):
    """Exporta o último relatório de previsão da sessão em arquivo Excel (.xlsx) multi-aba."""
    if dataset_id not in LAST_REPORTS:
        raise HTTPException(status_code=404, detail="Execute uma previsão antes de exportar o relatório Excel.")
    df_export, resp = LAST_REPORTS[dataset_id]
    xlsx_bytes = build_excel_report(df_export, resp)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="relatorio_sarimax_forecast.xlsx"'},
    )
