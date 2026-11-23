"""
Unified Time Series Analytics & Forecasting Platform — FastAPI Backend
Serves both the REST API (/api/*) and the pure HTML5/CSS3/JS SPA frontend (/).
"""
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.analytics import (
    compute_descriptive_stats,
    correct_outliers,
    generate_multi_resolution_views,
    identify_outliers,
    parse_date_column,
    run_seasonality_and_spc_analysis,
    safe_float,
)
from backend.forecasting import (
    run_arima_forecast,
    run_autots_forecast,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_EXCEL_PATH = BASE_DIR / "data" / "ei_total_(PA).xlsx"
if not SAMPLE_EXCEL_PATH.exists():
    SAMPLE_EXCEL_PATH = BASE_DIR / "arquivos_para_excluir" / "page_4_forecast_TS" / "www" / "ei_total_(PA).xlsx"

app = FastAPI(
    title="TimeSeries Studio Pro — Unified Analytics & Forecasting API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# IN-MEMORY WORKSPACE SESSION STATE
# ==============================================================================

class WorkspaceState:
    def __init__(self):
        self.raw_bytes: Optional[bytes] = None
        self.filename: str = ""
        self.file_type: str = "xlsx"
        self.sheets: List[str] = []
        self.selected_sheet: str = ""
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.date_col: str = "Data"
        self.original_date_col: str = "data"
        self.date_format: str = "auto"

        # Module 1: Outlier state & undo history
        self.outlier_col: Optional[str] = None
        self.outlier_ident_method: Optional[str] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self.outlier_history: List[Dict[str, Any]] = []

        # Module 3: Advanced Editor state & undo stack
        self.editor_col: Optional[str] = None
        self.editor_original_series: Optional[pd.DataFrame] = None
        self.editor_current_series: Optional[pd.DataFrame] = None
        self.editor_undo_stack: List[Dict[str, Any]] = []
        self.editor_log: List[Dict[str, Any]] = []

        # Module 4: Last ARIMA & AutoTS results for growth analysis & export
        self.last_arima_result: Optional[Dict[str, Any]] = None
        self.last_autots_result: Optional[Dict[str, Any]] = None


STATE = WorkspaceState()


def _detect_date_candidates(df: pd.DataFrame) -> List[str]:
    all_cols = list(df.columns)
    candidates = []
    for c in all_cols:
        if any(k in str(c).lower() for k in ["data", "date", "dt", "dia", "mes", "ano", "time", "periodo"]):
            candidates.append(c)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            candidates.append(c)
    ordered = candidates + [c for c in all_cols if c not in candidates]
    return ordered


def _load_excel_or_csv(content: bytes, filename: str, sheet_name: Optional[str] = None, csv_sep: Optional[str] = None):
    ext = filename.lower().split(".")[-1]
    STATE.raw_bytes = content
    STATE.filename = filename
    STATE.file_type = ext

    if ext in ("xlsx", "xls"):
        xl = pd.ExcelFile(io.BytesIO(content))
        STATE.sheets = xl.sheet_names
        chosen_sheet = sheet_name if (sheet_name and sheet_name in STATE.sheets) else STATE.sheets[0]
        STATE.selected_sheet = chosen_sheet
        df = xl.parse(chosen_sheet)
    elif ext == "csv":
        STATE.sheets = ["CSV_Data"]
        STATE.selected_sheet = "CSV_Data"
        text_sample = content[:4096].decode("utf-8", errors="ignore")
        sep = csv_sep if csv_sep else (";" if text_sample.count(";") > text_sample.count(",") else ",")
        df = pd.read_csv(io.BytesIO(content), sep=sep)
    else:
        raise ValueError("Formato de arquivo não suportado. Envie um arquivo .xlsx, .xls ou .csv.")

    STATE.raw_df = df
    # Auto-process default date column so the app is immediately usable
    date_candidates = _detect_date_candidates(df)
    default_date_col = date_candidates[0] if date_candidates else df.columns[0]
    _process_active_dataframe(default_date_col, "auto")


def _process_active_dataframe(date_column: str, date_format: str = "auto"):
    if STATE.raw_df is None:
        raise ValueError("Nenhum arquivo carregado.")
    df = STATE.raw_df.copy()
    if date_column not in df.columns:
        raise ValueError(f"Coluna '{date_column}' não encontrada.")

    parsed_dates = parse_date_column(df[date_column], date_format=date_format)
    if parsed_dates.isna().all():
        raise ValueError(f"Não foi possível converter a coluna '{date_column}' em datas válidas.")

    STATE.original_date_col = date_column
    STATE.date_format = date_format
    df["Data"] = parsed_dates
    if date_column != "Data":
        # Keep 'Data' as canonical date column at front
        cols = ["Data"] + [c for c in df.columns if c != "Data"]
        df = df[cols]

    df = df.sort_values("Data").reset_index(drop=True)
    STATE.df = df
    STATE.date_col = "Data"

    # Reset dependent states when dataset changes
    STATE.outlier_mask = None
    STATE.outlier_history = []
    STATE.editor_col = None
    STATE.editor_original_series = None
    STATE.editor_current_series = None
    STATE.editor_undo_stack = []
    STATE.editor_log = []


def _get_dataset_metadata() -> Dict[str, Any]:
    if STATE.df is None or STATE.raw_df is None:
        return {"loaded": False}
    df = STATE.df
    raw_cols = list(STATE.raw_df.columns)
    date_candidates = _detect_date_candidates(STATE.raw_df)
    num_cols = [
        c for c in df.columns
        if c != "Data" and c != STATE.original_date_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    valid_dates = df["Data"].dropna()
    min_date = valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) > 0 else None
    max_date = valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) > 0 else None
    years = sorted(valid_dates.dt.year.unique().tolist()) if len(valid_dates) > 0 else []

    # Preview rows
    preview_df = df.head(100).copy()
    preview_rows = []
    for _, r in preview_df.iterrows():
        row_dict: Dict[str, Any] = {}
        for col in preview_df.columns:
            val = r[col]
            if pd.isna(val):
                row_dict[col] = None
            elif isinstance(val, (pd.Timestamp, datetime)):
                row_dict[col] = val.strftime("%Y-%m-%d")
            elif isinstance(val, (float, np.floating)):
                row_dict[col] = safe_float(val)
            elif isinstance(val, (int, np.integer)):
                row_dict[col] = int(val)
            else:
                row_dict[col] = str(val)
        preview_rows.append(row_dict)

    # Column summaries
    col_stats = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        valid_s = s.dropna()
        non_na_dates = df.loc[s.notna(), "Data"]
        col_stats.append({
            "column": c,
            "non_null": int(len(valid_s)),
            "missing": int(s.isna().sum()),
            "mean": safe_float(valid_s.mean()) if len(valid_s) > 0 else None,
            "min": safe_float(valid_s.min()) if len(valid_s) > 0 else None,
            "max": safe_float(valid_s.max()) if len(valid_s) > 0 else None,
            "start_date": non_na_dates.min().strftime("%Y-%m-%d") if len(non_na_dates) > 0 else min_date,
            "end_date": non_na_dates.max().strftime("%Y-%m-%d") if len(non_na_dates) > 0 else max_date,
        })

    return {
        "loaded": True,
        "filename": STATE.filename,
        "file_type": STATE.file_type,
        "sheets": STATE.sheets,
        "selected_sheet": STATE.selected_sheet,
        "raw_columns": raw_cols,
        "date_candidates": date_candidates,
        "selected_date_column": STATE.original_date_col,
        "date_format": STATE.date_format,
        "numeric_columns": num_cols,
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "min_date": min_date,
        "max_date": max_date,
        "years": years,
        "column_summaries": col_stats,
        "preview_columns": list(preview_df.columns),
        "preview_rows": preview_rows,
    }


def _ensure_dataset_loaded():
    if STATE.df is None and SAMPLE_EXCEL_PATH.exists():
        try:
            content = SAMPLE_EXCEL_PATH.read_bytes()
            _load_excel_or_csv(content, SAMPLE_EXCEL_PATH.name)
        except Exception as e:
            print("Warning: failed to load sample dataset:", e)


# Auto-load the included sample dataset on module import & startup so the user can explore immediately
_ensure_dataset_loaded()


@app.on_event("startup")
def startup_load_sample():
    _ensure_dataset_loaded()


# ==============================================================================
# DATA & WORKSPACE ENDPOINTS
# ==============================================================================

class ConfigureDatasetRequest(BaseModel):
    sheet_name: Optional[str] = None
    date_column: str
    date_format: str = "auto"
    csv_separator: Optional[str] = None


@app.get("/api/dataset/info")
def get_dataset_info():
    if STATE.df is None and SAMPLE_EXCEL_PATH.exists():
        _load_excel_or_csv(SAMPLE_EXCEL_PATH.read_bytes(), SAMPLE_EXCEL_PATH.name)
    return _get_dataset_metadata()


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        _load_excel_or_csv(content, file.filename or "dataset.xlsx")
        return _get_dataset_metadata()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dataset/load-sample")
def load_sample_dataset():
    if not SAMPLE_EXCEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Arquivo de exemplo ei_total_(PA).xlsx não encontrado.")
    content = SAMPLE_EXCEL_PATH.read_bytes()
    _load_excel_or_csv(content, SAMPLE_EXCEL_PATH.name)
    return _get_dataset_metadata()


@app.post("/api/dataset/configure")
def configure_dataset(req: ConfigureDatasetRequest):
    if STATE.raw_bytes is None:
        raise HTTPException(status_code=400, detail="Nenhum arquivo carregado.")
    try:
        if req.sheet_name and req.sheet_name != STATE.selected_sheet and STATE.file_type in ("xlsx", "xls"):
            xl = pd.ExcelFile(io.BytesIO(STATE.raw_bytes))
            STATE.selected_sheet = req.sheet_name
            STATE.raw_df = xl.parse(req.sheet_name)
        elif req.csv_separator and STATE.file_type == "csv":
            STATE.raw_df = pd.read_csv(io.BytesIO(STATE.raw_bytes), sep=req.csv_separator)

        date_col = req.date_column if req.date_column in STATE.raw_df.columns else _detect_date_candidates(STATE.raw_df)[0]
        _process_active_dataframe(date_col, req.date_format)
        return _get_dataset_metadata()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dataset/export-sample")
def export_dataset_sample():
    if STATE.df is None:
        raise HTTPException(status_code=400, detail="Nenhum dado disponível.")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        STATE.df.to_excel(writer, index=False, sheet_name="Dados")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=dados_processados_{datetime.now().strftime('%Y%m%d')}.xlsx"}
    )


# ==============================================================================
# MODULE 1 ENDPOINTS: OUTLIERS (app_10_correcao_outliers)
# ==============================================================================

class OutlierIdentifyRequest(BaseModel):
    column: str
    method: str = "zscore"
    threshold: Optional[float] = None


class OutlierCorrectRequest(BaseModel):
    column: str
    method: str = "linear_interp"
    window_size: int = 5
    apply_to_workspace: bool = False


def _build_outlier_response(column: str) -> Dict[str, Any]:
    if STATE.df is None or column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna inválida ou dataset não carregado.")

    df = STATE.df
    original_arr = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    dates = [d.strftime("%Y-%m-%d") if pd.notna(d) else f"Obs {i+1}" for i, d in enumerate(df["Data"])]
    n_total = len(original_arr)
    na_count = int(np.isnan(original_arr).sum())

    if STATE.outlier_col != column or STATE.outlier_mask is None:
        mask = np.zeros(n_total, dtype=bool)
    else:
        mask = STATE.outlier_mask

    has_correction = len(STATE.outlier_history) > 0 and STATE.outlier_col == column
    treated_arr = STATE.outlier_history[-1]["result"] if has_correction else original_arr.copy()
    last_corr_method = STATE.outlier_history[-1]["method"] if has_correction else None

    n_outliers = int(mask.sum())
    valid_count = max(1, n_total - na_count)
    pct_outliers = round(100.0 * n_outliers / len(mask), 1) if len(mask) > 0 else 0.0

    orig_stats = compute_descriptive_stats(original_arr)
    treat_stats = compute_descriptive_stats(treated_arr)

    impact_mean = 0.0
    if has_correction and orig_stats["mean"] and treat_stats["mean"] and orig_stats["mean"] != 0:
        impact_mean = round(100.0 * abs(orig_stats["mean"] - treat_stats["mean"]) / abs(orig_stats["mean"]), 2)

    stat_labels = [
        ("mean", "Média"),
        ("median", "Mediana"),
        ("std", "Desvio Padrão"),
        ("min", "Mínimo"),
        ("max", "Máximo"),
        ("skewness", "Skewness (Assimetria)"),
        ("kurtosis", "Kurtosis (Curtose)"),
    ]
    stats_table = []
    for key, label in stat_labels:
        ov = orig_stats.get(key)
        tv = treat_stats.get(key) if has_correction else None
        var_pct = None
        if has_correction and ov is not None and tv is not None and ov != 0:
            var_pct = f"{round(100.0 * (tv - ov) / abs(ov), 1)}%"
        stats_table.append({
            "estatistica": label,
            "original": round(ov, 2) if ov is not None else None,
            "tratado": round(tv, 2) if (has_correction and tv is not None) else None,
            "variacao": var_pct if has_correction else "Pendente correção",
        })

    series_points = []
    for i in range(n_total):
        ov = safe_float(original_arr[i])
        tv = safe_float(treated_arr[i])
        series_points.append({
            "index": i + 1,
            "date": dates[i],
            "original": ov,
            "treated": tv,
            "is_outlier": bool(mask[i]),
            "is_na": bool(np.isnan(original_arr[i])),
        })

    return {
        "column": column,
        "ident_method": STATE.outlier_ident_method,
        "has_correction": has_correction,
        "correction_method": last_corr_method,
        "history_depth": len(STATE.outlier_history) if STATE.outlier_col == column else 0,
        "kpis": {
            "outliers_count": n_outliers,
            "na_count": na_count,
            "valid_count": valid_count,
            "percent_outliers": pct_outliers,
            "impact_mean_pct": impact_mean,
        },
        "stats_table": stats_table,
        "series": series_points,
    }


@app.post("/api/outliers/identify")
def api_identify_outliers(req: OutlierIdentifyRequest):
    if STATE.df is None or req.column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna não encontrada.")
    mask = identify_outliers(STATE.df, req.column, method=req.method, threshold=req.threshold)
    STATE.outlier_col = req.column
    STATE.outlier_ident_method = req.method
    STATE.outlier_mask = mask
    STATE.outlier_history = []
    return _build_outlier_response(req.column)


@app.post("/api/outliers/correct")
def api_correct_outliers(req: OutlierCorrectRequest):
    if STATE.df is None or req.column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna não encontrada.")
    if STATE.outlier_col != req.column or STATE.outlier_mask is None:
        # Auto-run default zscore if user didn't click identify first
        STATE.outlier_mask = identify_outliers(STATE.df, req.column, method="zscore", threshold=3.0)
        STATE.outlier_col = req.column
        STATE.outlier_ident_method = "zscore"

    base_arr = (
        STATE.outlier_history[-1]["result"]
        if (STATE.outlier_history and STATE.outlier_col == req.column)
        else pd.to_numeric(STATE.df[req.column], errors="coerce").to_numpy(dtype=float)
    )
    corrected = correct_outliers(
        base_arr,
        STATE.outlier_mask,
        method=req.method,
        window_size=req.window_size,
    )
    STATE.outlier_history.append({
        "method": req.method,
        "window_size": req.window_size,
        "result": corrected,
    })
    if req.apply_to_workspace:
        STATE.df[req.column] = corrected

    return _build_outlier_response(req.column)


@app.post("/api/outliers/undo")
def api_undo_outlier_correction(column: str = Query(...)):
    if STATE.outlier_col == column and len(STATE.outlier_history) > 0:
        STATE.outlier_history.pop()
    return _build_outlier_response(column)


@app.post("/api/outliers/commit")
def api_commit_outliers_to_workspace(column: str = Query(...)):
    """Saves the treated series into the active workspace dataframe so ARIMA/AutoTS can use it directly."""
    if STATE.df is None or column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna inválida.")
    if STATE.outlier_col == column and len(STATE.outlier_history) > 0:
        STATE.df[f"{column}_tratado"] = STATE.outlier_history[-1]["result"]
        return {"status": "ok", "new_column": f"{column}_tratado", "dataset": _get_dataset_metadata()}
    raise HTTPException(status_code=400, detail="Nenhuma correção ativa para salvar.")


@app.get("/api/outliers/export")
def api_export_outliers(column: str = Query(...)):
    if STATE.df is None or column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Dados não disponíveis.")
    export_df = STATE.df.copy()
    info = _build_outlier_response(column)
    if STATE.outlier_col == column and STATE.outlier_mask is not None:
        export_df[f"{column}_outlier"] = STATE.outlier_mask
    if STATE.outlier_col == column and len(STATE.outlier_history) > 0:
        export_df[f"{column}_tratado"] = STATE.outlier_history[-1]["result"]
        export_df[f"{column}_metodo"] = STATE.outlier_history[-1]["method"]
    else:
        export_df[f"{column}_metodo"] = "sem_correcao"

    meta_df = pd.DataFrame({
        "Informação": [
            "Data de Processamento",
            "Coluna Analisada",
            "Método de Identificação",
            "Método de Correção",
            "Quantidade de Outliers",
            "Impacto na Média (%)"
        ],
        "Valor": [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            column,
            str(STATE.outlier_ident_method or "Nenhum"),
            str(info["correction_method"] or "Nenhum"),
            str(info["kpis"]["outliers_count"]),
            str(info["kpis"]["impact_mean_pct"]),
        ]
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Dados_Tratados")
        meta_df.to_excel(writer, index=False, sheet_name="Metadados")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=dados_tratados_{column}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"}
    )


# ==============================================================================
# MODULE 2 ENDPOINTS: SEASONALITY, PARTICIPATION, XmR & STL ANOMALIES (app_5)
# ==============================================================================

class SeasonalityRequest(BaseModel):
    column: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    stl_threshold_sigma: float = 2.0


@app.post("/api/seasonality/analyze")
def api_analyze_seasonality(req: SeasonalityRequest):
    if STATE.df is None or req.column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna não encontrada no dataset.")
    try:
        return run_seasonality_and_spc_analysis(
            STATE.df,
            date_col="Data",
            value_col=req.column,
            start_year=req.start_year,
            end_year=req.end_year,
            stl_threshold_sigma=req.stl_threshold_sigma,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/seasonality/export")
def api_export_seasonality(
    column: str = Query(...),
    start_year: Optional[int] = Query(None),
    end_year: Optional[int] = Query(None)
):
    if STATE.df is None or column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna não encontrada.")
    res = run_seasonality_and_spc_analysis(STATE.df, "Data", column, start_year, end_year)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(res["limits_table"]).to_excel(writer, index=False, sheet_name="Participacao_Limites")
        pd.DataFrame(res["xmr"]["points"]).to_excel(writer, index=False, sheet_name="Controle_XmR")
        pd.DataFrame(res["stl"]["anomalies"]).to_excel(writer, index=False, sheet_name="Anomalias_STL")
        pd.DataFrame(res["stl"]["series"]).to_excel(writer, index=False, sheet_name="Decomposicao_STL")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=sazonalidade_xmr_{column}.xlsx"}
    )


# ==============================================================================
# MODULE 3 ENDPOINTS: ADVANCED INTERACTIVE SERIES EDITOR (page 6)
# ==============================================================================

class EditorInitRequest(BaseModel):
    column: str


class EditorModifyRequest(BaseModel):
    column: str
    indices: List[int]  # 0-based indices into the editor series
    change_type: str = "set"  # "set" (new exact value), "absolute" (+delta), "percentual" (+pct%)
    change_value: float


def _ensure_editor_initialized(column: str):
    if STATE.df is None or column not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Coluna não encontrada.")
    if STATE.editor_col != column or STATE.editor_current_series is None:
        sub = STATE.df[["Data", column]].copy()
        sub["Data"] = pd.to_datetime(sub["Data"], errors="coerce")
        sub["value"] = pd.to_numeric(sub[column], errors="coerce")
        sub = sub.dropna(subset=["Data", "value"]).sort_values("Data").reset_index(drop=True)
        STATE.editor_col = column
        STATE.editor_original_series = sub[["Data", "value"]].copy()
        STATE.editor_current_series = sub[["Data", "value"]].copy()
        STATE.editor_undo_stack = []
        STATE.editor_log = []


def _build_editor_payload(column: str) -> Dict[str, Any]:
    _ensure_editor_initialized(column)
    orig = STATE.editor_original_series
    curr = STATE.editor_current_series

    # Preparing `preparar_dataframe_final` from R app (Original vs Ajustada + MoM and YoY rates)
    comp = pd.DataFrame({
        "Data": orig["Data"],
        "Valor_Original": orig["value"].to_numpy(dtype=float),
        "Serie_Ajustada": curr["value"].to_numpy(dtype=float),
    })

    comp["Taxa_Crescimento_Original"] = comp["Valor_Original"].pct_change(1).replace([np.inf, -np.inf], np.nan) * 100.0
    comp["Taxa_Acrescimo_Original"] = comp["Valor_Original"].diff(1)
    comp["Taxa_Crescimento_Mes_Ano_Original"] = comp["Valor_Original"].pct_change(12).replace([np.inf, -np.inf], np.nan) * 100.0
    comp["Taxa_Acrescimo_Mes_Ano_Original"] = comp["Valor_Original"].diff(12)

    comp["Taxa_Crescimento_Ajustado"] = comp["Serie_Ajustada"].pct_change(1).replace([np.inf, -np.inf], np.nan) * 100.0
    comp["Taxa_Acrescimo_Ajustado"] = comp["Serie_Ajustada"].diff(1)
    comp["Taxa_Crescimento_Mes_Ano_Ajustado"] = comp["Serie_Ajustada"].pct_change(12).replace([np.inf, -np.inf], np.nan) * 100.0
    comp["Taxa_Acrescimo_Mes_Ano_Ajustado"] = comp["Serie_Ajustada"].diff(12)

    vals = comp["Serie_Ajustada"].dropna().to_numpy(dtype=float)
    kpis = {
        "num_periodos": int(len(comp)),
        "valor_max": safe_float(np.max(vals)) if len(vals) > 0 else None,
        "valor_min": safe_float(np.min(vals)) if len(vals) > 0 else None,
        "media_valor": safe_float(np.mean(vals)) if len(vals) > 0 else None,
        "modified_points_count": int(np.sum(np.abs(comp["Serie_Ajustada"] - comp["Valor_Original"]) > 1e-7)),
        "undo_available": len(STATE.editor_undo_stack),
    }

    series_rows = []
    for idx, r in comp.iterrows():
        is_mod = abs((r["Serie_Ajustada"] or 0.0) - (r["Valor_Original"] or 0.0)) > 1e-7
        series_rows.append({
            "index": int(idx),
            "Data": r["Data"].strftime("%Y-%m-%d"),
            "Valor_Original": safe_float(r["Valor_Original"]),
            "Serie_Ajustada": safe_float(r["Serie_Ajustada"]),
            "is_modified": bool(is_mod),
            "Taxa_Crescimento_Original": safe_float(r["Taxa_Crescimento_Original"]),
            "Taxa_Acrescimo_Original": safe_float(r["Taxa_Acrescimo_Original"]),
            "Taxa_Crescimento_Mes_Ano_Original": safe_float(r["Taxa_Crescimento_Mes_Ano_Original"]),
            "Taxa_Acrescimo_Mes_Ano_Original": safe_float(r["Taxa_Acrescimo_Mes_Ano_Original"]),
            "Taxa_Crescimento_Ajustado": safe_float(r["Taxa_Crescimento_Ajustado"]),
            "Taxa_Acrescimo_Ajustado": safe_float(r["Taxa_Acrescimo_Ajustado"]),
            "Taxa_Crescimento_Mes_Ano_Ajustado": safe_float(r["Taxa_Crescimento_Mes_Ano_Ajustado"]),
            "Taxa_Acrescimo_Mes_Ano_Ajustado": safe_float(r["Taxa_Acrescimo_Mes_Ano_Ajustado"]),
        })

    visoes = generate_multi_resolution_views(curr, "Data", "value")

    return {
        "column": column,
        "kpis": kpis,
        "series": series_rows,
        "modification_log": list(reversed(STATE.editor_log)),
        "visoes": visoes,
    }


@app.post("/api/editor/init")
def api_editor_init(req: EditorInitRequest):
    _ensure_editor_initialized(req.column)
    return _build_editor_payload(req.column)


@app.post("/api/editor/modify")
def api_editor_modify(req: EditorModifyRequest):
    _ensure_editor_initialized(req.column)
    curr = STATE.editor_current_series
    orig = STATE.editor_original_series

    valid_indices = [int(i) for i in req.indices if 0 <= int(i) < len(curr)]
    if not valid_indices:
        raise HTTPException(status_code=400, detail="Nenhum ponto válido selecionado.")

    # Save snapshot for single-step batch undo
    snapshot = []
    for idx in valid_indices:
        old_val = float(curr.loc[idx, "value"])
        snapshot.append((idx, old_val))
        if req.change_type == "set":
            new_val = float(req.change_value)
            tipo_label = "Valor Direto"
        elif req.change_type == "absolute":
            new_val = old_val + float(req.change_value)
            tipo_label = f"Absoluto ({req.change_value:+g})"
        else:  # percentual
            new_val = old_val * (1.0 + float(req.change_value) / 100.0)
            tipo_label = f"Percentual ({req.change_value:+g}%)"

        curr.loc[idx, "value"] = new_val
        STATE.editor_log.append({
            "Indice": idx + 1,
            "Data": curr.loc[idx, "Data"].strftime("%Y-%m-%d"),
            "Valor_Original": safe_float(orig.loc[idx, "value"]),
            "Valor_Anterior": safe_float(old_val),
            "Serie_Ajustada": safe_float(new_val),
            "Tipo": tipo_label,
        })

    STATE.editor_undo_stack.append({"changes": snapshot, "count": len(snapshot)})
    return _build_editor_payload(req.column)


@app.post("/api/editor/undo")
def api_editor_undo(column: str = Query(...)):
    _ensure_editor_initialized(column)
    if STATE.editor_undo_stack:
        last_batch = STATE.editor_undo_stack.pop()
        for idx, prev_val in last_batch["changes"]:
            STATE.editor_current_series.loc[idx, "value"] = prev_val
        for _ in range(min(last_batch["count"], len(STATE.editor_log))):
            STATE.editor_log.pop()
    return _build_editor_payload(column)


@app.post("/api/editor/reset")
def api_editor_reset(column: str = Query(...)):
    STATE.editor_col = None
    _ensure_editor_initialized(column)
    return _build_editor_payload(column)


@app.post("/api/editor/commit")
def api_editor_commit_to_workspace(column: str = Query(...)):
    """Saves the edited series back to the workspace DataFrame as `<column>_editado`."""
    _ensure_editor_initialized(column)
    curr = STATE.editor_current_series
    new_col = f"{column}_editado"
    lookup = dict(zip(curr["Data"], curr["value"]))
    STATE.df[new_col] = STATE.df["Data"].map(lookup)
    return {"status": "ok", "new_column": new_col, "dataset": _get_dataset_metadata()}


@app.get("/api/editor/export")
def api_editor_export(column: str = Query(...)):
    payload = _build_editor_payload(column)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(payload["series"]).drop(columns=["index", "is_modified"], errors="ignore").to_excel(
            writer, index=False, sheet_name="Series_Comparadas"
        )
        if payload["modification_log"]:
            pd.DataFrame(payload["modification_log"]).to_excel(writer, index=False, sheet_name="Log_Modificacoes")
        pd.DataFrame(payload["visoes"]["mensal"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Mensal_Valores")
        pd.DataFrame(payload["visoes"]["trimestral"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Trimestral_Valores")
        pd.DataFrame(payload["visoes"]["semestral"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Semestral_Valores")
        pd.DataFrame(payload["visoes"]["anual"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Anual_Valores")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=series_editadas_{column}_{datetime.now().strftime('%Y%m%d')}.xlsx"}
    )


# ==============================================================================
# MODULE 4 ENDPOINTS: FORECASTING (ARIMA UNI/MULTI + AUTOTS) & GROWTH ANALYSIS
# ==============================================================================

class ArimaForecastRequest(BaseModel):
    target_var: str
    exog_vars: Optional[List[str]] = None
    data_inicial: Optional[str] = None
    data_atual: Optional[str] = None
    end_predict: Optional[str] = None
    n_teste: int = 6
    seasonal: bool = True
    period: int = 12
    num_models: int = 5
    manual_mode: bool = False
    arima_params: Optional[Dict[str, int]] = None
    peso_recente: float = 2.0
    peso_desvio_max: float = 0.4


class AutoTSForecastRequest(BaseModel):
    variable: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    end_predict: Optional[str] = None
    selected_models: Optional[List[str]] = None
    periodo_maximo: int = 12
    peso_recente: float = 2.0
    peso_desvio_max: float = 0.4


class GrowthAnalysisRequest(BaseModel):
    source: str = "dataset"  # "dataset", "arima", "autots"
    model_or_col: str


@app.post("/api/forecast/arima")
def api_run_arima(req: ArimaForecastRequest):
    if STATE.df is None or req.target_var not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Variável alvo não encontrada.")
    try:
        res = run_arima_forecast(
            STATE.df,
            date_col="Data",
            target_var=req.target_var,
            exog_vars=req.exog_vars,
            data_inicial=req.data_inicial,
            data_atual=req.data_atual,
            end_predict=req.end_predict,
            n_teste=req.n_teste,
            seasonal=req.seasonal,
            period=req.period,
            num_models=req.num_models,
            manual_mode=req.manual_mode,
            arima_params=req.arima_params,
            peso_recente=req.peso_recente,
            peso_desvio_max=req.peso_desvio_max,
        )
        STATE.last_arima_result = res
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/forecast/autots")
def api_run_autots(req: AutoTSForecastRequest):
    if STATE.df is None or req.variable not in STATE.df.columns:
        raise HTTPException(status_code=400, detail="Variável não encontrada.")
    try:
        res = run_autots_forecast(
            STATE.df,
            date_col="Data",
            variable=req.variable,
            start_date=req.start_date,
            end_date=req.end_date,
            end_predict=req.end_predict,
            selected_models=req.selected_models,
            periodo_maximo=req.periodo_maximo,
            peso_recente=req.peso_recente,
            peso_desvio_max=req.peso_desvio_max,
        )
        STATE.last_autots_result = res
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _extract_growth_series(source: str, model_or_col: str) -> Tuple[pd.DataFrame, str]:
    if source == "arima":
        if not STATE.last_arima_result:
            raise ValueError("Execute a análise ARIMA primeiro.")
        comb = pd.DataFrame(STATE.last_arima_result["combined_series"])
        col_key = f"Pred_{model_or_col}" if not model_or_col.startswith("Pred_") else model_or_col
        if col_key not in comb.columns:
            avail = [c for c in comb.columns if c.startswith("Pred_")]
            if not avail:
                raise ValueError("Nenhum modelo ARIMA disponível.")
            col_key = avail[0]
        return comb[["Data", col_key]].rename(columns={col_key: "Valor"}), model_or_col

    elif source == "autots":
        if not STATE.last_autots_result:
            raise ValueError("Execute a análise AutoTS primeiro.")
        tl = pd.DataFrame(STATE.last_autots_result["timeline"])
        if model_or_col not in tl.columns:
            avail = STATE.last_autots_result["models_available"]
            model_or_col = avail[0]
        return tl[["Data", model_or_col]].rename(columns={model_or_col: "Valor"}), model_or_col

    else:  # dataset
        if STATE.df is None or model_or_col not in STATE.df.columns:
            raise ValueError(f"Coluna '{model_or_col}' não encontrada no dataset.")
        return STATE.df[["Data", model_or_col]].rename(columns={model_or_col: "Valor"}), model_or_col


@app.post("/api/growth/analyze")
def api_analyze_growth(req: GrowthAnalysisRequest):
    try:
        s_df, label = _extract_growth_series(req.source, req.model_or_col)
        views = generate_multi_resolution_views(s_df, "Data", "Valor")
        return {
            "source": req.source,
            "model_or_col": label,
            "views": views,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/forecast/export-arima")
def api_export_arima(model: str = Query("Todos")):
    if not STATE.last_arima_result:
        raise HTTPException(status_code=400, detail="Nenhum resultado ARIMA disponível para exportar.")
    res = STATE.last_arima_result
    comb_df = pd.DataFrame(res["combined_series"])
    if model != "Todos" and f"Pred_{model}" in comb_df.columns:
        cols = ["Data", "Target", "Period", f"Pred_{model}", f"Lo95_{model}", f"Hi95_{model}"]
        comb_df = comb_df[[c for c in cols if c in comb_df.columns]]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        comb_df.to_excel(writer, index=False, sheet_name="Projecoes_ARIMA")
        pd.DataFrame(res["errors"]).to_excel(writer, index=False, sheet_name="Metricas_Erro")
        pd.DataFrame(res["recommendation"]["ranking"]).to_excel(writer, index=False, sheet_name="Recomendacao")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=ARIMA_projections_{model}_{datetime.now().strftime('%Y%m%d')}.xlsx"}
    )


@app.get("/api/forecast/export-autots")
def api_export_autots(model: str = Query("Todos")):
    if not STATE.last_autots_result:
        raise HTTPException(status_code=400, detail="Nenhum resultado AutoTS disponível para exportar.")
    res = STATE.last_autots_result
    tl_df = pd.DataFrame(res["timeline"])
    var_name = res["variable"]

    if model != "Todos" and model in tl_df.columns:
        # Match `select_predict_autoTS_univ` from the R app
        export_df = pd.DataFrame({
            "Data": tl_df["Data"],
            "Period": tl_df["Period"],
            f"{var_name}_ts": tl_df[model].round(2),
            f"{var_name}_lower": tl_df.get(f"{model}_lower"),
            f"{var_name}_upper": tl_df.get(f"{model}_upper"),
        })
    else:
        export_df = tl_df

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Projecoes_AutoTS")
        pd.DataFrame(res["errors"]).to_excel(writer, index=False, sheet_name="Metricas_Erro")
        pd.DataFrame(res["recommendation"]["ranking"]).to_excel(writer, index=False, sheet_name="Ranking_Desvios")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=AutoTS_forecast_{model}_{datetime.now().strftime('%Y%m%d')}.xlsx"}
    )


@app.get("/api/growth/export")
def api_export_growth(source: str = Query("dataset"), model_or_col: str = Query(...)):
    try:
        s_df, label = _extract_growth_series(source, model_or_col)
        views = generate_multi_resolution_views(s_df, "Data", "Valor")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(views["mensal"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Mensal_Valores")
            pd.DataFrame(views["mensal"]["wide_taxas"]).to_excel(writer, index=False, sheet_name="Mensal_Taxas_%")
            pd.DataFrame(views["mensal"]["wide_variacoes"]).to_excel(writer, index=False, sheet_name="Mensal_Variacoes")
            pd.DataFrame(views["trimestral"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Trimestral_Valores")
            pd.DataFrame(views["trimestral"]["wide_taxas"]).to_excel(writer, index=False, sheet_name="Trimestral_Taxas_%")
            pd.DataFrame(views["semestral"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Semestral_Valores")
            pd.DataFrame(views["semestral"]["wide_taxas"]).to_excel(writer, index=False, sheet_name="Semestral_Taxas_%")
            pd.DataFrame(views["anual"]["wide_valores"]).to_excel(writer, index=False, sheet_name="Anual_Valores")
            pd.DataFrame(views["anual"]["wide_taxas"]).to_excel(writer, index=False, sheet_name="Anual_Taxas_%")
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=analise_crescimento_{label}_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==============================================================================
# STATIC FRONTEND SERVING (Pure HTML/CSS/JS Frontend)
# ==============================================================================

FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_spa_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return JSONResponse({"message": "TimeSeries Studio Pro Backend Running"})
