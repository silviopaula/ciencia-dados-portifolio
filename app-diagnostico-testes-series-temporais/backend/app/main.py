"""API FastAPI — TimeSeries Diagnostics Studio (Porta 8005)."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .datasets import (
    build_clean_series_df,
    get_demo_catalog,
    infer_columns_and_period,
    load_demo_dataset,
    parse_uploaded_file,
)
from .diagnostics import run_full_diagnostics
from .exporter import export_analysis_to_excel
from .schemas import AnalyzeDemoRequest, UploadInspectResponse

app = FastAPI(
    title="TimeSeries Diagnostics Studio API",
    description=(
        "Laboratório Econométrico Interativo de Testes Estatísticos para Séries Temporais: "
        "ADF, Phillips-Perron, KPSS, Zivot-Andrews, Box-Pierce, Ljung-Box, Engle ARCH-LM, "
        "Breusch-Pagan Temporal, Anderson-Darling, Jarque-Bera, Shapiro-Wilk, STL e SARIMA."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "TimeSeries Diagnostics Studio API",
        "version": "2.0.0",
    }


@app.get("/api/demos")
def list_demos() -> list:
    return get_demo_catalog()


@app.post("/api/analyze")
def analyze_demo(req: AnalyzeDemoRequest) -> dict:
    try:
        df, meta = load_demo_dataset(req.dataset_id)
        period = req.seasonal_period if req.seasonal_period is not None else meta["seasonal_period"]
        return run_full_diagnostics(
            df=df,
            dataset_title=meta["title"],
            dataset_id=meta["id"],
            transform=req.transform,
            alpha=req.alpha,
            regression=req.regression,
            max_lags=req.max_lags,
            seasonal_period=period,
            auto_arima=req.auto_arima,
            manual_order=(req.order_p, req.order_d, req.order_q),
            manual_seasonal=(req.seasonal_P, req.seasonal_D, req.seasonal_Q),
            forecast_horizon=req.forecast_horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao executar diagnóstico: {exc}") from exc


@app.post("/api/upload/inspect", response_model=UploadInspectResponse)
async def inspect_upload(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(default=None),
) -> UploadInspectResponse:
    try:
        raw = await file.read()
        if not raw:
            raise ValueError("Arquivo enviado está vazio.")
        df, info = parse_uploaded_file(raw, file.filename or "dados.csv", sheet_name=sheet_name)
        date_col, val_col, num_cols, period = infer_columns_and_period(df)
        preview = df.head(6).fillna("").astype(str).to_dict(orient="records")
        return UploadInspectResponse(
            filename=info["filename"],
            file_type=info["file_type"],
            detected_separator=info["detected_separator"],
            sheets=info["sheets"],
            selected_sheet=info["selected_sheet"],
            columns=list(df.columns),
            numeric_columns=num_cols,
            suggested_date_col=date_col,
            suggested_value_col=val_col,
            suggested_period=period,
            preview_rows=preview,
            total_rows=int(len(df)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Falha ao ler arquivo: {exc}") from exc


@app.post("/api/upload/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    sheet_name: Optional[str] = Form(default=None),
    date_format: str = Form(default="auto"),
    aggregation: str = Form(default="none"),
    transform: str = Form(default="level"),
    alpha: float = Form(default=0.05),
    regression: str = Form(default="ct"),
    max_lags: int = Form(default=12),
    seasonal_period: int = Form(default=12),
    auto_arima: bool = Form(default=True),
    order_p: int = Form(default=1),
    order_d: int = Form(default=1),
    order_q: int = Form(default=1),
    seasonal_P: int = Form(default=0),
    seasonal_D: int = Form(default=0),
    seasonal_Q: int = Form(default=0),
    forecast_horizon: int = Form(default=12),
) -> dict:
    try:
        raw = await file.read()
        df_raw, info = parse_uploaded_file(raw, file.filename or "upload.csv", sheet_name=sheet_name)
        clean_df = build_clean_series_df(
            df_raw,
            date_col=date_col,
            value_col=value_col,
            date_format=date_format,
            aggregation=aggregation,
        )
        title = f"{info['filename']} ({value_col})"
        return run_full_diagnostics(
            df=clean_df,
            dataset_title=title,
            dataset_id="upload",
            transform=transform,
            alpha=alpha,
            regression=regression,
            max_lags=max_lags,
            seasonal_period=seasonal_period,
            auto_arima=auto_arima,
            manual_order=(order_p, order_d, order_q),
            manual_seasonal=(seasonal_P, seasonal_D, seasonal_Q),
            forecast_horizon=forecast_horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar arquivo enviado: {exc}") from exc


@app.post("/api/export/excel")
def export_demo_excel(req: AnalyzeDemoRequest) -> Response:
    try:
        analysis = analyze_demo(req)
        xlsx_bytes = export_analysis_to_excel(analysis)
        fname = f"diagnostico_ts_{req.dataset_id}_{req.transform}.xlsx"
        return Response(
            content=xlsx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar Excel: {exc}") from exc


@app.post("/api/upload/export/excel")
async def export_upload_excel(
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    sheet_name: Optional[str] = Form(default=None),
    date_format: str = Form(default="auto"),
    aggregation: str = Form(default="none"),
    transform: str = Form(default="level"),
    alpha: float = Form(default=0.05),
    regression: str = Form(default="ct"),
    max_lags: int = Form(default=12),
    seasonal_period: int = Form(default=12),
    auto_arima: bool = Form(default=True),
    order_p: int = Form(default=1),
    order_d: int = Form(default=1),
    order_q: int = Form(default=1),
    seasonal_P: int = Form(default=0),
    seasonal_D: int = Form(default=0),
    seasonal_Q: int = Form(default=0),
    forecast_horizon: int = Form(default=12),
) -> Response:
    analysis = await analyze_upload(
        file=file,
        date_col=date_col,
        value_col=value_col,
        sheet_name=sheet_name,
        date_format=date_format,
        aggregation=aggregation,
        transform=transform,
        alpha=alpha,
        regression=regression,
        max_lags=max_lags,
        seasonal_period=seasonal_period,
        auto_arima=auto_arima,
        order_p=order_p,
        order_d=order_d,
        order_q=order_q,
        seasonal_P=seasonal_P,
        seasonal_D=seasonal_D,
        seasonal_Q=seasonal_Q,
        forecast_horizon=forecast_horizon,
    )
    xlsx_bytes = export_analysis_to_excel(analysis)
    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="diagnostico_ts_upload.xlsx"'},
    )
