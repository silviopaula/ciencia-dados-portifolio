"""
FastAPI Backend Application for App Forecasts.
Exposes REST APIs for Data Import & Preparation, ML (PyCaret/Caret Recipes),
ARIMA/SARIMAX, and AutoTS + Growth Analysis, and serves the Modern Web Frontend.
"""
from __future__ import annotations
import io
import os
import pickle
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.data_import import (
    converter_coluna_para_date_ok,
    detectar_tipo_arquivo,
    extrair_sazonalidade,
    find_candidate_date_columns,
    gerar_dummies,
    importar_dados,
    inspect_file_metadata,
    inspecionar_coluna_data,
    serialize_dataframe_rows,
    transformar_dados,
)
from backend.forecast_arima import (
    gerar_dashboard_recomendacao_arima,
    run_arima_models_extended,
)
from backend.forecast_autots import (
    generate_df_visoes_ATS,
    gerar_dashboard_autots,
    run_autots_pipeline,
)
from backend.forecast_caret import (
    HAS_PYCARET,
    LISTA_MODELOS_PADRAO,
    gerar_dashboard_recomendacao_adaptado,
    parse_custom_models_txt,
    treinar_avaliar_modelos_caret_recipes,
)
from backend.state import STATE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WWW_DIR = os.path.join(BASE_DIR, "www")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="App Forecasts API",
    description="Time-Series Analysis, Data Engineering, PyCaret/Caret ML, ARIMA, and AutoTS Forecasting Engine",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(WWW_DIR):
    app.mount("/www", StaticFiles(directory=WWW_DIR), name="www")


# ---------------------------------------------------------------------------
# Pydantic Request Schemas
# ---------------------------------------------------------------------------

class DateConvertRequest(BaseModel):
    coluna: str
    formato: str = "auto"


class DummiesRequest(BaseModel):
    anos: bool = True
    meses: bool = True
    tendencia_linear: bool = False
    funcoes_trig: bool = False
    estacoes: bool = False
    bro_bro: bool = False
    crises_selecionadas: List[str] = []
    ativar_dummy_personalizada: bool = False
    dummy_personalizada_nome: Optional[str] = None
    dummy_personalizada_inicio: Optional[str] = None
    dummy_personalizada_fim: Optional[str] = None


class TransformRequest(BaseModel):
    variaveis: List[str]
    transformacao: str
    parametro_boxcox: Optional[float] = None


class SeasonalityRequest(BaseModel):
    variaveis: List[str]
    data_inicio: str
    data_fim: str


class CaretTrainRequest(BaseModel):
    date_col: str
    var_Y: str
    explicativas: List[str] = []
    modelos: List[str]
    data_inicio: str
    data_teste: str
    n_test: int = 12
    metodo_impute: Optional[str] = None
    transformacoes_numericas: List[str] = []
    opcoes_transform: Dict[str, Any] = {}
    engenharia_features: List[str] = []
    opcoes_features: Dict[str, Any] = {}
    reducao_dimensao: Optional[str] = None
    opcoes_reducao: Dict[str, Any] = {}
    auto_tune: bool = False
    tune_length: int = 3
    validation_method: str = "timeslice"
    cv_folds: int = 5
    cv_repeats: Optional[int] = None
    periodo_analise: int = 12


class CaretBacktestUpdateRequest(BaseModel):
    periodo_analise: int = 12


class ArimaRunRequest(BaseModel):
    date_col: str
    target_var: str
    exog_vars: List[str] = []
    data_inicial: str
    data_atual: str
    end_predict: str
    n_teste: int = 3
    seasonal: bool = True
    period: int = 12
    num_models: int = 5
    manual_mode: bool = False
    arima_params: Dict[str, int] = {"p": 1, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1}
    peso_recente: float = 2.0
    peso_desvio_max: float = 0.4


class ArimaDashboardUpdateRequest(BaseModel):
    peso_recente: float = 2.0
    peso_desvio_max: float = 0.4
    periodo_maximo: int = 12


class AutoTSRunRequest(BaseModel):
    date_col: str
    variable: str
    start_date: str
    end_date: str
    end_predict: str
    selected_models: List[str] = ["prophet", "ets", "sarima"]
    periodo_maximo: int = 12
    peso_recente: float = 2.0
    peso_desvio_max: float = 0.4


class AutoTSDesviosUpdateRequest(BaseModel):
    periodo_maximo: int = 12
    peso_recente: float = 2.0
    peso_desvio_max: float = 0.4


class AutoTSGrowthRequest(BaseModel):
    model_name: str
    variable: str


# ---------------------------------------------------------------------------
# Helper to build full data state summary
# ---------------------------------------------------------------------------

def _build_state_summary() -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None:
        return {
            "has_data": False,
            "filename": None,
            "detected_type": None,
            "date_processing_complete": False,
            "official_date_col": None,
            "total_rows": 0,
            "total_cols": 0,
            "columns": [],
            "numeric_columns": [],
            "date_columns": [],
            "date_min": None,
            "date_max": None,
            "column_date_bounds": {},
            "is_monthly": False,
            "logs": list(reversed(STATE.operation_logs)),
            "seasonality_report": STATE.seasonality_report,
        }

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and c != "date_ok"
    ]
    date_cols = find_candidate_date_columns(df)

    date_min = None
    date_max = None
    is_monthly = False
    column_date_bounds: Dict[str, Dict[str, Optional[str]]] = {}
    ref_col = "date_ok" if "date_ok" in df.columns else (date_cols[0] if date_cols else None)
    if ref_col and ref_col in df.columns:
        dt_full = pd.to_datetime(df[ref_col], errors="coerce")
        dt_s = dt_full.dropna().sort_values()
        if len(dt_s) > 0:
            date_min = dt_s.min().strftime("%Y-%m-%d")
            date_max = dt_s.max().strftime("%Y-%m-%d")
            if len(dt_s) > 2:
                diffs = dt_s.diff().dropna().dt.days
                med_diff = float(diffs.median()) if len(diffs) > 0 else 0
                is_monthly = (27 <= med_diff <= 32) or bool((dt_s.dt.day == 1).all())

            for c in numeric_cols:
                valid_dt = dt_full[df[c].notna() & dt_full.notna()]
                if len(valid_dt) > 0:
                    column_date_bounds[c] = {
                        "min": valid_dt.min().strftime("%Y-%m-%d"),
                        "max": valid_dt.max().strftime("%Y-%m-%d"),
                    }

    return {
        "has_data": True,
        "filename": STATE.source_filename,
        "detected_type": STATE.detected_type,
        "date_processing_complete": STATE.date_processing_complete,
        "official_date_col": STATE.official_date_col,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "date_min": date_min,
        "date_max": date_max,
        "column_date_bounds": column_date_bounds,
        "is_monthly": is_monthly,
        "logs": list(reversed(STATE.operation_logs)),
        "seasonality_report": STATE.seasonality_report,
    }


# ---------------------------------------------------------------------------
# Health, General & Data Import Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    df = STATE.get_active_df()
    return {
        "status": "ok",
        "version": app.version,
        "has_data": df is not None,
        "filename": STATE.source_filename,
        "rows": int(len(df)) if df is not None else 0,
    }


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    return _build_state_summary()


@app.post("/api/reset")
def reset_state() -> Dict[str, Any]:
    STATE.reset_import_state()
    return _build_state_summary()


@app.get("/api/samples")
def list_sample_files() -> Dict[str, Any]:
    samples = []
    if os.path.isdir(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.isfile(fpath):
                ftype = detectar_tipo_arquivo(fname)
                if ftype != "desconhecido":
                    samples.append({
                        "filename": fname,
                        "type": ftype,
                        "size_kb": round(os.path.getsize(fpath) / 1024.0, 1),
                    })
    return {"samples": samples}


@app.post("/api/import/inspect")
async def inspect_uploaded_file(
    file: UploadFile = File(...),
    tipo_arquivo: str = Form("auto"),
) -> Dict[str, Any]:
    suffix = os.path.splitext(file.filename or "data.tmp")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        meta = inspect_file_metadata(tmp_path, file.filename or "arquivo", force_type=tipo_arquivo)
        return meta
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/api/import/inspect-sample")
def inspect_sample_file(filename: str = Query(...), tipo_arquivo: str = Query("auto")) -> Dict[str, Any]:
    fpath = os.path.join(DATA_DIR, os.path.basename(filename))
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"Arquivo de exemplo '{filename}' não encontrado.")
    return inspect_file_metadata(fpath, filename, force_type=tipo_arquivo)


def _finalize_imported_dataframe(df: pd.DataFrame, filename: str, detected_type: str) -> Dict[str, Any]:
    STATE.reset_import_state()
    STATE.raw_df = df.copy()
    STATE.processed_df = df.copy()
    STATE.source_filename = filename
    STATE.detected_type = detected_type
    STATE.log(f"Arquivo '{filename}' ({detected_type.upper()}) importado com {len(df)} linhas e {len(df.columns)} colunas.")

    candidates = find_candidate_date_columns(df)
    if candidates:
        best_col = candidates[0]
        STATE.official_date_col = best_col
        # Check if already date or cleanly 100% convertible
        insp = inspecionar_coluna_data(df, best_col)
        non_na_count = insp.get("total_registros", 0) - insp.get("registros_na", 0)
        if insp.get("e_data") or (
            insp.get("pode_converter") and non_na_count > 0 and insp.get("n_convertido", 0) == non_na_count
        ):
            df_conv, _info = converter_coluna_para_date_ok(df, best_col)
            STATE.processed_df = df_conv
            STATE.date_processing_complete = True
            STATE.log(f"Coluna de data '{best_col}' reconhecida automaticamente. 'date_ok' criada.")
    return {
        "state": _build_state_summary(),
        "preview": serialize_dataframe_rows(STATE.get_active_df(), limit=500),
    }


@app.post("/api/import/upload")
async def upload_and_import(
    file: UploadFile = File(...),
    tipo_arquivo: str = Form("auto"),
    planilha: Optional[str] = Form(None),
    tabela: Optional[str] = Form(None),
) -> Dict[str, Any]:
    fname = file.filename or "dados.csv"
    suffix = os.path.splitext(fname)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        detected = detectar_tipo_arquivo(fname) if tipo_arquivo == "auto" else tipo_arquivo
        df = importar_dados(
            file_path=tmp_path,
            filename=fname,
            tipo_arquivo=detected,
            planilha=planilha,
            tabela=tabela,
        )
        if len(df) == 0:
            raise ValueError("Arquivo importado está vazio.")
        return _finalize_imported_dataframe(df, fname, detected)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na importação: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/import/sample")
def import_sample_dataset(
    filename: str = Form(...),
    tipo_arquivo: str = Form("auto"),
    planilha: Optional[str] = Form(None),
    tabela: Optional[str] = Form(None),
) -> Dict[str, Any]:
    fpath = os.path.join(DATA_DIR, os.path.basename(filename))
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"Arquivo '{filename}' não encontrado na pasta data/.")
    try:
        detected = detectar_tipo_arquivo(filename) if tipo_arquivo == "auto" else tipo_arquivo
        df = importar_dados(
            file_path=fpath,
            filename=filename,
            tipo_arquivo=detected,
            planilha=planilha,
            tabela=tabela,
        )
        return _finalize_imported_dataframe(df, filename, detected)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao importar exemplo: {str(e)}")


@app.get("/api/date/inspect")
def api_inspect_date_col(coluna: str = Query(...)) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None:
        raise HTTPException(status_code=400, detail="Nenhum dado importado.")
    STATE.official_date_col = coluna
    res = inspecionar_coluna_data(df, coluna)
    if res.get("e_data"):
        df_conv, _ = converter_coluna_para_date_ok(df, coluna)
        STATE.processed_df = df_conv
        STATE.date_processing_complete = True
    return {
        "inspection": res,
        "state": _build_state_summary(),
    }


@app.post("/api/date/convert")
def api_convert_date_col(req: DateConvertRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None:
        raise HTTPException(status_code=400, detail="Importe dados primeiro.")
    try:
        df_conv, info = converter_coluna_para_date_ok(df, req.coluna, formato=req.formato)
        STATE.processed_df = df_conv
        STATE.official_date_col = req.coluna
        STATE.date_processing_complete = True
        msg = f"Coluna '{req.coluna}' convertida para 'date_ok' ({info['total_validas']} datas válidas)."
        if info["novos_nas"] > 0:
            msg += f" Atenção: {info['novos_nas']} valor(es) viraram NA."
        STATE.log(msg)
        return {
            "info": info,
            "message": msg,
            "state": _build_state_summary(),
            "preview": serialize_dataframe_rows(STATE.processed_df, limit=500),
        }
    except Exception as e:
        STATE.date_processing_complete = False
        STATE.log(f"ERRO ao converter '{req.coluna}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dummies/generate")
def api_generate_dummies(req: DummiesRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None or not STATE.date_processing_complete:
        raise HTTPException(status_code=400, detail="Converta a coluna de data para 'date_ok' primeiro.")

    periodo_pers = None
    if req.ativar_dummy_personalizada:
        if not req.dummy_personalizada_nome or not req.dummy_personalizada_nome.strip():
            raise HTTPException(status_code=400, detail="Dummy personalizada: Nome é obrigatório.")
        if not req.dummy_personalizada_inicio or not req.dummy_personalizada_fim:
            raise HTTPException(status_code=400, detail="Dummy personalizada: Datas de início e fim são obrigatórias.")
        periodo_pers = {
            "nome": req.dummy_personalizada_nome.strip(),
            "inicio": req.dummy_personalizada_inicio,
            "fim": req.dummy_personalizada_fim,
        }

    try:
        col_data = "date_ok" if "date_ok" in df.columns else (STATE.official_date_col or "Data")
        df_new, added_cols = gerar_dummies(
            df=df,
            col_data=col_data,
            anos=req.anos,
            meses=req.meses,
            tendencia_linear=req.tendencia_linear,
            funcoes_trig=req.funcoes_trig,
            estacoes=req.estacoes,
            bro_bro=req.bro_bro,
            crises_selecionadas=req.crises_selecionadas,
            periodo_personalizado=periodo_pers,
        )
        STATE.processed_df = df_new
        STATE.log(f"Variáveis/Dummies geradas ({len(added_cols)} colunas): {', '.join(added_cols[:12])}{'...' if len(added_cols) > 12 else ''}")
        return {
            "added_columns": added_cols,
            "message": f"{len(added_cols)} variáveis/dummies geradas com sucesso!",
            "state": _build_state_summary(),
            "preview": serialize_dataframe_rows(STATE.processed_df, limit=500),
        }
    except Exception as e:
        STATE.log(f"ERRO ao gerar dummies: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/transform/apply")
def api_transform_variables(req: TransformRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None or not STATE.date_processing_complete:
        raise HTTPException(status_code=400, detail="Importe dados e valide a data primeiro.")
    try:
        df_new, mensagens = transformar_dados(
            df=df,
            variaveis=req.variaveis,
            transformacao=req.transformacao,
            parametro_boxcox=req.parametro_boxcox,
        )
        STATE.processed_df = df_new
        STATE.log(f"Transformação '{req.transformacao}' aplicada a: {', '.join(req.variaveis)}")
        for var, m in mensagens.items():
            STATE.log(f"  * '{var}': {m}")
        return {
            "messages": mensagens,
            "state": _build_state_summary(),
            "preview": serialize_dataframe_rows(STATE.processed_df, limit=500),
        }
    except Exception as e:
        STATE.log(f"ERRO Transformação '{req.transformacao}': {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/seasonality/extract")
def api_extract_seasonality(req: SeasonalityRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None or not STATE.date_processing_complete:
        raise HTTPException(status_code=400, detail="Importe dados e converta a coluna de data primeiro.")
    try:
        col_data = "date_ok" if "date_ok" in df.columns else (STATE.official_date_col or "Data")
        df_new, report, added_cols = extrair_sazonalidade(
            df=df,
            colunas=req.variaveis,
            data_inicio=req.data_inicio,
            data_fim=req.data_fim,
            col_data=col_data,
        )
        STATE.processed_df = df_new
        STATE.seasonality_report = report
        STATE.log(f"Sazonalidade e tendência extraídas ({', '.join(added_cols)})")
        return {
            "added_columns": added_cols,
            "report": report,
            "state": _build_state_summary(),
            "preview": serialize_dataframe_rows(STATE.processed_df, limit=500),
        }
    except Exception as e:
        STATE.log(f"ERRO Sazonalidade: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/data/table")
def api_get_data_table(
    columns: Optional[str] = Query(None),
    limit: int = Query(5000),
) -> Dict[str, Any]:
    df = STATE.get_active_df()
    col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    return serialize_dataframe_rows(df, columns=col_list, limit=limit)


@app.get("/api/data/export")
def api_export_processed_data(fmt: str = Query("xlsx")) -> StreamingResponse:
    df = STATE.get_active_df()
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="Não há dados processados para exportar.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt_clean = fmt.lower().strip()

    if fmt_clean == "xlsx":
        buf = io.BytesIO()
        df_exp = df.copy()
        for c in df_exp.columns:
            if pd.api.types.is_datetime64_any_dtype(df_exp[c]):
                df_exp[c] = pd.to_datetime(df_exp[c]).dt.strftime("%Y-%m-%d")
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_exp.to_excel(writer, index=False, sheet_name="Dados_Processados")
        buf.seek(0)
        STATE.log("Dados exportados: XLSX")
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="dados_processados_{ts}.xlsx"'},
        )

    elif fmt_clean == "csv":
        buf = io.BytesIO()
        csv_text = df.to_csv(index=False)
        buf.write(csv_text.encode("utf-8-sig"))
        buf.seek(0)
        STATE.log("Dados exportados: CSV")
        return StreamingResponse(
            buf,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="dados_processados_{ts}.csv"'},
        )

    elif fmt_clean in ("rds", "pkl", "joblib"):
        buf = io.BytesIO()
        pickle.dump(df, buf)
        buf.seek(0)
        STATE.log("Dados exportados: Pickle/RDS-Python (.pkl)")
        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="dados_processados_{ts}.pkl"'},
        )

    raise HTTPException(status_code=400, detail=f"Formato de exportação inválido: {fmt}")


# ---------------------------------------------------------------------------
# Tab 3: Projeções ML (PyCaret / Caret Recipes) Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/caret/models")
def api_get_caret_models() -> Dict[str, Any]:
    active_catalog = STATE.custom_models_config if STATE.custom_models_config else LISTA_MODELOS_PADRAO
    items = []
    for k, v in active_catalog.items():
        items.append({
            "id": k,
            "method": v["method"],
            "preProcess": v.get("preProcess", []),
            "label": v.get("label", k),
            "group": v.get("group", "Padrão"),
        })
    return {
        "has_pycaret": HAS_PYCARET,
        "is_custom": STATE.custom_models_config is not None,
        "models": items,
        "default_selected": [m["id"] for m in items[:4]],
    }


@app.post("/api/caret/custom-models")
async def api_upload_custom_models(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = (await file.read()).decode("utf-8", errors="replace")
    parsed = parse_custom_models_txt(content)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="Nenhum modelo válido pôde ser identificado no arquivo .txt. Use formato R lista_modelos_custom ou nomes de modelos separados por linha.",
        )
    STATE.custom_models_config = parsed
    STATE.log(f"Lista de modelos customizados carregada ({len(parsed)} modelos).")
    return api_get_caret_models()


@app.post("/api/caret/reset-models")
def api_reset_custom_models() -> Dict[str, Any]:
    STATE.custom_models_config = None
    return api_get_caret_models()


@app.post("/api/caret/train")
def api_train_caret_models(req: CaretTrainRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None:
        raise HTTPException(status_code=400, detail="Importe dados primeiro na aba de Importação.")
    try:
        res = treinar_avaliar_modelos_caret_recipes(
            df=df,
            date_col=req.date_col,
            var_Y=req.var_Y,
            explicativas=req.explicativas,
            modelos_selecionados=req.modelos,
            data_inicio=req.data_inicio,
            data_teste=req.data_teste,
            n_test=req.n_test,
            metodo_impute=req.metodo_impute,
            transformacoes_numericas=req.transformacoes_numericas,
            opcoes_transform=req.opcoes_transform,
            engenharia_features=req.engenharia_features,
            opcoes_features=req.opcoes_features,
            reducao_dimensao=req.reducao_dimensao,
            opcoes_reducao=req.opcoes_reducao,
            auto_tune=req.auto_tune,
            tune_length=req.tune_length,
            validation_method=req.validation_method,
            cv_folds=req.cv_folds,
            cv_repeats=req.cv_repeats,
            periodo_analise=req.periodo_analise,
            custom_models=STATE.custom_models_config,
        )
        STATE.caret_results = res
        STATE.log(f"Treinamento ML concluído ({len(res['modelos_treinados_nomes'])} modelos para '{req.var_Y}').")

        return {
            "var_Y": res["var_Y"],
            "data_inicio": res["data_inicio"],
            "data_teste": res["data_teste"],
            "modelos_treinados": res["modelos_treinados_nomes"],
            "chart_series": res["chart_series"],
            "cv_metrics": res["cv_metrics"],
            "recomendacao_backtest": res["recomendacao_backtest"],
            "importancia_variaveis": res["importancia_variaveis"],
            "elasticidades": res["elasticidades"],
            "config": res["config"],
            "projecoes_rows": res["df_projecoes"].replace([np.inf, -np.inf], np.nan).replace({np.nan: None}).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/caret/backtest")
def api_update_caret_backtest(req: CaretBacktestUpdateRequest) -> Dict[str, Any]:
    if not STATE.caret_results:
        raise HTTPException(status_code=400, detail="Execute o treinamento de modelos primeiro.")
    res = STATE.caret_results
    recomendacao = gerar_dashboard_recomendacao_adaptado(
        df_projecoes=res["df_projecoes"],
        df_reais=res["df_reais"],
        var_Y=res["var_Y"],
        periodo_maximo=req.periodo_analise,
    )
    res["recomendacao_backtest"] = recomendacao
    return {"recomendacao_backtest": recomendacao}


@app.get("/api/caret/export-projections")
def api_export_caret_projections() -> StreamingResponse:
    if not STATE.caret_results:
        raise HTTPException(status_code=400, detail="Treine os modelos primeiro antes de exportar.")
    res = STATE.caret_results
    buf = io.BytesIO()
    ts = datetime.now().strftime("%Y%m%d")

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        res["df_projecoes"].to_excel(writer, index=False, sheet_name="Projecoes")
        pd.DataFrame(res["recomendacao_backtest"]).to_excel(writer, index=False, sheet_name="Backtest_Aderencia")
        pd.DataFrame(res["cv_metrics"]).to_excel(writer, index=False, sheet_name="Metricas_Internas")
        df_cfg = pd.DataFrame([{"Parametro": k, "Valor": str(v)} for k, v in res["config"].items()])
        df_cfg.to_excel(writer, index=False, sheet_name="Configuracoes")
        res["df_treino"].to_excel(writer, index=False, sheet_name="Dados_Treino_Usados")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="projecoes_resultados_ml_{ts}.xlsx"'},
    )


@app.get("/api/caret/export-importance")
def api_export_caret_importance(modelo: str = Query(...)) -> StreamingResponse:
    if not STATE.caret_results:
        raise HTTPException(status_code=400, detail="Treine os modelos primeiro.")
    imp_map = STATE.caret_results.get("importancia_variaveis", {})
    if modelo not in imp_map:
        raise HTTPException(status_code=404, detail=f"Importância não disponível para o modelo '{modelo}'.")
    df_imp = pd.DataFrame(imp_map[modelo]["dados"])
    buf = io.BytesIO()
    buf.write(df_imp.to_csv(index=False).encode("utf-8-sig"))
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d")
    return StreamingResponse(
        buf,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="importancia_cascata_{modelo}_{ts}.csv"'},
    )


@app.get("/api/caret/export-models")
def api_export_caret_models() -> StreamingResponse:
    if not STATE.caret_results or not STATE.caret_results.get("modelos_objetos"):
        raise HTTPException(status_code=400, detail="Nenhum modelo treinado disponível para exportação.")
    buf = io.BytesIO()
    joblib.dump(STATE.caret_results["modelos_objetos"], buf)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d")
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="modelos_forecast_{ts}.joblib"'},
    )


# ---------------------------------------------------------------------------
# Tab 4: Projeções ARIMA / SARIMAX Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/arima/run")
def api_run_arima(req: ArimaRunRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None:
        raise HTTPException(status_code=400, detail="Importe dados primeiro na aba de Importação.")
    try:
        res = run_arima_models_extended(
            df=df,
            date_col=req.date_col,
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
        STATE.arima_results = res
        STATE.log(f"Análise ARIMA concluída ({len(res['errors'])} modelos para '{req.target_var}').")
        return {
            "target_var": res["target_var"],
            "exog_vars": res["exog_vars"],
            "collinear_warning": res["collinear_warning"],
            "execution_time_sec": res["execution_time_sec"],
            "periods": res["periods"],
            "real_series": res["real_series"],
            "forecasts": res["forecasts"],
            "errors": res["errors"],
            "deviations": res["deviations"],
            "dashboard": res["dashboard"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/arima/dashboard")
def api_update_arima_dashboard(req: ArimaDashboardUpdateRequest) -> Dict[str, Any]:
    if not STATE.arima_results:
        raise HTTPException(status_code=400, detail="Execute a modelagem ARIMA primeiro.")
    res = STATE.arima_results
    dash = gerar_dashboard_recomendacao_arima(
        per_month_pct_deviations=res["per_month_pct_deviations"],
        errors_list=res["errors"],
        peso_recente=req.peso_recente,
        peso_desvio_max=req.peso_desvio_max,
        periodo_maximo=req.periodo_maximo,
    )
    res["dashboard"] = dash
    return {"dashboard": dash}


@app.get("/api/arima/export")
def api_export_arima_xlsx(model: str = Query("Todos")) -> StreamingResponse:
    if not STATE.arima_results:
        raise HTTPException(status_code=400, detail="Execute a análise ARIMA antes de exportar.")
    res = STATE.arima_results
    df_exp: pd.DataFrame = res["df_export"].copy()

    if model and model != "Todos":
        keep_cols = ["Data", "Realizado", "Tipo"]
        for c in [f"Projetado_{model}", f"Lo95_{model}", f"Hi95_{model}"]:
            if c in df_exp.columns:
                keep_cols.append(c)
        df_exp = df_exp[keep_cols]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_exp.to_excel(writer, index=False, sheet_name="Projecoes_ARIMA")
        pd.DataFrame(res["errors"]).to_excel(writer, index=False, sheet_name="Metricas_Erro")
        pd.DataFrame(res["dashboard"]).to_excel(writer, index=False, sheet_name="Recomendacao")
        if res["deviations"]:
            pd.DataFrame(res["deviations"]).to_excel(writer, index=False, sheet_name="Desvios_Teste")

    buf.seek(0)
    ts = datetime.now().strftime("%Y-%m-%d")
    safe_target = res["target_var"].replace(" ", "_")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="ARIMA_Projecao_{safe_target}_{ts}.xlsx"'},
    )


# ---------------------------------------------------------------------------
# Tab 5: Projeções AutoTS & Growth Analysis Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/autots/run")
def api_run_autots(req: AutoTSRunRequest) -> Dict[str, Any]:
    df = STATE.get_active_df()
    if df is None:
        raise HTTPException(status_code=400, detail="Importe dados primeiro na aba de Importação.")
    try:
        res = run_autots_pipeline(
            df=df,
            date_col=req.date_col,
            variable=req.variable,
            start_date=req.start_date,
            end_date=req.end_date,
            end_predict=req.end_predict,
            selected_models=req.selected_models,
            periodo_maximo=req.periodo_maximo,
            peso_recente=req.peso_recente,
            peso_desvio_max=req.peso_desvio_max,
        )
        STATE.autots_results = res
        STATE.log(f"Análise AutoTS concluída ({len(res['models'])} modelos para '{req.variable}').")

        # Also pre-calculate growth views for the best model automatically
        if res["best_model"]:
            STATE.autots_growth_results = generate_df_visoes_ATS(
                df_export=res["df_export"],
                model_name=res["best_model"],
                variable=req.variable,
            )

        return {
            "variable": res["variable"],
            "models": res["models"],
            "best_model": res["best_model"],
            "errors": res["errors"],
            "dashboard_desvios": res["dashboard_desvios"],
            "real_series": res["real_series"],
            "chart_models": res["chart_models"],
            "growth": STATE.autots_growth_results,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/autots/desvios")
def api_update_autots_desvios(req: AutoTSDesviosUpdateRequest) -> Dict[str, Any]:
    if not STATE.autots_results:
        raise HTTPException(status_code=400, detail="Execute a análise AutoTS primeiro.")
    res = STATE.autots_results
    dash = gerar_dashboard_autots(
        df_predict_ts=res["df_predict_ts"],
        modelos_usados=res["models"],
        periodo_maximo=req.periodo_maximo,
        peso_recente=req.peso_recente,
        peso_desvio_max=req.peso_desvio_max,
    )
    res["dashboard_desvios"] = dash
    return {"dashboard_desvios": dash}


@app.post("/api/autots/growth")
def api_run_autots_growth(req: AutoTSGrowthRequest) -> Dict[str, Any]:
    if not STATE.autots_results:
        raise HTTPException(status_code=400, detail="Execute a análise AutoTS primeiro.")
    res = STATE.autots_results
    visoes = generate_df_visoes_ATS(
        df_export=res["df_export"],
        model_name=req.model_name,
        variable=req.variable,
    )
    STATE.autots_growth_results = visoes
    return visoes


@app.get("/api/autots/export")
def api_export_autots_xlsx(model: Optional[str] = Query(None)) -> StreamingResponse:
    if not STATE.autots_results:
        raise HTTPException(status_code=400, detail="Execute a análise AutoTS antes de exportar.")
    res = STATE.autots_results
    var_name = res["variable"]
    df_exp: pd.DataFrame = res["df_export"].copy()
    ts = datetime.now().strftime("%Y-%m-%d")

    buf = io.BytesIO()
    if model and model in df_exp.columns:
        # Single selected model export matching `select_predict_autoTS_univ`
        df_single = pd.DataFrame({
            "Data": df_exp["Data"],
            f"{var_name}_ts": df_exp[model].round(2),
            f"{model}_lower": df_exp.get(f"{model}_lower"),
            f"{model}_upper": df_exp.get(f"{model}_upper"),
        })
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_single.to_excel(writer, index=False, sheet_name=f"AutoTS_{model}")
        fname = f"forecast-autots-{model}-{var_name}-{ts}.xlsx"
    else:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_exp.to_excel(writer, index=False, sheet_name="Projecoes_AutoTS")
            pd.DataFrame(res["errors"]).to_excel(writer, index=False, sheet_name="Erros_Treino")
            pd.DataFrame(res["dashboard_desvios"]).to_excel(writer, index=False, sheet_name="Analise_Desvios")
        fname = f"forecast-autots-completo-{var_name}-{ts}.xlsx"

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/autots/export-growth")
def api_export_autots_growth_xlsx() -> StreamingResponse:
    if not STATE.autots_growth_results:
        raise HTTPException(status_code=400, detail="Execute a análise de crescimento primeiro.")
    g = STATE.autots_growth_results
    buf = io.BytesIO()
    ts = datetime.now().strftime("%Y-%m-%d")

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for gran_key, sheet_prefix in [
            ("mensal", "Mensal"),
            ("trimestral", "Trim"),
            ("semestral", "Sem"),
            ("anual", "Anual"),
        ]:
            g_data = g[gran_key]
            pd.DataFrame(g_data["valores"]).to_excel(writer, index=False, sheet_name=f"Val_{sheet_prefix}")
            pd.DataFrame(g_data["taxas"]).to_excel(writer, index=False, sheet_name=f"Taxas_{sheet_prefix}")
            pd.DataFrame(g_data["variacoes"]).to_excel(writer, index=False, sheet_name=f"Var_{sheet_prefix}")
            pd.DataFrame(g_data["long"]).to_excel(writer, index=False, sheet_name=f"Serie_{sheet_prefix}")

    buf.seek(0)
    fname = f"analise-crescimento-{g['model']}-{g['variable']}-{ts}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------------
# Serve Frontend SPA (Zero Python in Frontend!)
# ---------------------------------------------------------------------------

if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend_assets")


@app.get("/")
def serve_frontend_index() -> FileResponse:
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html não encontrado.")
    return FileResponse(
        index_path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
