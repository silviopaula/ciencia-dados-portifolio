# app.py
import io
import os
import json
import random
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import torch
from chronos import BaseChronosPipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    median_absolute_error, r2_score
)

st.set_page_config(page_title="Forecast com Amazon Chronos", layout="wide")
st.title("⚡ Forecast com Amazon Chronos (Chronos/Bolt)")

# =========================
# Pequenos utilitários
# =========================
def parse_json_or_none(s: str):
    if not s or str(s).strip() == "":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def ensure_datetime(series: pd.Series):
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    try:
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    except Exception:
        return pd.to_datetime(series, errors="coerce")

# =========================
# Função principal (com melhorias)
# =========================
def chronos_forecast_multi_models_single_df(
    df: pd.DataFrame,
    models: List[str],
    target_col: str,
    test_periods: int = 12,
    n_periods: Optional[int] = None,      # horizonte desejado
    device: str = "cpu",
    force_offline: bool = False,
    date_col: str = "Data",
    interval_alpha: float = 0.95,         # usado se quantile_levels=None
    quantile_levels: Optional[List[float]] = None,  # ex.: [0.1, 0.5, 0.9]
    local_base_dir: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    random_seed: Optional[int] = 123,
    allow_long_horizon: bool = True,      # se >64, faz blocos e concatena
    ensemble: Optional[dict] = None,      # {"method":"mean"|"median"|"weighted","weights":{...},"models":[...]}
):
    # ---- Seeds ----
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        try:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # ---- Paleta ----
    default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    if color_map is None:
        color_map = {
            "amazon_chronos-t5-tiny":  "#2ca02c",
            "amazon_chronos-t5-mini":  "#1f77b4",
            "amazon_chronos-t5-small": "#ff7f0e",
            "amazon_chronos-t5-base":  "#9467bd",
            "amazon_chronos-t5-large": "#d62728",
            "amazon_chronos-bolt-tiny":  "#17becf",
            "amazon_chronos-bolt-mini":  "#bcbd22",
            "amazon_chronos-bolt-small": "#8c564b",
            "amazon_chronos-bolt-base":  "#e377c2",
        }
    used_colors = {}
    def get_color_for_label(label: str) -> str:
        if label in color_map:
            return color_map[label]
        for c in default_palette:
            if c not in used_colors.values():
                return c
        return "#7f7f7f"

    # ---- Preparação do DF ----
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada.")
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada.")

    df = df.copy()
    df[date_col] = ensure_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # split
    non_missing_idx = df[df[target_col].notna()].index
    if len(non_missing_idx) < max(1, test_periods):
        raise ValueError(f"Não há {test_periods} pontos não-nulos em '{target_col}' para formar o teste.")
    test_idx = non_missing_idx[-test_periods:]
    train_idx = non_missing_idx[:-test_periods]
    df_treino = df.loc[train_idx, [date_col, target_col]].reset_index(drop=True)
    df_teste  = df.loc[test_idx,  [date_col, target_col]].reset_index(drop=True)

    # horizonte
    if n_periods is None:
        n_periods = test_periods
    n_periods = int(n_periods)

    # índices de previsão
    start_date = df_teste[date_col].iloc[0] if len(df_teste) > 0 else (df_treino[date_col].max() + pd.offsets.MonthBegin(1))
    future_index_full = pd.date_range(start=start_date, periods=n_periods, freq="MS")

    # interseção p/ métricas
    overlap_len = min(n_periods, len(df_teste))
    df_teste_used = df_teste.iloc[:overlap_len].copy()
    shade_start = df_teste_used[date_col].iloc[0] if overlap_len > 0 else None
    shade_end   = df_teste_used[date_col].iloc[-1] if overlap_len > 0 else None

    # device
    use_device = "cuda" if (device.lower() == "cuda" and torch.cuda.is_available()) else "cpu"

    # gráfico base
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_treino[date_col], y=df_treino[target_col],
        mode="lines", name="Treino (real)",
        line=dict(color="black", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_teste[date_col], y=df_teste[target_col],
        mode="lines", name="Teste (real)",
        line=dict(color="#3366cc", width=2)
    ))
    if overlap_len > 0:
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=shade_start, x1=shade_end, y0=0, y1=1,
            fillcolor="rgba(0,0,0,0.07)", line=dict(width=0), layer="below"
        )

    # métricas enxutas
    metrics_cols = [
        "Label","Fonte","Variável","Previsto","Avaliado_sobre",
        "MAE","RMSE","MAPE (%)","MedAE","R2","Lower_q","Upper_q"
    ]
    metrics_rows = []
    forecasts = {}

    # contexto
    y_hist = df_treino[target_col].dropna().values.astype("float32")
    context_base = torch.from_numpy(y_hist).unsqueeze(0)

    # quantis
    def build_quantiles_for_model(model_id: str):
        if quantile_levels:
            q_sorted = sorted(quantile_levels)
        else:
            lo = (1.0 - interval_alpha) / 2.0
            hi = 1.0 - lo
            q_sorted = [lo, 0.5, hi]
        lo, mid, hi = q_sorted[0], 0.5, q_sorted[-1]
        if "bolt" in model_id.lower():
            lo = max(0.1, min(0.9, lo))
            hi = max(0.1, min(0.9, hi))
            if lo >= mid: lo = 0.1
            if hi <= mid: hi = 0.9
            q_sorted = [lo, mid, hi]
        return q_sorted, q_sorted[0], q_sorted[-1]

    # df_forecast consolidado
    df_forecast = df.copy()
    last_hist_date = df_forecast[date_col].max()
    extra_dates = [d for d in future_index_full if d > last_hist_date]
    if extra_dates:
        df_extra = pd.DataFrame({date_col: extra_dates})
        for c in df_forecast.columns:
            if c != date_col:
                df_extra[c] = np.nan
        df_forecast = pd.concat([df_forecast, df_extra], ignore_index=True, sort=False)
    if f"{target_col}_real" not in df_forecast.columns:
        df_forecast[f"{target_col}_real"] = df_forecast[target_col]

    # previsão em blocos
    def predict_in_blocks(pipe, ctx_tensor, pred_len, q_levels):
        max_block = 64
        remaining = pred_len
        y_ctx = ctx_tensor.clone()
        out_list = []
        while remaining > 0:
            step = min(max_block, remaining)
            qs, _ = pipe.predict_quantiles(
                context=y_ctx, prediction_length=step, quantile_levels=q_levels
            )
            np_block = qs[0].detach().cpu().numpy()
            out_list.append(np_block)
            # atualiza contexto com p50
            p50_idx = q_levels.index(0.5)
            p50_vals = qs[0, :, p50_idx].detach().cpu()
            y_ctx = torch.cat([y_ctx, p50_vals.unsqueeze(0)], dim=1)
            remaining -= step
        return np.concatenate(out_list, axis=0)

    # loop de modelos
    for model_str in models:
        is_online = ("/" in model_str)
        label = model_str.split("/")[-1] if is_online else model_str

        # carregar
        try:
            if is_online:
                # garantir modo online
                if "TRANSFORMERS_OFFLINE" in os.environ:
                    os.environ.pop("TRANSFORMERS_OFFLINE")
                pipe = BaseChronosPipeline.from_pretrained(
                    model_str, device_map=None, dtype=torch.float32, local_files_only=False
                )
                fonte = "huggingface"
                ident_for_quant = model_str
            else:
                if not local_base_dir:
                    raise ValueError("Modelo offline informado sem 'local_base_dir'.")
                model_dir = os.path.join(local_base_dir, model_str)
                if not (os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "config.json"))):
                    raise FileNotFoundError(f"Pasta do modelo local não encontrada ou inválida: {model_dir}")
                if force_offline:
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                pipe = BaseChronosPipeline.from_pretrained(
                    model_dir, local_files_only=True, device_map=None, dtype=torch.float32
                )
                fonte = "local"
                ident_for_quant = model_dir

            try:
                pipe.model.to(use_device)
            except Exception:
                pipe.model.to("cpu")
        except Exception as e:
            st.warning(f"Falha ao carregar modelo '{model_str}': {e}")
            metrics_rows.append([
                label, ("huggingface" if is_online else "local"), target_col,
                n_periods, overlap_len, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
            continue

        q_levels, lower_q, upper_q = build_quantiles_for_model(ident_for_quant)

        try:
            if allow_long_horizon and n_periods > 64:
                pvals = predict_in_blocks(pipe, context_base, n_periods, q_levels)
            else:
                qs, _ = pipe.predict_quantiles(
                    context=context_base, prediction_length=n_periods, quantile_levels=q_levels
                )
                pvals = qs[0].detach().cpu().numpy()
        except Exception as e:
            st.warning(f"Falha ao prever com '{label}': {e}")
            metrics_rows.append([
                label, fonte, target_col, n_periods, overlap_len,
                np.nan, np.nan, np.nan, np.nan, np.nan, lower_q, upper_q
            ])
            continue

        q_lo_idx, q_md_idx, q_hi_idx = 0, q_levels.index(0.5), len(q_levels)-1
        df_fc_model = pd.DataFrame({
            date_col: future_index_full,
            f"{target_col}__{label}__pL": pvals[:, q_lo_idx],
            f"{target_col}__{label}__p50": pvals[:, q_md_idx],
            f"{target_col}__{label}__pU": pvals[:, q_hi_idx],
        })
        forecasts[label] = df_fc_model

        # métricas
        if overlap_len > 0 and df_teste_used[target_col].notna().any():
            y_true = df_teste_used[target_col].astype("float64").values
            y_p50  = df_fc_model[f"{target_col}__{label}__p50"].iloc[:overlap_len].astype("float64").values
            mae   = mean_absolute_error(y_true, y_p50)
            rmse  = np.sqrt(mean_squared_error(y_true, y_p50))
            mape  = mean_absolute_percentage_error(y_true, y_p50) * 100.0
            medae = median_absolute_error(y_true, y_p50)
            r2    = r2_score(y_true, y_p50)
        else:
            mae=rmse=mape=medae=r2=np.nan

        metrics_rows.append([
            label, fonte, target_col, n_periods, overlap_len,
            mae, rmse, mape, medae, r2, lower_q, upper_q
        ])

        # plot
        base_color = get_color_for_label(label)
        used_colors[label] = base_color
        fig.add_trace(go.Scatter(
            x=df_fc_model[date_col],
            y=df_fc_model[f"{target_col}__{label}__p50"],
            mode="lines",
            name=f"P50 — {label}",
            line=dict(width=2, color=base_color)
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([df_fc_model[date_col], df_fc_model[date_col][::-1]]),
            y=np.concatenate([
                df_fc_model[f"{target_col}__{label}__pU"],
                df_fc_model[f"{target_col}__{label}__pL"][::-1]
            ]),
            fill='toself',
            name=f"Intervalo — {label}",
            line=dict(width=0),
            fillcolor=base_color,
            opacity=0.20
        ))

        # merge no consolidado
        df_forecast = df_forecast.merge(df_fc_model, on=date_col, how="left")

    metrics_df = pd.DataFrame(metrics_rows, columns=metrics_cols)

    # ensemble
    if ensemble is not None and len(forecasts) > 0:
        method = ensemble.get("method", "mean")
        weights = ensemble.get("weights", {})
        models_for_ens = ensemble.get("models", list(forecasts.keys()))
        cols_pL  = [f"{target_col}__{m}__pL"  for m in models_for_ens if f"{target_col}__{m}__pL"  in df_forecast.columns]
        cols_p50 = [f"{target_col}__{m}__p50" for m in models_for_ens if f"{target_col}__{m}__p50" in df_forecast.columns]
        cols_pU  = [f"{target_col}__{m}__pU"  for m in models_for_ens if f"{target_col}__{m}__pU"  in df_forecast.columns]

        def combine(cols, qname):
            if not cols:
                return
            if method == "mean":
                df_forecast[f"{target_col}__ensemble__{qname}"] = df_forecast[cols].mean(axis=1)
            elif method == "median":
                df_forecast[f"{target_col}__ensemble__{qname}"] = df_forecast[cols].median(axis=1)
            elif method == "weighted":
                ws = np.array([weights.get(c.split("__")[1], 0.0) for c in cols], dtype="float64")
                if ws.sum() == 0:
                    df_forecast[f"{target_col}__ensemble__{qname}"] = df_forecast[cols].mean(axis=1)
                else:
                    ws = ws / ws.sum()
                    df_forecast[f"{target_col}__ensemble__{qname}"] = (df_forecast[cols].values * ws).sum(axis=1)
            else:
                df_forecast[f"{target_col}__ensemble__{qname}"] = df_forecast[cols].mean(axis=1)

        combine(cols_pL, "pL"); combine(cols_p50, "p50"); combine(cols_pU, "pU")

        # plot ensemble
        ens_col = f"{target_col}__ensemble__p50"
        if ens_col in df_forecast.columns:
            ens_color = "#000000"
            fig.add_trace(go.Scatter(
                x=df_forecast[date_col],
                y=df_forecast[ens_col],
                mode="lines",
                name="P50 — ensemble",
                line=dict(width=3, color=ens_color, dash="dash")
            ))
            if f"{target_col}__ensemble__pL" in df_forecast.columns and f"{target_col}__ensemble__pU" in df_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([df_forecast[date_col], df_forecast[date_col][::-1]]),
                    y=np.concatenate([
                        df_forecast[f"{target_col}__ensemble__pU"],
                        df_forecast[f"{target_col}__ensemble__pL"][::-1]
                    ]),
                    fill='toself',
                    name="Intervalo — ensemble",
                    line=dict(width=0),
                    fillcolor=ens_color,
                    opacity=0.12
                ))

    title_extra = (f" | Avaliado (sombreado): {shade_start.date()} → {shade_end.date()}" if overlap_len > 0 else "")
    fig.update_layout(
        title=f"Forecast {target_col} — {len(models)} modelos | Previsto: {n_periods} meses{title_extra}",
        template="plotly_white",
        xaxis_title="Data",
        yaxis_title=target_col,
        height=820,
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1),
        margin=dict(l=60, r=200, t=80, b=60)
    )

    return fig, metrics_df, forecasts, df_treino, df_teste, df_forecast


# =========================
# Sidebar — Carregamento e parâmetros
# =========================
with st.sidebar:
    st.header("1) Carregar dados")
    up = st.file_uploader("CSV ou Excel", type=["csv", "xlsx", "xls"])
    read_kwargs = {}
    if up is not None and up.name.lower().endswith(".csv"):
        read_kwargs["sep"] = st.text_input("Delimitador (CSV)", value=",")
    enc = st.text_input("Encoding (opcional)", value="")
    if enc.strip():
        read_kwargs["encoding"] = enc.strip()

    st.header("2) Seleção de colunas")
    date_col = st.text_input("Nome da coluna de data", value="Data")
    target_col = st.text_input("Nome da coluna alvo", value="ec_total_BR")

    st.header("3) Modelos (Amazon Chronos)")
    mode = st.radio("Origem dos modelos", ["Online (Hugging Face)", "Offline (pastas locais)"])

    # Lista de opções (usaremos só o SUFIXO na UI)
    amazon_online_options = [
        "amazon/chronos-t5-tiny", "amazon/chronos-t5-mini", "amazon/chronos-t5-small",
        "amazon/chronos-t5-base", "amazon/chronos-t5-large",
        "amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini",
        "amazon/chronos-bolt-small", "amazon/chronos-bolt-base",
    ]
    amazon_offline_options = [
        "amazon_chronos-t5-tiny", "amazon_chronos-t5-mini", "amazon_chronos-t5-small",
        "amazon_chronos-t5-base", "amazon_chronos-t5-large",
        "amazon_chronos-bolt-tiny", "amazon_chronos-bolt-mini",
        "amazon_chronos-bolt-small", "amazon_chronos-bolt-base",
    ]

    selected_models = []
    if mode == "Online (Hugging Face)":
        st.caption("Marque os modelos (só o final do nome é exibido).")
        cols = st.columns(3)
        for i, full in enumerate(amazon_online_options):
            short = full.split("/")[-1]
            if cols[i % 3].checkbox(short, value=False, key=f"online_{short}"):
                selected_models.append(full)
        local_base_dir = None
        force_offline = False
    else:
        local_base_dir = st.text_input("Pasta base dos modelos locais", value="")  # sem pré-carregar
        st.caption("Marque as pastas de modelos (mostrando apenas o final do nome).")
        cols = st.columns(3)
        for i, full in enumerate(amazon_offline_options):
            short = full.split("_")[-1] if "_" in full else full
            if cols[i % 3].checkbox(short, value=False, key=f"offline_{full}"):
                selected_models.append(full)
        force_offline = st.checkbox("Forçar OFFLINE (sem internet)", value=True)

    st.header("4) Parâmetros")
    test_periods = st.number_input("Períodos de teste (últimos N)", min_value=1, max_value=120, value=6, step=1)
    n_periods = st.number_input("Horizonte de previsão (meses)", min_value=1, max_value=240, value=60, step=1)
    device = st.selectbox("Device", options=["cpu", "cuda"], index=1 if torch.cuda.is_available() else 0)
    interval_alpha = st.slider("Intervalo alvo (alpha)", 0.50, 0.99, 0.95, 0.01)
    quantiles_str = st.text_input("Quantis (JSON, opcional) ex.: [0.1,0.5,0.9]", value="")
    quantile_levels = parse_json_or_none(quantiles_str)
    random_seed = st.number_input("Random seed", value=123, step=1)
    allow_long_horizon = st.checkbox("Permitir horizonte longo (>64) via blocos", value=True)

    st.header("5) Ensemble (opcional)")
    use_ens = st.checkbox("Ativar ensemble", value=False)
    ensemble_dict = None
    if use_ens:
        method = st.selectbox("Método", ["mean", "median", "weighted"], index=0)
        # Para escolher os modelos do ensemble, use os marcados
        models_for_ens = selected_models.copy()
        weights_str = st.text_input("Pesos (JSON, só se weighted) ex.: {'chronos-t5-mini':0.4,'chronos-bolt-small':0.6}", value="")
        ens = {"method": method, "models": [m.split("/")[-1] if "/" in m else m for m in models_for_ens]}
        wjs = parse_json_or_none(weights_str)
        if method == "weighted" and wjs:
            ens["weights"] = wjs
        ensemble_dict = ens

    st.header("6) Exportação")
    out_filename = st.text_input("Nome do arquivo .xlsx", value="forecast_resultados.xlsx")

# =========================
# Área principal
# =========================
if up is None:
    st.info("👈 Carregue um arquivo para começar.")
    st.stop()

# Ler arquivo
try:
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up, **read_kwargs)
    else:
        df = pd.read_excel(up)
except Exception as e:
    st.error(f"Erro ao ler arquivo: {e}")
    st.stop()

st.success(f"Arquivo carregado: {up.name} | {df.shape[0]} linhas × {df.shape[1]} colunas")
with st.expander("Prévia dos dados", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# Botão de execução
run = st.button("🚀 Rodar previsão agora")

if run:
    if len(selected_models) == 0:
        st.warning("Selecione pelo menos um modelo nas caixinhas antes de rodar.")
        st.stop()

    try:
        fig, met_df, fc_dict, df_treino, df_teste, df_forecast = chronos_forecast_multi_models_single_df(
            df=df,
            models=selected_models,
            target_col=target_col,
            test_periods=test_periods,
            n_periods=n_periods,
            device=device,
            force_offline=force_offline,
            date_col=date_col,
            interval_alpha=interval_alpha,
            quantile_levels=quantile_levels,
            local_base_dir=local_base_dir if local_base_dir else None,
            color_map=None,
            random_seed=int(random_seed) if random_seed is not None else None,
            allow_long_horizon=allow_long_horizon,
            ensemble=ensemble_dict
        )

        st.subheader("📈 Gráfico")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Métricas")
        st.dataframe(met_df, use_container_width=True)

        st.subheader("🧮 df_forecast (consolidado)")
        st.dataframe(df_forecast.tail(30), use_container_width=True)

        # Exportar para XLSX (buffer)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            met_df.to_excel(writer, index=False, sheet_name="metrics")
            df_treino.to_excel(writer, index=False, sheet_name="treino")
            df_teste.to_excel(writer, index=False, sheet_name="teste")
            df_forecast.to_excel(writer, index=False, sheet_name="forecast")

        st.download_button(
            label="💾 Baixar resultados (.xlsx)",
            data=buffer.getvalue(),
            file_name=out_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.exception(e)
