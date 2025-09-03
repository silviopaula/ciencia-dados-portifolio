# App_Chronos.py
# --------------------------------------------------------------------------------------
# Autor: Dr. Silvio da Rosa Paula
# Requisitos:
#   pip install streamlit chronos-forecasting plotly scikit-learn pandas numpy torch openpyxl
# Execução:
#   streamlit run App_Chronos.py
# --------------------------------------------------------------------------------------

import os
import io
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from chronos import BaseChronosPipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")
!pip freeze > requirements.txt

# =========================================
# AJUDA / AUTORIA (fora do painel de inputs)
# =========================================
HELP_MD = r"""
### Objetivo
Função univariada com **Amazon Chronos (T5/Bolt)** para prever `n_periods` meses, gerar **quantis** (pL/p50/pU), calcular **métricas** no período de teste e montar um **DataFrame consolidado** (dados originais + projeções de vários modelos e **ensemble** opcional).

### Como usar (inputs essenciais)
- **Dados (`df`)**: selecione a **coluna de data** e a **coluna alvo** no app (qualquer nome é aceito). Recomenda-se frequência **mensal (MS)** e datas em ordem crescente.
- **Modelos (`models`)**:
  - **Online (Hugging Face)**: nomes **com “/”** (ex.: `amazon/chronos-bolt-small`).
  - **Offline (local)**: nomes **sem “/”** (ex.: `amazon_chronos-bolt-small`) + `local_base_dir`.
- **Horizonte**: `n_periods` (meses). Se >64, a função usa **blocos**.
- **Quantis**: informe `quantile_levels` (ex.: `[0.1,0.5,0.9]`) ou `interval_alpha` (ex.: `0.95`).
  - **Bolt** aceita quantis apenas em **[0.1, 0.9]** (ajuste automático).

### Saídas
- **Gráfico (Plotly)**: histórico (treino/teste), **P50** por modelo e **intervalos** translúcidos.
- **Métricas**: MAE, RMSE, MAPE, MedAE, R² (no período de teste).
- **`df_forecast`**: df original + linhas futuras + colunas `__pL/__p50/__pU` por modelo (e `__ensemble__*` se usado).

### Dicas
- **Bolt-small** costuma ter o melhor custo-benefício.
- **Ensemble** (média/mediana/ponderado) pode estabilizar e melhorar a acurácia.
- **Offline** evita downloads repetidos e acelera lotes de modelos.

---

**Dados demo (pré-carregados)**  
Usamos **Consumo de energia (Brasil)** já preparado (mensal, com totais e quebras regionais/classes).  
Arquivo Excel: **`consumo_energia_pronto.xlsx`** (coluna `Data` + colunas `ec_*`/`nc_*`, ex.: `ec_total_BR`).  
Fonte: https://github.com/silviopaula/ciencia-dados-portifolio/raw/refs/heads/main/Time_Series_Forecast_Chronos_Amazon/dados/consumo_energia_pronto.xlsx

---

**Autoria**  
**Dr. Silvio da Rosa Paula**  
GitHub: https://silviopaula.github.io/
"""


# =========================================
# FUNÇÃO PRINCIPAL DE FORECAST (univariada)
# =========================================
def function_chronos_forecast(
    df: pd.DataFrame,
    models: list[str],
    target_col: str = "ec_total_BR",
    test_periods: int = 12,
    n_periods: Optional[int] = None,      # horizonte desejado
    device: str = "cpu",
    force_offline: bool = False,
    date_col: str = "Data",
    interval_alpha: float = 0.95,         # usado se quantile_levels=None
    quantile_levels: Optional[list[float]] = None,  # ex.: [0.1, 0.5, 0.9]
    local_base_dir: Optional[str] = None,
    color_map: Optional[dict] = None,
    random_seed: Optional[int] = 123,
    allow_long_horizon: bool = True,      # se >64, faz blocos e concatena
    ensemble: Optional[dict] = None,      # {"method": "mean"|"median"|"weighted", "weights": {"label": w, ...}, "models": ["label", ...]}
):
    """
    Retorna:
      fig (Plotly), metrics_df, forecasts (dict[label] -> DF com Data+pL/p50/pU),
      df_treino, df_teste, df_forecast (df original + colunas de projeção por modelo e, se definido, ensemble_*).

    Observações:
    - Modelos com "/" são online (Hugging Face); sem "/" => pasta local em local_base_dir.
    - BOLT aceita quantis em [0.1, 0.9]; ajustamos automaticamente se precisar.
    - Intervalos usam a MESMA cor do modelo com transparência. A legenda do intervalo NÃO mostra valores dos quantis.
    """

    # Seeds (determinismo prático)
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        try:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Paleta e utilitários
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

    # Preparação do DF
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada.")
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada.")

    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        try:
            df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    non_missing_idx = df[df[target_col].notna()].index
    if len(non_missing_idx) < max(1, test_periods):
        raise ValueError(f"Não há {test_periods} pontos não-nulos em '{target_col}' para formar o teste.")

    test_idx = non_missing_idx[-test_periods:]
    train_idx = non_missing_idx[:-test_periods]
    df_treino = df.loc[train_idx, [date_col, target_col]].reset_index(drop=True)
    df_teste = df.loc[test_idx, [date_col, target_col]].reset_index(drop=True)

    # Horizonte desejado
    if n_periods is None:
        n_periods = test_periods
    n_periods = int(n_periods)

    # Índices temporais de previsão
    start_date = df_teste[date_col].iloc[0] if len(df_teste) > 0 else (df_treino[date_col].max() + pd.offsets.MonthBegin(1))
    future_index_full = pd.date_range(start=start_date, periods=n_periods, freq="MS")

    # Interseção p/ métricas
    overlap_len = min(n_periods, len(df_teste))
    df_teste_used = df_teste.iloc[:overlap_len].copy()
    shade_start = df_teste_used[date_col].iloc[0] if overlap_len > 0 else None
    shade_end = df_teste_used[date_col].iloc[-1] if overlap_len > 0 else None

    # Device
    use_device = "cuda" if (device.lower() == "cuda" and torch.cuda.is_available()) else "cpu"
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("[AVISO] CUDA não disponível. Usando CPU.")

    # Gráfico base
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

    # Métricas
    metrics_cols = [
        "Label", "Fonte", "Variável", "Previsto", "Avaliado_sobre",
        "MAE", "RMSE", "MAPE (%)", "MedAE", "R2", "Lower_q", "Upper_q"
    ]
    metrics_rows = []
    forecasts = {}

    # Contexto treino
    y_hist = df_treino[target_col].dropna().values.astype("float32")
    context_base = torch.from_numpy(y_hist).unsqueeze(0)  # [1, T]

    # Quantis
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
            if lo >= mid:
                lo = 0.1
            if hi <= mid:
                hi = 0.9
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

    # Previsão em blocos (>64)
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
            np_block = qs[0].detach().cpu().numpy()  # [step, len(q_levels)]
            out_list.append(np_block)
            # atualiza contexto com mediana (p50) para seguir
            p50_idx = q_levels.index(0.5)
            p50_vals = qs[0, :, p50_idx].detach().cpu()
            y_ctx = torch.cat([y_ctx, p50_vals.unsqueeze(0)], dim=1)
            remaining -= step
        return np.concatenate(out_list, axis=0)

    # Loop de modelos
    for model_str in models:
        is_online = ("/" in model_str)
        label = model_str.split("/")[-1] if is_online else model_str

        # Carregar
        try:
            if is_online:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
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
            print(f"[ERRO] Falha ao carregar modelo '{model_str}': {e}")
            metrics_rows.append([
                label, ("huggingface" if is_online else "local"), target_col,
                n_periods, overlap_len, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
            continue

        # Quantis do modelo
        q_levels, lower_q, upper_q = build_quantiles_for_model(ident_for_quant)

        # Prever
        try:
            if allow_long_horizon and n_periods > 64:
                pvals = predict_in_blocks(pipe, context_base, n_periods, q_levels)
            else:
                qs, _ = pipe.predict_quantiles(
                    context=context_base, prediction_length=n_periods, quantile_levels=q_levels
                )
                pvals = qs[0].detach().cpu().numpy()
        except Exception as e:
            print(f"[ERRO] Falha ao prever com '{label}': {e}")
            metrics_rows.append([
                label, fonte, target_col, n_periods, overlap_len,
                np.nan, np.nan, np.nan, np.nan, np.nan, lower_q, upper_q
            ])
            continue

        # Monta DF do modelo
        q_lo_idx, q_md_idx, q_hi_idx = 0, q_levels.index(0.5), len(q_levels) - 1
        df_fc_model = pd.DataFrame({
            date_col: future_index_full,
            f"{target_col}__{label}__pL": pvals[:, q_lo_idx],
            f"{target_col}__{label}__p50": pvals[:, q_md_idx],
            f"{target_col}__{label}__pU": pvals[:, q_hi_idx],
        })
        forecasts[label] = df_fc_model

        # Métricas na interseção com teste (usando p50)
        if overlap_len > 0 and df_teste_used[target_col].notna().any():
            y_true = df_teste_used[target_col].astype("float64").values
            y_p50 = df_fc_model[f"{target_col}__{label}__p50"].iloc[:overlap_len].astype("float64").values
            mae = mean_absolute_error(y_true, y_p50)
            rmse = np.sqrt(mean_squared_error(y_true, y_p50))
            mape = mean_absolute_percentage_error(y_true, y_p50) * 100.0
            medae = median_absolute_error(y_true, y_p50)
            r2 = r2_score(y_true, y_p50)
        else:
            mae = rmse = mape = medae = r2 = np.nan

        metrics_rows.append([
            label, fonte, target_col, n_periods, overlap_len,
            mae, rmse, mape, medae, r2, lower_q, upper_q
        ])

        # Plot linhas/intervalos
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
            fill="toself",
            name=f"Intervalo — {label}",
            line=dict(width=0),
            fillcolor=base_color,
            opacity=0.20
        ))

        # Mescla no consolidado
        df_forecast = df_forecast.merge(df_fc_model, on=date_col, how="left")

    # Metrics DF
    metrics_df = pd.DataFrame(metrics_rows, columns=metrics_cols)

    # Ensemble (opcional)
    if ensemble is not None:
        method = ensemble.get("method", "mean")
        weights = ensemble.get("weights", {})
        models_for_ens = ensemble.get("models", list(forecasts.keys()))

        cols_pL = [f"{target_col}__{m}__pL" for m in models_for_ens if f"{target_col}__{m}__pL" in df_forecast.columns]
        cols_p50 = [f"{target_col}__{m}__p50" for m in models_for_ens if f"{target_col}__{m}__p50" in df_forecast.columns]
        cols_pU = [f"{target_col}__{m}__pU" for m in models_for_ens if f"{target_col}__{m}__pU" in df_forecast.columns]

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

        combine(cols_pL, "pL")
        combine(cols_p50, "p50")
        combine(cols_pU, "pU")

        # Plot ensemble (p50)
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
                    fill="toself",
                    name="Intervalo — ensemble",
                    line=dict(width=0),
                    fillcolor=ens_color,
                    opacity=0.12
                ))

    # Layout final
    title_extra = f" | Avaliado (sombreado): {shade_start.date()} → {shade_end.date()}" if overlap_len > 0 else ""
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


# =========================================
# STREAMLIT APP
# =========================================
st.set_page_config(page_title="Chronos Forecast (Univariado)", layout="wide")

# Logo no título
st.markdown(
    """
<div align="center">
  <img src="https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/chronos-logo.png" width="30%">
</div>
""",
    unsafe_allow_html=True,
)
st.title("Chronos Forecast — Univariado (T5/Bolt)")

# Tabs: App e Ajuda (help fora do painel de inputs)
tab_app, tab_help = st.tabs(["App", "❓ Ajuda"])
with tab_help:
    st.markdown(HELP_MD, unsafe_allow_html=True)

with tab_app:
    # ----------------------------
    # Sidebar — Configuração (apenas inputs)
    # ----------------------------
    with st.sidebar:
        # Autor + Git acima dos inputs (sem LinkedIn)
        st.markdown("**Autor:** Dr. Silvio da Rosa Paula  \n**GitHub:** https://silviopaula.github.io/")
        st.markdown("---")

        st.header("Configuração")

        # Dataset: upload ou demo
        st.subheader("Dados")
        up = st.file_uploader("Envie um CSV/Excel (opcional) — se vazio, usamos o dataset demo",
                              type=["csv", "xlsx", "xls"])

        # Demo: Consumo de energia (Brasil)
        demo_url = "https://github.com/silviopaula/ciencia-dados-portifolio/raw/refs/heads/main/Time_Series_Forecast_Chronos_Amazon/dados/consumo_energia_pronto.xlsx"

        # Split e horizonte
        test_periods = st.number_input("Período de teste (meses)", min_value=1, max_value=48, value=12, step=1)
        n_periods = st.number_input("Horizonte de previsão (meses)", min_value=1, max_value=240, value=60, step=1)

        # Dispositivo
        dev_choice = st.selectbox("Dispositivo", options=["auto", "cpu", "cuda"], index=0)
        device = "cuda" if (dev_choice == "auto" and torch.cuda.is_available()) else (dev_choice if dev_choice != "auto" else "cpu")
        st.caption(f"GPU disponível: **{torch.cuda.is_available()}** · Usando: **{device}**")

        # Modo offline/online
        force_offline = st.checkbox("Rodar OFFLINE (pastas locais)", value=False)
        local_base_dir = None
        if force_offline:
            # Usuário informa se desejar (sem caminho pré-carregado)
            local_base_dir = st.text_input("Diretório raiz dos modelos (offline)", value="", placeholder="Ex.: D:/Models/chronos (opcional)")

        # Modelos
        st.subheader("Modelos")
        online_models = [
            "amazon/chronos-bolt-tiny",
            "amazon/chronos-bolt-mini",
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
            "amazon/chronos-t5-tiny",
            "amazon/chronos-t5-mini",
            "amazon/chronos-t5-small",
            "amazon/chronos-t5-base",
            "amazon/chronos-t5-large",
        ]
        offline_models = [m.replace("amazon/", "amazon_").replace("/", "-") for m in online_models]

        if force_offline:
            model_choices = offline_models
            default_sel = ["amazon_chronos-bolt-small"]
        else:
            model_choices = online_models
            default_sel = ["amazon/chronos-bolt-small"]

        selected_models = st.multiselect("Selecione os modelos", options=model_choices, default=default_sel)

        # Quantis
        st.subheader("Quantis / Intervalo")
        q_mode = st.radio("Escolha", options=["Usar interval_alpha", "Listar quantile_levels"], index=0)
        quantile_levels = None
        interval_alpha = 0.95
        if q_mode == "Usar interval_alpha":
            interval_alpha = st.slider("interval_alpha (nível do intervalo)", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
        else:
            q_text = st.text_input("quantile_levels (ex.: 0.1,0.5,0.9)", value="0.1,0.5,0.9")
            try:
                quantile_levels = [float(x.strip()) for x in q_text.split(",") if x.strip()]
                quantile_levels = sorted(set(quantile_levels))
                if 0.5 not in quantile_levels:
                    quantile_levels.append(0.5)
                    quantile_levels = sorted(quantile_levels)
            except Exception:
                st.warning("Não consegui interpretar os quantis. Voltando para interval_alpha=0.95.")
                quantile_levels = None
                interval_alpha = 0.95

        # Outras configs
        random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=123, step=1)
        allow_long_horizon = st.checkbox("Permitir horizonte longo (>64) com blocos", value=True)

        # Ensemble
        st.subheader("Ensemble (opcional)")
        use_ens = st.checkbox("Ativar ensemble", value=False)
        ensemble_dict = None
        if use_ens:
            method = st.selectbox("Método", options=["mean", "median", "weighted"], index=2)
            ens_models = st.multiselect("Modelos usados no ensemble", options=selected_models, default=selected_models)
            weights_map = {}
            if method == "weighted":
                st.caption("Defina pesos (só para os modelos escolhidos). Eles serão normalizados.")
                for m in ens_models:
                    w = st.number_input(f"Peso — {m}", min_value=0.0, max_value=9999.0, value=1.0, step=0.1, key=f"w_{m}")
                    weights_map[m.split("/")[-1] if "/" in m else m] = float(w)
            ensemble_dict = {"method": method}
            if method == "weighted":
                ensemble_dict["weights"] = {k.split("/")[-1] if "/" in k else k: v for k, v in weights_map.items()}
            ensemble_dict["models"] = [m.split("/")[-1] if "/" in m else m for m in ens_models]

    # ----------------------------
    # Carregamento de dados (demo se vazio)
    # ----------------------------
    df = None
    if up is None:
        st.info("Nenhum arquivo enviado — carregando **dataset demo: Consumo de energia (Brasil)**.")
        try:
            # Excel remoto com colunas "Data" e várias "ec_*" / "nc_*"
            df = pd.read_excel(demo_url)  # requer openpyxl
            st.success(f"Dataset demo carregado — {df.shape[0]} linhas × {df.shape[1]} colunas")
        except Exception as e:
            st.error(f"Falha ao carregar dataset demo: {e}")
            st.stop()
    else:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
            st.success(f"Arquivo carregado: {up.name} | {df.shape[0]} linhas × {df.shape[1]} colunas")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

    # Prévia
    with st.expander("Prévia dos dados", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # ----------------------------
    # Seleção de colunas (o usuário escolhe)
    # ----------------------------
    st.subheader("Seleção de colunas")
    all_cols = list(df.columns)

    default_date = "Data" if "Data" in all_cols else next((c for c in all_cols if c.lower() in ["data", "date", "timestamp", "time"]), all_cols[0])
    default_target = "ec_total_BR" if "ec_total_BR" in all_cols else next((c for c in all_cols if c.lower() in ["target", "valor", "y", "consumo"]), all_cols[-1])

    date_col = st.selectbox("Nome da coluna de data (date_col)", options=all_cols, index=all_cols.index(default_date))
    target_col = st.selectbox("Nome da coluna alvo (target_col)", options=all_cols, index=all_cols.index(default_target))

    # ----------------------------
    # Botão de execução
    # ----------------------------
    run = st.button("▶️ Rodar previsão", use_container_width=True)

    # ----------------------------
    # Execução
    # ----------------------------
    if run:
        if len(selected_models) == 0:
            st.warning("Selecione pelo menos um modelo.")
            st.stop()

        if date_col not in df.columns or target_col not in df.columns:
            st.error(f"O DataFrame precisa ter as colunas selecionadas '{date_col}' (data) e '{target_col}' (alvo).")
            st.stop()

        try:
            fig, met_df, fc_dict, df_treino, df_teste, df_forecast = function_chronos_forecast(
                df=df,
                models=selected_models,
                target_col=target_col,
                test_periods=int(test_periods),
                n_periods=int(n_periods),
                device=device,
                force_offline=force_offline,
                date_col=date_col,
                interval_alpha=float(interval_alpha),
                quantile_levels=quantile_levels,
                local_base_dir=local_base_dir if (force_offline and local_base_dir and local_base_dir.strip()) else None,
                color_map=None,
                random_seed=int(random_seed) if random_seed is not None else None,
                allow_long_horizon=allow_long_horizon,
                ensemble=ensemble_dict
            )

            # Gráfico
            st.subheader("Gráfico")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            # Métricas
            st.subheader("Métricas (período de teste)")
            fmt_cols = ["MAE", "RMSE", "MAPE (%)", "MedAE", "R2"]
            met_show = met_df.copy()
            for c in fmt_cols:
                if c in met_show.columns:
                    met_show[c] = pd.to_numeric(met_show[c], errors="coerce")
            st.dataframe(met_show, use_container_width=True)

            # Forecast consolidado (preview)
            st.subheader("Forecast consolidado (preview)")
            st.dataframe(df_forecast.tail(24), use_container_width=True)

            # Download Excel (robusto sem xlsxwriter)
            st.subheader("Baixar resultados")
            buffer = io.BytesIO()
            try:
                # tenta xlsxwriter (se instalado)
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as xw:
                    df_treino.to_excel(xw, "treino", index=False)
                    df_teste.to_excel(xw, "teste", index=False)
                    met_df.to_excel(xw, "metricas", index=False)
                    df_forecast.to_excel(xw, "forecast", index=False)
            except ModuleNotFoundError:
                # fallback para openpyxl (já no requirements)
                with pd.ExcelWriter(buffer, engine="openpyxl") as xw:
                    df_treino.to_excel(xw, "treino", index=False)
                    df_teste.to_excel(xw, "teste", index=False)
                    met_df.to_excel(xw, "metricas", index=False)
                    df_forecast.to_excel(xw, "forecast", index=False)

            st.download_button(
                label="📥 Baixar Excel (treino/teste/métricas/forecast)",
                data=buffer.getvalue(),
                file_name="chronos_forecast_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        except Exception as e:
            st.exception(e)
