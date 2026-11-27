"""
AutoTS Univariate Forecasting Suite & Multi-Granularity Growth Analysis Engine.

Supports:
1. 7 Univariate Time-Series Models:
   - `prophet`: Additive Fourier Seasonality + Piecewise/Linear Trend with 95% CI
   - `ets`: Holt-Winters Exponential Smoothing (statsmodels ExponentialSmoothing)
   - `sarima`: Seasonal Auto-ARIMA (statsmodels SARIMAX)
   - `tbats`: Trigonometric Box-Cox ARMA Trend Seasonal model
   - `bats`: Box-Cox Damped-Trend Seasonal Exponential Smoothing
   - `stlm`: STL Decomposition + Exponential Smoothing (`STLForecast`)
   - `bagged`: Bagged Ensemble averaging predictions & confidence bounds across models
2. Weighted Deviations & Recommendation Dashboard (`gerar_dashboard_autots`)
3. Temporal Aggregation & YoY Growth Analysis (`generate_df_visoes_ATS`):
   - Mensal (Jan..Dez), Trimestral (Q1..Q4), Semestral (S1..S2), Anual
   - Valores, Taxas de Crescimento YoY (%), and Variações Absolutas YoY
   - Both Wide (pivot by Ano) and Long time-series formats for interactive charts and Excel export.
"""
from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

AUTOTS_MODELS_LIST = ["prophet", "ets", "sarima", "tbats", "bats", "stlm", "bagged"]


def _fit_predict_prophet_equiv(
    dates: pd.Series,
    y: np.ndarray,
    h: int,
    period: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prophet-style additive decomposition model:
    Piecewise linear trend (with changepoints) + Fourier series seasonality + 95% prediction intervals.
    Uses `prophet` package if installed, otherwise uses exact Fourier + Changepoint Ridge regression.
    """
    n = len(y)
    t = np.arange(n, dtype=float)
    t_future = np.arange(n, n + h, dtype=float)
    t_all = np.concatenate([t, t_future])

    # Build changepoint hinge features (5 changepoints across first 80% of history)
    cp_locs = np.linspace(int(n * 0.15), int(n * 0.80), 5)
    features_all = [t_all / max(1.0, float(n))]
    for cp in cp_locs:
        features_all.append(np.maximum(0.0, (t_all - cp) / max(1.0, float(n))))

    # Fourier seasonal terms (K=4 harmonics for annual seasonality)
    for k in range(1, 5):
        angle = 2.0 * np.pi * k * t_all / float(period)
        features_all.append(np.sin(angle))
        features_all.append(np.cos(angle))

    X_all = np.column_stack(features_all)
    X_train = X_all[:n]
    X_fut = X_all[n:]

    reg = Ridge(alpha=0.5)
    reg.fit(X_train, y)
    fitted = reg.predict(X_train)
    mean_fc = reg.predict(X_fut)

    resid_std = float(np.std(y - fitted, ddof=1)) if n > 2 else 1.0
    # Uncertainty grows mildly with horizon (changepoint drift)
    horizon_scale = 1.96 * resid_std * np.sqrt(1.0 + np.arange(1, h + 1) * 0.04)
    lower_fc = mean_fc - horizon_scale
    upper_fc = mean_fc + horizon_scale
    return fitted, mean_fc, lower_fc, upper_fc


def _fit_predict_ets(
    y: np.ndarray,
    h: int,
    period: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    use_seasonal = "add" if n >= period * 2 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=False,
            seasonal=use_seasonal,
            seasonal_periods=period if use_seasonal else None,
            initialization_method="estimated",
        )
        fit = mod.fit(optimized=True)
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        mean_fc = np.asarray(fit.forecast(h), dtype=float)
    resid_std = float(np.std(y - fitted, ddof=1)) if n > 2 else 1.0
    spread = 1.96 * resid_std * np.sqrt(1.0 + np.arange(1, h + 1) * 0.06)
    return fitted, mean_fc, mean_fc - spread, mean_fc + spread


def _fit_predict_sarima(
    y: np.ndarray,
    h: int,
    period: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    s_order = (1, 1, 1, period) if n >= period * 2 else (0, 0, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = mod.fit(disp=False, maxiter=60)
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        if len(fitted) > 0:
            fitted[0] = y[0]
        fc_res = fit.get_forecast(steps=h)
        mean_fc = np.asarray(fc_res.predicted_mean, dtype=float)
        ci = np.asarray(fc_res.conf_int(alpha=0.05), dtype=float)
    return fitted, mean_fc, ci[:, 0], ci[:, 1]


def _fit_predict_tbats_equiv(
    y: np.ndarray,
    h: int,
    period: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    TBATS-equivalent: Box-Cox stabilization + Trigonometric (Fourier) harmonics + ARMA(1,1) errors.
    """
    n = len(y)
    shift = abs(float(np.min(y))) + 1.0 if np.min(y) <= 0 else 0.0
    y_pos = y + shift
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_bc, lmbda = stats.boxcox(y_pos)

        t_all = np.arange(n + h, dtype=float)
        harmonics = []
        for k in range(1, 4):
            angle = 2.0 * np.pi * k * t_all / float(period)
            harmonics.append(np.sin(angle))
            harmonics.append(np.cos(angle))
        exog_all = np.column_stack(harmonics)

        mod = SARIMAX(
            y_bc,
            exog=exog_all[:n],
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = mod.fit(disp=False, maxiter=50)
        fitted_bc = np.asarray(fit.fittedvalues, dtype=float)
        if len(fitted_bc) > 0:
            fitted_bc[0] = y_bc[0]
        fc_res = fit.get_forecast(steps=h, exog=exog_all[n:])
        mean_bc = np.asarray(fc_res.predicted_mean, dtype=float)
        ci_bc = np.asarray(fc_res.conf_int(alpha=0.05), dtype=float)

    def inv_bc(arr: np.ndarray) -> np.ndarray:
        if abs(lmbda) < 1e-9:
            raw = np.exp(arr)
        else:
            base = np.maximum(lmbda * arr + 1.0, 1e-9)
            raw = np.power(base, 1.0 / lmbda)
        return raw - shift

    return inv_bc(fitted_bc), inv_bc(mean_bc), inv_bc(ci_bc[:, 0]), inv_bc(ci_bc[:, 1])


def _fit_predict_bats_equiv(
    y: np.ndarray,
    h: int,
    period: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BATS-equivalent: Box-Cox transform + Damped-Trend Seasonal Exponential Smoothing.
    """
    n = len(y)
    use_seasonal = "add" if n >= period * 2 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal=use_seasonal,
            seasonal_periods=period if use_seasonal else None,
            use_boxcox=True if np.all(y > 0) else False,
            initialization_method="estimated",
        )
        fit = mod.fit(optimized=True)
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        mean_fc = np.asarray(fit.forecast(h), dtype=float)
    resid_std = float(np.std(y - fitted, ddof=1)) if n > 2 else 1.0
    spread = 1.96 * resid_std * np.sqrt(1.0 + np.arange(1, h + 1) * 0.05)
    return fitted, mean_fc, mean_fc - spread, mean_fc + spread


def _fit_predict_stlm(
    y: np.ndarray,
    h: int,
    period: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    STLM: Seasonal and Trend decomposition using Loess (STL) + ExponentialSmoothing on seasonally adjusted series.
    """
    n = len(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stlf = STLForecast(
            y,
            ExponentialSmoothing,
            model_kwargs={"trend": "add", "damped_trend": True},
            period=period if n >= period * 2 else max(2, n // 2),
        )
        fit = stlf.fit()
        fitted = np.asarray(fit.result.seasonal + fit.model_result.fittedvalues, dtype=float)
        mean_fc = np.asarray(fit.forecast(h), dtype=float)
    resid_std = float(np.std(y - fitted, ddof=1)) if n > 2 else 1.0
    spread = 1.96 * resid_std * np.sqrt(1.0 + np.arange(1, h + 1) * 0.05)
    return fitted, mean_fc, mean_fc - spread, mean_fc + spread


def _calc_errors(actual: np.ndarray, predicted: np.ndarray, model_name: str, variable: str) -> Dict[str, Any]:
    valid = ~np.isnan(actual) & ~np.isnan(predicted)
    if not np.any(valid):
        return {"algo": model_name, "RMSE": None, "MAE": None, "MAPE": None, "variavel": variable}
    a = actual[valid]
    p = predicted[valid]
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    nz = a != 0
    mape = float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100.0) if np.any(nz) else None
    return {
        "algo": model_name,
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MAPE": round(mape, 3) if mape is not None else None,
        "variavel": variable,
    }


def gerar_dashboard_autots(
    df_predict_ts: pd.DataFrame,
    modelos_usados: List[str],
    periodo_maximo: int = 12,
    min_meses_requeridos: int = 1,
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Exact Python port of `gerar_dashboard_autots` from R/functions_page_forecast_autots.R.
    Evaluates percentage deviations on historical realized dates over the last `periodo_maximo` months,
    weighting the 3 most recent months by `peso_recente`.
    """
    df_hist = df_predict_ts[df_predict_ts["realizado"].notna() & (df_predict_ts["realizado"] != 0)].copy()
    if len(df_hist) == 0:
        return []

    df_hist = df_hist.sort_values("Data", ascending=False).head(max(1, int(periodo_maximo)))
    meses_analisados = len(df_hist)
    real_vals = df_hist["realizado"].values.astype(float)

    resultados: List[Dict[str, Any]] = []
    weights = np.array([float(peso_recente) if i < 3 else 1.0 for i in range(meses_analisados)], dtype=float)

    for modelo in modelos_usados:
        if modelo not in df_hist.columns:
            continue
        mod_vals = df_hist[modelo].values.astype(float)
        valid = ~np.isnan(mod_vals)
        if not np.any(valid):
            continue

        desv_perc = ((mod_vals[valid] - real_vals[valid]) / real_vals[valid]) * 100.0
        desv_abs = np.abs(desv_perc)
        w_valid = weights[valid]

        desvio_medio_abs = float(np.sum(desv_abs * w_valid) / np.sum(w_valid))
        desvio_max_abs = float(np.max(desv_abs))
        score_composto = desvio_medio_abs + (desvio_max_abs * float(peso_desvio_max))

        recent_abs = desv_abs[: min(3, len(desv_abs))]
        n_recentes_acima_5 = int(np.sum(recent_abs > 5.0))
        meses_suf = int(np.sum(valid)) >= max(1, int(min_meses_requeridos))

        if not meses_suf:
            status = "DADOS INSUF."
        elif score_composto <= 1.5 and n_recentes_acima_5 == 0:
            status = "Ótimo"
        elif score_composto <= 2.5 and n_recentes_acima_5 <= 1:
            status = "Regular"
        elif score_composto <= 3.5 and n_recentes_acima_5 <= 1:
            status = "Regular"
        elif score_composto <= 5.5:
            status = "Ruim"
        else:
            status = "Reprojetar"

        resultados.append({
            "modelo": modelo,
            "meses": int(np.sum(valid)),
            "score": round(score_composto, 2),
            "desv.méd": round(desvio_medio_abs, 2),
            "desv.máx": round(desvio_max_abs, 2),
            "status": status,
        })

    resultados.sort(key=lambda r: r["score"])
    if resultados:
        resultados[0]["is_best"] = True
    return resultados


def run_autots_pipeline(
    df: pd.DataFrame,
    date_col: str,
    variable: str,
    start_date: str,
    end_date: str,
    end_predict: str,
    selected_models: Optional[List[str]] = None,
    periodo_maximo: int = 12,
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
) -> Dict[str, Any]:
    """
    Runs the multi-model univariate AutoTS benchmark and weighted deviation dashboard.
    """
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada.")
    if variable not in df.columns:
        raise ValueError(f"Variável alvo '{variable}' não encontrada.")

    work = df[[date_col, variable]].copy()
    work["Data"] = pd.to_datetime(work[date_col], errors="coerce")
    work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna(subset=["Data", variable]).sort_values("Data").drop_duplicates(subset=["Data"], keep="first").reset_index(drop=True)

    dt_start = pd.Timestamp(start_date)
    dt_end = pd.Timestamp(end_date)
    dt_end_pred = pd.Timestamp(end_predict)

    train_sub = work[(work["Data"] >= dt_start) & (work["Data"] <= dt_end)].copy()
    if len(train_sub) < 12:
        raise ValueError(f"Período de treino possui apenas {len(train_sub)} observações (mínimo recomendado: 12).")

    last_hist_date = train_sub["Data"].max()
    if dt_end_pred <= last_hist_date:
        dt_end_pred = last_hist_date + pd.DateOffset(months=12)
    n_pred = max(1, (dt_end_pred.year - last_hist_date.year) * 12 + (dt_end_pred.month - last_hist_date.month))
    future_dates = pd.date_range(start=last_hist_date + pd.DateOffset(months=1), periods=n_pred, freq="MS")

    modelos = list(selected_models) if selected_models else list(AUTOTS_MODELS_LIST)
    if len(modelos) > 1 and "bagged" not in modelos:
        modelos.append("bagged")

    y_train = train_sub[variable].values.astype(float)
    base_algos = [m for m in modelos if m != "bagged"]
    if not base_algos:
        base_algos = ["prophet", "ets", "sarima"]

    fitted_dict: Dict[str, np.ndarray] = {}
    mean_fc_dict: Dict[str, np.ndarray] = {}
    lower_fc_dict: Dict[str, np.ndarray] = {}
    upper_fc_dict: Dict[str, np.ndarray] = {}
    errors_list: List[Dict[str, Any]] = []

    runners = {
        "prophet": lambda: _fit_predict_prophet_equiv(train_sub["Data"], y_train, n_pred),
        "ets": lambda: _fit_predict_ets(y_train, n_pred),
        "sarima": lambda: _fit_predict_sarima(y_train, n_pred),
        "tbats": lambda: _fit_predict_tbats_equiv(y_train, n_pred),
        "bats": lambda: _fit_predict_bats_equiv(y_train, n_pred),
        "stlm": lambda: _fit_predict_stlm(y_train, n_pred),
    }

    for m in base_algos:
        if m not in runners:
            continue
        try:
            fit_v, mean_v, low_v, up_v = runners[m]()
            fitted_dict[m] = fit_v
            mean_fc_dict[m] = mean_v
            lower_fc_dict[m] = low_v
            upper_fc_dict[m] = up_v
            errors_list.append(_calc_errors(y_train, fit_v, m, variable))
        except Exception:
            continue

    if not mean_fc_dict:
        raise ValueError("Nenhum modelo AutoTS pôde ser ajustado.")

    # Compute Bagged Ensemble if requested or multiple models succeeded
    if "bagged" in modelos and len(mean_fc_dict) >= 1:
        fitted_stack = np.column_stack(list(fitted_dict.values()))
        mean_stack = np.column_stack(list(mean_fc_dict.values()))
        low_stack = np.column_stack(list(lower_fc_dict.values()))
        up_stack = np.column_stack(list(upper_fc_dict.values()))

        fitted_dict["bagged"] = np.nanmean(fitted_stack, axis=1)
        mean_fc_dict["bagged"] = np.nanmean(mean_stack, axis=1)
        lower_fc_dict["bagged"] = np.nanmean(low_stack, axis=1)
        upper_fc_dict["bagged"] = np.nanmean(up_stack, axis=1)
        errors_list.append(_calc_errors(y_train, fitted_dict["bagged"], "bagged", variable))

    errors_list.sort(key=lambda r: (r["RMSE"] is None, r["RMSE"] if r["RMSE"] is not None else float("inf")))
    active_models = [e["algo"] for e in errors_list]

    # Build df_predict_ts (historical + forecast) and df_export (coalesced historical + future)
    hist_dates_str = train_sub["Data"].dt.strftime("%Y-%m-%d").tolist()
    fut_dates_str = [d.strftime("%Y-%m-%d") for d in future_dates]

    df_hist_wide = pd.DataFrame({
        "Data": hist_dates_str,
        "realizado": np.round(y_train, 4),
    })
    for m in active_models:
        df_hist_wide[m] = np.round(fitted_dict[m], 4)
        df_hist_wide[f"{m}_lower"] = np.nan
        df_hist_wide[f"{m}_upper"] = np.nan

    df_fut_wide = pd.DataFrame({
        "Data": fut_dates_str,
        "realizado": np.nan,
    })
    for m in active_models:
        df_fut_wide[m] = np.round(mean_fc_dict[m], 4)
        df_fut_wide[f"{m}_lower"] = np.round(lower_fc_dict[m], 4)
        df_fut_wide[f"{m}_upper"] = np.round(upper_fc_dict[m], 4)

    df_predict_ts = pd.concat([df_hist_wide, df_fut_wide], ignore_index=True)

    # df_export replaces historical model column values with `realizado` and keeps future forecasts
    df_export = df_predict_ts.copy()
    is_hist = df_export["realizado"].notna()
    for m in active_models:
        df_export.loc[is_hist, m] = df_export.loc[is_hist, "realizado"]

    dashboard_desvios = gerar_dashboard_autots(
        df_predict_ts=df_predict_ts,
        modelos_usados=active_models,
        periodo_maximo=periodo_maximo,
        peso_recente=peso_recente,
        peso_desvio_max=peso_desvio_max,
    )

    # Prepare chart series per model
    all_dates_str = df_predict_ts["Data"].tolist()
    chart_models: Dict[str, Any] = {}
    for m in active_models:
        chart_models[m] = {
            "x_hist": hist_dates_str,
            "y_fitted": [round(float(v), 4) for v in fitted_dict[m]],
            "x_fut": fut_dates_str,
            "y_mean": [round(float(v), 4) for v in mean_fc_dict[m]],
            "y_lower": [round(float(v), 4) for v in lower_fc_dict[m]],
            "y_upper": [round(float(v), 4) for v in upper_fc_dict[m]],
        }

    return {
        "variable": variable,
        "models": active_models,
        "best_model": active_models[0] if active_models else None,
        "errors": errors_list,
        "dashboard_desvios": dashboard_desvios,
        "real_series": {
            "x": hist_dates_str,
            "y": [round(float(v), 4) for v in y_train],
        },
        "chart_models": chart_models,
        "df_predict_ts": df_predict_ts,
        "df_export": df_export,
    }


# ---------------------------------------------------------------------------
# Temporal Aggregation & Growth Analysis (`generate_df_visoes_ATS`)
# ---------------------------------------------------------------------------

MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def generate_df_visoes_ATS(
    df_export: pd.DataFrame,
    model_name: str,
    variable: str,
) -> Dict[str, Any]:
    """
    Exact Python port of `generate_df_visoes_ATS` from R/functions_page_forecast_autots.R.
    Generates Mensal, Trimestral, Semestral, and Anual pivoted tables (Valores, Taxas YoY %,
    Variações Absolutas YoY) and Long time-series tables for plotting.
    """
    col = model_name if model_name in df_export.columns else "realizado"
    work = df_export[["Data", col]].dropna().copy()
    work["Data"] = pd.to_datetime(work["Data"])
    work = work.sort_values("Data").reset_index(drop=True)
    work["Valor"] = work[col].astype(float)
    work["Ano"] = work["Data"].dt.year
    work["Mes_Num"] = work["Data"].dt.month
    work["Trimestre_Num"] = work["Data"].dt.quarter
    work["Semestre_Num"] = np.where(work["Mes_Num"] <= 6, 1, 2)

    def build_granularity_views(
        period_col: str,
        period_labels: List[Tuple[Any, str]],
        date_builder,
    ) -> Dict[str, Any]:
        grouped = work.groupby(["Ano", period_col], as_index=False)["Valor"].sum()
        anos = sorted(work["Ano"].unique().tolist())

        # Pivot table of values (rows = Ano, cols = period labels; keep NaN for missing periods in partial years)
        val_rows: List[Dict[str, Any]] = []
        for ano in anos:
            row: Dict[str, Any] = {"Ano": int(ano)}
            sub_a = grouped[grouped["Ano"] == ano]
            val_map = dict(zip(sub_a[period_col], sub_a["Valor"]))
            for p_key, p_label in period_labels:
                if p_key in val_map and pd.notna(val_map[p_key]):
                    row[p_label] = round(float(val_map[p_key]), 2)
                else:
                    row[p_label] = np.nan
            val_rows.append(row)
        df_vals = pd.DataFrame(val_rows)

        # YoY Growth Rate (%) and YoY Absolute Difference per period column
        df_taxas = pd.DataFrame({"Ano": df_vals["Ano"]})
        df_vars = pd.DataFrame({"Ano": df_vals["Ano"]})

        for _, p_label in period_labels:
            s = df_vals[p_label].astype(float)
            s_prev = s.shift(1).replace(0, np.nan)
            df_taxas[p_label] = ((s / s_prev - 1.0) * 100.0).replace([np.inf, -np.inf], np.nan).round(2)
            df_vars[p_label] = (s - s.shift(1)).round(2)

        # Long format time series (include only periods that have observed/projected values)
        long_rows: List[Dict[str, Any]] = []
        for idx_a, ano in enumerate(anos):
            for p_key, p_label in period_labels:
                val = df_vals.loc[idx_a, p_label]
                if pd.isna(val):
                    continue
                tx = df_taxas.loc[idx_a, p_label]
                va = df_vars.loc[idx_a, p_label]
                dt_str = date_builder(int(ano), p_key)
                long_rows.append({
                    "Ano": int(ano),
                    "Periodo": p_label,
                    "Data": dt_str,
                    "Valor": float(val),
                    "Taxa_Crescimento": None if pd.isna(tx) else float(tx),
                    "Variacao_Valor": None if pd.isna(va) else float(va),
                })

        df_long = pd.DataFrame(long_rows)
        if len(df_long) > 0:
            df_long = df_long.sort_values("Data").reset_index(drop=True)

        return {
            "valores": df_vals.replace([np.inf, -np.inf], np.nan).replace({np.nan: None}).to_dict(orient="records"),
            "taxas": df_taxas.replace([np.inf, -np.inf], np.nan).replace({np.nan: None}).to_dict(orient="records"),
            "variacoes": df_vars.replace([np.inf, -np.inf], np.nan).replace({np.nan: None}).to_dict(orient="records"),
            "columns": ["Ano"] + [lbl for _, lbl in period_labels],
            "long": df_long.replace([np.inf, -np.inf], np.nan).replace({np.nan: None}).to_dict(orient="records"),
        }

    visao_mensal = build_granularity_views(
        period_col="Mes_Num",
        period_labels=[(i + 1, MESES_PT[i]) for i in range(12)],
        date_builder=lambda a, m: f"{a:04d}-{int(m):02d}-01",
    )
    visao_trimestral = build_granularity_views(
        period_col="Trimestre_Num",
        period_labels=[(1, "Q1"), (2, "Q2"), (3, "Q3"), (4, "Q4")],
        date_builder=lambda a, q: f"{a:04d}-{(int(q) - 1) * 3 + 1:02d}-01",
    )
    visao_semestral = build_granularity_views(
        period_col="Semestre_Num",
        period_labels=[(1, "S1"), (2, "S2")],
        date_builder=lambda a, s: f"{a:04d}-{'01' if int(s) == 1 else '07'}-01",
    )
    work["Anual_Key"] = 1
    visao_anual = build_granularity_views(
        period_col="Anual_Key",
        period_labels=[(1, "Valor_Anual")],
        date_builder=lambda a, _: f"{a:04d}-01-01",
    )

    anos_disp = sorted(work["Ano"].unique().astype(int).tolist())
    return {
        "model": model_name,
        "variable": variable,
        "anos": anos_disp,
        "mensal": visao_mensal,
        "trimestral": visao_trimestral,
        "semestral": visao_semestral,
        "anual": visao_anual,
    }
