"""
ARIMA / SARIMAX Time-Series Forecasting Engine.
Ported from R/functions_page_forecast_arima.R and R/modulo_forecast_arima.R.

Supports:
- Date standardization to 1st of month
- Collinearity check (`check_collinearity`) for exogenous predictors
- Automatic order discovery (`auto.arima` via AIC search) + generation of `num_models` variations
- Manual ARIMA/SARIMA `(p, d, q)(P, D, Q)_m` configuration
- Both Univariate (`arima_uni_i`) and Multivariate with Exogenous regressors (`arima_multi_i`)
- 95% confidence intervals (`Lo95`, `Hi95`), error metrics (`MAPE`, `RMSE`, `MAE`),
  deviation statistics (`min`, `max`, `mean`, `cum` in value and %), and
  weighted Recommendation Dashboard (`gerar_dashboard_recomendacao_TS`).
"""
from __future__ import annotations
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def check_collinearity(df: pd.DataFrame, variables: List[str], threshold: float = 0.99) -> bool:
    if not variables or len(variables) <= 1:
        return False
    valid_vars = [v for v in variables if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
    if len(valid_vars) <= 1:
        return False
    sub = df[valid_vars].dropna()
    if len(sub) < 5:
        return True
    corr = sub.corr().abs().values
    np.fill_diagonal(corr, 0.0)
    return bool(np.any(corr > threshold))


def _standardize_to_month_start(dt_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(dt_series, errors="coerce")
    return pd.to_datetime(ts.dt.strftime("%Y-%m-01"), errors="coerce")


def _estimate_d(y: np.ndarray) -> int:
    try:
        res = adfuller(y, autolag="AIC")
        return 0 if res[1] < 0.05 else 1
    except Exception:
        return 1


def _find_best_auto_arima_order(y: np.ndarray, seasonal: bool = True, period: int = 12) -> Tuple[int, int, int, int, int, int]:
    """Fast AIC-based grid search for base ARIMA / SARIMA order (like forecast::auto.arima)."""
    d = _estimate_d(y)
    D = 1 if (seasonal and len(y) >= period * 2) else 0

    candidates: List[Tuple[int, int, int, int, int, int]] = [
        (1, d, 1, 1 if seasonal else 0, D, 1 if seasonal else 0),
        (1, d, 0, 1 if seasonal else 0, D, 0),
        (0, d, 1, 0, D, 1 if seasonal else 0),
        (2, d, 1, 1 if seasonal else 0, D, 0),
        (0, d, 0, 1 if seasonal else 0, D, 1 if seasonal else 0),
        (1, d, 2, 0, D, 1 if seasonal else 0),
    ]

    best_order = candidates[0]
    best_aic = float("inf")
    for order6 in candidates:
        p, d_o, q, P, D_o, Q = order6
        try:
            s_order = (P, D_o, Q, period) if seasonal else (0, 0, 0, 0)
            mod = SARIMAX(
                y,
                order=(p, d_o, q),
                seasonal_order=s_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=40)
            if math.isfinite(res.aic) and res.aic < best_aic:
                best_aic = float(res.aic)
                best_order = order6
        except Exception:
            continue

    return best_order


def _generate_order_variations(
    base_order: Tuple[int, int, int, int, int, int],
    num_variations: int,
    seasonal: bool,
    seed: int = 123,
) -> List[Tuple[int, int, int, int, int, int]]:
    rng = np.random.default_rng(seed)
    variations: List[Tuple[int, int, int, int, int, int]] = [base_order]
    seen = {base_order}
    p0, d0, q0, P0, D0, Q0 = base_order

    attempts = 0
    while len(variations) < num_variations and attempts < num_variations * 12:
        attempts += 1
        p = int(max(0, p0 + int(rng.choice([-1, 0, 1]))))
        d = int(max(0, min(2, d0 + int(rng.choice([0, 0, 1, -1])))))
        q = int(max(0, q0 + int(rng.choice([-1, 0, 1]))))
        if seasonal:
            P = int(max(0, min(2, P0 + int(rng.choice([-1, 0, 1])))))
            D = int(max(0, min(1, D0)))
            Q = int(max(0, min(2, Q0 + int(rng.choice([-1, 0, 1])))))
        else:
            P, D, Q = 0, 0, 0

        cand = (p, d, q, P, D, Q)
        if cand not in seen:
            seen.add(cand)
            variations.append(cand)

    return variations


def _calc_error_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Optional[float]]:
    if len(actual) == 0 or len(predicted) == 0:
        return {"MAPE": None, "RMSE": None, "MAE": None}
    n = min(len(actual), len(predicted))
    a = actual[:n].astype(float)
    p = predicted[:n].astype(float)
    valid = ~np.isnan(a) & ~np.isnan(p)
    if not np.any(valid):
        return {"MAPE": None, "RMSE": None, "MAE": None}
    a = a[valid]
    p = p[valid]
    a_safe = np.where(a == 0, 1e-10, a)
    mape = float(np.mean(np.abs((a - p) / np.abs(a_safe))) * 100.0)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    return {
        "MAPE": round(mape, 3),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
    }


def gerar_dashboard_recomendacao_arima(
    per_month_pct_deviations: Dict[str, List[Tuple[str, float]]],
    errors_list: List[Dict[str, Any]],
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
    periodo_maximo: int = 12,
) -> List[Dict[str, Any]]:
    """
    Generates the ARIMA recommendation table matching `gerar_dashboard_recomendacao`
    and `gerar_dashboard_recomendacao_TS`.
    """
    err_by_model = {e["model"]: e for e in errors_list}
    rows: List[Dict[str, Any]] = []

    for model_name, err_info in err_by_model.items():
        month_devs = per_month_pct_deviations.get(model_name, [])
        # Sort descending by date and keep up to `periodo_maximo`
        month_devs_sorted = sorted(month_devs, key=lambda x: x[0], reverse=True)[: max(1, int(periodo_maximo))]
        meses_analisados = len(month_devs_sorted)

        if meses_analisados == 0:
            rows.append({
                "modelo": model_name,
                "ordem": err_info.get("order", ""),
                "tipo": err_info.get("type", ""),
                "meses": 0,
                "RMSE": err_info.get("RMSE"),
                "MAE": err_info.get("MAE"),
                "MAPE": err_info.get("MAPE"),
                "desv.méd_pond": None,
                "desv.máx_perc": None,
                "score": None,
                "status": "SEM PERÍODO TESTE",
            })
            continue

        abs_devs = np.array([abs(v) for _, v in month_devs_sorted], dtype=float)
        # First 3 are most recent months weighted by `peso_recente`
        weights = np.array([float(peso_recente) if idx < 3 else 1.0 for idx in range(meses_analisados)], dtype=float)
        desv_med_pond = float(np.sum(abs_devs * weights) / np.sum(weights))
        desv_max = float(np.max(abs_devs))
        score_comp = desv_med_pond + (desv_max * float(peso_desvio_max))
        recent_over_5 = int(np.sum(abs_devs[:3] > 5.0))

        if score_comp <= 1.5 and recent_over_5 == 0:
            status = "Ótimo"
        elif score_comp <= 2.5 and recent_over_5 <= 1:
            status = "Recomendado"
        elif score_comp <= 3.5 and recent_over_5 <= 1:
            status = "Aceitável"
        elif score_comp <= 5.5:
            status = "Avaliar"
        else:
            status = "Reprojetar"

        rows.append({
            "modelo": model_name,
            "ordem": err_info.get("order", ""),
            "tipo": err_info.get("type", ""),
            "meses": meses_analisados,
            "RMSE": err_info.get("RMSE"),
            "MAE": err_info.get("MAE"),
            "MAPE": err_info.get("MAPE"),
            "desv.méd_pond": round(desv_med_pond, 2),
            "desv.máx_perc": round(desv_max, 2),
            "score": round(score_comp, 2),
            "status": status,
        })

    rows.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else (r["RMSE"] or float("inf"))))
    return rows


def run_arima_models_extended(
    df: pd.DataFrame,
    date_col: str,
    target_var: str,
    exog_vars: Optional[List[str]] = None,
    data_inicial: Optional[str] = None,
    data_atual: Optional[str] = None,
    end_predict: Optional[str] = None,
    n_teste: int = 3,
    seasonal: bool = True,
    period: int = 12,
    num_models: int = 5,
    manual_mode: bool = False,
    arima_params: Optional[Dict[str, int]] = None,
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Port of `run_arima_models_extended` from R/functions_page_forecast_arima.R.
    """
    start_time = time.time()
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada.")
    if target_var not in df.columns:
        raise ValueError(f"Variável alvo '{target_var}' não encontrada.")

    work = df.copy()
    work["Data"] = _standardize_to_month_start(work[date_col])
    work = work.dropna(subset=["Data", target_var]).sort_values("Data").drop_duplicates(subset=["Data"], keep="first").reset_index(drop=True)

    if len(work) == 0:
        raise ValueError("Não restaram dados válidos após limpeza da coluna de data e variável alvo.")

    dt_ini = _standardize_to_month_start(pd.Series([data_inicial or work["Data"].min()])).iloc[0]
    dt_atual = _standardize_to_month_start(pd.Series([data_atual or work["Data"].max()])).iloc[0]

    if end_predict:
        dt_end_pred = _standardize_to_month_start(pd.Series([end_predict])).iloc[0]
        if pd.isna(dt_end_pred) or dt_end_pred <= dt_atual:
            dt_end_pred = dt_atual + pd.DateOffset(months=12)
    else:
        dt_end_pred = dt_atual + pd.DateOffset(months=12)

    n_teste_int = max(0, int(n_teste or 0))
    if n_teste_int == 0:
        cutoff_date = dt_atual
        test_df = pd.DataFrame()
        test_y = np.array([], dtype=float)
        test_dates: List[pd.Timestamp] = []
    else:
        cutoff_date = dt_atual - pd.DateOffset(months=n_teste_int)
        test_df = work[(work["Data"] > cutoff_date) & (work["Data"] <= dt_atual)].copy()
        test_y = test_df[target_var].values.astype(float)
        test_dates = list(test_df["Data"])

    train_df = work[(work["Data"] >= dt_ini) & (work["Data"] <= cutoff_date)].copy()
    min_required = max(12, int(period) * 2) if seasonal else 8
    if len(train_df) < min_required:
        raise ValueError(
            f"Dados de treinamento insuficientes ({len(train_df)} observações). São necessários pelo menos {min_required} pontos."
        )

    train_y = train_df[target_var].values.astype(float)

    clean_exog = [v for v in (exog_vars or []) if v in work.columns and v not in (date_col, "Data", target_var) and pd.api.types.is_numeric_dtype(work[v])]
    has_exog = len(clean_exog) > 0
    collinear_warning = check_collinearity(train_df, clean_exog) if has_exog else False

    h_test = len(test_y)
    # Months from dt_atual to dt_end_pred
    h_future = max(1, (dt_end_pred.year - dt_atual.year) * 12 + (dt_end_pred.month - dt_atual.month))
    h_total = h_test + h_future

    future_dates = list(pd.date_range(start=dt_atual + pd.DateOffset(months=1), periods=h_future, freq="MS"))
    all_forecast_dates = test_dates + future_dates
    all_forecast_dates_str = [d.strftime("%Y-%m-%d") for d in all_forecast_dates]

    # Prepare Exogenous Matrices (train + total forecast horizon)
    train_exog_mat: Optional[np.ndarray] = None
    total_exog_mat: Optional[np.ndarray] = None
    if has_exog:
        train_exog_df = train_df[clean_exog].copy()
        exog_means = train_exog_df.mean()
        train_exog_df = train_exog_df.fillna(exog_means)
        train_exog_mat = train_exog_df.values.astype(float)

        future_lookup = work[work["Data"] > cutoff_date][["Data"] + clean_exog].copy()
        f_exog_df = pd.DataFrame({"Data": all_forecast_dates}).merge(future_lookup, on="Data", how="left")
        for v in clean_exog:
            f_exog_df[v] = f_exog_df[v].fillna(exog_means[v])
        total_exog_mat = f_exog_df[clean_exog].values.astype(float)

    # Determine ARIMA order variations
    if manual_mode and arima_params:
        variations = [(
            int(arima_params.get("p", 1)),
            int(arima_params.get("d", 1)),
            int(arima_params.get("q", 1)),
            int(arima_params.get("P", 0)) if seasonal else 0,
            int(arima_params.get("D", 0)) if seasonal else 0,
            int(arima_params.get("Q", 0)) if seasonal else 0,
        )]
    else:
        base_order = _find_best_auto_arima_order(train_y, seasonal=seasonal, period=int(period))
        variations = _generate_order_variations(base_order, num_variations=max(1, int(num_models)), seasonal=seasonal, seed=seed)

    forecast_results: Dict[str, pd.DataFrame] = {}
    errors_list: List[Dict[str, Any]] = []
    deviation_results: List[Dict[str, Any]] = []
    per_month_pct_deviations: Dict[str, List[Tuple[str, float]]] = {}

    for idx, var_order in enumerate(variations, start=1):
        p, d_o, q, P, D_o, Q = var_order
        s_order = (P, D_o, Q, int(period)) if seasonal else (0, 0, 0, 0)
        order_str = f"({p},{d_o},{q})({P},{D_o},{Q})[{period}]" if seasonal else f"({p},{d_o},{q})"

        # 1. Univariate ARIMA
        m_name_uni = f"arima_uni_{idx}"
        try:
            mod_uni = SARIMAX(
                train_y,
                order=(p, d_o, q),
                seasonal_order=s_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_uni = mod_uni.fit(disp=False, maxiter=60)
            fcast_res = fit_uni.get_forecast(steps=h_total)
            mean_pred = np.asarray(fcast_res.predicted_mean, dtype=float)
            conf_int = np.asarray(fcast_res.conf_int(alpha=0.05), dtype=float)
            lo95 = conf_int[:, 0]
            hi95 = conf_int[:, 1]

            if h_test > 0:
                test_pred = mean_pred[:h_test]
                desvios = test_pred - test_y
                safe_ty = np.where(test_y == 0, 1e-10, test_y)
                desvios_pct = (desvios / safe_ty) * 100.0
                per_month_pct_deviations[m_name_uni] = [
                    (test_dates[k].strftime("%Y-%m-%d"), float(desvios_pct[k])) for k in range(h_test)
                ]
                deviation_results.append({
                    "model": m_name_uni,
                    "order": order_str,
                    "min_deviation": round(float(np.min(desvios)), 2),
                    "max_deviation": round(float(np.max(desvios)), 2),
                    "mean_deviation": round(float(np.mean(desvios)), 2),
                    "cum_deviation": round(float(np.sum(desvios)), 2),
                    "min_pct_deviation": round(float(np.min(desvios_pct)), 2),
                    "max_pct_deviation": round(float(np.max(desvios_pct)), 2),
                    "mean_pct_deviation": round(float(np.mean(desvios_pct)), 2),
                    "cum_pct_deviation": round(float(np.sum(desvios_pct)), 2),
                })
                errs = _calc_error_metrics(test_y, test_pred)
            else:
                # Fallback to in-sample fitted errors when n_teste == 0
                fitted_vals = np.asarray(fit_uni.fittedvalues, dtype=float)
                errs = _calc_error_metrics(train_y[1:], fitted_vals[1:])

            df_fcast_uni = pd.DataFrame({
                "Data": all_forecast_dates_str,
                "predict": np.round(mean_pred, 4),
                "Lo95": np.round(lo95, 4),
                "Hi95": np.round(hi95, 4),
                "Model": m_name_uni,
            })
            forecast_results[m_name_uni] = df_fcast_uni
            errors_list.append({
                "model": m_name_uni,
                "order": order_str,
                "type": "Univariado",
                "AIC": round(float(fit_uni.aic), 2) if math.isfinite(fit_uni.aic) else None,
                "MAPE": errs["MAPE"],
                "RMSE": errs["RMSE"],
                "MAE": errs["MAE"],
            })
        except Exception as e:
            print(f"Erro no modelo {m_name_uni}: {e}")

        # 2. Multivariate ARIMAX / SARIMAX (when exogenous variables are selected)
        if has_exog and train_exog_mat is not None and total_exog_mat is not None:
            m_name_multi = f"arima_multi_{idx}"
            try:
                mod_multi = SARIMAX(
                    train_y,
                    exog=train_exog_mat,
                    order=(p, d_o, q),
                    seasonal_order=s_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_multi = mod_multi.fit(disp=False, maxiter=60)
                fcast_res_m = fit_multi.get_forecast(steps=h_total, exog=total_exog_mat)
                mean_pred_m = np.asarray(fcast_res_m.predicted_mean, dtype=float)
                conf_int_m = np.asarray(fcast_res_m.conf_int(alpha=0.05), dtype=float)
                lo95_m = conf_int_m[:, 0]
                hi95_m = conf_int_m[:, 1]

                if h_test > 0:
                    test_pred_m = mean_pred_m[:h_test]
                    desvios_m = test_pred_m - test_y
                    safe_ty = np.where(test_y == 0, 1e-10, test_y)
                    desvios_pct_m = (desvios_m / safe_ty) * 100.0
                    per_month_pct_deviations[m_name_multi] = [
                        (test_dates[k].strftime("%Y-%m-%d"), float(desvios_pct_m[k])) for k in range(h_test)
                    ]
                    deviation_results.append({
                        "model": m_name_multi,
                        "order": order_str,
                        "min_deviation": round(float(np.min(desvios_m)), 2),
                        "max_deviation": round(float(np.max(desvios_m)), 2),
                        "mean_deviation": round(float(np.mean(desvios_m)), 2),
                        "cum_deviation": round(float(np.sum(desvios_m)), 2),
                        "min_pct_deviation": round(float(np.min(desvios_pct_m)), 2),
                        "max_pct_deviation": round(float(np.max(desvios_pct_m)), 2),
                        "mean_pct_deviation": round(float(np.mean(desvios_pct_m)), 2),
                        "cum_pct_deviation": round(float(np.sum(desvios_pct_m)), 2),
                    })
                    errs_m = _calc_error_metrics(test_y, test_pred_m)
                else:
                    fitted_vals_m = np.asarray(fit_multi.fittedvalues, dtype=float)
                    errs_m = _calc_error_metrics(train_y[1:], fitted_vals_m[1:])

                df_fcast_multi = pd.DataFrame({
                    "Data": all_forecast_dates_str,
                    "predict": np.round(mean_pred_m, 4),
                    "Lo95": np.round(lo95_m, 4),
                    "Hi95": np.round(hi95_m, 4),
                    "Model": m_name_multi,
                })
                forecast_results[m_name_multi] = df_fcast_multi
                errors_list.append({
                    "model": m_name_multi,
                    "order": order_str,
                    "type": "Multivariado (ARIMAX)",
                    "AIC": round(float(fit_multi.aic), 2) if math.isfinite(fit_multi.aic) else None,
                    "MAPE": errs_m["MAPE"],
                    "RMSE": errs_m["RMSE"],
                    "MAE": errs_m["MAE"],
                })
            except Exception as e:
                print(f"Erro no modelo {m_name_multi}: {e}")

    if not forecast_results:
        raise ValueError("Nenhum modelo ARIMA pôde ser ajustado com sucesso. Tente ajustar os parâmetros.")

    # Sort errors by RMSE
    errors_list.sort(key=lambda r: (r["RMSE"] is None, r["RMSE"] if r["RMSE"] is not None else float("inf")))

    # Build combined df_arima table (Realizado + Projections for each model)
    df_realizado = work[(work["Data"] >= dt_ini) & (work["Data"] <= dt_atual)][["Data", target_var]].copy()
    df_realizado["Data"] = df_realizado["Data"].dt.strftime("%Y-%m-%d")
    df_realizado = df_realizado.rename(columns={target_var: "Realizado"})

    all_dates_str = sorted(list(set(df_realizado["Data"].tolist() + all_forecast_dates_str)))
    df_base = pd.DataFrame({"Data": all_dates_str}).merge(df_realizado, on="Data", how="left")
    dt_atual_str = dt_atual.strftime("%Y-%m-%d")
    df_base["Tipo"] = np.where(df_base["Data"] <= dt_atual_str, "Realizado", "Projetado")

    for m_name, m_df in forecast_results.items():
        pred_map = dict(zip(m_df["Data"], m_df["predict"]))
        lo_map = dict(zip(m_df["Data"], m_df["Lo95"]))
        hi_map = dict(zip(m_df["Data"], m_df["Hi95"]))

        df_base[f"Projetado_{m_name}"] = [
            (r_val if (d <= dt_atual_str and pd.notna(r_val)) else pred_map.get(d, None))
            for d, r_val in zip(df_base["Data"], df_base["Realizado"])
        ]
        df_base[f"Lo95_{m_name}"] = [lo_map.get(d, None) for d in df_base["Data"]]
        df_base[f"Hi95_{m_name}"] = [hi_map.get(d, None) for d in df_base["Data"]]

    # Build chart payload
    chart_forecasts = []
    for err_row in errors_list:
        m_name = err_row["model"]
        m_df = forecast_results[m_name]
        chart_forecasts.append({
            "model": m_name,
            "order": err_row["order"],
            "type": err_row["type"],
            "x": m_df["Data"].tolist(),
            "y": m_df["predict"].tolist(),
            "lo95": m_df["Lo95"].tolist(),
            "hi95": m_df["Hi95"].tolist(),
        })

    dashboard_rec = gerar_dashboard_recomendacao_arima(
        per_month_pct_deviations=per_month_pct_deviations,
        errors_list=errors_list,
        peso_recente=peso_recente,
        peso_desvio_max=peso_desvio_max,
        periodo_maximo=max(1, n_teste_int),
    )

    exec_time = round(time.time() - start_time, 2)
    return {
        "target_var": target_var,
        "exog_vars": clean_exog,
        "collinear_warning": collinear_warning,
        "execution_time_sec": exec_time,
        "periods": {
            "train_start": dt_ini.strftime("%Y-%m-%d"),
            "train_end": cutoff_date.strftime("%Y-%m-%d"),
            "test_start": (cutoff_date + pd.DateOffset(months=1)).strftime("%Y-%m-%d") if n_teste_int > 0 else None,
            "test_end": dt_atual_str,
            "forecast_start": (dt_atual + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
            "forecast_end": dt_end_pred.strftime("%Y-%m-%d"),
        },
        "real_series": {
            "x": df_realizado["Data"].tolist(),
            "y": [None if pd.isna(v) else float(v) for v in df_realizado["Realizado"].tolist()],
        },
        "forecasts": chart_forecasts,
        "errors": errors_list,
        "deviations": deviation_results,
        "dashboard": dashboard_rec,
        "per_month_pct_deviations": per_month_pct_deviations,
        "df_export": df_base,
    }
