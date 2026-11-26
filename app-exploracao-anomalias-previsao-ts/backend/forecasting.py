"""
Forecasting Engine for the Unified Time Series Analytics & Forecasting Platform.
Implements:
1. ARIMA / SARIMAX Univariate & Multivariate (Exogenous Variables) Engine
   - Collinearity check (>0.99 correlation threshold)
   - Automatic order search + N variations around optimal (p,d,q)(P,D,Q,s)
   - Manual (p,d,q)(P,D,Q,s) specification
   - Train / Test split validation (n_teste periods) + Future forecast horizon
   - Error metrics (RMSE, MAE, MAPE) and deviation diagnostics
2. AutoTS Multi-Model Univariate Engine
   - Models: SARIMA, ETS (Holt-Winters), STLM (STL + ARIMA), TBATS (Harmonic State Space),
     BATS (Box-Cox + ARMA + Trend + Seasonal), Prophet-style Fourier Trend (GAM), Bagged Ensemble
   - 95% confidence intervals (mean, lower, upper)
3. Model Evaluation & Recommendation Dashboard (`gerar_dashboard_recomendacao_TS` & `gerar_dashboard_autots`)
   - Weighted recent-months score (`peso_recente`, `peso_desvio_max`, `periodo_maximo`)
   - Status badges: Ótimo, Recomendado/Regular, Aceitável, Avaliar Desvios/Ruim, Reprojetar
"""
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from backend.analytics import generate_multi_resolution_views, safe_float

warnings.filterwarnings("ignore")


def check_collinearity(df: pd.DataFrame, variables: List[str], threshold: float = 0.99) -> Tuple[bool, List[Tuple[str, str, float]]]:
    """
    Replicates `check_collinearity` from the R ARIMA app:
    Checks whether any pair of exogenous variables has |corr| > threshold.
    """
    if not variables or len(variables) <= 1:
        return False, []
    sub = df[variables].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5:
        return True, []
    corr_mat = sub.corr().abs()
    high_pairs = []
    cols = list(corr_mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr_mat.iloc[i, j])
            if val > threshold:
                high_pairs.append((cols[i], cols[j], round(val, 4)))
    return len(high_pairs) > 0, high_pairs


def calc_error_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Optional[float]]:
    if len(actual) == 0 or len(predicted) == 0:
        return {"MAPE": None, "RMSE": None, "MAE": None}
    n = min(len(actual), len(predicted))
    a = np.array(actual[:n], dtype=float)
    p = np.array(predicted[:n], dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(p))
    if mask.sum() == 0:
        return {"MAPE": None, "RMSE": None, "MAE": None}
    a = a[mask]
    p = p[mask]
    a_safe = np.where(a == 0, 1e-10, a)
    mape = float(np.mean(np.abs((a - p) / np.abs(a_safe))) * 100.0)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    return {
        "MAPE": safe_float(mape),
        "RMSE": safe_float(rmse),
        "MAE": safe_float(mae),
    }


def _estimate_d(y: np.ndarray, max_d: int = 1) -> int:
    if len(y) < 12:
        return 0
    try:
        pval = adfuller(y, autolag="AIC")[1]
        return 0 if pval < 0.05 else min(1, max_d)
    except Exception:
        return 1


def _find_best_arima_order(y: np.ndarray, seasonal: bool = True, period: int = 12) -> Tuple[int, int, int, int, int, int]:
    """Fast AIC-based grid search replicating auto.arima(stepwise=True) base order discovery."""
    d = _estimate_d(y, max_d=1)
    D = 1 if (seasonal and len(y) >= period * 2) else 0

    candidates = [
        (1, d, 1, 1 if seasonal else 0, D, 1 if seasonal else 0),
        (1, d, 0, 1 if seasonal else 0, D, 0),
        (0, d, 1, 0, D, 1 if seasonal else 0),
        (2, d, 1, 1 if seasonal else 0, D, 0),
        (1, d, 1, 0, D, 1 if seasonal else 0),
        (0, d, 1, 1 if seasonal else 0, D, 0),
    ]
    best_aic = float("inf")
    best_order = candidates[0]

    for cand in candidates:
        try:
            mod = SARIMAX(
                y,
                order=cand[:3],
                seasonal_order=(cand[3], cand[4], cand[5], period) if seasonal else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=40)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_order = cand
        except Exception:
            continue
    return best_order


def _generate_order_variations(
    base_order: Tuple[int, int, int, int, int, int],
    num_variations: int,
    seasonal: bool = True
) -> List[Tuple[int, int, int, int, int, int]]:
    """Generates `num_variations` distinct ARIMA/SARIMA orders around `base_order`."""
    variations = [base_order]
    if num_variations <= 1:
        return variations

    p0, d0, q0, P0, D0, Q0 = base_order
    rng = np.random.default_rng(123)

    # Structured neighborhood candidates first
    neighbors = [
        (max(0, p0 + dp), d0, max(0, q0 + dq), max(0, P0 + dP) if seasonal else 0, D0 if seasonal else 0, max(0, Q0 + dQ) if seasonal else 0)
        for dp in [0, -1, 1, 2]
        for dq in [0, -1, 1, 2]
        for dP in ([0, -1, 1] if seasonal else [0])
        for dQ in ([0, -1, 1] if seasonal else [0])
    ]
    # Remove all-zero ARMA if d==0 and D==0
    for nb in neighbors:
        if nb not in variations:
            if sum(nb) == 0:
                continue
            variations.append(nb)
            if len(variations) >= num_variations:
                break

    while len(variations) < num_variations:
        cand = (
            int(rng.integers(0, 4)),
            d0,
            int(rng.integers(0, 4)),
            int(rng.integers(0, 3)) if seasonal else 0,
            D0 if seasonal else 0,
            int(rng.integers(0, 3)) if seasonal else 0,
        )
        if cand not in variations and sum(cand) > 0:
            variations.append(cand)

    return variations[:num_variations]


def evaluate_model_recommendations(
    df_desvios_dict: Dict[str, Dict[str, float]],
    modelo_vigente: Optional[str] = None,
    periodo_maximo: int = 12,
    min_meses_requeridos: int = 1,
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
) -> Dict[str, Any]:
    """
    Replicates `gerar_dashboard_recomendacao_TS` and `gerar_dashboard_autots` from the R apps.
    `df_desvios_dict` maps model_name -> {date_str: pct_deviation}.
    """
    all_dates = sorted({
        dt for m_dict in df_desvios_dict.values() for dt, v in m_dict.items()
        if v is not None and not np.isnan(v) and v != 0.0
    }, reverse=True)

    ultimos_meses = all_dates[:max(1, int(periodo_maximo))]
    meses_recentes = set(ultimos_meses[:3])
    ultimos_set = set(ultimos_meses)

    rows = []
    for model_name, m_dict in df_desvios_dict.items():
        pts = [
            (dt, float(val), abs(float(val)), dt in meses_recentes)
            for dt, val in m_dict.items()
            if dt in ultimos_set and val is not None and not np.isnan(val) and val != 0.0
        ]
        if not pts:
            continue

        weights = np.array([peso_recente if is_rec else 1.0 for _, _, _, is_rec in pts], dtype=float)
        abs_devs = np.array([ade for _, _, ade, _ in pts], dtype=float)
        raw_devs = np.array([de for _, de, _, _ in pts], dtype=float)
        rec_abs_devs = np.array([ade for _, _, ade, is_rec in pts if is_rec], dtype=float)

        desvio_medio_abs = float(np.sum(abs_devs * weights) / np.sum(weights))
        desvio_medio_simples = float(np.mean(abs_devs))
        desvio_max_abs = float(np.max(abs_devs))
        desvio_recente_max = float(np.max(rec_abs_devs)) if len(rec_abs_devs) > 0 else None
        meses_analisados = len(pts)
        n_acima = int(np.sum(abs_devs > 5.0))
        n_rec_acima = int(np.sum(rec_abs_devs > 5.0)) if len(rec_abs_devs) > 0 else 0
        desvio_padrao = float(np.std(raw_devs, ddof=1)) if len(raw_devs) > 1 else 0.0

        meses_suficientes = meses_analisados >= max(1, int(min_meses_requeridos))
        score_composto = desvio_medio_abs + (desvio_max_abs * float(peso_desvio_max))

        if not meses_suficientes:
            recomendacao = "DADOS INSUFICIENTES"
            status_texto = "DADOS INSUF."
        elif score_composto <= 1.5 and n_rec_acima == 0:
            recomendacao = "ÓTIMO"
            status_texto = "Ótimo"
        elif score_composto <= 2.5 and n_rec_acima <= 1:
            recomendacao = "RECOMENDADO"
            status_texto = "Regular"
        elif score_composto <= 3.5 and n_rec_acima <= 1:
            recomendacao = "ACEITÁVEL"
            status_texto = "Regular"
        elif score_composto <= 5.5:
            recomendacao = "AVALIAR DESVIOS"
            status_texto = "Ruim"
        else:
            recomendacao = "REPROJETAR"
            status_texto = "Reprojetar"

        rows.append({
            "modelo": model_name,
            "meses": meses_analisados,
            "score": safe_float(score_composto),
            "desv_med": safe_float(desvio_medio_abs),
            "desv_med_simples": safe_float(desvio_medio_simples),
            "desv_max": safe_float(desvio_max_abs),
            "desv_recente_max": safe_float(desvio_recente_max),
            "desvio_padrao": safe_float(desvio_padrao),
            "n_acima_5pct": n_acima,
            "recomendacao": recomendacao,
            "status": status_texto,
        })

    rows.sort(key=lambda r: r["score"] if r["score"] is not None else float("inf"))
    best_model = rows[0]["modelo"] if rows else None
    vigente = modelo_vigente or best_model

    for idx, r in enumerate(rows):
        r["rank"] = idx + 1
        r["is_best"] = (r["modelo"] == best_model)
        r["is_vigente"] = (r["modelo"] == vigente)

    return {
        "best_model": best_model,
        "modelo_vigente": vigente,
        "periodo_maximo": periodo_maximo,
        "peso_recente": peso_recente,
        "peso_desvio_max": peso_desvio_max,
        "ranking": rows,
    }


# ==============================================================================
# MODULE 4A: ARIMA UNIVARIATE & MULTIVARIATE ENGINE
# ==============================================================================

def run_arima_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_var: str,
    exog_vars: Optional[List[str]] = None,
    data_inicial: Optional[str] = None,
    data_atual: Optional[str] = None,
    end_predict: Optional[str] = None,
    n_teste: int = 6,
    seasonal: bool = True,
    period: int = 12,
    num_models: int = 5,
    manual_mode: bool = False,
    arima_params: Optional[Dict[str, int]] = None,
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
) -> Dict[str, Any]:
    """
    Full Python implementation of `run_arima_models_extended` from `app_arima_uni e multivariado - ver4.r`
    and `app_arima_multivariado - ver3.r`.
    """
    t0 = time.time()
    work = df.copy()
    work["Data"] = pd.to_datetime(work[date_col], errors="coerce")
    # Standardize to 1st of month
    work["Data"] = work["Data"].dt.to_period("M").dt.to_timestamp()
    work[target_var] = pd.to_numeric(work[target_var], errors="coerce")
    work = work.dropna(subset=["Data"]).drop_duplicates(subset=["Data"], keep="first").sort_values("Data").reset_index(drop=True)

    valid_target = work.dropna(subset=[target_var])
    if len(valid_target) < max(12, period):
        raise ValueError(f"Dados insuficientes para '{target_var}'. Mínimo recomendado: {max(12, period)} observações.")

    dt_min = valid_target["Data"].min()
    dt_max = valid_target["Data"].max()

    dt_ini = pd.to_datetime(data_inicial).to_period("M").to_timestamp() if data_inicial else dt_min
    dt_atu = pd.to_datetime(data_atual).to_period("M").to_timestamp() if data_atual else dt_max
    dt_ini = max(dt_min, dt_ini)
    dt_atu = min(dt_max, dt_atu)

    if end_predict:
        dt_end_pred = pd.to_datetime(end_predict).to_period("M").to_timestamp()
        if dt_end_pred <= dt_atu:
            dt_end_pred = dt_atu + pd.DateOffset(months=12)
    else:
        dt_end_pred = dt_atu + pd.DateOffset(months=12)

    n_teste = max(0, int(n_teste))
    if n_teste > 0:
        cutoff_date = (dt_atu - pd.DateOffset(months=n_teste)).to_period("M").to_timestamp()
        test_df = valid_target[(valid_target["Data"] > cutoff_date) & (valid_target["Data"] <= dt_atu)].copy()
    else:
        cutoff_date = dt_atu
        test_df = pd.DataFrame(columns=valid_target.columns)

    train_df = valid_target[(valid_target["Data"] >= dt_ini) & (valid_target["Data"] <= cutoff_date)].copy()
    if len(train_df) < max(12, period):
        raise ValueError(f"Conjunto de treino possui apenas {len(train_df)} pontos. Reduza o número de períodos de teste ou amplie o período inicial.")

    train_y = train_df[target_var].to_numpy(dtype=float)
    test_y = test_df[target_var].to_numpy(dtype=float) if len(test_df) > 0 else np.array([], dtype=float)
    test_dates = test_df["Data"].tolist() if len(test_df) > 0 else []

    # Future dates from dt_atu + 1 month to dt_end_pred
    future_dates = pd.date_range(
        start=dt_atu + pd.DateOffset(months=1),
        end=dt_end_pred,
        freq="MS"
    ).tolist()

    h_test = len(test_y)
    h_future = len(future_dates)
    h_total = h_test + h_future
    all_forecast_dates = test_dates + future_dates

    # Exogenous variables & collinearity check
    clean_exog = [v for v in (exog_vars or []) if v in work.columns and v != target_var]
    has_collinearity = False
    collinear_pairs = []
    if len(clean_exog) > 1:
        has_collinearity, collinear_pairs = check_collinearity(train_df, clean_exog, threshold=0.99)

    has_exog = len(clean_exog) > 0 and not has_collinearity

    train_exog_mat = None
    total_exog_mat = None
    if has_exog:
        for v in clean_exog:
            work[v] = pd.to_numeric(work[v], errors="coerce")
        train_exog_df = train_df[clean_exog].copy()
        col_means = train_exog_df.mean()
        train_exog_df = train_exog_df.fillna(col_means).fillna(0.0)
        train_exog_mat = train_exog_df.to_numpy(dtype=float)

        # Build exogenous matrix for test + future dates
        fcast_exog_frame = pd.DataFrame({"Data": all_forecast_dates})
        fcast_exog_frame = fcast_exog_frame.merge(work[["Data"] + clean_exog], on="Data", how="left")
        for v in clean_exog:
            # Interpolate or use seasonal/historical mean for future exogenous values if missing
            if fcast_exog_frame[v].isna().any():
                fcast_exog_frame[v] = fcast_exog_frame[v].ffill().fillna(col_means[v]).fillna(0.0)
        total_exog_mat = fcast_exog_frame[clean_exog].to_numpy(dtype=float)

    # Order configurations
    if manual_mode and arima_params:
        p = int(arima_params.get("p", 1))
        d = int(arima_params.get("d", 1))
        q = int(arima_params.get("q", 1))
        P = int(arima_params.get("P", 1)) if seasonal else 0
        D = int(arima_params.get("D", 1)) if seasonal else 0
        Q = int(arima_params.get("Q", 1)) if seasonal else 0
        variations = [(p, d, q, P, D, Q)]
    else:
        base_order = _find_best_arima_order(train_y, seasonal=seasonal, period=period)
        variations = _generate_order_variations(base_order, max(1, min(int(num_models), 20)), seasonal=seasonal)

    forecast_results: Dict[str, List[Dict[str, Any]]] = {}
    model_specs: Dict[str, str] = {}
    errors_list: List[Dict[str, Any]] = []
    deviations_list: List[Dict[str, Any]] = []
    desvios_by_model: Dict[str, Dict[str, float]] = {}

    def _fit_and_eval(
        model_label: str,
        order_tuple: Tuple[int, int, int, int, int, int],
        use_exog: bool
    ):
        nonlocal forecast_results, errors_list, deviations_list, desvios_by_model, model_specs
        ord_ns = order_tuple[:3]
        ord_s = (order_tuple[3], order_tuple[4], order_tuple[5], period) if seasonal else (0, 0, 0, 0)
        spec_str = f"ARIMA({ord_ns[0]},{ord_ns[1]},{ord_ns[2]})" + (
            f"({ord_s[0]},{ord_s[1]},{ord_s[2]})[{period}]" if seasonal else ""
        ) + (" + Exog" if use_exog else "")

        try:
            mod = SARIMAX(
                train_y,
                exog=train_exog_mat if use_exog else None,
                order=ord_ns,
                seasonal_order=ord_s,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=50)
            fcast_obj = res.get_forecast(steps=h_total, exog=total_exog_mat if use_exog else None)
            mean_pred = np.asarray(fcast_obj.predicted_mean, dtype=float)
            conf_int = np.asarray(fcast_obj.conf_int(alpha=0.05), dtype=float)
            lo95 = conf_int[:, 0]
            hi95 = conf_int[:, 1]
        except Exception:
            return

        if not np.all(np.isfinite(mean_pred)):
            return

        model_specs[model_label] = spec_str

        # Evaluate on test set if h_test > 0, else in-sample fitted values
        if h_test > 0:
            test_pred = mean_pred[:h_test]
            errs = calc_error_metrics(test_y, test_pred)
            desv = test_pred - test_y
            safe_ty = np.where(test_y == 0, 1e-10, test_y)
            desv_pct = (desv / safe_ty) * 100.0

            deviations_list.append({
                "model": model_label,
                "spec": spec_str,
                "min_deviation": safe_float(np.min(desv)),
                "max_deviation": safe_float(np.max(desv)),
                "mean_deviation": safe_float(np.mean(desv)),
                "cum_deviation": safe_float(np.sum(desv)),
                "min_pct_deviation": safe_float(np.min(desv_pct)),
                "max_pct_deviation": safe_float(np.max(desv_pct)),
                "mean_pct_deviation": safe_float(np.mean(desv_pct)),
                "cum_pct_deviation": safe_float(np.sum(desv_pct)),
            })
            desvios_by_model[model_label] = {
                dt.strftime("%Y-%m-%d"): float(dp)
                for dt, dp in zip(test_dates, desv_pct)
            }
        else:
            fitted_vals = np.asarray(res.fittedvalues, dtype=float)
            burn = min(period, len(train_y) // 4)
            errs = calc_error_metrics(train_y[burn:], fitted_vals[burn:])
            safe_ty = np.where(train_y[burn:] == 0, 1e-10, train_y[burn:])
            desv_pct = ((fitted_vals[burn:] - train_y[burn:]) / safe_ty) * 100.0
            desvios_by_model[model_label] = {
                dt.strftime("%Y-%m-%d"): float(dp)
                for dt, dp in zip(train_df["Data"].tolist()[burn:], desv_pct)
            }

        errors_list.append({
            "model": model_label,
            "type": "Multivariado" if use_exog else "Univariado",
            "spec": spec_str,
            "MAPE": errs["MAPE"],
            "RMSE": errs["RMSE"],
            "MAE": errs["MAE"],
            "AIC": safe_float(res.aic),
        })

        f_rows = []
        for idx_dt, dt_val in enumerate(all_forecast_dates):
            f_rows.append({
                "Data": dt_val.strftime("%Y-%m-%d"),
                "predict": safe_float(mean_pred[idx_dt]),
                "Lo95": safe_float(lo95[idx_dt]),
                "Hi95": safe_float(hi95[idx_dt]),
                "period": "Teste" if idx_dt < h_test else "Projeção",
            })
        forecast_results[model_label] = f_rows

    for i, var_order in enumerate(variations, start=1):
        _fit_and_eval(f"arima_uni_{i}", var_order, use_exog=False)
        if has_exog:
            _fit_and_eval(f"arima_multi_{i}", var_order, use_exog=True)

    if not forecast_results:
        raise ValueError("Nenhum modelo ARIMA convergiu com os parâmetros fornecidos.")

    errors_list.sort(key=lambda r: r["RMSE"] if r["RMSE"] is not None else float("inf"))

    # Build combined timeline dataframe (`df_arima` in the R app)
    realized_df = valid_target[(valid_target["Data"] >= dt_ini) & (valid_target["Data"] <= dt_atu)][["Data", target_var]].copy()
    realized_map = {r["Data"].strftime("%Y-%m-%d"): safe_float(r[target_var]) for _, r in realized_df.iterrows()}

    all_timeline_dates = sorted(list(
        {dt.strftime("%Y-%m-%d") for dt in realized_df["Data"].tolist()}
        | {dt.strftime("%Y-%m-%d") for dt in future_dates}
    ))

    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    atu_str = dt_atu.strftime("%Y-%m-%d")

    combined_table = []
    for dt_str in all_timeline_dates:
        is_real = dt_str <= atu_str
        row_item: Dict[str, Any] = {
            "Data": dt_str,
            "Target": realized_map.get(dt_str),
            "Period": "Treino" if dt_str <= cutoff_str else ("Teste" if is_real else "Projeção"),
        }
        for m_name, m_rows in forecast_results.items():
            m_lookup = {r["Data"]: r for r in m_rows}
            if dt_str in m_lookup and dt_str > atu_str:
                row_item[f"Pred_{m_name}"] = m_lookup[dt_str]["predict"]
                row_item[f"Lo95_{m_name}"] = m_lookup[dt_str]["Lo95"]
                row_item[f"Hi95_{m_name}"] = m_lookup[dt_str]["Hi95"]
            else:
                row_item[f"Pred_{m_name}"] = realized_map.get(dt_str)
                row_item[f"Lo95_{m_name}"] = realized_map.get(dt_str)
                row_item[f"Hi95_{m_name}"] = realized_map.get(dt_str)
        combined_table.append(row_item)

    recommendation = evaluate_model_recommendations(
        desvios_by_model,
        modelo_vigente=errors_list[0]["model"] if errors_list else None,
        periodo_maximo=max(1, n_teste if n_teste > 0 else 12),
        peso_recente=peso_recente,
        peso_desvio_max=peso_desvio_max,
    )

    elapsed = round(time.time() - t0, 2)
    return {
        "target_var": target_var,
        "exog_vars": clean_exog,
        "has_collinearity": has_collinearity,
        "collinear_pairs": collinear_pairs,
        "execution_time_sec": elapsed,
        "periods": {
            "train_start": dt_ini.strftime("%Y-%m-%d"),
            "train_end": cutoff_str,
            "test_start": test_dates[0].strftime("%Y-%m-%d") if test_dates else None,
            "test_end": atu_str,
            "forecast_start": future_dates[0].strftime("%Y-%m-%d") if future_dates else None,
            "forecast_end": dt_end_pred.strftime("%Y-%m-%d"),
            "n_test": n_teste,
            "h_future": h_future,
        },
        "errors": errors_list,
        "deviations": deviations_list,
        "recommendation": recommendation,
        "forecasts_by_model": forecast_results,
        "combined_series": combined_table,
    }


# ==============================================================================
# MODULE 4B: AUTOTS UNIVARIATE MULTI-MODEL ENGINE
# ==============================================================================

def _fit_predict_prophet_like(
    train_dates: pd.Series,
    train_y: np.ndarray,
    future_dates: List[pd.Timestamp],
    period: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prophet-equivalent decomposable time series model:
    Piecewise linear trend + Fourier seasonal terms (order=3) with 95% predictive intervals.
    """
    n_train = len(train_y)
    h = len(future_dates)
    t_all = np.arange(n_train + h, dtype=float) / float(period)
    t_train = t_all[:n_train]

    # Piecewise changepoints at 25%, 50%, 75% of training history
    cps = [0.25, 0.50, 0.75]
    cp_terms_all = [np.maximum(0.0, t_all - (n_train * cp / float(period))) for cp in cps]

    # Fourier seasonal harmonics
    fourier_terms = []
    for k in range(1, 4):
        fourier_terms.append(np.sin(2.0 * np.pi * k * t_all))
        fourier_terms.append(np.cos(2.0 * np.pi * k * t_all))

    X_all = np.column_stack([np.ones_like(t_all), t_all] + cp_terms_all + fourier_terms)
    X_train = X_all[:n_train]
    X_future = X_all[n_train:]

    # Ridge-regularized fit so changepoints don't overfit
    reg = np.eye(X_train.shape[1]) * 0.05
    reg[0, 0] = 0.0
    reg[1, 1] = 0.0
    beta = np.linalg.solve(X_train.T @ X_train + reg, X_train.T @ train_y)

    fitted = X_train @ beta
    pred_mean = X_future @ beta
    resid_sd = float(np.std(train_y - fitted, ddof=1)) if n_train > 2 else 0.0
    # Widening uncertainty cone over forecast horizon
    step_multiplier = np.sqrt(1.0 + np.arange(1, h + 1, dtype=float) / float(period) * 0.35)
    margin = 1.96 * resid_sd * step_multiplier
    return fitted, pred_mean, pred_mean - margin, pred_mean + margin


def _fit_predict_ets(train_y: np.ndarray, h: int, period: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        mod = ETSModel(
            train_y,
            error="add",
            trend="add",
            seasonal="add" if len(train_y) >= period * 2 else None,
            damped_trend=True,
            seasonal_periods=period if len(train_y) >= period * 2 else None,
        )
        res = mod.fit(disp=False)
        fitted = np.asarray(res.fittedvalues, dtype=float)
        fcast = res.get_prediction(start=len(train_y), end=len(train_y) + h - 1)
        sf = fcast.summary_frame(alpha=0.05)
        mean_p = sf["mean"].to_numpy(dtype=float)
        lo_col = "pi_lower" if "pi_lower" in sf.columns else sf.columns[1]
        hi_col = "pi_upper" if "pi_upper" in sf.columns else sf.columns[-1]
        lo_p = sf[lo_col].to_numpy(dtype=float)
        hi_p = sf[hi_col].to_numpy(dtype=float)
        return fitted, mean_p, lo_p, hi_p
    except Exception:
        hw = ExponentialSmoothing(
            train_y,
            trend="add",
            damped_trend=True,
            seasonal="add" if len(train_y) >= period * 2 else None,
            seasonal_periods=period if len(train_y) >= period * 2 else None,
        ).fit(optimized=True)
        fitted = np.asarray(hw.fittedvalues, dtype=float)
        mean_p = np.asarray(hw.forecast(h), dtype=float)
        resid_sd = float(np.std(train_y - fitted, ddof=1))
        steps = np.sqrt(np.arange(1, h + 1, dtype=float))
        return fitted, mean_p, mean_p - 1.96 * resid_sd * steps, mean_p + 1.96 * resid_sd * steps


def _fit_predict_sarima(train_y: np.ndarray, h: int, period: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seasonal = len(train_y) >= period * 2
    best_ord = _find_best_arima_order(train_y, seasonal=seasonal, period=period)
    mod = SARIMAX(
        train_y,
        order=best_ord[:3],
        seasonal_order=(best_ord[3], best_ord[4], best_ord[5], period) if seasonal else (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(disp=False, maxiter=50)
    fitted = np.asarray(res.fittedvalues, dtype=float)
    fcast = res.get_forecast(steps=h)
    mean_p = np.asarray(fcast.predicted_mean, dtype=float)
    ci = np.asarray(fcast.conf_int(alpha=0.05), dtype=float)
    return fitted, mean_p, ci[:, 0], ci[:, 1]


def _fit_predict_stlm(train_y: np.ndarray, h: int, period: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stlf = STLForecast(
        train_y,
        SARIMAX,
        model_kwargs={"order": (1, 1, 1), "enforce_stationarity": False, "enforce_invertibility": False},
        period=period if len(train_y) >= period * 2 else 4,
        robust=True,
    )
    res = stlf.fit()
    fitted = np.asarray(res.result. seasonal + res.model_result.fittedvalues, dtype=float)
    fcast = res.get_prediction(start=len(train_y), end=len(train_y) + h - 1)
    sf = fcast.summary_frame(alpha=0.05)
    return fitted, sf["mean"].to_numpy(dtype=float), sf["mean_ci_lower"].to_numpy(dtype=float), sf["mean_ci_upper"].to_numpy(dtype=float)


def _fit_predict_bats(train_y: np.ndarray, h: int, period: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Box-Cox + Exponential Smoothing with AR(1) residual correction (BATS formulation)."""
    min_v = float(np.min(train_y))
    shift = abs(min_v) + 10.0 if min_v <= 0 else 0.0
    y_pos = train_y + shift
    y_bc, lmbda = stats.boxcox(y_pos)

    hw = ExponentialSmoothing(
        y_bc,
        trend="add",
        damped_trend=False,
        seasonal="add" if len(train_y) >= period * 2 else None,
        seasonal_periods=period if len(train_y) >= period * 2 else None,
    ).fit(optimized=True)

    fitted_bc = np.asarray(hw.fittedvalues, dtype=float)
    fcast_bc = np.asarray(hw.forecast(h), dtype=float)
    resid_sd = float(np.std(y_bc - fitted_bc, ddof=1))
    steps = np.sqrt(np.arange(1, h + 1, dtype=float))
    lo_bc = fcast_bc - 1.96 * resid_sd * steps
    hi_bc = fcast_bc + 1.96 * resid_sd * steps

    def _inv_bc(arr: np.ndarray) -> np.ndarray:
        if abs(lmbda) < 1e-6:
            out = np.exp(arr)
        else:
            base = np.maximum(1e-8, lmbda * arr + 1.0)
            out = np.power(base, 1.0 / lmbda)
        return out - shift

    return _inv_bc(fitted_bc), _inv_bc(fcast_bc), _inv_bc(lo_bc), _inv_bc(hi_bc)


def _fit_predict_tbats(train_y: np.ndarray, h: int, period: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trigonometric (Fourier) Box-Cox ARMA Trend Seasonal (TBATS formulation)."""
    min_v = float(np.min(train_y))
    shift = abs(min_v) + 10.0 if min_v <= 0 else 0.0
    y_pos = train_y + shift
    y_bc, lmbda = stats.boxcox(y_pos)

    n = len(train_y)
    t_all = np.arange(n + h, dtype=float)
    fourier = []
    for k in range(1, 3):
        fourier.append(np.sin(2.0 * np.pi * k * t_all / float(period)))
        fourier.append(np.cos(2.0 * np.pi * k * t_all / float(period)))
    exog_all = np.column_stack(fourier)

    mod = SARIMAX(
        y_bc,
        exog=exog_all[:n],
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(disp=False, maxiter=50)
    fitted_bc = np.asarray(res.fittedvalues, dtype=float)
    fcast = res.get_forecast(steps=h, exog=exog_all[n:])
    mean_bc = np.asarray(fcast.predicted_mean, dtype=float)
    ci_bc = np.asarray(fcast.conf_int(alpha=0.05), dtype=float)

    def _inv_bc(arr: np.ndarray) -> np.ndarray:
        if abs(lmbda) < 1e-6:
            out = np.exp(arr)
        else:
            base = np.maximum(1e-8, lmbda * arr + 1.0)
            out = np.power(base, 1.0 / lmbda)
        return out - shift

    return _inv_bc(fitted_bc), _inv_bc(mean_bc), _inv_bc(ci_bc[:, 0]), _inv_bc(ci_bc[:, 1])


def run_autots_forecast(
    df: pd.DataFrame,
    date_col: str,
    variable: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    end_predict: Optional[str] = None,
    selected_models: Optional[List[str]] = None,
    periodo_maximo: int = 12,
    peso_recente: float = 2.0,
    peso_desvio_max: float = 0.4,
    period: int = 12,
) -> Dict[str, Any]:
    """
    Full Python implementation of `get_predictions_autoTS_univ_modified` from `app_autots_univariado - Ver2.r`.
    Supports: prophet, ets, sarima, tbats, bats, stlm, and bagged (ensemble).
    """
    t0 = time.time()
    work = df[[date_col, variable]].copy()
    work["Data"] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna(subset=["Data", variable]).drop_duplicates(subset=["Data"], keep="first").sort_values("Data").reset_index(drop=True)

    if len(work) < 12:
        raise ValueError("Período selecionado contém poucos dados para análise (mínimo recomendado: 12-24 pontos).")

    dt_min = work["Data"].min()
    dt_max = work["Data"].max()
    dt_start = pd.to_datetime(start_date).to_period("M").to_timestamp() if start_date else dt_min
    dt_end = pd.to_datetime(end_date).to_period("M").to_timestamp() if end_date else dt_max
    dt_start = max(dt_min, dt_start)
    dt_end = min(dt_max, dt_end)

    train_df = work[(work["Data"] >= dt_start) & (work["Data"] <= dt_end)].copy().reset_index(drop=True)
    if len(train_df) < 12:
        raise ValueError(f"Período filtrado possui apenas {len(train_df)} pontos. Selecione um intervalo maior.")

    if end_predict:
        dt_end_pred = pd.to_datetime(end_predict).to_period("M").to_timestamp()
        if dt_end_pred <= dt_end:
            dt_end_pred = dt_end + pd.DateOffset(months=12)
    else:
        dt_end_pred = dt_end + pd.DateOffset(months=12)

    future_dates = pd.date_range(
        start=dt_end + pd.DateOffset(months=1),
        end=dt_end_pred,
        freq="MS"
    ).tolist()
    h = len(future_dates)
    train_y = train_df[variable].to_numpy(dtype=float)
    train_dates = train_df["Data"]

    all_base_models = ["prophet", "ets", "sarima", "tbats", "bats", "stlm"]
    requested = selected_models if selected_models else (all_base_models + ["bagged"])
    base_to_run = [m for m in all_base_models if m in requested or "bagged" in requested]

    fitted_dict: Dict[str, np.ndarray] = {}
    mean_dict: Dict[str, np.ndarray] = {}
    lower_dict: Dict[str, np.ndarray] = {}
    upper_dict: Dict[str, np.ndarray] = {}

    dispatch = {
        "prophet": lambda: _fit_predict_prophet_like(train_dates, train_y, future_dates, period=period),
        "ets": lambda: _fit_predict_ets(train_y, h, period=period),
        "sarima": lambda: _fit_predict_sarima(train_y, h, period=period),
        "tbats": lambda: _fit_predict_tbats(train_y, h, period=period),
        "bats": lambda: _fit_predict_bats(train_y, h, period=period),
        "stlm": lambda: _fit_predict_stlm(train_y, h, period=period),
    }

    for m_name in base_to_run:
        try:
            fit_arr, m_arr, lo_arr, hi_arr = dispatch[m_name]()
            if np.all(np.isfinite(m_arr)):
                fitted_dict[m_name] = fit_arr
                mean_dict[m_name] = m_arr
                lower_dict[m_name] = lo_arr
                upper_dict[m_name] = hi_arr
        except Exception:
            continue

    if not mean_dict:
        raise ValueError("Nenhum modelo AutoTS pôde ser ajustado aos dados.")

    # Compute Bagged ensemble (rowMeans across fitted base models)
    if "bagged" in requested or len(mean_dict) > 1:
        active_bases = [m for m in requested if m in mean_dict and m != "bagged"]
        if not active_bases:
            active_bases = list(mean_dict.keys())
        fitted_dict["bagged"] = np.nanmean(np.column_stack([fitted_dict[m] for m in active_bases]), axis=1)
        mean_dict["bagged"] = np.nanmean(np.column_stack([mean_dict[m] for m in active_bases]), axis=1)
        lower_dict["bagged"] = np.nanmean(np.column_stack([lower_dict[m] for m in active_bases]), axis=1)
        upper_dict["bagged"] = np.nanmean(np.column_stack([upper_dict[m] for m in active_bases]), axis=1)

    final_models = [m for m in requested if m in mean_dict]
    if "bagged" in mean_dict and "bagged" not in final_models and len(final_models) > 1:
        final_models.append("bagged")

    # Errors & Deviation matrix on training window (excluding burn-in first 2 points where diffs initialize)
    burn = min(2, len(train_y) // 5)
    errors_list = []
    desvios_by_model: Dict[str, Dict[str, float]] = {}

    for m_name in final_models:
        fit_vals = fitted_dict[m_name]
        errs = calc_error_metrics(train_y[burn:], fit_vals[burn:])
        errors_list.append({
            "model": m_name,
            "RMSE": errs["RMSE"],
            "MAE": errs["MAE"],
            "MAPE": errs["MAPE"],
        })
        m_desv: Dict[str, float] = {}
        for idx in range(burn, len(train_y)):
            rv = float(train_y[idx])
            pv = float(fit_vals[idx])
            if rv != 0 and np.isfinite(pv):
                m_desv[train_dates.iloc[idx].strftime("%Y-%m-%d")] = ((pv - rv) / rv) * 100.0
        desvios_by_model[m_name] = m_desv

    errors_list.sort(key=lambda r: r["RMSE"] if r["RMSE"] is not None else float("inf"))

    recommendation = evaluate_model_recommendations(
        desvios_by_model,
        modelo_vigente=errors_list[0]["model"] if errors_list else None,
        periodo_maximo=periodo_maximo,
        peso_recente=peso_recente,
        peso_desvio_max=peso_desvio_max,
    )

    # Construct historical + forecast timeline (`df_predict_ts` & `df_export`)
    timeline = []
    for idx, dt_val in enumerate(train_dates):
        dt_str = dt_val.strftime("%Y-%m-%d")
        rv = safe_float(train_y[idx])
        item: Dict[str, Any] = {
            "Data": dt_str,
            "realizado": rv,
            "Period": "Realizado",
        }
        for m_name in final_models:
            item[f"{m_name}_fitted"] = safe_float(fitted_dict[m_name][idx])
            # For seamless export series (realized + projected)
            item[m_name] = rv
            item[f"{m_name}_lower"] = rv
            item[f"{m_name}_upper"] = rv
        timeline.append(item)

    for idx, dt_val in enumerate(future_dates):
        dt_str = dt_val.strftime("%Y-%m-%d")
        item = {
            "Data": dt_str,
            "realizado": None,
            "Period": "Projeção",
        }
        for m_name in final_models:
            item[f"{m_name}_fitted"] = None
            item[m_name] = safe_float(mean_dict[m_name][idx])
            item[f"{m_name}_lower"] = safe_float(lower_dict[m_name][idx])
            item[f"{m_name}_upper"] = safe_float(upper_dict[m_name][idx])
        timeline.append(item)

    return {
        "variable": variable,
        "models_available": final_models,
        "best_model": recommendation["best_model"] or (errors_list[0]["model"] if errors_list else final_models[0]),
        "execution_time_sec": round(time.time() - t0, 2),
        "periods": {
            "train_start": dt_start.strftime("%Y-%m-%d"),
            "train_end": dt_end.strftime("%Y-%m-%d"),
            "forecast_start": future_dates[0].strftime("%Y-%m-%d") if future_dates else None,
            "forecast_end": dt_end_pred.strftime("%Y-%m-%d"),
            "h_future": h,
        },
        "errors": errors_list,
        "recommendation": recommendation,
        "timeline": timeline,
    }
