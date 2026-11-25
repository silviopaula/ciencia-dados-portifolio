"""Motor estatístico Box-Jenkins / SARIMA / SARIMAX em Python (statsmodels).

Fornece:
- Diagnósticos de estacionariedade (ADF e KPSS) com recomendação didática de `d`;
- Cálculo de ACF e PACF com bandas de confiança de Bartlett/95%;
- Seleção automática por AIC/BIC (Auto-SARIMAX) ou ajuste manual;
- Backtest rigoroso fora da amostra vs. baseline Seasonal Naive;
- Diagnóstico de ruído branco nos resíduos (Ljung-Box);
- Projeção futura com intervalos de confiança e análise de crescimento anual (YoY).
"""

from __future__ import annotations

import io
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from app.schemas import (
    AnnualGrowthItem,
    CoefficientItem,
    CorrelationPoint,
    DiagnosticsRequest,
    DiagnosticsResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastSeriesPoint,
    MetricSet,
    ModelCandidateSummary,
    ResidualDiagnostic,
    ResidualPoint,
    SeriesPoint,
    StationarityTestResult,
)

warnings.filterwarnings("ignore")


def _prepare_dataframe(df_raw: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    df = df_raw.copy()
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' não encontrada.")
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df


def _infer_freq_offset(dates: pd.Series) -> pd.DateOffset:
    """Infere o passo temporal predominante (mensal por padrão)."""
    if len(dates) < 2:
        return pd.DateOffset(months=1)
    diffs = dates.diff().dropna().dt.days
    median_days = float(diffs.median())
    if 25 <= median_days <= 35:
        return pd.DateOffset(months=1)
    if 80 <= median_days <= 100:
        return pd.DateOffset(months=3)
    if 6 <= median_days <= 8:
        return pd.DateOffset(weeks=1)
    if median_days <= 2:
        return pd.DateOffset(days=1)
    return pd.DateOffset(years=1)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def run_diagnostics(df_raw: pd.DataFrame, req: DiagnosticsRequest) -> DiagnosticsResponse:
    """Executa testes ADF, KPSS e calcula ACF/PACF para orientar a escolha de (p, d, q)."""
    df = _prepare_dataframe(df_raw, req.date_col, req.target_col)
    obs = df[df[req.target_col].notna()].copy().reset_index(drop=True)

    if len(obs) < 24:
        raise ValueError("São necessárias pelo menos 24 observações válidas para diagnóstico ARIMA.")

    raw_series = obs[req.target_col].astype(float)
    dates = obs[req.date_col]

    # Transformação log opcional
    if req.use_log:
        if (raw_series <= 0).any():
            raise ValueError("Transformação log exige valores estritamente positivos na série alvo.")
        work = np.log(raw_series)
    else:
        work = raw_series.copy()

    # Recomendação automática de d (testando nível e 1ª diferença antes de aplicar a dif do usuário)
    adf_level_p = _safe_float(adfuller(work.dropna(), autolag="AIC")[1], 1.0)
    if adf_level_p < 0.05:
        recommended_d = 0
    else:
        diff1 = work.diff().dropna()
        adf_d1_p = _safe_float(adfuller(diff1, autolag="AIC")[1], 1.0)
        recommended_d = 1 if adf_d1_p < 0.05 else 2

    # Checa se há correlação sazonal forte no lag s (ex.: lag 12) para recomendar D
    recommended_D = 0
    if req.seasonal_period > 1 and len(work) > req.seasonal_period * 2:
        acf_check = acf(work.diff().dropna() if recommended_d > 0 else work.dropna(), nlags=req.seasonal_period, fft=False)
        if abs(acf_check[req.seasonal_period]) > 0.35:
            recommended_D = 1

    # Aplica as diferenciações escolhidas pelo usuário na aba interativa
    transformed = work.copy()
    for _ in range(req.diff_order):
        transformed = transformed.diff()
    if req.seasonal_diff == 1 and req.seasonal_period > 1:
        transformed = transformed.diff(req.seasonal_period)

    valid_mask = transformed.notna()
    series_clean = transformed[valid_mask]
    dates_clean = dates[valid_mask]

    if len(series_clean) < 16:
        raise ValueError("Poucos pontos restantes após diferenciação. Reduza d ou D.")

    # 1. Teste ADF (H0: Não-estacionária / raiz unitária)
    adf_stat, adf_p = adfuller(series_clean, autolag="AIC")[:2]
    adf_stat, adf_p = _safe_float(adf_stat), _safe_float(adf_p, 1.0)
    adf_ok = adf_p < 0.05
    adf_res = StationarityTestResult(
        test_name="Augmented Dickey-Fuller (ADF)",
        statistic=round(adf_stat, 4),
        p_value=round(adf_p, 5),
        is_stationary=adf_ok,
        interpretation=(
            "Rejeita H0 (p < 0,05): evidência estatística de que a série é ESTACIONÁRIA."
            if adf_ok
            else "Não rejeita H0 (p >= 0,05): há sinal de raiz unitária (série NÃO estacionária — teste d = 1)."
        ),
    )

    # 2. Teste KPSS (H0: Estacionária)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p = kpss(series_clean, regression="c", nlags="auto")[:2]
        kpss_stat, kpss_p = _safe_float(kpss_stat), _safe_float(kpss_p, 0.01)
    except Exception:
        kpss_stat, kpss_p = 0.0, 0.10
    kpss_ok = kpss_p >= 0.05
    kpss_res = StationarityTestResult(
        test_name="KPSS (Kwiatkowski–Phillips–Schmidt–Shin)",
        statistic=round(kpss_stat, 4),
        p_value=round(kpss_p, 5),
        is_stationary=kpss_ok,
        interpretation=(
            "Não rejeita H0 (p >= 0,05): confirma estacionariedade em torno da média."
            if kpss_ok
            else "Rejeita H0 (p < 0,05): indica presença de tendência determinística/estocástica (não estacionária)."
        ),
    )

    # 3. ACF e PACF
    max_lags = min(req.nlags, max(6, len(series_clean) // 3 - 1))
    acf_vals = acf(series_clean, nlags=max_lags, fft=False)
    pacf_vals = pacf(series_clean, nlags=max_lags, method="ywm")
    conf_bound = float(1.96 / np.sqrt(len(series_clean)))

    correlations = [
        CorrelationPoint(
            lag=lag,
            acf=round(_safe_float(acf_vals[lag]), 4),
            pacf=round(_safe_float(pacf_vals[lag]), 4),
            conf_bound=round(conf_bound, 4),
        )
        for lag in range(1, max_lags + 1)
    ]

    transformed_series = [
        SeriesPoint(date=d.strftime("%Y-%m-%d"), value=round(_safe_float(v), 4))
        for d, v in zip(dates_clean, series_clean)
    ]

    if adf_ok and kpss_ok:
        didactic_summary = (
            f"Excelente! Tanto o teste ADF quanto o KPSS concordam que a série atual (d={req.diff_order}, D={req.seasonal_diff}) "
            f"está estacionária. Observe as barras que ultrapassam a linha pontilhada (±{conf_bound:.2f}) no PACF para sugerir 'p' "
            f"e no ACF para sugerir 'q'."
        )
    else:
        didactic_summary = (
            f"A série ainda exibe sinais de não-estacionariedade (tendência ou sazonalidade). "
            f"Recomendação automática do diagnóstico: testar d = {recommended_d} e D = {recommended_D} (s = {req.seasonal_period}) "
            f"ou ativar a transformação logarítmica para estabilizar a variância."
        )

    return DiagnosticsResponse(
        n_obs=int(len(series_clean)),
        mean=round(_safe_float(raw_series.mean()), 2),
        std=round(_safe_float(raw_series.std()), 2),
        min_val=round(_safe_float(raw_series.min()), 2),
        max_val=round(_safe_float(raw_series.max()), 2),
        recommended_d=recommended_d,
        recommended_D=recommended_D,
        adf=adf_res,
        kpss=kpss_res,
        correlations=correlations,
        transformed_series=transformed_series,
        didactic_summary=didactic_summary,
    )


def _fit_sarimax(
    y: np.ndarray,
    exog: Optional[np.ndarray],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=150)


def _invert_scale(arr: np.ndarray, use_log: bool, smearing: float = 1.0) -> np.ndarray:
    if use_log:
        return np.exp(arr) * smearing
    return arr


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, aic: Optional[float] = None, bic: Optional[float] = None) -> MetricSet:
    return MetricSet(
        mape=round(_safe_float(mean_absolute_percentage_error(y_true, y_pred) * 100.0), 2),
        rmse=round(_safe_float(root_mean_squared_error(y_true, y_pred)), 2),
        mae=round(_safe_float(mean_absolute_error(y_true, y_pred)), 2),
        r2=round(_safe_float(r2_score(y_true, y_pred)), 4),
        aic=round(_safe_float(aic), 2) if aic is not None else None,
        bic=round(_safe_float(bic), 2) if bic is not None else None,
    )


def _project_missing_exog(
    df_obs: pd.DataFrame,
    df_fut: pd.DataFrame,
    exog_cols: List[str],
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Obtém a matriz de exógenas futuras: usa valores fornecidos na planilha quando existem;
    caso faltem períodos no final, projeta via média sazonal/último valor de forma segura.
    """
    rows = []
    n_avail = len(df_fut)
    for col in exog_cols:
        hist_vals = pd.to_numeric(df_obs[col], errors="coerce").ffill().bfill().values
        if n_avail >= horizon and df_fut[col].iloc[:horizon].notna().all():
            fut_vals = pd.to_numeric(df_fut[col].iloc[:horizon], errors="coerce").values
        else:
            # Aproveita os disponíveis e completa o restante repetindo o último ciclo sazonal
            avail_vals = (
                pd.to_numeric(df_fut[col].iloc[: min(n_avail, horizon)], errors="coerce")
                .ffill()
                .bfill()
                .values
                if n_avail > 0
                else np.array([])
            )
            missing_n = horizon - len(avail_vals)
            cycle = hist_vals[-min(seasonal_period, len(hist_vals)) :]
            padded = np.resize(cycle, missing_n) if missing_n > 0 else np.array([])
            fut_vals = np.concatenate([avail_vals, padded])
        rows.append(fut_vals)
    return np.column_stack(rows)


def run_sarimax_forecast(df_raw: pd.DataFrame, req: ForecastRequest) -> Tuple[ForecastResponse, pd.DataFrame]:
    """Executa o fluxo completo: Auto ou Manual SARIMAX, Backtest vs. Naive, Resíduos e Projeção Futura."""
    df = _prepare_dataframe(df_raw, req.date_col, req.target_col)
    obs = df[df[req.target_col].notna()].copy().reset_index(drop=True)
    fut_rows_in_file = df[df[req.target_col].isna()].copy().reset_index(drop=True)

    if len(obs) < req.test_periods + max(18, req.s + 6):
        raise ValueError(
            f"Histórico insuficiente ({len(obs)} obs) para test_periods={req.test_periods} e sazonalidade s={req.s}."
        )

    exog_cols = [c for c in req.exog_cols if c in df.columns and c not in (req.date_col, req.target_col)]
    for c in exog_cols:
        obs[c] = pd.to_numeric(obs[c], errors="coerce").ffill().bfill()

    y_raw = obs[req.target_col].astype(float).values
    if req.use_log and np.any(y_raw <= 0):
        raise ValueError("Transformação logarítmica exige que todos os valores da série alvo sejam > 0.")

    y_model = np.log(y_raw) if req.use_log else y_raw.copy()

    # Escalonamento Z-score das exógenas (ajustado SOMENTE no treino) para estabilidade numérica do MLE
    split_idx = len(obs) - req.test_periods
    y_train, y_test_raw = y_model[:split_idx], y_raw[split_idx:]
    y_train_raw = y_raw[:split_idx]

    exog_train = exog_test = exog_full = exog_future = None
    if exog_cols:
        X_obs_raw = obs[exog_cols].values.astype(float)
        exog_mean = X_obs_raw[:split_idx].mean(axis=0)
        exog_std = np.where(X_obs_raw[:split_idx].std(axis=0) == 0, 1.0, X_obs_raw[:split_idx].std(axis=0))

        X_obs_scaled = (X_obs_raw - exog_mean) / exog_std
        exog_train = X_obs_scaled[:split_idx]
        exog_test = X_obs_scaled[split_idx:]
        exog_full = X_obs_scaled

        X_fut_raw = _project_missing_exog(obs, fut_rows_in_file, exog_cols, req.forecast_horizon, req.s)
        exog_future = (X_fut_raw - exog_mean) / exog_std

    # 1. Seleção de Ordem: Auto-SARIMAX (grade inteligente rápida por AIC) ou Manual
    candidates_tested: List[ModelCandidateSummary] = []
    if req.mode == "auto":
        # Determina d e D rapidamente
        adf_p = _safe_float(adfuller(y_train, autolag="AIC")[1], 1.0)
        d_auto = 0 if adf_p < 0.05 else 1
        D_auto = 1 if (req.s > 1 and len(y_train) >= req.s * 3) else 0

        if req.s > 1:
            candidate_specs = [
                ((1, d_auto, 1), (1, D_auto, 1, req.s)),
                ((1, d_auto, 0), (1, D_auto, 1, req.s)),
                ((0, d_auto, 1), (0, D_auto, 1, req.s)),
                ((2, d_auto, 1), (1, D_auto, 1, req.s)),
                ((1, d_auto, 1), (0, D_auto, 1, req.s)),
                ((2, d_auto, 0), (1, D_auto, 0, req.s)),
            ]
        else:
            candidate_specs = [
                ((1, d_auto, 1), (0, 0, 0, 0)),
                ((2, d_auto, 1), (0, 0, 0, 0)),
                ((1, d_auto, 0), (0, 0, 0, 0)),
                ((0, d_auto, 1), (0, 0, 0, 0)),
                ((2, d_auto, 2), (0, 0, 0, 0)),
            ]

        best_aic = float("inf")
        best_order = candidate_specs[0][0]
        best_seasonal = candidate_specs[0][1]
        best_fit_train = None
        raw_candidates: List[Tuple[str, float, float, Tuple[int, int, int], Tuple[int, int, int, int]]] = []

        for ord_c, seas_c in candidate_specs:
            try:
                res_c = _fit_sarimax(y_train, exog_train, ord_c, seas_c)
                aic_val = _safe_float(res_c.aic, float("inf"))
                bic_val = _safe_float(res_c.bic, float("inf"))
                lbl = f"({ord_c[0]},{ord_c[1]},{ord_c[2]})x({seas_c[0]},{seas_c[1]},{seas_c[2]},{seas_c[3]})"
                raw_candidates.append((lbl, aic_val, bic_val, ord_c, seas_c))
                if aic_val < best_aic:
                    best_aic = aic_val
                    best_order = ord_c
                    best_seasonal = seas_c
                    best_fit_train = res_c
            except Exception:
                continue

        if best_fit_train is None:
            best_order = (1, 1, 1)
            best_seasonal = (0, 1, 1, req.s) if req.s > 1 else (0, 0, 0, 0)
            best_fit_train = _fit_sarimax(y_train, exog_train, best_order, best_seasonal)

        raw_candidates.sort(key=lambda item: item[1])
        for lbl, aic_v, bic_v, ord_c, seas_c in raw_candidates:
            candidates_tested.append(
                ModelCandidateSummary(
                    order=lbl,
                    aic=round(aic_v, 2),
                    bic=round(bic_v, 2),
                    selected=(ord_c == best_order and seas_c == best_seasonal),
                )
            )
    else:
        best_order = (req.p, req.d, req.q)
        best_seasonal = (req.P, req.D, req.Q, req.s) if req.s > 1 else (0, 0, 0, 0)
        best_fit_train = _fit_sarimax(y_train, exog_train, best_order, best_seasonal)
        lbl = f"({best_order[0]},{best_order[1]},{best_order[2]})x({best_seasonal[0]},{best_seasonal[1]},{best_seasonal[2]},{best_seasonal[3]})"
        candidates_tested.append(
            ModelCandidateSummary(
                order=lbl,
                aic=round(_safe_float(best_fit_train.aic), 2),
                bic=round(_safe_float(best_fit_train.bic), 2),
                selected=True,
            )
        )

    # 2. Backtest no período de teste (Out-of-Sample) + Baseline Seasonal Naive
    alpha_ci = 1.0 - req.confidence_level
    fc_test_obj = best_fit_train.get_forecast(steps=req.test_periods, exog=exog_test)
    pred_test_model = fc_test_obj.predicted_mean
    ci_test_model = fc_test_obj.conf_int(alpha=alpha_ci)

    # Smearing factor de Duan para des-transformação sem viés quando use_log=True
    # Respeita o loglikelihood_burn do filtro de Kalman (evita spikes difusos iniciais nos resíduos)
    base_burn = max(best_order[1] + best_seasonal[1] * max(best_seasonal[3], 1), 2)
    burn_in_train = max(int(getattr(best_fit_train, "loglikelihood_burn", 0)), base_burn)
    train_resids = np.asarray(best_fit_train.resid)[burn_in_train:]
    smearing_train = float(np.mean(np.exp(train_resids))) if (req.use_log and len(train_resids) > 5) else 1.0
    smearing_train = min(max(smearing_train, 0.95), 1.05)

    pred_test_orig = _invert_scale(np.asarray(pred_test_model), req.use_log, smearing_train)
    ci_test_lo = _invert_scale(np.asarray(ci_test_model)[:, 0], req.use_log, 1.0)
    ci_test_hi = _invert_scale(np.asarray(ci_test_model)[:, 1], req.use_log, 1.0)

    # Baseline Seasonal Naive (repete ciclo anterior de tamanho s)
    naive_cycle = y_train_raw[-min(req.s, len(y_train_raw)) :]
    pred_naive_test = np.resize(naive_cycle, req.test_periods)

    test_metrics = _compute_metrics(y_test_raw, pred_test_orig, aic=best_fit_train.aic, bic=best_fit_train.bic)
    baseline_metrics = _compute_metrics(y_test_raw, pred_naive_test)
    mape_improv = round(
        ((baseline_metrics.mape - test_metrics.mape) / max(baseline_metrics.mape, 1e-6)) * 100.0, 1
    )

    # 3. Reajuste em 100% do histórico observado para a Projeção Futura
    fit_full = _fit_sarimax(y_model, exog_full, best_order, best_seasonal)
    burn_in_full = max(int(getattr(fit_full, "loglikelihood_burn", 0)), base_burn)
    full_resids = np.asarray(fit_full.resid)[burn_in_full:]
    smearing_full = float(np.mean(np.exp(full_resids))) if (req.use_log and len(full_resids) > 5) else 1.0
    smearing_full = min(max(smearing_full, 0.95), 1.05)

    fc_fut_obj = fit_full.get_forecast(steps=req.forecast_horizon, exog=exog_future)
    pred_fut_orig = _invert_scale(np.asarray(fc_fut_obj.predicted_mean), req.use_log, smearing_full)
    ci_fut_arr = np.asarray(fc_fut_obj.conf_int(alpha=alpha_ci))
    ci_fut_lo = _invert_scale(ci_fut_arr[:, 0], req.use_log, 1.0)
    ci_fut_hi = _invert_scale(ci_fut_arr[:, 1], req.use_log, 1.0)

    # Valores ajustados no treino (para visualização contínua)
    fitted_train_orig = _invert_scale(np.asarray(best_fit_train.fittedvalues), req.use_log, smearing_train)

    # 4. Tabela de Coeficientes do Modelo Completo
    param_names = list(fit_full.param_names)
    params_arr = np.asarray(fit_full.params)
    bse_arr = np.asarray(fit_full.bse)
    tval_arr = np.asarray(fit_full.tvalues)
    pval_arr = np.asarray(fit_full.pvalues)
    ci_params = np.asarray(fit_full.conf_int(alpha=0.05))

    # Renomeia x1, x2... de volta para os nomes reais das variáveis exógenas
    pretty_names = []
    for pname in param_names:
        if pname.startswith("x") and pname[1:].isdigit():
            idx_ex = int(pname[1:]) - 1
            pretty_names.append(f"{exog_cols[idx_ex]} (padron.)" if idx_ex < len(exog_cols) else pname)
        else:
            pretty_names.append(pname)

    coefficients = [
        CoefficientItem(
            name=pretty_names[i],
            estimate=round(_safe_float(params_arr[i]), 5),
            std_err=round(_safe_float(bse_arr[i]), 5),
            z_stat=round(_safe_float(tval_arr[i]), 3),
            p_value=round(_safe_float(pval_arr[i], 1.0), 5),
            ci_lower=round(_safe_float(ci_params[i, 0]), 5),
            ci_upper=round(_safe_float(ci_params[i, 1]), 5),
            significant=_safe_float(pval_arr[i], 1.0) < 0.05,
        )
        for i in range(len(param_names))
    ]

    # 5. Diagnóstico de Resíduos (Ljung-Box)
    lb_lag = min(12, max(4, len(full_resids) // 5))
    lb_df = acorr_ljungbox(full_resids, lags=[lb_lag], return_df=True)
    lb_stat = _safe_float(lb_df["lb_stat"].iloc[0])
    lb_pval = _safe_float(lb_df["lb_pvalue"].iloc[0], 1.0)
    white_noise = lb_pval >= 0.05

    residuals_diagnostic = ResidualDiagnostic(
        ljung_box_lag=lb_lag,
        ljung_box_stat=round(lb_stat, 3),
        ljung_box_pvalue=round(lb_pval, 4),
        is_white_noise=white_noise,
        residual_mean=round(_safe_float(np.mean(full_resids)), 5),
        residual_std=round(_safe_float(np.std(full_resids)), 5),
        interpretation=(
            f"Aprovado (p = {lb_pval:.4f} >= 0,05): os resíduos comportam-se como ruído branco (sem autocorrelação remanescente relevante)."
            if white_noise
            else f"Atenção (p = {lb_pval:.4f} < 0,05): ainda há alguma estrutura autocorrelacionada nos resíduos; considere ajustar (p,q) ou incluir variáveis exógenas."
        ),
    )

    obs_dates = obs[req.date_col].tolist()
    residuals_series = [
        ResidualPoint(date=obs_dates[burn_in_full + i].strftime("%Y-%m-%d"), residual=round(_safe_float(full_resids[i]), 5))
        for i in range(len(full_resids))
    ]

    # 6. Montagem da série unificada (Treino + Teste + Futuro)
    step_offset = _infer_freq_offset(obs[req.date_col])
    if len(fut_rows_in_file) >= req.forecast_horizon:
        future_dates = fut_rows_in_file[req.date_col].iloc[: req.forecast_horizon].tolist()
    else:
        last_d = obs[req.date_col].iloc[-1]
        future_dates = [last_d + step_offset * (i + 1) for i in range(req.forecast_horizon)]

    series_points: List[ForecastSeriesPoint] = []
    export_rows: List[Dict[str, Any]] = []

    for i in range(split_idx):
        d_str = obs_dates[i].strftime("%Y-%m-%d")
        fit_val = round(_safe_float(fitted_train_orig[i]), 2) if i >= burn_in_train else None
        series_points.append(
            ForecastSeriesPoint(
                date=d_str,
                actual=round(_safe_float(y_raw[i]), 2),
                split="train",
                fitted_or_backtest=fit_val,
            )
        )
        export_rows.append(
            {"Data": d_str, "Periodo": "Treino", "Realizado": round(_safe_float(y_raw[i]), 2), "Previsto_SARIMAX": fit_val, "IC_Inferior": None, "IC_Superior": None}
        )

    for j in range(req.test_periods):
        idx_obs = split_idx + j
        d_str = obs_dates[idx_obs].strftime("%Y-%m-%d")
        p_val = round(_safe_float(pred_test_orig[j]), 2)
        lo_val = round(_safe_float(ci_test_lo[j]), 2)
        hi_val = round(_safe_float(ci_test_hi[j]), 2)
        series_points.append(
            ForecastSeriesPoint(
                date=d_str,
                actual=round(_safe_float(y_raw[idx_obs]), 2),
                split="test",
                fitted_or_backtest=p_val,
                lower_ci=lo_val,
                upper_ci=hi_val,
                baseline_naive=round(_safe_float(pred_naive_test[j]), 2),
            )
        )
        export_rows.append(
            {"Data": d_str, "Periodo": "Teste (Backtest)", "Realizado": round(_safe_float(y_raw[idx_obs]), 2), "Previsto_SARIMAX": p_val, "IC_Inferior": lo_val, "IC_Superior": hi_val}
        )

    for k in range(req.forecast_horizon):
        d_str = future_dates[k].strftime("%Y-%m-%d")
        p_val = round(_safe_float(pred_fut_orig[k]), 2)
        lo_val = round(_safe_float(ci_fut_lo[k]), 2)
        hi_val = round(_safe_float(ci_fut_hi[k]), 2)
        series_points.append(
            ForecastSeriesPoint(
                date=d_str,
                actual=None,
                split="future",
                forecast=p_val,
                lower_ci=lo_val,
                upper_ci=hi_val,
            )
        )
        export_rows.append(
            {"Data": d_str, "Periodo": "Projeção Futura", "Realizado": None, "Previsto_SARIMAX": p_val, "IC_Inferior": lo_val, "IC_Superior": hi_val}
        )

    # 7. Tabela de Crescimento Anual (YoY) nos últimos anos históricos + anos projetados
    df_export = pd.DataFrame(export_rows)
    df_growth = df_export.copy()
    df_growth["Ano"] = pd.to_datetime(df_growth["Data"]).dt.year
    df_growth["Valor_Ref"] = df_growth["Realizado"].fillna(df_growth["Previsto_SARIMAX"])

    annual_rows: List[AnnualGrowthItem] = []
    prev_mean = None
    for yr, grp in df_growth.groupby("Ano"):
        has_real = grp["Realizado"].notna().any()
        has_fut = (grp["Periodo"] == "Projeção Futura").any()
        src = "Misto" if (has_real and has_fut) else ("Projeção" if has_fut else "Histórico")
        mean_v = float(grp["Valor_Ref"].mean())
        total_v = float(grp["Valor_Ref"].sum())
        yoy = round(((mean_v / prev_mean) - 1.0) * 100.0, 2) if (prev_mean and prev_mean > 0) else None
        annual_rows.append(
            AnnualGrowthItem(
                year=int(yr),
                source=src,
                months_count=int(len(grp)),
                mean_value=round(mean_v, 2),
                total_value=round(total_v, 2),
                yoy_growth_pct=yoy,
            )
        )
        prev_mean = mean_v

    model_kind = "SARIMAX" if exog_cols else ("SARIMA" if req.s > 1 else "ARIMA")
    model_label = f"{model_kind}({best_order[0]},{best_order[1]},{best_order[2]})×({best_seasonal[0]},{best_seasonal[1]},{best_seasonal[2]},{best_seasonal[3]})"

    exog_note = f" com {len(exog_cols)} variável(is) exógena(s) ({', '.join(exog_cols)})" if exog_cols else " (univariado)"
    takeaway = (
        f"O modelo {model_label}{exog_note} atingiu MAPE de {test_metrics.mape:.2f}% e R² de {test_metrics.r2:.3f} "
        f"no backtest de {req.test_periods} períodos — uma redução de {mape_improv:+.1f}% no erro percentual frente ao "
        f"baseline Seasonal Naive (MAPE {baseline_metrics.mape:.2f}%)."
    )

    resp = ForecastResponse(
        model_label=model_label,
        order=list(best_order),
        seasonal_order=list(best_seasonal),
        use_log=req.use_log,
        exog_cols=exog_cols,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        mape_improvement_pct=mape_improv,
        coefficients=coefficients,
        residuals_diagnostic=residuals_diagnostic,
        residuals_series=residuals_series,
        series=series_points,
        annual_growth=annual_rows[-10:],  # Últimos 10 anos (histórico recente + projeção)
        candidates_tested=candidates_tested,
        executive_takeaway=takeaway,
    )
    return resp, df_export


def build_excel_report(df_export: pd.DataFrame, resp: ForecastResponse) -> bytes:
    """Gera relatório Excel (.xlsx) multi-aba com Projeções, Métricas, Coeficientes e Crescimento."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="Projecoes", index=False)

        pd.DataFrame([
            {"Modelo": resp.model_label, "MAPE (%)": resp.test_metrics.mape, "RMSE": resp.test_metrics.rmse, "MAE": resp.test_metrics.mae, "R2": resp.test_metrics.r2, "AIC": resp.test_metrics.aic, "BIC": resp.test_metrics.bic},
            {"Modelo": "Baseline Seasonal Naive", "MAPE (%)": resp.baseline_metrics.mape, "RMSE": resp.baseline_metrics.rmse, "MAE": resp.baseline_metrics.mae, "R2": resp.baseline_metrics.r2, "AIC": None, "BIC": None},
        ]).to_excel(writer, sheet_name="Metricas_Backtest", index=False)

        pd.DataFrame([c.model_dump() for c in resp.coefficients]).to_excel(writer, sheet_name="Coeficientes", index=False)
        pd.DataFrame([g.model_dump() for g in resp.annual_growth]).to_excel(writer, sheet_name="Crescimento_Anual", index=False)

    buf.seek(0)
    return buf.getvalue()
