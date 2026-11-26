"""Motor econométrico completo para o TimeSeries Diagnostics Studio.

Implementa com rigor estatístico:
- Estacionariedade & Raiz Unitária: ADF (c/ct/n), Phillips-Perron (Newey-West HAC exato),
  KPSS (c/ct), Zivot-Andrews (quebra estrutural endógena) e Consenso 4-Quadrantes.
- Matriz Comparativa de Transformações (Nível, Log, Box-Cox, Δ, Δ_s, Δ+Δ_s, Δ log).
- Autocorrelação & Dependência: Box-Pierce, Ljung-Box (série, resíduos com model_df e curva por lag),
  Durbin-Watson, ACF/PACF e diagnóstico automático Box-Jenkins.
- Heterocedasticidade & Volatilidade: Engle ARCH-LM, Breusch-Pagan Temporal (variância vs t) e Rolling Std.
- Normalidade & Outliers: Anderson-Darling, Jarque-Bera, Shapiro-Wilk, Q-Q Plot, IQR e Z-MAD Robusto.
- Decomposição STL Robusta (Força de Tendência F_T e Sazonalidade F_S — Wang, Smith & Hyndman 2006).
- Modelagem SARIMA Auto/Manual + Backtest Out-of-Sample vs Seasonal Naive + Previsão IC 80%/95%.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf, zivot_andrews

warnings.filterwarnings("ignore")

TRANSFORM_LABELS: Dict[str, str] = {
    "level": "Nível Original (yₜ)",
    "log": "Logaritmo Natural (ln yₜ)",
    "boxcox": "Box-Cox Ótimo (λ automático)",
    "diff1": "1ª Diferença (Δyₜ)",
    "diff_seasonal": "Diferença Sazonal (Δₛyₜ)",
    "diff1_seasonal": "1ª Diferença + Sazonal (ΔΔₛyₜ)",
    "log_diff": "Log-Retorno / Crescimento (Δ ln yₜ)",
}


def _sf(val: Any, digits: int = 4) -> Optional[float]:
    """Converte número para float JSON-safe arredondado (None se NaN/Inf)."""
    if val is None:
        return None
    try:
        f = float(val)
        if not math.isfinite(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    if p < 0.0001:
        return "< 0,0001"
    return f"{p:.4f}".replace(".", ",")


def apply_transformation(
    dates: pd.Series,
    values: np.ndarray,
    transform: str,
    seasonal_period: int,
) -> Tuple[pd.Series, np.ndarray, Dict[str, Any]]:
    """Aplica transformação estatística à série preservando alinhamento exato das datas."""
    y = np.asarray(values, dtype=float)
    d = pd.Series(dates).reset_index(drop=True)
    meta: Dict[str, Any] = {
        "transform": transform,
        "label": TRANSFORM_LABELS.get(transform, transform),
        "boxcox_lambda": None,
        "note": None,
    }

    if transform == "level":
        return d, y, meta

    if transform in ("log", "log_diff"):
        if np.any(y <= 0):
            raise ValueError(
                "A transformação logarítmica exige valores estritamente positivos (y > 0). "
                "Use Nível ou Diferenciação para séries com zeros ou valores negativos."
            )
        ly = np.log(y)
        if transform == "log":
            return d, ly, meta
        return d.iloc[1:].reset_index(drop=True), np.diff(ly), meta

    if transform == "boxcox":
        if np.any(y <= 0):
            raise ValueError(
                "A transformação Box-Cox exige valores estritamente positivos (y > 0)."
            )
        bc_y, lam = stats.boxcox(y)
        meta["boxcox_lambda"] = _sf(lam, 4)
        meta["label"] = f"Box-Cox (λ = {lam:.2f})"
        return d, np.asarray(bc_y, dtype=float), meta

    if transform == "diff1":
        return d.iloc[1:].reset_index(drop=True), np.diff(y), meta

    s = max(1, int(seasonal_period))
    if transform == "diff_seasonal":
        if s <= 1 or len(y) <= s + 10:
            raise ValueError(f"Diferença sazonal requer período s ≥ 2 e série com pelo menos {s + 11} obs.")
        return d.iloc[s:].reset_index(drop=True), y[s:] - y[:-s], meta

    if transform == "diff1_seasonal":
        if s <= 1 or len(y) <= s + 12:
            raise ValueError(f"Diferença Δ+Δₛ requer período s ≥ 2 e série com pelo menos {s + 13} obs.")
        ds = y[s:] - y[:-s]
        return d.iloc[s + 1:].reset_index(drop=True), np.diff(ds), meta

    return d, y, meta


def phillips_perron_test(y: np.ndarray, regression: str = "ct") -> Dict[str, Any]:
    """Teste Phillips-Perron (Z_t) com variância de longo prazo Newey-West (Bartlett) e MacKinnon p-valor."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 15:
        return {"statistic": None, "p_value": None, "lags": 0, "crit_1": None, "crit_5": None, "crit_10": None}

    y_dep = y[1:]
    y_lag = y[:-1]
    t_obs = len(y_dep)
    t_idx = np.arange(1, t_obs + 1, dtype=float)

    if regression == "ct":
        X = np.column_stack([y_lag, np.ones(t_obs), t_idx])
    elif regression == "c":
        X = np.column_stack([y_lag, np.ones(t_obs)])
    else:
        X = y_lag.reshape(-1, 1)

    k = X.shape[1]
    beta, _, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)
    u = y_dep - X @ beta
    s2 = float(np.sum(u ** 2) / max(t_obs - k, 1))
    xtx_inv = np.linalg.pinv(X.T @ X)
    se_rho = float(np.sqrt(max(s2 * xtx_inv[0, 0], 1e-18)))
    rho = float(beta[0])
    t_rho = (rho - 1.0) / se_rho

    # Newey-West truncation lag (Schwert/Bartlett)
    nlags = max(1, int(np.floor(4.0 * ((t_obs / 100.0) ** (2.0 / 9.0)))))
    gamma0 = float(np.dot(u, u) / t_obs)
    lam2 = gamma0
    for j in range(1, nlags + 1):
        w = 1.0 - (j / (nlags + 1.0))
        gamma_j = float(np.dot(u[j:], u[:-j]) / t_obs)
        lam2 += 2.0 * w * gamma_j

    sigma2 = max(gamma0, 1e-12)
    lam2 = max(lam2, 1e-12)
    s = math.sqrt(max(s2, 1e-12))
    lam = math.sqrt(lam2)

    z_t = math.sqrt(sigma2 / lam2) * t_rho - 0.5 * ((lam2 - sigma2) / lam) * ((t_obs * se_rho) / s)
    p_val = float(mackinnonp(z_t, regression=regression, N=1))
    p_val = min(max(p_val, 0.0), 1.0)
    crits = mackinnoncrit(N=1, regression=regression, nobs=t_obs)
    return {
        "statistic": _sf(z_t, 4),
        "p_value": _sf(p_val, 4),
        "lags": nlags,
        "crit_1": _sf(crits[0], 4),
        "crit_5": _sf(crits[1], 4),
        "crit_10": _sf(crits[2], 4),
    }


def _safe_adf(y: np.ndarray, regression: str = "ct") -> Dict[str, Any]:
    try:
        res = adfuller(y, regression=regression, autolag="AIC")
        return {
            "statistic": _sf(res[0], 4),
            "p_value": _sf(res[1], 4),
            "lags": int(res[2]),
            "nobs": int(res[3]),
            "crit_1": _sf(res[4]["1%"], 4),
            "crit_5": _sf(res[4]["5%"], 4),
            "crit_10": _sf(res[4]["10%"], 4),
        }
    except Exception:
        return {"statistic": None, "p_value": None, "lags": 0, "nobs": len(y), "crit_1": None, "crit_5": None, "crit_10": None}


def _safe_kpss(y: np.ndarray, regression: str = "ct") -> Dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, lags, crits = kpss(y, regression=regression, nlags="auto")
        note = None
        if pval <= 0.0101:
            note = "p-valor limitado inferiormente a ≤ 0,01 (tabela assintótica KPSS)"
        elif pval >= 0.0999:
            note = "p-valor limitado superiormente a ≥ 0,10 (tabela assintótica KPSS)"
        return {
            "statistic": _sf(stat, 4),
            "p_value": _sf(pval, 4),
            "lags": int(lags),
            "crit_1": _sf(crits["1%"], 4),
            "crit_5": _sf(crits["5%"], 4),
            "crit_10": _sf(crits["10%"], 4),
            "note": note,
        }
    except Exception:
        return {"statistic": None, "p_value": None, "lags": 0, "crit_1": None, "crit_5": None, "crit_10": None, "note": None}


def _safe_zivot_andrews(y: np.ndarray, dates: pd.Series, regression: str = "ct") -> Dict[str, Any]:
    try:
        reg_za = "ct" if regression == "ct" else "c"
        stat, pval, crits, baselag, bp_idx = zivot_andrews(y, trim=0.15, regression=reg_za, autolag="AIC")
        bp_int = int(bp_idx)
        bp_date = str(pd.Timestamp(dates.iloc[bp_int]).strftime("%Y-%m-%d")) if 0 <= bp_int < len(dates) else None
        return {
            "statistic": _sf(stat, 4),
            "p_value": _sf(pval, 4),
            "lags": int(baselag),
            "break_index": bp_int,
            "break_date": bp_date,
            "crit_1": _sf(crits["1%"], 4),
            "crit_5": _sf(crits["5%"], 4),
            "crit_10": _sf(crits["10%"], 4),
        }
    except Exception:
        return {
            "statistic": None,
            "p_value": None,
            "lags": 0,
            "break_index": None,
            "break_date": None,
            "crit_1": None,
            "crit_5": None,
            "crit_10": None,
        }


def _safe_arch(series: np.ndarray, lags: int) -> Dict[str, Any]:
    try:
        eff_lags = max(1, min(lags, len(series) // 5))
        lm_stat, lm_p, f_stat, f_p = het_arch(series, nlags=eff_lags)
        return {
            "statistic": _sf(lm_stat, 4),
            "p_value": _sf(lm_p, 4),
            "f_stat": _sf(f_stat, 4),
            "f_p": _sf(f_p, 4),
            "lags": eff_lags,
        }
    except Exception:
        return {"statistic": None, "p_value": None, "f_stat": None, "f_p": None, "lags": lags}


def _temporal_heteroskedasticity_test(series: np.ndarray) -> Dict[str, Any]:
    """Teste Breusch-Pagan temporal: regride desvios² contra [1, t, t²] para detectar variância crescente no tempo."""
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 20:
        return {"statistic": None, "p_value": None}
    centered2 = (y - np.mean(y)) ** 2
    t = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([np.ones(n), t, t ** 2])
    beta, _, _, _ = np.linalg.lstsq(X, centered2, rcond=None)
    pred = X @ beta
    ss_tot = float(np.sum((centered2 - np.mean(centered2)) ** 2))
    ss_res = float(np.sum((centered2 - pred) ** 2))
    r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
    lm_stat = n * r2
    p_val = float(1.0 - stats.chi2.cdf(lm_stat, df=2))
    return {"statistic": _sf(lm_stat, 4), "p_value": _sf(p_val, 4), "r2": _sf(r2, 4)}


def _anderson_darling_normal(series: np.ndarray) -> Dict[str, Any]:
    """Anderson-Darling para Normalidade com p-valor aproximado de D'Agostino & Stephens (1986)."""
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n < 8:
        return {"statistic": None, "p_value": None, "crit_5": None}
    res = stats.anderson(y, dist="norm")
    a2 = float(res.statistic)
    # Ajuste finito de Stephens (1986)
    a2_star = a2 * (1.0 + 0.75 / n + 2.25 / (n ** 2))
    if a2_star >= 0.6:
        p = math.exp(1.2937 - 5.709 * a2_star + 0.0186 * (a2_star ** 2))
    elif a2_star > 0.34:
        p = math.exp(0.9177 - 4.279 * a2_star - 1.38 * (a2_star ** 2))
    elif a2_star > 0.20:
        p = 1.0 - math.exp(-8.318 + 42.796 * a2_star - 59.938 * (a2_star ** 2))
    else:
        p = 1.0 - math.exp(-13.436 + 101.14 * a2_star - 223.73 * (a2_star ** 2))
    p = min(max(p, 0.0), 1.0)
    crit_5 = float(res.critical_values[2]) if len(res.critical_values) >= 3 else 0.787
    return {"statistic": _sf(a2, 4), "p_value": _sf(p, 4), "crit_5": _sf(crit_5, 4)}


def _build_stationarity_consensus(
    adf_p: Optional[float],
    pp_p: Optional[float],
    kpss_p: Optional[float],
    za_res: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """Veredito cruzado 4-quadrantes (ADF × PP × KPSS × Zivot-Andrews)."""
    adf_stat = adf_p is not None and adf_p < alpha
    pp_stat = pp_p is not None and pp_p < alpha
    kpss_stat = kpss_p is not None and kpss_p >= alpha
    za_p = za_res.get("p_value")
    za_stat = za_p is not None and za_p < alpha

    unit_root_rejected = adf_stat and pp_stat

    if unit_root_rejected and kpss_stat:
        return {
            "quadrant": "estacionaria",
            "badge_color": "emerald",
            "regime_title": "Série Estacionária (Consenso Forte)",
            "summary": (
                "ADF e Phillips-Perron rejeitam raiz unitária e o KPSS não rejeita estacionariedade. "
                "A média e a estrutura de autocovariância são estáveis no tempo."
            ),
            "recommendation": "Pronta para modelagem ARMA/SARIMA com d = 0 (avalie apenas efeitos ARCH e sazonalidade).",
            "suggested_d": 0,
        }

    if (not adf_stat) and za_stat:
        bp_dt = za_res.get("break_date") or "data identificada"
        return {
            "quadrant": "quebra_estrutural",
            "badge_color": "amber",
            "regime_title": f"Quebra Estrutural Endógena ({bp_dt})",
            "summary": (
                f"O ADF clássico não rejeita raiz unitária, mas Zivot-Andrews rejeita (p = {_fmt_p(za_p)}) "
                f"ao permitir quebra estrutural em {bp_dt} — clássico viés de Perron (1989)."
            ),
            "recommendation": (
                f"Incluir variável dummy de intervenção a partir de {bp_dt} (SARIMAX) ou modelar em 1ª diferença (d = 1)."
            ),
            "suggested_d": 1,
        }

    if (not unit_root_rejected) and (not kpss_stat):
        return {
            "quadrant": "nao_estacionaria",
            "badge_color": "rose",
            "regime_title": "Não-Estacionária — Raiz Unitária I(1)",
            "summary": (
                "ADF/PP não rejeitam raiz unitária e o KPSS rejeita estacionariedade. "
                "Modelar diretamente no nível produz regressões espúrias e previsões instáveis."
            ),
            "recommendation": (
                "Aplicar 1ª Diferença (Δyₜ) ou Log-Retorno (Δ ln yₜ); se houver forte sazonalidade, combinar com Δₛ."
            ),
            "suggested_d": 1,
        }

    return {
        "quadrant": "divergente",
        "badge_color": "indigo",
        "regime_title": "Sinais Divergentes (Tendência / Sazonalidade / Memória Longa)",
        "summary": (
            "Há divergência entre os testes de raiz unitária (ADF/PP) e o teste de estacionariedade (KPSS), "
            "comum quando há sazonalidade forte, tendência determinística ou raiz próxima de 1."
        ),
        "recommendation": (
            "Inspecione o Laboratório de Transformações (Δyₜ, Δₛyₜ ou tendência determinística) e verifique a queda da ACF."
        ),
        "suggested_d": 1,
    }


def _compute_transform_matrix(
    dates: pd.Series,
    raw_values: np.ndarray,
    seasonal_period: int,
    regression: str,
    alpha: float,
    max_lags: int,
) -> List[Dict[str, Any]]:
    """Testa todas as 7 transformações lado a lado para identificar a mínima transformação estacionária."""
    order = ["level", "log", "boxcox", "diff1", "diff_seasonal", "diff1_seasonal", "log_diff"]
    rows: List[Dict[str, Any]] = []
    first_stationary_key: Optional[str] = None

    for key in order:
        try:
            _, yt, tmeta = apply_transformation(dates, raw_values, key, seasonal_period)
            if len(yt) < 20 or np.std(yt) < 1e-10:
                raise ValueError("Série resultante insuficiente ou constante.")
            reg_eff = "c" if key in ("diff1", "diff_seasonal", "diff1_seasonal", "log_diff") else regression
            adf_r = _safe_adf(yt, regression=reg_eff)
            pp_r = phillips_perron_test(yt, regression=reg_eff)
            kpss_r = _safe_kpss(yt, regression=reg_eff)
            arch_r = _safe_arch(yt, lags=min(max_lags, 12))

            is_stat = (
                adf_r["p_value"] is not None
                and adf_r["p_value"] < alpha
                and pp_r["p_value"] is not None
                and pp_r["p_value"] < alpha
                and kpss_r["p_value"] is not None
                and kpss_r["p_value"] >= alpha
            )
            if is_stat:
                consensus = "Estacionária (Consenso)"
                status = "ok"
                if first_stationary_key is None:
                    first_stationary_key = key
            elif adf_r["p_value"] is not None and adf_r["p_value"] < alpha:
                consensus = "Parcial (ADF Ok / KPSS Alerta)"
                status = "warn"
            else:
                consensus = "Não-Estacionária"
                status = "fail"

            rows.append({
                "key": key,
                "label": tmeta["label"],
                "available": True,
                "reason": None,
                "n_obs": int(len(yt)),
                "adf_stat": adf_r["statistic"],
                "adf_p": adf_r["p_value"],
                "pp_p": pp_r["p_value"],
                "kpss_p": kpss_r["p_value"],
                "arch_p": arch_r["p_value"],
                "std": _sf(np.std(yt, ddof=1), 4),
                "consensus": consensus,
                "status": status,
                "recommended": False,
            })
        except Exception as exc:
            rows.append({
                "key": key,
                "label": TRANSFORM_LABELS[key],
                "available": False,
                "reason": str(exc),
                "n_obs": 0,
                "adf_stat": None,
                "adf_p": None,
                "pp_p": None,
                "kpss_p": None,
                "arch_p": None,
                "std": None,
                "consensus": "Indisponível",
                "status": "na",
                "recommended": False,
            })

    if first_stationary_key is None:
        # Fallback: escolhe o menor p-valor ADF disponível
        avail = [r for r in rows if r["available"] and r["adf_p"] is not None]
        if avail:
            first_stationary_key = min(avail, key=lambda r: r["adf_p"])["key"]

    for r in rows:
        if r["key"] == first_stationary_key:
            r["recommended"] = True
    return rows


def _fit_sarima_and_backtest(
    y: np.ndarray,
    dates: pd.Series,
    seasonal_period: int,
    auto_arima: bool,
    manual_order: Tuple[int, int, int],
    manual_seasonal: Tuple[int, int, int],
    forecast_horizon: int,
    suggested_d: int,
    seasonal_strength: float,
) -> Dict[str, Any]:
    """Ajusta SARIMA (Auto por AICc ou Manual), faz Backtest OOS vs Seasonal Naive e prevê com IC 80%/95%."""
    n = len(y)
    s = max(1, int(seasonal_period))
    use_seasonal = s >= 2 and n >= 3 * s and seasonal_strength >= 0.35

    if auto_arima:
        candidates: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        d_cands = [0, 1] if suggested_d == 0 else [1, 0]
        if use_seasonal:
            for d_v in d_cands:
                candidates.extend([
                    ((1, d_v, 0), (1, 0, 0, s)),
                    ((1, d_v, 1), (1, 0, 0, s)),
                    ((0, d_v, 1), (0, 1, 1, s)),
                    ((1, d_v, 0), (0, 1, 1, s)),
                    ((1, d_v, 1), (0, 1, 1, s)),
                ])
        else:
            for d_v in d_cands:
                candidates.extend([
                    ((1, d_v, 0), (0, 0, 0, 0)),
                    ((0, d_v, 1), (0, 0, 0, 0)),
                    ((1, d_v, 1), (0, 0, 0, 0)),
                    ((2, d_v, 0), (0, 0, 0, 0)),
                    ((2, d_v, 1), (0, 0, 0, 0)),
                ])
    else:
        s_ord = (*manual_seasonal, s) if (s >= 2 and any(manual_seasonal)) else (0, 0, 0, 0)
        candidates = [(manual_order, s_ord)]

    best_res = None
    best_aicc = float("inf")
    best_spec = candidates[0]

    for order, s_order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=s_order,
                    trend="c" if (order[1] == 0 and s_order[1] == 0) else "n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=80)
            k_p = len(res.params)
            aicc = float(res.aic) + (2.0 * k_p * (k_p + 1.0)) / max(n - k_p - 1.0, 1.0)
            if math.isfinite(aicc) and aicc < best_aicc:
                best_aicc = aicc
                best_res = res
                best_spec = (order, s_order)
        except Exception:
            continue

    if best_res is None:
        # Fallback ultra-estável AR(1)
        best_spec = ((1, 0, 0), (0, 0, 0, 0))
        best_res = SARIMAX(y, order=(1, 0, 0), trend="c", enforce_stationarity=False).fit(disp=False)
        best_aicc = float(best_res.aic)

    order, s_order = best_spec
    if s_order[3] > 0:
        spec_label = f"SARIMA{order}×({s_order[0]},{s_order[1]},{s_order[2]})[{s_order[3]}]"
    else:
        spec_label = f"ARIMA{order}"

    # Descarta primeiros burn-in passos de diferenciação nos resíduos diagnósticos
    burn = min(order[1] + s_order[1] * max(s_order[3], 1), max(n // 5, 1))
    raw_resid = np.asarray(best_res.resid, dtype=float)
    diag_resid = raw_resid[burn:] if len(raw_resid) > burn + 12 else raw_resid
    diag_resid = diag_resid[np.isfinite(diag_resid)]
    model_df = int(order[0] + order[2] + s_order[0] + s_order[2])

    # Previsão futura + datas inferidas
    fc_obj = best_res.get_forecast(steps=forecast_horizon)
    fc_mean = np.asarray(fc_obj.predicted_mean, dtype=float)
    ci80 = np.asarray(fc_obj.conf_int(alpha=0.20), dtype=float)
    ci95 = np.asarray(fc_obj.conf_int(alpha=0.05), dtype=float)

    last_dt = pd.Timestamp(dates.iloc[-1])
    med_days = float(pd.Series(dates).diff().dt.days.dropna().median()) if len(dates) >= 3 else 30.0
    if med_days <= 2.5:
        fc_dates = pd.date_range(last_dt + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    elif med_days <= 10:
        fc_dates = pd.date_range(last_dt + pd.Timedelta(days=7), periods=forecast_horizon, freq="7D")
    elif med_days <= 45:
        fc_dates = pd.date_range(last_dt + pd.DateOffset(months=1), periods=forecast_horizon, freq="MS")
    elif med_days <= 110:
        fc_dates = pd.date_range(last_dt + pd.DateOffset(months=3), periods=forecast_horizon, freq="QS")
    else:
        fc_dates = pd.date_range(last_dt + pd.DateOffset(years=1), periods=forecast_horizon, freq="YS")

    forecast_rows = [
        {
            "date": str(fc_dates[i].strftime("%Y-%m-%d")),
            "mean": _sf(fc_mean[i], 4),
            "lower_80": _sf(ci80[i, 0], 4),
            "upper_80": _sf(ci80[i, 1], 4),
            "lower_95": _sf(ci95[i, 0], 4),
            "upper_95": _sf(ci95[i, 1], 4),
        }
        for i in range(forecast_horizon)
    ]

    # Backtest Out-of-Sample (últimos h_bt períodos) vs Seasonal Naive
    h_bt = min(12, max(4, n // 6))
    y_train, y_test = y[:-h_bt], y[-h_bt:]
    bt_dates = dates.iloc[-h_bt:].reset_index(drop=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt_mod = SARIMAX(
                y_train,
                order=order,
                seasonal_order=s_order,
                trend="c" if (order[1] == 0 and s_order[1] == 0) else "n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=60)
            y_pred_bt = np.asarray(bt_mod.get_forecast(steps=h_bt).predicted_mean, dtype=float)
    except Exception:
        y_pred_bt = np.repeat(y_train[-1], h_bt)

    if s >= 2 and len(y_train) >= s:
        y_naive = np.array([y_train[-s + (i % s)] for i in range(h_bt)], dtype=float)
    else:
        y_naive = np.repeat(y_train[-1], h_bt)

    rmse_m = float(np.sqrt(np.mean((y_test - y_pred_bt) ** 2)))
    mae_m = float(np.mean(np.abs(y_test - y_pred_bt)))
    rmse_n = float(np.sqrt(np.mean((y_test - y_naive) ** 2)))
    mae_n = float(np.mean(np.abs(y_test - y_naive)))
    denom = np.where(np.abs(y_test) > 1e-6, np.abs(y_test), np.nan)
    mape_m = float(np.nanmean(np.abs((y_test - y_pred_bt) / denom)) * 100.0) if np.any(np.isfinite(denom)) else None
    mape_n = float(np.nanmean(np.abs((y_test - y_naive) / denom)) * 100.0) if np.any(np.isfinite(denom)) else None
    skill_pct = ((rmse_n - rmse_m) / max(rmse_n, 1e-9)) * 100.0

    backtest_series = [
        {
            "date": str(pd.Timestamp(bt_dates.iloc[i]).strftime("%Y-%m-%d")),
            "actual": _sf(y_test[i], 4),
            "sarima": _sf(y_pred_bt[i], 4),
            "naive": _sf(y_naive[i], 4),
        }
        for i in range(h_bt)
    ]

    return {
        "spec_label": spec_label,
        "order": list(order),
        "seasonal_order": list(s_order),
        "model_df": model_df,
        "aic": _sf(best_res.aic, 2),
        "aicc": _sf(best_aicc, 2),
        "bic": _sf(best_res.bic, 2),
        "fitted_values": np.asarray(best_res.fittedvalues, dtype=float),
        "raw_residuals": raw_resid,
        "diag_residuals": diag_resid,
        "forecast": forecast_rows,
        "backtest": {
            "horizon": h_bt,
            "rmse": _sf(rmse_m, 4),
            "mae": _sf(mae_m, 4),
            "mape": _sf(mape_m, 2),
            "naive_rmse": _sf(rmse_n, 4),
            "naive_mae": _sf(mae_n, 4),
            "naive_mape": _sf(mape_n, 2),
            "skill_vs_naive_pct": _sf(skill_pct, 1),
            "series": backtest_series,
        },
    }


def _histogram_with_normal(values: np.ndarray, bins: int = 18) -> List[Dict[str, Any]]:
    v = values[np.isfinite(values)]
    if len(v) < 5:
        return []
    counts, edges = np.histogram(v, bins=bins)
    mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
    sd = max(sd, 1e-9)
    width = float(edges[1] - edges[0])
    out = []
    for i in range(len(counts)):
        mid = 0.5 * (edges[i] + edges[i + 1])
        expected = float(stats.norm.pdf(mid, loc=mu, scale=sd) * len(v) * width)
        out.append({
            "bin_label": f"{mid:.2f}",
            "bin_mid": _sf(mid, 4),
            "count": int(counts[i]),
            "normal_expected": _sf(expected, 2),
        })
    return out


def _qq_normal_points(values: np.ndarray) -> List[Dict[str, Any]]:
    v = values[np.isfinite(values)]
    if len(v) < 8:
        return []
    z = (v - np.mean(v)) / max(float(np.std(v, ddof=1)), 1e-9)
    (osm, osr), _ = stats.probplot(z, dist="norm")
    # Amostra até 120 pontos ordenados se série for longa para manter gráfico rápido
    if len(osm) > 120:
        idx = np.linspace(0, len(osm) - 1, 120).astype(int)
        osm, osr = osm[idx], osr[idx]
    return [
        {"theoretical": _sf(osm[i], 3), "sample": _sf(osr[i], 3), "ref_line": _sf(osm[i], 3)}
        for i in range(len(osm))
    ]


def run_full_diagnostics(
    df: pd.DataFrame,
    dataset_title: str,
    dataset_id: Optional[str] = None,
    transform: str = "level",
    alpha: float = 0.05,
    regression: str = "ct",
    max_lags: int = 12,
    seasonal_period: int = 12,
    auto_arima: bool = True,
    manual_order: Tuple[int, int, int] = (1, 1, 1),
    manual_seasonal: Tuple[int, int, int] = (0, 0, 0),
    forecast_horizon: int = 12,
) -> Dict[str, Any]:
    """Executa bateria econométrica completa e devolve payload pronto para UI React + Excel."""
    raw_dates = pd.to_datetime(df["Date"]).reset_index(drop=True)
    raw_vals = df["Value"].to_numpy(dtype=float)

    t_dates, y, t_meta = apply_transformation(raw_dates, raw_vals, transform, seasonal_period)
    n = len(y)
    if n < 20:
        raise ValueError(f"Série transformada possui apenas {n} observações (mínimo exigido: 20).")

    s = max(1, int(seasonal_period))
    # 1. Estacionariedade (especificação principal + todas 3 especificações ADF)
    adf_c = _safe_adf(y, regression="c")
    adf_ct = _safe_adf(y, regression="ct")
    adf_n = _safe_adf(y, regression="n")
    adf_main = adf_ct if regression == "ct" else adf_c

    pp_main = phillips_perron_test(y, regression=regression)
    kpss_c = _safe_kpss(y, regression="c")
    kpss_ct = _safe_kpss(y, regression="ct")
    kpss_main = kpss_ct if regression == "ct" else kpss_c

    za_main = _safe_zivot_andrews(y, t_dates, regression=regression)
    consensus = _build_stationarity_consensus(
        adf_main["p_value"], pp_main["p_value"], kpss_main["p_value"], za_main, alpha
    )

    # 2. Decomposição STL Robusta + Forças (Wang, Smith & Hyndman 2006)
    if s >= 2 and n >= 2 * s + 2:
        stl_res = STL(y, period=s, robust=True).fit()
        trend = np.asarray(stl_res.trend, dtype=float)
        seasonal = np.asarray(stl_res.seasonal, dtype=float)
        remainder = np.asarray(stl_res.resid, dtype=float)
        var_r = float(np.var(remainder))
        var_tr = float(np.var(trend + remainder))
        var_sr = float(np.var(seasonal + remainder))
        trend_strength = max(0.0, 1.0 - var_r / max(var_tr, 1e-12))
        seasonal_strength = max(0.0, 1.0 - var_r / max(var_sr, 1e-12))
    else:
        win = max(3, min(7, n // 5))
        trend = pd.Series(y).rolling(win, center=True, min_periods=1).mean().to_numpy()
        seasonal = np.zeros(n)
        remainder = y - trend
        var_r = float(np.var(remainder))
        trend_strength = max(0.0, 1.0 - var_r / max(float(np.var(y)), 1e-12))
        seasonal_strength = 0.0

    # 3. SARIMA Diagnóstico + Backtest + Previsão
    arima_out = _fit_sarima_and_backtest(
        y=y,
        dates=t_dates,
        seasonal_period=s,
        auto_arima=auto_arima,
        manual_order=manual_order,
        manual_seasonal=manual_seasonal,
        forecast_horizon=forecast_horizon,
        suggested_d=consensus["suggested_d"],
        seasonal_strength=seasonal_strength,
    )
    resid = arima_out["diag_residuals"]

    # 4. Testes de Autocorrelação (Série e Resíduos + Curva de p-valores)
    eff_lag = max(2, min(max_lags, max(2, len(resid) // 4)))
    lb_model_df = arima_out["model_df"] if eff_lag > arima_out["model_df"] else 0

    lb_series_df = acorr_ljungbox(y, lags=[eff_lag], boxpierce=True, return_df=True)
    lb_resid_df = acorr_ljungbox(resid, lags=[eff_lag], model_df=lb_model_df, boxpierce=True, return_df=True)

    bp_stat = _sf(lb_resid_df["bp_stat"].iloc[0], 4)
    bp_p = _sf(lb_resid_df["bp_pvalue"].iloc[0], 4)
    lb_stat = _sf(lb_resid_df["lb_stat"].iloc[0], 4)
    lb_p = _sf(lb_resid_df["lb_pvalue"].iloc[0], 4)
    lb_series_p = _sf(lb_series_df["lb_pvalue"].iloc[0], 4)
    dw_stat = _sf(durbin_watson(resid), 4)

    curve_max = min(24, max(6, len(resid) // 4))
    lb_curve_df = acorr_ljungbox(resid, lags=list(range(1, curve_max + 1)), return_df=True)
    lb_curve = [
        {"lag": int(lg), "lb_p": _sf(lb_curve_df.loc[lg, "lb_pvalue"], 4), "alpha": alpha}
        for lg in range(1, curve_max + 1)
    ]

    # 5. Heterocedasticidade (Engle ARCH-LM & Breusch-Pagan Temporal)
    arch_resid = _safe_arch(resid, lags=eff_lag)
    arch_series = _safe_arch(y, lags=eff_lag)
    bp_temp = _temporal_heteroskedasticity_test(y)

    # 6. Normalidade (Anderson-Darling, Jarque-Bera, Shapiro-Wilk)
    ad_resid = _anderson_darling_normal(resid)
    jb_stat, jb_p = stats.jarque_bera(resid)
    shapiro_sample = resid[:4500] if len(resid) > 4500 else resid
    sw_stat, sw_p = stats.shapiro(shapiro_sample)

    # 7. Estatísticas Descritivas & Outliers (IQR + Z-MAD Robusto)
    q1, med, q3 = float(np.percentile(y, 25)), float(np.median(y)), float(np.percentile(y, 75))
    iqr = q3 - q1
    iqr_low, iqr_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mad = float(np.median(np.abs(y - med)))
    z_mad = 0.6745 * (y - med) / max(mad, 1e-9)

    is_iqr_out = (y < iqr_low) | (y > iqr_high)
    is_mad_out = np.abs(z_mad) > 3.5
    is_any_out = is_iqr_out | is_mad_out

    outlier_rows = []
    for idx in np.where(is_any_out)[0]:
        methods = []
        if is_iqr_out[idx]:
            methods.append("IQR 1,5×")
        if is_mad_out[idx]:
            methods.append("Z-MAD > 3,5")
        outlier_rows.append({
            "date": str(pd.Timestamp(t_dates.iloc[idx]).strftime("%Y-%m-%d")),
            "value": _sf(y[idx], 4),
            "z_mad": _sf(z_mad[idx], 2),
            "methods": " + ".join(methods),
        })
    outlier_rows = sorted(outlier_rows, key=lambda r: abs(r["z_mad"] or 0.0), reverse=True)[:25]

    desc_stats = {
        "n": int(n),
        "mean": _sf(np.mean(y), 4),
        "std": _sf(np.std(y, ddof=1), 4),
        "min": _sf(np.min(y), 4),
        "q1": _sf(q1, 4),
        "median": _sf(med, 4),
        "q3": _sf(q3, 4),
        "max": _sf(np.max(y), 4),
        "iqr": _sf(iqr, 4),
        "cv_pct": _sf((np.std(y, ddof=1) / max(abs(np.mean(y)), 1e-9)) * 100.0, 2),
        "skewness": _sf(stats.skew(y), 4),
        "kurtosis": _sf(stats.kurtosis(y), 4),
        "resid_skewness": _sf(stats.skew(resid), 4),
        "resid_kurtosis": _sf(stats.kurtosis(resid), 4),
    }

    # 8. ACF & PACF + Heurística Box-Jenkins
    nlags_corr = min(24, max(8, n // 3 - 1))
    acf_vals = acf(y, nlags=nlags_corr, fft=True)
    pacf_vals = pacf(y, nlags=nlags_corr, method="ywm")
    conf_bound = 1.96 / math.sqrt(n)
    acf_pacf_rows = [
        {
            "lag": i,
            "acf": _sf(acf_vals[i], 4),
            "pacf": _sf(pacf_vals[i], 4) if i > 0 else 1.0,
            "upper": _sf(conf_bound, 4),
            "lower": _sf(-conf_bound, 4),
            "acf_sig": bool(i > 0 and abs(acf_vals[i]) > conf_bound),
            "pacf_sig": bool(i > 0 and abs(pacf_vals[i]) > conf_bound),
        }
        for i in range(1, nlags_corr + 1)
    ]

    sig_acf = [r["lag"] for r in acf_pacf_rows if r["acf_sig"]]
    sig_pacf = [r["lag"] for r in acf_pacf_rows if r["pacf_sig"]]
    slow_decay = len([lg for lg in sig_acf if lg <= 8]) >= 6
    seas_spike = s >= 2 and s in sig_acf
    if slow_decay:
        bj_hint = "Decaimento lento na ACF indica forte persistência / raiz unitária (recomenda-se d ≥ 1)."
    elif len(sig_pacf) <= 3 and len(sig_acf) > len(sig_pacf):
        bj_hint = f"PACF corta após o(s) lag(s) {sig_pacf[:3]} enquanto ACF decai → assinatura compatível com AR({max(sig_pacf[:3]) if sig_pacf else 1})."
    elif len(sig_acf) <= 3 and len(sig_pacf) > len(sig_acf):
        bj_hint = f"ACF corta após o(s) lag(s) {sig_acf[:3]} → assinatura compatível com MA({max(sig_acf[:3]) if sig_acf else 1})."
    elif not sig_acf and not sig_pacf:
        bj_hint = "Nenhuma autocorrelação significativa fora das bandas de 95% → comportamento compatível com Ruído Branco."
    else:
        bj_hint = "Ambas ACF e PACF apresentam picos significativos → assinatura compatível com processo misto ARMA(p,q)."
    if seas_spike:
        bj_hint += f" Pico expressivo detectado no lag sazonal s = {s}."

    # 9. Tabela Resumo Unificada dos 10 Testes Estatísticos
    def _mk_test(
        tid: str,
        cat: str,
        name: str,
        target_series: str,
        stat: Optional[float],
        pval: Optional[float],
        crit5: Optional[float],
        h0: str,
        ok_cond: bool,
        ok_label: str,
        fail_label: str,
        interp_ok: str,
        interp_fail: str,
        extra: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "id": tid,
            "category": cat,
            "name": name,
            "target_series": target_series,
            "statistic": stat,
            "p_value": pval,
            "p_formatted": _fmt_p(pval),
            "critical_5pct": crit5,
            "h0": h0,
            "passed": bool(ok_cond),
            "status_label": ok_label if ok_cond else fail_label,
            "interpretation": interp_ok if ok_cond else interp_fail,
            "extra_note": extra,
        }

    tests_summary = [
        _mk_test(
            "adf", "Estacionariedade", f"Dickey-Fuller Aumentado (ADF — {regression})", t_meta["label"],
            adf_main["statistic"], adf_main["p_value"], adf_main["crit_5"],
            "Existe raiz unitária (série não-estacionária)",
            adf_main["p_value"] is not None and adf_main["p_value"] < alpha,
            "Estacionária", "Não-Estacionária",
            f"Rejeita H₀ (p = {_fmt_p(adf_main['p_value'])} < {alpha}): ausência de raiz unitária (lags AIC = {adf_main['lags']}).",
            f"Não rejeita H₀ (p = {_fmt_p(adf_main['p_value'])} ≥ {alpha}): evidência de raiz unitária (lags AIC = {adf_main['lags']}).",
            extra=f"Lags AIC: {adf_main['lags']}",
        ),
        _mk_test(
            "pp", "Estacionariedade", f"Phillips-Perron (PP — {regression})", t_meta["label"],
            pp_main["statistic"], pp_main["p_value"], pp_main["crit_5"],
            "Existe raiz unitária (robusto HAC Newey-West)",
            pp_main["p_value"] is not None and pp_main["p_value"] < alpha,
            "Estacionária", "Não-Estacionária",
            f"Rejeita H₀ (Zₜ = {pp_main['statistic']}, p = {_fmt_p(pp_main['p_value'])}): estacionariedade confirmada com correção HAC.",
            f"Não rejeita H₀ (Zₜ = {pp_main['statistic']}, p = {_fmt_p(pp_main['p_value'])}): raiz unitária mesmo após correção Newey-West.",
            extra=f"Bandwidth Bartlett: {pp_main['lags']}",
        ),
        _mk_test(
            "kpss", "Estacionariedade", f"KPSS ({regression})", t_meta["label"],
            kpss_main["statistic"], kpss_main["p_value"], kpss_main["crit_5"],
            "A série É estacionária (H₀ invertida)",
            kpss_main["p_value"] is not None and kpss_main["p_value"] >= alpha,
            "Estacionária", "Não-Estacionária",
            f"Não rejeita H₀ (p = {_fmt_p(kpss_main['p_value'])} ≥ {alpha}): consistente com estacionariedade.",
            f"Rejeita H₀ (p = {_fmt_p(kpss_main['p_value'])} < {alpha}): rejeita estacionariedade em favor de tendência estocástica.",
            extra=kpss_main.get("note"),
        ),
        _mk_test(
            "zivot_andrews", "Quebra Estrutural", "Zivot-Andrews (Quebra Endógena)", t_meta["label"],
            za_main["statistic"], za_main["p_value"], za_main["crit_5"],
            "Raiz unitária mesmo permitindo 1 quebra estrutural",
            za_main["p_value"] is not None and za_main["p_value"] < alpha,
            f"Quebra em {za_main['break_date'] or '—'}", "Raiz Unitária Persistente",
            f"Rejeita raiz unitária (p = {_fmt_p(za_main['p_value'])}) ao acomodar quebra estrutural endógena em {za_main['break_date']}.",
            f"Não rejeita raiz unitária (p = {_fmt_p(za_main['p_value'])}); principal ponto candidato de quebra: {za_main['break_date']}.",
            extra=f"Data de quebra: {za_main['break_date']}",
        ),
        _mk_test(
            "ljung_box", "Autocorrelação", f"Ljung-Box (Q*, lag={eff_lag})", f"Resíduos {arima_out['spec_label']}",
            lb_stat, lb_p, None,
            "Ausência de autocorrelação nos resíduos (Ruído Branco)",
            lb_p is not None and lb_p >= alpha,
            "Resíduos Limpos", "Autocorrelação Residual",
            f"Não rejeita H₀ (p = {_fmt_p(lb_p)} ≥ {alpha}): o modelo capturou adequadamente a dinâmica temporal.",
            f"Rejeita H₀ (p = {_fmt_p(lb_p)} < {alpha}): ainda resta dependência temporal nos resíduos (ajustar p, q ou parte sazonal).",
            extra=f"Ajuste model_df={lb_model_df} | Série bruta p={_fmt_p(lb_series_p)}",
        ),
        _mk_test(
            "box_pierce", "Autocorrelação", f"Box-Pierce (Q, lag={eff_lag})", f"Resíduos {arima_out['spec_label']}",
            bp_stat, bp_p, None,
            "Ausência de autocorrelação serial nos resíduos",
            bp_p is not None and bp_p >= alpha,
            "Resíduos Limpos", "Autocorrelação Residual",
            f"Não rejeita H₀ (p = {_fmt_p(bp_p)} ≥ {alpha}): sem autocorrelação significativa até o lag {eff_lag}.",
            f"Rejeita H₀ (p = {_fmt_p(bp_p)} < {alpha}): dependência serial detectada nos resíduos.",
            extra=f"Durbin-Watson = {dw_stat}",
        ),
        _mk_test(
            "arch_lm", "Heterocedasticidade", f"Engle ARCH-LM (lag={arch_resid['lags']})", f"Resíduos {arima_out['spec_label']}",
            arch_resid["statistic"], arch_resid["p_value"], None,
            "Ausência de efeitos ARCH (variância condicional constante)",
            arch_resid["p_value"] is not None and arch_resid["p_value"] >= alpha,
            "Homocedástica", "Efeito ARCH / GARCH",
            f"Não rejeita H₀ (p = {_fmt_p(arch_resid['p_value'])} ≥ {alpha}): ausência de clusters de volatilidade condicional.",
            f"Rejeita H₀ (p = {_fmt_p(arch_resid['p_value'])} < {alpha}): forte clusterização de volatilidade → considere modelos GARCH.",
            extra=f"Série direta ARCH p = {_fmt_p(arch_series['p_value'])}",
        ),
        _mk_test(
            "bp_temporal", "Heterocedasticidade", "Breusch-Pagan Temporal (Var vs t)", t_meta["label"],
            bp_temp["statistic"], bp_temp["p_value"], None,
            "Variância incondicional estável ao longo do tempo",
            bp_temp["p_value"] is not None and bp_temp["p_value"] >= alpha,
            "Variância Estável", "Variância Crescente",
            f"Não rejeita H₀ (p = {_fmt_p(bp_temp['p_value'])} ≥ {alpha}): dispersão não varia sistematicamente com o tempo.",
            f"Rejeita H₀ (p = {_fmt_p(bp_temp['p_value'])} < {alpha}): dispersão varia com o nível/tempo → testar Log ou Box-Cox.",
            extra=f"R² auxiliar = {bp_temp.get('r2')}",
        ),
        _mk_test(
            "anderson_darling", "Normalidade", "Anderson-Darling (Resíduos)", f"Resíduos {arima_out['spec_label']}",
            ad_resid["statistic"], ad_resid["p_value"], ad_resid["crit_5"],
            "Os resíduos seguem distribuição Normal (sensível a caudas)",
            ad_resid["p_value"] is not None and ad_resid["p_value"] >= alpha,
            "Resíduos Normais", "Não-Normais (Caudas)",
            f"Não rejeita normalidade (A² = {ad_resid['statistic']}, p = {_fmt_p(ad_resid['p_value'])}): intervalos de confiança válidos.",
            f"Rejeita normalidade (A² = {ad_resid['statistic']}, p = {_fmt_p(ad_resid['p_value'])}): caudas pesadas ou outliers (use bootstrap / t-Student).",
            extra=f"Crítico 5%: {ad_resid['crit_5']}",
        ),
        _mk_test(
            "jarque_bera", "Normalidade", "Jarque-Bera & Shapiro-Wilk", f"Resíduos {arima_out['spec_label']}",
            _sf(jb_stat, 4), _sf(jb_p, 4), 5.9915,
            "Assimetria = 0 e Excesso de Curtose = 0 (Gaussianidade)",
            jb_p >= alpha,
            "Gaussianos", "Desvio de Normalidade",
            f"Não rejeita H₀ (JB p = {_fmt_p(_sf(jb_p, 4))} | Shapiro p = {_fmt_p(_sf(sw_p, 4))}): boa aderência Gaussiana.",
            f"Rejeita H₀ (JB p = {_fmt_p(_sf(jb_p, 4))} | Shapiro p = {_fmt_p(_sf(sw_p, 4))}): assimetria ({desc_stats['resid_skewness']}) / curtose ({desc_stats['resid_kurtosis']}).",
            extra=f"Shapiro-Wilk W = {_sf(sw_stat, 4)} (p = {_fmt_p(_sf(sw_p, 4))})",
        ),
    ]

    # 10. Matriz Comparativa das 7 Transformações
    transform_matrix = _compute_transform_matrix(raw_dates, raw_vals, s, regression, alpha, max_lags)

    # 11. Rolling Mean & Rolling Std + Série Completa + STL
    roll_w = max(4, min(s if s >= 4 else 12, n // 4))
    roll_mean = pd.Series(y).rolling(roll_w, min_periods=max(2, roll_w // 2)).mean().to_numpy()
    roll_std = pd.Series(y).rolling(roll_w, min_periods=max(2, roll_w // 2)).std().to_numpy()

    raw_align = raw_vals[-n:]
    fitted_arr = arima_out["fitted_values"]
    raw_resid_arr = arima_out["raw_residuals"]

    series_table = [
        {
            "date": str(pd.Timestamp(t_dates.iloc[i]).strftime("%Y-%m-%d")),
            "raw_value": _sf(raw_align[i], 4),
            "value": _sf(y[i], 4),
            "rolling_mean": _sf(roll_mean[i], 4),
            "rolling_std": _sf(roll_std[i], 4),
            "stl_trend": _sf(trend[i], 4),
            "stl_seasonal": _sf(seasonal[i], 4),
            "stl_remainder": _sf(remainder[i], 4),
            "fitted": _sf(fitted_arr[i], 4) if i < len(fitted_arr) else None,
            "residual": _sf(raw_resid_arr[i], 4) if i < len(raw_resid_arr) else None,
            "is_outlier": bool(is_any_out[i]),
            "outlier_value": _sf(y[i], 4) if is_any_out[i] else None,
        }
        for i in range(n)
    ]

    return {
        "meta": {
            "dataset_id": dataset_id,
            "dataset_title": dataset_title,
            "transform": transform,
            "transform_label": t_meta["label"],
            "boxcox_lambda": t_meta["boxcox_lambda"],
            "n_original": int(len(raw_vals)),
            "n_analyzed": int(n),
            "date_start": str(pd.Timestamp(t_dates.iloc[0]).strftime("%Y-%m-%d")),
            "date_end": str(pd.Timestamp(t_dates.iloc[-1]).strftime("%Y-%m-%d")),
            "seasonal_period": s,
            "alpha": alpha,
            "regression": regression,
            "max_lags": eff_lag,
            "rolling_window": roll_w,
        },
        "stationarity_consensus": consensus,
        "adf_specifications": {
            "c": adf_c,
            "ct": adf_ct,
            "n": adf_n,
        },
        "kpss_specifications": {
            "c": kpss_c,
            "ct": kpss_ct,
        },
        "zivot_andrews": za_main,
        "tests_summary": tests_summary,
        "ljung_box_curve": lb_curve,
        "durbin_watson": dw_stat,
        "acf_pacf": acf_pacf_rows,
        "box_jenkins_hint": bj_hint,
        "transform_comparison": transform_matrix,
        "descriptive_stats": desc_stats,
        "outliers": {
            "iqr_low": _sf(iqr_low, 4),
            "iqr_high": _sf(iqr_high, 4),
            "iqr_count": int(np.sum(is_iqr_out)),
            "mad_count": int(np.sum(is_mad_out)),
            "total_flagged": int(np.sum(is_any_out)),
            "top_outliers": outlier_rows,
        },
        "distributions": {
            "values_histogram": _histogram_with_normal(y, bins=18),
            "residuals_histogram": _histogram_with_normal(resid, bins=18),
            "residuals_qq": _qq_normal_points(resid),
        },
        "stl": {
            "trend_strength": _sf(trend_strength, 4),
            "seasonal_strength": _sf(seasonal_strength, 4),
            "has_seasonality": bool(s >= 2 and n >= 2 * s + 2),
        },
        "arima": {
            "spec_label": arima_out["spec_label"],
            "order": arima_out["order"],
            "seasonal_order": arima_out["seasonal_order"],
            "aic": arima_out["aic"],
            "aicc": arima_out["aicc"],
            "bic": arima_out["bic"],
            "forecast": arima_out["forecast"],
            "backtest": arima_out["backtest"],
        },
        "series_table": series_table,
    }
