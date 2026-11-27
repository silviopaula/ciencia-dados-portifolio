"""
Core Data Processing, Outliers, Seasonality, XmR, and Multi-Resolution Growth Analytics
for the Unified Time Series Analytics & Forecasting Platform.
"""
import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

MESES_PT = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]

MESES_ABBR_PT = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez"
]


def safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val))):
        return None
    if pd.isna(val):
        return None
    try:
        return round(float(val), 4)
    except Exception:
        return None


def parse_date_column(series: pd.Series, date_format: str = "auto") -> pd.Series:
    """Safely converts a column to datetime standardized to YYYY-MM-DD (or 1st of month when monthly)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    if pd.api.types.is_numeric_dtype(series):
        # Excel serial date fallback
        return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")

    fmt_map = {
        "ymd": "%Y-%m-%d",
        "dmy": "%d/%m/%Y",
        "mdy": "%m/%d/%Y",
        "ydm": "%Y-%d-%m",
    }
    if date_format in fmt_map:
        parsed = pd.to_datetime(series, format=fmt_map[date_format], errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed

    # Fallback flexible parsing
    dayfirst = date_format == "dmy"
    return pd.to_datetime(series, dayfirst=dayfirst, errors="coerce")


def compute_descriptive_stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return {
            "count": 0, "mean": None, "median": None, "std": None,
            "min": None, "max": None, "skewness": None, "kurtosis": None
        }
    return {
        "count": int(len(clean)),
        "mean": safe_float(np.mean(clean)),
        "median": safe_float(np.median(clean)),
        "std": safe_float(np.std(clean, ddof=1) if len(clean) > 1 else 0.0),
        "min": safe_float(np.min(clean)),
        "max": safe_float(np.max(clean)),
        "skewness": safe_float(stats.skew(clean, bias=False) if len(clean) > 2 else 0.0),
        "kurtosis": safe_float(stats.kurtosis(clean, fisher=False, bias=False) if len(clean) > 3 else 0.0),
    }


# ==============================================================================
# MODULE 1: OUTLIER IDENTIFICATION & CORRECTION
# ==============================================================================

def identify_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "zscore",
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Identifies outliers in `column` using one of 6 statistical/ML methods:
    - zscore (default threshold 3.0)
    - iqr (1.5 * IQR)
    - mahalanobis (multivariate if >1 numeric col, else univariate, chi2 0.95)
    - isolation_forest (default threshold 0.6 on anomaly score)
    - lof (Local Outlier Factor, default threshold 1.5)
    - svm (One-Class SVM, nu=0.05)
    """
    x = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    n = len(x)
    is_outlier = np.zeros(n, dtype=bool)
    na_mask = np.isnan(x)
    valid_idx = np.where(~na_mask)[0]
    x_clean = x[valid_idx]

    if len(x_clean) < 4:
        return is_outlier

    if method == "zscore":
        thresh = threshold if threshold is not None else 3.0
        std_val = np.std(x_clean, ddof=1)
        if std_val > 0:
            z_scores = np.abs((x_clean - np.mean(x_clean)) / std_val)
            is_outlier[valid_idx] = z_scores > thresh

    elif method == "iqr":
        mult = threshold if threshold is not None else 1.5
        q1, q3 = np.percentile(x_clean, [25, 75])
        iqr = q3 - q1
        lower = q1 - mult * iqr
        upper = q3 + mult * iqr
        is_outlier[valid_idx] = (x_clean < lower) | (x_clean > upper)

    elif method == "mahalanobis":
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if df[c].notna().sum() > 3
        ]
        multivariate_computed = False
        if len(num_cols) > 1:
            num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
            complete_mask = num_df.notna().all(axis=1).to_numpy()
            if complete_mask.sum() > len(num_cols) + 2:
                mat = num_df.loc[complete_mask].to_numpy(dtype=float)
                cov_mat = np.cov(mat, rowvar=False)
                cov_mat += np.eye(cov_mat.shape[0]) * 1e-4
                try:
                    cov_inv = np.linalg.inv(cov_mat)
                    means = np.mean(mat, axis=0)
                    diff = mat - means
                    md = np.sum(diff @ cov_inv * diff, axis=1)
                    chi_thresh = stats.chi2.ppf(0.95, df=mat.shape[1])
                    is_outlier[complete_mask] = md > chi_thresh
                    multivariate_computed = True
                except Exception:
                    multivariate_computed = False
        if not multivariate_computed:
            var_val = np.var(x_clean, ddof=1)
            if var_val > 0:
                md = ((x_clean - np.mean(x_clean)) ** 2) / var_val
                chi_thresh = stats.chi2.ppf(0.95, df=1)
                is_outlier[valid_idx] = md > chi_thresh

    elif method == "isolation_forest":
        thresh = threshold if threshold is not None else 0.6
        # Map R isotree score threshold (~0.5-0.8) to decision function
        iso = IsolationForest(n_estimators=100, random_state=42)
        X_mat = x_clean.reshape(-1, 1)
        iso.fit(X_mat)
        # score_samples returns negative values; convert to 0..1 anomaly score like isotree
        raw_scores = -iso.score_samples(X_mat)
        is_outlier[valid_idx] = raw_scores > thresh

    elif method == "lof":
        thresh = threshold if threshold is not None else 1.5
        k = min(30, max(2, len(x_clean) - 1))
        lof = LocalOutlierFactor(n_neighbors=k)
        X_mat = x_clean.reshape(-1, 1)
        lof.fit(X_mat)
        lof_scores = -lof.negative_outlier_factor_
        is_outlier[valid_idx] = lof_scores > thresh

    elif method == "svm":
        nu_val = min(0.5, max(0.01, threshold if threshold is not None else 0.05))
        svm = OneClassSVM(nu=nu_val, kernel="rbf", gamma="scale")
        X_mat = x_clean.reshape(-1, 1)
        preds = svm.fit_predict(X_mat)
        is_outlier[valid_idx] = preds == -1

    is_outlier[na_mask] = False
    return is_outlier


def correct_outliers(
    series: np.ndarray,
    is_outlier: np.ndarray,
    method: str = "linear_interp",
    window_size: int = 5,
    period: int = 12
) -> np.ndarray:
    """
    Corrects identified outliers using one of 7 methods from the original R app:
    - mean
    - median
    - moving_average (centered rolling mean + linear interpolation fallback)
    - linear_interp
    - winsorize (5th and 95th percentiles of non-outlier data)
    - arima (SARIMAX / auto-arima fitted values on non-outlier observations)
    - seasonal_decomp (STL trend component replacement)
    """
    x = np.array(series, dtype=float).copy()
    corrected = x.copy()
    out_mask = np.array(is_outlier, dtype=bool) & (~np.isnan(x))

    if not out_mask.any():
        return corrected

    valid_non_outliers = x[(~out_mask) & (~np.isnan(x))]
    if len(valid_non_outliers) == 0:
        return corrected

    if method == "mean":
        corrected[out_mask] = np.mean(valid_non_outliers)

    elif method == "median":
        corrected[out_mask] = np.median(valid_non_outliers)

    elif method == "moving_average":
        w = max(2, min(int(window_size), len(x) - 1))
        temp_s = pd.Series(x)
        temp_s[out_mask] = np.nan
        ma = temp_s.rolling(window=w, center=True, min_periods=1).mean()
        ma = ma.interpolate(method="linear").bfill().ffill()
        corrected[out_mask] = ma.to_numpy()[out_mask]

    elif method == "linear_interp":
        temp_s = pd.Series(x)
        temp_s[out_mask] = np.nan
        interp = temp_s.interpolate(method="linear").bfill().ffill()
        corrected[out_mask] = interp.to_numpy()[out_mask]

    elif method == "winsorize":
        q05, q95 = np.percentile(valid_non_outliers, [5, 95])
        low_mask = out_mask & (x < q05)
        high_mask = out_mask & (x > q95)
        mid_mask = out_mask & (~low_mask) & (~high_mask)
        corrected[low_mask] = q05
        corrected[high_mask] = q95
        if mid_mask.any():
            corrected[mid_mask] = np.median(valid_non_outliers)

    elif method == "arima":
        temp_s = pd.Series(x)
        temp_s[out_mask] = np.nan
        fallback_interp = temp_s.interpolate(method="linear").bfill().ffill().to_numpy()
        try:
            # Fit state space SARIMAX(1,1,1) which natively interpolates missing observations via Kalman filter
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(temp_s.to_numpy(), order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
            fitted = np.asarray(res.fittedvalues, dtype=float)
            # Guard against d=1 burn-in zero artifact at leading index (index 0 / before first non-outlier)
            first_valid_pos = int(np.where((~out_mask) & (~np.isnan(x)))[0][0])
            burn_in_mask = np.arange(len(x)) <= max(0, first_valid_pos)
            bad_fit = (~np.isfinite(fitted)) | burn_in_mask
            fitted_safe = np.where(bad_fit, fallback_interp, fitted)
            corrected[out_mask] = fitted_safe[out_mask]
        except Exception:
            corrected[out_mask] = fallback_interp[out_mask]

    elif method == "seasonal_decomp":
        try:
            temp_s = pd.Series(x).interpolate(method="linear").bfill().ffill()
            p = max(2, period if len(temp_s) >= period * 2 else 4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stl = STL(temp_s.to_numpy(), period=p, robust=True).fit()
            corrected[out_mask] = stl.trend[out_mask]
        except Exception:
            corrected[out_mask] = np.mean(valid_non_outliers)

    return corrected


# ==============================================================================
# MODULE 2: SEASONALITY TESTS, PARTICIPATION, XmR & STL ANOMALIES
# ==============================================================================

def run_seasonality_and_spc_analysis(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    stl_threshold_sigma: float = 2.0
) -> Dict[str, Any]:
    """
    Runs comprehensive seasonality and statistical process control diagnostics:
    1. Monthly participation & control limits (+/- 1 SD) over [start_year, end_year]
    2. Yearly monthly participation breakdown across all years
    3. XmR Statistical Process Control Chart (Individual X and Moving Range MR with 2.66*MR_bar and 3.0*MR_bar)
    4. STL Decomposition (Trend, Seasonal, Remainder) & Anomaly detection on STL residuals
    5. Statistical Seasonality F-Test / Kruskal-Wallis test across months
    """
    work = df[[date_col, value_col]].copy()
    work["Data"] = pd.to_datetime(work[date_col], errors="coerce")
    work["Valor"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["Data", "Valor"]).sort_values("Data").reset_index(drop=True)

    if len(work) < 2:
        raise ValueError("São necessárias pelo menos 2 observações válidas para a análise.")

    work["Ano"] = work["Data"].dt.year
    work["Mes_Num"] = work["Data"].dt.month
    work["Mes"] = work["Mes_Num"].apply(lambda m: MESES_PT[m - 1])

    all_years = sorted(work["Ano"].unique().tolist())
    s_yr = int(start_year) if start_year is not None else min(all_years)
    e_yr = int(end_year) if end_year is not None else max(all_years)

    # 1. All-years monthly participation
    yearly_month = (
        work.groupby(["Ano", "Mes_Num", "Mes"], as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "Total_Mes"})
    )
    year_totals = yearly_month.groupby("Ano")["Total_Mes"].transform("sum")
    yearly_month["Participacao"] = np.where(
        year_totals != 0,
        np.round(yearly_month["Total_Mes"] / year_totals, 5),
        0.0
    )

    # 2. Limits filtered by [s_yr, e_yr]
    filt = yearly_month[(yearly_month["Ano"] >= s_yr) & (yearly_month["Ano"] <= e_yr)].copy()
    limits_rows = []
    for m_num in range(1, 13):
        m_name = MESES_PT[m_num - 1]
        sub = filt[filt["Mes_Num"] == m_num]["Participacao"].dropna()
        media_p = float(np.round(sub.mean(), 5)) if len(sub) > 0 else 0.0
        sd_p = float(np.round(sub.std(ddof=1), 5)) if len(sub) > 1 else 0.0
        row_dict: Dict[str, Any] = {
            "Mes_Num": m_num,
            "Mes": m_name,
            "Media_Participacao": media_p,
            "Desvio_Padrao": sd_p,
            "Limite_Superior": float(np.round(media_p + sd_p, 5)),
            "Limite_Inferior": float(np.round(media_p - sd_p, 5)),
        }
        # Pivot each year's participation into the row
        for yr in all_years:
            val_yr = yearly_month[(yearly_month["Ano"] == yr) & (yearly_month["Mes_Num"] == m_num)]["Participacao"]
            row_dict[str(yr)] = safe_float(val_yr.iloc[0]) if len(val_yr) > 0 else None
        limits_rows.append(row_dict)

    # Series by year for plotting
    yearly_traces = []
    for yr in all_years:
        sub_yr = yearly_month[yearly_month["Ano"] == yr].sort_values("Mes_Num")
        yearly_traces.append({
            "year": int(yr),
            "months": sub_yr["Mes_Num"].tolist(),
            "month_names": sub_yr["Mes"].tolist(),
            "participations": [safe_float(v) for v in sub_yr["Participacao"].tolist()],
            "totals": [safe_float(v) for v in sub_yr["Total_Mes"].tolist()]
        })

    # 3. XmR Chart Data & Control Limits
    work["MR"] = work["Valor"].diff().abs()
    xmr_df = work.dropna(subset=["MR"]).copy()
    media_x = float(xmr_df["Valor"].mean())
    media_mr = float(xmr_df["MR"].mean())
    ucl_x = media_x + 2.66 * media_mr
    lcl_x = media_x - 2.66 * media_mr
    ucl_mr = 3.0 * media_mr
    lcl_mr = 0.0

    xmr_points = []
    for _, r in xmr_df.iterrows():
        x_val = float(r["Valor"])
        mr_val = float(r["MR"])
        xmr_points.append({
            "date": r["Data"].strftime("%Y-%m-%d"),
            "x": safe_float(x_val),
            "mr": safe_float(mr_val),
            "x_violation": bool(x_val > ucl_x or x_val < lcl_x),
            "mr_violation": bool(mr_val > ucl_mr),
        })

    xmr_summary = {
        "media_x": safe_float(media_x),
        "ucl_x": safe_float(ucl_x),
        "lcl_x": safe_float(lcl_x),
        "media_mr": safe_float(media_mr),
        "ucl_mr": safe_float(ucl_mr),
        "lcl_mr": safe_float(lcl_mr),
        "x_violations_count": sum(1 for p in xmr_points if p["x_violation"]),
        "mr_violations_count": sum(1 for p in xmr_points if p["mr_violation"]),
        "points": xmr_points,
    }

    # 4. STL Residuals Anomaly Detection
    stl_anomalies = []
    stl_series = []
    seasonality_strength = None
    if len(work) >= 12:
        period = 12 if len(work) >= 24 else max(2, len(work) // 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl_res = STL(work["Valor"].to_numpy(), period=period, robust=True).fit()
        rem = stl_res.resid
        trend = stl_res.trend
        seas = stl_res.seasonal
        rem_sd = float(np.std(rem, ddof=1))
        thresh = stl_threshold_sigma * rem_sd

        # Seasonality strength: max(0, 1 - Var(R) / Var(S + R))
        var_sr = np.var(seas + rem, ddof=1)
        if var_sr > 0:
            seasonality_strength = safe_float(max(0.0, 1.0 - (np.var(rem, ddof=1) / var_sr)))

        for idx, r in work.iterrows():
            is_anom = bool(abs(rem[idx]) > thresh)
            item = {
                "date": r["Data"].strftime("%Y-%m-%d"),
                "value": safe_float(r["Valor"]),
                "trend": safe_float(trend[idx]),
                "seasonal": safe_float(seas[idx]),
                "remainder": safe_float(rem[idx]),
                "upper_bound": safe_float(thresh),
                "lower_bound": safe_float(-thresh),
                "is_anomaly": is_anom,
            }
            stl_series.append(item)
            if is_anom:
                stl_anomalies.append(item)

    # 5. Kruskal-Wallis & ANOVA Seasonality Test across months
    month_groups = [g["Valor"].dropna().to_numpy() for _, g in work.groupby("Mes_Num") if len(g) > 0]
    kw_stat, kw_p = (None, None)
    f_stat, f_p = (None, None)
    if len(month_groups) >= 2 and all(len(g) >= 1 for g in month_groups):
        try:
            if any(len(g) > 1 for g in month_groups):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f_res = stats.f_oneway(*month_groups)
                    f_stat, f_p = safe_float(f_res.statistic), safe_float(f_res.pvalue)
                    kw_res = stats.kruskal(*month_groups)
                    kw_stat, kw_p = safe_float(kw_res.statistic), safe_float(kw_res.pvalue)
        except Exception:
            pass

    return {
        "years_available": all_years,
        "start_year": s_yr,
        "end_year": e_yr,
        "limits_table": limits_rows,
        "yearly_traces": yearly_traces,
        "xmr": xmr_summary,
        "stl": {
            "threshold_sigma": stl_threshold_sigma,
            "anomalies_count": len(stl_anomalies),
            "anomalies": stl_anomalies,
            "series": stl_series,
            "seasonality_strength": seasonality_strength,
        },
        "seasonality_tests": {
            "anova_f": f_stat,
            "anova_p": f_p,
            "kruskal_h": kw_stat,
            "kruskal_p": kw_p,
            "has_significant_seasonality": bool((f_p is not None and f_p < 0.05) or (kw_p is not None and kw_p < 0.05)),
        },
    }


# ==============================================================================
# MODULE 3: MULTI-RESOLUTION GROWTH & VARIATION ENGINE
# ==============================================================================

def generate_multi_resolution_views(
    df: pd.DataFrame,
    date_col: str,
    value_col: str
) -> Dict[str, Any]:
    """
    Aggregates time series into multiple temporal resolutions:
    - Mensal (Jan..Dez)
    - Trimestral (Q1..Q4)
    - Semestral (S1, S2)
    - Anual (Valor_Anual)
    Computes YoY Growth Rates (%) and YoY Absolute Variations for both wide tables and long series,
    plus MoM (month-over-month) and YoY (month-over-year) comparisons for the Advanced Editor.
    """
    work = df[[date_col, value_col]].copy()
    work["Data"] = pd.to_datetime(work[date_col], errors="coerce")
    work["Valor"] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["Data", "Valor"]).sort_values("Data").reset_index(drop=True)

    if len(work) == 0:
        return {
            "mensal": {"wide_valores": [], "wide_taxas": [], "wide_variacoes": [], "long": []},
            "trimestral": {"wide_valores": [], "wide_taxas": [], "wide_variacoes": [], "long": []},
            "semestral": {"wide_valores": [], "wide_taxas": [], "wide_variacoes": [], "long": []},
            "anual": {"wide_valores": [], "wide_taxas": [], "wide_variacoes": [], "long": []},
        }

    work["Ano"] = work["Data"].dt.year
    work["Mes_Num"] = work["Data"].dt.month
    work["Mes_Label"] = work["Mes_Num"].apply(lambda m: MESES_ABBR_PT[m - 1])
    work["Quarter_Num"] = work["Data"].dt.quarter
    work["Quarter_Label"] = "Q" + work["Quarter_Num"].astype(str)
    work["Semestre_Num"] = np.where(work["Mes_Num"] <= 6, 1, 2)
    work["Semestre_Label"] = "S" + work["Semestre_Num"].astype(str)

    def _build_period_views(
        period_col: str,
        ordered_labels: List[str],
        label_to_date_fn
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = work.groupby(["Ano", period_col], as_index=False)["Valor"].sum()
        piv = grouped.pivot(index="Ano", columns=period_col, values="Valor")
        for lbl in ordered_labels:
            if lbl not in piv.columns:
                piv[lbl] = np.nan
        piv = piv[ordered_labels].sort_index()

        # YoY growth rate (%) per period column across consecutive years
        piv_growth = piv.pct_change(fill_method=None) * 100.0
        piv_growth = piv_growth.replace([np.inf, -np.inf], np.nan)
        # YoY absolute difference per period column
        piv_diff = piv.diff()

        wide_valores = []
        wide_taxas = []
        wide_variacoes = []
        long_rows = []

        for ano in piv.index:
            row_val: Dict[str, Any] = {"Ano": int(ano)}
            row_tx: Dict[str, Any] = {"Ano": int(ano)}
            row_var: Dict[str, Any] = {"Ano": int(ano)}

            for lbl in ordered_labels:
                v = safe_float(piv.loc[ano, lbl])
                tx = safe_float(piv_growth.loc[ano, lbl])
                dff = safe_float(piv_diff.loc[ano, lbl])
                row_val[lbl] = v
                row_tx[lbl] = tx
                row_var[lbl] = dff

                if v is not None:
                    dt_str = label_to_date_fn(int(ano), lbl)
                    long_rows.append({
                        "Data": dt_str,
                        "Ano": int(ano),
                        "Periodo": lbl,
                        "Valor": v,
                        "Taxa_Crescimento": tx,
                        "Variacao_Valor": dff,
                    })

            wide_valores.append(row_val)
            wide_taxas.append(row_tx)
            wide_variacoes.append(row_var)

        long_rows.sort(key=lambda r: r["Data"])
        return {
            "wide_valores": wide_valores,
            "wide_taxas": wide_taxas,
            "wide_variacoes": wide_variacoes,
            "long": long_rows,
        }

    mensal_views = _build_period_views(
        "Mes_Label",
        MESES_ABBR_PT,
        lambda yr, lbl: f"{yr:04d}-{MESES_ABBR_PT.index(lbl) + 1:02d}-01"
    )
    trimestral_views = _build_period_views(
        "Quarter_Label",
        ["Q1", "Q2", "Q3", "Q4"],
        lambda yr, lbl: f"{yr:04d}-{(int(lbl[1]) - 1) * 3 + 1:02d}-01"
    )
    semestral_views = _build_period_views(
        "Semestre_Label",
        ["S1", "S2"],
        lambda yr, lbl: f"{yr:04d}-{'01' if lbl == 'S1' else '07'}-01"
    )
    work["Anual_Label"] = "Valor_Anual"
    anual_views = _build_period_views(
        "Anual_Label",
        ["Valor_Anual"],
        lambda yr, lbl: f"{yr:04d}-01-01"
    )

    # Also compute MoM (month-over-month) and YoY (lag 12) series for the Editor
    mom_yoy_series = []
    s_df = work.groupby("Data", as_index=False)["Valor"].sum().sort_values("Data").reset_index(drop=True)
    s_df["MoM_Growth"] = s_df["Valor"].pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan) * 100.0
    s_df["MoM_Diff"] = s_df["Valor"].diff(1)
    s_df["YoY_Growth"] = s_df["Valor"].pct_change(12, fill_method=None).replace([np.inf, -np.inf], np.nan) * 100.0
    s_df["YoY_Diff"] = s_df["Valor"].diff(12)

    for _, r in s_df.iterrows():
        mom_yoy_series.append({
            "Data": r["Data"].strftime("%Y-%m-%d"),
            "Valor": safe_float(r["Valor"]),
            "MoM_Growth": safe_float(r["MoM_Growth"]),
            "MoM_Diff": safe_float(r["MoM_Diff"]),
            "YoY_Growth": safe_float(r["YoY_Growth"]),
            "YoY_Diff": safe_float(r["YoY_Diff"]),
        })

    return {
        "mensal": mensal_views,
        "trimestral": trimestral_views,
        "semestral": semestral_views,
        "anual": anual_views,
        "mom_yoy_series": mom_yoy_series,
    }
