"""Motor Econométrico e de Conjuntura do Observatório Macro Brasil Studio.

Implementa:
- Painel de Conjuntura Brasil (Selic, IPCA 12m vs. Meta, DBGG/PIB, Câmbio, IBC-Br, PNAD)
- 6 Modelos Econométricos Dedicados com parâmetros interativos (M01, M02, M06, M12, M13, M14)
- Motor Temático Unificado (VAR+IRF+Granger, Engle-Granger+ECM, Filtro HP, SARIMA, EWMA, MQO-HAC)
  para os demais 24 estudos autorais, com opção de atualização ao vivo pela API SGS/Olinda do BCB.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, coint, grangercausalitytests, pacf

from .catalog import STUDIES_BY_CODE
from .data_store import (
    CORE_CONJUNTURA_SERIES,
    fetch_bcb_olinda_expinf,
    fetch_bcb_sgs,
    get_store_stats,
    load_bcb_series,
    load_study_table,
)

PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}


def _sf(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _parse_pt_month_year(s: str) -> pd.Timestamp:
    raw = str(s).strip().lower()
    parts = raw.split("/")
    if len(parts) == 2:
        m = PT_MONTHS.get(parts[0][:3])
        if m is not None:
            y = int(parts[1])
            if y < 100:
                y += 2000 if y <= 50 else 1900
            return pd.Timestamp(year=y, month=m, day=1)
    return pd.to_datetime(raw, dayfirst=True)


def _adf_summary(series: pd.Series, name: str) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = adfuller(s, autolag="AIC")
    return {
        "series": name,
        "statistic": round(_sf(res[0]), 4),
        "pvalue": round(_sf(res[1], 1.0), 4),
        "lags_used": int(res[2]),
        "nobs": int(res[3]),
        "stationary_5pct": bool(res[1] < 0.05),
    }


# ---------------------------------------------------------------------------
# PAINEL DE CONJUNTURA BRASIL (BCB SGS + DuckDB)
# ---------------------------------------------------------------------------

def get_conjuntura_dashboard(force_live: bool = False) -> Dict[str, Any]:
    """Consolida os 6 grandes termômetros macroeconômicos do Brasil (DuckDB + BCB ao vivo)."""
    t0 = time.perf_counter()
    live_hits = 0

    def _get(sid: int, start: str = "01/01/2016") -> Optional[pd.Series]:
        nonlocal live_hits
        df, is_live = fetch_bcb_sgs(sid, start, force_live=force_live)
        if is_live:
            live_hits += 1
        if df is None or df.empty:
            return None
        return df["value"].resample("MS").mean().dropna()

    selic = _get(4189)
    ipca12 = _get(13522)
    meta = _get(13521)
    dbgg = _get(13762, "01/01/2014")
    cambio = _get(3698)
    ibcbr = _get(24364)
    desoc = _get(24369)

    def _kpi(key: str, label: str, unit: str, s: Optional[pd.Series], dec: int = 2) -> Dict[str, Any]:
        if s is None or len(s) == 0:
            return {"key": key, "label": label, "unit": unit, "value": None, "delta_12m": None, "last_date": None}
        latest = _sf(s.iloc[-1])
        prev = _sf(s.iloc[-13], latest) if len(s) > 12 else _sf(s.iloc[0], latest)
        return {
            "key": key,
            "label": label,
            "unit": unit,
            "value": round(latest, dec),
            "delta_12m": round(latest - prev, dec),
            "last_date": s.index[-1].strftime("%Y-%m"),
        }

    kpis = [
        _kpi("selic", "Taxa Selic (BCB 4189)", "% a.a.", selic),
        _kpi("ipca12", "IPCA Acumulado 12m", "%", ipca12),
        _kpi("dbgg", "Dívida Bruta (DBGG)", "% PIB", dbgg),
        _kpi("cambio", "Câmbio Comercial", "R$/US$", cambio),
        _kpi("ibcbr", "Atividade IBC-Br", "índice", ibcbr, 1),
        _kpi("desocupacao", "Desocupação PNAD", "%", desoc, 1),
    ]

    # Gráfico 1: Selic vs IPCA 12m vs Meta
    juros_df = pd.DataFrame({
        "selic": selic,
        "ipca12": ipca12,
        "meta": meta.reindex(ipca12.index, method="ffill").bfill() if (meta is not None and ipca12 is not None) else None,
    }).dropna(how="all").tail(108)
    chart_juros_inflacao = [
        {
            "date": idx.strftime("%Y-%m"),
            "selic": round(_sf(row["selic"]), 2) if pd.notna(row.get("selic")) else None,
            "ipca12": round(_sf(row["ipca12"]), 2) if pd.notna(row.get("ipca12")) else None,
            "meta": round(_sf(row["meta"]), 2) if pd.notna(row.get("meta")) else None,
        }
        for idx, row in juros_df.iterrows()
    ]

    # Gráfico 2: Fiscal & Câmbio (DBGG vs Dólar)
    fiscal_df = pd.DataFrame({"dbgg": dbgg, "cambio": cambio}).dropna(how="all").tail(108)
    chart_fiscal_cambio = [
        {
            "date": idx.strftime("%Y-%m"),
            "dbgg": round(_sf(row["dbgg"]), 2) if pd.notna(row.get("dbgg")) else None,
            "cambio": round(_sf(row["cambio"]), 2) if pd.notna(row.get("cambio")) else None,
        }
        for idx, row in fiscal_df.iterrows()
    ]

    # Gráfico 3: Atividade (IBC-Br) vs Desocupação (PNAD)
    ativ_df = pd.DataFrame({"ibcbr": ibcbr, "desoc": desoc}).dropna(how="all").tail(108)
    chart_atividade_trabalho = [
        {
            "date": idx.strftime("%Y-%m"),
            "ibcbr": round(_sf(row["ibcbr"]), 2) if pd.notna(row.get("ibcbr")) else None,
            "desoc": round(_sf(row["desoc"]), 2) if pd.notna(row.get("desoc")) else None,
        }
        for idx, row in ativ_df.iterrows()
    ]

    return {
        "synced_live_count": live_hits,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "kpis": kpis,
        "charts": {
            "juros_inflacao": chart_juros_inflacao,
            "fiscal_cambio": chart_fiscal_cambio,
            "atividade_trabalho": chart_atividade_trabalho,
        },
        "db_stats": get_store_stats(),
    }


def sync_all_core_bcb() -> Dict[str, Any]:
    """Força atualização ao vivo das séries centrais na API do BCB e atualiza DuckDB/Parquet."""
    updated = []
    for sid, label, _unit in CORE_CONJUNTURA_SERIES:
        df, is_live = fetch_bcb_sgs(sid, "01/01/2014", force_live=True)
        if df is not None and not df.empty:
            updated.append({
                "sgs_id": sid,
                "label": label,
                "live": is_live,
                "rows": int(len(df)),
                "last_date": df.index[-1].strftime("%Y-%m-%d"),
            })
    fetch_bcb_olinda_expinf("2015-01-01", force_live=True)
    dash = get_conjuntura_dashboard(force_live=False)
    dash["sync_details"] = updated
    return dash


# ---------------------------------------------------------------------------
# 6 MODELOS ECONOMÉTRICOS DEDICADOS (M01, M02, M06, M12, M13, M14)
# ---------------------------------------------------------------------------

def _run_m01_dominancia_fiscal(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    subsample_start = str(params.get("subsample_start", "2014-09-05"))
    max_lags = max(2, min(20, int(params.get("max_lags", 12))))
    use_live = bool(params.get("use_live", False))

    cds, _ = load_study_table("M01", "cds5y")
    selic_local, _ = load_study_table("M01", "selic")
    if cds is None or selic_local is None:
        raise ValueError("Bases históricas de M01 não encontradas no DuckDB.")

    cds = cds.copy()
    cds["date"] = pd.to_datetime(cds["date"], format="%d/%m/%Y")
    cds = cds.rename(columns={"cds5y": "CDS5Y"}).set_index("date").sort_index()
    cds["CDS5Y"] = pd.to_numeric(cds["CDS5Y"], errors="coerce")

    selic_local = selic_local.copy()
    selic_local["Date"] = pd.to_datetime(selic_local["Date"], format="%d/%m/%Y")
    selic_local = selic_local.rename(columns={"Date": "date", "Selic": "Selic"}).set_index("date").sort_index()
    selic_local["Selic"] = pd.to_numeric(selic_local["Selic"], errors="coerce")

    source_note = "Séries diárias históricas armazenadas no DuckDB (CDS 5Y e Selic Meta)."
    selic_df = selic_local
    if use_live:
        live, ok = fetch_bcb_sgs(432, "01/01/2001", force_live=True)
        if live is not None and not live.empty:
            selic_df = live.rename(columns={"value": "Selic"})
            source_note = (
                f"Selic Meta atualizada via BCB (SGS 432) até {selic_df.index.max().strftime('%Y-%m-%d')} "
                f"({selic_df['Selic'].iloc[-1]:.2f}% a.a.); pareada nas datas em comum com o CDS 5Y."
            )

    data = cds.join(selic_df, how="inner").dropna()
    data = data[~data.index.duplicated(keep="last")]
    datadif = data.diff().dropna()
    subdata = datadif[datadif.index >= pd.to_datetime(subsample_start)]
    if len(subdata) < max_lags + 15:
        raise ValueError(f"Subamostra a partir de {subsample_start} possui poucas observações ({len(subdata)}).")

    adf_table = [
        _adf_summary(data["CDS5Y"], "CDS 5 Anos (nível)"),
        _adf_summary(data["Selic"], "Taxa Selic (nível)"),
        _adf_summary(datadif["CDS5Y"], "Δ CDS 5 Anos (1ª dif.)"),
        _adf_summary(datadif["Selic"], "Δ Selic (1ª dif.)"),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag_sel = VAR(datadif[["CDS5Y", "Selic"]].reset_index(drop=True)).select_order(maxlags=max_lags)
    lag_criteria = {
        "AIC": int(lag_sel.aic),
        "BIC": int(lag_sel.bic),
        "HQIC": int(lag_sel.hqic),
        "FPE": int(lag_sel.fpe),
    }

    ccf_points = [
        {"lag": k, "corr": round(_sf(datadif["Selic"].shift(k).corr(datadif["CDS5Y"])), 4)}
        for k in range(-max_lags, max_lags + 1)
    ]

    def _granger(df_in: pd.DataFrame) -> List[Dict[str, Any]]:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            r1 = grangercausalitytests(df_in[["CDS5Y", "Selic"]], maxlag=max_lags)
            r2 = grangercausalitytests(df_in[["Selic", "CDS5Y"]], maxlag=max_lags)
        rows = []
        for lg in range(1, max_lags + 1):
            p1, f1 = float(r1[lg][0]["ssr_ftest"][1]), float(r1[lg][0]["ssr_ftest"][0])
            p2, f2 = float(r2[lg][0]["ssr_ftest"][1]), float(r2[lg][0]["ssr_ftest"][0])
            rows.append({
                "lag": lg,
                "f_selic_to_cds": round(f1, 3),
                "p_selic_to_cds": round(p1, 4),
                "sig_selic_to_cds": bool(p1 < 0.05),
                "f_cds_to_selic": round(f2, 3),
                "p_cds_to_selic": round(p2, 4),
                "sig_cds_to_selic": bool(p2 < 0.05),
            })
        return rows

    g_full = _granger(datadif)
    g_sub = _granger(subdata)
    sig_s2c = [r["lag"] for r in g_sub if r["sig_selic_to_cds"]]
    sig_c2s = [r["lag"] for r in g_sub if r["sig_cds_to_selic"]]

    if sig_s2c and not sig_c2s:
        conclusion = (
            f"Na subamostra iniciada em {subsample_start}, ΔSelic Granger-causa ΔCDS 5Y a 5% nas defasagens {sig_s2c}, "
            "sem evidência de causalidade reversa. Esse padrão empírico é compatível com indícios de dominância fiscal."
        )
    elif sig_s2c and sig_c2s:
        conclusion = f"Na subamostra, observa-se retroalimentação bidirecional: Selic → CDS ({sig_s2c}) e CDS → Selic ({sig_c2s})."
    else:
        conclusion = f"Na subamostra a partir de {subsample_start}, não houve precedência estatística significativa (p < 0,05) de ΔSelic → ΔCDS."

    step = max(1, len(data) // 600)
    level_series = [
        {"date": d.strftime("%Y-%m-%d"), "cds5y": round(_sf(row["CDS5Y"]), 2), "selic": round(_sf(row["Selic"]), 2)}
        for d, row in data.iloc[::step].iterrows()
    ]
    diff_series = [
        {"date": d.strftime("%Y-%m-%d"), "d_cds5y": round(_sf(row["CDS5Y"]), 3), "d_selic": round(_sf(row["Selic"]), 3)}
        for d, row in datadif.iloc[::step].iterrows()
    ]

    return {
        "kind": "dedicated_m01",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": source_note,
        "kpis": [
            {"label": "Observações Pareadas", "value": str(len(data)), "sub": f"{data.index.min().strftime('%Y-%m')} a {data.index.max().strftime('%Y-%m')}"},
            {"label": "Subamostra", "value": str(len(subdata)), "sub": f"a partir de {subsample_start}"},
            {"label": "Último CDS 5Y", "value": f"{data['CDS5Y'].iloc[-1]:.1f}", "sub": "pontos-base"},
            {"label": "Última Selic", "value": f"{data['Selic'].iloc[-1]:.2f}%", "sub": "% ao ano"},
            {"label": "Lag Ótimo (AIC / BIC)", "value": f"{lag_criteria['AIC']} / {lag_criteria['BIC']}", "sub": "Seleção VAR"},
        ],
        "level_series": level_series,
        "diff_series": diff_series,
        "adf": adf_table,
        "lag_criteria": lag_criteria,
        "ccf": ccf_points,
        "granger_full": g_full,
        "granger_sub": g_sub,
        "conclusion": conclusion,
        "export_df": data.reset_index(),
    }


def _run_m02_divida_publica(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    horizon = max(3, min(36, int(params.get("horizon", 12))))
    use_live = bool(params.get("use_live", True))

    df, _ = load_study_table("M02", "painel_fiscal")
    if df is None:
        raise ValueError("Painel fiscal histórico de M02 não encontrado no DuckDB.")
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y")
    df = df.set_index("DATE").sort_index()

    dbgg_series = pd.to_numeric(df["DBGG"], errors="coerce").dropna()
    source_note = "Painel fiscal histórico local (DBGG/PIB, PIB 12m, Juro Real e Superávit Primário)."
    if use_live:
        live, _ = fetch_bcb_sgs(13762, "01/12/2006", force_live=False)
        if live is not None and not live.empty:
            dbgg_series = live["value"]
            source_note = f"Série oficial da DBGG/PIB via BCB (SGS 13762 / DuckDB) até {dbgg_series.index.max().strftime('%m/%Y')} ({dbgg_series.iloc[-1]:.2f}% do PIB)."

    dbgg_series = dbgg_series.asfreq("MS").ffill()
    adf_table = [
        _adf_summary(dbgg_series, "DBGG / PIB (% — nível)"),
        _adf_summary(dbgg_series.diff().dropna(), "Δ DBGG / PIB (1ª diferença)"),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(dbgg_series, order=(1, 1, 1), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    fc_m = fc.predicted_mean
    ci95 = fc.conf_int(alpha=0.05)
    ci80 = fc.conf_int(alpha=0.20)

    last_val = float(dbgg_series.iloc[-1])
    end_val = float(fc_m.iloc[-1])
    delta_pp = end_val - last_val

    forecast_series = [
        {"date": d.strftime("%Y-%m"), "observed": round(_sf(v), 2), "forecast": None, "ci95": None, "ci80": None}
        for d, v in dbgg_series.tail(96).items()
    ]
    for i, d in enumerate(fc_m.index):
        forecast_series.append({
            "date": d.strftime("%Y-%m"),
            "observed": None,
            "forecast": round(_sf(fc_m.iloc[i]), 2),
            "ci95": [round(_sf(ci95.iloc[i, 0]), 2), round(_sf(ci95.iloc[i, 1]), 2)],
            "ci80": [round(_sf(ci80.iloc[i, 0]), 2), round(_sf(ci80.iloc[i, 1]), 2)],
        })

    fiscal_panel = [
        {
            "date": d.strftime("%Y-%m"),
            "s_primario": round(_sf(row["SPRIMARIO"]), 2),
            "s_req": round(_sf(row["SPREQ"]), 2),
            "juro_real": round(_sf(row["JUROREAL"]), 2),
            "pib": round(_sf(row["PIB"]), 2),
        }
        for d, row in df.iterrows()
    ]

    return {
        "kind": "dedicated_m02",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": source_note,
        "kpis": [
            {"label": "DBGG Atual", "value": f"{last_val:.2f}%", "sub": f"Ref. {dbgg_series.index.max().strftime('%m/%Y')}"},
            {"label": f"Projeção ({horizon}m)", "value": f"{end_val:.2f}%", "sub": f"Em {fc_m.index[-1].strftime('%m/%Y')}"},
            {"label": "Variação Projetada", "value": f"{delta_pp:+.2f} p.p.", "sub": "Cenário inercial ARIMA"},
            {"label": "AIC do Modelo", "value": f"{fit.aic:.1f}", "sub": f"N = {len(dbgg_series)} meses"},
        ],
        "forecast_series": forecast_series,
        "fiscal_panel": fiscal_panel,
        "adf": adf_table,
        "conclusion": (
            f"Partindo de {last_val:.2f}% do PIB ({dbgg_series.index.max().strftime('%m/%Y')}), a projeção inercial "
            f"ARIMA(1,1,1) com drift aponta DBGG de {end_val:.2f}% do PIB ao fim de {horizon} meses ({delta_pp:+.2f} p.p.). "
            "No painel fiscal, desvios persistentes do primário efetivo abaixo do requerido amplificam a trajetória de alta."
        ),
        "export_df": pd.DataFrame({"date": fc_m.index.strftime("%Y-%m-%d"), "dbgg_proj": fc_m.values, "ic95_inf": ci95.iloc[:, 0].values, "ic95_sup": ci95.iloc[:, 1].values}),
    }


def _run_m06_regra_taylor(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hp_lambda = max(100.0, min(100000.0, float(params.get("hp_lambda", 14400.0))))
    use_live = bool(params.get("use_live", True))

    df_local, _ = load_study_table("M06", "curva_reacao")
    if df_local is None:
        raise ValueError("Base histórica de M06 não encontrada no DuckDB.")
    df_local = df_local.copy()
    df_local["date"] = df_local["DATE"].apply(_parse_pt_month_year)
    df_local = df_local.set_index("date").sort_index()
    df = df_local[["SELIC", "EXPINF", "META", "PIB"]].copy()
    source_note = "Base histórica local do DuckDB (nov/2001 a jun/2016)."

    if use_live:
        s_selic, _ = fetch_bcb_sgs(4189, "01/01/2015", force_live=False)
        s_meta, _ = fetch_bcb_sgs(13521, "01/01/2015", force_live=False)
        s_ibc, _ = fetch_bcb_sgs(24364, "01/01/2015", force_live=False)
        s_exp, _ = fetch_bcb_olinda_expinf("2015-01-01", force_live=False)
        exp_lbl = "Focus 12m (Olinda BCB)"
        if s_exp is None or s_exp.empty:
            s_ipca12, _ = fetch_bcb_sgs(13522, "01/01/2015", force_live=False)
            if s_ipca12 is not None and not s_ipca12.empty:
                s_exp = s_ipca12["value"].resample("MS").mean()
                exp_lbl = "IPCA 12m (SGS 13522)"
        if all(x is not None and not x.empty for x in (s_selic, s_meta, s_ibc)) and s_exp is not None and not s_exp.empty:
            live_df = pd.DataFrame({
                "SELIC": s_selic["value"].resample("MS").mean(),
                "PIB": s_ibc["value"].resample("MS").mean(),
                "EXPINF": s_exp,
            })
            meta_m = s_meta["value"].resample("MS").ffill()
            live_df["META"] = meta_m.reindex(live_df.index, method="ffill").bfill()
            live_df = live_df.dropna()
            if len(live_df) >= 36:
                df = live_df
                source_note = (
                    f"Séries oficiais do BCB (Selic 4189, Meta 13521, IBC-Br 24364 + {exp_lbl}) "
                    f"de {df.index.min().strftime('%m/%Y')} a {df.index.max().strftime('%m/%Y')} ({len(df)} meses)."
                )

    log_pib = np.log(df["PIB"].clip(lower=1e-6)) * 100.0
    cycle, _trend = hpfilter(log_pib, lamb=hp_lambda)
    df["hiato"] = cycle
    df["desvio_inf"] = df["EXPINF"] - df["META"]
    df["selic_lag1"] = df["SELIC"].shift(1)

    reg_df = df[["SELIC", "selic_lag1", "desvio_inf", "hiato"]].dropna()
    X = sm.add_constant(reg_df[["selic_lag1", "desvio_inf", "hiato"]])
    y = reg_df["SELIC"]
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    names_pt = {
        "const": "Intercepto (c)",
        "selic_lag1": "Suavização Selic t−1 (ρ)",
        "desvio_inf": "Desvio da Expectativa (β₁)",
        "hiato": "Hiato do Produto (β₂)",
    }
    coefs = [
        {
            "term": names_pt.get(k, k),
            "coef": round(_sf(ols.params[k]), 4),
            "std_err": round(_sf(ols.bse[k]), 4),
            "t_stat": round(_sf(ols.tvalues[k]), 3),
            "pvalue": round(_sf(ols.pvalues[k], 1.0), 4),
            "sig": bool(ols.pvalues[k] < 0.05),
        }
        for k in ols.params.index
    ]

    rho = float(ols.params["selic_lag1"])
    beta1 = float(ols.params["desvio_inf"])
    lr_beta = beta1 / (1.0 - rho) if abs(1.0 - rho) > 1e-5 else 0.0
    taylor_ok = bool(lr_beta > 1.0)

    series_points = [
        {
            "date": d.strftime("%Y-%m"),
            "selic_obs": round(_sf(y.loc[d]), 2),
            "selic_fit": round(_sf(ols.fittedvalues.loc[d]), 2),
            "hiato": round(_sf(reg_df.loc[d, "hiato"]), 2),
            "desvio_inf": round(_sf(reg_df.loc[d, "desvio_inf"]), 2),
        }
        for d in reg_df.index
    ]

    return {
        "kind": "dedicated_m06",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": source_note,
        "kpis": [
            {"label": "R² da Regra de Taylor", "value": f"{ols.rsquared * 100:.1f}%", "sub": f"N = {len(reg_df)} meses"},
            {"label": "Suavização (ρ)", "value": f"{rho:.3f}", "sub": "Inércia da taxa Selic"},
            {"label": "Resp. Longo Prazo β₁/(1−ρ)", "value": f"{lr_beta:.2f}", "sub": "Princípio de Taylor (> 1,0)"},
            {"label": "Princípio de Taylor", "value": "Satisfeito ✓" if taylor_ok else "Abaixo de 1", "sub": f"λ HP = {int(hp_lambda)}"},
        ],
        "series": series_points,
        "coefs": coefs,
        "conclusion": (
            f"A função de reação explica {ols.rsquared * 100:.1f}% da variância da Selic, com suavização ρ = {rho:.3f} e "
            f"resposta de longo prazo à inflação β₁/(1−ρ) = {lr_beta:.2f}. "
            + (
                "Como supera 1,0, o Princípio de Taylor é satisfeito: o Banco Central eleva o juro real quando a inflação esperada supera a meta."
                if taylor_ok
                else "A resposta de longo prazo ficou abaixo de 1,0 nesta janela amostral."
            )
        ),
        "export_df": pd.DataFrame(series_points),
    }


def _run_m12_inercia_ipca(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    window = max(18, min(96, int(params.get("window", 48))))
    use_live = bool(params.get("use_live", True))

    df_loc, _ = load_study_table("M12", "ipca_inercia")
    meta_df, _ = load_study_table("M12", "meta_inflacao")
    if df_loc is None or meta_df is None:
        raise ValueError("Bases de M12 não encontradas no DuckDB.")

    df_loc = df_loc.copy()
    df_loc["date"] = df_loc.iloc[:, 0].apply(_parse_pt_month_year)
    df_loc = df_loc.set_index("date").sort_index()

    meta_df = meta_df.copy()
    meta_df["date"] = meta_df.iloc[:, 0].apply(_parse_pt_month_year)
    meta_df = meta_df.set_index("date").sort_index()

    ipca_m = pd.to_numeric(df_loc["ipca"], errors="coerce").dropna()
    source_note = "Série histórica local de IPCA mensal e Meta de Inflação (DuckDB)."
    if use_live:
        live, _ = fetch_bcb_sgs(433, "01/01/2000", force_live=False)
        if live is not None and not live.empty:
            ipca_m = live["value"]
            source_note = f"IPCA mensal oficial via BCB (SGS 433 / DuckDB) até {ipca_m.index.max().strftime('%m/%Y')}."

    ipca_12m = ((1.0 + ipca_m / 100.0).rolling(12).apply(np.prod, raw=True) - 1.0) * 100.0
    ipca_12m = ipca_12m.dropna()
    live_meta, _ = fetch_bcb_sgs(13521, "01/01/2000", force_live=False)
    meta_series = live_meta["value"].resample("MS").ffill() if (live_meta is not None and not live_meta.empty) else meta_df["meta"]
    joined = ipca_12m.to_frame("ipca12").join(meta_series.rename("meta"), how="left").ffill().bfill()

    vals = ipca_m.values
    idx = ipca_m.index
    rolling_pts = []
    for i in range(window, len(vals)):
        sub = vals[i - window : i]
        ols = sm.OLS(sub[1:], sm.add_constant(sub[:-1])).fit()
        g = _sf(ols.params[1])
        se = _sf(ols.bse[1])
        rolling_pts.append({
            "date": idx[i].strftime("%Y-%m"),
            "gamma": round(g, 4),
            "band": [round(g - 1.96 * se, 4), round(g + 1.96 * se, 4)],
        })

    gammas = [p["gamma"] for p in rolling_pts]
    latest_g = gammas[-1]
    mean_g = float(np.mean(gammas))

    ipca_series = [
        {"date": d.strftime("%Y-%m"), "ipca12": round(_sf(row["ipca12"]), 2), "meta": round(_sf(row["meta"]), 2)}
        for d, row in joined.iterrows()
    ]

    return {
        "kind": "dedicated_m12",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": source_note,
        "kpis": [
            {"label": "IPCA Acum. 12m", "value": f"{joined['ipca12'].iloc[-1]:.2f}%", "sub": f"Meta: {joined['meta'].iloc[-1]:.2f}%"},
            {"label": "Inércia Atual γ(t)", "value": f"{latest_g:.3f}", "sub": f"Janela de {window} meses"},
            {"label": "Média Histórica γ", "value": f"{mean_g:.3f}", "sub": f"Pico: {max(gammas):.3f}"},
            {"label": "Observações", "value": str(len(ipca_m)), "sub": "Meses de IPCA"},
        ],
        "ipca_series": ipca_series,
        "rolling_series": rolling_pts,
        "conclusion": (
            f"Com janela móvel de {window} meses, a persistência autorregressiva atual do IPCA é γ(t) = {latest_g:.3f} "
            f"(média histórica de {mean_g:.3f}). Coeficientes elevados indicam propagação lenta de choques e maior esforço "
            "monetário para reconduzir a inflação à meta."
        ),
        "export_df": pd.DataFrame([{"date": p["date"], "gamma": p["gamma"], "ic_inf": p["band"][0], "ic_sup": p["band"][1]} for p in rolling_pts]),
    }


def _run_m13_sarima_ipca(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    p = max(0, min(4, int(params.get("p", 1))))
    d = max(0, min(2, int(params.get("d", 0))))
    q = max(0, min(4, int(params.get("q", 1))))
    P = max(0, min(2, int(params.get("P", 1))))
    D = max(0, min(1, int(params.get("D", 0))))
    Q = max(0, min(2, int(params.get("Q", 1))))
    horizon = max(3, min(24, int(params.get("horizon", 12))))

    live, _ = fetch_bcb_sgs(433, "01/01/2003", force_live=False)
    if live is not None and not live.empty:
        ipca = live["value"].asfreq("MS").ffill()
        source_note = f"IPCA mensal oficial via BCB (SGS 433 / DuckDB) até {ipca.index.max().strftime('%m/%Y')}."
    else:
        fb, _ = load_study_table("M13", "ipca_historico")
        if fb is None:
            raise ValueError("Série de IPCA não encontrada.")
        fb = fb.copy()
        fb["date"] = fb.iloc[:, 0].apply(_parse_pt_month_year)
        ipca = pd.to_numeric(fb.set_index("date")["ipca"], errors="coerce").dropna().asfreq("MS").ffill()
        source_note = "Série histórica de IPCA mensal do DuckDB."

    adf_res = _adf_summary(ipca, "IPCA mensal (% m/m)")
    nlags = min(24, len(ipca) // 3)
    acf_v = acf(ipca.dropna(), nlags=nlags)
    pacf_v = pacf(ipca.dropna(), nlags=nlags)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            ipca,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    fc_m = fc.predicted_mean
    ci95 = fc.conf_int(alpha=0.05)

    rmse = float(np.sqrt(np.mean(fit.resid ** 2)))
    acc_12m = float((np.prod(1.0 + fc_m.values[:12] / 100.0) - 1.0) * 100.0) if horizon >= 12 else None

    fc_series = [
        {"date": dt.strftime("%Y-%m"), "observed": round(_sf(v), 2), "forecast": None, "ci95": None}
        for dt, v in ipca.tail(60).items()
    ]
    for i, dt in enumerate(fc_m.index):
        fc_series.append({
            "date": dt.strftime("%Y-%m"),
            "observed": None,
            "forecast": round(_sf(fc_m.iloc[i]), 2),
            "ci95": [round(_sf(ci95.iloc[i, 0]), 2), round(_sf(ci95.iloc[i, 1]), 2)],
        })

    correlogram = [
        {"lag": k, "acf": round(_sf(acf_v[k]), 4), "pacf": round(_sf(pacf_v[k]), 4)}
        for k in range(1, nlags + 1)
    ]

    return {
        "kind": "dedicated_m13",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": source_note,
        "kpis": [
            {"label": "Último IPCA (% m/m)", "value": f"{ipca.iloc[-1]:.2f}%", "sub": f"Ref. {ipca.index.max().strftime('%m/%Y')}"},
            {"label": "Acum. 12m Projetado", "value": f"{acc_12m:.2f}%" if acc_12m is not None else "N/A", "sub": f"Horizonte {horizon}m"},
            {"label": "AIC / BIC", "value": f"{fit.aic:.1f}", "sub": f"BIC: {fit.bic:.1f}"},
            {"label": "RMSE In-Sample", "value": f"{rmse:.3f} p.p.", "sub": f"SARIMA({p},{d},{q})({P},{D},{Q})[12]"},
        ],
        "forecast_series": fc_series,
        "correlogram": correlogram,
        "adf": [adf_res],
        "conclusion": (
            f"O modelo SARIMA({p},{d},{q})({P},{D},{Q})[12] apresentou AIC = {fit.aic:.1f} e RMSE = {rmse:.3f} p.p. "
            + (f"A inflação acumulada projetada para os próximos 12 meses é de {acc_12m:.2f}%." if acc_12m is not None else "")
        ),
        "export_df": pd.DataFrame({"date": fc_m.index.strftime("%Y-%m-%d"), "ipca_proj": fc_m.values, "ic95_inf": ci95.iloc[:, 0].values, "ic95_sup": ci95.iloc[:, 1].values}),
    }


def _run_m14_curva_phillips(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    maxlags_hac = max(1, min(12, int(params.get("hac_lags", 4))))
    df, _ = load_study_table("M14", "curva_phillips")
    if df is None:
        raise ValueError("Base histórica de M14 não encontrada no DuckDB.")
    df = df.copy()

    # Identifica colunas numéricas relevantes da Curva de Phillips
    x_dates = df.iloc[:, 0].astype(str).tolist()
    for c in df.columns[1:]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not str(c).startswith("Unnamed")]

    # Escolhe Y (inflação) e até 4 regressores principais não colineares
    y_col = next((c for c in num_cols if "ipca" in c.lower() or "inf" in c.lower()), num_cols[0])
    cand_x = [c for c in num_cols if c != y_col]
    # Prioriza expectativas, hiato/desemprego e câmbio se presentes
    priority_keys = ("exp", "hiato", "desemp", "camb", "ic", "pib", "meta")
    cand_x.sort(key=lambda c: (0 if any(k in c.lower() for k in priority_keys) else 1, c))
    x_cols = cand_x[:4]

    clean = df[[y_col] + x_cols].dropna()
    X = sm.add_constant(clean[x_cols])
    y = clean[y_col]
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags_hac})

    coefs = [
        {
            "term": "Intercepto" if k == "const" else str(k),
            "coef": round(_sf(ols.params[k]), 4),
            "std_err": round(_sf(ols.bse[k]), 4),
            "t_stat": round(_sf(ols.tvalues[k]), 3),
            "pvalue": round(_sf(ols.pvalues[k], 1.0), 4),
            "sig": bool(ols.pvalues[k] < 0.05),
        }
        for k in ols.params.index
    ]

    reg_dates = [x_dates[i] if i < len(x_dates) else str(i) for i in clean.index]
    series_pts = [
        {
            "date": reg_dates[i],
            "actual": round(_sf(y.iloc[i]), 3),
            "fitted": round(_sf(ols.fittedvalues.iloc[i]), 3),
            "resid": round(_sf(ols.resid.iloc[i]), 3),
        }
        for i in range(len(clean))
    ]
    adf_resid = _adf_summary(ols.resid, "Resíduos da Curva de Phillips")
    sig_vars = [c["term"] for c in coefs if c["sig"] and c["term"] != "Intercepto"]

    return {
        "kind": "dedicated_m14",
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": f"Painel econométrico da Curva de Phillips no DuckDB ({len(clean)} obs., erros robustos Newey-West HAC lags={maxlags_hac}).",
        "kpis": [
            {"label": "R² / R² Ajust.", "value": f"{ols.rsquared * 100:.1f}%", "sub": f"Ajustado: {ols.rsquared_adj * 100:.1f}%"},
            {"label": "Estatística F", "value": f"{_sf(ols.fvalue):.1f}", "sub": f"N = {int(ols.nobs)} obs."},
            {"label": "Variável Alvo", "value": str(y_col), "sub": f"{len(x_cols)} regressores"},
            {"label": "Resíduos (ADF)", "value": "Estacionários ✓" if adf_resid["stationary_5pct"] else "Não estac.", "sub": f"p = {adf_resid['pvalue']:.4f}"},
        ],
        "dep_var": str(y_col),
        "series": series_pts,
        "coefs": coefs,
        "adf": [adf_resid],
        "conclusion": (
            f"A Curva de Phillips estimada por MQO-HAC explica {ols.rsquared * 100:.1f}% da variância de '{y_col}'. "
            + (
                f"Os regressores estatisticamente significativos a 5% são: {', '.join(sig_vars)}."
                if sig_vars
                else "Nenhum regressor isolado atingiu significância a 5% nessa especificação."
            )
        ),
        "export_df": pd.DataFrame(series_pts),
    }


# ---------------------------------------------------------------------------
# MOTOR ECONOMÉTRICO TEMÁTICO PARA OS DEMAIS 24 ESTUDOS (M03..M30)
# ---------------------------------------------------------------------------

def _run_thematic_study(study: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    code = study["code"]
    use_live = bool(params.get("use_live", False))
    model_type = study["model_type"]

    df: Optional[pd.DataFrame] = None
    source_note = ""

    def _build_bcb_panel(series_spec: List[Any], live_flag: bool) -> Tuple[Optional[pd.DataFrame], List[str]]:
        s_dict: Dict[str, pd.Series] = {}
        for sid, lbl in series_spec[:4]:
            sid_int = int(sid)
            if sid_int == -1:
                exp_s, _ = fetch_bcb_olinda_expinf("2014-01-01", force_live=live_flag)
                if exp_s is not None and not exp_s.empty:
                    s_dict[lbl] = exp_s.resample("MS").mean()
            else:
                s_df, _ = fetch_bcb_sgs(sid_int, "01/01/2014", force_live=live_flag)
                if s_df is not None and not s_df.empty:
                    s_dict[lbl] = s_df["value"].resample("MS").mean()
        if not s_dict:
            return None, []
        raw_df = pd.DataFrame(s_dict).sort_index()
        # Séries anuais/degrau (ex.: Meta CMN 13521) são propagadas mensalmente com ffill completo
        max_len = max(int(raw_df[c].notna().sum()) for c in raw_df.columns)
        for c in raw_df.columns:
            col_n = int(raw_df[c].notna().sum())
            if col_n < max_len * 0.35:
                raw_df[c] = raw_df[c].ffill().bfill()
            else:
                raw_df[c] = raw_df[c].ffill(limit=2)
        inner = raw_df.dropna()
        panel = inner if len(inner) >= 30 else raw_df.ffill().bfill().dropna(how="all")
        panel = panel.copy()
        panel.insert(0, "data", panel.index.strftime("%Y-%m-%d"))
        return panel.reset_index(drop=True), list(s_dict.keys())

    # 1. Se usuário pediu atualização ao vivo do BCB
    if use_live and study.get("bcb_series"):
        panel_df, loaded_labels = _build_bcb_panel(study["bcb_series"], live_flag=True)
        if panel_df is not None and not panel_df.empty:
            df = panel_df
            source_note = f"Séries atualizadas ao vivo via API SGS/Olinda do Banco Central ({', '.join(loaded_labels)})."

    # 2. Caso padrão: carrega base própria histórica do estudo no DuckDB
    if df is None:
        # Caso especial M27 (3 séries diárias separadas -> junta por data!)
        if code == "M27":
            c_df, _ = load_study_table("M27", "cambio_diario")
            r_df, _ = load_study_table("M27", "cds_cambio")
            s_df, _ = load_study_table("M27", "selic_cambio")
            if c_df is not None and r_df is not None:
                c_df = c_df.copy()
                r_df = r_df.copy()
                c_df["d"] = pd.to_datetime(c_df.iloc[:, 0], dayfirst=True, errors="coerce")
                r_df["d"] = pd.to_datetime(r_df.iloc[:, 0], dayfirst=True, errors="coerce")
                merged = pd.DataFrame({
                    "Cambio": pd.to_numeric(c_df.set_index("d").iloc[:, 1], errors="coerce"),
                    "CDS5Y": pd.to_numeric(r_df.set_index("d").iloc[:, 1], errors="coerce"),
                }).dropna()
                if s_df is not None:
                    s_df = s_df.copy()
                    s_df["d"] = pd.to_datetime(s_df.iloc[:, 0], dayfirst=True, errors="coerce")
                    merged["Selic"] = pd.to_numeric(s_df.set_index("d").iloc[:, 1], errors="coerce")
                merged = merged.resample("MS").mean().dropna().reset_index()
                merged["d"] = merged["d"].dt.strftime("%Y-%m-%d")
                df = merged.rename(columns={"d": "data"})
                source_note = "Painel mensal consolidado no DuckDB (Câmbio R$/US$, CDS 5Y e Selic)."
        else:
            for dkey in study.get("dataset_keys", []):
                loaded, _ = load_study_table(code, dkey)
                if loaded is not None and not loaded.empty:
                    df = loaded.copy()
                    source_note = f"Base histórica do estudo no DuckDB ({code} • {dkey}, {len(df)} obs.)."
                    break

    # 3. Séries oficiais BCB armazenadas no DuckDB/Parquet (estudos M31–M60 ou fallback)
    if df is None and study.get("bcb_series"):
        panel_df, loaded_labels = _build_bcb_panel(study["bcb_series"], live_flag=False)
        if panel_df is not None and not panel_df.empty:
            df = panel_df
            source_note = (
                f"Séries oficiais do BCB (SGS/Olinda) em cache DuckDB/Parquet "
                f"({', '.join(loaded_labels)} — {len(df)} meses)."
            )

    if df is None or df.empty:
        raise ValueError(f"Nenhuma base disponível para o estudo {code}.")

    # Limpeza automática de eixo X e colunas numéricas
    x_vals = [str(i + 1) for i in range(len(df))]
    for col in df.columns[:2]:
        col_low = str(col).lower()
        if df[col].dtype == object or "date" in col_low or "data" in col_low or "mes" in col_low:
            sample = df[col].dropna().astype(str).head(5).tolist()
            if sample and any("/" in s or "-" in s for s in sample):
                x_vals = df[col].astype(str).tolist()
                break

    for c in df.columns:
        clow = str(c).lower()
        if df[c].dtype == object and not any(k in clow for k in ("date", "data", "mes", "ano", "trim")):
            coerced = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if coerced.notna().sum() > len(df) * 0.55:
                df[c] = coerced

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not str(c).startswith(("Unnamed", "col_0"))
    ]
    if not num_cols:
        num_cols = list(df.select_dtypes(include=[np.number]).columns)

    # Garante ao menos 2 séries para modelos multivariados (usando defasagem se univariado)
    if len(num_cols) == 1:
        base = num_cols[0]
        s0 = pd.to_numeric(df[base], errors="coerce")
        df[f"{base}_t-1"] = s0.shift(1).bfill()
        df[f"d_{base}"] = s0.diff().fillna(0.0)
        num_cols = [base, f"{base}_t-1", f"d_{base}"]

    clean = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    x_aligned = [x_vals[i] if i < len(x_vals) else str(i) for i in clean.index]

    # Estatísticas descritivas & ADF & Correlação
    stats_rows = []
    adf_rows = []
    for col in num_cols[:6]:
        s = clean[col]
        stats_rows.append({
            "column": str(col),
            "count": int(len(s)),
            "mean": round(_sf(s.mean()), 3),
            "std": round(_sf(s.std()), 3),
            "min": round(_sf(s.min()), 3),
            "median": round(_sf(s.median()), 3),
            "max": round(_sf(s.max()), 3),
        })
        if len(s) >= 24 and len(adf_rows) < 4 and s.std() > 1e-9:
            try:
                adf_rows.append(_adf_summary(s, str(col)))
            except Exception:
                pass

    corr_cols = [str(c) for c in num_cols[:6]]
    corr_mat = clean[num_cols[:6]].corr().fillna(0.0)
    corr_matrix = {
        "cols": corr_cols,
        "rows": [[round(_sf(v), 3) for v in row] for row in corr_mat.values],
    }

    # Execução do modelo econométrico específico do estudo
    model_payload = _execute_econometric_block(model_type, clean, x_aligned, num_cols)

    # Série temporal amostrada (máx 450 pontos) para gráfico exploratório
    step = max(1, len(clean) // 450)
    raw_chart = []
    for pos in range(0, len(clean), step):
        row_d: Dict[str, Any] = {"date": x_aligned[pos]}
        for c in num_cols[:4]:
            row_d[str(c)] = round(_sf(clean[c].iloc[pos]), 3)
        raw_chart.append(row_d)

    return {
        "kind": "thematic",
        "model_type": model_payload["model_type"],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "source_note": source_note,
        "kpis": [
            {"label": "Observações Válidas", "value": str(len(clean)), "sub": f"{len(num_cols)} variáveis numéricas"},
            {"label": "Série Principal", "value": str(num_cols[0])[:22], "sub": f"Último: {_sf(clean[num_cols[0]].iloc[-1]):.2f}"},
            {"label": "Metodologia", "value": model_payload["short_label"], "sub": "Python / Statsmodels"},
            {"label": "Tempo de Cálculo", "value": f"{round((time.perf_counter() - t0) * 1000.0, 0):.0f} ms", "sub": "Backend FastAPI"},
        ],
        "series_keys": [str(c) for c in num_cols[:4]],
        "raw_chart": raw_chart,
        "model_result": model_payload,
        "adf": adf_rows,
        "summary_stats": stats_rows,
        "corr_matrix": corr_matrix,
        "conclusion": model_payload["conclusion"],
        "export_df": clean.reset_index(drop=True),
    }


def _execute_econometric_block(
    model_type: str,
    clean: pd.DataFrame,
    x_aligned: List[str],
    num_cols: List[str],
) -> Dict[str, Any]:
    try:
        if model_type == "var_granger" and len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            sub = clean[[c1, c2]].diff().dropna()
            if len(sub) >= 25:
                max_l = min(6, max(2, len(sub) // 14))
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    vmod = VAR(sub.reset_index(drop=True))
                    fit = vmod.fit(maxlags=max_l, ic="aic")
                    if fit.k_ar == 0:
                        fit = vmod.fit(1)
                    irf = fit.irf(10)
                    g1 = grangercausalitytests(sub[[c1, c2]], maxlag=max_l)
                    g2 = grangercausalitytests(sub[[c2, c1]], maxlag=max_l)
                irf_pts = [
                    {"h": h, "shock_2to1": round(_sf(irf.irfs[h, 0, 1]), 4), "shock_1to2": round(_sf(irf.irfs[h, 1, 0]), 4)}
                    for h in range(11)
                ]
                g_rows = [
                    {
                        "lag": lg,
                        "f_2to1": round(_sf(g1[lg][0]["ssr_ftest"][0]), 3),
                        "p_2to1": round(_sf(g1[lg][0]["ssr_ftest"][1], 1.0), 4),
                        "sig_2to1": bool(g1[lg][0]["ssr_ftest"][1] < 0.05),
                        "f_1to2": round(_sf(g2[lg][0]["ssr_ftest"][0]), 3),
                        "p_1to2": round(_sf(g2[lg][0]["ssr_ftest"][1], 1.0), 4),
                        "sig_1to2": bool(g2[lg][0]["ssr_ftest"][1] < 0.05),
                    }
                    for lg in range(1, max_l + 1)
                ]
                return {
                    "model_type": "var_granger",
                    "short_label": f"VAR({fit.k_ar}) + IRF",
                    "var_names": [str(c1), str(c2)],
                    "used_lag": int(fit.k_ar),
                    "aic": round(_sf(fit.aic), 2),
                    "irf_points": irf_pts,
                    "granger_rows": g_rows,
                    "conclusion": (
                        f"Modelo VAR({fit.k_ar}) estimado entre Δ({c1}) e Δ({c2}) (AIC = {fit.aic:.2f}). "
                        "A Função Impulso-Resposta (IRF) mapeia a propagação dinâmica em 10 passos acompanhada pelos testes de Granger."
                    ),
                }

        if model_type == "coint_vecm" and len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            s1, s2 = clean[c1], clean[c2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_stat, c_pval, _ = coint(s1, s2)
            lr = sm.OLS(s1, sm.add_constant(s2)).fit()
            ect = lr.resid
            ecm_df = pd.DataFrame({"dy": s1.diff(), "ect_lag": ect.shift(1), "dx": s2.diff()}).dropna()
            ecm = sm.OLS(ecm_df["dy"], sm.add_constant(ecm_df[["ect_lag", "dx"]])).fit()
            a_ect = _sf(ecm.params["ect_lag"])
            p_ect = _sf(ecm.pvalues["ect_lag"], 1.0)
            step = max(1, len(s1) // 400)
            pts = [
                {
                    "date": x_aligned[i],
                    "actual": round(_sf(s1.iloc[i]), 3),
                    "equilibrium": round(_sf(lr.fittedvalues.iloc[i]), 3),
                    "ect": round(_sf(ect.iloc[i]), 3),
                }
                for i in range(0, len(s1), step)
            ]
            return {
                "model_type": "coint_vecm",
                "short_label": "Engle-Granger + ECM",
                "var_names": [str(c1), str(c2)],
                "coint_stat": round(_sf(c_stat), 3),
                "coint_pval": round(_sf(c_pval, 1.0), 4),
                "is_cointegrated": bool(c_pval < 0.05),
                "beta_lr": round(_sf(lr.params.iloc[1]), 4),
                "alpha_ect": round(a_ect, 4),
                "p_ect": round(p_ect, 4),
                "points": pts,
                "conclusion": (
                    f"Cointegração Engle-Granger entre {c1} e {c2}: t = {c_stat:.3f} (p = {c_pval:.4f}). "
                    f"Velocidade de ajuste ao equilíbrio de longo prazo α_ECT = {a_ect:.4f} (p = {p_ect:.4f})."
                ),
            }

        if model_type == "hp_neutral":
            target = num_cols[0]
            s = clean[target]
            lam = 14400 if len(s) > 60 else 1600
            cycle, trend = hpfilter(s, lamb=lam)
            step = max(1, len(s) // 450)
            pts = [
                {
                    "date": x_aligned[i],
                    "observed": round(_sf(s.iloc[i]), 3),
                    "trend": round(_sf(trend.iloc[i]), 3),
                    "cycle": round(_sf(cycle.iloc[i]), 3),
                }
                for i in range(0, len(s), step)
            ]
            return {
                "model_type": "hp_neutral",
                "short_label": f"Filtro HP (λ={lam})",
                "target": str(target),
                "hp_lambda": lam,
                "latest_trend": round(_sf(trend.iloc[-1]), 2),
                "latest_cycle": round(_sf(cycle.iloc[-1]), 2),
                "points": pts,
                "conclusion": (
                    f"Filtro Hodrick-Prescott (λ = {lam}) sobre '{target}': tendência estrutural atual em "
                    f"{trend.iloc[-1]:.2f} e desvio cíclico (hiato) em {cycle.iloc[-1]:+.2f}."
                ),
            }

        if model_type == "volatility":
            target = num_cols[0]
            s = clean[target]
            if (s > 0).all():
                ret = np.log(s / s.shift(1)).dropna() * 100.0
                ret_lbl = f"Retorno Log {target} (%)"
            else:
                ret = s.diff().dropna()
                ret_lbl = f"Variação Δ {target}"
            ewma_vol = np.sqrt(ret.ewm(alpha=0.06).var().dropna())
            kurt = _sf(ret.kurtosis())
            skew = _sf(ret.skew())
            ret_al = ret.iloc[-len(ewma_vol) :]
            dates_al = x_aligned[-len(ewma_vol) :]
            step = max(1, len(ewma_vol) // 450)
            pts = [
                {"date": dates_al[i], "ret": round(_sf(ret_al.iloc[i]), 4), "vol": round(_sf(ewma_vol.iloc[i]), 4)}
                for i in range(0, len(ewma_vol), step)
            ]
            return {
                "model_type": "volatility",
                "short_label": "Volatilidade EWMA",
                "target": str(target),
                "ret_label": ret_lbl,
                "latest_vol": round(_sf(ewma_vol.iloc[-1]), 3),
                "max_vol": round(_sf(ewma_vol.max()), 3),
                "kurtosis": round(kurt, 2),
                "skewness": round(skew, 2),
                "points": pts,
                "conclusion": (
                    f"Volatilidade condicional EWMA (λ=0,94) de '{target}': atual = {ewma_vol.iloc[-1]:.3f} "
                    f"(pico = {ewma_vol.max():.3f}). Curtose = {kurt:.2f} "
                    f"({'caudas pesadas / leptocúrtica' if kurt > 1 else 'aderente à normal'})."
                ),
            }

        if model_type == "sarima_forecast":
            target = num_cols[0]
            s = clean[target].reset_index(drop=True)
            nlags = min(20, max(4, len(s) // 4))
            acf_v = acf(s, nlags=nlags)
            pacf_v = pacf(s, nlags=nlags)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(s, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = fit.get_forecast(steps=12)
            fc_m = fc.predicted_mean
            ci = fc.conf_int(alpha=0.05)
            hist_n = min(72, len(s))
            fc_pts = [
                {"date": x_aligned[len(s) - hist_n + i], "observed": round(_sf(s.iloc[len(s) - hist_n + i]), 3), "forecast": None, "ci95": None}
                for i in range(hist_n)
            ]
            for i in range(12):
                fc_pts.append({
                    "date": f"T+{i + 1}",
                    "observed": None,
                    "forecast": round(_sf(fc_m.iloc[i]), 3),
                    "ci95": [round(_sf(ci.iloc[i, 0]), 3), round(_sf(ci.iloc[i, 1]), 3)],
                })
            corr_pts = [{"lag": k, "acf": round(_sf(acf_v[k]), 3), "pacf": round(_sf(pacf_v[k]), 3)} for k in range(1, nlags + 1)]
            return {
                "model_type": "sarima_forecast",
                "short_label": "ARIMA(1,1,1)",
                "target": str(target),
                "aic": round(_sf(fit.aic), 1),
                "bic": round(_sf(fit.bic), 1),
                "forecast_points": fc_pts,
                "correlogram": corr_pts,
                "conclusion": (
                    f"Modelo ARIMA(1,1,1) ajustado para '{target}' (AIC = {fit.aic:.1f}). "
                    f"Último valor observado: {s.iloc[-1]:.2f} → Projeção T+12: {fc_m.iloc[-1]:.2f}."
                ),
            }
    except Exception:
        pass

    # Fallback / Padrão: Regressão MQO com Erros Robustos Newey-West HAC
    y_col = num_cols[0]
    x_cols = num_cols[1 : min(6, len(num_cols))]
    X = sm.add_constant(clean[x_cols])
    y = clean[y_col]
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(4, max(1, len(clean) // 15))})
    coefs = [
        {
            "term": "Intercepto" if k == "const" else str(k),
            "coef": round(_sf(ols.params[k]), 4),
            "std_err": round(_sf(ols.bse[k]), 4),
            "t_stat": round(_sf(ols.tvalues[k]), 3),
            "pvalue": round(_sf(ols.pvalues[k], 1.0), 4),
            "sig": bool(ols.pvalues[k] < 0.05),
        }
        for k in ols.params.index
    ]
    step = max(1, len(clean) // 450)
    pts = [
        {"date": x_aligned[i], "actual": round(_sf(y.iloc[i]), 3), "fitted": round(_sf(ols.fittedvalues.iloc[i]), 3), "resid": round(_sf(ols.resid.iloc[i]), 3)}
        for i in range(0, len(clean), step)
    ]
    return {
        "model_type": "ols_regression",
        "short_label": "MQO Newey-West HAC",
        "dep_var": str(y_col),
        "r2": round(_sf(ols.rsquared), 4),
        "adj_r2": round(_sf(ols.rsquared_adj), 4),
        "f_stat": round(_sf(ols.fvalue), 2),
        "nobs": int(ols.nobs),
        "coefs": coefs,
        "points": pts,
        "conclusion": (
            f"Regressão MQO (erros robustos Newey-West HAC) explicando '{y_col}' por {', '.join(str(c) for c in x_cols)}: "
            f"R² = {ols.rsquared * 100:.1f}% (R² ajustado = {ols.rsquared_adj * 100:.1f}%, N = {int(ols.nobs)})."
        ),
    }


# Cache em memória do último dataframe exportável por estudo
_LAST_EXPORT_DF: Dict[str, pd.DataFrame] = {}


def execute_study(code: str, params: Dict[str, Any]) -> Dict[str, Any]:
    study = STUDIES_BY_CODE.get(code.upper())
    if not study:
        raise KeyError(f"Estudo '{code}' não encontrado no catálogo.")

    mtype = study["model_type"]
    if mtype == "dedicated_m01":
        res = _run_m01_dominancia_fiscal(params)
    elif mtype == "dedicated_m02":
        res = _run_m02_divida_publica(params)
    elif mtype == "dedicated_m06":
        res = _run_m06_regra_taylor(params)
    elif mtype == "dedicated_m12":
        res = _run_m12_inercia_ipca(params)
    elif mtype == "dedicated_m13":
        res = _run_m13_sarima_ipca(params)
    elif mtype == "dedicated_m14":
        res = _run_m14_curva_phillips(params)
    else:
        res = _run_thematic_study(study, params)

    export_df = res.pop("export_df", None)
    if isinstance(export_df, pd.DataFrame):
        _LAST_EXPORT_DF[study["code"]] = export_df

    return {
        "study": study,
        "result": res,
    }


def get_study_export_csv(code: str) -> bytes:
    up = code.upper()
    if up not in _LAST_EXPORT_DF:
        execute_study(up, {})
    df = _LAST_EXPORT_DF.get(up)
    if df is None:
        raise KeyError(f"Nenhum dado exportável para '{up}'.")
    return df.to_csv(index=False).encode("utf-8")
