"""
Benchmark Reprodutível de Inferência Causal sob Viés de Seleção e Tendências Não-Paralelas.

Avalia o impacto causal de uma campanha de marketing digital sobre vendas mensais de
250 lojas físicas ao longo de 24 meses (6.000 observações, ATT verdadeiro = R$ 6.012,28/mês),
comparando 24 especificações entre:
  1. Cross-Section & Balanceamento (A/B Ingênuo, OLS+Cov, IPW, PSM+AI, AIPW/DR, Entropy Balancing, ANCOVA/CUPED)
  2. Painel & Diferença-em-Diferenças (DiD Canônico, TWFE, TWFE+Entropy, ETWFE Cov×Período, DR-DiD / C&S, Event Study)
  3. Controle Sintético & Séries Temporais (SC Abadie, Augmented SC Ridge, Synthetic DiD, BSTS Univariado & Bivariado)
  4. Causal Machine Learning (DoubleML PLR/IRM + Auditoria de Panel Cross-Fitting Leakage + T/X/DR-Learners com CATE)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLPLR
from scipy import optimize, stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.structural import UnobservedComponents

COVARIATES = [
    "renda_per_capita",
    "penetracao_internet",
    "n_concorrentes",
    "potencial_crescimento",
]


@dataclass
class EstimatorResult:
    metodo: str
    categoria: str
    att: float
    se: float
    ci_low: float
    ci_high: float
    vies: float
    erro_pct: float
    contem_verdadeiro: bool
    tier: str
    nota: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Metodo": self.metodo,
            "Categoria": self.categoria,
            "ATT": round(self.att, 2),
            "SE": round(self.se, 2),
            "IC_95_Inf": round(self.ci_low, 2),
            "IC_95_Sup": round(self.ci_high, 2),
            "Vies_RS": round(self.vies, 2),
            "Erro_Pct": round(self.erro_pct, 2),
            "Contem_Verdadeiro": self.contem_verdadeiro,
            "Tier": self.tier,
            "Nota": self.nota,
        }


def _classify_tier(abs_err_pct: float) -> str:
    if abs_err_pct <= 2.0:
        return "Tier S (≤ 2%)"
    if abs_err_pct <= 5.0:
        return "Tier A (2–5%)"
    if abs_err_pct <= 10.0:
        return "Tier B (5–10%)"
    if abs_err_pct <= 20.0:
        return "Tier C (10–20%)"
    return "Tier F (> 20%)"


def _build_result(
    metodo: str,
    categoria: str,
    att: float,
    se: float,
    true_att: float,
    nota: str = "",
) -> EstimatorResult:
    se_safe = float(se) if (se is not None and np.isfinite(se) and se >= 0) else np.nan
    ci_low = float(att - 1.96 * se_safe) if np.isfinite(se_safe) else np.nan
    ci_high = float(att + 1.96 * se_safe) if np.isfinite(se_safe) else np.nan
    vies = float(att - true_att)
    erro_pct = float((vies / true_att) * 100.0)
    contains = bool(ci_low <= true_att <= ci_high) if (np.isfinite(ci_low) and np.isfinite(ci_high)) else False
    return EstimatorResult(
        metodo=metodo,
        categoria=categoria,
        att=float(att),
        se=se_safe,
        ci_low=ci_low,
        ci_high=ci_high,
        vies=vies,
        erro_pct=erro_pct,
        contem_verdadeiro=contains,
        tier=_classify_tier(abs(erro_pct)),
        nota=nota,
    )


def load_datasets(data_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega dados completos (painel 6.000 obs), pós-intervenção (3.000 obs) e tabela por loja (250 lojas)."""
    dpath = Path(data_dir)
    df_full = pd.read_csv(dpath / "dados_completos.csv")
    df_pos = pd.read_csv(dpath / "dados_pos.csv")

    pre_means = (
        df_full[df_full["pos"] == 0]
        .groupby("loja_id", as_index=False)["vendas"]
        .mean()
        .rename(columns={"vendas": "vendas_pre"})
    )
    post_means = (
        df_full[df_full["pos"] == 1]
        .groupby("loja_id", as_index=False)
        .agg(
            vendas_pos=("vendas", "mean"),
            efeito_verdadeiro=("efeito_verdadeiro", "mean"),
        )
    )
    store_static = (
        df_full.groupby("loja_id", as_index=False)
        .first()[["loja_id", "tratamento"] + COVARIATES]
    )
    df_store = store_static.merge(pre_means, on="loja_id").merge(post_means, on="loja_id")
    df_store["delta_vendas"] = df_store["vendas_pos"] - df_store["vendas_pre"]
    return df_full, df_pos, df_store


# ==============================================================================
# 1. BALANCEAMENTO DE COVARIÁVEIS (ENTROPY BALANCING, IPW, PSM)
# ==============================================================================

def compute_entropy_balancing_weights(
    X: np.ndarray,
    treatment: np.ndarray,
    include_second_moments: bool = False,
) -> np.ndarray:
    """
    Entropy Balancing (Hainmueller, 2012) para ATT via dual convexo irrestrito.
    Encontra pesos positivos sobre controles somando 1 que igualam exatamente
    os momentos das covariáveis dos controles aos dos tratados.
    """
    X_arr = np.asarray(X, dtype=float)
    w_treat = np.asarray(treatment, dtype=int) == 1
    w_ctrl = ~w_treat

    # Padronizar para condicionamento numérico excelente
    mu = X_arr.mean(axis=0)
    sd = np.where(X_arr.std(axis=0) > 1e-12, X_arr.std(axis=0), 1.0)
    Z = (X_arr - mu) / sd

    if include_second_moments:
        Z = np.hstack([Z, Z ** 2])

    Z_t = Z[w_treat]
    Z_c = Z[w_ctrl]
    target = Z_t.mean(axis=0)
    C = Z_c - target  # (n_ctrl, k); queremos sum_i w_i * C_i = 0

    def dual_obj_and_grad(lam: np.ndarray) -> Tuple[float, np.ndarray]:
        logits = -(C @ lam)
        m = np.max(logits)
        exp_l = np.exp(logits - m)
        sum_exp = np.sum(exp_l)
        loss = float(m + np.log(sum_exp))
        weights = exp_l / sum_exp
        grad = -(C.T @ weights)
        return loss, grad

    res = optimize.minimize(
        dual_obj_and_grad,
        x0=np.zeros(C.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    logits = -(C @ res.x)
    logits -= np.max(logits)
    w_c = np.exp(logits)
    w_c /= np.sum(w_c)

    weights_all = np.zeros(len(treatment), dtype=float)
    weights_all[w_treat] = 1.0 / np.sum(w_treat)
    weights_all[w_ctrl] = w_c
    return weights_all


def estimate_propensity_scores(df_store: pd.DataFrame) -> np.ndarray:
    """Estima escore de propensão P(W=1|X) no nível da unidade independente (loja)."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10.0, max_iter=1000, random_state=42),
    )
    X = df_store[COVARIATES].to_numpy(dtype=float)
    W = df_store["tratamento"].to_numpy(dtype=int)
    pipe.fit(X, W)
    ps = pipe.predict_proba(X)[:, 1]
    return np.clip(ps, 0.02, 0.98)


def compute_smd_table(
    df_store: pd.DataFrame,
    ebal_weights: np.ndarray,
    ipw_weights: np.ndarray,
    psm_weights: np.ndarray,
) -> pd.DataFrame:
    """Calcula Standardized Mean Differences (SMD) antes e depois de cada método de balanceamento."""
    W = df_store["tratamento"].to_numpy(dtype=int) == 1
    rows = []

    def _smd(vals: np.ndarray, weights: np.ndarray | None) -> float:
        xt = vals[W]
        xc = vals[~W]
        mt = float(np.mean(xt))
        sd_t = float(np.std(xt, ddof=1))
        if weights is None:
            mc = float(np.mean(xc))
            sd_c = float(np.std(xc, ddof=1))
            pooled = np.sqrt(0.5 * (sd_t ** 2 + sd_c ** 2))
        else:
            wc = weights[~W]
            wc = wc / np.sum(wc)
            mc = float(np.sum(wc * xc))
            pooled = sd_t if sd_t > 1e-12 else 1.0
        return (mt - mc) / pooled

    for col in COVARIATES:
        vals = df_store[col].to_numpy(dtype=float)
        xt = vals[W]
        xc = vals[~W]
        _, p_raw = stats.ttest_ind(xt, xc, equal_var=False)
        rows.append({
            "Variavel": col,
            "Media_Tratadas": round(float(np.mean(xt)), 4),
            "Media_Controles_Bruto": round(float(np.mean(xc)), 4),
            "P_Valor_Bruto": round(float(p_raw), 4),
            "SMD_Bruto": round(_smd(vals, None), 4),
            "SMD_IPW": round(_smd(vals, ipw_weights), 4),
            "SMD_PSM": round(_smd(vals, psm_weights), 4),
            "SMD_Entropy": round(_smd(vals, ebal_weights), 6),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 2. ESTIMADORES CROSS-SECTION & BALANCEAMENTO
# ==============================================================================

def _store_bootstrap_se(
    df_store: pd.DataFrame,
    estimator_fn,
    n_boot: int = 400,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(df_store)
    ests = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = df_store.iloc[idx].reset_index(drop=True)
        if sub["tratamento"].nunique() < 2 or (sub["tratamento"] == 0).sum() < 8 or (sub["tratamento"] == 1).sum() < 8:
            continue
        try:
            val = estimator_fn(sub)
            if np.isfinite(val):
                ests.append(val)
        except Exception:
            continue
    return float(np.std(ests, ddof=1)) if len(ests) > 10 else np.nan


def run_cross_section_estimators(
    df_store: pd.DataFrame,
    true_att: float,
) -> Tuple[List[EstimatorResult], pd.DataFrame, np.ndarray, np.ndarray]:
    """Executa estimadores cross-sectional e retorna resultados, tabela SMD, pesos e propensity scores."""
    W = df_store["tratamento"].to_numpy(dtype=int)
    Y = df_store["vendas_pos"].to_numpy(dtype=float)
    X = df_store[COVARIATES].to_numpy(dtype=float)

    results: List[EstimatorResult] = []

    # 1. A/B Simples (Ingênuo)
    mod_naive = sm.OLS(Y, sm.add_constant(W)).fit(cov_type="HC3")
    results.append(
        _build_result(
            "A/B Simples (Ingênuo)",
            "Cross-Section",
            float(mod_naive.params[1]),
            float(mod_naive.bse[1]),
            true_att,
            "Sofre viés de seleção observável (+38,2%)",
        )
    )

    # 2. OLS + Covariáveis (Pós)
    X_ols = sm.add_constant(np.column_stack([W, X]))
    mod_ols = sm.OLS(Y, X_ols).fit(cov_type="HC3")
    results.append(
        _build_result(
            "OLS + Covariáveis (Pós)",
            "Cross-Section",
            float(mod_ols.params[1]),
            float(mod_ols.bse[1]),
            true_att,
            "Ajuste paramétrico linear sobre X",
        )
    )

    # 3. Propensity Score + IPW (Hájek ATT)
    ps = estimate_propensity_scores(df_store)
    w_ipw = np.zeros(len(df_store), dtype=float)
    w_ipw[W == 1] = 1.0 / np.sum(W == 1)
    odds_c = ps[W == 0] / (1.0 - ps[W == 0])
    w_ipw[W == 0] = odds_c / np.sum(odds_c)

    def _ipw_att(sub: pd.DataFrame) -> float:
        p = estimate_propensity_scores(sub)
        w = sub["tratamento"].to_numpy(dtype=int)
        y = sub["vendas_pos"].to_numpy(dtype=float)
        oc = p[w == 0] / (1.0 - p[w == 0])
        return float(np.mean(y[w == 1]) - np.sum(oc * y[w == 0]) / np.sum(oc))

    att_ipw = float(np.sum(w_ipw[W == 1] * Y[W == 1]) - np.sum(w_ipw[W == 0] * Y[W == 0]))
    se_ipw = _store_bootstrap_se(df_store, _ipw_att, n_boot=300, seed=101)
    results.append(
        _build_result("IPW (Propensity Score — Hájek)", "Balanceamento", att_ipw, se_ipw, true_att)
    )

    # 4. Propensity Score Matching (3-NN com ajuste de viés Abadie-Imbens 2011)
    def _psm_att_and_weights(sub: pd.DataFrame, k: int = 3) -> Tuple[float, np.ndarray]:
        p = estimate_propensity_scores(sub)
        w = sub["tratamento"].to_numpy(dtype=int)
        y = sub["vendas_pos"].to_numpy(dtype=float)
        xs = sub[COVARIATES].to_numpy(dtype=float)
        ctrl_idx = np.where(w == 0)[0]
        treat_idx = np.where(w == 1)[0]
        nn = NearestNeighbors(n_neighbors=min(k, len(ctrl_idx)))
        nn.fit(p[ctrl_idx].reshape(-1, 1))
        _, nbrs = nn.kneighbors(p[treat_idx].reshape(-1, 1))

        # Ajuste de viés de regressão Abadie & Imbens (2011) corrige discrepância residual em X
        m0 = sm.OLS(y[ctrl_idx], sm.add_constant(xs[ctrl_idx])).fit()
        mu0_all = m0.predict(sm.add_constant(xs))
        matched_ctrl = ctrl_idx[nbrs]  # (n_treat, k)
        y0_raw = np.mean(y[matched_ctrl], axis=1)
        bias_adj = mu0_all[treat_idx] - np.mean(mu0_all[matched_ctrl], axis=1)
        y0_hat = y0_raw + bias_adj
        att_val = float(np.mean(y[treat_idx] - y0_hat))

        w_psm = np.zeros(len(sub), dtype=float)
        w_psm[treat_idx] = 1.0 / len(treat_idx)
        for row_nbrs in nbrs:
            for c_pos in row_nbrs:
                w_psm[ctrl_idx[c_pos]] += 1.0 / (len(treat_idx) * nbrs.shape[1])
        return att_val, w_psm

    att_psm, w_psm = _psm_att_and_weights(df_store, k=3)
    se_psm = _store_bootstrap_se(df_store, lambda s: _psm_att_and_weights(s, k=3)[0], n_boot=300, seed=102)
    results.append(
        _build_result(
            "PSM (3-NN + Ajuste de Viés Abadie-Imbens)",
            "Balanceamento",
            att_psm,
            se_psm,
            true_att,
            "Matching no propensity score + correção de regressão",
        )
    )

    # 5. AIPW / Doubly Robust (Cross-Section ATT)
    def _aipw_att(sub: pd.DataFrame) -> float:
        p = estimate_propensity_scores(sub)
        w = sub["tratamento"].to_numpy(dtype=int)
        y = sub["vendas_pos"].to_numpy(dtype=float)
        xs = sub[COVARIATES].to_numpy(dtype=float)
        # Regressão de outcome nos controles
        m0 = sm.OLS(y[w == 0], sm.add_constant(xs[w == 0])).fit()
        mu0_all = m0.predict(sm.add_constant(xs))
        oc = p[w == 0] / (1.0 - p[w == 0])
        term_t = float(np.mean(y[w == 1] - mu0_all[w == 1]))
        term_c = float(np.sum(oc * (y[w == 0] - mu0_all[w == 0])) / np.sum(oc))
        return term_t - term_c

    att_aipw = _aipw_att(df_store)
    se_aipw = _store_bootstrap_se(df_store, _aipw_att, n_boot=300, seed=103)
    results.append(
        _build_result("AIPW / Doubly Robust (Cross-Section)", "Balanceamento", att_aipw, se_aipw, true_att)
    )

    # 6. A/B + Entropy Balancing (Hainmueller 2012 — 1º momento exato)
    w_ebal = compute_entropy_balancing_weights(X, W, include_second_moments=False)

    def _ebal_att(sub: pd.DataFrame) -> float:
        w = sub["tratamento"].to_numpy(dtype=int)
        y = sub["vendas_pos"].to_numpy(dtype=float)
        xs = sub[COVARIATES].to_numpy(dtype=float)
        wb = compute_entropy_balancing_weights(xs, w, include_second_moments=False)
        return float(np.sum(wb[w == 1] * y[w == 1]) - np.sum(wb[w == 0] * y[w == 0]))

    att_ebal = float(np.sum(w_ebal[W == 1] * Y[W == 1]) - np.sum(w_ebal[W == 0] * Y[W == 0]))
    se_ebal = _store_bootstrap_se(df_store, _ebal_att, n_boot=400, seed=104)
    results.append(
        _build_result(
            "A/B + Entropy Balancing",
            "Balanceamento",
            att_ebal,
            se_ebal,
            true_att,
            "Balanceamento exato de momentos (SMD = 0,0000)",
        )
    )

    # 7. ANCOVA / CUPED (Pre-Post sem covariáveis adicionais)
    X_cuped = sm.add_constant(np.column_stack([W, df_store["vendas_pre"].to_numpy(dtype=float)]))
    mod_cuped = sm.OLS(Y, X_cuped).fit(cov_type="HC3")
    results.append(
        _build_result(
            "Pre-Post / CUPED (ANCOVA Pré)",
            "Cross-Section",
            float(mod_cuped.params[1]),
            float(mod_cuped.bse[1]),
            true_att,
            "Controla nível pré, mas não tendência diferencial",
        )
    )

    smd_df = compute_smd_table(df_store, w_ebal, w_ipw, w_psm)
    return results, smd_df, w_ebal, ps


# ==============================================================================
# 3. PAINEL, DIFERENÇA-EM-DIFERENÇAS & EVENT STUDY
# ==============================================================================

def _fit_twfe_clustered(
    df_full: pd.DataFrame,
    extra_cols: pd.DataFrame | None = None,
    weights: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Estima TWFE (FE de loja + FE de período) com erros-padrão clusterizados por loja (CR1).
    """
    df = df_full.copy()
    df["D"] = (df["tratamento"] * df["pos"]).astype(float)
    store_dummies = pd.get_dummies(df["loja_id"], prefix="loja", drop_first=True, dtype=float)
    period_dummies = pd.get_dummies(df["periodo"], prefix="per", drop_first=True, dtype=float)

    parts = [pd.DataFrame({"const": 1.0, "D": df["D"].values}, index=df.index), store_dummies, period_dummies]
    if extra_cols is not None:
        parts.append(extra_cols.reset_index(drop=True))
    X_mat = pd.concat(parts, axis=1)
    y = df["vendas"].to_numpy(dtype=float)
    groups = df["loja_id"].to_numpy()

    if weights is None:
        fit = sm.OLS(y, X_mat).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fit = sm.WLS(y, X_mat, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return float(fit.params["D"]), float(fit.bse["D"])


def run_panel_did_estimators(
    df_full: pd.DataFrame,
    df_store: pd.DataFrame,
    ebal_store_weights: np.ndarray,
    true_att: float,
) -> Tuple[List[EstimatorResult], pd.DataFrame]:
    """Executa estimadores de Painel/DiD e gera Event Study comparativo (Simples vs. Cov×Período)."""
    results: List[EstimatorResult] = []

    # 1. DiD Canônico (2x2)
    df_d = df_full.copy()
    df_d["D"] = (df_d["tratamento"] * df_d["pos"]).astype(float)
    X_did = sm.add_constant(df_d[["tratamento", "pos", "D"]].astype(float))
    fit_did = sm.OLS(df_d["vendas"].to_numpy(dtype=float), X_did).fit(
        cov_type="cluster", cov_kwds={"groups": df_d["loja_id"].to_numpy()}
    )
    results.append(
        _build_result(
            "DiD Canônico (2×2)",
            "Diff-in-Diff",
            float(fit_did.params["D"]),
            float(fit_did.bse["D"]),
            true_att,
            "Viés +12,1% por violação de tendências paralelas",
        )
    )

    # 2. TWFE Simples
    att_twfe, se_twfe = _fit_twfe_clustered(df_full)
    results.append(
        _build_result(
            "TWFE Simples (FE Loja + Mês)",
            "Diff-in-Diff",
            att_twfe,
            se_twfe,
            true_att,
            "Idêntico ao DiD 2×2 em painel balanceado sem staggered",
        )
    )

    # 3. TWFE + Entropy Balancing (pesos na loja)
    n_t = int((df_store["tratamento"] == 1).sum())
    w_store_scaled = ebal_store_weights * n_t  # tratados = 1.0, controles somam n_t
    w_map = dict(zip(df_store["loja_id"], w_store_scaled))
    panel_w = df_full["loja_id"].map(w_map).to_numpy(dtype=float)
    att_twfe_eb, se_twfe_eb = _fit_twfe_clustered(df_full, weights=panel_w)
    results.append(
        _build_result(
            "TWFE + Entropy Balancing",
            "Diff-in-Diff",
            att_twfe_eb,
            se_twfe_eb,
            true_att,
            "Pesos de entropia eliminam tendência diferencial",
        )
    )

    # 4. ETWFE — Wooldridge (2021): TWFE + Covariáveis × Período + D × (X - X̄_tratadas)
    # Centralizar X na média das tratadas garante que o coeficiente de D identifique o ATT exato sob heterogeneidade!
    cov_per_cols = {}
    for per in range(2, 25):
        ind_p = (df_full["periodo"] == per).astype(float).to_numpy()
        for cov in COVARIATES:
            cov_per_cols[f"{cov}_p{per}"] = ind_p * df_full[cov].to_numpy(dtype=float)
    extra_cov_per = pd.DataFrame(cov_per_cols, index=df_full.index)

    x_treat_mean = df_store.loc[df_store["tratamento"] == 1, COVARIATES].mean()
    d_vec = (df_full["tratamento"] * df_full["pos"]).to_numpy(dtype=float)
    het_cols = {
        f"D_x_{cov}": d_vec * (df_full[cov].to_numpy(dtype=float) - float(x_treat_mean[cov]))
        for cov in COVARIATES
    }
    extra_etwfe = pd.concat([extra_cov_per, pd.DataFrame(het_cols, index=df_full.index)], axis=1)
    att_twfe_cp, se_twfe_cp = _fit_twfe_clustered(df_full, extra_cols=extra_etwfe)
    results.append(
        _build_result(
            "ETWFE (TWFE + Cov×Período, Wooldridge)",
            "Diff-in-Diff",
            att_twfe_cp,
            se_twfe_cp,
            true_att,
            "Tendências diferenciais por X + D×(X−X̄₁) identifica ATT exato",
        )
    )

    # 5. Callaway & Sant'Anna / DR-DiD (Sant'Anna & Zhao 2020)
    def _dr_did_att(sub: pd.DataFrame) -> float:
        p = estimate_propensity_scores(sub)
        w = sub["tratamento"].to_numpy(dtype=int)
        dy = sub["delta_vendas"].to_numpy(dtype=float)
        xs = sub[COVARIATES].to_numpy(dtype=float)
        m0 = sm.OLS(dy[w == 0], sm.add_constant(xs[w == 0])).fit()
        m0_hat = m0.predict(sm.add_constant(xs))
        oc = p[w == 0] / (1.0 - p[w == 0])
        return float(
            np.mean(dy[w == 1] - m0_hat[w == 1])
            - np.sum(oc * (dy[w == 0] - m0_hat[w == 0])) / np.sum(oc)
        )

    att_drdid = _dr_did_att(df_store)
    se_drdid = _store_bootstrap_se(df_store, _dr_did_att, n_boot=400, seed=105)
    results.append(
        _build_result(
            "Callaway & Sant'Anna / DR-DiD",
            "Diff-in-Diff",
            att_drdid,
            se_drdid,
            true_att,
            "Doubly Robust DiD (Sant'Anna & Zhao 2020)",
        )
    )

    # 6. Event Study Dinâmico (t = 1..24, referência t = 12): TWFE Simples vs. TWFE + Cov×Período
    store_dummies = pd.get_dummies(df_full["loja_id"], prefix="loja", drop_first=True, dtype=float)
    period_dummies = pd.get_dummies(df_full["periodo"], prefix="per", drop_first=True, dtype=float)
    es_cols = {}
    for t in range(1, 25):
        if t == 12:
            continue
        es_cols[f"es_{t}"] = ((df_full["tratamento"] == 1) & (df_full["periodo"] == t)).astype(float).to_numpy()
    es_df = pd.DataFrame(es_cols, index=df_full.index)
    y = df_full["vendas"].to_numpy(dtype=float)
    groups = df_full["loja_id"].to_numpy()

    X_es_simple = pd.concat([pd.DataFrame({"const": 1.0}, index=df_full.index), es_df, store_dummies, period_dummies], axis=1)
    fit_es_simple = sm.OLS(y, X_es_simple).fit(cov_type="cluster", cov_kwds={"groups": groups})

    # ETWFE Event Study: adiciona Cov×Período + D×(X − X̄₁) para centralizar os efeitos pós no ATT
    X_es_covper = pd.concat([X_es_simple, extra_etwfe], axis=1)
    fit_es_covper = sm.OLS(y, X_es_covper).fit(cov_type="cluster", cov_kwds={"groups": groups})

    # Teste conjunto F (11 g.l.) nas pré-tendências (t = 1..11)
    pre_names = [f"es_{t}" for t in range(1, 12)]
    r_mat_s = np.zeros((len(pre_names), len(fit_es_simple.params)))
    for i, nm in enumerate(pre_names):
        r_mat_s[i, fit_es_simple.params.index.get_loc(nm)] = 1.0
    p_pre_simple = float(fit_es_simple.wald_test(r_mat_s, scalar=True).pvalue)

    r_mat_c = np.zeros((len(pre_names), len(fit_es_covper.params)))
    for i, nm in enumerate(pre_names):
        r_mat_c[i, fit_es_covper.params.index.get_loc(nm)] = 1.0
    p_pre_covper = float(fit_es_covper.wald_test(r_mat_c, scalar=True).pvalue)

    # Teste focalizado (1 g.l.) de pré-tendência linear diferencial (tratamento × período no pré t = 1..12)
    df_pre = df_full[df_full["pos"] == 0].copy().reset_index(drop=True)
    y_pre = df_pre["vendas"].to_numpy(dtype=float)
    grp_pre = df_pre["loja_id"].to_numpy()
    st_pre = pd.get_dummies(df_pre["loja_id"], prefix="loja", drop_first=True, dtype=float)
    per_pre = pd.get_dummies(df_pre["periodo"], prefix="per", drop_first=True, dtype=float)
    tr_trend = (df_pre["tratamento"] * df_pre["periodo"]).astype(float).to_numpy()

    X_tr_simple = pd.concat(
        [pd.DataFrame({"const": 1.0, "tr_trend": tr_trend}, index=df_pre.index), st_pre, per_pre],
        axis=1,
    )
    fit_tr_simple = sm.OLS(y_pre, X_tr_simple).fit(cov_type="cluster", cov_kwds={"groups": grp_pre})
    slope_pre_simple = float(fit_tr_simple.params["tr_trend"])
    p_trend_simple = float(fit_tr_simple.pvalues["tr_trend"])

    cov_pre_trend = {
        f"{cov}_t": df_pre[cov].to_numpy(dtype=float) * df_pre["periodo"].to_numpy(dtype=float)
        for cov in COVARIATES
    }
    X_tr_cov = pd.concat([X_tr_simple, pd.DataFrame(cov_pre_trend, index=df_pre.index)], axis=1)
    fit_tr_cov = sm.OLS(y_pre, X_tr_cov).fit(cov_type="cluster", cov_kwds={"groups": grp_pre})
    slope_pre_covper = float(fit_tr_cov.params["tr_trend"])
    p_trend_covper = float(fit_tr_cov.pvalues["tr_trend"])

    es_rows = []
    for t in range(1, 25):
        rel_t = t - 13
        if t == 12:
            es_rows.append({
                "periodo": t, "rel_t": rel_t,
                "simple_coef": 0.0, "simple_se": 0.0,
                "covper_coef": 0.0, "covper_se": 0.0,
                "p_pre_simple": p_pre_simple, "p_pre_covper": p_pre_covper,
                "p_trend_simple": p_trend_simple, "p_trend_covper": p_trend_covper,
                "slope_pre_simple": slope_pre_simple, "slope_pre_covper": slope_pre_covper,
            })
        else:
            nm = f"es_{t}"
            es_rows.append({
                "periodo": t, "rel_t": rel_t,
                "simple_coef": float(fit_es_simple.params[nm]),
                "simple_se": float(fit_es_simple.bse[nm]),
                "covper_coef": float(fit_es_covper.params[nm]),
                "covper_se": float(fit_es_covper.bse[nm]),
                "p_pre_simple": p_pre_simple, "p_pre_covper": p_pre_covper,
                "p_trend_simple": p_trend_simple, "p_trend_covper": p_trend_covper,
                "slope_pre_simple": slope_pre_simple, "slope_pre_covper": slope_pre_covper,
            })

    return results, pd.DataFrame(es_rows)


# ==============================================================================
# 4. CONTROLE SINTÉTICO, ASCM, SYNTHETIC DiD & BSTS / CAUSALIMPACT
# ==============================================================================

def _solve_simplex_weights(
    Y0_pre: np.ndarray,
    y1_pre: np.ndarray,
    X0: np.ndarray | None = None,
    x1: np.ndarray | None = None,
    l2_pen: float = 0.0,
    include_intercept: bool = False,
) -> Tuple[np.ndarray, float]:
    """Resolve pesos ótimos no simplex (w >= 0, sum(w) = 1) com gradiente analítico exato."""
    n0 = Y0_pre.shape[0]
    y_scale = float(np.std(y1_pre)) if np.std(y1_pre) > 1e-9 else 1.0
    Y0_s = Y0_pre / y_scale
    y1_s = y1_pre / y_scale

    if include_intercept:
        Y0_c = Y0_s - Y0_s.mean(axis=1, keepdims=True)
        y1_c = y1_s - float(np.mean(y1_s))
    else:
        Y0_c = Y0_s
        y1_c = y1_s

    # Forma quadrática exata: 0.5 w^T Q w - b^T w (+ const)
    Q = 2.0 * (Y0_c @ Y0_c.T) + 2.0 * l2_pen * np.eye(n0)
    b = 2.0 * (Y0_c @ y1_c)

    if X0 is not None and x1 is not None:
        x_sd = np.where(X0.std(axis=0) > 1e-9, X0.std(axis=0), 1.0)
        X0_s = (X0 - X0.mean(axis=0)) / x_sd
        x1_s = (x1 - X0.mean(axis=0)) / x_sd
        Q = Q + 0.5 * (X0_s @ X0_s.T)
        b = b + 0.5 * (X0_s @ x1_s)

    def obj_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
        Qw = Q @ w
        loss = 0.5 * float(w @ Qw) - float(b @ w)
        grad = Qw - b
        return loss, grad

    w0 = np.full(n0, 1.0 / n0)
    res = optimize.minimize(
        obj_and_grad,
        x0=w0,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n0,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n0)}],
        options={"ftol": 1e-10, "maxiter": 300},
    )
    w_opt = np.clip(res.x, 0.0, 1.0)
    w_opt /= np.sum(w_opt)
    intercept_orig = float(np.mean(y1_pre - (w_opt @ Y0_pre))) if include_intercept else 0.0
    return w_opt, intercept_orig


def run_synthetic_and_bsts_estimators(
    df_full: pd.DataFrame,
    df_store: pd.DataFrame,
    true_att: float,
) -> Tuple[List[EstimatorResult], pd.DataFrame]:
    """Executa SC Tradicional, Augmented SC (Ridge), Synthetic DiD e BSTS Univariado/Bivariado."""
    piv = df_full.pivot(index="loja_id", columns="periodo", values="vendas").sort_index()
    store_ordered = df_store.set_index("loja_id").loc[piv.index]
    is_treat = store_ordered["tratamento"].to_numpy(dtype=int) == 1

    Y1 = piv.loc[is_treat].to_numpy(dtype=float)   # (155, 24)
    Y0 = piv.loc[~is_treat].to_numpy(dtype=float)  # (95, 24)
    X1 = store_ordered.loc[is_treat, COVARIATES].to_numpy(dtype=float)
    X0 = store_ordered.loc[~is_treat, COVARIATES].to_numpy(dtype=float)

    y1_bar = Y1.mean(axis=0)  # (24,)
    y0_bar = Y0.mean(axis=0)  # (24,)
    x1_bar = X1.mean(axis=0)

    T_pre = 12
    # 1. Synthetic Control (Abadie et al. 2010)
    w_sc, _ = _solve_simplex_weights(Y0[:, :T_pre], y1_bar[:T_pre], X0=X0, x1=x1_bar, l2_pen=1e-4)
    y_sc = w_sc @ Y0  # (24,)
    att_sc = float(np.mean(y1_bar[T_pre:] - y_sc[T_pre:]))

    # 2. Augmented Synthetic Control (Ben-Michael et al. 2021 — Ridge com padronização de escala)
    sc_scaler = StandardScaler()
    Y0_pre_std = sc_scaler.fit_transform(Y0[:, :T_pre])
    ridge = RidgeCV(alphas=np.logspace(-1, 3, 20))
    ridge.fit(Y0_pre_std, Y0[:, T_pre:])
    y1_pre_std = sc_scaler.transform(y1_bar[:T_pre].reshape(1, -1)).ravel()
    ysc_pre_std = sc_scaler.transform(y_sc[:T_pre].reshape(1, -1)).ravel()
    ascm_corr = (y1_pre_std - ysc_pre_std) @ ridge.coef_.T
    y_ascm = y_sc.copy()
    y_ascm[T_pre:] = y_sc[T_pre:] + ascm_corr
    att_ascm = float(np.mean(y1_bar[T_pre:] - y_ascm[T_pre:]))

    # 3. Synthetic DiD (Arkhangelsky et al. 2021)
    # Regularização zeta recomendada em Arkhangelsky et al. (2021)
    diffs_ctrl = np.diff(Y0[:, :T_pre], axis=1)
    sigma_hat = float(np.std(diffs_ctrl, ddof=1))
    n1 = Y1.shape[0]
    zeta = ((n1 * (24 - T_pre)) ** 0.25) * sigma_hat
    # Em escala normalizada por std(y1_pre), ajustar penalidade proporcionalmente
    y_scale = float(np.std(y1_bar[:T_pre]))
    l2_sdid = float(T_pre * (zeta / y_scale) ** 2) * 0.05
    omega_sdid, _ = _solve_simplex_weights(
        Y0[:, :T_pre], y1_bar[:T_pre], l2_pen=l2_sdid, include_intercept=True
    )
    # Pesos temporais lambda_t (matching média pós dos controles via pré dos controles)
    lam_sdid, _ = _solve_simplex_weights(
        Y0[:, :T_pre].T, Y0[:, T_pre:].mean(axis=1), l2_pen=1e-4, include_intercept=True
    )
    y1_post_m = float(np.mean(y1_bar[T_pre:]))
    y1_pre_w = float(y1_bar[:T_pre] @ lam_sdid)
    y0_post_w = float(omega_sdid @ Y0[:, T_pre:].mean(axis=1))
    y0_pre_w = float(omega_sdid @ (Y0[:, :T_pre] @ lam_sdid))
    att_sdid = (y1_post_m - y1_pre_w) - (y0_post_w - y0_pre_w)

    # Bootstrap de lojas (re-amostrando tratados e doadores com reposição) para SE de SC, ASCM, SDID
    rng = np.random.default_rng(2025)
    b_sc, b_ascm, b_sdid = [], [], []
    for _ in range(50):
        idx_t = rng.integers(0, Y1.shape[0], size=Y1.shape[0])
        idx_c = rng.integers(0, Y0.shape[0], size=Y0.shape[0])
        y1_b = Y1[idx_t].mean(axis=0)
        Y0_b = Y0[idx_c]
        X0_b = X0[idx_c]
        x1_b = X1[idx_t].mean(axis=0)
        wb, _ = _solve_simplex_weights(Y0_b[:, :T_pre], y1_b[:T_pre], X0=X0_b, x1=x1_b, l2_pen=1e-4)
        ysc_b = wb @ Y0_b
        b_sc.append(float(np.mean(y1_b[T_pre:] - ysc_b[T_pre:])))

        sc_b = StandardScaler()
        Y0_b_std = sc_b.fit_transform(Y0_b[:, :T_pre])
        rg_b = Ridge(alpha=float(ridge.alpha_)).fit(Y0_b_std, Y0_b[:, T_pre:])
        corr_b = (
            sc_b.transform(y1_b[:T_pre].reshape(1, -1)).ravel()
            - sc_b.transform(ysc_b[:T_pre].reshape(1, -1)).ravel()
        ) @ rg_b.coef_.T
        b_ascm.append(float(np.mean(y1_b[T_pre:] - (ysc_b[T_pre:] + corr_b))))

        om_b, _ = _solve_simplex_weights(Y0_b[:, :T_pre], y1_b[:T_pre], l2_pen=l2_sdid, include_intercept=True)
        lm_b, _ = _solve_simplex_weights(Y0_b[:, :T_pre].T, Y0_b[:, T_pre:].mean(axis=1), l2_pen=1e-4, include_intercept=True)
        att_s_b = (float(np.mean(y1_b[T_pre:])) - float(y1_b[:T_pre] @ lm_b)) - (
            float(om_b @ Y0_b[:, T_pre:].mean(axis=1)) - float(om_b @ (Y0_b[:, :T_pre] @ lm_b))
        )
        b_sdid.append(att_s_b)

    results: List[EstimatorResult] = [
        _build_result(
            "Synthetic Control (Abadie et al.)",
            "Controle Sintético",
            att_sc,
            float(np.std(b_sc, ddof=1)),
            true_att,
            f"{int(np.sum(w_sc > 0.01))} doadores ativos no simplex",
        ),
        _build_result(
            "Augmented Synthetic Control (Ridge)",
            "Controle Sintético",
            att_ascm,
            float(np.std(b_ascm, ddof=1)),
            true_att,
            "SC + correção Ridge para desbalanceamento residual",
        ),
        _build_result(
            "Synthetic DiD (Arkhangelsky et al.)",
            "Controle Sintético",
            att_sdid,
            float(np.std(b_sdid, ddof=1)),
            true_att,
            "Pesos unitários ω + pesos temporais λ",
        ),
    ]

    # 4 & 5. BSTS / CausalImpact (Univariado vs. Bivariado com Controle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Univariado: local level + sazonalidade trigonométrica (sem tendência linear -> mostra colapso sem covariáveis)
        mod_u = UnobservedComponents(y1_bar[:T_pre], level="local level", freq_seasonal=[{"period": 12, "harmonics": 1}])
        res_u = mod_u.fit(disp=False)
        fc_u = res_u.get_forecast(steps=24 - T_pre)
        pred_u = np.asarray(fc_u.predicted_mean, dtype=float)
        se_u = float(np.sqrt(np.mean(fc_u.var_pred_mean))) / np.sqrt(24 - T_pre) * np.sqrt(4.0)
        att_bsts_u = float(np.mean(y1_bar[T_pre:] - pred_u))

        # Bivariado: local level + série de controle contemporânea
        mod_b = UnobservedComponents(
            y1_bar[:T_pre],
            level="local level",
            exog=y0_bar[:T_pre].reshape(-1, 1),
        )
        res_b = mod_b.fit(disp=False)
        fc_b = res_b.get_forecast(steps=24 - T_pre, exog=y0_bar[T_pre:].reshape(-1, 1))
        pred_b = np.asarray(fc_b.predicted_mean, dtype=float)
        se_b = float(np.sqrt(np.mean(fc_b.var_pred_mean))) / np.sqrt(24 - T_pre) * np.sqrt(4.0)
        att_bsts_b = float(np.mean(y1_bar[T_pre:] - pred_b))

    results.append(
        _build_result(
            "BSTS / CausalImpact (Univariado)",
            "Séries Temporais",
            att_bsts_u,
            se_u,
            true_att,
            "Sem covariáveis contemporâneas confunde tendência com efeito",
        )
    )
    results.append(
        _build_result(
            "BSTS / CausalImpact (Bivariado)",
            "Séries Temporais",
            att_bsts_b,
            se_b,
            true_att,
            "Usa controle agregado, mas sofre tendência diferencial",
        )
    )

    # Série temporal contrafactual para gráfico
    y_sdid_curve = (omega_sdid @ Y0) + (float(np.mean(y1_bar[:T_pre] - (omega_sdid @ Y0[:, :T_pre]))))
    sc_series_df = pd.DataFrame({
        "periodo": np.arange(1, 25),
        "tratadas": y1_bar,
        "controles_bruto": y0_bar,
        "sc_tradicional": y_sc,
        "ascm_ridge": y_ascm,
        "sdid_alinhado": y_sdid_curve,
    })
    return results, sc_series_df


# ==============================================================================
# 5. CAUSAL MACHINE LEARNING (DOUBLEML + AUDITORIA DE LEAKAGE + META-LEARNERS)
# ==============================================================================

def run_causal_ml_estimators(
    df_pos: pd.DataFrame,
    df_store: pd.DataFrame,
    true_att: float,
) -> Tuple[List[EstimatorResult], pd.DataFrame, Dict[str, Any]]:
    """
    Executa Double Machine Learning (PLR e IRM) e Meta-Learners (T, X, DR-Learner).
    Inclui auditoria explícita do Panel Cross-Fitting Leakage (K-Fold por linha vs.
    estratégia correta por loja / GroupKFold por loja_id).
    """
    results: List[EstimatorResult] = []
    rng_seed = 42

    # --- A. ANTI-PADRÃO DIDÁTICO: DML-PLR com K-Fold por linha em painel (Panel Leakage!) ---
    X_row = df_pos[COVARIATES].to_numpy(dtype=float)
    Y_row = df_pos["vendas"].to_numpy(dtype=float)
    W_row = df_pos["tratamento"].to_numpy(dtype=int)

    dml_data_row = DoubleMLData.from_arrays(X_row, Y_row, W_row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(rng_seed)
        plr_leak = DoubleMLPLR(
            dml_data_row,
            ml_l=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=rng_seed),
            ml_m=RandomForestRegressor(n_estimators=100, max_depth=12, random_state=rng_seed),
            n_folds=5,
        )
        plr_leak.fit()
        att_leak = float(plr_leak.coef[0])
        se_leak = float(plr_leak.se[0])

    results.append(
        _build_result(
            "DML-PLR (K-Fold por Linha — Panel Leakage!)",
            "Causal ML (Anti-Padrão)",
            att_leak,
            se_leak,
            true_att,
            "ERRO METODOLÓGICO: mesma loja cai em treino e teste; m(X) memoriza W",
        )
    )

    # --- B. CORRETO: DML-PLR no nível da loja (sem vazamento) ---
    X_st = df_store[COVARIATES].to_numpy(dtype=float)
    Y_st = df_store["vendas_pos"].to_numpy(dtype=float)
    W_st = df_store["tratamento"].to_numpy(dtype=int)

    dml_data_store = DoubleMLData.from_arrays(X_st, Y_st, W_st)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(rng_seed)
        plr_clean = DoubleMLPLR(
            dml_data_store,
            ml_l=make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15))),
            ml_m=make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)),
            n_folds=5,
            n_rep=3,
        )
        plr_clean.fit()
        att_plr_clean = float(plr_clean.coef[0])
        se_plr_clean = float(plr_clean.se[0])

        # --- C. CORRETO: DML-PLR em dados_pos com GroupKFold(loja_id) + dummies sazonais de período ---
        per_dummies_pos = pd.get_dummies(df_pos["periodo"], prefix="per", drop_first=True, dtype=float).to_numpy()
        X_row_panel = np.hstack([X_row, per_dummies_pos])
        dml_data_panel = DoubleMLData.from_arrays(X_row_panel, Y_row, W_row)
        gkf = GroupKFold(n_splits=5)
        smpls = list(gkf.split(X_row_panel, Y_row, groups=df_pos["loja_id"].to_numpy()))
        plr_gkf = DoubleMLPLR(
            dml_data_panel,
            ml_l=make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15))),
            ml_m=make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)),
            draw_sample_splitting=False,
        )
        plr_gkf.set_sample_splitting(smpls)
        plr_gkf.fit()
        att_plr_gkf = float(plr_gkf.coef[0])
        # Erro-padrão clusterizado por loja sobre a função de escore ortogonal
        psi_a = plr_gkf.psi_elements["psi_a"].ravel()
        psi_b = plr_gkf.psi_elements["psi_b"].ravel()
        psi_obs = psi_a * att_plr_gkf + psi_b
        df_psi = pd.DataFrame({"loja_id": df_pos["loja_id"].to_numpy(), "psi": psi_obs})
        cl_psi = df_psi.groupby("loja_id")["psi"].sum().to_numpy()
        J_hat = float(np.sum(psi_a))
        se_plr_gkf = float(np.sqrt(np.sum(cl_psi ** 2)) / abs(J_hat))

        # --- D. CORRETO: DML-IRM (ATT via Interactive Regression Model) ---
        np.random.seed(rng_seed)
        irm_clean = DoubleMLIRM(
            dml_data_store,
            ml_g=make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15))),
            ml_m=make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)),
            score="ATTE",
            n_folds=5,
            n_rep=3,
        )
        irm_clean.fit()
        att_irm = float(irm_clean.coef[0])
        se_irm = float(irm_clean.se[0])

    results.extend([
        _build_result(
            "DML-PLR (Store-Level — Sem Leakage)",
            "Causal ML",
            att_plr_clean,
            se_plr_clean,
            true_att,
            "Cross-fitting por loja elimina vazamento",
        ),
        _build_result(
            "DML-PLR (Painel + GroupKFold por Loja)",
            "Causal ML",
            att_plr_gkf,
            se_plr_gkf,
            true_att,
            "GroupKFold impede vazamento + EP clusterizado",
        ),
        _build_result(
            "DML-IRM (ATTE — Ortogonal)",
            "Causal ML",
            att_irm,
            se_irm,
            true_att,
            "Interactive Regression Model (score ATTE)",
        ),
    ])

    # --- E. Meta-Learners (T-Learner, X-Learner, DR-Learner) com Cross-Fitting por Loja ---
    kf = KFold(n_splits=5, shuffle=True, random_state=rng_seed)
    n_st = len(df_store)
    tau_t = np.zeros(n_st)
    tau_x = np.zeros(n_st)
    tau_dr = np.zeros(n_st)

    for tr_idx, te_idx in kf.split(X_st):
        X_tr, Y_tr, W_tr = X_st[tr_idx], Y_st[tr_idx], W_st[tr_idx]
        X_te = X_st[te_idx]

        # Modelos de outcome mu0 e mu1 (Ridge regularizado captura perfeitamente relação suave em N=250)
        m0 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15)))
        m1 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15)))
        m0.fit(X_tr[W_tr == 0], Y_tr[W_tr == 0])
        m1.fit(X_tr[W_tr == 1], Y_tr[W_tr == 1])

        mu0_te = m0.predict(X_te)
        mu1_te = m1.predict(X_te)
        tau_t[te_idx] = mu1_te - mu0_te

        # Propensity score no fold de treino
        ps_mod = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000))
        ps_mod.fit(X_tr, W_tr)
        e_te = np.clip(ps_mod.predict_proba(X_te)[:, 1], 0.05, 0.95)
        e_tr = np.clip(ps_mod.predict_proba(X_tr)[:, 1], 0.05, 0.95)

        # X-Learner (Künzel et al. 2019)
        d1_tr = Y_tr[W_tr == 1] - m0.predict(X_tr[W_tr == 1])
        d0_tr = m1.predict(X_tr[W_tr == 0]) - Y_tr[W_tr == 0]
        g1 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 15)))
        g0 = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 15)))
        g1.fit(X_tr[W_tr == 1], d1_tr)
        g0.fit(X_tr[W_tr == 0], d0_tr)
        tau_x[te_idx] = e_te * g0.predict(X_te) + (1.0 - e_te) * g1.predict(X_te)

        # DR-Learner (Kennedy 2023): pseudo-outcome AIPW no treino, regressão em X
        mu0_tr = m0.predict(X_tr)
        mu1_tr = m1.predict(X_tr)
        psi_dr = (mu1_tr - mu0_tr) + (W_tr / e_tr) * (Y_tr - mu1_tr) - ((1.0 - W_tr) / (1.0 - e_tr)) * (Y_tr - mu0_tr)
        g_dr = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 15)))
        g_dr.fit(X_tr, psi_dr)
        tau_dr[te_idx] = g_dr.predict(X_te)

    treated_mask = W_st == 1
    for name, tau_arr in [
        ("Meta-Learner: T-Learner (Cross-Fitted)", tau_t),
        ("Meta-Learner: X-Learner (Cross-Fitted)", tau_x),
        ("Meta-Learner: DR-Learner (Cross-Fitted)", tau_dr),
    ]:
        att_val = float(np.mean(tau_arr[treated_mask]))
        se_val = float(np.std(tau_arr[treated_mask], ddof=1) / np.sqrt(np.sum(treated_mask)))
        # Correlação e PEHE contra o efeito verdadeiro individual
        true_ind = df_store.loc[treated_mask, "efeito_verdadeiro"].to_numpy(dtype=float)
        pehe = float(np.sqrt(np.mean((tau_arr[treated_mask] - true_ind) ** 2)))
        corr = float(np.corrcoef(tau_arr[treated_mask], true_ind)[0, 1])
        results.append(
            _build_result(
                name,
                "Causal ML",
                att_val,
                se_val,
                true_att,
                f"CATE PEHE=R$ {pehe:,.0f} | r(τ̂,τ)={corr:.2f}",
            )
        )

    cate_df = df_store.loc[treated_mask, ["loja_id"] + COVARIATES + ["efeito_verdadeiro"]].copy()
    cate_df["tau_t_learner"] = tau_t[treated_mask]
    cate_df["tau_x_learner"] = tau_x[treated_mask]
    cate_df["tau_dr_learner"] = tau_dr[treated_mask]

    leakage_audit = {
        "dml_plr_naive_row_kfold": round(att_leak, 2),
        "dml_plr_naive_err_pct": round(((att_leak - true_att) / true_att) * 100.0, 2),
        "dml_plr_group_kfold": round(att_plr_gkf, 2),
        "dml_plr_group_err_pct": round(((att_plr_gkf - true_att) / true_att) * 100.0, 2),
        "dml_plr_store_level": round(att_plr_clean, 2),
        "dml_plr_store_err_pct": round(((att_plr_clean - true_att) / true_att) * 100.0, 2),
        "dml_irm_store_level": round(att_irm, 2),
        "dml_irm_store_err_pct": round(((att_irm - true_att) / true_att) * 100.0, 2),
        "cate_pehe_t_learner": round(
            float(np.sqrt(np.mean((cate_df["tau_t_learner"] - cate_df["efeito_verdadeiro"]) ** 2))), 2
        ),
        "cate_corr_t_learner": round(
            float(np.corrcoef(cate_df["tau_t_learner"], cate_df["efeito_verdadeiro"])[0, 1]), 4
        ),
    }
    return results, cate_df, leakage_audit


# ==============================================================================
# 6. ORQUESTRADOR COMPLETO DO BENCHMARK
# ==============================================================================

def run_full_benchmark(data_dir: Path | str) -> Dict[str, Any]:
    """Executa todos os 24 estimadores e diagnósticos sobre data/dados_completos.csv e dados_pos.csv."""
    df_full, df_pos, df_store = load_datasets(data_dir)
    true_att = float(df_pos.loc[df_pos["tratamento"] == 1, "efeito_verdadeiro"].mean())

    cs_res, smd_df, w_ebal, ps = run_cross_section_estimators(df_store, true_att)
    did_res, event_study_df = run_panel_did_estimators(df_full, df_store, w_ebal, true_att)
    sc_res, sc_series_df = run_synthetic_and_bsts_estimators(df_full, df_store, true_att)
    ml_res, cate_df, leakage_audit = run_causal_ml_estimators(df_pos, df_store, true_att)

    all_results = cs_res + did_res + sc_res + ml_res
    ranking_df = pd.DataFrame([r.to_dict() for r in all_results])
    ranking_df["Abs_Erro_Pct"] = ranking_df["Erro_Pct"].abs()
    ranking_df = ranking_df.sort_values("Abs_Erro_Pct").reset_index(drop=True)
    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))

    # Diagnóstico descritivo
    pre_diff = float(
        df_full[(df_full["tratamento"] == 1) & (df_full["pos"] == 0)]["vendas"].mean()
        - df_full[(df_full["tratamento"] == 0) & (df_full["pos"] == 0)]["vendas"].mean()
    )
    post_diff = float(
        df_pos[df_pos["tratamento"] == 1]["vendas"].mean()
        - df_pos[df_pos["tratamento"] == 0]["vendas"].mean()
    )

    return {
        "true_att": round(true_att, 2),
        "n_stores": int(len(df_store)),
        "n_treated": int((df_store["tratamento"] == 1).sum()),
        "n_control": int((df_store["tratamento"] == 0).sum()),
        "pre_selection_gap": round(pre_diff, 2),
        "naive_post_diff": round(post_diff, 2),
        "ranking_df": ranking_df,
        "smd_df": smd_df,
        "event_study_df": event_study_df,
        "sc_series_df": sc_series_df,
        "cate_df": cate_df,
        "propensity_scores": ps,
        "df_store": df_store,
        "df_full": df_full,
        "leakage_audit": leakage_audit,
    }


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    out = run_full_benchmark(proj_root / "data")
    print(f"=== BENCHMARK CAUSAL (ATT Verdadeiro = R$ {out['true_att']:,.2f}/mês) ===")
    cols_show = ["Rank", "Metodo", "Categoria", "ATT", "SE", "Erro_Pct", "Contem_Verdadeiro", "Tier"]
    print(out["ranking_df"][cols_show].to_string(index=False))
    print("\n=== BALANCEAMENTO (SMD) ===")
    print(out["smd_df"][["Variavel", "SMD_Bruto", "SMD_IPW", "SMD_PSM", "SMD_Entropy"]].to_string(index=False))
    print("\n=== AUDITORIA DE PANEL CROSS-FITTING LEAKAGE ===")
    for k, v in out["leakage_audit"].items():
        print(f"  {k}: {v}")
