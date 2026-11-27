"""
Motor Reprodutível de Causal Machine Learning & Uplift Modeling para Promoções.

Reproduz e estende os dois estudos de caso do paper DoorDash (KDD 2025):
  "Causal Machine Learning for Promotions: Industry Evidence and Applications"
  (Muxi Xu, Kun Hu, Sudeep Das, Bruce Wang — DoorDash Inc.)

Módulos incluídos:
  1. Caso 3.1 — Targeting Discreto (Binário):
     - Simulação realista com confundimento observável e heterogeneidade rica (Persuadables,
       Sure Things, Lost Causes e Sleeping Dogs)
     - 6 estimadores: S-Learner, T-Learner, X-Learner, DML-PLR (constante, cross-fitted),
       DR-Learner (DML Heterogêneo) e R-Learner (Nie & Wager, 2021)
     - Auditoria metodológica: Qini Ingênuo (viesado sob confundimento) vs. DR-AUUC
       (debiased out-of-sample) e Oracle AUUC / PEHE / r(τ̂, τ)
     - Simulador de política de targeting sensível a custos (Fronteira Custo × Pedidos Incrementais)

  2. Caso 3.2 — Profundidade de Desconto Contínuo (Dose-Response):
     - Simulação com função dose-resposta côncava (retornos decrescentes em t ∈ [$0, $5]) e
       non-compliance + holdout de controle (15%)
     - Comparação entre DML Contínuo Ingênuo (pseudo-outcome recortado sem pesos) vs.
       R-Learner Contínuo Ponderado (sample_weight = Z_res², sem viés de truncamento)
     - Otimização Lagrangiana de DoorDash (Eq. 2.2.3): max_t [V·ΔP(t, x) − λ·c(t, x)]
       traçando a Fronteira de Pareto (Desconto Uniforme vs. Personalizado vs. Oráculo)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DISCRETE_FEATURES = [
    "ordem_historica",
    "engajamento_vertical",
    "redemption_hist",
    "dias_desde_ultimo",
]

CONTINUOUS_FEATURES = [
    "orders_last_30d",
    "avg_order_value",
    "days_since_last_order",
    "is_new_user",
    "merchant_fav_count",
    "grocery_category_engagement",
    "region_gmv_tier_encoded",
]


# ==============================================================================
# 1. SIMULAÇÃO DE DADOS — CASO 3.1 (TARGETING DISCRETO) & CASO 3.2 (CONTÍNUO)
# ==============================================================================

def simulate_discrete_promo_data(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Caso 3.1 (DoorDash Non-Restaurant Delivery):
    Tratamento binário (T ∈ {0, 1}) com forte confundimento observável e efeito
    heterogêneo não-linear (incluindo Persuadables, Sure Things e Sleeping Dogs).
    """
    rng = np.random.default_rng(seed)

    ordem_historica = rng.poisson(3, size=n)
    engajamento_vertical = rng.beta(2, 5, size=n)
    redemption_hist = rng.binomial(5, 0.3, size=n)
    dias_desde_ultimo = rng.exponential(14.0, size=n)

    # Confundimento realista: usuários mais ativos e engajados historicamente receberam mais promoções
    ps_logit = -1.0 + 0.50 * ordem_historica + 2.00 * engajamento_vertical
    propensity_true = expit(ps_logit)
    T = (rng.random(n) < propensity_true).astype(int)

    # Probabilidade base de conversão sem promoção p0(X)
    y0_logit = (
        -1.50
        + 0.30 * ordem_historica
        + 1.20 * engajamento_vertical
        - 0.05 * dias_desde_ultimo
    )
    p0_true = expit(y0_logit)

    # CATE verdadeiro heterogêneo:
    # - Baixo engajamento na vertical (novos na categoria) + histórico de resgate = Persuadables
    # - Usuários super-ativos com alto engajamento = Sure Things / leves Sleeping Dogs
    tau_raw = (
        0.24 * ((1.0 - engajamento_vertical) ** 2)
        + 0.08 * (redemption_hist >= 2).astype(float)
        - 0.06 * (ordem_historica >= 5).astype(float)
        - 0.04
    )
    p1_true = np.clip(p0_true + tau_raw, 0.01, 0.99)
    true_cate = p1_true - p0_true

    prob_obs = np.where(T == 1, p1_true, p0_true)
    Y = (rng.random(n) < prob_obs).astype(int)

    return pd.DataFrame({
        "ordem_historica": ordem_historica,
        "engajamento_vertical": engajamento_vertical,
        "redemption_hist": redemption_hist,
        "dias_desde_ultimo": dias_desde_ultimo,
        "T": T,
        "Y": Y,
        "propensity_true": propensity_true,
        "p0_true": p0_true,
        "p1_true": p1_true,
        "true_cate": true_cate,
    })


def simulate_continuous_discount_data(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """
    Caso 3.2 (DoorDash Restaurant Discount Depth):
    Tratamento contínuo t ∈ [$0, $5] com holdout de controle (15%), confundimento
    por tipo de usuário, non-compliance (caps de cupom) e resposta côncava com
    retornos decrescentes: ΔP(t, X) = τ(X) · ln(1 + t).
    """
    rng = np.random.default_rng(seed)

    orders_last_30d = rng.poisson(3, size=n)
    avg_order_value = rng.gamma(shape=2.0, scale=30.0, size=n)
    days_since_last_order = rng.exponential(5.0, size=n)
    is_new_user = rng.binomial(1, 0.30, size=n)
    merchant_fav_count = rng.poisson(5, size=n)
    grocery_engagement = rng.beta(2, 5, size=n)
    region_tier = rng.choice(["low", "medium", "high"], size=n, p=[0.4, 0.4, 0.2])
    region_enc = pd.Series(region_tier).map({"low": 0, "medium": 1, "high": 2}).to_numpy(dtype=int)

    # Probabilidade base de conversão (t = 0):
    # Usuários existentes têm alta conversão orgânica ("Sure Things"); novos têm baixa conversão orgânica
    p0_true = np.clip(
        0.34
        + 0.08 * np.log1p(orders_last_30d)
        + 0.0008 * avg_order_value
        - 0.03 * np.log1p(days_since_last_order)
        - 0.28 * is_new_user
        + 0.012 * merchant_fav_count
        + 0.22 * grocery_engagement
        + 0.03 * region_enc,
        0.06,
        0.72,
    )

    # Sensibilidade marginal verdadeira τ(X) por unidade de dose z = ln(1 + t):
    # Forte heterogeneidade realista (DoorDash Sec. 3.2):
    # - Existentes habituais: baixa elasticidade a desconto (τ ≈ 0.005–0.025)
    # - Novos / baixo engajamento: altíssima elasticidade (τ ≈ 0.11–0.20)
    true_tau = np.clip(
        0.006
        + 0.135 * is_new_user
        + 0.045 * (1.0 - grocery_engagement) * (0.35 + 0.65 * is_new_user)
        - 0.012 * np.log1p(orders_last_30d) * (1.0 - 0.5 * is_new_user)
        + 0.018 * (days_since_last_order > 8.0).astype(float),
        0.005,
        0.20,
    )

    # Alocação endógena (confundida) do desconto + 15% de holdout com t = 0 (DoorDash Sec 3.2.2)
    intended_discount = np.where(
        is_new_user == 1,
        rng.uniform(1.5, 5.0, size=n),
        rng.uniform(0.5, 3.2, size=n),
    )
    # Non-compliance por teto de desconto
    effective_discount = np.clip(intended_discount * rng.uniform(0.85, 1.0, size=n), 0.5, 5.0)
    is_holdout = rng.random(n) < 0.15
    T = np.where(is_holdout, 0.0, effective_discount)

    # Dose côncava z = ln(1 + t): retornos decrescentes em profundidade de desconto
    dose_z = np.log1p(T)
    p_conv = np.clip(p0_true + true_tau * dose_z, 0.01, 0.98)
    Y = rng.binomial(1, p_conv)

    return pd.DataFrame({
        "user_id": np.arange(n),
        "orders_last_30d": orders_last_30d,
        "avg_order_value": avg_order_value,
        "days_since_last_order": days_since_last_order,
        "is_new_user": is_new_user,
        "merchant_fav_count": merchant_fav_count,
        "grocery_category_engagement": grocery_engagement,
        "region_gmv_tier": region_tier,
        "region_gmv_tier_encoded": region_enc,
        "T": T,
        "dose_z": dose_z,
        "Y": Y,
        "p0_true": p0_true,
        "true_tau": true_tau,
        "conversion_prob": p_conv,
    })


# ==============================================================================
# 2. ESTIMADORES CAUSAL ML — CASO 3.1 (TRATAMENTO BINÁRIO)
# ==============================================================================

class SLearner:
    """S-Learner: modelo único Y ~ [X, T], prevê P(Y=1|X,1) − P(Y=1|X,0)."""

    def __init__(self, seed: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=180, max_depth=5, learning_rate=0.05, min_samples_leaf=30, random_state=seed
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        self.model.fit(np.column_stack([X, T]), Y)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        p1 = self.model.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]
        return p1 - p0


class TLearner:
    """T-Learner: dois modelos independentes μ₁(X) em tratados e μ₀(X) em controles."""

    def __init__(self, seed: int = 42):
        self.m1 = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )
        self.m0 = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        self.m1.fit(X[T == 1], Y[T == 1])
        self.m0.fit(X[T == 0], Y[T == 0])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]

    def predict_p0(self, X: np.ndarray) -> np.ndarray:
        return self.m0.predict_proba(X)[:, 1]


class XLearner:
    """X-Learner (Künzel et al., 2019): imputação contrafactual ponderada por propensity score."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.m1 = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )
        self.m0 = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )
        self.g1 = HistGradientBoostingRegressor(
            max_iter=120, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )
        self.g0 = HistGradientBoostingRegressor(
            max_iter=120, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )
        self.ps = make_pipeline(StandardScaler(), LogisticRegression(C=2.0, max_iter=500))

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        self.m1.fit(X[T == 1], Y[T == 1])
        self.m0.fit(X[T == 0], Y[T == 0])
        self.ps.fit(X, T)

        d1 = Y[T == 1] - self.m0.predict_proba(X[T == 1])[:, 1]
        d0 = self.m1.predict_proba(X[T == 0])[:, 1] - Y[T == 0]
        self.g1.fit(X[T == 1], d1)
        self.g0.fit(X[T == 0], d0)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        e = np.clip(self.ps.predict_proba(X)[:, 1], 0.05, 0.95)
        return e * self.g0.predict(X) + (1.0 - e) * self.g1.predict(X)


class DMLConstantPLR:
    """
    DML-PLR Básico com Cross-Fitting em 5 folds (Chernozhukov et al., 2018):
    Estima o ATE ortogonal constante (sem viés de seleção), porém sem poder de ranking (CATE plano).
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.ate_: float = 0.0
        self.se_: float = 0.0

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "DMLConstantPLR":
        n = len(Y)
        m_oof = np.zeros(n)
        e_oof = np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for tr, va in kf.split(X):
            m = HistGradientBoostingClassifier(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr], Y[tr])
            e = HistGradientBoostingClassifier(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr], T[tr])
            m_oof[va] = m.predict_proba(X[va])[:, 1]
            e_oof[va] = e.predict_proba(X[va])[:, 1]

        y_res = Y - m_oof
        t_res = T - e_oof
        denom = float(np.sum(t_res ** 2))
        self.ate_ = float(np.sum(t_res * y_res) / denom)
        psi = (y_res - self.ate_ * t_res) * t_res
        self.se_ = float(np.sqrt(np.sum(psi ** 2)) / denom)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.ate_, dtype=float)


class DRLearnerDML:
    """
    DR-Learner / DML Heterogêneo com Cross-Fitting em 5 folds (Kennedy, 2023):
    Constrói o pseudo-outcome duplamente robusto ψ_DR out-of-fold e ajusta τ̂(X) na 2ª etapa.
    """

    def __init__(self, n_splits: int = 5, min_prop: float = 0.05, seed: int = 42):
        self.n_splits = n_splits
        self.min_prop = min_prop
        self.seed = seed
        self.ridge_stage2 = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            RidgeCV(alphas=np.logspace(0, 4, 20)),
        )
        self.gbr_stage2 = HistGradientBoostingRegressor(
            max_iter=120, max_depth=3, learning_rate=0.04, min_samples_leaf=80, l2_regularization=5.0, random_state=seed
        )
        self.m0_full = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "DRLearnerDML":
        n = len(Y)
        m0_oof = np.zeros(n)
        m1_oof = np.zeros(n)
        e_oof = np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for tr, va in kf.split(X):
            m0 = HistGradientBoostingClassifier(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
            m1 = HistGradientBoostingClassifier(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1])
            e = make_pipeline(StandardScaler(), LogisticRegression(C=2.0, max_iter=500)).fit(X[tr], T[tr])

            m0_oof[va] = m0.predict_proba(X[va])[:, 1]
            m1_oof[va] = m1.predict_proba(X[va])[:, 1]
            e_oof[va] = np.clip(e.predict_proba(X[va])[:, 1], self.min_prop, 1.0 - self.min_prop)

        psi = (
            (m1_oof - m0_oof)
            + (T / e_oof) * (Y - m1_oof)
            - ((1.0 - T) / (1.0 - e_oof)) * (Y - m0_oof)
        )
        self.ridge_stage2.fit(X, psi)
        self.gbr_stage2.fit(X, psi)
        self.m0_full.fit(X[T == 0], Y[T == 0])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return 0.55 * self.ridge_stage2.predict(X) + 0.45 * self.gbr_stage2.predict(X)

    def predict_p0(self, X: np.ndarray) -> np.ndarray:
        return self.m0_full.predict_proba(X)[:, 1]


class RLearnerDiscrete:
    """
    R-Learner para tratamento binário (Nie & Wager, 2021):
    Minimiza a perda de Robinson ponderada por T_res² com cross-fitting em 5 folds:
      min_τ  Σ T_res² · (Y_res / T_res − τ(X))²
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.ridge_stage2 = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            RidgeCV(alphas=np.logspace(0, 4, 20)),
        )
        self.gbr_stage2 = HistGradientBoostingRegressor(
            max_iter=120, max_depth=3, learning_rate=0.04, min_samples_leaf=80, l2_regularization=5.0, random_state=seed
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "RLearnerDiscrete":
        n = len(Y)
        m_oof = np.zeros(n)
        e_oof = np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for tr, va in kf.split(X):
            m = HistGradientBoostingClassifier(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr], Y[tr])
            e = make_pipeline(StandardScaler(), LogisticRegression(C=2.0, max_iter=500)).fit(X[tr], T[tr])
            m_oof[va] = m.predict_proba(X[va])[:, 1]
            e_oof[va] = np.clip(e.predict_proba(X[va])[:, 1], 0.05, 0.95)

        y_res = Y - m_oof
        t_res = T - e_oof
        valid = np.abs(t_res) > 1e-3
        target = y_res[valid] / t_res[valid]
        weights = t_res[valid] ** 2
        self.ridge_stage2.fit(X[valid], target, ridgecv__sample_weight=weights)
        self.gbr_stage2.fit(X[valid], target, sample_weight=weights)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return 0.55 * self.ridge_stage2.predict(X) + 0.45 * self.gbr_stage2.predict(X)


# ==============================================================================
# 3. MÉTRICAS HONESTAS DE UPLIFT (INGÊNUO vs. DEBIASED DR-AUUC vs. ORÁCULO)
# ==============================================================================

def compute_test_dr_pseudo_outcome(
    Xte: np.ndarray,
    Tte: np.ndarray,
    Yte: np.ndarray,
    seed: int = 99,
) -> np.ndarray:
    """
    Constrói pseudo-outcomes duplamente robustos ψ_DR via 5-fold cross-fitting no teste
    para avaliação offline SEM VIÉS DE CONFUNDIMENTO (DR-AUUC / DR-Uplift@k).
    """
    n = len(Yte)
    m0 = np.zeros(n)
    m1 = np.zeros(n)
    e = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in kf.split(Xte):
        f0 = HistGradientBoostingClassifier(max_iter=120, max_depth=4, learning_rate=0.04, min_samples_leaf=30, random_state=seed).fit(Xte[tr][Tte[tr] == 0], Yte[tr][Tte[tr] == 0])
        f1 = HistGradientBoostingClassifier(max_iter=120, max_depth=4, learning_rate=0.04, min_samples_leaf=30, random_state=seed).fit(Xte[tr][Tte[tr] == 1], Yte[tr][Tte[tr] == 1])
        fe = make_pipeline(StandardScaler(), LogisticRegression(C=2.0, max_iter=500)).fit(Xte[tr], Tte[tr])
        m0[va] = f0.predict_proba(Xte[va])[:, 1]
        m1[va] = f1.predict_proba(Xte[va])[:, 1]
        e[va] = np.clip(fe.predict_proba(Xte[va])[:, 1], 0.05, 0.95)
    return (m1 - m0) + (Tte / e) * (Yte - m1) - ((1.0 - Tte) / (1.0 - e)) * (Yte - m0)


def compute_uplift_curves(
    pred_cate: np.ndarray,
    psi_eval: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> Dict[str, Any]:
    """
    Calcula:
      - Curva de Ganho Cumulativo Debiased (sobre ψ_eval = true_cate ou ψ_DR)
      - AUUC (Area Under Uplift Curve ACIMA da linha aleatória — modelo constante = 0,0000!)
      - Uplift@30% (média de ψ_eval no top 30% ranqueado)
      - Qini Ingênuo (sem ajuste de confundimento — mantido apenas para auditoria didática!)
    """
    n = len(pred_cate)
    # Desempate determinístico aleatório para modelos constantes (evita viés de ordem do índice)
    rng = np.random.default_rng(7)
    tie_break = rng.uniform(-1e-12, 1e-12, size=n)
    idx = np.argsort(-(pred_cate + tie_break))

    frac = np.arange(1, n + 1) / n
    cum_gain = np.cumsum(psi_eval[idx]) / n
    random_line = frac * float(np.mean(psi_eval))
    auuc_above_random = float(np.trapezoid(cum_gain - random_line, frac))

    k30 = int(n * 0.30)
    uplift_at_30 = float(np.mean(psi_eval[idx[:k30]]))

    # Métrica ingênua (viesada sob confundimento) apenas para demonstrar o perigo!
    y_s, t_s = Y[idx], T[idx]
    nt = np.maximum(np.cumsum(t_s), 1)
    nc = np.maximum(np.cumsum(1 - t_s), 1)
    naive_curve = np.cumsum(y_s * t_s) / nt - np.cumsum(y_s * (1 - t_s)) / nc
    naive_qini = float(np.trapezoid(naive_curve, frac))

    return {
        "frac": frac,
        "cum_gain": cum_gain,
        "random_line": random_line,
        "auuc": auuc_above_random,
        "uplift_at_30": uplift_at_30,
        "naive_qini": naive_qini,
    }


def evaluate_discrete_targeting_policy(
    pred_cate: np.ndarray,
    true_cate: np.ndarray,
    p0_true: np.ndarray,
    coupon_cost: float = 4.50,
    margin_per_order: float = 12.0,
) -> pd.DataFrame:
    """
    Simula política de targeting top-q% (DoorDash Case 3.1 / Tabela 2) sobre o conjunto de teste:
    - Margem de contribuição realista de marketplace: US$ 12,00/pedido (~25% de take-rate em cesta de US$ 48)
    - Cupom promocional: US$ 4,50 por resgate -> custo esperado c_i = 4,50 * P(Y=1 | T=1, X_i)
    - Pedidos incrementais verdadeiros: Σ_{i ∈ S} true_cate_i
    - Lucro incremental líquido: Σ_{i ∈ S} (margin_per_order * true_cate_i − c_i)
    """
    n = len(pred_cate)
    rng = np.random.default_rng(7)
    idx = np.argsort(-(pred_cate + rng.uniform(-1e-12, 1e-12, size=n)))
    p1_true = np.clip(p0_true + true_cate, 0.0, 1.0)
    exp_cost_i = coupon_cost * p1_true

    rows = []
    for q in np.linspace(0.05, 1.0, 20):
        m = int(round(n * q))
        sel = idx[:m]
        inc_orders = float(np.sum(true_cate[sel]))
        total_cost = float(np.sum(exp_cost_i[sel]))
        net_profit = float(margin_per_order * inc_orders - total_cost)
        cpio = total_cost / inc_orders if inc_orders > 1e-6 else np.nan
        rows.append({
            "pct_targeted": round(float(q), 2),
            "n_targeted": m,
            "incremental_orders": round(inc_orders, 2),
            "order_rate_lift_pp": round((inc_orders / n) * 100.0, 3),
            "total_cost": round(total_cost, 2),
            "net_profit": round(net_profit, 2),
            "cpio": round(cpio, 2),
        })
    df_pol = pd.DataFrame(rows)
    uniform_cost = float(df_pol.loc[df_pol["pct_targeted"] == 1.0, "total_cost"].iloc[0])
    df_pol["cost_index"] = (df_pol["total_cost"] / uniform_cost * 100.0).round(1)
    return df_pol


# ==============================================================================
# 4. ESTIMADORES CAUSAL ML — CASO 3.2 (DESCONTO CONTÍNUO & OTIMIZAÇÃO LAGRANGIANA)
# ==============================================================================

class NaiveClippedDMLContinuous:
    """
    Anti-padrão didático (baseado na implementação original):
    - Divide Y_res / Z_res sem ponderação por Z_res²
    - Descarta 20% dos dados com |Z_res| pequeno e trunca pseudo-outcome em [0, 0.10]
    - Usa árvore profunda (max_depth=10) sujeita a overfitting / encolhimento severo.
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.effect_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=8, learning_rate=0.02, random_state=seed
        )

    def fit(self, X: np.ndarray, Z: np.ndarray, Y: np.ndarray) -> "NaiveClippedDMLContinuous":
        n = len(Y)
        m_oof = np.zeros(n)
        e_oof = np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            m = GradientBoostingRegressor(n_estimators=120, max_depth=8, learning_rate=0.02, random_state=self.seed).fit(X[tr], Y[tr])
            e = GradientBoostingRegressor(n_estimators=120, max_depth=8, learning_rate=0.02, random_state=self.seed).fit(X[tr], Z[tr])
            m_oof[va] = m.predict(X[va])
            e_oof[va] = e.predict(X[va])

        y_res = Y - m_oof
        z_res = Z - e_oof
        thr = np.percentile(np.abs(z_res), 20)
        mask = np.abs(z_res) > thr
        tau_pseudo = np.clip(y_res[mask] / z_res[mask], 0.0, 0.10)
        self.effect_model.fit(X[mask], tau_pseudo)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.effect_model.predict(X), 0.005, 0.10)


class RLearnerContinuousDML:
    """
    R-Learner / DML Contínuo Ortogonal Ponderado (Nie & Wager, 2021; DoorDash Case 3.2):
    Minimiza a perda de Robinson exata sobre a dose côncava Z = ln(1 + T):
      min_τ  Σ Z_res² · (Y_res / Z_res − τ(X))²
    - Sem truncamento enviesado do pseudo-outcome!
    - sample_weight = Z_res² elimina automaticamente a explosão de variância quando Z_res ≈ 0.
    - Regressão regularizada na 2ª etapa (Ridge Polinomial + HistGBR em ensemble) evita shrinkage.
    """

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.ridge_stage2 = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            RidgeCV(alphas=np.logspace(-1, 4, 20)),
        )
        self.gbr_stage2 = HistGradientBoostingRegressor(
            max_iter=150, max_depth=3, learning_rate=0.04, min_samples_leaf=60, l2_regularization=2.0, random_state=seed
        )
        self.p0_model = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=seed
        )
        self.z_res_std_: float = 0.0
        self.y_res_std_: float = 0.0

    def fit(self, X: np.ndarray, Z: np.ndarray, Y: np.ndarray, T_raw: np.ndarray | None = None) -> "RLearnerContinuousDML":
        n = len(Y)
        m_oof = np.zeros(n)
        e_oof = np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for tr, va in kf.split(X):
            m = HistGradientBoostingRegressor(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr], Y[tr])
            e = HistGradientBoostingRegressor(
                max_iter=150, max_depth=4, learning_rate=0.04, min_samples_leaf=40, random_state=self.seed
            ).fit(X[tr], Z[tr])
            m_oof[va] = m.predict(X[va])
            e_oof[va] = e.predict(X[va])

        y_res = Y - m_oof
        z_res = Z - e_oof
        self.y_res_std_ = float(np.std(y_res))
        self.z_res_std_ = float(np.std(z_res))

        valid = np.abs(z_res) > 1e-4
        target = y_res[valid] / z_res[valid]
        weights = z_res[valid] ** 2

        # Ensemble estável de Ridge Polinomial + HistGBR ambos ponderados por Z_res²
        self.ridge_stage2.fit(X[valid], target, ridgecv__sample_weight=weights)
        self.gbr_stage2.fit(X[valid], target, sample_weight=weights)

        # Modelo baseline p0(X) treinado nos usuários do holdout (T = 0)
        if T_raw is not None and np.sum(T_raw == 0) >= 100:
            self.p0_model.fit(X[T_raw == 0], Y[T_raw == 0])
        else:
            self.p0_model.fit(X, Y)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        pred = 0.55 * self.ridge_stage2.predict(X) + 0.45 * self.gbr_stage2.predict(X)
        return np.clip(pred, 0.0, 0.25)

    def predict_p0(self, X: np.ndarray) -> np.ndarray:
        return self.p0_model.predict_proba(X)[:, 1]


def optimize_continuous_discount_lagrangian(
    tau_hat: np.ndarray,
    p0_hat: np.ndarray,
    discount_grid: np.ndarray,
    lam: float,
    margin_per_order: float = 15.0,
) -> np.ndarray:
    """
    Otimização Lagrangiana do paper DoorDash (Eq. 2.2.3 & Sec. 3.2.4):
      t*_i(λ) = argmax_{t ∈ grid} [ V · ΔP(t, X_i) − λ · c_i(t) ]
    onde:
      - ΔP(t, X_i) = min(1 − p0_hat_i, tau_hat_i · ln(1 + t))
      - c_i(t)     = t · (p0_hat_i + ΔP(t, X_i))  (custo esperado de resgate do cupom!)
    """
    # Vetorizado: matrizes (n_users, n_grid)
    t_mat = discount_grid[None, :]                     # (1, G)
    z_mat = np.log1p(t_mat)                            # (1, G)
    p0_col = p0_hat[:, None]                           # (N, 1)
    tau_col = tau_hat[:, None]                         # (N, 1)

    delta_p = np.minimum(1.0 - p0_col, tau_col * z_mat)
    exp_cost = t_mat * (p0_col + delta_p)
    reward = margin_per_order * delta_p - lam * exp_cost

    best_idx = np.argmax(reward, axis=1)
    return discount_grid[best_idx]


def evaluate_continuous_policy_frontier(
    tau_hat: np.ndarray,
    p0_hat: np.ndarray,
    true_tau: np.ndarray,
    p0_true: np.ndarray,
    discount_grid: np.ndarray | None = None,
    margin_per_order: float = 15.0,
) -> pd.DataFrame:
    """Traça a Fronteira de Pareto (Custo vs. Pedidos Incrementais Verdadeiros) variando λ."""
    if discount_grid is None:
        discount_grid = np.arange(0.0, 5.25, 0.5)

    n = len(true_tau)
    lambdas = np.concatenate([np.linspace(0.05, 0.9, 12), np.linspace(1.0, 4.5, 14)])
    rows = []
    for lam in lambdas:
        t_opt = optimize_continuous_discount_lagrangian(
            tau_hat, p0_hat, discount_grid, lam=float(lam), margin_per_order=margin_per_order
        )
        # Avaliação CONTRATUAL VERDADEIRA (ground truth) fora da amostra
        true_dp = np.minimum(1.0 - p0_true, true_tau * np.log1p(t_opt))
        true_cost = t_opt * (p0_true + true_dp)
        inc_orders = float(np.sum(true_dp))
        total_cost = float(np.sum(true_cost))
        net_profit = float(margin_per_order * inc_orders - total_cost)
        roi = (margin_per_order * inc_orders - total_cost) / total_cost if total_cost > 1e-6 else 0.0
        rows.append({
            "lambda": round(float(lam), 3),
            "avg_discount": round(float(np.mean(t_opt)), 3),
            "pct_treated": round(float(np.mean(t_opt > 0.0)) * 100.0, 1),
            "incremental_orders": round(inc_orders, 2),
            "order_rate_lift_pp": round((inc_orders / n) * 100.0, 3),
            "total_cost": round(total_cost, 2),
            "net_profit": round(net_profit, 2),
            "roi_x": round(roi, 2),
            "cpio": round(total_cost / inc_orders, 2) if inc_orders > 1e-6 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)


def evaluate_uniform_discount_curve(
    true_tau: np.ndarray,
    p0_true: np.ndarray,
    discount_grid: np.ndarray | None = None,
    margin_per_order: float = 15.0,
) -> pd.DataFrame:
    """Baseline Uniforme: dá o mesmo desconto fixo t para 100% dos usuários."""
    if discount_grid is None:
        discount_grid = np.arange(0.0, 5.25, 0.5)
    n = len(true_tau)
    rows = []
    for t_val in discount_grid:
        t_vec = np.full(n, float(t_val))
        true_dp = np.minimum(1.0 - p0_true, true_tau * np.log1p(t_vec))
        true_cost = t_vec * (p0_true + true_dp)
        inc_orders = float(np.sum(true_dp))
        total_cost = float(np.sum(true_cost))
        net_profit = float(margin_per_order * inc_orders - total_cost)
        roi = (margin_per_order * inc_orders - total_cost) / total_cost if total_cost > 1e-6 else 0.0
        rows.append({
            "discount": float(t_val),
            "incremental_orders": round(inc_orders, 2),
            "order_rate_lift_pp": round((inc_orders / n) * 100.0, 3),
            "total_cost": round(total_cost, 2),
            "net_profit": round(net_profit, 2),
            "roi_x": round(roi, 2),
            "cpio": round(total_cost / inc_orders, 2) if inc_orders > 1e-6 else np.nan,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 5. ORQUESTRADOR COMPLETO (PARA TESTES E CLI)
# ==============================================================================

def run_case1_discrete(n: int = 15000, seed: int = 42) -> Dict[str, Any]:
    """Executa o pipeline completo do Caso 3.1 (Targeting Discreto)."""
    df = simulate_discrete_promo_data(n=n, seed=seed)
    strat = df["T"].astype(str) + "_" + df["Y"].astype(str)
    df_tr, df_te = train_test_split(df, test_size=0.30, random_state=seed, stratify=strat)

    Xtr, Ttr, Ytr = df_tr[DISCRETE_FEATURES].to_numpy(float), df_tr["T"].to_numpy(int), df_tr["Y"].to_numpy(int)
    Xte, Tte, Yte = df_te[DISCRETE_FEATURES].to_numpy(float), df_te["T"].to_numpy(int), df_te["Y"].to_numpy(int)
    true_cate_te = df_te["true_cate"].to_numpy(float)
    p0_te = df_te["p0_true"].to_numpy(float)

    psi_dr_te = compute_test_dr_pseudo_outcome(Xte, Tte, Yte, seed=seed + 50)

    models = {
        "S-Learner": SLearner(seed=seed),
        "T-Learner": TLearner(seed=seed),
        "X-Learner": XLearner(seed=seed),
        "DML-PLR (Constante)": DMLConstantPLR(n_splits=5, seed=seed),
        "DR-Learner (DML-Het)": DRLearnerDML(n_splits=5, min_prop=0.05, seed=seed),
        "R-Learner (Discrete)": RLearnerDiscrete(n_splits=5, seed=seed),
    }

    preds: Dict[str, np.ndarray] = {}
    metrics_rows: List[Dict[str, Any]] = []
    curves_oracle: Dict[str, Dict[str, Any]] = {}
    curves_dr: Dict[str, Dict[str, Any]] = {}

    for name, mod in models.items():
        mod.fit(Xtr, Ttr, Ytr)
        p_te = mod.predict_cate(Xte)
        preds[name] = p_te

        c_ora = compute_uplift_curves(p_te, true_cate_te, Yte, Tte)
        c_dr = compute_uplift_curves(p_te, psi_dr_te, Yte, Tte)
        curves_oracle[name] = c_ora
        curves_dr[name] = c_dr

        pehe = float(np.sqrt(np.mean((p_te - true_cate_te) ** 2)))
        corr = float(np.corrcoef(p_te, true_cate_te)[0, 1]) if np.std(p_te) > 1e-10 else 0.0
        sp_rho = float(spearmanr(p_te, true_cate_te).statistic) if np.std(p_te) > 1e-10 else 0.0

        metrics_rows.append({
            "Modelo": name,
            "ATE_Estimado_pp": round(float(np.mean(p_te)) * 100.0, 2),
            "PEHE_pp": round(pehe * 100.0, 2),
            "Corr_Pearson": round(corr, 4),
            "Spearman_Rho": round(sp_rho, 4),
            "DR_AUUC": round(c_dr["auuc"], 5),
            "Oracle_AUUC": round(c_ora["auuc"], 5),
            "True_Uplift_Top30_pp": round(c_ora["uplift_at_30"] * 100.0, 2),
            "Naive_Qini_Viesado": round(c_ora["naive_qini"] * 100.0, 2),
        })

    # Oráculo para teto de referência
    c_ora_ceil = compute_uplift_curves(true_cate_te, true_cate_te, Yte, Tte)
    curves_oracle["Oráculo (Teto)"] = c_ora_ceil

    metrics_df = pd.DataFrame(metrics_rows).sort_values("Oracle_AUUC", ascending=False).reset_index(drop=True)
    best_model_name = str(metrics_df.iloc[0]["Modelo"])

    policy_best = evaluate_discrete_targeting_policy(preds[best_model_name], true_cate_te, p0_te)
    policy_oracle = evaluate_discrete_targeting_policy(true_cate_te, true_cate_te, p0_te)
    policy_random = evaluate_discrete_targeting_policy(np.zeros_like(true_cate_te), true_cate_te, p0_te)

    # Ponto ótimo de lucro líquido do melhor modelo vs. Uniforme (100%)
    best_row = policy_best.loc[policy_best["net_profit"].idxmax()]
    unif_row = policy_best.loc[policy_best["pct_targeted"] == 1.0].iloc[0]

    return {
        "df_train": df_tr,
        "df_test": df_te,
        "true_ate": round(float(np.mean(true_cate_te)), 4),
        "naive_ate": round(float(Yte[Tte == 1].mean() - Yte[Tte == 0].mean()), 4),
        "metrics_df": metrics_df,
        "best_model": best_model_name,
        "preds": preds,
        "curves_oracle": curves_oracle,
        "curves_dr": curves_dr,
        "policy_best": policy_best,
        "policy_oracle": policy_oracle,
        "policy_random": policy_random,
        "summary_policy": {
            "best_pct_targeted": float(best_row["pct_targeted"]),
            "best_cost_index": float(best_row["cost_index"]),
            "cost_reduction_pct": round(100.0 - float(best_row["cost_index"]), 1),
            "best_net_profit": float(best_row["net_profit"]),
            "uniform_net_profit": float(unif_row["net_profit"]),
            "profit_delta_usd": round(float(best_row["net_profit"]) - float(unif_row["net_profit"]), 2),
            "best_inc_orders": float(best_row["incremental_orders"]),
            "uniform_inc_orders": float(unif_row["incremental_orders"]),
        },
    }


def run_case2_continuous(n: int = 20000, seed: int = 42) -> Dict[str, Any]:
    """Executa o pipeline completo do Caso 3.2 (Profundidade de Desconto Contínuo)."""
    df = simulate_continuous_discount_data(n=n, seed=seed)
    df_tr, df_te = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["Y"])

    Xtr = df_tr[CONTINUOUS_FEATURES].to_numpy(float)
    Ztr = df_tr["dose_z"].to_numpy(float)
    Ttr = df_tr["T"].to_numpy(float)
    Ytr = df_tr["Y"].to_numpy(float)

    Xte = df_te[CONTINUOUS_FEATURES].to_numpy(float)
    true_tau_te = df_te["true_tau"].to_numpy(float)
    p0_te = df_te["p0_true"].to_numpy(float)

    # 1. Anti-padrão: DML Contínuo Ingênuo (truncado e sem pesos Z_res²)
    naive_mod = NaiveClippedDMLContinuous(n_splits=5, seed=seed).fit(Xtr, Ztr, Ytr)
    tau_naive = naive_mod.predict_tau(Xte)

    # 2. Solução rigorosa: R-Learner Contínuo Ponderado (sample_weight = Z_res²)
    r_mod = RLearnerContinuousDML(n_splits=5, seed=seed).fit(Xtr, Ztr, Ytr, T_raw=Ttr)
    tau_r = r_mod.predict_tau(Xte)
    p0_hat_te = r_mod.predict_p0(Xte)

    comp_df = pd.DataFrame([
        {
            "Modelo": "DML Contínuo Ingênuo (Truncado s/ Pesos)",
            "Tau_Medio_Est": round(float(np.mean(tau_naive)), 4),
            "Tau_Std_Est": round(float(np.std(tau_naive)), 4),
            "PEHE_Tau": round(float(np.sqrt(np.mean((tau_naive - true_tau_te) ** 2))), 4),
            "Corr_Pearson": round(float(np.corrcoef(tau_naive, true_tau_te)[0, 1]), 4),
            "Spearman_Rho": round(float(spearmanr(tau_naive, true_tau_te).statistic), 4),
        },
        {
            "Modelo": "R-Learner Contínuo (Ponderado Z_res²)",
            "Tau_Medio_Est": round(float(np.mean(tau_r)), 4),
            "Tau_Std_Est": round(float(np.std(tau_r)), 4),
            "PEHE_Tau": round(float(np.sqrt(np.mean((tau_r - true_tau_te) ** 2))), 4),
            "Corr_Pearson": round(float(np.corrcoef(tau_r, true_tau_te)[0, 1]), 4),
            "Spearman_Rho": round(float(spearmanr(tau_r, true_tau_te).statistic), 4),
        },
    ])

    discount_grid = np.arange(0.0, 5.25, 0.5)
    frontier_r = evaluate_continuous_policy_frontier(tau_r, p0_hat_te, true_tau_te, p0_te, discount_grid)
    frontier_naive = evaluate_continuous_policy_frontier(tau_naive, p0_hat_te, true_tau_te, p0_te, discount_grid)
    frontier_oracle = evaluate_continuous_policy_frontier(true_tau_te, p0_te, true_tau_te, p0_te, discount_grid)
    uniform_curve = evaluate_uniform_discount_curve(true_tau_te, p0_te, discount_grid)

    # Comparação DoorDash Tabela 3:
    # Baseline Uniforme = desconto fixo $2.50 para 100% dos usuários
    unif_ref = uniform_curve.loc[uniform_curve["discount"] == 2.5].iloc[0]
    target_orders = float(unif_ref["incremental_orders"])
    # Menor custo na fronteira R-Learner que entrega >= pedidos incrementais do uniforme $2.50
    feasible = frontier_r[frontier_r["incremental_orders"] >= target_orders]
    if len(feasible) > 0:
        iso_lift_row = feasible.iloc[0]
    else:
        iso_lift_row = frontier_r.iloc[-1]

    # Ponto de lucro líquido máximo (λ = 1.0 → 1 dólar de margem vale 1 dólar de custo)
    opt_profit_row = frontier_r.loc[frontier_r["net_profit"].idxmax()]
    t_opt_profit = optimize_continuous_discount_lagrangian(tau_r, p0_hat_te, discount_grid, lam=1.0, margin_per_order=15.0)

    df_te_out = df_te.copy()
    df_te_out["tau_pred_naive"] = tau_naive
    df_te_out["tau_pred_rlearner"] = tau_r
    df_te_out["optimal_discount_lam1"] = t_opt_profit

    cost_index_iso = round(float(iso_lift_row["total_cost"]) / float(unif_ref["total_cost"]) * 100.0, 1)
    return {
        "df_train": df_tr,
        "df_test": df_te_out,
        "true_tau_mean": round(float(np.mean(true_tau_te)), 4),
        "true_tau_std": round(float(np.std(true_tau_te)), 4),
        "comp_df": comp_df,
        "frontier_r": frontier_r,
        "frontier_naive": frontier_naive,
        "frontier_oracle": frontier_oracle,
        "uniform_curve": uniform_curve,
        "summary_policy": {
            "uniform_discount_ref": 2.50,
            "uniform_cost": float(unif_ref["total_cost"]),
            "uniform_inc_orders": target_orders,
            "uniform_net_profit": float(unif_ref["net_profit"]),
            "rlearner_iso_cost": float(iso_lift_row["total_cost"]),
            "rlearner_iso_orders": float(iso_lift_row["incremental_orders"]),
            "rlearner_iso_cost_index": cost_index_iso,
            "rlearner_iso_savings_pct": round(100.0 - cost_index_iso, 1),
            "rlearner_max_profit": float(opt_profit_row["net_profit"]),
            "rlearner_max_profit_avg_disc": float(opt_profit_row["avg_discount"]),
            "rlearner_max_profit_pct_treated": float(opt_profit_row["pct_treated"]),
        },
    }


def run_all_experiments(seed: int = 42) -> Dict[str, Any]:
    return {
        "case1": run_case1_discrete(n=15000, seed=seed),
        "case2": run_case2_continuous(n=20000, seed=seed),
    }


if __name__ == "__main__":
    res = run_all_experiments(seed=42)
    c1, c2 = res["case1"], res["case2"]
    print("=== CASO 3.1: TARGETING DISCRETO ===")
    print(f"ATE Verdadeiro: {c1['true_ate']*100:.2f} pp | Dif. Ingênua (Confundida): {c1['naive_ate']*100:.2f} pp")
    print(c1["metrics_df"].to_string(index=False))
    print("Política Ótima:", c1["summary_policy"])
    print("\n=== CASO 3.2: DESCONTO CONTÍNUO ===")
    print(f"Tau Verdadeiro: média={c2['true_tau_mean']:.4f} | std={c2['true_tau_std']:.4f}")
    print(c2["comp_df"].to_string(index=False))
    print("Política Lagrangiana:", c2["summary_policy"])
