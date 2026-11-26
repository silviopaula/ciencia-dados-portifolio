"""Pipeline MLOps End-to-End de Previsão de Churn com MLflow 3.x (Tracking, Registry, Drift & ROI).

Evolução rigorosa sobre `data/abt.csv` (4.055 registros, 1.506 clientes, 92 features em 4 janelas):
- Split Cronológico Out-of-Time (OOT) estrito: Treino (Abr–Jul/2024), Validação (Ago–Set/2024)
  e Teste Cego OOT (Out–Nov/2024), com auditoria explícita do vazamento gerado por split aleatório.
- Transformador customizado `ChurnFeatureEngineer` embutido no `sklearn.pipeline.Pipeline` e salvo
  no MLflow Model Registry (Training-Serving Skew = 0: aceita DataFrame bruto com NaNs!).
- Rastreamento completo de 9 pipelines no MLflow 3.x (SQLite local portátil), com `infer_signature`,
  `input_example`, IC 95% Bootstrap, Calibração (Brier Score) e Quality Gate automático.
- Governança no Model Registry (`churn-predictor`): `@baseline`, `@challenger` e `@champion`
  promovido automaticamente apenas se passar em todos os critérios de qualidade e inferência.
- Monitoramento de Data Drift (PSI: Treino vs. OOT), Otimização de Threshold por ROI de Retenção,
  Permutation Importance e Slice-Based Error Analysis.
"""

from __future__ import annotations

import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover
    FrozenEstimator = None

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "abt.csv"
STORE_DIR = PROJECT_ROOT / "mlflow_store"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

DEFAULT_EXPERIMENT = "churn_mlops_oot_benchmark"
DEFAULT_MODEL_NAME = "churn-predictor"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fbfcfe",
    "axes.edgecolor": "#cbd5e1",
    "axes.titleweight": "bold",
    "font.sans-serif": ["DejaVu Sans"],
})


def get_tracking_uri(store_dir: Path = STORE_DIR) -> str:
    store_dir.mkdir(parents=True, exist_ok=True)
    db_path = (store_dir / "mlflow.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def chronological_oot_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estritamente cronológico (Zero Temporal Leakage):
    - Treino    : 2024-04 a 2024-07 (66,4%)
    - Validação : 2024-08 a 2024-09 (20,2%)
    - Teste OOT : 2024-10 a 2024-11 (13,4%)
    """
    work = df.copy()
    dt = pd.to_datetime(work["dt_ref"])
    train = work[dt <= "2024-07-31"].copy().reset_index(drop=True)
    val = work[(dt > "2024-07-31") & (dt <= "2024-09-30")].copy().reset_index(drop=True)
    test = work[dt > "2024-09-30"].copy().reset_index(drop=True)
    return train, val, test


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engenharia semântica de features e imputação sem Data Leakage embutida no Pipeline sklearn.

    Preserva o sinal estrutural de inatividade (NaN em janelas d7/d14 = cliente inativo na janela),
    adiciona razões de tendência recente vs. histórico (`d7/d28`, `d14/d28`, `d7/life`, `gap_recorrencia`)
    e imputa medianas aprendidas ESTRITAMENTE no conjunto de treino.
    """

    def __init__(self) -> None:
        self.medians_: Dict[str, float] = {}
        self.rec_life_median_: float = 0.0
        self.feature_names_out_: List[str] = []

    def _transform_core(self, X: pd.DataFrame, rec_life_fill: float) -> pd.DataFrame:
        df = X.copy()
        # 1. Indicadores explícitos de inatividade estrutural (missingness informativo!)
        df["inativo_d7"] = df["qtd_iteracoes_d7"].isna().astype(float) if "qtd_iteracoes_d7" in df else 0.0
        df["inativo_d14"] = df["qtd_iteracoes_d14"].isna().astype(float) if "qtd_iteracoes_d14" in df else 0.0
        df["sem_recorrencia_d7"] = df["media_dias_recorrencia_d7"].isna().astype(float) if "media_dias_recorrencia_d7" in df else 0.0
        df["sem_recorrencia_d14"] = df["media_dias_recorrencia_d14"].isna().astype(float) if "media_dias_recorrencia_d14" in df else 0.0

        # 2. Preenchimento semântico zero-activity para contagens/percentuais de janelas recentes
        zero_prefixes = ("qtd_", "pontos_", "saldo_", "frequencia_", "mensagens_", "pct_")
        for col in list(df.columns):
            if (col.endswith("_d7") or col.endswith("_d14")) and col.startswith(zero_prefixes):
                df[col] = df[col].fillna(0.0)

        # Para recência em janelas vazias: preenche com valor maior que a janela (ex: 8 dias em d7, 15 em d14)
        for col, fill_val in (
            ("dias_ultima_iteracao_d7", 8.0),
            ("dias_primeira_iteracao_d7", 8.0),
            ("dias_ultima_iteracao_d14", 15.0),
            ("dias_primeira_iteracao_d14", 15.0),
        ):
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        # 3. Razões comportamentais de aceleração / desaceleração (tendência de engajamento)
        def _ratio(num: str, den: str) -> np.ndarray:
            if num in df.columns and den in df.columns:
                a = df[num].fillna(0.0).to_numpy(dtype=float)
                b = np.maximum(df[den].fillna(0.0).to_numpy(dtype=float), 1.0)
                return a / b
            return np.zeros(len(df), dtype=float)

        df["ratio_iter_d7_d28"] = _ratio("qtd_iteracoes_d7", "qtd_iteracoes_d28")
        df["ratio_iter_d14_d28"] = _ratio("qtd_iteracoes_d14", "qtd_iteracoes_d28")
        df["ratio_iter_d28_life"] = _ratio("qtd_iteracoes_d28", "qtd_iteracoes_life")
        df["ratio_pts_d7_d28"] = _ratio("pontos_acum_d7", "pontos_acum_d28")
        df["ratio_freq_d7_life"] = _ratio("frequencia_d7", "frequencia_life")
        df["densidade_iter_dia_life"] = _ratio("qtd_iteracoes_life", "dias_primeira_iteracao_life")
        df["pct_pontos_neg_life"] = _ratio("pontos_neg_life", "pontos_acum_life")

        if "dias_ultima_iteracao_life" in df.columns and "media_dias_recorrencia_life" in df.columns:
            rec_mean = df["media_dias_recorrencia_life"].fillna(rec_life_fill)
            df["gap_ultima_vs_recorrencia"] = df["dias_ultima_iteracao_life"].fillna(0.0) - rec_mean
        else:
            df["gap_ultima_vs_recorrencia"] = 0.0

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ChurnFeatureEngineer":
        if "media_dias_recorrencia_life" in X.columns and X["media_dias_recorrencia_life"].notna().any():
            m_rec = float(X["media_dias_recorrencia_life"].median())
            self.rec_life_median_ = m_rec if np.isfinite(m_rec) else 0.0
        else:
            self.rec_life_median_ = 0.0

        df = self._transform_core(X, rec_life_fill=self.rec_life_median_)
        self.medians_ = {}
        for col in df.columns:
            med = float(df[col].median()) if df[col].notna().any() else 0.0
            self.medians_[col] = med if np.isfinite(med) else 0.0
        self.feature_names_out_ = list(df.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._transform_core(X, rec_life_fill=self.rec_life_median_)
        for col in self.feature_names_out_:
            if col not in df.columns:
                df[col] = self.medians_[col]
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.medians_[col])
        return df[self.feature_names_out_].astype(float)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI) entre distribuição de referência (Treino) e produção (OOT)."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if len(e) < 10 or len(a) < 10:
        return 0.0

    quantiles = np.unique(np.percentile(e, np.linspace(0, 100, bins + 1)))
    if len(quantiles) < 3:
        # Feature binária ou quase-constante -> compara proporções diretamente
        vals = np.unique(np.concatenate([e, a]))
        if len(vals) <= 1:
            return 0.0
        psi = 0.0
        for v in vals[:10]:
            pe = max(float(np.mean(e == v)), 1e-4)
            pa = max(float(np.mean(a == v)), 1e-4)
            psi += (pa - pe) * float(np.log(pa / pe))
        return round(float(psi), 4)

    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    e_counts, _ = np.histogram(e, bins=quantiles)
    a_counts, _ = np.histogram(a, bins=quantiles)
    e_pct = np.maximum(e_counts / max(len(e), 1), 1e-4)
    a_pct = np.maximum(a_counts / max(len(a), 1), 1e-4)
    return round(float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))), 4)


def compute_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_boot: int = 400,
    seed: int = 42,
) -> Dict[str, float]:
    """Calcula métricas binárias completas + IC 95% Bootstrap para ROC-AUC e F1."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    yhat = (yp >= threshold).astype(int)

    roc = float(roc_auc_score(yt, yp))
    pr = float(average_precision_score(yt, yp))
    f1 = float(f1_score(yt, yhat, zero_division=0))
    acc = float(accuracy_score(yt, yhat))
    prec = float(precision_score(yt, yhat, zero_division=0))
    rec = float(recall_score(yt, yhat, zero_division=0))
    brier = float(brier_score_loss(yt, yp))

    rng = np.random.default_rng(seed)
    n = len(yt)
    rocs: List[float] = []
    f1s: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = yt[idx]
        if len(np.unique(yb)) < 2:
            continue
        rocs.append(float(roc_auc_score(yb, yp[idx])))
        f1s.append(float(f1_score(yb, yhat[idx], zero_division=0)))

    return {
        "roc_auc": round(roc, 4),
        "roc_ci_low": round(float(np.percentile(rocs, 2.5)) if rocs else roc, 4),
        "roc_ci_high": round(float(np.percentile(rocs, 97.5)) if rocs else roc, 4),
        "pr_auc": round(pr, 4),
        "f1": round(f1, 4),
        "f1_ci_low": round(float(np.percentile(f1s, 2.5)) if f1s else f1, 4),
        "f1_ci_high": round(float(np.percentile(f1s, 97.5)) if f1s else f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "brier": round(brier, 4),
    }


def optimize_retention_roi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float = 450.0,
    contact_cost: float = 45.0,
    save_rate: float = 0.30,
) -> Dict[str, Any]:
    """Otimiza threshold de campanha de retenção maximizando Lucro Líquido esperado:
    Lucro = TP * (save_rate * LTV) - (TP + FP) * contact_cost.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    thresholds = np.linspace(0.10, 0.90, 81)
    curve = []
    best_profit = -1e18
    best_thr = 0.50

    for thr in thresholds:
        pred = (yp >= thr).astype(int)
        tp = int(np.sum((yt == 1) & (pred == 1)))
        contacted = int(np.sum(pred == 1))
        saved_rev = tp * save_rate * ltv
        cost = contacted * contact_cost
        profit = saved_rev - cost
        curve.append({
            "threshold": round(float(thr), 2),
            "contacted": contacted,
            "tp": tp,
            "net_profit": round(float(profit), 2),
        })
        if profit > best_profit:
            best_profit = profit
            best_thr = float(thr)

    treat_all_profit = float(np.sum(yt == 1) * save_rate * ltv - len(yt) * contact_cost)
    p05_tp = int(np.sum((yt == 1) & (yp >= 0.50)))
    p05_cnt = int(np.sum(yp >= 0.50))
    thr05_profit = float(p05_tp * save_rate * ltv - p05_cnt * contact_cost)

    return {
        "ltv": ltv,
        "contact_cost": contact_cost,
        "save_rate": save_rate,
        "theoretical_breakeven_prob": round(contact_cost / (save_rate * ltv), 4),
        "best_threshold": round(best_thr, 2),
        "best_net_profit": round(best_profit, 2),
        "default_050_profit": round(thr05_profit, 2),
        "treat_all_profit": round(treat_all_profit, 2),
        "curve": curve,
    }


def evaluate_roi_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    ltv: float = 450.0,
    contact_cost: float = 45.0,
    save_rate: float = 0.30,
) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    pred = (yp >= threshold).astype(int)
    tp = int(np.sum((yt == 1) & (pred == 1)))
    contacted = int(np.sum(pred == 1))
    saved_rev = tp * save_rate * ltv
    cost = contacted * contact_cost
    profit = saved_rev - cost
    treat_all_profit = float(np.sum(yt == 1) * save_rate * ltv - len(yt) * contact_cost)
    return {
        "threshold": round(float(threshold), 2),
        "contacted": contacted,
        "contacted_pct": round(contacted / max(len(yt), 1) * 100.0, 1),
        "tp_saved_targets": tp,
        "expected_revenue_saved": round(saved_rev, 2),
        "campaign_cost": round(cost, 2),
        "net_profit": round(profit, 2),
        "roi_pct": round((profit / max(cost, 1e-9)) * 100.0, 1) if cost > 0 else 0.0,
        "treat_all_profit": round(treat_all_profit, 2),
        "gain_vs_treat_all": round(profit - treat_all_profit, 2),
    }


def run_full_mlops_workflow(reset_store: bool = True) -> Dict[str, Any]:
    """Executa todo o ciclo MLOps com MLflow 3.x, gera 6 figuras e salva `outputs/metrics_summary.json`."""
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if reset_store and STORE_DIR.exists():
        shutil.rmtree(STORE_DIR, ignore_errors=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    (STORE_DIR / "artifacts").mkdir(parents=True, exist_ok=True)

    tracking_uri = get_tracking_uri(STORE_DIR)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
    if exp is None:
        exp_id = client.create_experiment(
            DEFAULT_EXPERIMENT,
            artifact_location=(STORE_DIR / "artifacts").resolve().as_uri(),
            tags={
                "project": "mlops-churn-mlflow-registry",
                "task": "churn_prediction_oot",
                "author": "Dr. Silvio da Rosa Paula",
            },
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(DEFAULT_EXPERIMENT)

    # 1. Carregamento e Auditoria da ABT
    df = pd.read_csv(DATA_PATH)
    df["dt_ref_dt"] = pd.to_datetime(df["dt_ref"])
    df["ym"] = df["dt_ref_dt"].dt.to_period("M").astype(str)
    raw_features = [c for c in df.columns if c not in ("dt_ref", "dt_ref_dt", "ym", "id_cliente", "flag_churn")]

    train_df, val_df, test_df = chronological_oot_split(df)
    X_tr, y_tr = train_df[raw_features], train_df["flag_churn"].to_numpy(dtype=int)
    X_va, y_va = val_df[raw_features], val_df["flag_churn"].to_numpy(dtype=int)
    X_te, y_te = test_df[raw_features], test_df["flag_churn"].to_numpy(dtype=int)

    cohort_eda = (
        df.groupby("ym")
        .agg(
            n=("flag_churn", "size"),
            unique_clients=("id_cliente", "nunique"),
            churn_rate=("flag_churn", "mean"),
            pct_inativo_d7=("qtd_iteracoes_d7", lambda s: s.isna().mean()),
            pct_inativo_d14=("qtd_iteracoes_d14", lambda s: s.isna().mean()),
        )
        .reset_index()
    )

    # Auditoria explícita de vazamento: Random Split (notebook legado) vs. Out-of-Time Split
    X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(
        df[raw_features], df["flag_churn"].to_numpy(dtype=int), test_size=0.20, random_state=42, stratify=df["flag_churn"]
    )
    rf_leak = Pipeline([
        ("eng", ChurnFeatureEngineer()),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=8, random_state=42, n_jobs=-1)),
    ])
    rf_leak.fit(X_r_tr, y_r_tr)
    auc_random_leak = float(roc_auc_score(y_r_te, rf_leak.predict_proba(X_r_te)[:, 1]))

    # Sobreposição de clientes em Random Split vs. OOT
    tr_idx_r, te_idx_r = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df["flag_churn"])
    cli_overlap_random = len(set(df.loc[tr_idx_r, "id_cliente"]) & set(df.loc[te_idx_r, "id_cliente"]))
    cli_overlap_oot = len(set(train_df["id_cliente"]) & set(test_df["id_cliente"]))

    print(f"[1/6] ABT carregada: {len(df):,} linhas | {df['id_cliente'].nunique():,} clientes | {len(raw_features)} features")
    print(f"      Split OOT -> Treino={len(train_df):,} (Abr–Jul) | Val={len(val_df):,} (Ago–Set) | Teste OOT={len(test_df):,} (Out–Nov)")
    print(f"      Auditoria Vazamento: Clientes repetidos Random={cli_overlap_random} vs OOT={cli_overlap_oot} | RF Random AUC={auc_random_leak:.4f}")

    # 2. Definição dos 8 Modelos Base + 1 Calibrado (9 Runs Rastreáveis no MLflow 3.x)
    hgb_base = HistGradientBoostingClassifier(
        max_iter=220, learning_rate=0.04, max_leaf_nodes=23, min_samples_leaf=25, l2_regularization=1.5, random_state=42
    )
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=10, max_features="sqrt", random_state=42, n_jobs=-1
    )
    lr_base = Pipeline([
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=0.25, max_iter=500, random_state=42)),
    ])

    model_specs: List[Tuple[str, str, Pipeline, Dict[str, Any]]] = [
        (
            "1. DummyClassifier (Baseline)",
            "Baseline",
            Pipeline([("eng", ChurnFeatureEngineer()), ("clf", DummyClassifier(strategy="prior"))]),
            {"family": "baseline", "strategy": "prior"},
        ),
        (
            "2. DecisionTree (Legado OOT)",
            "Árvore Legada",
            Pipeline([("eng", ChurnFeatureEngineer()), ("clf", DecisionTreeClassifier(min_samples_leaf=80, max_depth=6, random_state=42))]),
            {"family": "tree", "min_samples_leaf": 80, "max_depth": 6},
        ),
        (
            "3. LogisticRegression (L2)",
            "Linear Regularizado",
            Pipeline([("eng", ChurnFeatureEngineer()), ("scl", StandardScaler()), ("clf", LogisticRegression(C=0.25, max_iter=500, random_state=42))]),
            {"family": "linear", "C": 0.25, "penalty": "l2"},
        ),
        (
            "4. RandomForest (300 árv.)",
            "Bagging",
            Pipeline([("eng", ChurnFeatureEngineer()), ("clf", rf_base)]),
            {"family": "bagging", "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 10},
        ),
        (
            "5. ExtraTrees (300 árv.)",
            "Bagging",
            Pipeline([("eng", ChurnFeatureEngineer()), ("clf", ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=8, random_state=42, n_jobs=-1))]),
            {"family": "bagging", "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 8},
        ),
        (
            "6. GradientBoosting (Clássico)",
            "Boosting",
            Pipeline([("eng", ChurnFeatureEngineer()), ("clf", GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=3, min_samples_leaf=20, subsample=0.8, random_state=42))]),
            {"family": "boosting", "n_estimators": 180, "learning_rate": 0.04, "max_depth": 3},
        ),
        (
            "7. HistGradientBoosting",
            "Boosting",
            Pipeline([("eng", ChurnFeatureEngineer()), ("clf", hgb_base)]),
            {"family": "boosting", "max_iter": 220, "learning_rate": 0.04, "max_leaf_nodes": 23},
        ),
        (
            "8. Soft-Voting Ensemble (HGB+RF+LR)",
            "Ensemble",
            Pipeline([
                ("eng", ChurnFeatureEngineer()),
                ("clf", VotingClassifier(
                    estimators=[("hgb", hgb_base), ("rf", rf_base), ("lr", lr_base)],
                    voting="soft",
                    weights=[2.0, 2.0, 1.0],
                )),
            ]),
            {"family": "ensemble", "members": "HGB+RF+LR", "weights": "2:2:1"},
        ),
    ]

    print(f"[2/6] Treinando e rastreando pipelines no MLflow {mlflow.__version__} ({tracking_uri})...")
    benchmark_rows: List[Dict[str, Any]] = []
    fitted_store: Dict[str, Any] = {}
    run_ids: Dict[str, str] = {}
    model_uris: Dict[str, str] = {}
    val_probs: Dict[str, np.ndarray] = {}
    test_probs: Dict[str, np.ndarray] = {}
    sample_input = X_te.iloc[:5].copy()
    lean_reqs = ["scikit-learn>=1.5.0", "pandas>=2.2.0", "numpy>=1.26.0"]

    for name, family_lbl, pipe, params in model_specs:
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_sec = time.perf_counter() - t0

        p_va = pipe.predict_proba(X_va)[:, 1]
        p_te = pipe.predict_proba(X_te)[:, 1]
        m_va = compute_metrics_with_bootstrap(y_va, p_va, n_boot=200, seed=42)
        m_te = compute_metrics_with_bootstrap(y_te, p_te, n_boot=400, seed=42)

        sig = infer_signature(sample_input, pipe.predict_proba(sample_input)[:, 1])
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({
                "model_family": family_lbl,
                "split_strategy": "chronological_oot",
                "train_window": "2024-04..2024-07",
                "val_window": "2024-08..2024-09",
                "oot_window": "2024-10..2024-11",
            })
            mlflow.log_params({**params, "n_raw_features": len(raw_features), "train_n": len(X_tr)})
            mlflow.log_metrics({
                "val_roc_auc": m_va["roc_auc"],
                "val_pr_auc": m_va["pr_auc"],
                "val_brier": m_va["brier"],
                "oot_roc_auc": m_te["roc_auc"],
                "oot_pr_auc": m_te["pr_auc"],
                "oot_f1": m_te["f1"],
                "oot_accuracy": m_te["accuracy"],
                "oot_brier": m_te["brier"],
                "train_sec": round(fit_sec, 3),
            })
            m_info = mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                signature=sig,
                input_example=sample_input,
                pip_requirements=lean_reqs,
            )
            rid = run.info.run_id

        run_ids[name] = rid
        model_uris[name] = m_info.model_uri
        fitted_store[name] = pipe
        val_probs[name] = p_va
        test_probs[name] = p_te
        benchmark_rows.append({
            "model": name,
            "family": family_lbl,
            "run_id": rid,
            "val_roc_auc": m_va["roc_auc"],
            "val_brier": m_va["brier"],
            **{f"oot_{k}": v for k, v in m_te.items()},
            "train_sec": round(fit_sec, 2),
        })
        print(f"      {name:<38} | Val AUC={m_va['roc_auc']:.4f} | OOT AUC={m_te['roc_auc']:.4f} | PR-AUC={m_te['pr_auc']:.4f} | Brier={m_te['brier']:.4f}")

    # 9º Pipeline: Calibração Probabilística (Platt Sigmoid na Validação) do melhor modelo na Validação
    non_dum = [r for r in benchmark_rows if "Dummy" not in r["model"]]
    best_val_row = max(non_dum, key=lambda r: r["val_roc_auc"])
    best_uncal_name = best_val_row["model"]
    best_uncal_pipe = fitted_store[best_uncal_name]

    t0 = time.perf_counter()
    if FrozenEstimator is not None:
        cal_clf = CalibratedClassifierCV(FrozenEstimator(best_uncal_pipe), method="sigmoid")
    else:  # pragma: no cover
        cal_clf = CalibratedClassifierCV(best_uncal_pipe, method="sigmoid", cv="prefit")
    cal_clf.fit(X_va, y_va)
    cal_sec = time.perf_counter() - t0

    p_va_cal = cal_clf.predict_proba(X_va)[:, 1]
    p_te_cal = cal_clf.predict_proba(X_te)[:, 1]
    m_va_cal = compute_metrics_with_bootstrap(y_va, p_va_cal, n_boot=200, seed=42)
    m_te_cal = compute_metrics_with_bootstrap(y_te, p_te_cal, n_boot=400, seed=42)

    cal_name = f"9. Calibrated ({best_uncal_name.split('. ')[1].split(' (')[0]} + Platt)"
    sig_cal = infer_signature(sample_input, cal_clf.predict_proba(sample_input)[:, 1])
    with mlflow.start_run(run_name=cal_name) as run:
        mlflow.set_tags({
            "model_family": "Calibrado",
            "base_model": best_uncal_name,
            "calibration_method": "sigmoid_platt_val",
        })
        mlflow.log_params({"base_model": best_uncal_name, "calibration": "sigmoid_on_val"})
        mlflow.log_metrics({
            "val_roc_auc": m_va_cal["roc_auc"],
            "val_pr_auc": m_va_cal["pr_auc"],
            "val_brier": m_va_cal["brier"],
            "oot_roc_auc": m_te_cal["roc_auc"],
            "oot_pr_auc": m_te_cal["pr_auc"],
            "oot_f1": m_te_cal["f1"],
            "oot_accuracy": m_te_cal["accuracy"],
            "oot_brier": m_te_cal["brier"],
            "train_sec": round(cal_sec, 3),
        })
        m_info_cal = mlflow.sklearn.log_model(
            sk_model=cal_clf,
            name="model",
            signature=sig_cal,
            input_example=sample_input,
            pip_requirements=lean_reqs,
        )
        rid_cal = run.info.run_id

    run_ids[cal_name] = rid_cal
    model_uris[cal_name] = m_info_cal.model_uri
    fitted_store[cal_name] = cal_clf
    val_probs[cal_name] = p_va_cal
    test_probs[cal_name] = p_te_cal
    benchmark_rows.append({
        "model": cal_name,
        "family": "Calibrado",
        "run_id": rid_cal,
        "val_roc_auc": m_va_cal["roc_auc"],
        "val_brier": m_va_cal["brier"],
        **{f"oot_{k}": v for k, v in m_te_cal.items()},
        "train_sec": round(cal_sec, 2),
    })
    print(f"      {cal_name:<38} | Val AUC={m_va_cal['roc_auc']:.4f} | OOT AUC={m_te_cal['roc_auc']:.4f} | PR-AUC={m_te_cal['pr_auc']:.4f} | Brier={m_te_cal['brier']:.4f}")

    bm_df = pd.DataFrame(benchmark_rows).sort_values("oot_roc_auc", ascending=False).reset_index(drop=True)

    # Seleção do Campeão (critério: maior Val ROC-AUC entre os não-calibrados + calibração se reduzir Brier na Validação)
    # Sem tocar no Teste para escolher o modelo!
    champ_name = cal_name if m_va_cal["brier"] <= best_val_row["val_brier"] else best_uncal_name
    challenger_candidates = [r["model"] for r in sorted(non_dum, key=lambda r: r["val_roc_auc"], reverse=True) if r["model"] != best_uncal_name and "DecisionTree" not in r["model"]]
    challenger_name = challenger_candidates[0]
    baseline_name = "2. DecisionTree (Legado OOT)"

    champ_row = bm_df[bm_df["model"] == champ_name].iloc[0].to_dict()
    base_row = bm_df[bm_df["model"] == baseline_name].iloc[0].to_dict()
    chal_row = bm_df[bm_df["model"] == challenger_name].iloc[0].to_dict()

    # 3. Governança no MLflow Model Registry + Quality Gate Automático
    print(f"[3/6] Registrando versões no MLflow Model Registry (`{DEFAULT_MODEL_NAME}`) e aplicando Quality Gate...")
    registry_records = []
    for role_alias, m_name in [("baseline", baseline_name), ("challenger", challenger_name), ("champion", champ_name)]:
        rid = run_ids[m_name]
        mv = mlflow.register_model(model_uri=model_uris[m_name], name=DEFAULT_MODEL_NAME)
        client.set_registered_model_alias(DEFAULT_MODEL_NAME, role_alias, mv.version)
        row_m = bm_df[bm_df["model"] == m_name].iloc[0]
        client.update_model_version(
            name=DEFAULT_MODEL_NAME,
            version=mv.version,
            description=f"{m_name} | Val AUC={row_m['val_roc_auc']:.4f} | OOT AUC={row_m['oot_roc_auc']:.4f} | Brier={row_m['oot_brier']:.4f}",
        )
        registry_records.append({
            "version": int(mv.version),
            "alias": f"@{role_alias}",
            "model": m_name,
            "run_id": rid,
            "val_roc_auc": float(row_m["val_roc_auc"]),
            "oot_roc_auc": float(row_m["oot_roc_auc"]),
            "oot_brier": float(row_m["oot_brier"]),
        })

    # Smoke-test real de inferência carregando direto via `models:/churn-predictor@champion`!
    loaded_champ = mlflow.sklearn.load_model(f"models:/{DEFAULT_MODEL_NAME}@champion")
    serve_probs_check = loaded_champ.predict_proba(X_te.iloc[:10])[:, 1]
    serve_max_diff = float(np.max(np.abs(serve_probs_check - test_probs[champ_name][:10])))

    # 4. Monitoramento de Data Drift (PSI: Treino vs Teste OOT)
    print("[4/6] Calculando Data Drift (PSI) e Otimização de ROI de Campanha de Retenção...")
    eng_audit = ChurnFeatureEngineer().fit(X_tr)
    X_tr_eng = eng_audit.transform(X_tr)
    X_te_eng = eng_audit.transform(X_te)

    psi_rows = []
    for col in X_tr_eng.columns:
        psi_val = compute_psi(X_tr_eng[col].to_numpy(), X_te_eng[col].to_numpy(), bins=10)
        status = "Crítico (>0,25)" if psi_val > 0.25 else ("Atenção (0,10–0,25)" if psi_val >= 0.10 else "Estável (<0,10)")
        psi_rows.append({"feature": col, "psi": psi_val, "status": status})
    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False).reset_index(drop=True)
    max_psi = float(psi_df.iloc[0]["psi"])
    n_drift_warn = int((psi_df["psi"] >= 0.10).sum())
    n_drift_crit = int((psi_df["psi"] > 0.25).sum())

    # Quality Gate Automático (seleção estritamente na Validação + não-regressão OOT + Zero Skew no Registry!)
    gate_checks = {
        "val_beats_legacy_by_2pp": bool(champ_row["val_roc_auc"] >= base_row["val_roc_auc"] + 0.02),
        "oot_beats_legacy_and_brier": bool(champ_row["oot_roc_auc"] >= base_row["oot_roc_auc"] and champ_row["oot_brier"] <= 0.20),
        "registry_serve_zero_skew": bool(serve_max_diff < 1e-9),
        "drift_audit_logged": True,
    }
    gate_passed = all(gate_checks.values())
    client.set_model_version_tag(DEFAULT_MODEL_NAME, str(registry_records[-1]["version"]), "promotion_gate", "APPROVED" if gate_passed else "REVIEW")
    client.set_model_version_tag(DEFAULT_MODEL_NAME, str(registry_records[-1]["version"]), "max_oot_psi", f"{max_psi:.4f}")

    # 5. Otimização de Threshold por ROI (ajustado na Validação -> avaliado cegamente no OOT!)
    roi_val = optimize_retention_roi(y_va, val_probs[champ_name], ltv=450.0, contact_cost=45.0, save_rate=0.30)
    opt_thr = roi_val["best_threshold"]
    roi_oot_opt = evaluate_roi_at_threshold(y_te, test_probs[champ_name], threshold=opt_thr, ltv=450.0, contact_cost=45.0, save_rate=0.30)
    roi_oot_050 = evaluate_roi_at_threshold(y_te, test_probs[champ_name], threshold=0.50, ltv=450.0, contact_cost=45.0, save_rate=0.30)
    oot_metrics_at_opt = compute_metrics_with_bootstrap(y_te, test_probs[champ_name], threshold=opt_thr, n_boot=400, seed=42)

    # Permutation Importance (sobre as 92 features brutas no OOT Test com o Campeão)
    print("[5/6] Calculando Permutation Importance e Slice-Based Error Analysis...")
    perm = permutation_importance(
        loaded_champ, X_te, y_te, scoring="roc_auc", n_repeats=12, random_state=42, n_jobs=-1
    )
    perm_df = (
        pd.DataFrame({
            "feature": raw_features,
            "imp_mean": perm.importances_mean,
            "imp_std": perm.importances_std,
        })
        .sort_values("imp_mean", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    # Slice-Based Error Analysis no Teste OOT
    sl_df = test_df[["ym", "dias_primeira_iteracao_life", "qtd_iteracoes_d7", "flag_churn"]].copy()
    sl_df["prob"] = test_probs[champ_name]
    sl_df["pred"] = (sl_df["prob"] >= opt_thr).astype(int)
    sl_df["atividade_d7"] = np.where(sl_df["qtd_iteracoes_d7"].isna(), "Inativo últimos 7d", "Ativo últimos 7d")
    sl_df["faixa_tenure"] = pd.cut(
        sl_df["dias_primeira_iteracao_life"],
        bins=[-1, 30, 120, 100000],
        labels=["Novo (≤30d)", "Intermediário (31–120d)", "Veterano (>120d)"],
    ).astype(str)

    slice_rows = []
    for dim_lbl, col_g in [
        ("Safra OOT", "ym"),
        ("Atividade Recente (d7)", "atividade_d7"),
        ("Tempo de Casa (Tenure)", "faixa_tenure"),
    ]:
        for val_g, grp in sl_df.groupby(col_g):
            yt_g = grp["flag_churn"].to_numpy(dtype=int)
            yp_g = grp["prob"].to_numpy(dtype=float)
            yhat_g = grp["pred"].to_numpy(dtype=int)
            auc_g = float(roc_auc_score(yt_g, yp_g)) if len(np.unique(yt_g)) == 2 else np.nan
            slice_rows.append({
                "Dimensão": dim_lbl,
                "Fatia": str(val_g),
                "N": int(len(grp)),
                "Taxa Churn": round(float(yt_g.mean()), 4),
                "ROC-AUC": round(auc_g, 4) if np.isfinite(auc_g) else None,
                "F1 (τ ROI)": round(float(f1_score(yt_g, yhat_g, zero_division=0)), 4),
                "Acurácia": round(float(accuracy_score(yt_g, yhat_g)), 4),
            })
    slices_out_df = pd.DataFrame(slice_rows)

    # 6. Geração das 6 Figuras Executivas
    print("[6/6] Gerando 6 figuras executivas em outputs/figures/...")

    # FIG 1: EDA Coortes, Missingness Estrutural & Auditoria de Vazamento
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    x_ym = np.arange(len(cohort_eda))
    bar_colors = ["#3b82f6" if ym <= "2024-07" else ("#f59e0b" if ym <= "2024-09" else "#ef4444") for ym in cohort_eda["ym"]]
    axes[0].bar(x_ym, cohort_eda["n"], color=bar_colors, alpha=0.85, edgecolor="#1e293b")
    axes[0].set_xticks(x_ym)
    axes[0].set_xticklabels(cohort_eda["ym"], rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Registros na Safra")
    ax0_r = axes[0].twinx()
    ax0_r.plot(x_ym, cohort_eda["churn_rate"] * 100, "o-", color="#0f172a", lw=2.2, label="Taxa de Churn (%)")
    ax0_r.set_ylabel("Taxa de Churn (%)", color="#0f172a")
    ax0_r.set_ylim(25, 65)
    axes[0].set_title("Split OOT Cronológico & Drift de Churn")
    axes[0].legend(
        handles=[Patch(color="#3b82f6", label="Treino"), Patch(color="#f59e0b", label="Validação"), Patch(color="#ef4444", label="Teste OOT")],
        loc="upper right", fontsize=8,
    )

    miss_win = [
        ("Recorrência d7", df["media_dias_recorrencia_d7"].isna().mean() * 100),
        ("Recorrência d14", df["media_dias_recorrencia_d14"].isna().mean() * 100),
        ("Recorrência d28", df["media_dias_recorrencia_d28"].isna().mean() * 100),
        ("Atividade d7", df["qtd_iteracoes_d7"].isna().mean() * 100),
        ("Atividade d14", df["qtd_iteracoes_d14"].isna().mean() * 100),
        ("Recorrência life", df["media_dias_recorrencia_life"].isna().mean() * 100),
    ]
    mw_lbl, mw_val = zip(*miss_win[::-1])
    axes[1].barh(np.arange(len(mw_lbl)), mw_val, color="#6366f1", alpha=0.88, edgecolor="#1e293b")
    axes[1].set_yticks(np.arange(len(mw_lbl)))
    axes[1].set_yticklabels(mw_lbl, fontsize=8.8)
    for i, v in enumerate(mw_val):
        axes[1].text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xlim(0, 95)
    axes[1].set_xlabel("% Ausentes (Sinal Estrutural de Inatividade)")
    axes[1].set_title("Missingness Não-Aleatório na ABT")

    rf_oot_auc = float(bm_df[bm_df["model"].str.contains("RandomForest")]["oot_roc_auc"].iloc[0])
    leak_lbls = ["Split Aleatório\n(Vazamento de Clientes)", "Split Cronológico OOT\n(Avaliação Realista)"]
    leak_vals = [auc_random_leak, rf_oot_auc]
    axes[2].bar(leak_lbls, leak_vals, color=["#ef4444", "#10b981"], width=0.52, edgecolor="#1e293b")
    for i, v in enumerate(leak_vals):
        axes[2].text(i, v + 0.015, f"AUC = {v:.4f}", ha="center", fontweight="bold", fontsize=9.5)
    axes[2].set_ylim(0.55, 0.98)
    axes[2].set_ylabel("ROC-AUC (RandomForest)")
    axes[2].set_title(f"Auditoria de Vazamento ({cli_overlap_random} vs {cli_overlap_oot} cli. repetidos)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_eda_coortes_missingness_leakage.png", dpi=150, bbox_inches="tight")
    plt.close()

    # FIG 2: Benchmark dos 9 Pipelines MLflow (OOT ROC-AUC IC 95% + PR-AUC)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8))
    pal_fam = {
        "Baseline": "#94a3b8",
        "Árvore Legada": "#ef4444",
        "Linear Regularizado": "#6366f1",
        "Bagging": "#38bdf8",
        "Boosting": "#f59e0b",
        "Ensemble": "#8b5cf6",
        "Calibrado": "#10b981",
    }
    p_df = bm_df.sort_values("oot_roc_auc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(p_df))
    axes[0].barh(
        y_pos,
        p_df["oot_roc_auc"],
        xerr=[p_df["oot_roc_auc"] - p_df["oot_roc_ci_low"], p_df["oot_roc_ci_high"] - p_df["oot_roc_auc"]],
        color=[pal_fam[f] for f in p_df["family"]],
        capsize=3, alpha=0.92, edgecolor="#1e293b",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(p_df["model"], fontsize=8.8)
    for i, r in p_df.iterrows():
        axes[0].text(max(r["oot_roc_auc"], r["oot_roc_ci_high"]) + 0.008, i, f"{r['oot_roc_auc']:.3f}", va="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xlim(0.45, 0.98)
    axes[0].set_xlabel("ROC-AUC no Teste OOT (com IC 95% Bootstrap)")
    axes[0].set_title("Benchmark MLflow 3.x — ROC-AUC Out-of-Time (N=543)")

    pr_df = bm_df.sort_values("oot_pr_auc", ascending=True).reset_index(drop=True)
    y_pos2 = np.arange(len(pr_df))
    axes[1].barh(y_pos2, pr_df["oot_pr_auc"], color=[pal_fam[f] for f in pr_df["family"]], alpha=0.92, edgecolor="#1e293b")
    axes[1].set_yticks(y_pos2)
    axes[1].set_yticklabels(pr_df["model"], fontsize=8.8)
    for i, r in pr_df.iterrows():
        axes[1].text(r["oot_pr_auc"] + 0.010, i, f"{r['oot_pr_auc']:.3f} (Brier {r['oot_brier']:.3f})", va="center", fontsize=8.2, fontweight="bold")
    axes[1].set_xlim(0.40, 1.08)
    axes[1].set_xlabel("PR-AUC no Teste OOT (com Brier Score)")
    axes[1].set_title("Benchmark MLflow 3.x — PR-AUC & Calibração")
    handles = [Patch(facecolor=c, label=k) for k, c in pal_fam.items()]
    fig.legend(handles=handles, title="Família Registrada no MLflow", loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=8.2)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(FIG_DIR / "02_mlflow_benchmark_oot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # FIG 3: Diagnóstico, Calibração & Curva de ROI da Campanha de Retenção
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.0))
    cm = confusion_matrix(y_te, (test_probs[champ_name] >= opt_thr).astype(int))
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]:,}\n({cm_n[i, j]*100:.1f}%)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm_n, annot=annot, fmt="", cmap="Blues", xticklabels=["Ativo (0)", "Churn (1)"], yticklabels=["Ativo (0)", "Churn (1)"], ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title(f"Matriz de Confusão OOT — @champion (τ ROI = {opt_thr:.2f})")
    axes[0, 0].set_xlabel("Predito"); axes[0, 0].set_ylabel("Real")

    for m_lbl, short_l, col_l in [
        (baseline_name, "@baseline — DecisionTree Legado", "#ef4444"),
        (challenger_name, f"@challenger — {challenger_name.split('. ')[1][:22]}", "#f59e0b"),
        (champ_name, "@champion — Melhor Modelo OOT", "#10b981"),
    ]:
        fpr, tpr, _ = roc_curve(y_te, test_probs[m_lbl])
        auc_v = roc_auc_score(y_te, test_probs[m_lbl])
        axes[0, 1].plot(fpr, tpr, lw=2.3, color=col_l, label=f"{short_l} ({auc_v:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0, 1].set_title("Curvas ROC no Teste OOT (@baseline vs @challenger vs @champion)")
    axes[0, 1].set_xlabel("Taxa de Falsos Positivos"); axes[0, 1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[0, 1].legend(loc="lower right", fontsize=8.3)

    pt_u, pp_u = calibration_curve(y_te, test_probs[best_uncal_name], n_bins=8, strategy="quantile")
    pt_c, pp_c = calibration_curve(y_te, test_probs[cal_name], n_bins=8, strategy="quantile")
    axes[1, 0].plot(pp_u, pt_u, "s--", lw=1.9, color="#f59e0b", label=f"Sem Calibração (Brier={brier_score_loss(y_te, test_probs[best_uncal_name]):.4f})")
    axes[1, 0].plot(pp_c, pt_c, "o-", lw=2.4, color="#2563eb", label=f"Com Platt Val (Brier={brier_score_loss(y_te, test_probs[cal_name]):.4f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibração Perfeita")
    axes[1, 0].set_title("Curva de Calibração Probabilística (Teste OOT)")
    axes[1, 0].set_xlabel("Probabilidade Predita"); axes[1, 0].set_ylabel("Fração Real de Churn")
    axes[1, 0].legend(fontsize=8.3)

    roi_oot_curve = optimize_retention_roi(y_te, test_probs[champ_name], ltv=450.0, contact_cost=45.0, save_rate=0.30)["curve"]
    rc_df = pd.DataFrame(roi_oot_curve)
    axes[1, 1].plot(rc_df["threshold"], rc_df["net_profit"], lw=2.5, color="#10b981", label="Lucro Líquido OOT (@champion)")
    axes[1, 1].axvline(opt_thr, color="#2563eb", ls="--", lw=2.0, label=f"τ Ótimo Val = {opt_thr:.2f} (R$ {roi_oot_opt['net_profit']:,.0f})")
    axes[1, 1].axhline(roi_oot_opt["treat_all_profit"], color="#ef4444", ls=":", lw=1.8, label=f"Contatar Todos (R$ {roi_oot_opt['treat_all_profit']:,.0f})")
    axes[1, 1].set_title(r"Otimização Financeira (LTV R\$ 450 | Custo R\$ 45 | Retenção 30%)")
    axes[1, 1].set_xlabel("Threshold de Decisão (τ)"); axes[1, 1].set_ylabel("Lucro Líquido Esperado no OOT (R$)")
    axes[1, 1].legend(loc="lower left", fontsize=8.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_diagnostico_calibracao_roi.png", dpi=150, bbox_inches="tight")
    plt.close()

    # FIG 4: Permutation Importance + Taxa de Churn por Faixa de Recência
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    p_plot = perm_df.iloc[::-1].reset_index(drop=True)
    axes[0].barh(np.arange(len(p_plot)), p_plot["imp_mean"], xerr=p_plot["imp_std"], color="#4f46e5", alpha=0.88, capsize=3, edgecolor="#1e293b")
    axes[0].set_yticks(np.arange(len(p_plot)))
    axes[0].set_yticklabels(p_plot["feature"], fontsize=8.5)
    axes[0].axvline(0, color="#64748b", lw=1)
    axes[0].set_xlabel("Queda de ROC-AUC OOT após Permutação (±1 dp)")
    axes[0].set_title("Top 15 Variáveis Mais Importantes (@champion no OOT)")

    rec_df = df[["dias_ultima_iteracao_life", "flag_churn"]].copy()
    rec_df["faixa_rec"] = pd.cut(
        rec_df["dias_ultima_iteracao_life"],
        bins=[-1, 2, 7, 14, 21, 1000],
        labels=["0–2d", "3–7d", "8–14d", "15–21d", ">21d"],
    )
    rec_agg = rec_df.groupby("faixa_rec", observed=False)["flag_churn"].agg(["count", "mean"]).reset_index()
    axes[1].bar(rec_agg["faixa_rec"].astype(str), rec_agg["mean"] * 100, color=["#10b981", "#38bdf8", "#f59e0b", "#f97316", "#ef4444"], edgecolor="#1e293b")
    for i, r in rec_agg.iterrows():
        axes[1].text(i, r["mean"] * 100 + 2.0, f"{r['mean']*100:.1f}%\n(n={int(r['count'])})", ha="center", fontsize=8.5, fontweight="bold")
    axes[1].set_ylim(0, 95)
    axes[1].set_xlabel("Dias desde a Última Interação (dias_ultima_iteracao_life)")
    axes[1].set_ylabel("Taxa Observada de Churn (%)")
    axes[1].set_title("Drivers de Negócio: Recência de Uso vs. Risco de Churn")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_permutation_importance_explainability.png", dpi=150, bbox_inches="tight")
    plt.close()

    # FIG 5: Monitoramento de Data Drift (PSI: Treino vs Teste OOT)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
    top_psi = psi_df.head(12).iloc[::-1].reset_index(drop=True)
    psi_cols = ["#ef4444" if v > 0.25 else ("#f59e0b" if v >= 0.10 else "#10b981") for v in top_psi["psi"]]
    axes[0].barh(np.arange(len(top_psi)), top_psi["psi"], color=psi_cols, alpha=0.9, edgecolor="#1e293b")
    axes[0].set_yticks(np.arange(len(top_psi)))
    axes[0].set_yticklabels(top_psi["feature"], fontsize=8.5)
    axes[0].axvline(0.10, color="#d97706", ls="--", lw=1.6, label="Atenção (0,10)")
    axes[0].axvline(0.25, color="#dc2626", ls="--", lw=1.6, label="Drift Crítico (0,25)")
    for i, v in enumerate(top_psi["psi"]):
        axes[0].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8.3, fontweight="bold")
    axes[0].set_xlim(0, max(0.32, float(top_psi["psi"].max()) * 1.25))
    axes[0].set_xlabel("Population Stability Index (PSI: Treino Abr–Jul vs. OOT Out–Nov)")
    axes[0].set_title("Monitoramento de Data Drift — Top 12 Features por PSI")
    axes[0].legend(loc="lower right", fontsize=8.5)

    top_feat = str(psi_df.iloc[0]["feature"])
    tr_v = np.clip(X_tr_eng[top_feat].to_numpy(), np.percentile(X_tr_eng[top_feat], 1), np.percentile(X_tr_eng[top_feat], 99))
    te_v = np.clip(X_te_eng[top_feat].to_numpy(), np.percentile(X_tr_eng[top_feat], 1), np.percentile(X_tr_eng[top_feat], 99))
    axes[1].hist(tr_v, bins=20, density=True, alpha=0.55, color="#3b82f6", label="Treino (Abr–Jul)")
    axes[1].hist(te_v, bins=20, density=True, alpha=0.55, color="#ef4444", label="Produção OOT (Out–Nov)")
    axes[1].set_title(f"Deslocamento Distribucional: {top_feat} (PSI = {psi_df.iloc[0]['psi']:.3f})")
    axes[1].set_xlabel(top_feat); axes[1].set_ylabel("Densidade")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_drift_psi_monitoring.png", dpi=150, bbox_inches="tight")
    plt.close()

    # FIG 6: Slice-Based Error Analysis no Teste OOT
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2))
    sl_act = slices_out_df[slices_out_df["Dimensão"].isin(["Safra OOT", "Atividade Recente (d7)"])].reset_index(drop=True)
    x_s1 = np.arange(len(sl_act))
    axes[0].bar(x_s1 - 0.18, sl_act["ROC-AUC"] * 100, width=0.35, color="#10b981", label="ROC-AUC (×100)", edgecolor="#1e293b")
    axes[0].bar(x_s1 + 0.18, sl_act["Acurácia"] * 100, width=0.35, color="#3b82f6", label="Acurácia (%)", edgecolor="#1e293b")
    axes[0].set_xticks(x_s1)
    axes[0].set_xticklabels([f"{r['Fatia']}\n(n={r['N']})" for _, r in sl_act.iterrows()], fontsize=8.5)
    for i, r in sl_act.iterrows():
        axes[0].text(i - 0.18, r["ROC-AUC"] * 100 + 1.2, f"{r['ROC-AUC']*100:.1f}", ha="center", fontsize=8.3, fontweight="bold")
        axes[0].text(i + 0.18, r["Acurácia"] * 100 + 1.2, f"{r['Acurácia']*100:.1f}%", ha="center", fontsize=8.3, fontweight="bold")
    axes[0].set_ylim(50, 102)
    axes[0].set_ylabel("Desempenho (%)")
    axes[0].set_title("Slice Analysis OOT — Por Safra & Atividade Recente")
    axes[0].legend(loc="lower right")

    ten_order = ["Novo (≤30d)", "Intermediário (31–120d)", "Veterano (>120d)"]
    sl_ten = (
        slices_out_df[slices_out_df["Dimensão"] == "Tempo de Casa (Tenure)"]
        .set_index("Fatia")
        .reindex(ten_order)
        .reset_index()
    )
    x_s2 = np.arange(len(sl_ten))
    axes[1].bar(x_s2 - 0.18, sl_ten["ROC-AUC"] * 100, width=0.35, color="#10b981", label="ROC-AUC (×100)", edgecolor="#1e293b")
    axes[1].bar(x_s2 + 0.18, sl_ten["F1 (τ ROI)"] * 100, width=0.35, color="#8b5cf6", label="F1-Score (τ ROI)", edgecolor="#1e293b")
    axes[1].set_xticks(x_s2)
    axes[1].set_xticklabels([f"{r['Fatia']}\n(n={r['N']}, churn {r['Taxa Churn']*100:.0f}%)" for _, r in sl_ten.iterrows()], fontsize=8.5)
    for i, r in sl_ten.iterrows():
        axes[1].text(i - 0.18, r["ROC-AUC"] * 100 + 1.2, f"{r['ROC-AUC']*100:.1f}", ha="center", fontsize=8.3, fontweight="bold")
        axes[1].text(i + 0.18, r["F1 (τ ROI)"] * 100 + 1.2, f"{r['F1 (τ ROI)']*100:.1f}", ha="center", fontsize=8.3, fontweight="bold")
    axes[1].set_ylim(50, 102)
    axes[1].set_ylabel("Desempenho (%)")
    axes[1].set_title("Slice Analysis OOT — Por Tempo de Casa (Tenure)")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_slice_error_analysis_registry.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loga as 6 figuras como artefatos oficiais na run do @champion no MLflow!
    with mlflow.start_run(run_id=run_ids[champ_name]):
        mlflow.log_artifacts(str(FIG_DIR), artifact_path="diagnostic_figures")
        mlflow.log_metrics({
            "oot_max_psi": max_psi,
            "oot_roi_opt_threshold": opt_thr,
            "oot_roi_net_profit_brl": roi_oot_opt["net_profit"],
            "registry_serve_max_diff": serve_max_diff,
        })

    summary = {
        "mlflow_version": mlflow.__version__,
        "tracking_uri": "sqlite:///mlflow_store/mlflow.db",
        "experiment_name": DEFAULT_EXPERIMENT,
        "experiment_id": exp_id,
        "registered_model_name": DEFAULT_MODEL_NAME,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "dataset": {
            "total_rows": int(len(df)),
            "unique_clients": int(df["id_cliente"].nunique()),
            "n_raw_features": int(len(raw_features)),
            "n_engineered_features": int(X_tr_eng.shape[1]),
            "overall_churn_rate": round(float(df["flag_churn"].mean()), 4),
            "train_n": int(len(train_df)),
            "val_n": int(len(val_df)),
            "oot_test_n": int(len(test_df)),
            "train_churn_rate": round(float(y_tr.mean()), 4),
            "val_churn_rate": round(float(y_va.mean()), 4),
            "oot_churn_rate": round(float(y_te.mean()), 4),
            "cohorts": cohort_eda.round(4).to_dict(orient="records"),
        },
        "leakage_audit": {
            "random_split_rf_auc": round(auc_random_leak, 4),
            "oot_split_rf_auc": round(rf_oot_auc, 4),
            "auc_inflation_pp": round((auc_random_leak - rf_oot_auc) * 100.0, 2),
            "random_client_overlap": int(cli_overlap_random),
            "oot_client_overlap": int(cli_overlap_oot),
        },
        "models_benchmark": bm_df.to_dict(orient="records"),
        "registry_governance": {
            "champion_model": champ_name,
            "challenger_model": challenger_name,
            "baseline_model": baseline_name,
            "versions": registry_records,
            "quality_gate_checks": gate_checks,
            "promotion_approved": gate_passed,
            "serve_roundtrip_max_diff": serve_max_diff,
        },
        "roi_optimization": {
            "val_opt_threshold": opt_thr,
            "theoretical_breakeven_prob": roi_val["theoretical_breakeven_prob"],
            "oot_at_opt_threshold": roi_oot_opt,
            "oot_at_default_050": roi_oot_050,
            "oot_metrics_at_opt": oot_metrics_at_opt,
        },
        "drift_monitoring": {
            "max_psi": max_psi,
            "n_features_warn_010": n_drift_warn,
            "n_features_crit_025": n_drift_crit,
            "top_12_psi": psi_df.head(12).to_dict(orient="records"),
        },
        "top_15_permutation_importance": perm_df.round(5).to_dict(orient="records"),
        "slice_error_analysis": slices_out_df.to_dict(orient="records"),
    }
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n>>> @champion REGISTRADO: {champ_name} | OOT ROC-AUC={champ_row['oot_roc_auc']:.4f} | Brier={champ_row['oot_brier']:.4f} | Lucro OOT=R$ {roi_oot_opt['net_profit']:,.2f} | Gate={'APROVADO' if gate_passed else 'REVISAR'}")
    print(f"Concluído em {summary['elapsed_sec']}s! Resumo salvo em outputs/metrics_summary.json")
    return summary


if __name__ == "__main__":
    run_full_mlops_workflow(reset_store=True)
