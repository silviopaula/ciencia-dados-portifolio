"""Executor completo do benchmark NLP + Multimodal (Olist E-Commerce Brasil).

Roda todo o pipeline livre de vazamento com split cronológico estrito, salva
métricas verificadas em `outputs/metrics_summary.json` e gera as 6 figuras
executivas de alta resolução em `outputs/figures/`.
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from nlp_pipeline import (
    ASPECT_PATTERNS,
    CLASSES_ORDER,
    CustomerSupportRAG,
    bootstrap_ci,
    build_tabular_preprocessor,
    build_text_vectorizer,
    chronological_split,
    clean_pt_text,
    enrich_text_signals,
    evaluate_multiclass,
    evaluate_slices,
    lexicon_predict_labels,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PARQUET = PROJECT_ROOT / "data" / "olist_reviews_nlp.parquet"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fbfcfe",
    "axes.edgecolor": "#cbd5e1",
    "axes.titleweight": "bold",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
})


def _proba_aligned(model: Any, X: Any) -> np.ndarray:
    """Garante colunas de probabilidade na ordem exata CLASSES_ORDER = [Negativo, Neutro, Positivo]."""
    raw_p = model.predict_proba(X)
    cls = list(model.classes_)
    if cls == CLASSES_ORDER:
        return raw_p
    if cls == [0, 1, 2]:
        return raw_p
    idx = [cls.index(c) for c in CLASSES_ORDER]
    return raw_p[:, idx]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Recompacta o parquet com zstd para ficar leve no repositório
    df_all = pd.read_parquet(DATA_PARQUET)
    df_all.to_parquet(DATA_PARQUET, index=False, compression="zstd")
    pq_mb = DATA_PARQUET.stat().st_size / (1024 * 1024)
    print(f"[1/7] Base carregada: {len(df_all):,} pedidos únicos ({pq_mb:.2f} MB zstd)")

    # Subconjunto de avaliações com texto escrito (>= 3 caracteres)
    df_txt = df_all[df_all["texto_completo"].fillna("").str.len() >= 3].copy().reset_index(drop=True)
    df_txt = enrich_text_signals(df_txt)
    print(f"      Avaliações com texto escrito (>=3 chars): {len(df_txt):,} ({len(df_txt)/len(df_all)*100:.1f}%)")

    # Split cronológico estrito (70% Treino / 15% Validação / 15% Teste Out-of-Time)
    train_df, val_df, test_df = chronological_split(df_txt, 0.70, 0.15)
    print(
        f"[2/7] Split cronológico: "
        f"Treino={len(train_df):,} ({train_df['order_purchase_timestamp'].min():%Y-%m-%d}..{train_df['order_purchase_timestamp'].max():%Y-%m-%d}) | "
        f"Val={len(val_df):,} ({val_df['order_purchase_timestamp'].min():%Y-%m-%d}..{val_df['order_purchase_timestamp'].max():%Y-%m-%d}) | "
        f"Teste={len(test_df):,} ({test_df['order_purchase_timestamp'].min():%Y-%m-%d}..{test_df['order_purchase_timestamp'].max():%Y-%m-%d})"
    )

    # --- FIGURA 1: EDA Sentimento, Logística, Temporal e Comprimento ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    score_counts = df_txt["review_score"].value_counts().sort_index()
    colors_s = ["#ef4444", "#f97316", "#f59e0b", "#38bdf8", "#10b981"]
    axes[0, 0].bar(score_counts.index.astype(str) + "★", score_counts.values, color=colors_s, edgecolor="#1e293b", alpha=0.9)
    for i, v in enumerate(score_counts.values):
        axes[0, 0].text(i, v + 350, f"{v:,}\n({v/len(df_txt)*100:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[0, 0].set_title("Distribuição Real das Notas (Avaliações com Texto, N=42.173)")
    axes[0, 0].set_ylabel("Quantidade de Avaliações")
    axes[0, 0].set_ylim(0, score_counts.max() * 1.18)

    delay_by_score = df_txt.groupby("review_score")["entregue_com_atraso"].mean() * 100
    not_deliv_by_score = df_txt.groupby("review_score")["pedido_nao_entregue"].mean() * 100
    x_pos = np.arange(len(delay_by_score))
    axes[0, 1].bar(x_pos - 0.18, delay_by_score.values, width=0.35, label="Entregue com Atraso (%)", color="#ef4444", alpha=0.9)
    axes[0, 1].bar(x_pos + 0.18, not_deliv_by_score.values, width=0.35, label="Pedido Não Entregue (%)", color="#64748b", alpha=0.9)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([f"{s}★" for s in delay_by_score.index])
    axes[0, 1].set_title("Impacto Logístico: Atraso e Não-Entrega por Nota")
    axes[0, 1].set_ylabel("Taxa (%)")
    axes[0, 1].legend()

    monthly = df_txt.set_index("order_purchase_timestamp").resample("ME")["detrator"].agg(["count", "mean"])
    monthly = monthly[monthly["count"] >= 30]
    ax_m = axes[1, 0]
    ax_m.bar(monthly.index, monthly["count"], width=22, color="#93c5fd", alpha=0.65, label="Volume mensal")
    ax_m.axvline(val_df["order_purchase_timestamp"].min(), color="#f59e0b", ls="--", lw=2, label="Início Validação")
    ax_m.axvline(test_df["order_purchase_timestamp"].min(), color="#ef4444", ls="--", lw=2, label="Início Teste OOT")
    ax_m.set_title("Evolução Temporal & Corte Cronológico (Treino / Val / Teste)")
    ax_m.set_ylabel("Avaliações / Mês")
    ax_m2 = ax_m.twinx()
    ax_m2.plot(monthly.index, monthly["mean"] * 100, color="#dc2626", lw=2.2, marker="o", ms=3.5, label="% Detratores (1–2★)")
    ax_m2.set_ylabel("% Detratores", color="#dc2626")
    ax_m.tick_params(axis="x", rotation=30)
    lines1, lbl1 = ax_m.get_legend_handles_labels()
    lines2, lbl2 = ax_m2.get_legend_handles_labels()
    ax_m.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="upper left")

    len_clip = df_txt.assign(palavras_clip=df_txt["qtd_palavras"].clip(upper=50))
    sns.boxplot(
        data=len_clip, x="sentimento_real", y="palavras_clip", order=CLASSES_ORDER,
        palette={"Negativo": "#ef4444", "Neutro": "#f59e0b", "Positivo": "#10b981"},
        ax=axes[1, 1], showfliers=False,
    )
    axes[1, 1].set_title("Comprimento do Comentário (Palavras) por Polaridade")
    axes[1, 1].set_xlabel("Polaridade Real (review_score)")
    axes[1, 1].set_ylabel("Palavras por Avaliação (até 50)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_eda_sentimento_logistica.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- ABLATION DE PRÉ-PROCESSAMENTO: PRESERVAR NEGAÇÕES vs REMOVER NEGAÇÕES (Validação) ---
    print("[3/7] Testando impacto de preservar negações em PT-BR na validação...")
    y_train = train_df["sentimento_real"].values
    y_val = val_df["sentimento_real"].values
    y_test = test_df["sentimento_real"].values
    y_train_idx = np.array([CLASSES_ORDER.index(c) for c in y_train])
    y_val_idx = np.array([CLASSES_ORDER.index(c) for c in y_val])
    y_test_idx = np.array([CLASSES_ORDER.index(c) for c in y_test])

    tr_naive = train_df["texto_completo"].map(lambda s: clean_pt_text(s, preserve_negations=False))
    va_naive = val_df["texto_completo"].map(lambda s: clean_pt_text(s, preserve_negations=False))
    vec_naive = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_features=20000)
    lr_naive = LogisticRegression(C=1.5, max_iter=500, class_weight="balanced", random_state=42)
    lr_naive.fit(vec_naive.fit_transform(tr_naive), y_train)
    f1_naive_val = float(f1_score(y_val, lr_naive.predict(vec_naive.transform(va_naive)), average="macro"))

    # --- VETORIZAÇÃO ZERO-LEAKAGE (FIT APENAS NO TREINO) ---
    text_vec = build_text_vectorizer()
    X_tr_txt = text_vec.fit_transform(train_df["texto_limpo"])
    X_va_txt = text_vec.transform(val_df["texto_limpo"])
    X_te_txt = text_vec.transform(test_df["texto_limpo"])

    lr_neg_check = LogisticRegression(C=1.5, max_iter=500, class_weight="balanced", random_state=42)
    lr_neg_check.fit(X_tr_txt, y_train)
    f1_neg_val = float(f1_score(y_val, lr_neg_check.predict(X_va_txt), average="macro"))
    print(f"      Validação Macro-F1: Stopwords ingênuas (sem negação) = {f1_naive_val:.4f} -> Com negações + Char-Ngram = {f1_neg_val:.4f} ({(f1_neg_val-f1_naive_val)*100:+.2f} p.p.)")

    tab_prep = build_tabular_preprocessor()
    X_tr_tab = sparse.csr_matrix(tab_prep.fit_transform(train_df))
    X_va_tab = sparse.csr_matrix(tab_prep.transform(val_df))
    X_te_tab = sparse.csr_matrix(tab_prep.transform(test_df))

    # Fusão esparsa Multimodal (Texto TF-IDF + Operacional/Logística escalada com peso equilibrado)
    X_tr_mm = sparse.hstack([X_tr_txt, X_tr_tab * 0.25], format="csr")
    X_va_mm = sparse.hstack([X_va_txt, X_va_tab * 0.25], format="csr")
    X_te_mm = sparse.hstack([X_te_txt, X_te_tab * 0.25], format="csr")

    # Features densas para GBDTs (LightGBM & XGBoost): SVD 64D + OOF NLP Probs (5-fold zero leakage!) + Tabular
    print("[4/7] Construindo representação densa zero-leakage (SVD 64D + OOF NLP + Logística) para GBDTs...")
    svd = TruncatedSVD(n_components=64, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_txt)
    X_va_svd = svd.transform(X_va_txt)
    X_te_svd = svd.transform(X_te_txt)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lr = cross_val_predict(
        LogisticRegression(C=1.5, max_iter=500, random_state=42),
        X_tr_txt, y_train, cv=skf, method="predict_proba", n_jobs=-1
    )
    oof_nb = cross_val_predict(
        ComplementNB(alpha=0.5),
        X_tr_txt, y_train, cv=skf, method="predict_proba", n_jobs=-1
    )
    lr_txt_full = LogisticRegression(C=1.5, max_iter=500, random_state=42).fit(X_tr_txt, y_train)
    nb_txt_full = ComplementNB(alpha=0.5).fit(X_tr_txt, y_train)

    X_tr_dense = np.hstack([X_tr_tab.toarray(), X_tr_svd, oof_lr, oof_nb])
    X_va_dense = np.hstack([X_va_tab.toarray(), X_va_svd, _proba_aligned(lr_txt_full, X_va_txt), _proba_aligned(nb_txt_full, X_va_txt)])
    X_te_dense = np.hstack([X_te_tab.toarray(), X_te_svd, _proba_aligned(lr_txt_full, X_te_txt), _proba_aligned(nb_txt_full, X_te_txt)])

    # --- BENCHMARK DOS 10 MODELOS NO TESTE OUT-OF-TIME ---
    print("[5/7] Treinando e avaliando os 10 modelos no split cronológico Out-of-Time...")
    records: List[Dict[str, Any]] = []
    stored_preds: Dict[str, Dict[str, Any]] = {}

    def _register(name: str, modality: str, fit_sec: float, inf_ms_1k: float, size_mb: float, y_pred: np.ndarray, y_proba: np.ndarray | None) -> None:
        m = evaluate_multiclass(y_test, y_pred, y_proba)
        ci = bootstrap_ci(y_test, y_pred, y_proba[:, 0] if y_proba is not None else None, n_boot=500, seed=42)
        rec = {
            "model": name,
            "modality": modality,
            **{k: round(v, 4) if not np.isnan(v) else None for k, v in m.items()},
            "f1_macro_ci_low": round(ci["f1_macro_ci95"][0], 4),
            "f1_macro_ci_high": round(ci["f1_macro_ci95"][1], 4),
            "pr_auc_ci_low": round(ci["detrator_pr_auc_ci95"][0], 4) if "detrator_pr_auc_ci95" in ci else None,
            "pr_auc_ci_high": round(ci["detrator_pr_auc_ci95"][1], 4) if "detrator_pr_auc_ci95" in ci else None,
            "train_sec": round(fit_sec, 2),
            "infer_ms_1k": round(inf_ms_1k, 2),
            "size_mb": round(size_mb, 2),
        }
        records.append(rec)
        stored_preds[name] = {"pred": y_pred, "proba": y_proba}
        print(f"      {name:<42} | Macro-F1={m['f1_macro']:.4f} | Acc={m['accuracy']:.4f} | Det PR-AUC={m['detrator_pr_auc']:.4f} | {fit_sec:.2f}s")

    # 1. Dummy Baseline
    t0 = time.perf_counter()
    dum = DummyClassifier(strategy="most_frequent").fit(X_tr_tab, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    p_dum = dum.predict(X_te_tab)
    pr_dum = _proba_aligned(dum, X_te_tab)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("1. Dummy (Classe Majoritária)", "Baseline", fit_s, inf_ms, 0.01, p_dum, pr_dum)

    # 2. Lexicon PT-BR (Negation-Aware)
    t1 = time.perf_counter()
    p_lex = lexicon_predict_labels(test_df["lexicon_score"].values)
    # Transforma score léxico em probabilidade sigmoide de detrator para PR/ROC-AUC
    p_det_lex = 1.0 / (1.0 + np.exp(test_df["lexicon_score"].values))
    pr_lex = np.column_stack([p_det_lex, np.full_like(p_det_lex, 0.15) * (1 - p_det_lex), (1 - p_det_lex) * 0.85])
    pr_lex = pr_lex / pr_lex.sum(axis=1, keepdims=True)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("2. Léxico PT-BR (Com Negações)", "Léxico", 0.0, inf_ms, 0.02, p_lex, pr_lex)

    # 3. Tabular Logística Only (LightGBM — sem TF-IDF!)
    # Remove lexicon_score para medir sinal estritamente operacional/logístico
    t0 = time.perf_counter()
    lgb_tab = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", random_state=42, verbosity=-1, n_jobs=-1,
    )
    lgb_tab.fit(X_tr_tab, y_train_idx)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_lt = lgb_tab.predict_proba(X_te_tab)
    p_lt = np.array([CLASSES_ORDER[i] for i in pr_lt.argmax(axis=1)])
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("3. LightGBM (Logística + Metadados Only)", "Tabular Only", fit_s, inf_ms, len(pickle.dumps(lgb_tab)) / 1e6, p_lt, pr_lt)

    # 4. Complement Naive Bayes (Texto Only)
    t0 = time.perf_counter()
    cnb = ComplementNB(alpha=0.5).fit(X_tr_txt, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_cnb = _proba_aligned(cnb, X_te_txt)
    p_cnb = cnb.predict(X_te_txt)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("4. Complement Naive Bayes (TF-IDF)", "Texto Only", fit_s, inf_ms, len(pickle.dumps(cnb)) / 1e6, p_cnb, pr_cnb)

    # 5. Logistic Regression (Texto Only)
    t0 = time.perf_counter()
    lr_txt = LogisticRegression(C=1.5, max_iter=600, class_weight="balanced", random_state=42).fit(X_tr_txt, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_lrt = _proba_aligned(lr_txt, X_te_txt)
    p_lrt = lr_txt.predict(X_te_txt)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("5. Regressão Logística L2 (TF-IDF)", "Texto Only", fit_s, inf_ms, len(pickle.dumps(lr_txt)) / 1e6, p_lrt, pr_lrt)

    # 6. Calibrated LinearSVC (Texto Only)
    t0 = time.perf_counter()
    svc_txt = CalibratedClassifierCV(LinearSVC(C=0.45, class_weight="balanced", random_state=42), cv=3, method="sigmoid")
    svc_txt.fit(X_tr_txt, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_svt = _proba_aligned(svc_txt, X_te_txt)
    p_svt = svc_txt.predict(X_te_txt)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("6. Calibrated LinearSVC (TF-IDF)", "Texto Only", fit_s, inf_ms, len(pickle.dumps(svc_txt)) / 1e6, p_svt, pr_svt)

    # 7. Logistic Regression (Multimodal: TF-IDF + Logística)
    t0 = time.perf_counter()
    lr_mm = LogisticRegression(C=1.0, max_iter=600, class_weight="balanced", random_state=42).fit(X_tr_mm, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_lrmm = _proba_aligned(lr_mm, X_te_mm)
    p_lrmm = lr_mm.predict(X_te_mm)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("7. Regressão Logística (Multimodal)", "Multimodal", fit_s, inf_ms, len(pickle.dumps(lr_mm)) / 1e6, p_lrmm, pr_lrmm)

    # 8. Calibrated LinearSVC (Multimodal: TF-IDF + Logística)
    t0 = time.perf_counter()
    svc_mm = CalibratedClassifierCV(LinearSVC(C=0.35, class_weight="balanced", random_state=42), cv=3, method="sigmoid")
    svc_mm.fit(X_tr_mm, y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_svmm = _proba_aligned(svc_mm, X_te_mm)
    p_svmm = svc_mm.predict(X_te_mm)
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("8. Calibrated LinearSVC (Multimodal)", "Multimodal", fit_s, inf_ms, len(pickle.dumps(svc_mm)) / 1e6, p_svmm, pr_svmm)

    # 9. LightGBM (Multimodal Stacking: OOF NLP + SVD 64D + Logística)
    t0 = time.perf_counter()
    lgb_mm = lgb.LGBMClassifier(
        n_estimators=320, learning_rate=0.04, num_leaves=31, subsample_for_bin=50000,
        colsample_bytree=0.8, subsample=0.85, reg_alpha=1.0, reg_lambda=2.0,
        class_weight="balanced", random_state=42, verbosity=-1, n_jobs=-1,
    )
    lgb_mm.fit(X_tr_dense, y_train_idx, eval_set=[(X_va_dense, y_val_idx)], callbacks=[lgb.early_stopping(30, verbose=False)])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_lgb = lgb_mm.predict_proba(X_te_dense)
    p_lgb = np.array([CLASSES_ORDER[i] for i in pr_lgb.argmax(axis=1)])
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("9. LightGBM (Multimodal Stacking)", "Multimodal", fit_s, inf_ms, len(pickle.dumps(lgb_mm)) / 1e6, p_lgb, pr_lgb)

    # 10. XGBoost (Multimodal Stacking: OOF NLP + SVD 64D + Logística)
    # Calcula sample_weight balanceado para XGBoost multiclasse
    cls_counts = np.bincount(y_train_idx)
    w_map = len(y_train_idx) / (len(cls_counts) * cls_counts)
    sw_tr = w_map[y_train_idx]
    t0 = time.perf_counter()
    xgb_mm = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0, eval_metric="mlogloss", early_stopping_rounds=25,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    xgb_mm.fit(X_tr_dense, y_train_idx, sample_weight=sw_tr, eval_set=[(X_va_dense, y_val_idx)], verbose=False)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr_xgb = xgb_mm.predict_proba(X_te_dense)
    p_xgb = np.array([CLASSES_ORDER[i] for i in pr_xgb.argmax(axis=1)])
    inf_ms = (time.perf_counter() - t1) * 1000 / (len(y_test) / 1000)
    _register("10. XGBoost (Multimodal Stacking)", "Multimodal", fit_s, inf_ms, len(pickle.dumps(xgb_mm)) / 1e6, p_xgb, pr_xgb)

    res_df = pd.DataFrame(records).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    best_name = str(res_df.iloc[0]["model"])
    best_pred = stored_preds[best_name]["pred"]
    best_proba = stored_preds[best_name]["proba"]
    print(f"\n>>> CAMPEÃO OUT-OF-TIME (Macro-F1): {best_name} | Macro-F1={res_df.iloc[0]['f1_macro']:.4f} | Detrator PR-AUC={res_df.iloc[0]['detrator_pr_auc']:.4f}")

    # --- FIGURA 2: Benchmark de Modelos + IC 95% Bootstrap + Ablation Multimodal ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    plot_df = res_df.sort_values("f1_macro", ascending=True)
    pal_mod = {"Baseline": "#94a3b8", "Léxico": "#f59e0b", "Tabular Only": "#fb7185", "Texto Only": "#38bdf8", "Multimodal": "#10b981"}
    bar_cols = [pal_mod[m] for m in plot_df["modality"]]
    y_pos = np.arange(len(plot_df))
    err_low = plot_df["f1_macro"] - plot_df["f1_macro_ci_low"]
    err_high = plot_df["f1_macro_ci_high"] - plot_df["f1_macro"]
    axes[0].barh(y_pos, plot_df["f1_macro"], xerr=[err_low, err_high], color=bar_cols, alpha=0.92, capsize=3, edgecolor="#1e293b")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["model"], fontsize=9)
    for idx_b, val in enumerate(plot_df["f1_macro"]):
        axes[0].text(val + 0.012, idx_b, f"{val:.3f}", va="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xlim(0, 0.88)
    axes[0].set_title("Benchmark Out-of-Time: Macro-F1 com IC 95% Bootstrap")
    axes[0].set_xlabel("Macro-F1 (Teste Cronológico OOT)")

    pr_df = res_df.dropna(subset=["detrator_pr_auc"]).sort_values("detrator_pr_auc", ascending=True)
    y_pos2 = np.arange(len(pr_df))
    axes[1].barh(y_pos2, pr_df["detrator_pr_auc"], color=[pal_mod[m] for m in pr_df["modality"]], alpha=0.92, edgecolor="#1e293b")
    axes[1].set_yticks(y_pos2)
    axes[1].set_yticklabels(pr_df["model"], fontsize=9)
    for idx_b, val in enumerate(pr_df["detrator_pr_auc"]):
        axes[1].text(val + 0.01, idx_b, f"{val:.3f}", va="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_title("Detecção de Detratores (1–2★): PR-AUC (Ablation Multimodal)")
    axes[1].set_xlabel("Precision-Recall AUC (Classe Negativo)")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=k) for k, c in pal_mod.items()]
    fig.legend(handles=handles, title="Modalidade", loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(FIG_DIR / "02_benchmark_modelos_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- FIGURA 3: Matriz de Confusão + Curva ROC OvR + Curva PR Detratores ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    cm = confusion_matrix(y_test, best_pred, labels=CLASSES_ORDER)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)" for j in range(3)] for i in range(3)])
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", xticklabels=CLASSES_ORDER, yticklabels=CLASSES_ORDER, ax=axes[0], cbar=False)
    axes[0].set_title(f"Matriz de Confusão — {best_name.split('. ')[1]}")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")

    # ROC Multiclasse OvR
    roc_colors = {"Negativo": "#ef4444", "Neutro": "#f59e0b", "Positivo": "#10b981"}
    for i, cname in enumerate(CLASSES_ORDER):
        y_bin = (y_test == cname).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, best_proba[:, i])
        from sklearn.metrics import auc as sk_auc
        axes[1].plot(fpr, tpr, lw=2.2, color=roc_colors[cname], label=f"{cname} (AUC={sk_auc(fpr, tpr):.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("Curvas ROC (One-vs-Rest) — Teste OOT")
    axes[1].set_xlabel("Taxa de Falsos Positivos")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].legend(loc="lower right")

    # PR Curve Detratores comparando modalidades
    y_det_test = (y_test == "Negativo").astype(int)
    for m_key, col_c in [
        ("3. LightGBM (Logística + Metadados Only)", "#fb7185"),
        ("5. Regressão Logística L2 (TF-IDF)", "#38bdf8"),
        ("7. Regressão Logística (Multimodal)", "#8b5cf6"),
        (best_name, "#10b981"),
    ]:
        if m_key not in stored_preds:
            continue
        p_d = stored_preds[m_key]["proba"][:, 0]
        prec, rec, _ = precision_recall_curve(y_det_test, p_d)
        ap = float(res_df.loc[res_df["model"] == m_key, "detrator_pr_auc"].iloc[0])
        axes[2].plot(rec, prec, lw=2.1, color=col_c, label=f"{m_key.split('. ')[1]} (AP={ap:.3f})")
    axes[2].axhline(y_det_test.mean(), color="#64748b", ls=":", label=f"Prevalência ({y_det_test.mean()*100:.1f}%)")
    axes[2].set_title("Curva Precision-Recall — Triagem de Detratores")
    axes[2].set_xlabel("Recall (Detratores Capturados)")
    axes[2].set_ylabel("Precisão na Fila SAC")
    axes[2].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_matriz_confusao_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- FIGURA 4: Calibração & Otimização de Threshold para SAC ---
    brier_best = float(brier_score_loss(y_det_test, best_proba[:, 0]))
    prob_true, prob_pred = calibration_curve(y_det_test, best_proba[:, 0], n_bins=10, strategy="quantile")
    # Otimização de threshold no conjunto de VALIDAÇÃO (zero leakage no teste!)
    if "LightGBM (Multimodal" in best_name:
        val_p_det = lgb_mm.predict_proba(X_va_dense)[:, 0]
    elif "XGBoost" in best_name:
        val_p_det = xgb_mm.predict_proba(X_va_dense)[:, 0]
    elif "LinearSVC (Multimodal" in best_name:
        val_p_det = _proba_aligned(svc_mm, X_va_mm)[:, 0]
    else:
        val_p_det = _proba_aligned(lr_mm, X_va_mm)[:, 0]

    y_det_val = (y_val == "Negativo").astype(int)
    thrs = np.linspace(0.15, 0.90, 76)
    prec_l, rec_l, f1_l, vol_l = [], [], [], []
    for t in thrs:
        pred_t = (val_p_det >= t).astype(int)
        prec_l.append(float((y_det_val[pred_t == 1].mean()) if pred_t.sum() > 0 else 1.0))
        rec_l.append(float((pred_t[y_det_val == 1].mean())))
        f1_l.append(float(f1_score(y_det_val, pred_t, zero_division=0)))
        vol_l.append(float(pred_t.mean() * 100))

    best_thr_idx = int(np.argmax(f1_l))
    opt_thr = float(thrs[best_thr_idx])
    # Threshold operacional alta precisão (>= 85% precisão na validação)
    hi_prec_idx = next((i for i, p in enumerate(prec_l) if p >= 0.85), best_thr_idx)
    sac_thr = float(thrs[hi_prec_idx])

    test_pred_opt = (best_proba[:, 0] >= opt_thr).astype(int)
    test_pred_sac = (best_proba[:, 0] >= sac_thr).astype(int)
    sac_metrics = {
        "brier_score_detrator": round(brier_best, 4),
        "opt_f1_threshold_val": round(opt_thr, 2),
        "opt_f1_test_f1": round(float(f1_score(y_det_test, test_pred_opt)), 4),
        "opt_f1_test_precision": round(float(y_det_test[test_pred_opt == 1].mean()), 4),
        "opt_f1_test_recall": round(float(test_pred_opt[y_det_test == 1].mean()), 4),
        "sac_p85_threshold_val": round(sac_thr, 2),
        "sac_p85_test_precision": round(float(y_det_test[test_pred_sac == 1].mean()), 4),
        "sac_p85_test_recall": round(float(test_pred_sac[y_det_test == 1].mean()), 4),
        "sac_p85_queue_pct": round(float(test_pred_sac.mean() * 100), 1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    axes[0].plot(prob_pred, prob_true, "o-", color="#10b981", lw=2.2, label=f"Campeão (Brier = {brier_best:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibração Perfeita")
    axes[0].set_title("Curva de Calibração — Probabilidade de Detrator (1–2★)")
    axes[0].set_xlabel("Probabilidade Média Predita")
    axes[0].set_ylabel("Fração Real de Detratores")
    axes[0].legend()

    axes[1].plot(thrs, prec_l, lw=2, color="#38bdf8", label="Precisão SAC")
    axes[1].plot(thrs, rec_l, lw=2, color="#ef4444", label="Recall Detratores")
    axes[1].plot(thrs, f1_l, lw=2.2, color="#10b981", label="F1-Score Detratores")
    axes[1].axvline(opt_thr, color="#10b981", ls="--", alpha=0.8, label=f"Ótimo F1 (τ={opt_thr:.2f})")
    axes[1].axvline(sac_thr, color="#f59e0b", ls=":", lw=2, label=f"SLA Alta Precisão (τ={sac_thr:.2f})")
    axes[1].set_title("Otimização de Threshold para Triagem SAC (Conjunto Validação)")
    axes[1].set_xlabel("Threshold de Probabilidade de Detrator (τ)")
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_calibracao_threshold_sac.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- FIGURA 5: Interpretabilidade (Termos Discriminativos) & Slice-Based Error Analysis ---
    print("[6/7] Gerando interpretabilidade, slice error analysis, ABSA e Grafo NetworkX...")
    word_names = text_vec.transformer_list[0][1].get_feature_names_out()
    n_word = len(word_names)
    # Coeficientes da Regressão Logística Texto Only (classe 0 = Negativo vs classe 2 = Positivo)
    w_neg = lr_txt.coef_[0, :n_word]
    w_pos = lr_txt.coef_[2, :n_word]
    top_neg_idx = np.argsort(w_neg)[-14:][::-1]
    top_pos_idx = np.argsort(w_pos)[-14:][::-1]
    top_neg_terms = [(str(word_names[i]), round(float(w_neg[i]), 3)) for i in top_neg_idx]
    top_pos_terms = [(str(word_names[i]), round(float(w_pos[i]), 3)) for i in top_pos_idx]

    slices_df = evaluate_slices(test_df, y_test, best_pred, best_proba[:, 0])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    terms_lbl = [t for t, _ in reversed(top_neg_terms[:12])] + [t for t, _ in top_pos_terms[:12]]
    terms_val = [-v for _, v in reversed(top_neg_terms[:12])] + [v for _, v in top_pos_terms[:12]]
    t_colors = ["#ef4444"] * 12 + ["#10b981"] * 12
    axes[0].barh(np.arange(len(terms_lbl)), terms_val, color=t_colors, alpha=0.9)
    axes[0].set_yticks(np.arange(len(terms_lbl)))
    axes[0].set_yticklabels(terms_lbl, fontsize=9)
    axes[0].axvline(0, color="#334155", lw=1)
    axes[0].set_title("Interpretabilidade NLP: Top Termos Detratores (−) vs. Promotores (+)")
    axes[0].set_xlabel("Peso Linear L2 (Regressão Logística)")

    s_plot = slices_df[slices_df["Dimensão"].isin(["Status Logístico", "Tamanho do Review"])].copy()
    s_plot["label"] = s_plot["Dimensão"].str.slice(0, 4) + ": " + s_plot["Fatia"]
    y_s = np.arange(len(s_plot))
    axes[1].barh(y_s - 0.18, s_plot["Macro-F1"], height=0.35, color="#38bdf8", label="Macro-F1")
    axes[1].barh(y_s + 0.18, s_plot["Recall Detrator"], height=0.35, color="#ef4444", label="Recall Detrator")
    axes[1].set_yticks(y_s)
    axes[1].set_yticklabels(s_plot["label"], fontsize=9)
    axes[1].set_xlim(0, 1.08)
    for i_s, (_, r_s) in enumerate(s_plot.iterrows()):
        axes[1].text(r_s["Recall Detrator"] + 0.015, i_s + 0.18, f"{r_s['Recall Detrator']:.2f}", va="center", fontsize=8)
    axes[1].set_title("Slice-Based Error Analysis (Teste Out-of-Time)")
    axes[1].set_xlabel("Score")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_interpretabilidade_slices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- FIGURA 6: Mineração de Causa-Raiz (ABSA + NMF) & Grafo Categorias <-> Dores ---
    neg_df = df_txt[df_txt["sentimento_real"] == "Negativo"].copy()
    aspect_counts = {}
    for asp in list(ASPECT_PATTERNS.keys()) + ["Outros / Geral"]:
        aspect_counts[asp] = int(neg_df["aspectos"].map(lambda lst: asp in lst).sum())

    # Tópicos NMF sobre avaliações negativas
    nmf_vec = TfidfVectorizer(max_features=3000, min_df=10, max_df=0.85, ngram_range=(1, 2))
    X_neg_nmf = nmf_vec.fit_transform(neg_df["texto_limpo"])
    nmf = NMF(n_components=5, random_state=42, max_iter=300)
    W_nmf = nmf.fit_transform(X_neg_nmf)
    nmf_feats = nmf_vec.get_feature_names_out()
    nmf_topics = []
    for t_i, comp in enumerate(nmf.components_):
        top_w = [str(nmf_feats[j]) for j in np.argsort(comp)[-7:][::-1]]
        nmf_topics.append({"topic": t_i + 1, "top_terms": top_w, "share_pct": round(float((W_nmf.argmax(axis=1) == t_i).mean() * 100), 1)})

    # Matriz Categoria x Causa-Raiz (Top 8 categorias com mais reclamações)
    top_cats = neg_df["categoria_principal"].value_counts().head(8).index.tolist()
    heat_data = []
    for cat in top_cats:
        sub = neg_df[neg_df["categoria_principal"] == cat]
        row = {asp: float(sub["aspectos"].map(lambda l: asp in l).mean() * 100) for asp in ASPECT_PATTERNS.keys()}
        heat_data.append(row)
    heat_df = pd.DataFrame(heat_data, index=[c.replace("_", " ").title() for c in top_cats])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0], cbar_kws={"label": "% dos Reviews Negativos"})
    axes[0].set_title("Mapa de Causa-Raiz: Top 8 Categorias × Aspecto de Reclamação (%)")
    axes[0].tick_params(axis="x", rotation=28)

    # Grafo Bipartido NetworkX (Categorias <-> Aspectos com peso >= 10%)
    G = nx.Graph()
    for cat_lbl in heat_df.index:
        G.add_node(cat_lbl, bipartite=0, kind="cat")
    for asp_lbl in heat_df.columns:
        G.add_node(asp_lbl, bipartite=1, kind="asp")
    for cat_lbl in heat_df.index:
        for asp_lbl in heat_df.columns:
            w = float(heat_df.loc[cat_lbl, asp_lbl])
            if w >= 10.0:
                G.add_edge(cat_lbl, asp_lbl, weight=w)

    pos = {}
    cats_list = list(heat_df.index)
    asps_list = list(heat_df.columns)
    for idx_c, n in enumerate(cats_list):
        pos[n] = (0.0, 1.0 - idx_c / max(1, len(cats_list) - 1))
    for idx_a, n in enumerate(asps_list):
        pos[n] = (1.0, 0.95 - idx_a * 0.9 / max(1, len(asps_list) - 1))

    edge_w = [G[u][v]["weight"] / 8.0 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=axes[1], width=edge_w, alpha=0.55, edge_color="#f97316")
    nx.draw_networkx_nodes(G, pos, nodelist=cats_list, ax=axes[1], node_color="#38bdf8", node_size=520)
    nx.draw_networkx_nodes(G, pos, nodelist=asps_list, ax=axes[1], node_color="#ef4444", node_size=650)
    for n, (x_n, y_n) in pos.items():
        ha = "right" if x_n < 0.5 else "left"
        dx = -0.04 if x_n < 0.5 else 0.04
        axes[1].text(x_n + dx, y_n, n, ha=ha, va="center", fontsize=8.5, fontweight="bold", color="#0f172a")
    axes[1].set_xlim(-0.55, 1.55)
    axes[1].set_ylim(-0.12, 1.12)
    axes[1].axis("off")
    axes[1].set_title("Grafo de Conhecimento: Categorias ↔ Principais Causas-Raiz (≥10%)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_causa_raiz_grafo_aspectos.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- COPILOTO RAG DE ATENDIMENTO (3 EXEMPLOS REAIS DO TESTE OUT-OF-TIME) ---
    print("[7/7] Executando demonstração do Copiloto RAG de SAC em casos reais Out-of-Time...")
    rag = CustomerSupportRAG(text_vec, train_df, X_tr_txt)
    # Seleciona 1 detrator com atraso, 1 detrator com defeito/divergência, 1 promotor
    idx_late = int(test_df[(test_df["sentimento_real"] == "Negativo") & (test_df["entregue_com_atraso"] == 1) & (test_df["qtd_palavras"] >= 10)].index[0])
    idx_def = int(test_df[(test_df["sentimento_real"] == "Negativo") & (test_df["aspecto_principal"] == "Defeito / Avaria") & (test_df["qtd_palavras"] >= 8)].index[0])
    idx_pos = int(test_df[(test_df["sentimento_real"] == "Positivo") & (test_df["qtd_palavras"] >= 8)].index[0])

    rag_demos = []
    for idx_ex in (idx_late, idx_def, idx_pos):
        row = test_df.iloc[idx_ex]
        res_rag = rag.assist(
            review_text=str(row["texto_completo"]),
            order_context=row.to_dict(),
            pred_label=str(best_pred[idx_ex]),
            prob_detrator=float(best_proba[idx_ex, 0]),
            top_k=3,
        )
        rag_demos.append({
            "real_score": int(row["review_score"]),
            "real_label": str(row["sentimento_real"]),
            "texto_cliente": str(row["texto_completo"])[:220],
            "categoria": str(row["categoria_principal"]),
            "atraso_dias": round(float(row["atraso_entrega_dias"]), 1) if pd.notna(row["atraso_entrega_dias"]) else None,
            **res_rag,
        })

    summary = {
        "dataset": {
            "total_unique_orders_reviewed": int(len(df_all)),
            "written_reviews_n": int(len(df_txt)),
            "written_reviews_pct": round(len(df_txt) / len(df_all) * 100, 1),
            "parquet_size_mb": round(pq_mb, 2),
            "date_min": str(df_txt["order_purchase_timestamp"].min().date()),
            "date_max": str(df_txt["order_purchase_timestamp"].max().date()),
            "class_distribution": {k: int(v) for k, v in df_txt["sentimento_real"].value_counts().items()},
            "delay_rate_by_sentiment_pct": {
                k: round(float(v * 100), 2)
                for k, v in df_txt.groupby("sentimento_real")["entregue_com_atraso"].mean().items()
            },
            "not_delivered_rate_by_sentiment_pct": {
                k: round(float(v * 100), 2)
                for k, v in df_txt.groupby("sentimento_real")["pedido_nao_entregue"].mean().items()
            },
        },
        "splits": {
            "train_n": int(len(train_df)),
            "val_n": int(len(val_df)),
            "test_n": int(len(test_df)),
            "train_end": str(train_df["order_purchase_timestamp"].max().date()),
            "val_end": str(val_df["order_purchase_timestamp"].max().date()),
            "test_end": str(test_df["order_purchase_timestamp"].max().date()),
        },
        "negation_ablation_val": {
            "naive_stopwords_f1_macro": round(f1_naive_val, 4),
            "negation_preserved_char_tfidf_f1_macro": round(f1_neg_val, 4),
            "gain_pp": round((f1_neg_val - f1_naive_val) * 100, 2),
        },
        "champion_model": best_name,
        "models_benchmark": res_df.to_dict(orient="records"),
        "calibration_and_sac_threshold": sac_metrics,
        "top_negative_terms": top_neg_terms,
        "top_positive_terms": top_pos_terms,
        "slice_error_analysis": slices_df.to_dict(orient="records"),
        "negative_aspects_counts": aspect_counts,
        "nmf_negative_topics": nmf_topics,
        "rag_demos": rag_demos,
    }

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo salvo em: {OUT_DIR / 'metrics_summary.json'}")
    print(f"Figuras salvas em: {FIG_DIR} ({len(list(FIG_DIR.glob('*.png')))} PNGs)")


if __name__ == "__main__":
    main()
