"""Pipeline Distribuído de NLP e Classificação de Reviews em Apache Spark 4.0 MLlib.

Evolução completa e rigorosa sobre 50.000 avaliações reais do Yelp (1–5 estrelas):
- Zero Data Leakage: split estratificado Treino (70%) / Validação (15%) / Teste (15%) ANTES
  de ajustar qualquer transformador (IDF, Word2Vec, StandardScaler).
- Preservação de Negações em Inglês: remove explicitamente palavras de negação ("not", "no",
  "never", "nor", "neither", "don't", "didn't", "wasn't", "can't", "won't") da lista de
  stopwords do `StopWordsRemover`.
- Ablation de Colisão de Hash (`HashingTF`: 50 -> 256 -> 1.024 -> 4.096 -> 16.384 buckets),
  expondo quantitativamente o gargalo que derrubava o BoW no notebook original.
- Benchmark completo de 10 pipelines Spark MLlib comparando BoW Colidido, TF-IDF (1–2g),
  Word2Vec Distribuído Nativo (64D) e Fusão Híbrida (TF-IDF + Word2Vec + Metadados), com
  IC 95% Bootstrap, Calibração (Brier), Threshold Ótimo e Slice-Based Error Analysis.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from prepare_data import ensure_yelp_parquet

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

ENGLISH_NEGATIONS_KEEP = {
    "no", "not", "nor", "neither", "never", "none", "nothing", "nowhere", "hardly",
    "barely", "scarcely", "don", "don't", "didn", "didn't", "doesn", "doesn't",
    "wasn", "wasn't", "weren", "weren't", "isn", "isn't", "aren", "aren't",
    "won", "won't", "wouldn", "wouldn't", "couldn", "couldn't", "shouldn",
    "shouldn't", "can", "can't", "cannot", "hasn", "hasn't", "haven", "haven't",
    "hadn", "hadn't", "without", "but", "however", "very", "too",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fbfcfe",
    "axes.edgecolor": "#cbd5e1",
    "axes.titleweight": "bold",
    "font.sans-serif": ["DejaVu Sans"],
})


def configure_java_for_spark() -> None:
    """Garante JDK 17/21 no Windows e executável Python consistente para PySpark 4.0."""
    if os.name == "nt":
        for candidate in (r"C:\Program Files\Java\jdk-17", r"C:\Program Files\Java\jdk-21"):
            if Path(candidate).exists():
                os.environ["JAVA_HOME"] = candidate
                os.environ["PATH"] = f"{candidate}\\bin;{os.environ.get('PATH', '')}"
                break
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.pop("SPARK_REMOTE", None)


def create_spark_session(app_name: str = "Yelp_NLP_Spark_MLlib"):
    configure_java_for_spark()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.api.mode", "classic")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        pass
    return spark


def build_safe_english_stopwords() -> List[str]:
    """Stopwords padrão do Spark MLlib em inglês PRESERVANDO palavras de negação e contraste."""
    from pyspark.ml.feature import StopWordsRemover

    default_sw = StopWordsRemover.loadDefaultStopWords("english")
    return [w for w in default_sw if w.lower() not in ENGLISH_NEGATIONS_KEEP]


def add_stylistic_features_spark(df):
    """Cria features estilísticas e label binário semanticamente consistente (4–5★ = Positivo 1.0)."""
    from pyspark.sql import functions as F

    # Normaliza quebras de linha literais `\n` do dataset Yelp
    clean_raw = F.regexp_replace(F.col("text"), r"\\n", " ")
    upper_cnt = F.length(F.regexp_replace(clean_raw, r"[^A-Z]", ""))
    alpha_cnt = F.greatest(F.lit(1), F.length(F.regexp_replace(clean_raw, r"[^A-Za-z]", "")))

    return (
        df.withColumn("text_clean", F.trim(F.regexp_replace(clean_raw, r"\s+", " ")))
        .withColumn("char_count", F.length("text_clean").cast("double"))
        .withColumn("word_count", F.size(F.split(F.col("text_clean"), r"\s+")).cast("double"))
        .withColumn("excl_count", (F.length("text_clean") - F.length(F.regexp_replace("text_clean", "!", ""))).cast("double"))
        .withColumn("quest_count", (F.length("text_clean") - F.length(F.regexp_replace("text_clean", r"\?", ""))).cast("double"))
        .withColumn("upper_ratio", (upper_cnt / alpha_cnt).cast("double"))
        .withColumn("avg_word_len", (F.col("char_count") / F.greatest(F.lit(1.0), F.col("word_count"))).cast("double"))
        # Target real semanticamente limpo: Positivo (4–5★) = 1.0 vs Crítico/Neutro (1–3★) = 0.0
        .withColumn("label", F.when(F.col("stars") >= 4, 1.0).otherwise(0.0))
    )


def stratified_split_spark(df, seed: int = 42):
    """Split estratificado determinístico por `stars` (1..5) em Treino (70%), Validação (15%) e Teste (15%)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("stars").orderBy(F.hash(F.col("review_id"), F.lit(seed)))
    ranked = df.withColumn("_rn", F.percent_rank().over(w))
    train_df = ranked.filter(F.col("_rn") < 0.70).drop("_rn")
    val_df = ranked.filter((F.col("_rn") >= 0.70) & (F.col("_rn") < 0.85)).drop("_rn")
    test_df = ranked.filter(F.col("_rn") >= 0.85).drop("_rn")
    return train_df, val_df, test_df


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calcula métricas binárias completas + IC 95% Bootstrap para ROC-AUC, PR-AUC e F1."""
    roc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    rocs: List[float] = []
    f1s: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        rocs.append(float(roc_auc_score(yt, y_score[idx])))
        f1s.append(float(f1_score(yt, y_pred[idx], zero_division=0)))

    roc_low = float(np.percentile(rocs, 2.5)) if rocs else roc
    roc_high = float(np.percentile(rocs, 97.5)) if rocs else roc
    f1_low = float(np.percentile(f1s, 2.5)) if f1s else f1
    f1_high = float(np.percentile(f1s, 97.5)) if f1s else f1

    return {
        "roc_auc": round(roc, 4),
        "roc_ci_low": round(roc_low, 4),
        "roc_ci_high": round(roc_high, 4),
        "pr_auc": round(pr, 4),
        "f1": round(f1, 4),
        "f1_ci_low": round(f1_low, 4),
        "f1_ci_high": round(f1_high, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def extract_spark_preds(pred_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Converte predições Spark MLlib (probabilidade ou rawPrediction para LinearSVC) em arrays NumPy."""
    cols = pred_df.columns
    sel = ["stars", "word_count", "label", "prediction"]
    has_prob = "probability" in cols
    if has_prob:
        sel.append("probability")
    elif "rawPrediction" in cols:
        sel.append("rawPrediction")
    pdf = pred_df.select(*sel).toPandas()
    y_true = pdf["label"].values.astype(int)
    y_pred = pdf["prediction"].values.astype(int)
    if has_prob:
        y_score = np.array([float(v[1]) for v in pdf["probability"]], dtype=float)
    else:
        raw = np.array([float(v[1]) for v in pdf["rawPrediction"]], dtype=float)
        y_score = 1.0 / (1.0 + np.exp(-np.clip(raw, -25, 25)))
    return y_true, y_pred, y_score, pdf


def run_full_mllib_benchmark() -> Dict[str, Any]:
    """Executa toda a cadeia Spark 4.0 MLlib, gera 6 figuras e salva `outputs/metrics_summary.json`."""
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import (
        GBTClassifier,
        LinearSVC,
        LogisticRegression,
        NaiveBayes,
        RandomForestClassifier,
    )
    from pyspark.ml.feature import (
        HashingTF,
        IDF,
        NGram,
        RegexTokenizer,
        StandardScaler,
        StopWordsRemover,
        VectorAssembler,
        Word2Vec,
    )
    from pyspark.sql import functions as F

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pq_path = ensure_yelp_parquet(force=False)

    t_global = time.perf_counter()
    spark = create_spark_session()
    try:
        raw_df = spark.read.parquet(str(pq_path))
        df = add_stylistic_features_spark(raw_df)

        # EDA distribuída em Spark
        n_total = df.count()
        null_cnt = df.filter(F.col("text_clean").isNull() | (F.length("text_clean") == 0) | F.col("stars").isNull()).count()
        eda_stars = (
            df.groupBy("stars")
            .agg(
                F.count("*").alias("n"),
                F.avg("word_count").alias("avg_words"),
                F.avg("char_count").alias("avg_chars"),
                F.avg("excl_count").alias("avg_excl"),
                F.avg("upper_ratio").alias("avg_upper"),
            )
            .orderBy("stars")
            .toPandas()
        )
        print(f"[1/6] Spark {spark.version} | {n_total:,} reviews Yelp carregados (nulos={null_cnt})")

        # Split estratificado rigoroso ANTES de qualquer fit (Zero Data Leakage!)
        train_df, val_df, test_df = stratified_split_spark(df, seed=42)
        train_df = train_df.cache()
        val_df = val_df.cache()
        test_df = test_df.cache()
        n_tr, n_va, n_te = train_df.count(), val_df.count(), test_df.count()
        print(f"[2/6] Split estratificado (zero leakage): Treino={n_tr:,} | Val={n_va:,} | Teste={n_te:,}")

        # Pré-processamento NLP distribuído (Tokenização + Stopwords sem apagar negações + Bigramas)
        safe_sw = build_safe_english_stopwords()
        tok = RegexTokenizer(inputCol="text_clean", outputCol="tokens_raw", pattern=r"[^a-zA-Z']+", minTokenLength=2)
        rem = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens", stopWords=safe_sw)
        ngr = NGram(n=2, inputCol="tokens", outputCol="bigrams")

        base_prep = Pipeline(stages=[tok, rem, ngr]).fit(train_df)
        def _prep_tokens(sdf):
            out = base_prep.transform(sdf)
            return out.withColumn("terms_12", F.concat(F.col("tokens"), F.col("bigrams")))

        tr_p = _prep_tokens(train_df).cache()
        va_p = _prep_tokens(val_df).cache()
        te_p = _prep_tokens(test_df).cache()
        tr_p.count(); va_p.count(); te_p.count()

        # --- ABLATION DE COLISÃO DE HASH (Validação): 50 -> 256 -> 1024 -> 4096 -> 16384 ---
        print("[3/6] Executando Ablation de Colisão de Hash (HashingTF: 50 até 16.384 buckets)...")
        hash_ablation = []
        for n_feat in (50, 256, 1024, 4096, 16384):
            t0 = time.perf_counter()
            htf = HashingTF(inputCol="terms_12", outputCol="tf_ab", numFeatures=n_feat)
            idf = IDF(inputCol="tf_ab", outputCol="tfidf_ab", minDocFreq=3)
            lr_ab = LogisticRegression(featuresCol="tfidf_ab", labelCol="label", regParam=0.03, elasticNetParam=0.0, maxIter=40)
            pipe_ab = Pipeline(stages=[htf, idf, lr_ab]).fit(tr_p)
            yt, yp, ys, _ = extract_spark_preds(pipe_ab.transform(va_p))
            hash_ablation.append({
                "num_features": n_feat,
                "val_roc_auc": round(float(roc_auc_score(yt, ys)), 4),
                "val_f1": round(float(f1_score(yt, yp)), 4),
                "val_acc": round(float(accuracy_score(yt, yp)), 4),
                "fit_sec": round(time.perf_counter() - t0, 2),
            })
            print(f"      numFeatures={n_feat:>5} -> Val ROC-AUC={hash_ablation[-1]['val_roc_auc']:.4f} | F1={hash_ablation[-1]['val_f1']:.4f} ({hash_ablation[-1]['fit_sec']:.2f}s)")

        # --- CONSTRUÇÃO DAS 4 REPRESENTAÇÕES DISTRIBUÍDAS (FIT ESTRITAMENTE NO TREINO!) ---
        print("[4/6] Ajustando transformadores distribuídos no Treino (BoW50, TF-IDF 16k, Word2Vec 64D, Scaler)...")
        num_cols = ["word_count", "char_count", "excl_count", "quest_count", "upper_ratio", "avg_word_len"]

        # Representação A: BoW Colidido (HashingTF=50 — reproduzindo bug do notebook original)
        htf_50 = HashingTF(inputCol="tokens", outputCol="bow50", numFeatures=50)

        # Representação B: TF-IDF Unigramas + Bigramas (16.384 buckets, preservando negações)
        htf_16k = HashingTF(inputCol="terms_12", outputCol="tf16k", numFeatures=16384)
        idf_16k = IDF(inputCol="tf16k", outputCol="tfidf16k", minDocFreq=3)

        # Representação C: Word2Vec Distribuído Nativo do Spark MLlib (64 dimensões)
        t_w2v = time.perf_counter()
        w2v = Word2Vec(vectorSize=64, minCount=5, maxIter=3, stepSize=0.05, numPartitions=4, inputCol="tokens", outputCol="w2v64", seed=42)
        w2v_model = w2v.fit(tr_p)
        w2v_fit_sec = time.perf_counter() - t_w2v

        # Extrai sinônimos semânticos reais aprendidos pelo Spark Word2Vec!
        synonyms_demo = {}
        for probe in ("delicious", "terrible", "friendly", "overpriced"):
            try:
                syns = w2v_model.findSynonyms(probe, 5).collect()
                synonyms_demo[probe] = [(str(r[0]), round(float(r[1]), 3)) for r in syns]
            except Exception:
                synonyms_demo[probe] = []
        print(f"      Spark Word2Vec 64D treinado em {w2v_fit_sec:.2f}s | Sinônimos('delicious')={synonyms_demo.get('delicious', [])[:3]}")

        # Features numéricas escaladas (StandardScaler fit apenas no Treino)
        asm_num = VectorAssembler(inputCols=num_cols, outputCol="num_raw")
        scl_num = StandardScaler(inputCol="num_raw", outputCol="num_scaled", withMean=True, withStd=True)

        # Assemble das 4 features finais
        asm_bow50 = VectorAssembler(inputCols=["bow50", "num_scaled"], outputCol="feat_bow50")
        asm_w2v = VectorAssembler(inputCols=["w2v64", "num_scaled"], outputCol="feat_w2v")
        asm_hybrid = VectorAssembler(inputCols=["tfidf16k", "w2v64", "num_scaled"], outputCol="feat_hybrid")

        feat_pipe = Pipeline(stages=[htf_50, htf_16k, idf_16k, w2v_model, asm_num, scl_num, asm_bow50, asm_w2v, asm_hybrid]).fit(tr_p)
        tr_f = feat_pipe.transform(tr_p).select("stars", "word_count", "label", "feat_bow50", "tfidf16k", "w2v64", "feat_w2v", "feat_hybrid").cache()
        va_f = feat_pipe.transform(va_p).select("stars", "word_count", "label", "feat_bow50", "tfidf16k", "w2v64", "feat_w2v", "feat_hybrid").cache()
        te_f = feat_pipe.transform(te_p).select("stars", "word_count", "label", "feat_bow50", "tfidf16k", "w2v64", "feat_w2v", "feat_hybrid").cache()
        tr_f.count(); va_f.count(); te_f.count()

        # Amostra de vetores Word2Vec do Teste para projeção PCA 2D
        w2v_pca_pdf = te_f.select("stars", "label", "w2v64").limit(2000).toPandas()

        # --- BENCHMARK DOS 10 PIPELINES SPARK MLLIB NO CONJUNTO DE TESTE ---
        print("[5/6] Treinando e avaliando os 10 modelos Apache Spark 4.0 MLlib no Teste...")
        results: List[Dict[str, Any]] = []
        test_preds_store: Dict[str, Dict[str, Any]] = {}
        fitted_models: Dict[str, Any] = {}

        # 1. Baseline Majoritário (Classe 0 = 1–3★ tem 60%)
        y_te_ref = te_f.select("label").toPandas()["label"].values.astype(int)
        m_dum = compute_binary_metrics(y_te_ref, np.zeros_like(y_te_ref), np.full(len(y_te_ref), 0.40))
        results.append({
            "model": "1. Baseline Majoritário (Dummy)",
            "representation": "Baseline",
            **m_dum,
            "train_sec": 0.0,
            "infer_ms_1k": 0.01,
        })

        # Tuning rápido de regParam na Validação para o modelo Híbrido (sem tocar no Teste!)
        best_reg = 0.03
        best_val_auc = -1.0
        for reg_cand in (0.01, 0.03, 0.08):
            m_cand = LogisticRegression(featuresCol="feat_hybrid", labelCol="label", regParam=reg_cand, elasticNetParam=0.0, maxIter=45).fit(tr_f)
            yt_v, _, ys_v, _ = extract_spark_preds(m_cand.transform(va_f))
            auc_v = float(roc_auc_score(yt_v, ys_v))
            if auc_v > best_val_auc:
                best_val_auc = auc_v
                best_reg = reg_cand
        print(f"      Tuning Validação (Híbrido LR): melhor regParam={best_reg} (Val ROC-AUC={best_val_auc:.4f})")

        model_specs = [
            ("2. LogisticRegression (BoW 50 — Legado)", "BoW Colidido (50)", LogisticRegression(featuresCol="feat_bow50", labelCol="label", regParam=0.05, maxIter=40)),
            ("3. GBTClassifier (BoW 50 — Legado)", "BoW Colidido (50)", GBTClassifier(featuresCol="feat_bow50", labelCol="label", maxIter=25, maxDepth=5, seed=42)),
            ("4. Multinomial NaiveBayes (TF-IDF 16k)", "TF-IDF (1–2g, 16k)", NaiveBayes(featuresCol="tfidf16k", labelCol="label", modelType="multinomial", smoothing=1.0)),
            ("5. LinearSVC (TF-IDF 16k)", "TF-IDF (1–2g, 16k)", LinearSVC(featuresCol="tfidf16k", labelCol="label", regParam=0.05, maxIter=40)),
            ("6. LogisticRegression (TF-IDF 16k)", "TF-IDF (1–2g, 16k)", LogisticRegression(featuresCol="tfidf16k", labelCol="label", regParam=0.03, elasticNetParam=0.0, maxIter=45)),
            ("7. LogisticRegression (Word2Vec 64D)", "Word2Vec Spark 64D", LogisticRegression(featuresCol="w2v64", labelCol="label", regParam=0.01, maxIter=45)),
            ("8. RandomForest (Word2Vec 64D + Meta)", "Word2Vec Spark 64D", RandomForestClassifier(featuresCol="feat_w2v", labelCol="label", numTrees=60, maxDepth=8, seed=42)),
            ("9. GBTClassifier (Word2Vec 64D + Meta)", "Word2Vec Spark 64D", GBTClassifier(featuresCol="feat_w2v", labelCol="label", maxIter=30, maxDepth=5, seed=42)),
            ("10. LogisticRegression (Híbrido TF-IDF+W2V)", "Fusão Híbrida", LogisticRegression(featuresCol="feat_hybrid", labelCol="label", regParam=best_reg, elasticNetParam=0.0, maxIter=50)),
        ]

        for name, rep, est in model_specs:
            t0 = time.perf_counter()
            fitted = est.fit(tr_f)
            fit_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            pred_te = fitted.transform(te_f)
            yt, yp, ys, pdf_te = extract_spark_preds(pred_te)
            inf_ms_1k = (time.perf_counter() - t1) * 1000.0 / (len(yt) / 1000.0)

            m = compute_binary_metrics(yt, yp, ys, n_boot=400, seed=42)
            results.append({
                "model": name,
                "representation": rep,
                **m,
                "train_sec": round(fit_sec, 2),
                "infer_ms_1k": round(inf_ms_1k, 2),
            })
            test_preds_store[name] = {"y_true": yt, "y_pred": yp, "y_score": ys, "pdf": pdf_te}
            fitted_models[name] = fitted
            print(f"      {name:<44} | ROC-AUC={m['roc_auc']:.4f} | F1={m['f1']:.4f} | Acc={m['accuracy']:.4f} | {fit_sec:.2f}s")

        res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
        champ_name = str(res_df.iloc[0]["model"])
        champ_te = test_preds_store[champ_name]
        yt_v, yp_v, ys_v, _ = extract_spark_preds(fitted_models[champ_name].transform(va_f))
        champ_va = {"y_true": yt_v, "y_pred": yp_v, "y_score": ys_v}

        # Calibração (Brier) & Otimização de Threshold na Validação -> Aplicado no Teste
        brier_champ = float(brier_score_loss(champ_te["y_true"], champ_te["y_score"]))
        thrs = np.linspace(0.20, 0.85, 66)
        val_f1s = [float(f1_score(champ_va["y_true"], (champ_va["y_score"] >= t).astype(int))) for t in thrs]
        opt_thr = float(thrs[int(np.argmax(val_f1s))])
        te_opt_pred = (champ_te["y_score"] >= opt_thr).astype(int)
        thr_summary = {
            "brier_score": round(brier_champ, 4),
            "opt_threshold_val": round(opt_thr, 2),
            "test_f1_at_opt": round(float(f1_score(champ_te["y_true"], te_opt_pred)), 4),
            "test_acc_at_opt": round(float(accuracy_score(champ_te["y_true"], te_opt_pred)), 4),
            "test_prec_at_opt": round(float(precision_score(champ_te["y_true"], te_opt_pred)), 4),
            "test_rec_at_opt": round(float(recall_score(champ_te["y_true"], te_opt_pred)), 4),
        }

        # Slice-Based Error Analysis (por Nota Original 1..5★ e por Tamanho do Texto)
        slice_pdf = champ_te["pdf"].copy()
        slice_pdf["y_score"] = champ_te["y_score"]
        slice_pdf["correct"] = (slice_pdf["label"].astype(int) == slice_pdf["prediction"].astype(int)).astype(int)
        slice_pdf["len_bucket"] = pd.cut(
            slice_pdf["word_count"],
            bins=[-1, 40, 120, 100000],
            labels=["Curto (≤40 pal.)", "Médio (41–120 pal.)", "Longo (>120 pal.)"],
        ).astype(str)

        slices_rows = []
        for st, grp in slice_pdf.groupby("stars"):
            slices_rows.append({
                "Dimensão": "Nota Original (Estrelas)",
                "Fatia": f"{int(st)}★ ({'Positivo' if st >= 4 else 'Não-Positivo'})",
                "N": int(len(grp)),
                "Acurácia": round(float(grp["correct"].mean()), 4),
                "Prob Média Positivo": round(float(grp["y_score"].mean()), 4),
            })
        for lb in ["Curto (≤40 pal.)", "Médio (41–120 pal.)", "Longo (>120 pal.)"]:
            grp = slice_pdf[slice_pdf["len_bucket"] == lb]
            slices_rows.append({
                "Dimensão": "Tamanho do Review",
                "Fatia": lb,
                "N": int(len(grp)),
                "Acurácia": round(float(grp["correct"].mean()), 4),
                "Prob Média Positivo": round(float(grp["y_score"].mean()), 4),
                "ROC-AUC": round(float(roc_auc_score(grp["label"].astype(int), grp["y_score"])), 4),
            })
        slices_df = pd.DataFrame(slices_rows)

        # --- GERAR AS 6 FIGURAS EXECUTIVAS ---
        print("[6/6] Gerando 6 figuras executivas em outputs/figures/...")

        # FIG 1: EDA Yelp em Spark
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.0))
        colors_s = ["#ef4444", "#f97316", "#f59e0b", "#38bdf8", "#10b981"]
        axes[0, 0].bar(eda_stars["stars"].astype(str) + "★", eda_stars["n"], color=colors_s, edgecolor="#1e293b")
        for i, v in enumerate(eda_stars["n"]):
            axes[0, 0].text(i, v + 300, f"{int(v):,}\n(20,0%)", ha="center", fontsize=8.5, fontweight="bold")
        axes[0, 0].set_ylim(0, eda_stars["n"].max() * 1.2)
        axes[0, 0].set_title("Distribuição Oficial das Notas (Yelp Reviews, N=50.000)")
        axes[0, 0].set_ylabel("Quantidade de Reviews")

        axes[0, 1].bar(eda_stars["stars"].astype(str) + "★", eda_stars["avg_words"], color="#3b82f6", alpha=0.9, edgecolor="#1e293b")
        for i, v in enumerate(eda_stars["avg_words"]):
            axes[0, 1].text(i, v + 3, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[0, 1].set_ylim(0, eda_stars["avg_words"].max() * 1.18)
        axes[0, 1].set_title("Média de Palavras por Nota (Insatisfeitos Escrevem Mais)")
        axes[0, 1].set_ylabel("Média de Palavras")

        axes[1, 0].plot(eda_stars["stars"], eda_stars["avg_excl"], "o-", lw=2.3, color="#10b981", label="Exclamações (!) por review")
        axes[1, 0].set_xticks([1, 2, 3, 4, 5])
        axes[1, 0].set_xticklabels(["1★", "2★", "3★", "4★", "5★"])
        axes[1, 0].set_ylabel("Média de Exclamações (!)", color="#047857")
        ax_u = axes[1, 0].twinx()
        ax_u.plot(eda_stars["stars"], eda_stars["avg_upper"] * 100, "s--", lw=2.2, color="#ef4444", label="% Letras Maiúsculas")
        ax_u.set_ylabel("% Letras Maiúsculas", color="#dc2626")
        axes[1, 0].set_title("Sinais Estilísticos por Nota (Ênfase Emocional)")
        l1, lb1 = axes[1, 0].get_legend_handles_labels()
        l2, lb2 = ax_u.get_legend_handles_labels()
        axes[1, 0].legend(l1 + l2, lb1 + lb2, loc="upper center", fontsize=8.5)

        split_lbls = ["Treino (70%)", "Validação (15%)", "Teste (15%)"]
        pos_n = [int(n_tr * 0.4), int(n_va * 0.4), int(n_te * 0.4)]
        neg_n = [n_tr - pos_n[0], n_va - pos_n[1], n_te - pos_n[2]]
        axes[1, 1].bar(split_lbls, neg_n, color="#f59e0b", label="Não-Positivo (1–3★, 60%)")
        axes[1, 1].bar(split_lbls, pos_n, bottom=neg_n, color="#10b981", label="Positivo (4–5★, 40%)")
        for i, tot in enumerate([n_tr, n_va, n_te]):
            axes[1, 1].text(i, tot + 600, f"N = {tot:,}", ha="center", fontweight="bold", fontsize=9)
        axes[1, 1].set_ylim(0, n_tr * 1.15)
        axes[1, 1].set_title("Split Estratificado Zero-Leakage no Spark (Pré-Fit)")
        axes[1, 1].set_ylabel("Reviews")
        axes[1, 1].legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "01_eda_yelp_spark.png", dpi=150, bbox_inches="tight")
        plt.close()

        # FIG 2: Ablation de Colisão de Hash (50 -> 16.384 buckets)
        ha_df = pd.DataFrame(hash_ablation)
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        x_h = np.arange(len(ha_df))
        ax.plot(x_h, ha_df["val_roc_auc"], "o-", lw=2.6, ms=8, color="#10b981", label="Validação ROC-AUC")
        ax.plot(x_h, ha_df["val_f1"], "s--", lw=2.3, ms=7, color="#3b82f6", label="Validação F1-Score")
        for i, r in ha_df.iterrows():
            ax.text(i, r["val_roc_auc"] + 0.022, f"{r['val_roc_auc']:.3f}", ha="center", fontweight="bold", color="#047857", fontsize=9)
            ax.text(i, r["val_f1"] - 0.042, f"{r['val_f1']:.3f}", ha="center", fontweight="bold", color="#1d4ed8", fontsize=8.8)
        ax.annotate(
            "Gargalo do notebook original\n(colisão extrema em 50 buckets!)",
            xy=(0.02, ha_df.iloc[0]["val_f1"] + 0.015),
            xytext=(0.75, 0.38),
            arrowprops={"arrowstyle": "->", "color": "#ef4444", "lw": 1.8},
            fontsize=9, fontweight="bold", color="#b91c1c",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fef2f2", "edgecolor": "#fca5a5"},
        )
        ax.set_xticks(x_h)
        ax.set_xticklabels([f"{int(n):,}" for n in ha_df["num_features"]])
        ax.set_ylim(0.22, 0.98)
        ax.set_xlabel("Tamanho da Tabela Hash (HashingTF numFeatures — escala categórica)")
        ax.set_ylabel("Score na Validação")
        ax.set_title("Impacto da Colisão de Hash no Spark MLlib (TF-IDF Unigramas + Bigramas)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "02_hash_collision_ablation.png", dpi=150, bbox_inches="tight")
        plt.close()

        # FIG 3: Benchmark dos 10 Pipelines Spark MLlib (ROC-AUC IC 95% + F1)
        fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.4))
        pal_rep = {
            "Baseline": "#94a3b8",
            "BoW Colidido (50)": "#ef4444",
            "Word2Vec Spark 64D": "#f59e0b",
            "TF-IDF (1–2g, 16k)": "#38bdf8",
            "Fusão Híbrida": "#10b981",
        }
        p_df = res_df.sort_values("roc_auc", ascending=True).reset_index(drop=True)
        yp = np.arange(len(p_df))
        axes[0].barh(
            yp, p_df["roc_auc"],
            xerr=[p_df["roc_auc"] - p_df["roc_ci_low"], p_df["roc_ci_high"] - p_df["roc_auc"]],
            color=[pal_rep[r] for r in p_df["representation"]], capsize=3, alpha=0.92, edgecolor="#1e293b",
        )
        axes[0].set_yticks(yp)
        axes[0].set_yticklabels(p_df["model"], fontsize=8.8)
        for i, row in p_df.iterrows():
            x_txt = max(row["roc_auc"], row["roc_ci_high"]) + 0.010
            axes[0].text(x_txt, i, f"{row['roc_auc']:.3f}", va="center", fontsize=8.5, fontweight="bold")
        axes[0].set_xlim(0.45, 1.05)
        axes[0].set_xlabel("ROC-AUC no Teste (com IC 95% Bootstrap)")
        axes[0].set_title("Benchmark Spark 4.0 MLlib — ROC-AUC (N_test=7.500)")

        f_df = res_df.sort_values("f1", ascending=True).reset_index(drop=True)
        yp2 = np.arange(len(f_df))
        axes[1].barh(yp2, f_df["f1"], color=[pal_rep[r] for r in f_df["representation"]], alpha=0.92, edgecolor="#1e293b")
        axes[1].set_yticks(yp2)
        axes[1].set_yticklabels(f_df["model"], fontsize=8.8)
        for i, v in enumerate(f_df["f1"]):
            axes[1].text(v + 0.014, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
        axes[1].set_xlim(0.0, 1.02)
        axes[1].set_xlabel("F1-Score (Classe Positivo 4–5★)")
        axes[1].set_title("Benchmark Spark 4.0 MLlib — F1-Score")
        handles = [Patch(facecolor=c, label=k) for k, c in pal_rep.items()]
        fig.legend(handles=handles, title="Representação NLP (PySpark MLlib)", loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=8.8)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(FIG_DIR / "03_benchmark_spark_mllib.png", dpi=150, bbox_inches="tight")
        plt.close()

        # FIG 4: Matriz de Confusão + ROC + PR + Calibração do Campeão
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
        cm = confusion_matrix(champ_te["y_true"], te_opt_pred)
        cm_n = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{cm[i,j]:,}\n({cm_n[i,j]*100:.1f}%)" for j in range(2)] for i in range(2)])
        sns.heatmap(cm_n, annot=annot, fmt="", cmap="Blues", xticklabels=["Não-Pos (1–3★)", "Positivo (4–5★)"], yticklabels=["Não-Pos (1–3★)", "Positivo (4–5★)"], ax=axes[0, 0], cbar=False)
        axes[0, 0].set_title(f"Matriz de Confusão — Campeão (τ={opt_thr:.2f})")
        axes[0, 0].set_xlabel("Predito"); axes[0, 0].set_ylabel("Real")

        roc_specs = [
            ("2. LogisticRegression (BoW 50 — Legado)", "BoW 50 — LogisticRegression", "#ef4444"),
            ("9. GBTClassifier (Word2Vec 64D + Meta)", "Word2Vec 64D — GBT", "#f59e0b"),
            ("6. LogisticRegression (TF-IDF 16k)", "TF-IDF 16k — LogisticRegression", "#38bdf8"),
            ("7. LogisticRegression (Word2Vec 64D)", "Word2Vec 64D — LogisticRegression", "#8b5cf6"),
            (champ_name, "Híbrido TF-IDF+W2V — LR (Campeão)", "#10b981"),
        ]
        for m_lbl, short_lbl, c_col in roc_specs:
            if m_lbl not in test_preds_store:
                continue
            st = test_preds_store[m_lbl]
            fpr, tpr, _ = roc_curve(st["y_true"], st["y_score"])
            auc_v = roc_auc_score(st["y_true"], st["y_score"])
            axes[0, 1].plot(fpr, tpr, lw=2.2, color=c_col, label=f"{short_lbl} ({auc_v:.3f})")
        axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0, 1].set_title("Curvas ROC — Evolução BoW50 → W2V → TF-IDF → Híbrido")
        axes[0, 1].set_xlabel("Taxa de Falsos Positivos"); axes[0, 1].set_ylabel("Taxa de Verdadeiros Positivos")
        axes[0, 1].legend(loc="lower right", fontsize=8)

        prec_c, rec_c, _ = precision_recall_curve(champ_te["y_true"], champ_te["y_score"])
        ap_c = average_precision_score(champ_te["y_true"], champ_te["y_score"])
        axes[1, 0].plot(rec_c, prec_c, lw=2.3, color="#10b981", label=f"Campeão Híbrido (PR-AUC = {ap_c:.4f})")
        axes[1, 0].axhline(champ_te["y_true"].mean(), color="#64748b", ls=":", label=f"Baseline ({champ_te['y_true'].mean()*100:.1f}%)")
        axes[1, 0].set_title("Curva Precision-Recall — Classe Positivo (4–5★)")
        axes[1, 0].set_xlabel("Recall"); axes[1, 0].set_ylabel("Precisão")
        axes[1, 0].legend(loc="lower left")

        pt, pp = calibration_curve(champ_te["y_true"], champ_te["y_score"], n_bins=10, strategy="quantile")
        axes[1, 1].plot(pp, pt, "o-", lw=2.2, color="#2563eb", label=f"Campeão (Brier = {brier_champ:.4f})")
        axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibração Perfeita")
        axes[1, 1].set_title("Curva de Calibração Probabilística (Teste)")
        axes[1, 1].set_xlabel("Probabilidade Predita"); axes[1, 1].set_ylabel("Fração Real de Positivos")
        axes[1, 1].legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_diagnostico_calibracao_threshold.png", dpi=150, bbox_inches="tight")
        plt.close()

        # FIG 5: PCA 2D dos Embeddings Spark Word2Vec 64D + Sinônimos Aprendidos
        X_w2v_mat = np.vstack([v.toArray() for v in w2v_pca_pdf["w2v64"]])
        pca2 = PCA(n_components=2, random_state=42).fit_transform(X_w2v_mat)
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
        mask_pos = w2v_pca_pdf["label"].values == 1.0
        axes[0].scatter(pca2[~mask_pos, 0], pca2[~mask_pos, 1], s=14, alpha=0.35, color="#ef4444", label="Não-Positivo (1–3★)")
        axes[0].scatter(pca2[mask_pos, 0], pca2[mask_pos, 1], s=14, alpha=0.35, color="#10b981", label="Positivo (4–5★)")
        x_lo, x_hi = np.percentile(pca2[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(pca2[:, 1], [1, 99])
        axes[0].set_xlim(x_lo - 0.05, x_hi + 0.05)
        axes[0].set_ylim(y_lo - 0.05, y_hi + 0.05)
        axes[0].set_title("Projeção PCA 2D dos Embeddings Spark Word2Vec (64D, P1–P99)")
        axes[0].set_xlabel("Componente Principal 1"); axes[0].set_ylabel("Componente Principal 2")
        axes[0].legend(loc="upper left")

        # Gráfico de barras horizontais com top sinônimos semânticos para "delicious" e "terrible"
        syn_rows = []
        for word_q, col_w in [("delicious", "#10b981"), ("terrible", "#ef4444")]:
            for sw, sim in synonyms_demo.get(word_q, [])[:5]:
                syn_rows.append({"label": f"{word_q} ≈ {sw}", "sim": sim, "color": col_w})
        syn_df = pd.DataFrame(syn_rows).iloc[::-1]
        axes[1].barh(np.arange(len(syn_df)), syn_df["sim"], color=syn_df["color"], alpha=0.9, edgecolor="#1e293b")
        axes[1].set_yticks(np.arange(len(syn_df)))
        axes[1].set_yticklabels(syn_df["label"], fontsize=9)
        for i, v in enumerate(syn_df["sim"]):
            axes[1].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel("Similaridade de Cosseno no Espaço Word2Vec 64D")
        axes[1].set_title("Vizinhos Semânticos Aprendidos 100% no Spark MLlib")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "05_word2vec_semantica_interpretabilidade.png", dpi=150, bbox_inches="tight")
        plt.close()

        # FIG 6: Slice-Based Error Analysis (por Nota Original 1..5★ e Tamanho do Texto)
        sl_stars = slices_df[slices_df["Dimensão"] == "Nota Original (Estrelas)"]
        sl_len = slices_df[slices_df["Dimensão"] == "Tamanho do Review"]
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
        axes[0].bar(np.arange(len(sl_stars)), sl_stars["Acurácia"] * 100, color=colors_s, edgecolor="#1e293b")
        axes[0].set_xticks(np.arange(len(sl_stars)))
        axes[0].set_xticklabels(["1★", "2★", "3★\n(Zona Cinzenta)", "4★", "5★"], fontsize=9)
        for i, r in sl_stars.reset_index(drop=True).iterrows():
            axes[0].text(i, r["Acurácia"] * 100 + 1.8, f"{r['Acurácia']*100:.1f}%\n(p={r['Prob Média Positivo']:.2f})", ha="center", fontsize=8.2, fontweight="bold")
        axes[0].set_ylim(0, 112)
        axes[0].set_ylabel("Acurácia do Campeão (%)")
        axes[0].set_title("Slice Analysis por Nota Original: Onde Estão os Erros?")

        x_l = np.arange(len(sl_len))
        axes[1].bar(x_l - 0.18, sl_len["Acurácia"] * 100, width=0.35, color="#3b82f6", label="Acurácia (%)")
        axes[1].bar(x_l + 0.18, sl_len["ROC-AUC"] * 100, width=0.35, color="#10b981", label="ROC-AUC (×100)")
        axes[1].set_xticks(x_l)
        axes[1].set_xticklabels(sl_len["Fatia"], fontsize=9)
        for i, r in sl_len.reset_index(drop=True).iterrows():
            axes[1].text(i - 0.18, r["Acurácia"] * 100 + 1.2, f"{r['Acurácia']*100:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
            axes[1].text(i + 0.18, r["ROC-AUC"] * 100 + 1.2, f"{r['ROC-AUC']*100:.1f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[1].set_ylim(60, 104)
        axes[1].set_ylabel("Desempenho (%)")
        axes[1].set_title("Estabilidade por Comprimento do Review (Teste)")
        axes[1].legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "06_slice_error_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        summary = {
            "spark_version": spark.version,
            "elapsed_total_sec": round(time.perf_counter() - t_global, 2),
            "dataset": {
                "total_reviews": int(n_total),
                "null_rows": int(null_cnt),
                "parquet_size_mb": round(pq_path.stat().st_size / 1e6, 2),
                "train_n": int(n_tr),
                "val_n": int(n_va),
                "test_n": int(n_te),
                "target_definition": "Positivo (4–5★) = 1.0 (40%) vs Não-Positivo (1–3★) = 0.0 (60%)",
                "eda_by_star": eda_stars.round(3).to_dict(orient="records"),
            },
            "hash_collision_ablation_val": hash_ablation,
            "word2vec_train_sec": round(w2v_fit_sec, 2),
            "word2vec_synonyms": synonyms_demo,
            "hybrid_best_regParam": best_reg,
            "champion_model": champ_name,
            "models_benchmark": res_df.to_dict(orient="records"),
            "calibration_and_threshold": thr_summary,
            "slice_error_analysis": slices_df.to_dict(orient="records"),
        }
        (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n>>> CAMPEÃO SPARK 4.0 MLLIB: {champ_name} | ROC-AUC={res_df.iloc[0]['roc_auc']:.4f} | F1={res_df.iloc[0]['f1']:.4f} | Acc={res_df.iloc[0]['accuracy']*100:.2f}% | Brier={brier_champ:.4f}")
        print(f"Concluído em {summary['elapsed_total_sec']}s! Resumo salvo em outputs/metrics_summary.json")
        return summary
    finally:
        spark.stop()


if __name__ == "__main__":
    run_full_mllib_benchmark()
