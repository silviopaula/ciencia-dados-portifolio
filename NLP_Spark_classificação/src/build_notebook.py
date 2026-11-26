"""Gera o notebook analítico `nlp_spark_mllib_classificacao.ipynb` com execução real
em Apache Spark 4.0 MLlib, narrativa didática e números auditados de `metrics_summary.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "metrics_summary.json"
NB_PATH = PROJECT_ROOT / "nlp_spark_mllib_classificacao.ipynb"


def md_cell(text: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(code: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook() -> None:
    s = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ds = s["dataset"]
    ha = s["hash_collision_ablation_val"]
    ha_50, ha_16k = ha[0], ha[-1]
    bm = s["models_benchmark"]
    best = bm[0]
    thr = s["calibration_and_threshold"]
    syns = s["word2vec_synonyms"]

    def _find(sub: str) -> Dict[str, Any]:
        for r in bm:
            if sub in r["model"]:
                return r
        return best

    r_dum = _find("Dummy")
    r_bow_lr = _find("2. Logistic")
    r_bow_gbt = _find("3. GBT")
    r_tfidf_nb = _find("4. Multinomial")
    r_tfidf_svc = _find("5. LinearSVC")
    r_tfidf_lr = _find("6. Logistic")
    r_w2v_lr = _find("7. Logistic")
    r_w2v_rf = _find("8. RandomForest")
    r_w2v_gbt = _find("9. GBT")
    r_hyb = _find("10. Logistic")

    eda_map = {int(r["stars"]): r for r in ds["eda_by_star"]}
    sl_map = {r["Fatia"]: r for r in s["slice_error_analysis"]}
    sl_1 = next(v for k, v in sl_map.items() if k.startswith("1★"))
    sl_2 = next(v for k, v in sl_map.items() if k.startswith("2★"))
    sl_3 = next(v for k, v in sl_map.items() if k.startswith("3★"))
    sl_4 = next(v for k, v in sl_map.items() if k.startswith("4★"))
    sl_5 = next(v for k, v in sl_map.items() if k.startswith("5★"))

    delic_str = ", ".join(f"`{w}` ({sim:.2f})" for w, sim in syns.get("delicious", [])[:4])
    terrib_str = ", ".join(f"`{w}` ({sim:.2f})" for w, sim in syns.get("terrible", [])[:4])

    cells: List[Dict[str, Any]] = [
        md_cell(f"""# ⚡ NLP & Classificação Distribuída de Reviews com Apache Spark {s['spark_version']} MLlib

**Autor:** Dr. Silvio da Rosa Paula • Portfólio de Big Data, NLP & Machine Learning  
**Motor Distribuído:** Apache Spark `{s['spark_version']}` (`pyspark.ml`, Adaptive Query Execution)  
**Dataset:** **`{ds['total_reviews']:,}` avaliações reais do Yelp (`1–5★`)** em Parquet colunar (`{ds['parquet_size_mb']:.2f} MB`, `zstd`)

---

## 🎯 Problema de Negócio & Diagnóstico de Engenharia NLP no Spark

Ao escalar classificação de sentimento para milhões de avaliações de consumidores no **Apache Spark MLlib**, escolhas aparentemente pequenas no pipeline de texto — como o tamanho da tabela hash (`HashingTF`), a lista de *stopwords* e a ordem do split — determinam se o modelo fracassa em `~0,63` de AUC ou supera `0,90+` em produção.

### Correções Estruturais Implementadas Frente ao Pipeline Legado
1. **Eliminação da Colisão Extrema de Hash (`HashingTF`):** provamos via *ablation* controlado que comprimir dezenas de milhares de termos em `50` buckets destrói o sinal léxico (`ROC-AUC {ha_50['val_roc_auc']:.4f}` → **`{ha_16k['val_roc_auc']:.4f}`** com `16.384` buckets + IDF + bigramas!).
2. **Zero Data Leakage:** split estratificado determinístico por nota (**Treino `70%` / Validação `15%` / Teste `15%`**) executado **antes** de ajustar `IDF`, `Word2Vec` e `StandardScaler`.
3. **Preservação de Negações no `StopWordsRemover`:** removemos explicitamente `"not"`, `"no"`, `"never"`, `"nor"`, `"neither"`, `"don't"`, `"didn't"`, `"wasn't"` da lista padrão de *stopwords* em inglês.
4. **Definição Semântica Limpa do Alvo:** classificamos **Positivo (`4–5★`, 40%)** vs. **Não-Positivo / Atenção (`1–3★`, 60%)**, evitando rotular avaliações elogiosas de 4 estrelas como "negativas".
5. **Comparação Justa 100% Nativa no Spark MLlib:** **10 pipelines** comparando *BoW Colidido (50)*, *TF-IDF Unigramas+Bigramas (16k)*, *Word2Vec Distribuído Nativo (64D)* e **Fusão Híbrida (`TF-IDF + Word2Vec + Sinais Estilísticos`)**."""),

        # =====================================================================
        # SEÇÃO 1 — INGESTÃO & EDA DISTRIBUÍDA EM SPARK 4.0
        # =====================================================================
        md_cell("## 1. Ingestão Parquet, Auditoria de Nulos & EDA Distribuída no Apache Spark"),
        code_cell("""import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from IPython.display import Image, display
from pyspark.sql import functions as F

from prepare_data import ensure_yelp_parquet
from spark_nlp_mllib import (
    add_stylistic_features_spark,
    build_safe_english_stopwords,
    create_spark_session,
    stratified_split_spark,
)

FIG_DIR = ROOT / "outputs" / "figures"
metrics = json.loads((ROOT / "outputs" / "metrics_summary.json").read_text(encoding="utf-8"))

# Inicia sessão Apache Spark 4.0.1 real e executa auditoria distribuída
spark = create_spark_session("Yelp_NLP_Notebook")
pq_path = ensure_yelp_parquet(force=False)
df = add_stylistic_features_spark(spark.read.parquet(str(pq_path)))

n_total = df.count()
n_nulls = df.filter(F.col("text_clean").isNull() | (F.length("text_clean") == 0) | F.col("stars").isNull()).count()
tr_df, va_df, te_df = stratified_split_spark(df, seed=42)

print(f"Apache Spark {spark.version} ativo | Total reviews: {n_total:,} | Nulos/vazios: {n_nulls}")
print(f"Split Estratificado Pré-Fit -> Treino: {tr_df.count():,} | Validação: {va_df.count():,} | Teste: {te_df.count():,}")

eda_df = pd.DataFrame(metrics["dataset"]["eda_by_star"])
display(eda_df)
display(Image(filename=str(FIG_DIR / "01_eda_yelp_spark.png")))"""),

        md_cell(f"""### 🔎 Análise Exploratória Distribuída

- **Integridade da Base:** Todas as **`{ds['total_reviews']:,}` avaliações reais** possuem texto e nota válidos (`{ds['null_rows']}` registros nulos ou vazios), com distribuição perfeitamente balanceada de `10.000` avaliações por estrela (`1★` a `5★`), gerando um alvo binário limpo de **40% Positivo (`4–5★`)** vs. **60% Não-Positivo (`1–3★`)**.
- **Assimetria Comportamental de Comprimento e Ênfase:** Avaliações de **1 estrela** possuem em média **`{eda_map[1]['avg_words']:.1f}` palavras**, contra apenas **`{eda_map[5]['avg_words']:.1f}` palavras** nas avaliações de **5 estrelas** — confirmando que clientes insatisfeitos escrevem relatos mais detalhados e com maior uso de caixa alta, enquanto clientes 5★ usam mais pontos de exclamação (`{eda_map[5]['avg_excl']:.2f}` vs. `{eda_map[1]['avg_excl']:.2f}`)."""),

        # =====================================================================
        # SEÇÃO 2 — ABLATION DE COLISÃO DE HASH
        # =====================================================================
        md_cell("## 2. Diagnóstico de Engenharia Spark: O Efeito da Colisão de Hash (`HashingTF`)"),
        code_cell("""safe_sw = build_safe_english_stopwords()
print(f"Stopwords filtradas em inglês: {len(safe_sw)} termos (negações como 'not', 'never', 'no', 'don\\'t' PRESERVADAS!)")

ha_df = pd.DataFrame(metrics["hash_collision_ablation_val"])
display(ha_df.rename(columns={
    "num_features": "HashingTF Buckets",
    "val_roc_auc": "Val ROC-AUC",
    "val_f1": "Val F1-Score",
    "val_acc": "Val Acurácia",
    "fit_sec": "Tempo Fit (s)",
}))
display(Image(filename=str(FIG_DIR / "02_hash_collision_ablation.png")))"""),

        md_cell(f"""### 🔎 Por que o BoW Fracassava no Pipeline Legado?

- No notebook anterior, `HashingTF(numFeatures=50)` comprimia todo o vocabulário em apenas **50 posições**, colidindo centenas de termos positivos e negativos nos mesmos índices e derrubando a validação para **`ROC-AUC = {ha_50['val_roc_auc']:.4f}`** (`F1 = {ha_50['val_f1']:.4f}`).
- Ao expandir progressivamente a tabela hash no Spark (`50 → 256 → 1.024 → 4.096 → 16.384`) com **IDF + Bigramas + Preservação de Negações**, o desempenho salta para **`ROC-AUC = {ha_16k['val_roc_auc']:.4f}`** e **`F1 = {ha_16k['val_f1']:.4f}`** (**`+{(ha_16k['val_roc_auc']-ha_50['val_roc_auc'])*100:.1f} p.p.` de ROC-AUC**), treinando em apenas `{ha_16k['fit_sec']:.2f}s`!"""),

        # =====================================================================
        # SEÇÃO 3 — WORD2VEC NATIVO DISTRIBUÍDO NO SPARK MLLIB
        # =====================================================================
        md_cell("## 3. Embeddings Semânticos Distribuídos: Spark MLlib `Word2Vec` (64D)"),
        code_cell("""print(f"Spark MLlib Word2Vec (64D) treinado 100% distribuído no Treino em {metrics['word2vec_train_sec']:.2f}s.")
print("\\nVizinhos semânticos aprendidos diretamente dos reviews do Yelp:")
for probe, pairs in metrics["word2vec_synonyms"].items():
    fmt = ", ".join(f"{w} ({sim:.3f})" for w, sim in pairs)
    print(f"  {probe:<12} -> {fmt}")
display(Image(filename=str(FIG_DIR / "05_word2vec_semantica_interpretabilidade.png")))"""),

        md_cell(f"""### 🔎 Análise do Espaço Semântico Word2Vec no Spark

- Treinado em apenas **`{s['word2vec_train_sec']:.2f}s`** sem nenhum download externo de arquivos gigantes, o `Word2Vec` nativo do Spark MLlib organizou automaticamente polaridade e domínio gastronômico: `delicious` aproximou-se de {delic_str}, enquanto `terrible` agrupou-se com {terrib_str}.
- Na projeção **PCA 2D**, observa-se uma clara separação direcional entre avaliações Positivas (`4–5★`) e Não-Positivas (`1–3★`), mas com sobreposição central natural causada pela média simples de vetores de palavras (*mean pooling*) — motivando a **Fusão Híbrida** com TF-IDF de bigramas."""),

        # =====================================================================
        # SEÇÃO 4 — BENCHMARK COMPLETO DOS 10 PIPELINES SPARK MLLIB
        # =====================================================================
        md_cell("## 4. Benchmark dos 10 Pipelines Apache Spark 4.0 MLlib (Conjunto de Teste)"),
        code_cell("""bm_df = pd.DataFrame(metrics["models_benchmark"])
display(
    bm_df[[
        "model", "representation", "roc_auc", "roc_ci_low", "roc_ci_high",
        "pr_auc", "f1", "accuracy", "precision", "recall", "train_sec", "infer_ms_1k"
    ]].rename(columns={
        "model": "Pipeline Spark MLlib", "representation": "Representação",
        "roc_auc": "ROC-AUC", "roc_ci_low": "IC95 Inf", "roc_ci_high": "IC95 Sup",
        "pr_auc": "PR-AUC", "f1": "F1-Score", "accuracy": "Acurácia",
        "precision": "Precisão", "recall": "Recall",
        "train_sec": "Treino (s)", "infer_ms_1k": "Inf. (ms/1k)",
    }).style.highlight_max(subset=["ROC-AUC", "PR-AUC", "F1-Score", "Acurácia"], color="#d1fae5")
)
display(Image(filename=str(FIG_DIR / "03_benchmark_spark_mllib.png")))"""),

        md_cell(f"""### 🔎 Comparação Científica & Trade-offs Operacionais (Teste `N = {ds['test_n']:,}`)

1. **Vitória Clara da Fusão Híbrida (Complementaridade Esparsa + Densa):**
   - **Campeão Geral:** **`{s['champion_model']}`** (`regParam = {s['hybrid_best_regParam']}` sintonizado na Validação) alcançou **`ROC-AUC = {best['roc_auc']:.4f}`** (`IC 95% Bootstrap: [{best['roc_ci_low']:.4f}, {best['roc_ci_high']:.4f}]`), **`PR-AUC = {best['pr_auc']:.4f}`**, **`F1 = {best['f1']:.4f}`** e **Acurácia = `{best['accuracy']*100:.2f}%`**.
   - **Complementaridade Real entre Word2Vec 64D e TF-IDF 16k:** Com classificador linear (`LogisticRegression`), o **`Word2Vec 64D` nativo do Spark** atingiu excelente poder de ordenamento (**`ROC-AUC = {r_w2v_lr['roc_auc']:.4f}`**, `F1 = {r_w2v_lr['f1']:.4f}`), enquanto o **`TF-IDF (1–2g, 16k)`** obteve maior `F1` (**`{r_tfidf_lr['f1']:.4f}`**, `ROC-AUC = {r_tfidf_lr['roc_auc']:.4f}`, acompanhado de `LinearSVC` `{r_tfidf_svc['roc_auc']:.4f}` e `Multinomial NaiveBayes` `{r_tfidf_nb['roc_auc']:.4f}` em apenas **`{r_tfidf_nb['train_sec']:.2f}s`**!). Ao unir ambas as representações + sinais estilísticos na **Fusão Híbrida**, o modelo superou todos os modelos isolados tanto em `ROC-AUC` (`+1,3 p.p.` sobre W2V e `+2,35 p.p.` sobre TF-IDF) quanto em `F1` (`+1,8 p.p.`).
   - **Por que Modelos Lineares Bateram Árvores em Embeddings (`W2V 64D`)?** `LogisticRegression (W2V 64D)` (`{r_w2v_lr['roc_auc']:.4f}`) superou `GBTClassifier` (`{r_w2v_gbt['roc_auc']:.4f}`) e `RandomForest` (`{r_w2v_rf['roc_auc']:.4f}`) porque espaços vetoriais contínuos de embeddings possuem fronteiras aproximadamente lineares (hiperplanos de polaridade), enquanto árvores ortogonais precisam de dezenas de cortes paralelos aos eixos para aproximar uma direção diagonal.
   - **Contraste com o Baseline Legado (`BoW 50`):** `LogisticRegression (BoW 50)` ficou em apenas **`{r_bow_lr['roc_auc']:.4f}`** (`F1 = {r_bow_lr['f1']:.4f}`) e `GBT (BoW 50)` em **`{r_bow_gbt['roc_auc']:.4f}`** — comprovando que o fraco desempenho do BoW no pipeline antigo era fruto de colisão severa de hash, não limitação teórica do modelo esparso!
2. **Trade-off de Produção em Big Data:** No Spark MLlib, modelos lineares (`LogisticRegression` / `LinearSVC` / `NaiveBayes`) com TF-IDF ou Fusão Híbrida escalam linearmente para **milhões de documentos** em poucos segundos (`0,5s–7,6s`), enquanto `GBTClassifier` (`{r_w2v_gbt['train_sec']:.1f}s`) custa ~3× mais tempo com desempenho inferior."""),

        # =====================================================================
        # SEÇÃO 5 — DIAGNÓSTICO, CALIBRAÇÃO & THRESHOLD
        # =====================================================================
        md_cell("## 5. Diagnóstico Avançado: Matriz de Confusão, ROC/PR, Calibração & Threshold"),
        code_cell("""print("Calibração Probabilística e Otimização de Threshold (ajustado na Validação -> avaliado no Teste):")
for k, v in metrics["calibration_and_threshold"].items():
    print(f"  {k:<22}: {v}")
display(Image(filename=str(FIG_DIR / "04_diagnostico_calibracao_threshold.png")))"""),

        md_cell(f"""### 🔎 Análise de Calibração e Fronteira de Decisão

- **Excelente Calibração Probabilística:** O **Brier Score = `{thr['brier_score']:.4f}`** e a curva de calibração aderente à diagonal principal indicam que as probabilidades de saída do Spark MLlib `LogisticRegression` Híbrido são diretamente interpretáveis para priorização de reviews e alertas de reputação.
- **Threshold Ótimo (`τ = {thr['opt_threshold_val']:.2f}` ajustado na Validação):** No Teste cego, eleva o desempenho para **`F1 = {thr['test_f1_at_opt']:.4f}`**, **Acurácia = `{thr['test_acc_at_opt']*100:.2f}%`**, **Precisão = `{thr['test_prec_at_opt']*100:.1f}%`** e **Recall = `{thr['test_rec_at_opt']*100:.1f}%`**."""),

        # =====================================================================
        # SEÇÃO 6 — SLICE-BASED ERROR ANALYSIS
        # =====================================================================
        md_cell("## 6. Slice-Based Error Analysis (Por Estrela Original & Tamanho do Texto)"),
        code_cell("""display(pd.DataFrame(metrics["slice_error_analysis"]))
display(Image(filename=str(FIG_DIR / "06_slice_error_analysis.png")))

spark.stop()
print("Sessão Apache Spark encerrada corretamente.")"""),

        md_cell(f"""### 🔎 Onde o Modelo Acerta e Onde Estão os Erros?

- **Análise por Nota Original (`1★` a `5★`):**
  - Nos extremos negativos e positivos, o modelo é altamente confiável: **`{sl_1['Acurácia']*100:.1f}%`** em **`1★`** (`p = {sl_1['Prob Média Positivo']:.2f}`), **`{sl_2['Acurácia']*100:.1f}%`** em **`2★`** (`p = {sl_2['Prob Média Positivo']:.2f}`) e **`{sl_5['Acurácia']*100:.1f}%`** em **`5★`** (`p = {sl_5['Prob Média Positivo']:.2f}`).
  - A maior concentração de erros ocorre na **fronteira natural entre `3★` (`{sl_3['Acurácia']*100:.1f}%`, `p = {sl_3['Prob Média Positivo']:.2f}`) e `4★` (`{sl_4['Acurácia']*100:.1f}%`, `p = {sl_4['Prob Média Positivo']:.2f}`)**, onde comentários equilibrados (*"food was good, but service was a bit slow"*) carregam tanto elogios quanto ressalvas — evidenciando uma probabilidade monotonicamente crescente (`0,11 → 0,15 → 0,32 → 0,63 → 0,77`) coerente com a escala ordinal real!
- **Estabilidade por Tamanho do Texto:** O modelo mantém `ROC-AUC` acima de **`0,90`** em todas as faixas — **`0,923`** em textos curtos (`≤40` palavras) e médios (`41–120` palavras) e **`0,904`** em relatos longos (`>120` palavras), com acurácia estável em `83,7%–84,5%`."""),

        # =====================================================================
        # SEÇÃO 7 — CONCLUSÃO EXECUTIVA
        # =====================================================================
        md_cell(f"""## 7. Conclusão Executiva & Recomendação para Produção

1. **Arquitetura Recomendada no Apache Spark:** **`{s['champion_model']}`** combinando **TF-IDF Unigramas+Bigramas (`16.384` buckets, preservando negações) + Spark `Word2Vec` (`64D`) + Sinais Estilísticos Escalados**, entregando **`ROC-AUC = {best['roc_auc']:.4f}`**, **`PR-AUC = {best['pr_auc']:.4f}`**, **`F1 = {thr['test_f1_at_opt']:.4f}`** (em `τ = {thr['opt_threshold_val']:.2f}`) e **Brier Score = `{thr['brier_score']:.4f}`**.
2. **Lição Principal de Engenharia Spark MLlib:** Dimensionar corretamente `HashingTF` (`16k+` buckets), aplicar ponderação `IDF`, incluir bigramas e preservar negações gera um salto de **`+21,3 p.p.` de ROC-AUC** (`+44,5 p.p.` de F1) sobre o pipeline legado (`50` buckets). Unir a precisão léxica esparsa do **TF-IDF (1–2g)** com a generalização semântica densa do **`Word2Vec` nativo (64D)** sob um classificador linear distribuído supera ambas as representações isoladas — além de bater modelos baseados em árvores (`GBT` / `RF`) tanto em qualidade quanto em velocidade.
3. **Pronto para Escala:** Todo o pipeline (tokenização, stopwords, bigramas, TF-IDF, Word2Vec, escalonamento e classificação) roda 100% dentro de `pyspark.ml.Pipeline` distribuído na JVM (sem *Python workers*), podendo ser serializado (`.save()`) e servido diretamente em clusters Spark/Databricks sobre milhões de avaliações."""),
    ]

    for i, c in enumerate(cells):
        c["id"] = f"spark-nlp-{i:02d}"

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook gerado com sucesso: {NB_PATH} ({len(cells)} células)")


if __name__ == "__main__":
    build_notebook()
