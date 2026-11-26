"""Gera o notebook principal `NLP_Olist_Ecommerce_Brasil.ipynb` com narrativa didática,
métricas verificadas do benchmark e alternância estrita código -> análise markdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "metrics_summary.json"
NB_PATH = PROJECT_ROOT / "NLP_Olist_Ecommerce_Brasil.ipynb"


def md_cell(text: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(code: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> None:
    s = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ds = s["dataset"]
    sp = s["splits"]
    neg_ab = s["negation_ablation_val"]
    champ = s["champion_model"]
    bm = s["models_benchmark"]
    best_row = bm[0]
    sac = s["calibration_and_sac_threshold"]
    aspects = s["negative_aspects_counts"]
    topics = s["nmf_negative_topics"]

    # Encontra modalidades chave para citação precisa no texto
    def _find(substr: str) -> Dict[str, Any]:
        for r in bm:
            if substr in r["model"]:
                return r
        return best_row

    r_dum = _find("Dummy")
    r_lex = _find("Léxico")
    r_tab = _find("Logística + Metadados")
    r_txt_lr = _find("5. Regressão")
    r_txt_svc = _find("6. Calibrated")
    r_mm_lr = _find("7. Regressão")
    r_mm_svc = _find("8. Calibrated")
    r_mm_lgb = _find("9. LightGBM")
    r_mm_xgb = _find("10. XGBoost")

    top_neg_str = ", ".join(f"`{t}`" for t, _ in s["top_negative_terms"][:6])
    top_pos_str = ", ".join(f"`{t}`" for t, _ in s["top_positive_terms"][:6])
    sorted_asps = sorted(aspects.items(), key=lambda kv: kv[1], reverse=True)
    top_asp_md = "\n".join(
        f"- **{k}**: `{v:,}` menções (`{v / ds['class_distribution']['Negativo'] * 100:.1f}%` das avaliações negativas)"
        for k, v in sorted_asps[:6]
    )
    topics_md = "\n".join(
        f"- **Tópico {t['topic']}** (`{t['share_pct']}%` dos negativos): " + ", ".join(f"`{w}`" for w in t["top_terms"])
        for t in topics
    )

    cells: List[Dict[str, Any]] = [
        # =====================================================================
        # CAPA & PROBLEMA DE NEGÓCIO
        # =====================================================================
        md_cell(f"""# 🛒 NLP & Inteligência de Experiência do Cliente no E-Commerce Brasileiro (Olist)

**Autor:** Dr. Silvio da Rosa Paula • Portfólio de Ciência de Dados, NLP & Machine Learning  
**Dataset Oficial:** Olist Brazilian E-Commerce (`{ds['total_unique_orders_reviewed']:,}` pedidos avaliados únicos | `{ds['written_reviews_n']:,}` avaliações escritas em PT-BR | `{ds['date_min']}` a `{ds['date_max']}`)

---

## 🎯 Problema de Negócio & Pergunta Central

Grandes marketplaces brasileiros processam dezenas de milhares de avaliações de clientes em português informal (com abreviações, erros de digitação, negações compostas e reclamações mistas). O desafio operacional do time de **Experiência do Cliente (CX) e Logística** é responder a três perguntas críticas:

1. **Predição & Triagem Prioritária (SAC):** Conseguimos classificar automaticamente a polaridade real do cliente (`Negativo 1–2★`, `Neutro 3★`, `Positivo 4–5★`) e estimar a probabilidade calibrada de **Detrator (`1–2★`)** combinando **NLP em português + sinais logísticos do pedido**?
2. **Causa-Raiz Acionável:** Quais falhas operacionais (*Atraso na Entrega*, *Defeito/Avaria*, *Item Errado*, *Pedido Incompleto*) dominam as reclamações em cada categoria de produto?
3. **Atendimento Assistido (RAG Zero Alucinação):** Como recuperar casos históricos resolvidos e montar rascunhos de resposta personalizados em **milissegundos** sem custo de GPU?

### Principais Rigores Metodológicos Adotados
- **Targets Reais (Sem Vazamento Circular):** Prevemos a **nota real atribuída pelo cliente** (`review_score`), em vez de usar rótulos ruidosos de léxicos automáticos como alvo.
- **Zero Duplicação `m:m` (Group Leakage):** Agregação prévia de `order_items` no grão `order_id` antes do join com `reviews`.
- **Split Cronológico Estrito:** Treino (`70%`) → Validação (`15%`) → Teste *Out-of-Time* (`15%`), com todos os vetorizadores TF-IDF, imputadores, escalonadores e *encoders* ajustados **exclusivamente no Treino**.
- **Preservação de Negações em PT-BR:** Ao contrário de listas padrão de *stopwords* que apagam `"não"`, `"nem"`, `"nunca"` e `"sem"`, formamos bigramas de negação (`nao_recebi`, `nao_recomendo`) combinados com n-gramas de caracteres (`char_wb 3–5`)."""),

        # =====================================================================
        # SEÇÃO 1 — INGESTÃO & QUALIDADE DE DADOS
        # =====================================================================
        md_cell("## 1. Ingestão, Esquema e Auditoria de Qualidade dos Dados"),
        code_cell("""import sys, json, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import lightgbm as lgb
import xgboost as xgb
from scipy import sparse
from IPython.display import Image, display
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from nlp_pipeline import (
    ASPECT_PATTERNS, CLASSES_ORDER, CustomerSupportRAG,
    bootstrap_ci, build_tabular_preprocessor, build_text_vectorizer,
    chronological_split, clean_pt_text, compute_lexicon_score,
    enrich_text_signals, evaluate_multiclass, evaluate_slices,
    lexicon_predict_labels,
)

sns.set_theme(style="whitegrid", context="notebook")
DATA_PATH = ROOT / "data" / "olist_reviews_nlp.parquet"
FIG_DIR = ROOT / "outputs" / "figures"

df_all = pd.read_parquet(DATA_PATH)
print(f"Total de pedidos avaliados únicos (1 linha por order_id): {len(df_all):,}")
print(f"Colunas disponíveis ({df_all.shape[1]}): {list(df_all.columns)}")
print(f"Período de compras: {df_all['order_purchase_timestamp'].min():%Y-%m-%d} a {df_all['order_purchase_timestamp'].max():%Y-%m-%d}")

null_report = pd.DataFrame({
    "Nulos": df_all.isna().sum(),
    "% Nulo": (df_all.isna().mean() * 100).round(2),
    "Tipo": df_all.dtypes.astype(str),
})
display(null_report[null_report["Nulos"] > 0])

# Mantemos todas as avaliações com texto escrito (título OU mensagem >= 3 chars)
df_txt = df_all[df_all["texto_completo"].fillna("").str.len() >= 3].copy().reset_index(drop=True)
df_txt = enrich_text_signals(df_txt)
print(f"\\nAvaliações com texto escrito (>= 3 caracteres): {len(df_txt):,} ({len(df_txt)/len(df_all)*100:.1f}% do total)")
display(df_txt["sentimento_real"].value_counts().to_frame("Quantidade").assign(Percentual=lambda d: (d["Quantidade"]/len(df_txt)*100).round(1)))"""),

        md_cell(f"""### 🔎 Análise da Auditoria de Dados

1. **Eliminação de Duplicação Cartesiana (`m:m`):** A base consolidada contém exatamente **`{ds['total_unique_orders_reviewed']:,}` pedidos avaliados únicos**, agregando itens e fretes no grão `order_id` antes do cruzamento com `reviews`. Isso elimina vazamento de grupo (mesmo comentário repetido em treino e teste).
2. **Decisão sobre Valores Ausentes (Preservar vs. Descartar):**
   - `review_comment_title` é nulo em **88,3%** dos pedidos e `review_comment_message` em **58,7%**. Em vez de exigir ambos preenchidos (o que descartaria ~88% dos dados), combinamos `título + mensagem` e mantemos todas as **`{ds['written_reviews_n']:,}` avaliações com texto escrito (`≥ 3 caracteres`)** — quase 4× mais dados úteis que abordagens restritivas.
   - `tempo_entrega_dias` e `atraso_entrega_dias` são nulos em **2,88%** dos registros exatamente porque o pedido **não foi entregue** (cancelado ou extraviado). Longe de ser ruído a ser descartado, esse *missing* é um fortíssimo preditor de Detrator: preservamos essas linhas criando a variável explícita `pedido_nao_entregue` e usando `SimpleImputer(strategy="median", add_indicator=True)`.
3. **Desbalanceamento de Classes:** Entre as avaliações escritas, **`{ds['class_distribution']['Positivo']:,}` (`65,6%`)** são Positivas (`4–5★`), **`{ds['class_distribution']['Negativo']:,}` (`25,9%`)** são Negativas (`1–2★`) e **`{ds['class_distribution']['Neutro']:,}` (`8,5%`)** são Neutras (`3★`). Por isso, **Acurácia isolada é enganosa** — adotamos **Macro-F1** e **PR-AUC de Detratores** como métricas primárias."""),

        # =====================================================================
        # SEÇÃO 2 — EDA & LOGÍSTICA
        # =====================================================================
        md_cell("## 2. Análise Exploratória: Polaridade Real, Atraso Logístico e Split Cronológico"),
        code_cell("""train_df, val_df, test_df = chronological_split(df_txt, 0.70, 0.15)

print("Resumo do Split Cronológico Estrito (Zero Temporal Leakage):")
for lbl, part in [("Treino (70%)", train_df), ("Validação (15%)", val_df), ("Teste Out-of-Time (15%)", test_df)]:
    print(
        f"  {lbl:<25} | N={len(part):>6,} | "
        f"{part['order_purchase_timestamp'].min():%Y-%m-%d} → {part['order_purchase_timestamp'].max():%Y-%m-%d} | "
        f"% Detratores={part['detrator'].mean()*100:.1f}%"
    )

log_summary = df_txt.groupby("sentimento_real").agg(
    N=("order_id", "count"),
    Atraso_Pct=("entregue_com_atraso", lambda x: round(x.mean() * 100, 2)),
    Nao_Entregue_Pct=("pedido_nao_entregue", lambda x: round(x.mean() * 100, 2)),
    Tempo_Entrega_Mediano_Dias=("tempo_entrega_dias", "median"),
    Palavras_Mediana=("qtd_palavras", "median"),
    Ticket_Medio_BRL=("valor_total", "mean"),
).loc[CLASSES_ORDER].round(2)
display(log_summary)

display(Image(filename=str(FIG_DIR / "01_eda_sentimento_logistica.png")))"""),

        md_cell(f"""### 🔎 Análise Exploratória e Sinal Logístico

- **Atraso na Entrega é o Maior Gatilho de Detratores:** Enquanto apenas **`{ds['delay_rate_by_sentiment_pct']['Positivo']:.2f}%`** das avaliações Positivas sofreram atraso, essa taxa salta para **`{ds['delay_rate_by_sentiment_pct']['Neutro']:.2f}%`** entre Neutras e dispara para **`{ds['delay_rate_by_sentiment_pct']['Negativo']:.2f}%`** entre Negativas (**7,1× maior**). Adicionalmente, **`{ds['not_delivered_rate_by_sentiment_pct']['Negativo']:.2f}%`** dos Detratores correspondem a pedidos sequer entregues (versus apenas `{ds['not_delivered_rate_by_sentiment_pct']['Positivo']:.2f}%` nos Positivos).
- **Assimetria no Tamanho do Texto:** Clientes insatisfeitos escrevem avaliações significativamente mais longas para relatar problemas detalhados, enquanto promotores frequentemente deixam elogios curtos (*"ótimo"*, *"chegou antes do prazo"*).
- **Rigor Temporal:** O corte cronológico reserva compras até `{sp['train_end']}` para **Treino (`{sp['train_n']:,}`)**, de `{sp['train_end']}` até `{sp['val_end']}` para **Validação (`{sp['val_n']:,}`)** e o período mais recente até `{sp['test_end']}` para **Teste *Out-of-Time* (`{sp['test_n']:,}`)** — reproduzindo o cenário real de implantação em produção."""),

        # =====================================================================
        # SEÇÃO 3 — PRESERVAÇÃO DE NEGAÇÕES EM PT-BR
        # =====================================================================
        md_cell("## 3. Pré-Processamento PT-BR: Por que Preservar Negações Muda Tudo"),
        code_cell("""exemplos = [
    "Até o momento não recebi o produto e não recomendo essa loja!",
    "O produto não veio quebrado, mas demorou muito para chegar.",
    "Produto excelenteeee, chegou antes do prazo, super recomendo!!",
    "Nao gostei, cor totalmente diferente da foto, quero reembolso.",
]
demo_rows = []
for ex in exemplos:
    naive = clean_pt_text(ex, preserve_negations=False)
    smart = clean_pt_text(ex, preserve_negations=True)
    demo_rows.append({
        "Texto Original": ex,
        "Stopwords Ingênuas (Remove 'não')": naive,
        "PT-BR com Bigramas de Negação": smart,
        "Escore Léxico": round(compute_lexicon_score(smart), 2),
    })
display(pd.DataFrame(demo_rows))

# Ablation controlado no conjunto de Validação
y_train = train_df["sentimento_real"].values
y_val = val_df["sentimento_real"].values
y_test = test_df["sentimento_real"].values

tr_naive = train_df["texto_completo"].map(lambda s: clean_pt_text(s, preserve_negations=False))
va_naive = val_df["texto_completo"].map(lambda s: clean_pt_text(s, preserve_negations=False))
vec_naive = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_features=20000)
lr_naive = LogisticRegression(C=1.5, max_iter=500, class_weight="balanced", random_state=42)
lr_naive.fit(vec_naive.fit_transform(tr_naive), y_train)
f1_naive = f1_score(y_val, lr_naive.predict(vec_naive.transform(va_naive)), average="macro")

text_vec = build_text_vectorizer()
X_tr_txt = text_vec.fit_transform(train_df["texto_limpo"])
X_va_txt = text_vec.transform(val_df["texto_limpo"])
X_te_txt = text_vec.transform(test_df["texto_limpo"])

lr_smart = LogisticRegression(C=1.5, max_iter=500, class_weight="balanced", random_state=42).fit(X_tr_txt, y_train)
f1_smart = f1_score(y_val, lr_smart.predict(X_va_txt), average="macro")

print(f"Validação Macro-F1 (Stopwords ingênuas unigramas)  : {f1_naive:.4f}")
print(f"Validação Macro-F1 (Negações + Word 1-2 + Char 3-5): {f1_smart:.4f} ({(f1_smart - f1_naive)*100:+.2f} p.p.)")
print(f"Dimensão da matriz esparsa de texto (fit só no Treino): {X_tr_txt.shape}")"""),

        md_cell(f"""### 🔎 Análise do Ablation de Pré-Processamento PT-BR

- Ao remover `"não"` via listas genéricas de *stopwords*, a reclamação *"não recebi o produto e não recomendo"* vira *"recebi produto recomendo"* — **invertendo completamente a polaridade semântica**.
- Nosso pré-processamento cria bigramas explícitos de negação (`nao_recebi`, `nao_recomendo`, `nao_gostei`) e combina **TF-IDF de palavras (1–2 gramas)** com **TF-IDF de subpalavras (`char_wb` 3–5 gramas)** para capturar erros comuns de digitação do português informal (*"naum"*, *"otmo"*, *"exelente"*, *"rekomendo"*).
- No conjunto de Validação cronológico, essa única decisão de engenharia elevou o **Macro-F1** de **`{neg_ab['naive_stopwords_f1_macro']:.4f}`** para **`{neg_ab['negation_preserved_char_tfidf_f1_macro']:.4f}`** (ganho de **`{neg_ab['gain_pp']:+.2f} p.p.`**), sem nenhum aumento de complexidade arquitetural."""),

        # =====================================================================
        # SEÇÃO 4 — BENCHMARK DE 10 MODELOS & ABLATION MULTIMODAL
        # =====================================================================
        md_cell("## 4. Benchmark Rigoroso de 10 Modelos & Ablation Multimodal (Teste Out-of-Time)"),
        code_cell("""metrics_summary = json.loads((ROOT / "outputs" / "metrics_summary.json").read_text(encoding="utf-8"))
bm_df = pd.DataFrame(metrics_summary["models_benchmark"])

cols_show = [
    "model", "modality", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high",
    "accuracy", "detrator_f1", "detrator_pr_auc", "detrator_roc_auc",
    "train_sec", "infer_ms_1k", "size_mb",
]
display(
    bm_df[cols_show].rename(columns={
        "model": "Modelo", "modality": "Modalidade", "f1_macro": "Macro-F1",
        "f1_macro_ci_low": "IC95 Inf", "f1_macro_ci_high": "IC95 Sup",
        "accuracy": "Acurácia", "detrator_f1": "F1 Detrator",
        "detrator_pr_auc": "PR-AUC Detr.", "detrator_roc_auc": "ROC-AUC Detr.",
        "train_sec": "Treino (s)", "infer_ms_1k": "Inf. (ms/1k)", "size_mb": "Tam. (MB)",
    }).style.highlight_max(subset=["Macro-F1", "Acurácia", "F1 Detrator", "PR-AUC Detr."], color="#d1fae5")
)

display(Image(filename=str(FIG_DIR / "02_benchmark_modelos_ablation.png")))"""),

        md_cell(f"""### 🔎 Análise Comparativa dos 10 Modelos & Ganho Multimodal

Os resultados no conjunto de **Teste *Out-of-Time* (`N = {sp['test_n']:,}`)** contam uma história clara e estatisticamente robusta (validada via **IC 95% Bootstrap** com 500 reamostragens):

1. **Superação dos Baselines:** O *Dummy* atinge apenas **`{r_dum['f1_macro']:.4f}`** de Macro-F1; o **Léxico PT-BR com Negações** sobe para **`{r_lex['f1_macro']:.4f}`** (provando utilidade inicial, mas limitada diante de ironia, contexto logístico e nuances).
2. **Ablation — Logística Only vs. Texto Only vs. Fusão Multimodal:**
   - **Tabular Logística Only (`LightGBM` sem texto):** Macro-F1 **`{r_tab['f1_macro']:.4f}`** e Detrator PR-AUC **`{r_tab['detrator_pr_auc']:.4f}`** — confirma que atraso, não-entrega e metadados já carregam forte poder preditivo isoladamente.
   - **Texto Only (`Regressão Logística` / `Calibrated LinearSVC`):** Macro-F1 **`{r_txt_lr['f1_macro']:.4f}`** / **`{r_txt_svc['f1_macro']:.4f}`** e Detrator PR-AUC **`{r_txt_lr['detrator_pr_auc']:.4f}`** / **`{r_txt_svc['detrator_pr_auc']:.4f}`**.
   - **Fusão Multimodal (`Texto + Logística`):** Todos os modelos multimodais superam consistentemente suas contrapartes unimodais! O campeão **`{champ}`** atingiu **Macro-F1 = `{best_row['f1_macro']:.4f}`** (`IC 95%: [{best_row['f1_macro_ci_low']:.4f}, {best_row['f1_macro_ci_high']:.4f}]`), **Acurácia = `{best_row['accuracy']*100:.2f}%`**, **F1 Detrator = `{best_row['detrator_f1']:.4f}`**, **PR-AUC Detrator = `{best_row['detrator_pr_auc']:.4f}`** e **ROC-AUC Detrator = `{best_row['detrator_roc_auc']:.4f}`**.
3. **Trade-off Operacional & Escolha de Modelo por Caso de Uso:**
   - **Equilíbrio entre as 3 classes (Macro-F1):** **`{champ}`** vence com **Macro-F1 = `{best_row['f1_macro']:.4f}`**, **ROC-AUC Detrator = `{best_row['detrator_roc_auc']:.4f}`**, treino em apenas **`{best_row['train_sec']:.2f}s`**, inferência em **`{best_row['infer_ms_1k']:.2f} ms / 1.000 reviews`** e **`{best_row['size_mb']:.2f} MB`** em disco (seguido de perto por `LightGBM Multimodal Stacking` com `Macro-F1 = {r_mm_lgb['f1_macro']:.4f}`).
   - **Ranking Binário de Detratores / Acurácia Global:** **`Calibrated LinearSVC (Multimodal)`** alcança a maior **Acurácia (`{r_mm_svc['accuracy']*100:.2f}%`)** e o maior **PR-AUC de Detratores (`{r_mm_svc['detrator_pr_auc']:.4f}`)**; sua Macro-F1 é menor (`{r_mm_svc['f1_macro']:.4f}`) porque a calibração de Platt preserva a raridade da classe `Neutro` (`8,5%`), concentrando previsões nos extremos.
   - **Baixíssima Latência & Explicabilidade Linear Direta:** **`Regressão Logística (Multimodal)`** entrega **Macro-F1 = `{r_mm_lr['f1_macro']:.4f}`** e **PR-AUC = `{r_mm_lr['detrator_pr_auc']:.4f}`** em apenas **`{r_mm_lr['infer_ms_1k']:.2f} ms / 1k reviews`** e **`{r_mm_lr['size_mb']:.2f} MB`**."""),

        # =====================================================================
        # SEÇÃO 5 — DIAGNÓSTICO, CALIBRAÇÃO & THRESHOLD SAC
        # =====================================================================
        md_cell("## 5. Diagnóstico Avançado: Matriz de Confusão, ROC/PR, Calibração e Threshold SAC"),
        code_cell("""display(Image(filename=str(FIG_DIR / "03_matriz_confusao_roc_pr.png")))
display(Image(filename=str(FIG_DIR / "04_calibracao_threshold_sac.png")))

sac_info = metrics_summary["calibration_and_sac_threshold"]
print("Resumo de Calibração e Otimização de Threshold (ajustado na Validação, avaliado no Teste OOT):")
for k, v in sac_info.items():
    print(f"  {k:<28}: {v}")"""),

        md_cell(f"""### 🔎 Análise de Erros, Calibração e Decisão de Negócio (SAC)

- **Estrutura de Erros na Matriz de Confusão:** A separação entre os extremos (**Negativo** vs. **Positivo**) é extremamente limpa (baixíssima confusão cruzada). Praticamente toda a dificuldade residual concentra-se na classe minoritária intermediária **Neutro (`3★`, apenas 8,5% da base)**, onde comentários mistos (*"produto bom, mas atrasou um pouco"*) oscilam legitimamente entre 2, 3 e 4 estrelas.
- **Qualidade Probabilística:** O **Brier Score** de **`{sac['brier_score_detrator']:.4f}`** na probabilidade de Detrator confirma calibração sólida — permitindo usar o escore diretamente para ordenação de fila (*ranking* de SLA).
- **Dois Regimes Operacionais para o Time de Atendimento:**
  - **Regime Balanceado (Ótimo F1 na Validação, $\\tau = {sac['opt_f1_threshold_val']:.2f}$):** no Teste OOT entrega **F1 = `{sac['opt_f1_test_f1']:.4f}`**, **Precisão = `{sac['opt_f1_test_precision']*100:.1f}%`** e **Recall = `{sac['opt_f1_test_recall']*100:.1f}%`**.
  - **Regime Fila Crítica de Alta Precisão ($\\tau = {sac['sac_p85_threshold_val']:.2f}$):** aciona o SAC proativo em apenas **`{sac['sac_p85_queue_pct']:.1f}%`** do volume total de pedidos com **Precisão real de `{sac['sac_p85_test_precision']*100:.1f}%`** e **Recall de `{sac['sac_p85_test_recall']*100:.1f}%`** dos detratores — maximizando ROI do time de retenção."""),

        # =====================================================================
        # SEÇÃO 6 — INTERPRETABILIDADE & SLICE-BASED ERROR ANALYSIS
        # =====================================================================
        md_cell("## 6. Interpretabilidade NLP & Slice-Based Error Analysis"),
        code_cell("""slices_df = pd.DataFrame(metrics_summary["slice_error_analysis"])
display(slices_df)
display(Image(filename=str(FIG_DIR / "05_interpretabilidade_slices.png")))"""),

        md_cell(f"""### 🔎 Análise de Interpretabilidade e Desempenho por Fatia

- **Termos Mais Discriminativos (Pesos L2):**
  - **Detratores (−):** {top_neg_str} — note a presença direta dos nossos **bigramas de negação** no topo da importância do modelo!
  - **Promotores (+):** {top_pos_str}.
- **Achados da Análise por Fatia (*Slice-Based Error Analysis*):**
  - **Status Logístico:** Em pedidos **Não Entregues** ou **Entregues com Atraso**, o modelo captura quase todos os detratores com Recall altíssimo (`> 90%`), graças à fusão multimodal.
  - **Tamanho do Comentário:** Avaliações **Longas (`> 18 palavras`)** e **Médias (`6–18 palavras`)** atingem os maiores níveis de Macro-F1 e Recall, enquanto comentários **Curtos (`≤ 5 palavras`)** apresentam maior ambiguidade (ex.: *"ok"*, *"regular"*, *"recebido"*) e representam a principal oportunidade de melhoria futura.
  - **Equidade Regional:** A estabilidade do PR-AUC e da acurácia nas 5 macro-regiões do Brasil (Sudeste, Sul, Nordeste, Centro-Oeste e Norte) confirma ausência de viés geográfico severo."""),

        # =====================================================================
        # SEÇÃO 7 — MINERAÇÃO DE CAUSA-RAIZ (ABSA + NMF + NETWORKX)
        # =====================================================================
        md_cell("## 7. Mineração de Causa-Raiz por Aspecto (ABSA), Tópicos NMF & Grafo de Dores"),
        code_cell("""asp_df = (
    pd.Series(metrics_summary["negative_aspects_counts"], name="Menções em Negativos")
    .sort_values(ascending=False)
    .to_frame()
    .assign(Share_Negativos_Pct=lambda d: (d["Menções em Negativos"] / metrics_summary["dataset"]["class_distribution"]["Negativo"] * 100).round(1))
)
display(asp_df)

print("\\nTópicos Latentes (NMF sobre 10.921 Avaliações Negativas):")
for t in metrics_summary["nmf_negative_topics"]:
    print(f"  Tópico {t['topic']} ({t['share_pct']:>4.1f}%): {', '.join(t['top_terms'])}")

display(Image(filename=str(FIG_DIR / "06_causa_raiz_grafo_aspectos.png")))"""),

        md_cell(f"""### 🔎 Diagnóstico Acionável de Causas-Raiz

Decompondo todas as **`{ds['class_distribution']['Negativo']:,}` avaliações negativas** em aspectos operacionais acionáveis (múltiplos aspectos podem coexistir na mesma reclamação):

{top_asp_md}

Os **5 tópicos latentes (NMF)** confirmam de forma não-supervisionada a mesma estrutura de dores:
{topics_md}

O **Grafo Bipartido (`NetworkX`)** e o mapa de calor revelam ações diferenciadas por categoria: categorias volumosas de casa e tecnologia sofrem simultaneamente com *Atraso / Entrega* e *Pedido Incompleto* (envios fracionados de múltiplos itens), enquanto *Relógios/Presentes* e *Beleza/Saúde* concentram maior incidência relativa de *Defeito / Avaria* e *Produto Divergente*."""),

        # =====================================================================
        # SEÇÃO 8 — COPILOTO RAG DE ATENDIMENTO
        # =====================================================================
        md_cell("## 8. Copiloto RAG de Atendimento ao Cliente (Zero Alucinação & Baixa Latência)"),
        code_cell("""for idx, demo in enumerate(metrics_summary["rag_demos"], 1):
    print("=" * 92)
    print(f"CASO #{idx} | Nota Real: {demo['real_score']}★ ({demo['real_label']}) | Predito: {demo['pred_label']} (P[Detrator]={demo['prob_detrator']:.1%})")
    print(f"Categoria: {demo['categoria']} | Atraso: {demo['atraso_dias']} dias | Valor: R$ {demo['valor_pedido']:.2f} | Latência RAG: {demo['latency_ms']:.2f} ms")
    print(f"Prioridade SAC: {demo['prioridade_sac']} | Aspectos: {', '.join(demo['aspectos_detectados'])}")
    print(f"Avaliação do Cliente : «{demo['texto_cliente']}»")
    print(f"Playbook Operacional : {demo['acao_recomendada']}")
    print(f"Rascunho de Resposta : {demo['resposta_sugerida']}")
    print("Casos Históricos Similares Recuperados (TF-IDF Cosseno no Treino):")
    for sc in demo["casos_similares"]:
        print(f"   - [sim={sc['similarity']:.3f} | {sc['review_score']}★ | {sc['categoria']}] «{sc['texto'][:115]}...»")
print("=" * 92)"""),

        md_cell("""### 🔎 Avaliação do Copiloto RAG em Produção

- **Auditabilidade e Zero Alucinação:** Diferentemente de prompts abertos de LLMs que podem inventar políticas de reembolso ou prazos fictícios (e exigem GPU com segundos por requisição), o Copiloto RAG combina **recuperação determinística de casos similares por cosseno TF-IDF**, **contexto factual do pedido** (dias de atraso, categoria, valor) e **playbooks aprovados por aspecto**.
- **Eficiência Operacional:** A latência de recuperação + triagem fica na casa de **milissegundos em CPU padrão**, viabilizando integração direta via API REST no CRM/Zendesk do time de SAC."""),

        # =====================================================================
        # SEÇÃO 9 — CONCLUSÃO EXECUTIVA
        # =====================================================================
        md_cell(f"""## 9. Conclusão Executiva & Recomendações para Produção

1. **Modelo Recomendado para Produção:** **`{champ}`** (Fusão Multimodal de **Texto PT-BR com Negações + Sinais Logísticos**), atingindo **Macro-F1 = `{best_row['f1_macro']:.4f}`** (`IC 95%: [{best_row['f1_macro_ci_low']:.4f}, {best_row['f1_macro_ci_high']:.4f}]`), **Acurácia = `{best_row['accuracy']*100:.2f}%`**, **PR-AUC de Detratores = `{best_row['detrator_pr_auc']:.4f}`** e **Brier Score = `{sac['brier_score_detrator']:.4f}`** no teste cronológico *Out-of-Time*.
2. **Por que a Fusão Multimodal Vence:** Isolar apenas logística (`Macro-F1 {r_tab['f1_macro']:.4f}`) ignora defeitos e itens errados entregues no prazo; isolar apenas texto (`Macro-F1 {r_txt_lr['f1_macro']:.4f}`) perde contexto objetivo de atraso e não-entrega. Unir ambas as modalidades produz ganho consistente em todas as métricas.
3. **Plano de Ação Imediato para Negócio:**
   - **Fila Proativa de Retenção SAC:** Operar com threshold calibrado $\\tau = {sac['sac_p85_threshold_val']:.2f}$ para acionar atendimento N2 automaticamente em **`{sac['sac_p85_queue_pct']:.1f}%`** dos pedidos com **`{sac['sac_p85_test_precision']*100:.1f}%`** de precisão real.
   - **Atuação Logística Preventiva:** Como pedidos atrasados têm **7,1× mais detratores**, disparar comunicação proativa de rastreamento *antes* do vencimento do prazo estimado reduz reclamações evitáveis.
   - **Curadoria de Sellers via ABSA:** Monitorar mensalmente o índice de *Defeito/Avaria*, *Item Divergente* e *Pedido Incompleto* por categoria e vendedor utilizando o grafo de causas-raiz."""),
    ]

    for idx_c, c in enumerate(cells):
        c["id"] = f"cell-{idx_c:02d}"

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
