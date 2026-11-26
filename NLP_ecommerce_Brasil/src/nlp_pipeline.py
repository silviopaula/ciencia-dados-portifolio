"""Pipeline modular de NLP em Português (PT-BR), Engenharia de Features Multimodais,
Diagnóstico Estatístico e Copiloto RAG de Atendimento para o E-Commerce Olist Brasil.

Princípios adotados:
- Zero Data Leakage: todos os vetorizadores TF-IDF, imputadores, scalers e encoders são
  ajustados exclusivamente no conjunto de Treino após split cronológico estrito.
- Preservação de Negações em PT-BR: ao contrário de listas ingênuas de stopwords que
  removem "não", "nem", "nunca", "sem", preservamos negações e formamos bigramas de negação.
- Targets Reais: predição da polaridade real da nota do cliente (review_score 1..5) e
  detecção de Detratores (1–2 estrelas) para triagem priorizada de SAC.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

CLASSES_ORDER = ["Negativo", "Neutro", "Positivo"]

# Stopwords PT-BR cuidadosamente curadas SEM palavras de negação ou intensidade crítica!
PT_STOPWORDS_SAFE = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para",
    "com", "que", "e", "ou", "se", "ao", "aos", "à", "às", "eu", "ele", "ela", "eles",
    "elas", "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas", "nosso",
    "nossa", "este", "esta", "isso", "isto", "aquele", "aquela", "aqui", "ali", "lá",
    "ja", "já", "foi", "era", "ser", "esta", "está", "estao", "estão", "estava",
    "ter", "tem", "tinha", "fui", "sao", "são", "como", "quando", "onde", "qual",
    "quem", "pois", "tambem", "também", "ate", "até", "so", "só", "pra", "pro",
}

NEGATION_WORDS = {"nao", "não", "nem", "nunca", "jamais", "sem", "nada", "nenhum", "nenhuma"}

POS_LEXICON = {
    "otimo": 2.0, "excelente": 2.2, "perfeito": 2.2, "maravilhoso": 2.2, "adorei": 2.0,
    "amei": 2.2, "recomendo": 1.8, "satisfeito": 1.7, "satisfeita": 1.7, "rapido": 1.5,
    "rapida": 1.5, "antes": 1.0, "prazo": 0.6, "bom": 1.3, "boa": 1.3, "lindo": 1.6,
    "linda": 1.6, "parabens": 1.8, "chegou": 0.4, "embalado": 1.1, "qualidade": 0.8,
    "eficiente": 1.5, "pontual": 1.5, "original": 1.2, "super": 0.9, "nota": 0.6,
    "dez": 1.2, "top": 1.6, "muito_bom": 2.0, "antes_prazo": 2.0, "super_recomendo": 2.4,
    "entrega_rapida": 2.0, "bem_embalado": 1.7, "gostei": 1.6, "certinho": 1.6,
}

NEG_LEXICON = {
    "pessimo": -2.4, "pessima": -2.4, "horrivel": -2.4, "ruim": -1.8, "lixo": -2.5,
    "porcaria": -2.4, "atraso": -2.0, "atrasado": -2.0, "atrasada": -2.0, "atrasou": -2.0,
    "demorou": -1.7, "demora": -1.6, "quebrado": -2.3, "quebrada": -2.3, "defeito": -2.2,
    "danificado": -2.2, "errado": -2.1, "errada": -2.1, "diferente": -1.5, "faltou": -2.1,
    "faltando": -2.1, "incompleto": -2.0, "cancelado": -2.0, "estorno": -1.8, "reembolso": -1.8,
    "devolver": -1.7, "devolucao": -1.7, "enganosa": -2.4, "falsificado": -2.5, "fragil": -1.4,
    "decepcionado": -2.1, "decepcionada": -2.1, "decepcao": -2.1, "insatisfeito": -2.1,
    "insatisfeita": -2.1, "absurdo": -2.2, "vergonha": -2.2, "desrespeito": -2.3,
    "extraviado": -2.3, "nunca": -1.2, "pior": -2.2, "reclamacao": -1.8, "procon": -2.4,
    "nao_recebi": -2.8, "nao_chegou": -2.8, "nao_recomendo": -2.8, "nao_gostei": -2.4,
    "nao_funciona": -2.6, "nao_veio": -2.6, "nao_entregue": -2.7, "nao_entregaram": -2.7,
    "nao_cumpre": -2.3, "nao_responde": -2.3, "nao_serve": -2.2, "nao_vale": -2.2,
}

ASPECT_PATTERNS: Dict[str, re.Pattern[str]] = {
    "Atraso / Entrega": re.compile(
        r"\b(atras\w*|demor\w*|prazo|correio\w*|transportadora|extrav\w*|nao_recebi|nao_chegou|nao_entreg\w*|entrega)\b"
    ),
    "Defeito / Avaria": re.compile(
        r"\b(quebrad\w*|defeit\w*|danific\w*|amassad\w*|riscad\w*|estragad\w*|rasgad\w*|vazand\w*|nao_funciona|parou|trincad\w*)\b"
    ),
    "Produto Divergente / Errado": re.compile(
        r"\b(errad\w*|diferente|outro\s+produto|cor\s+diferente|tamanho\s+errad\w*|propaganda|enganos\w*|divergent\w*|trocad\w*|foto)\b"
    ),
    "Pedido Incompleto / Faltando": re.compile(
        r"\b(faltou|faltando|falta\s+de|incomplet\w*|metade|apenas\s+(1|um|uma)|veio\s+so|nao_veio|sem\s+o\s+|sem\s+a\s+)\b"
    ),
    "Baixa Qualidade / Frágil": re.compile(
        r"\b(fragil|qualidade|material\s+ruim|falsific\w*|replica|horrivel|lixo|porcaria|pessim\w*|fraco|fraca|fino)\b"
    ),
    "Atendimento / Estorno / Troca": re.compile(
        r"\b(estorn\w*|reembols\w*|cancel\w*|devolv\w*|devoluc\w*|troca\w*|atendimento|nao_responde|procon|dinheiro|contato|suporte)\b"
    ),
}

ASPECT_PLAYBOOK: Dict[str, str] = {
    "Atraso / Entrega": "Acionar rastreamento prioritário junto à transportadora, abrir protocolo de atraso e oferecer cupom de compensação ou reenvio expresso.",
    "Defeito / Avaria": "Liberar código de logística reversa imediata sem custo, priorizar troca expressa ou reembolso integral em até 48h e notificar controle de qualidade do seller.",
    "Produto Divergente / Errado": "Confirmar SKU expedido vs. anunciado, autorizar troca gratuita imediata e abrir auditoria no anúncio do vendedor para evitar recorrência.",
    "Pedido Incompleto / Faltando": "Conferir nota fiscal e peso expedido; despachar item faltante em frete expresso ou realizar estorno proporcional imediato.",
    "Baixa Qualidade / Frágil": "Oferecer devolução facilitada (direito de arrependimento / garantia) e registrar alerta de qualidade na curadoria do catálogo.",
    "Atendimento / Estorno / Troca": "Escalar ticket para fila N2 de resolução definitiva com confirmação formal do prazo de estorno/troca por e-mail e WhatsApp.",
}

NUMERIC_COLS = [
    "tempo_entrega_dias",
    "atraso_entrega_dias",
    "prazo_estimado_dias",
    "valor_total",
    "frete_total",
    "razao_frete",
    "qtd_itens",
    "qtd_vendedores",
    "comprimento_chars",
    "qtd_palavras",
    "tem_titulo",
    "entregue_com_atraso",
    "pedido_nao_entregue",
    "lexicon_score",
    "n_exclamacoes",
    "n_interrogacoes",
    "razao_maiusculas",
]

CATEGORICAL_COLS = ["categoria_principal", "customer_state", "order_status"]


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def clean_pt_text(text: str, preserve_negations: bool = True) -> str:
    """Normaliza texto em PT-BR preservando negações e criando bigramas de negação."""
    if not isinstance(text, str) or not text.strip():
        return ""
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\r|\n|\t", " ", s)
    # Normaliza letras exageradamente repetidas (ex.: otimoooo -> otimoo)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s_no_acc = strip_accents(s)
    tokens = re.findall(r"[a-z0-9]+", s_no_acc)

    if not preserve_negations:
        tokens = [t for t in tokens if t not in PT_STOPWORDS_SAFE and t not in NEGATION_WORDS]
        return " ".join(tokens)

    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATION_WORDS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            # Pula artigo curto entre "nao" e verbo/substantivo (ex.: "nao me entregaram" -> nao_entregaram)
            if nxt in {"me", "o", "a", "os", "as", "foi", "esta", "veio"} and i + 2 < len(tokens) and nxt != "veio":
                nxt2 = tokens[i + 2]
                out.append(f"{tok}_{nxt2}")
                i += 3
                continue
            out.append(f"{tok}_{nxt}")
            i += 2
            continue
        if tok not in PT_STOPWORDS_SAFE or tok in NEGATION_WORDS:
            out.append(tok)
        i += 1
    return " ".join(out)


def compute_lexicon_score(cleaned_text: str) -> float:
    """Calcula escore de polaridade léxico sensível a negações em PT-BR."""
    if not cleaned_text:
        return 0.0
    toks = cleaned_text.split()
    score = 0.0
    for idx, t in enumerate(toks):
        if t in NEG_LEXICON:
            score += NEG_LEXICON[t]
        elif t in POS_LEXICON:
            score += POS_LEXICON[t]
        elif t.startswith("nao_"):
            base = t[4:]
            if base in POS_LEXICON:
                score -= POS_LEXICON[base] * 1.15
            elif base in NEG_LEXICON:
                score += abs(NEG_LEXICON[base]) * 0.7
            else:
                score -= 0.45
        if idx + 1 < len(toks):
            bi = f"{t}_{toks[idx + 1]}"
            if bi in POS_LEXICON:
                score += POS_LEXICON[bi] * 0.8
            elif bi in NEG_LEXICON:
                score += NEG_LEXICON[bi] * 0.8
    return float(score)


def lexicon_predict_labels(scores: np.ndarray, neg_thr: float = -0.35, pos_thr: float = 0.75) -> np.ndarray:
    return np.select(
        [scores <= neg_thr, scores >= pos_thr],
        ["Negativo", "Positivo"],
        default="Neutro",
    )


def detect_aspects(cleaned_text: str) -> List[str]:
    if not cleaned_text:
        return ["Outros / Geral"]
    hits = [asp for asp, pat in ASPECT_PATTERNS.items() if pat.search(cleaned_text)]
    return hits if hits else ["Outros / Geral"]


def enrich_text_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona texto limpo com preservação de negações e sinais estilísticos básicos (sem vazamento)."""
    out = df.copy()
    raw_txt = out["texto_completo"].fillna("").astype(str)
    out["texto_limpo"] = raw_txt.map(lambda s: clean_pt_text(s, preserve_negations=True))
    out["lexicon_score"] = out["texto_limpo"].map(compute_lexicon_score)
    out["n_exclamacoes"] = raw_txt.str.count("!")
    out["n_interrogacoes"] = raw_txt.str.count(r"\?")
    upper_cnt = raw_txt.map(lambda s: sum(1 for ch in s if ch.isupper()))
    alpha_cnt = raw_txt.map(lambda s: max(1, sum(1 for ch in s if ch.isalpha())))
    out["razao_maiusculas"] = upper_cnt / alpha_cnt
    out["aspectos"] = out["texto_limpo"].map(detect_aspects)
    out["aspecto_principal"] = out["aspectos"].map(lambda lst: lst[0])
    return out


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split cronológico estrito (Treino -> Validação -> Teste Out-of-Time)."""
    ordered = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(ordered)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return (
        ordered.iloc[:i1].copy().reset_index(drop=True),
        ordered.iloc[i1:i2].copy().reset_index(drop=True),
        ordered.iloc[i2:].copy().reset_index(drop=True),
    )


def build_text_vectorizer() -> FeatureUnion:
    """União TF-IDF Palavras (1–2 gramas) + Subpalavras Char (3–5 gramas)."""
    return FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.95,
                max_features=30000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b[a-z0-9_]{2,}\b",
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=5,
                max_df=0.95,
                max_features=25000,
                sublinear_tf=True,
            ),
        ),
    ])


def build_tabular_preprocessor() -> ColumnTransformer:
    """Pré-processamento tabular livre de leakage (imputação + indicador de missing + escala + OHE)."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="nao_informada")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=25)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ],
        sparse_threshold=0.3,
    )


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """Calcula métricas multiclasse (Macro-F1, Weighted-F1, Acc) e binárias para Detrator (Negativo)."""
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    y_true_det = (y_true == "Negativo").astype(int)
    y_pred_det = (y_pred == "Negativo").astype(int)
    metrics["detrator_f1"] = float(f1_score(y_true_det, y_pred_det, zero_division=0))
    metrics["detrator_recall"] = float(recall_score(y_true_det, y_pred_det, zero_division=0))
    metrics["detrator_precision"] = float(precision_score(y_true_det, y_pred_det, zero_division=0))

    if y_proba is not None:
        # Assume ordem CLASSES_ORDER = ["Negativo", "Neutro", "Positivo"] -> coluna 0 = Negativo
        p_det = y_proba[:, 0]
        metrics["detrator_roc_auc"] = float(roc_auc_score(y_true_det, p_det))
        metrics["detrator_pr_auc"] = float(average_precision_score(y_true_det, p_det))
        y_true_idx = np.array([CLASSES_ORDER.index(c) for c in y_true])
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="macro")
        )
    else:
        metrics["detrator_roc_auc"] = float("nan")
        metrics["detrator_pr_auc"] = float("nan")
        metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_det: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Intervalos de confiança bootstrap (95%) para Macro-F1 e PR-AUC de Detratores."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_boot, dtype=float)
    prs = np.empty(n_boot, dtype=float)
    y_det = (y_true == "Negativo").astype(int)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s[b] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        if y_proba_det is not None:
            prs[b] = average_precision_score(y_det[idx], y_proba_det[idx])

    out: Dict[str, Tuple[float, float]] = {
        "f1_macro_ci95": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
    }
    if y_proba_det is not None:
        out["detrator_pr_auc_ci95"] = (float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5)))
    return out


def evaluate_slices(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_det: np.ndarray,
) -> pd.DataFrame:
    """Análise de erros por fatia (Slice-Based Error Analysis) exigida em produção."""
    work = df_test.copy()
    work["y_true"] = y_true
    work["y_pred"] = y_pred
    work["p_det"] = p_det

    work["fatia_logistica"] = np.select(
        [work["pedido_nao_entregue"] == 1, work["entregue_com_atraso"] == 1],
        ["Não Entregue", "Entregue com Atraso"],
        default="Entregue no Prazo",
    )
    work["fatia_tamanho"] = pd.cut(
        work["qtd_palavras"],
        bins=[-1, 5, 18, 10000],
        labels=["Curto (≤5 pal.)", "Médio (6–18 pal.)", "Longo (>18 pal.)"],
    ).astype(str)

    rows = []
    for dim, col in [
        ("Status Logístico", "fatia_logistica"),
        ("Tamanho do Review", "fatia_tamanho"),
        ("Macro-Região", "regiao_br"),
    ]:
        for val, grp in work.groupby(col):
            if len(grp) < 40:
                continue
            yt = grp["y_true"].values
            yp = grp["y_pred"].values
            y_d = (yt == "Negativo").astype(int)
            yp_d = (yp == "Negativo").astype(int)
            rows.append({
                "Dimensão": dim,
                "Fatia": str(val),
                "N": int(len(grp)),
                "% Detrator Real": round(float(y_d.mean() * 100), 1),
                "Accuracy": round(float(accuracy_score(yt, yp)), 4),
                "Macro-F1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
                "Recall Detrator": round(float(recall_score(y_d, yp_d, zero_division=0)), 4),
                "PR-AUC Detrator": round(float(average_precision_score(y_d, grp["p_det"].values)), 4)
                if len(np.unique(y_d)) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


class CustomerSupportRAG:
    """Copiloto de Atendimento RAG Determinístico (Zero Alucinação) para Triagem SAC Olist.

    Recupera avaliações históricas semanticamente similares via TF-IDF cosseno no conjunto de
    Treino e combina contexto logístico real do pedido + aspecto detectado + playbook operacional
    para gerar rascunhos de resposta empáticos e auditáveis em < 5 ms.
    """

    def __init__(self, vectorizer: FeatureUnion, kb_df: pd.DataFrame, kb_matrix: sparse.csr_matrix):
        self.vectorizer = vectorizer
        self.kb_df = kb_df.reset_index(drop=True)
        self.kb_matrix = normalize(kb_matrix, norm="l2", copy=True)

    def assist(
        self,
        review_text: str,
        order_context: Dict[str, Any],
        pred_label: str,
        prob_detrator: float,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        cleaned = clean_pt_text(review_text, preserve_negations=True)
        aspects = detect_aspects(cleaned) if pred_label != "Positivo" else ["Experiência Positiva / Entrega OK"]
        main_aspect = aspects[0]

        q_vec = normalize(self.vectorizer.transform([cleaned]), norm="l2", copy=True)
        sims = (self.kb_matrix @ q_vec.T).toarray().ravel()
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        similar_cases = []
        for idx in top_idx:
            row = self.kb_df.iloc[int(idx)]
            similar_cases.append({
                "similarity": round(float(np.clip(sims[int(idx)], 0.0, 1.0)), 3),
                "review_score": int(row["review_score"]),
                "categoria": str(row["categoria_principal"]),
                "atraso_dias": round(float(row["atraso_entrega_dias"]), 1)
                if pd.notna(row["atraso_entrega_dias"]) else None,
                "texto": str(row["texto_completo"])[:180],
            })

        # Prioridade de SLA baseada na probabilidade calibrada de Detrator e severidade operacional
        atraso = order_context.get("atraso_entrega_dias")
        valor = order_context.get("valor_total", 0.0)
        if pred_label != "Positivo" and (prob_detrator >= 0.80 or (atraso is not None and atraso > 5)):
            prioridade = "ALTA (SLA 2h — Retenção Crítica)"
        elif pred_label != "Positivo" and prob_detrator >= 0.45:
            prioridade = "MÉDIA (SLA 12h — Acompanhamento Ativo)"
        else:
            prioridade = "BAIXA (Encantamento / Pós-Venda)"

        if pred_label == "Positivo":
            acao = "Registrar elogio no índice NPS do vendedor parceiro e acionar fluxo automático de fidelização."
        else:
            acao = ASPECT_PLAYBOOK.get(
                main_aspect,
                "Agradecer o feedback, confirmar detalhes da ocorrência e acionar suporte humanizado.",
            )

        categoria = str(order_context.get("categoria_principal", "produto")).replace("_", " ")
        if pred_label == "Negativo":
            atraso_msg = (
                f"Identificamos que houve um atraso de {atraso:.0f} dias na entrega do seu pedido de {categoria}. "
                if (atraso is not None and atraso > 0)
                else f"Lamentamos sinceramente o transtorno relatado com seu pedido de {categoria}. "
            )
            draft = (
                f"Olá! Agradecemos por nos reportar sua experiência. {atraso_msg}"
                f"Nossa equipe já abriu prioridade para tratar a ocorrência ({main_aspect}) — "
                f"próximo passo imediato: {acao}"
            )
        elif pred_label == "Neutro":
            draft = (
                f"Olá! Obrigado por avaliar sua compra de {categoria}. Notamos pontos de melhoria "
                f"relacionados a '{main_aspect}'. Caso precise de qualquer ajuste ou suporte adicional, "
                "estamos à disposição no canal de atendimento."
            )
        else:
            draft = (
                f"Olá! Ficamos muito felizes em saber da sua ótima experiência com seu pedido de {categoria}. "
                "Seu feedback positivo foi compartilhado com o vendedor parceiro. Conte sempre conosco!"
            )

        return {
            "pred_label": pred_label,
            "prob_detrator": round(float(prob_detrator), 4),
            "prioridade_sac": prioridade,
            "aspectos_detectados": aspects,
            "acao_recomendada": acao,
            "resposta_sugerida": draft,
            "casos_similares": similar_cases,
            "valor_pedido": round(float(valor), 2) if pd.notna(valor) else 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
