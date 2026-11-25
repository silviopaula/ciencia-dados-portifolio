"""Motor modular de NLP Financeiro e Política Monetária para Atas do COPOM (em inglês).

Implementa sem dependências obsoletas:
1. Pré-processamento textual limpo de PDFs oficiais do BCB (remoção de cabeçalhos/rodapés);
2. Índice de Sentimento Financeiro (estilo Loughran–McDonald: Positivo, Negativo e Incerteza)
   com tratamento explícito de janela de negação;
3. Índice Hawkish–Dovish específico para comunicação de Bancos Centrais (unigramas + bigramas
   de postura monetária: aperto / pressões inflacionárias vs. afrouxamento / desinflação);
4. Modelagem de tópicos via TF-IDF + NMF;
5. Mapeamento oficial de presidências do Banco Central do Brasil (sem lacunas de data).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================================================================
# 1. Léxico Financeiro estilo Loughran–McDonald (núcleo de alta relevância macro)
# =============================================================================
LM_POSITIVE = {
    "achieve", "achieved", "achieving", "advance", "advanced", "advancing",
    "beneficial", "benefit", "benefits", "better", "boost", "boosted", "buoyant",
    "calm", "calmer", "confidence", "confident", "constructive", "contribute",
    "efficiency", "efficient", "encourage", "encouraging", "enhance", "enhanced",
    "expand", "expanded", "expanding", "expansion", "favorable", "favourable",
    "gain", "gained", "gains", "good", "grow", "growing", "growth", "healthy",
    "improve", "improved", "improvement", "improvements", "improving", "mitigate",
    "mitigated", "normalization", "normalize", "normalized", "opportunities",
    "opportunity", "optimism", "optimistic", "outperform", "positive", "positively",
    "progress", "progressed", "rebound", "rebounded", "recover", "recovered",
    "recovering", "recovery", "resilience", "resilient", "robust", "solid",
    "sound", "stability", "stabilization", "stabilize", "stabilized", "stable",
    "strength", "strengthen", "strengthened", "strengthening", "strong", "stronger",
    "success", "successful", "supportive", "upturn", "well",
}

LM_NEGATIVE = {
    "adverse", "adversely", "aggravate", "aggravated", "concern", "concerned",
    "concerning", "concerns", "constrain", "constrained", "constraints", "contract",
    "contracted", "contracting", "contraction", "crisis", "critical", "damage",
    "decelerate", "decelerated", "decelerating", "deceleration", "decline",
    "declined", "declining", "default", "deficit", "deficits", "delay", "delayed",
    "deteriorate", "deteriorated", "deteriorating", "deterioration", "difficult",
    "difficulties", "difficulty", "disrupt", "disrupted", "disruption", "disruptions",
    "downturn", "drop", "dropped", "erosion", "excessive", "fail", "failure",
    "fall", "fallen", "falling", "fear", "fragile", "fragility", "frustration",
    "halt", "halted", "headwinds", "hurt", "imbalance", "imbalances", "impair",
    "impaired", "instability", "insufficient", "loss", "losses", "negative",
    "negatively", "obstacle", "obstacles", "pessimism", "pessimistic", "plunge",
    "poor", "pressure", "pressured", "pressures", "problem", "problems", "recession",
    "recessionary", "restrict", "restricted", "restriction", "restrictions", "risk",
    "risks", "risky", "severe", "severely", "shock", "shocks", "shortage",
    "shortages", "shrink", "shrinking", "slowdown", "sluggish", "slump", "stress",
    "stresses", "suffer", "suffered", "tense", "tension", "tensions", "threat",
    "turbulence", "turmoil", "unable", "unanchored", "unemployment", "unfavorable",
    "unfavourable", "unstable", "vulnerability", "vulnerabilities", "vulnerable",
    "weak", "weaken", "weakened", "weakening", "weakness", "worse", "worsen",
    "worsened", "worsening",
}

LM_UNCERTAINTY = {
    "ambiguous", "approximate", "approximately", "assume", "assumption", "assumptions",
    "caution", "cautious", "cautiously", "conditional", "contingency", "depend",
    "dependent", "depending", "depends", "doubt", "doubts", "estimate", "estimated",
    "fluctuate", "fluctuation", "fluctuations", "hesitation", "imprecise",
    "indefinite", "instability", "may", "might", "pending", "perhaps", "possible",
    "possibly", "precaution", "precautionary", "predict", "preliminary", "probable",
    "probably", "project", "projected", "projection", "projections", "reassess",
    "risk", "risks", "scenario", "scenarios", "speculative", "tentative", "uncertain",
    "uncertainties", "uncertainty", "unclear", "unpredictable", "variable", "volatile",
    "volatility",
}

# =============================================================================
# 2. Léxico Especializado Hawkish–Dovish (Comunicação de Bancos Centrais)
# =============================================================================
# Hawkish (+): postura contracionista, alerta inflacionário, juros altos/aperto
HAWKISH_UNIGRAMS = {
    "tighten", "tightened", "tightening", "restrictive", "contractionary",
    "hike", "hikes", "hiked", "hiking", "overheating", "overheated",
    "deanchoring", "unanchored", "deanchored", "reanchoring", "persistence",
    "persistent", "resilient", "resilience", "upside", "hawkish", "vigilant",
    "vigilance", "perseverance", "firmness", "serene", "caution", "cautious",
    "pressures", "inflationary", "acceleration", "accelerating", "above",
    "elevated", "tight", "tightness", "dynamism", "heated", "passthrough",
    "deteriorated", "deterioration", "fiscal", "depreciation",
}

HAWKISH_PHRASES = [
    "upward pressure", "upward pressures", "upside risk", "upside risks",
    "inflationary pressure", "inflationary pressures", "inflation expectations",
    "deanchoring of inflation", "unanchored expectations", "higher for longer",
    "contractionary territory", "significantly contractionary", "restrictive territory",
    "tight labor market", "tighter financial", "additional adjustment",
    "further adjustment", "increase in the selic", "raise the selic",
    "hiking cycle", "tightening cycle", "monetary tightening", "prolonged period",
    "above the target", "above target", "fiscal risk", "fiscal risks",
    "asymmetric balance of risks", "upward asymmetry", "services inflation",
    "core inflation", "output gap", "positive output gap", "resilient activity",
]

# Dovish (-): postura expansionista, desinflação, cortes de juros, ociosidade
DOVISH_UNIGRAMS = {
    "easing", "ease", "eased", "accommodative", "stimulative", "stimulus",
    "cut", "cuts", "cutting", "reduction", "reductions", "reduce", "reduced",
    "disinflation", "disinflationary", "deflation", "benign", "moderation",
    "moderating", "moderate", "slack", "idleness", "spare", "subdued",
    "downside", "dovish", "gradual", "flexibility", "anchored", "convergence",
    "converging", "deceleration", "decelerating", "slowdown", "cooling",
    "softening", "weakness", "weakening", "recession", "contraction", "below",
    "lower", "fall", "falling", "decline", "declining", "normalizing",
}

DOVISH_PHRASES = [
    "downside risk", "downside risks", "disinflation process", "disinflationary process",
    "monetary easing", "easing cycle", "reduction of the selic", "cut in the selic",
    "pace of reduction", "parsimonious pace", "economic slack", "idleness in the economy",
    "negative output gap", "below the target", "below target", "converging to the target",
    "well anchored", "benign inflation", "benign scenario", "moderation in activity",
    "activity slowdown", "gradual recovery", "accommodative monetary", "stimulative monetary",
    "lower inflation", "inflation deceleration", "subdued inflation", "spare capacity",
    "credit slowdown", "credit moderation",
]

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "less", "hardly", "barely"}


def clean_copom_text(raw_text: str) -> str:
    """Remove ruídos comuns de extração de PDF das atas oficiais do BCB."""
    if not isinstance(raw_text, str):
        return ""
    text = raw_text.replace("\xa0", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\bbcb\.gov\.br\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_english(text: str) -> List[str]:
    """Tokeniza texto em palavras alfabéticas em minúsculas (>= 2 letras)."""
    return re.findall(r"[a-z]{2,}", text.lower())


def _score_lexicon_with_negation(
    tokens: List[str],
    pos_set: set[str],
    neg_set: set[str],
    negation_window: int = 3,
) -> Tuple[int, int, Dict[str, int], Dict[str, int]]:
    """Conta ocorrências de dois polos invertendo polaridade quando precedidos por negação."""
    pos_count = 0
    neg_count = 0
    pos_hits: Dict[str, int] = {}
    neg_hits: Dict[str, int] = {}

    for idx, tok in enumerate(tokens):
        window = tokens[max(0, idx - negation_window) : idx]
        is_negated = any(w in NEGATION_WORDS for w in window)

        if tok in pos_set:
            if is_negated:
                neg_count += 1
                neg_hits[f"not_{tok}"] = neg_hits.get(f"not_{tok}", 0) + 1
            else:
                pos_count += 1
                pos_hits[tok] = pos_hits.get(tok, 0) + 1
        elif tok in neg_set:
            if is_negated:
                pos_count += 1
                pos_hits[f"not_{tok}"] = pos_hits.get(f"not_{tok}", 0) + 1
            else:
                neg_count += 1
                neg_hits[tok] = neg_hits.get(tok, 0) + 1

    return pos_count, neg_count, pos_hits, neg_hits


def score_copom_minute(raw_text: str) -> Dict[str, float]:
    """Calcula métricas NLP completas para uma ata do COPOM:
    - Sentimento Financeiro (Polaridade LM [-1, +1]) e Índice de Incerteza (%)
    - Índice Hawkish–Dovish [-1, +1] combinando unigramas (peso 1) e bigramas/frases (peso 2.5)
    - Métricas estruturais (n_palavras, diversidade lexical)
    """
    cleaned = clean_copom_text(raw_text)
    lower = cleaned.lower()
    tokens = tokenize_english(cleaned)
    n_tokens = max(len(tokens), 1)

    # 1. Sentimento Financeiro (estilo Loughran–McDonald com janela de negação)
    lm_pos, lm_neg, _, _ = _score_lexicon_with_negation(tokens, LM_POSITIVE, LM_NEGATIVE)
    lm_unc = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    lm_polarity = (lm_pos - lm_neg) / max(lm_pos + lm_neg, 1)
    lm_uncertainty_pct = (lm_unc / n_tokens) * 100.0

    # 2. Índice Hawkish–Dovish (unigramas com negação + frases de política monetária)
    hk_uni, dv_uni, _, _ = _score_lexicon_with_negation(tokens, HAWKISH_UNIGRAMS, DOVISH_UNIGRAMS)
    hk_phr = sum(lower.count(p) for p in HAWKISH_PHRASES)
    dv_phr = sum(lower.count(p) for p in DOVISH_PHRASES)

    hk_score = hk_uni + 2.5 * hk_phr
    dv_score = dv_uni + 2.5 * dv_phr
    hawkish_index = (hk_score - dv_score) / max(hk_score + dv_score, 1.0)

    return {
        "n_palavras": int(len(tokens)),
        "vocabulario_unico": int(len(set(tokens))),
        "lm_positivas": int(lm_pos),
        "lm_negativas": int(lm_neg),
        "sentimento_lm": round(float(lm_polarity), 4),
        "incerteza_pct": round(float(lm_uncertainty_pct), 4),
        "hawkish_score": round(float(hk_score), 2),
        "dovish_score": round(float(dv_score), 2),
        "indice_hawkish": round(float(hawkish_index), 4),
    }


def extract_top_policy_terms(texts: pd.Series, top_n: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna os bigramas/frases Hawkish e Dovish mais frequentes no corpus."""
    combined = " ".join(clean_copom_text(t).lower() for t in texts)
    hk_counts = [(p, combined.count(p)) for p in HAWKISH_PHRASES]
    dv_counts = [(p, combined.count(p)) for p in DOVISH_PHRASES]

    df_hk = pd.DataFrame(hk_counts, columns=["expressao", "frequencia"]).sort_values("frequencia", ascending=False).head(top_n)
    df_dv = pd.DataFrame(dv_counts, columns=["expressao", "frequencia"]).sort_values("frequencia", ascending=False).head(top_n)
    return df_hk.reset_index(drop=True), df_dv.reset_index(drop=True)


def fit_copom_topics(texts: pd.Series, n_topics: int = 5, random_state: int = 42) -> Tuple[pd.DataFrame, List[Dict]]:
    """Extrai tópicos latentes das atas do COPOM via TF-IDF + NMF."""
    cleaned = [clean_copom_text(t) for t in texts]
    custom_stops = list(
        TfidfVectorizer(stop_words="english").get_stop_words()
        | {
            "copom", "meeting", "minutes", "committee", "members", "bcb", "bank",
            "central", "brazil", "brazilian", "percent", "percentage", "points",
            "period", "month", "months", "quarter", "year", "years", "data",
            "noted", "assessed", "highlighted", "discussed", "considered",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "head", "deputy", "attending", "office", "department", "division",
            "governor", "director", "banco", "brasil", "interno", "secretariat",
            "press", "advisor", "advisors", "staff", "session", "minutes",
            "page", "public", "confidential", "restricted",
        }
    )
    vec = TfidfVectorizer(
        max_df=0.85,
        min_df=3,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        stop_words=custom_stops,
        max_features=1200,
    )
    X_tfidf = vec.fit_transform(cleaned)
    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X_tfidf)
    W_norm = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)

    feature_names = np.array(vec.get_feature_names_out())
    topic_meta = []
    for k in range(n_topics):
        top_idx = np.argsort(nmf.components_[k])[::-1][:8]
        words = feature_names[top_idx].tolist()
        topic_meta.append({"topic_id": k, "top_terms": words})

    df_W = pd.DataFrame(W_norm, columns=[f"topico_{k+1}" for k in range(n_topics)])
    return df_W, topic_meta


def map_bcb_president(date_series: pd.Series) -> pd.Series:
    """Mapeia rigorosamente a presidência do BCB na data de cada reunião do COPOM (sem lacunas)."""
    dt = pd.to_datetime(date_series)
    pres = pd.Series("Desconhecido", index=dt.index, dtype="object")
    pres[dt < pd.Timestamp("2016-06-09")] = "Alexandre Tombini"
    pres[(dt >= pd.Timestamp("2016-06-09")) & (dt < pd.Timestamp("2019-02-28"))] = "Ilan Goldfajn"
    pres[(dt >= pd.Timestamp("2019-02-28")) & (dt < pd.Timestamp("2025-01-01"))] = "Roberto Campos Neto"
    pres[dt >= pd.Timestamp("2025-01-01")] = "Gabriel Galípolo"
    return pres
