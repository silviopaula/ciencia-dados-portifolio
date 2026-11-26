"""Motor RAG Híbrido, Transparente e Didático para Artigos Científicos em PDF.

Arquitetura em 6 etapas observáveis:
1. Ingestão PDF por página (pypdf) com limpeza tipográfica (normalização NFKC de
   ligaduras LaTeX como ﬁ/ﬀ/ﬂ e des-hifenização entre quebras de linha).
2. Chunking consciente de sentenças com overlap configurável e rastreamento exato
   da zona de sobreposição (`overlap_prefix_chars`).
3. Indexação Dupla:
   - Canal Lexical: Okapi BM25 (k1=1.5, b=0.75) com Ponte Bilíngue PT <-> EN.
   - Canal Semântico: Vetores Densos LSA (TF-IDF + TruncatedSVD) combinados com
     vetores subword (character n-grams 3..5) para cognatos multilíngues.
4. Recuperação Híbrida via fusão ponderada + Reciprocal Rank Fusion (RRF) e
   diversificação opcional via Maximal Marginal Relevance (MMR).
5. Montagem explícita de Prompt Grounded com citações numeradas [1]..[k] e
   guardrail anti-alucinação para consultas fora do escopo.
6. Geração Multi-Provedor: Extrativo Grounded Offline (zero dependências externas),
   Ollama Local, OpenAI API ou Google Gemini API.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .schemas import (
    ChunkConfig,
    ChunkHistogramBin,
    ChunkItem,
    ChunksPageResponse,
    DocumentSummary,
    IndexStats,
    LibraryResponse,
    PromptInspection,
    ProviderInfo,
    ProvidersResponse,
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SuggestedQuestion,
)

# ---------------------------------------------------------------------------
# Stopwords bilíngues (PT + EN) & Ponte Bilíngue Acadêmica (PT -> EN)
# ---------------------------------------------------------------------------

STOPWORDS_PT_EN = {
    # Português
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do",
    "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa",
    "essas", "esse", "esses", "esta", "estao", "estas", "estava", "este", "estes",
    "estou", "eu", "foi", "fomos", "foram", "ha", "isso", "isto", "ja", "lhe", "lhes",
    "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nao",
    "nas", "nem", "no", "nos", "nossa", "nosso", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando", "que", "quem",
    "sao", "se", "seja", "sem", "ser", "seu", "seus", "so", "sobre", "somos", "sua",
    "suas", "tambem", "tem", "ter", "teu", "toda", "todas", "todo", "todos", "tu", "um",
    "uma", "umas", "uns", "voce", "voces", "porque", "quais", "explique", "descreva",
    "resuma", "diga", "fale", "mostre", "qual", "artigo", "artigos", "paper", "papers",
    "estudo", "estudos", "texto", "pdf", "pdfs", "autores", "autor", "segundo",
    # Inglês
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have",
    "has", "had", "not", "but", "what", "which", "who", "whom", "when", "where", "why",
    "how", "can", "could", "should", "would", "may", "might", "will", "shall", "into",
    "onto", "upon", "about", "above", "below", "between", "through", "during", "before",
    "after", "each", "other", "such", "only", "same", "than", "then", "there", "these",
    "those", "their", "they", "them", "its", "our", "your", "his", "her", "also", "been",
    "being", "does", "did", "doing", "done", "more", "most", "very", "some", "any", "all",
    "both", "few", "many", "much", "own", "here", "over", "under", "again", "further",
    "once", "just", "now", "use", "used", "using", "paper", "article", "study", "authors",
}

PT_INDICATOR_WORDS = {
    "como", "qual", "quais", "porque", "para", "pelo", "pela", "sobre", "entre", "diferenca",
    "tendencia", "tendencias", "tratamento", "periodo", "periodos", "efeito", "efeitos",
    "vies", "estimador", "hipotese", "hipoteses", "artigo", "artigos", "metodo", "metodos",
    "pesos", "negativos", "continuo", "escalonado", "multiplos", "problema", "explica",
    "explique", "quais", "quando", "nao", "sao", "com", "dos", "das", "nos", "nas",
}

# Frases e termos frequentes em pesquisa científica / econometria / ML (PT -> EN)
BILINGUAL_PHRASE_MAP: List[Tuple[str, List[str]]] = [
    ("tendencia paralela forte", ["strong", "parallel", "trends", "spt"]),
    ("tendencias paralelas fortes", ["strong", "parallel", "trends", "spt"]),
    ("tendencia paralela", ["parallel", "trends", "assumption"]),
    ("tendencias paralelas", ["parallel", "trends", "assumption"]),
    ("tratamento continuo", ["continuous", "treatment", "dose"]),
    ("multiplos periodos", ["multiple", "time", "periods", "staggered"]),
    ("adocao escalonada", ["staggered", "adoption", "timing", "variation"]),
    ("efeito medio do tratamento", ["average", "treatment", "effect", "att"]),
    ("efeito causal", ["causal", "response", "treatment", "effect", "acr", "acrt"]),
    ("vies de selecao", ["selection", "bias"]),
    ("pesos negativos", ["negative", "weights", "twfe"]),
    ("efeitos fixos bidirecionais", ["two", "way", "fixed", "effects", "twfe"]),
    ("grupo tempo", ["group", "time", "att"]),
    ("estudo de evento", ["event", "study", "dynamic", "aggregation"]),
    ("duplamente robusto", ["doubly", "robust", "dr"]),
    ("escore de propensao", ["propensity", "score", "ipw"]),
    ("salario minimo", ["minimum", "wage", "employment", "teen"]),
    ("dose resposta", ["dose", "response", "att", "acr"]),
    ("bandas de confianca", ["simultaneous", "confidence", "bands", "bootstrap"]),
    ("aplicacao empirica", ["empirical", "application"]),
    ("analise de dados", ["data", "analysis", "empirical"]),
    ("revisao de literatura", ["literature", "review"]),
    ("aprendizado de maquina", ["machine", "learning"]),
    ("series temporais", ["time", "series"]),
    ("regressao linear", ["linear", "regression", "ols"]),
]

BILINGUAL_TOKEN_MAP: Dict[str, List[str]] = {
    "tendencia": ["trend", "trends"],
    "tendencias": ["trends", "trend"],
    "paralela": ["parallel"],
    "paralelas": ["parallel"],
    "forte": ["strong"],
    "fraca": ["weak"],
    "tratamento": ["treatment", "treated"],
    "tratados": ["treated"],
    "controle": ["control", "untreated", "comparison"],
    "controles": ["control", "untreated", "comparison"],
    "continuo": ["continuous", "dose"],
    "continua": ["continuous", "dose"],
    "binario": ["binary"],
    "escalonado": ["staggered", "timing"],
    "escalonada": ["staggered", "timing"],
    "multiplos": ["multiple"],
    "periodo": ["period", "time"],
    "periodos": ["periods", "time"],
    "efeito": ["effect", "att", "acr"],
    "efeitos": ["effects", "att", "acrt"],
    "medio": ["average"],
    "media": ["average", "mean"],
    "vies": ["bias", "biased"],
    "selecao": ["selection"],
    "pesos": ["weights", "weighting"],
    "peso": ["weight", "weights"],
    "negativos": ["negative"],
    "negativo": ["negative"],
    "estimador": ["estimator", "estimation"],
    "estimadores": ["estimators", "estimation"],
    "estimacao": ["estimation", "estimator"],
    "identificacao": ["identification", "identified"],
    "hipotese": ["assumption", "hypothesis"],
    "hipoteses": ["assumptions"],
    "suposicao": ["assumption"],
    "suposicoes": ["assumptions"],
    "heterogeneidade": ["heterogeneity", "heterogeneous"],
    "heterogeneo": ["heterogeneous"],
    "heterogeneos": ["heterogeneous"],
    "dinamico": ["dynamic", "event"],
    "dinamicos": ["dynamic", "event"],
    "agregacao": ["aggregation", "aggregate", "summary"],
    "agregado": ["aggregate", "aggregated"],
    "covariaveis": ["covariates", "conditioning"],
    "covariavel": ["covariate"],
    "inferencia": ["inference", "bootstrap"],
    "contrafactual": ["counterfactual", "untreated"],
    "resultado": ["outcome", "results"],
    "resultados": ["results", "outcomes", "findings"],
    "diferenca": ["difference", "differences"],
    "diferencas": ["differences", "did"],
    "teorema": ["theorem", "proposition"],
    "proposicao": ["proposition"],
    "corolario": ["corollary"],
    "simulacao": ["simulation", "monte", "carlo"],
    "simulacoes": ["simulations", "monte", "carlo"],
    "empirica": ["empirical", "application"],
    "empirico": ["empirical", "application"],
    "salario": ["wage", "wages"],
    "minimo": ["minimum"],
    "emprego": ["employment"],
    "hospital": ["hospital", "medicare"],
    "hospitais": ["hospitals", "medicare"],
    "dose": ["dose", "doses", "intensity"],
    "doses": ["doses", "dose"],
    "intensidade": ["intensity", "dose"],
    "grupo": ["group", "cohort"],
    "grupos": ["groups", "cohorts"],
    "coorte": ["cohort", "group"],
    "coortes": ["cohorts", "groups"],
    "nunca": ["never", "untreated"],
    "ainda": ["not", "yet", "treated"],
    "antecipacao": ["anticipation"],
    "limitacao": ["limitation", "caveat"],
    "limitacoes": ["limitations", "caveats"],
    "conclusao": ["conclusion", "conclude"],
    "resumo": ["abstract", "summary"],
    "contribuicao": ["contribution", "contributions"],
    "contribuicoes": ["contributions", "contribution"],
    "metodologia": ["methodology", "method", "framework"],
    "metodo": ["method", "estimator"],
    "equacao": ["equation"],
    "amostra": ["sample"],
    "dados": ["data", "dataset"],
    "problema": ["problem", "issue", "drawback", "limitation"],
    "problemas": ["problems", "issues", "drawbacks"],
    "comparacao": ["comparison", "compare"],
    "comparar": ["compare", "comparison"],
    "vantagem": ["advantage", "benefit"],
    "vantagens": ["advantages", "benefits"],
}


# ---------------------------------------------------------------------------
# Utilitários de limpeza de texto e tokenização
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Remove acentos mantendo caracteres ASCII limpos para matching lexical."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def clean_pdf_text(raw_text: str) -> str:
    """Normaliza texto extraído de PDFs científicos (especialmente gerados em LaTeX).

    Resolve:
    - Ligaduras tipográficas Unicode (ﬁ -> fi, ﬀ -> ff, ﬂ -> fl, ﬃ -> ffi) via NFKC.
    - Soft hyphens (\xad) e hifenizações de fim de linha (`identi-\\nfication` -> `identification`).
    - Espaçamentos múltiplos preservando quebras de parágrafo.
    """
    if not raw_text:
        return ""
    text = unicodedata.normalize("NFKC", raw_text)
    text = text.replace("\xad", "").replace("\u200b", "").replace("\r\n", "\n").replace("\r", "\n")
    # Junta palavras quebradas por hífen no fim de linha
    text = re.sub(r"([A-Za-zÀ-ÿ])-\s*\n\s*([a-zà-ÿ])", r"\1\2", text)
    # Normaliza espaços em cada linha e colapsa quebras excessivas
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_paragraphs: List[str] = []
    buffer: List[str] = []
    for line in lines:
        if not line:
            if buffer:
                cleaned_paragraphs.append(" ".join(buffer))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        cleaned_paragraphs.append(" ".join(buffer))
    return "\n\n".join(cleaned_paragraphs).strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokeniza texto preservando siglas científicas (ex.: DiD, TWFE, ATT, ACR, ACRT)."""
    normalized = strip_accents(text.lower())
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized)
    tokens: List[str] = []
    for tok in raw_tokens:
        if len(tok) < 2:
            continue
        if tok.isdigit() and len(tok) < 4:
            continue
        if remove_stopwords and tok in STOPWORDS_PT_EN:
            continue
        tokens.append(tok)
        # Se for composto com hífen (ex.: difference-in-differences), inclui partes úteis
        if "-" in tok:
            for part in tok.split("-"):
                if len(part) >= 2 and (not remove_stopwords or part not in STOPWORDS_PT_EN):
                    tokens.append(part)
    return tokens


def detect_query_language(question: str) -> str:
    """Detecta se a consulta está em português ou inglês."""
    has_pt_accents = bool(re.search(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]", question))
    plain_words = set(re.findall(r"[a-z]+", strip_accents(question.lower())))
    pt_hits = len(plain_words & PT_INDICATOR_WORDS)
    if has_pt_accents or pt_hits >= 2:
        return "pt-BR"
    return "en"


def expand_query_bilingual(question: str) -> Tuple[List[str], List[str], str]:
    """Extrai tokens originais e termos expandidos via Ponte Bilíngue PT <-> EN."""
    lang = detect_query_language(question)
    original_tokens = tokenize(question, remove_stopwords=True)
    expanded: List[str] = []
    seen = set(original_tokens)

    plain_q = " ".join(re.findall(r"[a-z0-9]+", strip_accents(question.lower())))
    # 1. Expansão por sintagmas/frases técnicas
    for phrase, targets in BILINGUAL_PHRASE_MAP:
        if phrase in plain_q:
            for t in targets:
                if t not in seen:
                    expanded.append(t)
                    seen.add(t)

    # 2. Expansão por tokens individuais
    for tok in original_tokens:
        if tok in BILINGUAL_TOKEN_MAP:
            for t in BILINGUAL_TOKEN_MAP[tok]:
                if t not in seen:
                    expanded.append(t)
                    seen.add(t)

    # Deduplica original_tokens mantendo ordem
    unique_orig = list(dict.fromkeys(original_tokens))
    return unique_orig, expanded, lang


def infer_paper_metadata(filename: str, pages_text: List[str], pdf_meta: Optional[dict]) -> Tuple[str, str, Optional[str]]:
    """Infere título limpo, autores e ano a partir do nome do arquivo, metadados PDF e 1ª página."""
    stem = Path(filename).stem
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", filename)
    year = year_match.group(1) if year_match else None

    # Padrão frequente: "(Autores) Título (Ano).pdf"
    authors = "Autores não identificados"
    title = stem

    paren_match = re.match(r"^\(([^)]+)\)\s*(.+)$", stem)
    if paren_match:
        raw_authors = re.sub(r"\s+", " ", paren_match.group(1)).strip()
        raw_title = re.sub(r"\(\s*(?:19|20)\d{2}\s*\)", "", paren_match.group(2)).strip()
        # Extrai ano dos parênteses dos autores se não havia antes
        if not year:
            ym = re.search(r"\b(19\d{2}|20\d{2})\b", raw_authors)
            if ym:
                year = ym.group(1)
        clean_authors = re.sub(r"\b(19|20)\d{2}\b", "", raw_authors).strip(" ,;-")
        if clean_authors:
            authors = clean_authors
        if raw_title:
            title = raw_title
    else:
        if pdf_meta:
            meta_title = str(pdf_meta.get("/Title") or "").strip()
            meta_author = str(pdf_meta.get("/Author") or "").strip()
            if meta_title and len(meta_title) > 5 and meta_title.lower() not in {"untitled", "document"}:
                title = meta_title
            if meta_author and len(meta_author) > 2:
                authors = meta_author

    # Refina autores nos artigos demo conhecidos caso venha abreviado no nome do arquivo
    first_page = pages_text[0][:1200] if pages_text else ""
    if "Brantly Callaway" in first_page and "Goodman-Bacon" in first_page:
        authors = "Brantly Callaway, Andrew Goodman-Bacon & Pedro H. C. Sant'Anna"
        title = "Difference-in-Differences with a Continuous Treatment"
        year = year or "2021"
    elif "Brantly Callaway" in first_page and "Multiple Time Periods" in first_page:
        authors = "Brantly Callaway & Pedro H. C. Sant'Anna"
        title = "Difference-in-Differences with Multiple Time Periods"
        year = year or "2020"

    if not year and first_page:
        ym2 = re.search(r"\b(20\d{2}|19\d{2})\b", first_page)
        if ym2:
            year = ym2.group(1)

    return title, authors, year


# ---------------------------------------------------------------------------
# Estruturas internas de Documento e Chunk
# ---------------------------------------------------------------------------

@dataclass
class PageRecord:
    page_number: int  # 1-based
    text: str


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    title: str
    authors: str
    year: Optional[str]
    pages: List[PageRecord]
    is_demo: bool
    ingested_at: str

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def total_words(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    @property
    def preview(self) -> str:
        joined = " ".join(p.text for p in self.pages[:2]).strip()
        return joined[:280] + ("..." if len(joined) > 280 else "")


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    document_name: str
    document_title: str
    page: int
    chunk_index: int
    text: str
    overlap_prefix_chars: int
    tokens: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking consciente de sentenças com rastreamento de Overlap
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n\n+")


def split_page_into_chunks(
    doc: DocumentRecord,
    page: PageRecord,
    chunk_size: int,
    chunk_overlap: int,
    start_chunk_idx: int,
) -> List[ChunkRecord]:
    """Segmenta o texto de uma página respeitando fronteiras de sentença e rastreando overlap."""
    text = page.text.strip()
    if not text:
        return []

    effective_overlap = min(chunk_overlap, max(0, chunk_size // 2))
    raw_units = [u.strip() for u in SENTENCE_SPLIT_RE.split(text) if u.strip()]

    # Garante que nenhuma sentença/unidade individual exceda chunk_size
    units: List[str] = []
    for u in raw_units:
        if len(u) <= chunk_size:
            units.append(u)
        else:
            step = max(200, chunk_size - effective_overlap)
            for start in range(0, len(u), step):
                piece = u[start : start + chunk_size].strip()
                if piece:
                    units.append(piece)

    chunks: List[ChunkRecord] = []
    current_parts: List[str] = []
    current_len = 0
    overlap_prefix_len = 0
    idx = start_chunk_idx

    for unit in units:
        add_len = len(unit) + (1 if current_parts else 0)
        if current_parts and current_len + add_len > chunk_size:
            chunk_text = " ".join(current_parts).strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}-c{idx:04d}",
                    doc_id=doc.doc_id,
                    document_name=doc.filename,
                    document_title=doc.title,
                    page=page.page_number,
                    chunk_index=idx,
                    text=chunk_text,
                    overlap_prefix_chars=overlap_prefix_len,
                    tokens=tokenize(chunk_text, remove_stopwords=True),
                )
            )
            idx += 1

            # Calcula sufixo de overlap preservando palavras inteiras
            if effective_overlap > 0:
                tail = chunk_text[-effective_overlap:]
                # Ajusta para começar na próxima palavra completa se cortou no meio
                space_pos = tail.find(" ")
                if 0 < space_pos < len(tail) - 10:
                    tail = tail[space_pos + 1 :].strip()
                else:
                    tail = tail.strip()
                if tail:
                    current_parts = [tail, unit]
                    overlap_prefix_len = len(tail)
                    current_len = len(tail) + 1 + len(unit)
                else:
                    current_parts = [unit]
                    overlap_prefix_len = 0
                    current_len = len(unit)
            else:
                current_parts = [unit]
                overlap_prefix_len = 0
                current_len = len(unit)
        else:
            current_parts.append(unit)
            current_len += add_len

    if current_parts:
        chunk_text = " ".join(current_parts).strip()
        if chunk_text:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}-c{idx:04d}",
                    doc_id=doc.doc_id,
                    document_name=doc.filename,
                    document_title=doc.title,
                    page=page.page_number,
                    chunk_index=idx,
                    text=chunk_text,
                    overlap_prefix_chars=overlap_prefix_len,
                    tokens=tokenize(chunk_text, remove_stopwords=True),
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Motor Principal PaperRAG
# ---------------------------------------------------------------------------

class PaperRAGEngine:
    """Motor RAG Híbrido (BM25 + LSA/Char Embeddings + RRF + MMR) para PDFs Científicos."""

    def __init__(self, demo_papers_dir: Path) -> None:
        self.demo_papers_dir = demo_papers_dir
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.chunk_size: int = 1000
        self.chunk_overlap: int = 180

        self.documents: Dict[str, DocumentRecord] = {}
        self.chunks: List[ChunkRecord] = []

        # Estruturas do Canal Lexical (Okapi BM25)
        self._bm25_tf: List[Counter[str]] = []
        self._bm25_df: Counter[str] = Counter()
        self._bm25_doc_lens: np.ndarray = np.array([], dtype=np.float64)
        self._bm25_avgdl: float = 1.0

        # Estruturas do Canal Semântico (LSA + Subword Char Embeddings)
        self._word_vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._lsa_matrix: Optional[np.ndarray] = None
        self._char_vectorizer: Optional[TfidfVectorizer] = None
        self._char_matrix = None

        self._last_build_ms: float = 0.0
        self._lsa_dims: int = 0
        self._vocab_size: int = 0

    # --- Ingestão de PDFs --------------------------------------------------

    def ingest_pdf_bytes(self, filename: str, content: bytes, is_demo: bool = False) -> DocumentRecord:
        """Extrai texto página a página de um PDF em memória e registra na biblioteca."""
        reader = PdfReader(io.BytesIO(content))
        pages: List[PageRecord] = []
        raw_page_texts: List[str] = []

        for i, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = clean_pdf_text(extracted)
            raw_page_texts.append(cleaned)
            if cleaned:
                pages.append(PageRecord(page_number=i, text=cleaned))

        if not pages:
            raise ValueError(
                f"Não foi possível extrair texto pesquisável de '{filename}'. "
                "Verifique se o PDF contém texto selecionável (não apenas imagens escaneadas)."
            )

        pdf_meta = dict(reader.metadata) if reader.metadata else None
        title, authors, year = infer_paper_metadata(filename, raw_page_texts, pdf_meta)

        digest = hashlib.sha1(f"{filename}:{len(content)}".encode("utf-8")).hexdigest()[:10]
        doc_id = f"doc-{digest}"

        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            title=title,
            authors=authors,
            year=year,
            pages=pages,
            is_demo=is_demo,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        self.documents[doc_id] = record
        return record

    def load_demo_papers(self) -> List[DocumentRecord]:
        """Carrega os artigos científicos pré-empacotados em data/papers/ e reconstrói o índice."""
        loaded: List[DocumentRecord] = []
        if not self.demo_papers_dir.exists():
            return loaded

        for pdf_path in sorted(self.demo_papers_dir.glob("*.pdf")):
            content = pdf_path.read_bytes()
            rec = self.ingest_pdf_bytes(pdf_path.name, content, is_demo=True)
            loaded.append(rec)

        self.rebuild_index(self.chunk_size, self.chunk_overlap)
        return loaded

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.rebuild_index(self.chunk_size, self.chunk_overlap)
        return True

    def clear_library(self) -> None:
        self.documents.clear()
        self.rebuild_index(self.chunk_size, self.chunk_overlap)

    # --- Indexação Híbrida (BM25 + LSA/Char Embeddings) --------------------

    def rebuild_index(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> IndexStats:
        """Reconstrói chunks, índice BM25 e espaço vetorial semântico denso."""
        t0 = time.perf_counter()
        if chunk_size is not None:
            self.chunk_size = int(chunk_size)
        if chunk_overlap is not None:
            self.chunk_overlap = min(int(chunk_overlap), max(0, self.chunk_size // 2))

        all_chunks: List[ChunkRecord] = []
        for doc in self.documents.values():
            doc_chunk_idx = 0
            for page in doc.pages:
                page_chunks = split_page_into_chunks(
                    doc=doc,
                    page=page,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    start_chunk_idx=doc_chunk_idx,
                )
                all_chunks.extend(page_chunks)
                doc_chunk_idx += len(page_chunks)

        self.chunks = all_chunks

        # Reset se biblioteca vazia
        if not self.chunks:
            self._bm25_tf = []
            self._bm25_df = Counter()
            self._bm25_doc_lens = np.array([], dtype=np.float64)
            self._bm25_avgdl = 1.0
            self._word_vectorizer = None
            self._svd = None
            self._lsa_matrix = None
            self._char_vectorizer = None
            self._char_matrix = None
            self._lsa_dims = 0
            self._vocab_size = 0
            self._last_build_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self.get_index_stats()

        # 1. Índice Okapi BM25
        self._bm25_tf = []
        self._bm25_df = Counter()
        lens: List[int] = []
        for ch in self.chunks:
            tf = Counter(ch.tokens)
            self._bm25_tf.append(tf)
            for term in tf:
                self._bm25_df[term] += 1
            lens.append(len(ch.tokens))
        self._bm25_doc_lens = np.array(lens, dtype=np.float64)
        self._bm25_avgdl = float(np.mean(self._bm25_doc_lens)) if len(lens) else 1.0
        self._vocab_size = len(self._bm25_df)

        # 2. Índice Semântico Vetorial Denso (Word TF-IDF -> LSA SVD + Subword Char N-grams)
        corpus_texts = [strip_accents(ch.text.lower()) for ch in self.chunks]

        self._word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=14000,
            token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
            stop_words=list(STOPWORDS_PT_EN),
        )
        word_tfidf = self._word_vectorizer.fit_transform(corpus_texts)

        n_docs, n_feats = word_tfidf.shape
        target_dims = min(96, max(2, min(n_docs - 1, n_feats - 1)))
        if n_docs >= 3 and n_feats >= 3:
            self._svd = TruncatedSVD(n_components=target_dims, random_state=42)
            lsa_raw = self._svd.fit_transform(word_tfidf)
            self._lsa_matrix = normalize(lsa_raw, norm="l2", axis=1)
            self._lsa_dims = int(self._lsa_matrix.shape[1])
        else:
            self._svd = None
            dense = word_tfidf.toarray()
            self._lsa_matrix = normalize(dense, norm="l2", axis=1)
            self._lsa_dims = int(self._lsa_matrix.shape[1])

        self._char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=22000,
        )
        self._char_matrix = normalize(self._char_vectorizer.fit_transform(corpus_texts), norm="l2", axis=1)

        self._last_build_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return self.get_index_stats()

    # --- Estatísticas, Biblioteca e Sugestões Didáticas --------------------

    def get_index_stats(self) -> IndexStats:
        doc_count = len(self.documents)
        total_pages = sum(len(d.pages) for d in self.documents.values())
        total_chars = sum(d.total_chars for d in self.documents.values())
        total_chunks = len(self.chunks)

        if total_chunks == 0:
            return IndexStats(
                document_count=doc_count,
                total_pages=total_pages,
                total_chunks=0,
                total_chars=total_chars,
                avg_chunk_chars=0.0,
                median_chunk_chars=0.0,
                min_chunk_chars=0,
                max_chunk_chars=0,
                vocab_size=0,
                lsa_dimensions=0,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                overlap_ratio_pct=0.0,
                build_time_ms=self._last_build_ms,
                histogram=[],
            )

        lengths = np.array([len(c.text) for c in self.chunks], dtype=np.float64)
        overlap_chars = sum(c.overlap_prefix_chars for c in self.chunks)
        chunk_sum_chars = float(np.sum(lengths))
        overlap_ratio = round((overlap_chars / chunk_sum_chars) * 100.0, 1) if chunk_sum_chars > 0 else 0.0

        # Histograma em 8 faixas para o Recharts no Laboratório de Chunking
        min_l, max_l = int(lengths.min()), int(lengths.max())
        bins_n = min(8, max(3, len(set(lengths.tolist()))))
        if min_l == max_l:
            histogram = [
                ChunkHistogramBin(
                    range_label=f"{min_l} chars",
                    min_chars=min_l,
                    max_chars=max_l,
                    count=total_chunks,
                )
            ]
        else:
            counts, edges = np.histogram(lengths, bins=bins_n)
            histogram = []
            for i in range(len(counts)):
                lo = int(round(edges[i]))
                hi = int(round(edges[i + 1]))
                histogram.append(
                    ChunkHistogramBin(
                        range_label=f"{lo}-{hi}",
                        min_chars=lo,
                        max_chars=hi,
                        count=int(counts[i]),
                    )
                )

        return IndexStats(
            document_count=doc_count,
            total_pages=total_pages,
            total_chunks=total_chunks,
            total_chars=total_chars,
            avg_chunk_chars=round(float(np.mean(lengths)), 1),
            median_chunk_chars=round(float(np.median(lengths)), 1),
            min_chunk_chars=min_l,
            max_chunk_chars=max_l,
            vocab_size=self._vocab_size,
            lsa_dimensions=self._lsa_dims,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            overlap_ratio_pct=overlap_ratio,
            build_time_ms=self._last_build_ms,
            histogram=histogram,
        )

    def get_suggested_questions(self) -> List[SuggestedQuestion]:
        """Gera perguntas guiadas adaptadas aos artigos carregados (incluindo teste anti-alucinação)."""
        titles_joined = " ".join(d.title.lower() for d in self.documents.values())
        if "continuous treatment" in titles_joined or "multiple time periods" in titles_joined:
            return [
                SuggestedQuestion(
                    id="q-spt",
                    category="Identificação Causal",
                    question="Qual a diferença entre a hipótese de Tendências Paralelas padrão e Tendências Paralelas Fortes (Strong Parallel Trends) no tratamento contínuo?",
                    hint="Mostra como a ponte bilíngue PT↔EN encontra a definição de Strong Parallel Trends e o viés de seleção em Callaway, Goodman-Bacon & Sant'Anna (2021).",
                ),
                SuggestedQuestion(
                    id="q-att-gt",
                    category="Múltiplos Períodos",
                    question="Como é definido o parâmetro group-time average treatment effect ATT(g,t) e como ele pode ser agregado em estudos de evento?",
                    hint="Recupera diretamente a definição formal de ATT(g,t) e os esquemas de agregação em Callaway & Sant'Anna (2020).",
                ),
                SuggestedQuestion(
                    id="q-twfe",
                    category="Diagnóstico TWFE",
                    question="Quais são os problemas e o viés de seleção do estimador TWFE (Two-Way Fixed Effects) quando o tratamento é contínuo ou escalonado?",
                    hint="Compara evidências dos dois artigos sobre decomposição de pesos e viés de seleção em TWFE.",
                ),
                SuggestedQuestion(
                    id="q-empirical",
                    category="Aplicação Empírica",
                    question="Quais aplicações empíricas (como salário mínimo ou Medicare) são analisadas nos artigos?",
                    hint="Testa recuperação multi-documento combinando a aplicação do salário mínimo (2020) e Medicare (2021).",
                ),
                SuggestedQuestion(
                    id="q-guardrail",
                    category="Teste Anti-Alucinação",
                    question="Qual foi a receita líquida da Petrobras no terceiro trimestre de 2025 e a política de dividendos?",
                    hint="Pergunta propositalmente fora do escopo dos PDFs para demonstrar o guardrail anti-alucinação do RAG.",
                    is_out_of_scope=True,
                ),
            ]

        return [
            SuggestedQuestion(
                id="q-gen-1",
                category="Visão Geral",
                question="Qual é o objetivo principal e a principal contribuição metodológica dos artigos carregados?",
                hint="Síntese geral dos objetivos e achados presentes nos PDFs da biblioteca.",
            ),
            SuggestedQuestion(
                id="q-gen-2",
                category="Metodologia",
                question="Quais dados, hipóteses de identificação e métodos de estimação são utilizados nos estudos?",
                hint="Recupera seções metodológicas e especificação empírica.",
            ),
            SuggestedQuestion(
                id="q-gen-3",
                category="Resultados",
                question="Quais são os principais resultados empíricos e limitações discutidos pelos autores?",
                hint="Foca em conclusões, resultados e limitações.",
            ),
            SuggestedQuestion(
                id="q-gen-guardrail",
                category="Teste Anti-Alucinação",
                question="Qual a receita culinária tradicional de bolo de cenoura com cobertura de chocolate?",
                hint="Pergunta fora do escopo para testar o bloqueio de alucinação.",
                is_out_of_scope=True,
            ),
        ]

    def get_library_state(self) -> LibraryResponse:
        chunk_counts = Counter(c.doc_id for c in self.chunks)
        docs = [
            DocumentSummary(
                doc_id=d.doc_id,
                filename=d.filename,
                title=d.title,
                authors=d.authors,
                year=d.year,
                pages=len(d.pages),
                total_chars=d.total_chars,
                total_words=d.total_words,
                chunk_count=chunk_counts.get(d.doc_id, 0),
                is_demo=d.is_demo,
                preview=d.preview,
                ingested_at=d.ingested_at,
            )
            for d in self.documents.values()
        ]
        return LibraryResponse(
            documents=docs,
            stats=self.get_index_stats(),
            suggested_questions=self.get_suggested_questions(),
        )

    def list_chunks(
        self,
        doc_id: Optional[str] = None,
        search: Optional[str] = None,
        offset: int = 0,
        limit: int = 24,
    ) -> ChunksPageResponse:
        filtered = self.chunks
        if doc_id:
            filtered = [c for c in filtered if c.doc_id == doc_id]
        if search and search.strip():
            needle = strip_accents(search.strip().lower())
            filtered = [c for c in filtered if needle in strip_accents(c.text.lower())]

        total = len(filtered)
        page_slice = filtered[offset : offset + limit]
        items = [
            ChunkItem(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                document_name=c.document_name,
                document_title=c.document_title,
                page=c.page,
                chunk_index=c.chunk_index,
                char_count=len(c.text),
                word_count=len(c.text.split()),
                overlap_prefix_chars=c.overlap_prefix_chars,
                text=c.text,
                preview=c.text[:220] + ("..." if len(c.text) > 220 else ""),
            )
            for c in page_slice
        ]
        return ChunksPageResponse(total=total, offset=offset, limit=limit, items=items)

    # --- Descoberta de Provedores LLM (Offline, Ollama, OpenAI, Gemini) ----

    def inspect_providers(self) -> ProvidersResponse:
        ollama_models: List[str] = []
        ollama_up = False
        ollama_detail = f"Ollama não detectado em {self.ollama_url} (opcional)"
        try:
            with httpx.Client(timeout=1.2) as client:
                r = client.get(f"{self.ollama_url}/api/tags")
                if r.status_code == 200:
                    payload = r.json()
                    ollama_models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
                    ollama_up = True
                    if ollama_models:
                        ollama_detail = f"Online em {self.ollama_url} ({len(ollama_models)} modelo(s) disponível(is))"
                    else:
                        ollama_detail = f"Online em {self.ollama_url}, mas nenhum modelo baixado (rode: ollama pull qwen2.5:7b)"
        except Exception:
            ollama_up = False

        has_openai_env = bool(os.getenv("OPENAI_API_KEY"))
        has_gemini_env = bool(os.getenv("GEMINI_API_KEY"))

        providers = [
            ProviderInfo(
                id="extractive",
                name="Síntese Extrativa Grounded (Offline)",
                available=True,
                requires_key=False,
                models=["extractive-grounded-v2"],
                default_model="extractive-grounded-v2",
                description="Funciona 100% offline, sem GPU e sem chave de API. Sintetiza evidências literais dos trechos recuperados com citações exatas [1..k].",
                status_detail="Sempre pronto (zero latência externa)",
            ),
            ProviderInfo(
                id="ollama",
                name="Ollama Local (LLM Privado)",
                available=ollama_up and len(ollama_models) > 0,
                requires_key=False,
                models=ollama_models if ollama_models else ["qwen2.5:7b", "llama3.2:3b", "mistral:7b"],
                default_model=ollama_models[0] if ollama_models else "qwen2.5:7b",
                description="Executa LLMs open-source localmente com privacidade total e tradução/síntese fluente em português.",
                status_detail=ollama_detail,
            ),
            ProviderInfo(
                id="openai",
                name="OpenAI API",
                available=True,
                requires_key=not has_openai_env,
                models=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
                default_model="gpt-4o-mini",
                description="Geração generativa em PT-BR com alta fidelidade às citações via chave de sessão ou variável OPENAI_API_KEY.",
                status_detail="Chave detectada no ambiente" if has_openai_env else "Informe sua chave na sessão (nunca salva em disco)",
            ),
            ProviderInfo(
                id="gemini",
                name="Google Gemini API",
                available=True,
                requires_key=not has_gemini_env,
                models=["gemini-2.5-flash", "gemini-2.0-flash"],
                default_model="gemini-2.5-flash",
                description="Geração rápida e longa janela de contexto via chave de sessão ou variável GEMINI_API_KEY.",
                status_detail="Chave detectada no ambiente" if has_gemini_env else "Informe sua chave na sessão (nunca salva em disco)",
            ),
        ]
        return ProvidersResponse(providers=providers, ollama_url=self.ollama_url)

    def is_ollama_available(self) -> bool:
        try:
            with httpx.Client(timeout=0.8) as client:
                r = client.get(f"{self.ollama_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    # --- Retrieval Híbrido (BM25 + Semântico Vetorial + RRF + MMR) ---------

    def _score_bm25(
        self,
        orig_tokens: Sequence[str],
        expanded_tokens: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """Calcula Okapi BM25 exato por chunk e retorna termos casados em cada chunk."""
        n_docs = len(self.chunks)
        scores = np.zeros(n_docs, dtype=np.float64)
        matched_per_chunk: List[List[str]] = [[] for _ in range(n_docs)]

        if n_docs == 0:
            return scores, matched_per_chunk

        weighted_query: Dict[str, float] = {}
        for t in orig_tokens:
            weighted_query[t] = max(weighted_query.get(t, 0.0), 1.0)
        for t in expanded_tokens:
            weighted_query[t] = max(weighted_query.get(t, 0.0), 0.78)

        if not weighted_query:
            return scores, matched_per_chunk

        for term, q_weight in weighted_query.items():
            df = self._bm25_df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for i, tf_map in enumerate(self._bm25_tf):
                tf = tf_map.get(term, 0)
                if tf > 0:
                    dl = self._bm25_doc_lens[i]
                    denom = tf + k1 * (1.0 - b + b * (dl / self._bm25_avgdl))
                    scores[i] += q_weight * idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)
                    matched_per_chunk[i].append(term)

        return scores, matched_per_chunk

    def _score_semantic(self, query_text: str) -> np.ndarray:
        """Calcula similaridade cosseno no espaço semântico denso (LSA + Char Subword)."""
        n_docs = len(self.chunks)
        if n_docs == 0 or self._word_vectorizer is None or self._lsa_matrix is None:
            return np.zeros(n_docs, dtype=np.float64)

        norm_q = strip_accents(query_text.lower())

        # 1. Projeção LSA Densa
        q_word = self._word_vectorizer.transform([norm_q])
        if self._svd is not None:
            q_lsa = normalize(self._svd.transform(q_word), norm="l2", axis=1)
        else:
            q_lsa = normalize(q_word.toarray(), norm="l2", axis=1)
        cos_lsa = np.clip((self._lsa_matrix @ q_lsa.T).ravel(), 0.0, 1.0)

        # 2. Subword Char N-gram Cosine (captura cognatos PT <-> EN e morfologia)
        if self._char_vectorizer is not None and self._char_matrix is not None:
            q_char = normalize(self._char_vectorizer.transform([norm_q]), norm="l2", axis=1)
            cos_char = np.clip((self._char_matrix @ q_char.T).toarray().ravel(), 0.0, 1.0)
            combined = 0.62 * cos_lsa + 0.38 * cos_char
        else:
            combined = cos_lsa

        return np.clip(combined, 0.0, 1.0)

    def _select_with_mmr(
        self,
        candidate_indices: List[int],
        relevance_scores: np.ndarray,
        top_k: int,
        mmr_lambda: float,
    ) -> List[int]:
        """Seleciona top_k chunks aplicando Maximal Marginal Relevance sobre vetores densos LSA."""
        if len(candidate_indices) <= top_k or self._lsa_matrix is None:
            return candidate_indices[:top_k]

        selected: List[int] = [candidate_indices[0]]
        remaining = candidate_indices[1:]

        while len(selected) < top_k and remaining:
            best_idx = remaining[0]
            best_mmr = -1e9
            sel_vecs = self._lsa_matrix[selected]

            for cand in remaining:
                rel = float(relevance_scores[cand])
                sims = sel_vecs @ self._lsa_matrix[cand]
                max_sim = float(np.max(sims)) if len(sims) else 0.0
                mmr_score = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = cand

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    def _build_excerpt(self, text: str, matched_terms: Sequence[str], max_chars: int = 340) -> str:
        """Extrai o trecho mais denso em termos relevantes para pré-visualização didática."""
        if len(text) <= max_chars:
            return text
        if not matched_terms:
            return text[:max_chars].rstrip() + "..."

        plain = strip_accents(text.lower())
        best_pos = 0
        best_hits = -1
        step = 60
        for start in range(0, max(1, len(plain) - max_chars + 1), step):
            window = plain[start : start + max_chars]
            hits = sum(1 for t in matched_terms if t in window)
            if hits > best_hits:
                best_hits = hits
                best_pos = start

        # Alinha ao início da palavra
        if best_pos > 0:
            sp = text.find(" ", max(0, best_pos - 25), best_pos + 25)
            if sp != -1:
                best_pos = sp + 1

        end_pos = min(len(text), best_pos + max_chars)
        prefix = "..." if best_pos > 0 else ""
        suffix = "..." if end_pos < len(text) else ""
        return f"{prefix}{text[best_pos:end_pos].strip()}{suffix}"

    # --- Montagem de Prompt Grounded e Geração Multi-Provedor --------------

    def _build_rag_prompt(
        self,
        question: str,
        citations: List[RetrievedChunk],
        response_language: str,
        out_of_scope_warning: bool,
    ) -> PromptInspection:
        lang_instruction = (
            "Responda SEMPRE em Português Brasileiro claro, técnico e didático (traduzindo e explicando os conceitos dos trechos em inglês quando necessário)."
            if response_language == "pt-BR"
            else "Respond in clear, rigorous academic English."
        )

        system_prompt = (
            "Você é o PaperRAG Studio, um assistente científico especialista em análise rigorosa de artigos acadêmicos.\n"
            "REGRAS OBRIGATÓRIAS DE FIDELIDADE (GROUNDING):\n"
            "1. Responda EXCLUSIVAMENTE com base nos trechos numerados [1], [2], ... fornecidos no contexto.\n"
            "2. Cite explicitamente a origem de cada afirmação usando colchetes [1], [2], etc.\n"
            "3. NUNCA invente números, teoremas, resultados empíricos ou informações ausentes do contexto.\n"
            "4. Se os trechos não contiverem evidência suficiente para responder, declare isso com transparência.\n"
            f"5. {lang_instruction}"
        )

        context_blocks: List[str] = []
        for c in citations:
            context_blocks.append(
                f"[{c.rank}] Fonte: {c.document_title} ({c.document_name}) | Página {c.page} | Score Híbrido: {c.score_final:.3f}\n"
                f"{c.text}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        scope_note = (
            "\n[ALERTA DO RETRIEVER]: Os scores de similaridade ficaram abaixo do limiar de confiança. "
            "Verifique cuidadosamente se o contexto realmente responde à pergunta; caso contrário, recuse especular.\n"
            if out_of_scope_warning
            else ""
        )

        user_prompt = (
            f"CONTEXTO RECUPERADO DOS ARTIGOS CIENTÍFICOS:\n\n{context_str}\n"
            f"{scope_note}\n"
            f"PERGUNTA DO PESQUISADOR:\n{question}\n\n"
            "Estruture sua resposta em: (a) Resumo Direto, (b) Detalhamento Técnico com citações [1], [2]..., e (c) Síntese Comparativa (se houver múltiplos artigos)."
        )

        total_chars = len(system_prompt) + len(user_prompt)
        est_tokens = max(1, int(round(total_chars / 3.8)))
        return PromptInspection(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            estimated_tokens=est_tokens,
            context_chars=len(context_str),
        )

    def _generate_extractive_answer(
        self,
        question: str,
        citations: List[RetrievedChunk],
        query_terms: List[str],
        out_of_scope_warning: bool,
        grounding_score: float,
    ) -> str:
        """Gera síntese estruturada 100% offline baseada estritamente nos trechos recuperados."""
        if not citations or out_of_scope_warning:
            closest_lines = "\n".join(
                f"- **[{c.rank}] {c.document_title}** (pág. {c.page}, relevância `{c.score_final:.2f}`): *\"{c.highlighted_excerpt[:160]}...\"*"
                for c in citations[:2]
            )
            return (
                "### Evidência Insuficiente nos Artigos Carregados (Guardrail Anti-Alucinação)\n\n"
                "Não encontrei evidências confiáveis na biblioteca atual de PDFs para responder à sua pergunta com rigor científico "
                f"(confiança de grounding: **{grounding_score:.0f}%**).\n\n"
                "**Por que isso aconteceu?**\n"
                "Em uma arquitetura **RAG Grounded**, quando os termos e conceitos da consulta apresentam baixa aderência lexical (BM25) "
                "e semântica (vetores LSA/subword) aos trechos indexados, o sistema **recusa alucinar** uma resposta com conhecimento externo.\n\n"
                + (
                    f"**Trechos mais próximos apenas para auditoria (baixa aderência):**\n{closest_lines}\n\n"
                    if closest_lines
                    else ""
                )
                + "**Sugestão:** Envie um PDF sobre esse tema na aba **Biblioteca & Chunking** ou selecione uma das perguntas guiadas sobre os artigos carregados."
            )

        # Seleciona sentenças mais informativas dentro dos chunks recuperados
        unique_titles = list(dict.fromkeys(c.document_title for c in citations))
        pages_by_doc: Dict[str, List[int]] = {}
        for c in citations:
            pages_by_doc.setdefault(c.document_title, [])
            if c.page not in pages_by_doc[c.document_title]:
                pages_by_doc[c.document_title].append(c.page)

        q_set = set(query_terms)
        scored_sentences: List[Tuple[float, int, str, RetrievedChunk]] = []

        for c in citations:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c.text) if 45 <= len(s.strip()) <= 440]
            if not sents:
                sents = [c.highlighted_excerpt]
            for s in sents:
                # Evita linhas puramente numéricas / cabeçalhos de tabela quebrados
                alpha_ratio = sum(ch.isalpha() for ch in s) / max(len(s), 1)
                if alpha_ratio < 0.58:
                    continue
                s_toks = set(tokenize(s, remove_stopwords=True))
                overlap = len(s_toks & q_set)
                density = overlap / max(math.sqrt(len(s_toks) + 1), 1.0)
                score = (density * 1.4) + (c.score_final * 0.9) + (0.15 if overlap >= 2 else 0.0)
                scored_sentences.append((score, c.rank, s, c))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)

        # Escolhe até 5 sentenças diversas garantindo cobertura de múltiplos chunks/artigos
        chosen: List[Tuple[int, str, RetrievedChunk]] = []
        used_per_rank: Counter[int] = Counter()
        seen_prefixes = set()

        for _, rank, sent, chunk in scored_sentences:
            if len(chosen) >= 5:
                break
            if used_per_rank[rank] >= 2:
                continue
            prefix_key = strip_accents(sent[:55].lower())
            if prefix_key in seen_prefixes:
                continue
            seen_prefixes.add(prefix_key)
            used_per_rank[rank] += 1
            chosen.append((rank, sent, chunk))

        # Garante que o chunk #1 apareça
        if chosen and all(r != 1 for r, _, _ in chosen):
            c1 = citations[0]
            chosen.insert(0, (1, c1.highlighted_excerpt, c1))
            chosen = chosen[:5]

        sources_summary = "; ".join(
            f"**{title}** (pág. {', '.join(str(p) for p in sorted(pages_by_doc[title]))})"
            for title in unique_titles
        )

        bullets: List[str] = []
        for rank, sent, chunk in chosen:
            clean_sent = re.sub(r"\s+", " ", sent).strip()
            bullets.append(
                f"- **[{rank}]** *(pág. {chunk.page} — {chunk.document_title})*: \"{clean_sent}\" [{rank}]"
            )

        terms_found = list(dict.fromkeys(t for c in citations for t in c.matched_terms))[:10]
        terms_str = ", ".join(f"`{t}`" for t in terms_found) if terms_found else "`busca semântica densa`"

        answer = (
            f"### Síntese Grounded (Modo Extrativo Offline)\n\n"
            f"A busca híbrida recuperou **{len(citations)} trechos de alta relevância** em {sources_summary}, "
            f"com aderência consolidada de **{grounding_score:.0f}%**.\n\n"
            f"#### Principais Evidências Extraídas dos Artigos\n\n"
            + "\n".join(bullets)
            + "\n\n"
            f"#### Conexão Conceitual & Auditoria\n"
            f"- **Conceitos-âncora localizados:** {terms_str}.\n"
            f"- **Transparência:** Clique em qualquer selo **[1]..[{len(citations)}]** abaixo ou abra a aba **Raio-X do Retrieval** "
            "para comparar os scores BM25 vs. Semântico e inspecionar o prompt pronto para LLM.\n"
            "- **Dica:** Para obter síntese e tradução generativa fluida em português sobre estes mesmos trechos, selecione **Ollama Local**, **OpenAI** ou **Gemini** na barra lateral."
        )
        return answer

    def _call_ollama(self, model: str, prompt: PromptInspection, temperature: float) -> str:
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        }
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content") or data.get("response") or ""
            if not content.strip():
                raise RuntimeError("Ollama retornou resposta vazia.")
            return content.strip()

    def _call_openai(self, model: str, api_key: str, prompt: PromptInspection, temperature: float) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, model: str, api_key: str, prompt: PromptInspection, temperature: float) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": prompt.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt.user_prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, params={"key": api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini não retornou candidatos de resposta.")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

    # --- Pipeline Completo de Consulta RAG ---------------------------------

    def query(self, req: QueryRequest) -> QueryResponse:
        if not self.chunks:
            raise ValueError("A biblioteca está vazia. Carregue os artigos de demonstração ou envie um PDF primeiro.")

        t_start = time.perf_counter()

        # 1. Análise da consulta e Ponte Bilíngue PT <-> EN
        orig_tokens, expanded_terms, detected_lang = expand_query_bilingual(req.question)
        all_query_terms = list(dict.fromkeys(orig_tokens + expanded_terms))
        enriched_query = f"{req.question} {' '.join(expanded_terms)}".strip()

        # 2. Scoring nos dois canais (BM25 Lexical + Semântico Vetorial Denso)
        raw_bm25, matched_per_chunk = self._score_bm25(orig_tokens, expanded_terms)
        raw_sem = self._score_semantic(enriched_query)

        # Filtro opcional por documentos selecionados
        allowed_mask = np.ones(len(self.chunks), dtype=bool)
        if req.doc_ids:
            doc_set = set(req.doc_ids)
            allowed_mask = np.array([c.doc_id in doc_set for c in self.chunks], dtype=bool)
            if not np.any(allowed_mask):
                raise ValueError("Nenhum chunk corresponde aos documentos filtrados.")

        valid_indices = np.where(allowed_mask)[0]
        bm25_valid = raw_bm25[valid_indices]
        sem_valid = raw_sem[valid_indices]

        # 3. Calibração absoluta (impede inflação artificial em consultas fora do escopo!)
        max_bm25 = float(np.max(bm25_valid)) if len(bm25_valid) else 0.0
        max_sem = float(np.max(sem_valid)) if len(sem_valid) else 0.0

        # Normalização suave com piso de escala para preservar sinal absoluto
        bm25_scale = max(max_bm25, 4.5)
        sem_scale = max(max_sem, 0.32)

        norm_bm25_all = np.clip(raw_bm25 / bm25_scale, 0.0, 1.0)
        norm_sem_all = np.clip(raw_sem / sem_scale, 0.0, 1.0)

        # Ranks individuais (1-based) dentro dos candidatos permitidos
        order_lex = valid_indices[np.argsort(-raw_bm25[valid_indices])]
        order_sem = valid_indices[np.argsort(-raw_sem[valid_indices])]

        rank_lex_map = {int(idx): r for r, idx in enumerate(order_lex, start=1)}
        rank_sem_map = {int(idx): r for r, idx in enumerate(order_sem, start=1)}

        # 4. Reciprocal Rank Fusion (RRF) + Score Híbrido Ponderado por alpha
        k_rrf = 60.0
        alpha = float(req.alpha)
        rrf_scores = np.zeros(len(self.chunks), dtype=np.float64)
        final_scores = np.zeros(len(self.chunks), dtype=np.float64)

        max_rrf_theoretical = 1.0 / (k_rrf + 1.0)
        for idx in valid_indices:
            i = int(idx)
            rrf_val = (1.0 - alpha) / (k_rrf + rank_lex_map[i]) + alpha / (k_rrf + rank_sem_map[i])
            rrf_norm = rrf_val / max_rrf_theoretical
            rrf_scores[i] = rrf_val
            # Combina fidelidade absoluta (82%) + consenso de ranking RRF (18%)
            weighted_abs = (1.0 - alpha) * norm_bm25_all[i] + alpha * norm_sem_all[i]
            final_scores[i] = 0.82 * weighted_abs + 0.18 * (rrf_norm * max(weighted_abs, 0.15))

        # Pool candidato para MMR (top 3x ou min 20)
        sorted_candidates = [int(i) for i in valid_indices[np.argsort(-final_scores[valid_indices])]]
        pool_size = min(len(sorted_candidates), max(req.top_k * 3, 20))
        candidate_pool = sorted_candidates[:pool_size]

        if req.use_mmr:
            chosen_indices = self._select_with_mmr(candidate_pool, final_scores, req.top_k, req.mmr_lambda)
        else:
            chosen_indices = candidate_pool[: req.top_k]

        # Reordena selecionados por score final decrescente
        chosen_indices.sort(key=lambda idx: float(final_scores[idx]), reverse=True)

        citations: List[RetrievedChunk] = []
        for rank_num, idx in enumerate(chosen_indices, start=1):
            ch = self.chunks[idx]
            matched = list(dict.fromkeys(matched_per_chunk[idx]))[:12]
            citations.append(
                RetrievedChunk(
                    rank=rank_num,
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    document_name=ch.document_name,
                    document_title=ch.document_title,
                    page=ch.page,
                    chunk_index=ch.chunk_index,
                    text=ch.text,
                    highlighted_excerpt=self._build_excerpt(ch.text, matched),
                    score_final=round(float(final_scores[idx]), 4),
                    score_lexical=round(float(norm_bm25_all[idx]), 4),
                    score_semantic=round(float(norm_sem_all[idx]), 4),
                    score_rrf=round(float(rrf_scores[idx]), 5),
                    rank_lexical=rank_lex_map[idx],
                    rank_semantic=rank_sem_map[idx],
                    matched_terms=matched,
                )
            )

        t_retrieval = (time.perf_counter() - t_start) * 1000.0

        # 5. Guardrail Anti-Alucinação & Score de Grounding
        top_matched_union = set(t for c in citations[:3] for t in c.matched_terms)
        query_coverage = len(top_matched_union) / max(len(orig_tokens), 1)
        top_score = citations[0].score_final if citations else 0.0

        out_of_scope_warning = (max_bm25 < 2.2 and max_sem < 0.20) or (
            max_bm25 < 1.2 and len(top_matched_union) == 0
        )

        if out_of_scope_warning:
            grounding_score = round(min(32.0, max(6.0, top_score * 55.0)), 1)
            confidence_level = "baixa"
        else:
            raw_conf = 48.0 + 38.0 * min(1.0, top_score / 0.78) + 14.0 * min(1.0, query_coverage)
            grounding_score = round(min(98.5, max(35.0, raw_conf)), 1)
            confidence_level = "alta" if grounding_score >= 74.0 else ("media" if grounding_score >= 52.0 else "baixa")

        # 6. Montagem explícita do Prompt Grounded + Geração Multi-Provedor
        prompt_inspection = self._build_rag_prompt(
            question=req.question,
            citations=citations,
            response_language=req.response_language,
            out_of_scope_warning=out_of_scope_warning,
        )

        t_gen_start = time.perf_counter()
        provider_used = req.provider
        model_used = req.model or ""
        fallback_notice: Optional[str] = None
        answer = ""

        if req.provider == "extractive":
            model_used = "extractive-grounded-v2"
            answer = self._generate_extractive_answer(
                question=req.question,
                citations=citations,
                query_terms=all_query_terms,
                out_of_scope_warning=out_of_scope_warning,
                grounding_score=grounding_score,
            )
        elif req.provider == "ollama":
            model_used = req.model or "qwen2.5:7b"
            try:
                answer = self._call_ollama(model_used, prompt_inspection, req.temperature)
            except Exception as exc:
                fallback_notice = (
                    f"Ollama ({model_used}) indisponível ({exc}). "
                    "Resposta gerada automaticamente pelo modo Extrativo Grounded Offline."
                )
                provider_used = "extractive"
                model_used = "extractive-grounded-v2"
                answer = self._generate_extractive_answer(
                    req.question, citations, all_query_terms, out_of_scope_warning, grounding_score
                )
        elif req.provider == "openai":
            model_used = req.model or "gpt-4o-mini"
            key = (req.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not key:
                fallback_notice = "Chave OpenAI não fornecida. Usando síntese Extrativa Grounded Offline."
                provider_used = "extractive"
                model_used = "extractive-grounded-v2"
                answer = self._generate_extractive_answer(
                    req.question, citations, all_query_terms, out_of_scope_warning, grounding_score
                )
            else:
                try:
                    answer = self._call_openai(model_used, key, prompt_inspection, req.temperature)
                except Exception as exc:
                    fallback_notice = f"Falha ao chamar OpenAI ({exc}). Aplicado fallback Extrativo Grounded Offline."
                    provider_used = "extractive"
                    model_used = "extractive-grounded-v2"
                    answer = self._generate_extractive_answer(
                        req.question, citations, all_query_terms, out_of_scope_warning, grounding_score
                    )
        elif req.provider == "gemini":
            model_used = req.model or "gemini-2.5-flash"
            key = (req.api_key or os.getenv("GEMINI_API_KEY") or "").strip()
            if not key:
                fallback_notice = "Chave Gemini não fornecida. Usando síntese Extrativa Grounded Offline."
                provider_used = "extractive"
                model_used = "extractive-grounded-v2"
                answer = self._generate_extractive_answer(
                    req.question, citations, all_query_terms, out_of_scope_warning, grounding_score
                )
            else:
                try:
                    answer = self._call_gemini(model_used, key, prompt_inspection, req.temperature)
                except Exception as exc:
                    fallback_notice = f"Falha ao chamar Gemini ({exc}). Aplicado fallback Extrativo Grounded Offline."
                    provider_used = "extractive"
                    model_used = "extractive-grounded-v2"
                    answer = self._generate_extractive_answer(
                        req.question, citations, all_query_terms, out_of_scope_warning, grounding_score
                    )

        t_gen = (time.perf_counter() - t_gen_start) * 1000.0
        t_total = (time.perf_counter() - t_start) * 1000.0

        if alpha <= 0.15:
            mode_label = "Lexical BM25 (Palavras-chave)"
        elif alpha >= 0.85:
            mode_label = "Semântico Vetorial (LSA + Subword)"
        else:
            mode_label = f"Híbrido RRF ({int(round((1 - alpha) * 100))}% BM25 / {int(round(alpha * 100))}% Semântico)"

        diagnostics = QueryDiagnostics(
            detected_language=detected_lang,
            original_tokens=orig_tokens,
            expanded_terms=expanded_terms,
            alpha=alpha,
            retrieval_mode_label=mode_label,
            use_mmr=req.use_mmr,
            candidates_evaluated=int(len(valid_indices)),
            retrieval_ms=round(t_retrieval, 2),
            generation_ms=round(t_gen, 2),
            total_ms=round(t_total, 2),
            grounding_score=grounding_score,
            confidence_level=confidence_level,
            out_of_scope_warning=out_of_scope_warning,
            prompt_tokens_estimate=prompt_inspection.estimated_tokens,
            unique_docs_cited=len({c.doc_id for c in citations}),
            unique_pages_cited=len({(c.doc_id, c.page) for c in citations}),
        )

        return QueryResponse(
            question=req.question,
            answer=answer,
            provider_used=provider_used,
            model_used=model_used,
            fallback_notice=fallback_notice,
            citations=citations,
            diagnostics=diagnostics,
            prompt_inspection=prompt_inspection,
        )
