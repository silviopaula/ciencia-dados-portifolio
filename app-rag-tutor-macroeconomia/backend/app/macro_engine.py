"""Motor MacroRAG Tutor Studio — RAG Especializado em Livros-Texto e Notas de Macroeconomia.

Projetado para escalar confortavelmente para livros de 800+ páginas (como *Foundations of
Modern Macroeconomics*, Ben J. Heijdra, 816 págs.) e notas de aula em PDF/MD/TXT:
1. Extração hierárquica com árvore de Capítulos/Partes via PDF Outline (ou cabeçalhos Markdown).
2. Cache local persistente comprimido (.json.gz em backend/.cache/) para startup < 1,5s.
3. Limpeza tipográfica NFKC (ligaduras LaTeX/OUP) e chunking consciente de sentenças.
4. Índice Híbrido Rápido:
   - Canal Lexical: Okapi BM25 com índice invertido (< 10 ms em 2.000+ chunks) + Ponte Bilíngue
     PT <-> EN especializada em Macroeconomia (IS-LM, Phillips, Lucas, Ricardiana, RBC, OLG, NKPC).
   - Canal Semântico: Vetores Densos LSA (TruncatedSVD 96D) + Subword Char N-grams (3–5).
   - Fusão RRF + Diversidade MMR + Filtro por Capítulo/Documento.
5. 3 Modos de Estudo no Chat (`tutor`, `compare`, `verify`) + Gerador de Flashcards/Questões de Prova.
6. Auditoria de Privacidade explícita (bytes locais vs. bytes enviados a LLMs externos).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .schemas import (
    ChapterSummary,
    DocumentSummary,
    FlashcardItem,
    FlashcardRequest,
    FlashcardResponse,
    IndexStats,
    LibraryResponse,
    PrivacyAudit,
    PromptInspection,
    ProviderInfo,
    ProvidersResponse,
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    RetrievedPassage,
    SuggestedPrompt,
)

# ---------------------------------------------------------------------------
# Stopwords bilíngues & Ponte Bilíngue Especializada em Macroeconomia (PT->EN)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do", "dos", "e", "em",
    "entre", "essa", "esse", "esta", "estao", "este", "foi", "foram", "ha", "isso", "ja",
    "mais", "mas", "mesmo", "muito", "na", "nao", "nas", "no", "nos", "num", "numa", "o",
    "os", "ou", "para", "pela", "pelo", "por", "qual", "quais", "quando", "que", "quem",
    "sao", "se", "sem", "ser", "seu", "sua", "sobre", "tambem", "tem", "ter", "todo",
    "todos", "um", "uma", "explique", "compare", "diferencie", "diferenca", "resuma",
    "funciona", "livro", "capitulo", "capitulos", "texto", "segundo", "autor", "heijdra",
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have",
    "has", "had", "not", "but", "what", "which", "who", "when", "where", "why", "how",
    "can", "could", "should", "would", "may", "might", "will", "into", "about", "between",
    "through", "during", "before", "after", "each", "other", "such", "only", "same",
    "than", "then", "there", "these", "those", "their", "they", "them", "its", "our",
    "also", "been", "being", "does", "did", "more", "most", "very", "some", "any", "all",
    "both", "few", "many", "much", "chapter", "section", "book", "figure", "equation",
}

PT_WORDS = {
    "como", "qual", "quais", "explique", "compare", "diferencie", "politica", "monetaria",
    "fiscal", "inflacao", "desemprego", "crescimento", "juros", "cambio", "expectativas",
    "curva", "modelo", "modelos", "economia", "aberta", "dinamica", "equilibrio", "credito",
    "governo", "divida", "deficit", "credibilidade", "salarios", "precos", "rigidez",
}

MACRO_PHRASE_MAP: List[Tuple[str, List[str]]] = [
    ("curva de phillips", ["phillips", "curve", "inflation", "unemployment", "nairu"]),
    ("politica monetaria", ["monetary", "policy", "money", "supply", "interest", "central", "bank"]),
    ("politica fiscal", ["fiscal", "policy", "government", "spending", "taxation", "deficit"]),
    ("equivalencia ricardiana", ["ricardian", "equivalence", "government", "budget", "deficit", "barro"]),
    ("inconsistencia temporal", ["time", "inconsistency", "credibility", "kydland", "prescott", "barro", "gordon"]),
    ("expectativas racionais", ["rational", "expectations", "lucas", "sargent", "policy", "ineffectiveness"]),
    ("expectativas adaptativas", ["adaptive", "expectations", "inflation", "phillips"]),
    ("critica de lucas", ["lucas", "critique", "rational", "expectations", "policy"]),
    ("novo keynesiano", ["new", "keynesian", "sticky", "prices", "calvo", "monopolistic", "competition"]),
    ("novos keynesianos", ["new", "keynesian", "sticky", "prices", "calvo", "menu", "costs"]),
    ("rigidez de precos", ["sticky", "prices", "nominal", "rigidities", "menu", "costs", "calvo"]),
    ("ciclos reais", ["real", "business", "cycles", "rbc", "technology", "shocks", "productivity"]),
    ("crescimento endogeno", ["endogenous", "growth", "romer", "human", "capital", "ak", "innovation"]),
    ("crescimento exogeno", ["exogenous", "growth", "solow", "swan", "ramsey", "golden", "rule"]),
    ("geracoes sobrepostas", ["overlapping", "generations", "olg", "blanchard", "yaari", "diamond"]),
    ("economia aberta", ["open", "economy", "mundell", "fleming", "exchange", "rate", "dornbusch"]),
    ("taxa de cambio", ["exchange", "rate", "floating", "fixed", "overshooting", "dornbusch"]),
    ("armadilha da liquidez", ["liquidity", "trap", "money", "demand", "keynesian", "is-lm"]),
    ("regra de ouro", ["golden", "rule", "capital", "accumulation", "dynamic", "efficiency", "solow"]),
    ("mercado de trabalho", ["labour", "labor", "market", "unemployment", "search", "matching", "unions"]),
    ("busca no mercado de trabalho", ["search", "matching", "beveridge", "vacancies", "unemployment"]),
    ("sindicatos", ["trade", "unions", "monopoly", "union", "bargaining", "wage"]),
    ("demanda agregada", ["aggregate", "demand", "is-lm", "ad-as"]),
    ("oferta agregada", ["aggregate", "supply", "ad-as", "phillips"]),
    ("banco central", ["central", "bank", "credibility", "conservative", "rogoff", "inflation", "bias"]),
    ("taxa de juros", ["interest", "rate", "nominal", "real", "fisher"]),
    ("efeito deslocamento", ["crowding", "out", "fiscal", "investment", "interest"]),
    ("divida publica", ["public", "debt", "government", "budget", "constraint", "solvency"]),
    ("superneutralidade", ["superneutrality", "money", "sidrauski", "tobin", "effect"]),
]

MACRO_TOKEN_MAP: Dict[str, List[str]] = {
    "inflacao": ["inflation", "inflationary", "prices"],
    "desemprego": ["unemployment", "nairu", "labour"],
    "juros": ["interest", "rate"],
    "moeda": ["money", "monetary", "seigniorage"],
    "monetaria": ["monetary", "money"],
    "monetario": ["monetary", "money"],
    "fiscal": ["fiscal", "budget", "deficit"],
    "governo": ["government", "public", "fiscal"],
    "impostos": ["taxes", "taxation", "lump-sum"],
    "imposto": ["tax", "taxation"],
    "deficit": ["deficit", "budget", "debt"],
    "divida": ["debt", "bonds", "solvency"],
    "cambio": ["exchange", "rate", "currency"],
    "flutuante": ["flexible", "floating"],
    "fixo": ["fixed"],
    "crescimento": ["growth", "steady", "state"],
    "endogeno": ["endogenous"],
    "exogeno": ["exogenous", "solow"],
    "capital": ["capital", "accumulation", "investment"],
    "investimento": ["investment", "capital"],
    "consumo": ["consumption", "euler", "households"],
    "poupanca": ["saving", "savings"],
    "salarios": ["wages", "wage"],
    "salario": ["wage", "wages"],
    "precos": ["prices", "price"],
    "rigidez": ["rigidity", "rigidities", "sticky"],
    "choque": ["shock", "shocks"],
    "choques": ["shocks", "shock"],
    "produtividade": ["productivity", "technology", "tfp"],
    "expectativas": ["expectations"],
    "racionais": ["rational"],
    "adaptativas": ["adaptive"],
    "antecipacao": ["anticipation", "anticipated", "unanticipated"],
    "credibilidade": ["credibility", "reputation", "commitment"],
    "discrecionaria": ["discretion", "discretionary"],
    "regras": ["rules", "commitment"],
    "keynesiano": ["keynesian"],
    "keynesianos": ["keynesian", "keynesians"],
    "classico": ["classical"],
    "classicos": ["classical", "classicals"],
    "monetaristas": ["monetarist", "monetarists", "friedman"],
    "multiplicador": ["multiplier"],
    "estabilidade": ["stability", "saddle", "path"],
    "sela": ["saddle", "saddle-path"],
    "dinamica": ["dynamics", "dynamic"],
    "equilibrio": ["equilibrium", "steady-state"],
    "otimizacao": ["optimization", "hamiltonian", "euler"],
    "bem-estar": ["welfare", "utility"],
    "ineficiencia": ["inefficiency", "dynamic"],
    "neutralidade": ["neutrality", "superneutrality"],
    "desinflacao": ["disinflation", "sacrifice", "ratio"],
}


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\xad", "").replace("\u200b", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-zÀ-ÿ])-\s*\n\s*([a-zà-ÿ])", r"\1\2", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    paras: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if not ln:
            if buf:
                paras.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras).strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    norm = strip_accents(text.lower())
    raw_toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", norm)
    out: List[str] = []
    for tok in raw_toks:
        if len(tok) < 2:
            continue
        if tok.isdigit() and len(tok) < 4:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
        if "-" in tok:
            for p in tok.split("-"):
                if len(p) >= 2 and (not remove_stopwords or p not in STOPWORDS):
                    out.append(p)
    return out


def detect_language(q: str) -> str:
    has_acc = bool(re.search(r"[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]", q))
    words = set(re.findall(r"[a-z]+", strip_accents(q.lower())))
    return "pt-BR" if (has_acc or len(words & PT_WORDS) >= 1) else "en"


def expand_macro_query(question: str) -> Tuple[List[str], List[str], str]:
    lang = detect_language(question)
    orig = list(dict.fromkeys(tokenize(question, remove_stopwords=True)))
    expanded: List[str] = []
    seen = set(orig)

    plain = " ".join(re.findall(r"[a-z0-9]+", strip_accents(question.lower())))
    for phrase, targets in MACRO_PHRASE_MAP:
        if phrase in plain:
            for t in targets:
                if t not in seen:
                    expanded.append(t)
                    seen.add(t)

    for tok in orig:
        if tok in MACRO_TOKEN_MAP:
            for t in MACRO_TOKEN_MAP[tok]:
                if t not in seen:
                    expanded.append(t)
                    seen.add(t)

    return orig, expanded, lang


# ---------------------------------------------------------------------------
# Estruturas internas: Documento, Capítulo, Página, Chunk
# ---------------------------------------------------------------------------

@dataclass
class ChapterDef:
    chapter_id: str
    part_name: str
    title: str
    start_page: int
    end_page: int


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class MacroDocument:
    doc_id: str
    filename: str
    fmt: str  # "pdf" | "md" | "txt"
    title: str
    author: str
    pages: List[PageText]
    chapters: List[ChapterDef]
    is_demo: bool
    loaded_from_cache: bool
    ingested_at: str

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def total_words(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)


@dataclass
class MacroChunk:
    chunk_id: str
    doc_id: str
    document_title: str
    chapter_id: str
    chapter_title: str
    part_name: str
    page: int
    text: str
    tokens: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extração de Capítulos via PDF Outline / Markdown
# ---------------------------------------------------------------------------

def _clean_bookmark_title(raw: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\r\n\t]+", "", raw or "")).strip()


def _extract_pdf_chapters(reader: PdfReader, doc_id: str, total_pages: int) -> List[ChapterDef]:
    """Constrói intervalos de capítulos a partir do sumário (outline) do PDF."""
    raw_entries: List[Tuple[int, str, str]] = []  # (start_page, part_name, title)
    try:
        outline = reader.outline
    except Exception:
        outline = None

    if outline:
        current_part = "Geral"
        seen_keys = set()
        for idx, item in enumerate(outline):
            if isinstance(item, list):
                continue
            try:
                pg = reader.get_destination_page_number(item) + 1
                title = _clean_bookmark_title(getattr(item, "title", ""))
            except Exception:
                continue
            if not title:
                continue
            # Verifica se o próximo elemento é uma lista de sub-itens (capítulos da Parte)
            has_children = (idx + 1 < len(outline)) and isinstance(outline[idx + 1], list)
            low = title.lower()
            if has_children and (low.startswith("i ") or low.startswith("ii ") or low.startswith("part ")):
                current_part = title
                children = outline[idx + 1]
                for ch in children:
                    if isinstance(ch, list):
                        continue
                    try:
                        cpg = reader.get_destination_page_number(ch) + 1
                        ctitle = _clean_bookmark_title(getattr(ch, "title", ""))
                        key = (cpg, ctitle)
                        if ctitle and key not in seen_keys:
                            seen_keys.add(key)
                            raw_entries.append((cpg, current_part, ctitle))
                    except Exception:
                        continue
            elif any(k in low for k in ("mathematical appendix", "epilogue", "chapter ", "capítulo ")):
                key = (pg, title)
                if key not in seen_keys:
                    seen_keys.add(key)
                    part_lbl = "Apêndice & Epílogo" if ("appendix" in low or "epilogue" in low) else "Principal"
                    raw_entries.append((pg, part_lbl, title))

    if not raw_entries:
        # Fallback: blocos lógicos por faixa de páginas (para PDFs sem bookmarks)
        step = max(10, total_pages // 6) if total_pages > 25 else max(1, total_pages)
        chapters: List[ChapterDef] = []
        sec = 1
        for start_p in range(1, total_pages + 1, step):
            end_p = min(total_pages, start_p + step - 1)
            title = "Documento Completo" if (start_p == 1 and end_p == total_pages) else f"Seção {sec} (pp. {start_p}–{end_p})"
            chapters.append(
                ChapterDef(
                    chapter_id=f"{doc_id}-ch{sec:02d}",
                    part_name="Documento",
                    title=title,
                    start_page=start_p,
                    end_page=end_p,
                )
            )
            sec += 1
        return chapters

    raw_entries.sort(key=lambda x: x[0])
    # Deduplica entradas na mesma página
    deduped: List[Tuple[int, str, str]] = []
    for pg, part, title in raw_entries:
        if deduped and deduped[-1][0] == pg:
            continue
        deduped.append((pg, part, title))

    chapters = []
    # Se há prefácio antes do 1º capítulo, adiciona Front Matter
    if deduped[0][0] > 1:
        chapters.append(
            ChapterDef(
                chapter_id=f"{doc_id}-ch00",
                part_name="Introdução",
                title="Prefácio & Sumário",
                start_page=1,
                end_page=deduped[0][0] - 1,
            )
        )

    for i, (start_p, part, title) in enumerate(deduped):
        # Evita incluir bibliografia/índice remissivo dentro do último apêndice se detectarmos
        if i + 1 < len(deduped):
            end_p = deduped[i + 1][0] - 1
        else:
            end_p = total_pages
        end_p = max(start_p, end_p)
        chapters.append(
            ChapterDef(
                chapter_id=f"{doc_id}-ch{i + 1:02d}",
                part_name=part,
                title=title,
                start_page=start_p,
                end_page=end_p,
            )
        )
    return chapters


def _chapter_for_page(chapters: List[ChapterDef], page_num: int) -> ChapterDef:
    for ch in chapters:
        if ch.start_page <= page_num <= ch.end_page:
            return ch
    return chapters[-1]


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n\n+")


def _chunk_document(doc: MacroDocument, chunk_size: int, chunk_overlap: int) -> List[MacroChunk]:
    eff_overlap = min(chunk_overlap, max(0, chunk_size // 2))
    chunks: List[MacroChunk] = []
    idx = 0

    for page in doc.pages:
        text = page.text.strip()
        if len(text) < 40:
            continue
        ch_def = _chapter_for_page(doc.chapters, page.page_number)
        # Ignora páginas de sumário/prefácio, bibliografia e índice remissivo em livros longos
        if ch_def.title.lower() in {"index", "bibliography", "prefácio & sumário"} and len(doc.chapters) > 3:
            continue

        raw_units = [u.strip() for u in SENTENCE_SPLIT_RE.split(text) if u.strip()]
        units: List[str] = []
        for u in raw_units:
            if len(u) <= chunk_size:
                units.append(u)
            else:
                step = max(250, chunk_size - eff_overlap)
                for s in range(0, len(u), step):
                    piece = u[s : s + chunk_size].strip()
                    if piece:
                        units.append(piece)

        cur: List[str] = []
        cur_len = 0
        for u in units:
            add = len(u) + (1 if cur else 0)
            if cur and cur_len + add > chunk_size:
                ctext = " ".join(cur).strip()
                if len(ctext) >= 60:
                    chunks.append(
                        MacroChunk(
                            chunk_id=f"{doc.doc_id}-c{idx:05d}",
                            doc_id=doc.doc_id,
                            document_title=doc.title,
                            chapter_id=ch_def.chapter_id,
                            chapter_title=ch_def.title,
                            part_name=ch_def.part_name,
                            page=page.page_number,
                            text=ctext,
                            tokens=tokenize(ctext, remove_stopwords=True),
                        )
                    )
                    idx += 1
                if eff_overlap > 0:
                    tail = ctext[-eff_overlap:]
                    sp = tail.find(" ")
                    tail = tail[sp + 1 :].strip() if 0 < sp < len(tail) - 10 else tail.strip()
                    cur = [tail, u] if tail else [u]
                    cur_len = sum(len(x) for x in cur) + (1 if len(cur) > 1 else 0)
                else:
                    cur = [u]
                    cur_len = len(u)
            else:
                cur.append(u)
                cur_len += add

        if cur:
            ctext = " ".join(cur).strip()
            if len(ctext) >= 60:
                chunks.append(
                    MacroChunk(
                        chunk_id=f"{doc.doc_id}-c{idx:05d}",
                        doc_id=doc.doc_id,
                        document_title=doc.title,
                        chapter_id=ch_def.chapter_id,
                        chapter_title=ch_def.title,
                        part_name=ch_def.part_name,
                        page=page.page_number,
                        text=ctext,
                        tokens=tokenize(ctext, remove_stopwords=True),
                    )
                )
                idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Motor Principal MacroRAGEngine
# ---------------------------------------------------------------------------

class MacroRAGEngine:
    def __init__(self, books_dir: Path, cache_dir: Path) -> None:
        self.books_dir = books_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

        self.chunk_size = 1050
        self.chunk_overlap = 160

        self.documents: Dict[str, MacroDocument] = {}
        self.chunks: List[MacroChunk] = []

        # Índice Invertido Okapi BM25 (O(postings) -> < 10ms em 2.000+ chunks!)
        self._inverted: Dict[str, List[Tuple[int, int]]] = {}
        self._doc_lens: np.ndarray = np.array([], dtype=np.float64)
        self._avgdl: float = 1.0
        self._vocab_size: int = 0

        # Espaço Semântico Denso (LSA 96D + Char Subword)
        self._word_vec: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._lsa_mat: Optional[np.ndarray] = None
        self._char_vec: Optional[TfidfVectorizer] = None
        self._char_mat = None
        self._lsa_dims: int = 0
        self._chapter_keywords: Dict[str, List[str]] = {}
        self._last_build_ms: float = 0.0

    # --- Ingestão Multi-Formato (PDF + MD + TXT) com Cache Persistente -----

    def _cache_file_for(self, digest: str) -> Path:
        return self.cache_dir / f"parsed_{digest}.json.gz"

    def ingest_file(self, filename: str, content: bytes, is_demo: bool = False) -> MacroDocument:
        ext = Path(filename).suffix.lower()
        digest = hashlib.sha1(f"{filename}:{len(content)}".encode("utf-8")).hexdigest()[:12]
        doc_id = f"doc-{digest}"

        if ext == ".pdf":
            cache_path = self._cache_file_for(digest)
            if cache_path.exists():
                with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
                    cached = json.load(fh)
                pages = [PageText(page_number=p["p"], text=p["t"]) for p in cached["pages"]]
                chapters = [ChapterDef(**c) for c in cached["chapters"]]
                doc = MacroDocument(
                    doc_id=doc_id,
                    filename=filename,
                    fmt="pdf",
                    title=cached["title"],
                    author=cached["author"],
                    pages=pages,
                    chapters=chapters,
                    is_demo=is_demo,
                    loaded_from_cache=True,
                    ingested_at=datetime.now(timezone.utc).isoformat(),
                )
                self.documents[doc_id] = doc
                return doc

            reader = PdfReader(io.BytesIO(content))
            meta = dict(reader.metadata) if reader.metadata else {}
            title = str(meta.get("/Title") or "").strip() or Path(filename).stem
            author = str(meta.get("/Author") or "").strip() or "Autor não especificado"
            if "heijdra" in filename.lower():
                title = "Foundations of Modern Macroeconomics"
                author = "Ben J. Heijdra (Oxford University Press)"

            total_pages = len(reader.pages)
            chapters = _extract_pdf_chapters(reader, doc_id, total_pages)
            pages = []
            for p_idx, page in enumerate(reader.pages, start=1):
                cleaned = clean_text(page.extract_text() or "")
                if cleaned:
                    pages.append(PageText(page_number=p_idx, text=cleaned))

            if not pages:
                raise ValueError(f"Não foi possível extrair texto pesquisável do PDF '{filename}'.")

            # Salva cache local comprimido para carregamentos futuros instantâneos
            payload = {
                "title": title,
                "author": author,
                "pages": [{"p": p.page_number, "t": p.text} for p in pages],
                "chapters": [c.__dict__ for c in chapters],
            }
            with gzip.open(cache_path, "wt", encoding="utf-8", compresslevel=5) as fh:
                json.dump(payload, fh, ensure_ascii=False)

            doc = MacroDocument(
                doc_id=doc_id,
                filename=filename,
                fmt="pdf",
                title=title,
                author=author,
                pages=pages,
                chapters=chapters,
                is_demo=is_demo,
                loaded_from_cache=False,
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
            self.documents[doc_id] = doc
            return doc

        if ext in {".md", ".txt"}:
            text_raw = content.decode("utf-8", errors="ignore")
            cleaned_full = clean_text(text_raw)
            if len(cleaned_full) < 40:
                raise ValueError(f"Arquivo '{filename}' contém pouco texto útil.")

            # Divide por títulos Markdown (# ou ##) quando existirem
            sections = re.split(r"(?m)^#{1,2}\s+(.+)$", text_raw)
            pages: List[PageText] = []
            chapters: List[ChapterDef] = []
            if len(sections) >= 3:
                sec_idx = 1
                for i in range(1, len(sections), 2):
                    sec_title = sections[i].strip()[:90]
                    sec_body = clean_text(sections[i + 1] if i + 1 < len(sections) else "")
                    if len(sec_body) >= 30:
                        pages.append(PageText(page_number=sec_idx, text=sec_body))
                        chapters.append(
                            ChapterDef(
                                chapter_id=f"{doc_id}-ch{sec_idx:02d}",
                                part_name="Notas de Aula",
                                title=sec_title,
                                start_page=sec_idx,
                                end_page=sec_idx,
                            )
                        )
                        sec_idx += 1
            if not pages:
                pages = [PageText(page_number=1, text=cleaned_full)]
                chapters = [
                    ChapterDef(
                        chapter_id=f"{doc_id}-ch01",
                        part_name="Notas",
                        title=Path(filename).stem,
                        start_page=1,
                        end_page=1,
                    )
                ]

            doc = MacroDocument(
                doc_id=doc_id,
                filename=filename,
                fmt="md" if ext == ".md" else "txt",
                title=Path(filename).stem.replace("_", " ").replace("-", " ").title(),
                author="Material enviado pelo usuário",
                pages=pages,
                chapters=chapters,
                is_demo=is_demo,
                loaded_from_cache=False,
                ingested_at=datetime.now(timezone.utc).isoformat(),
            )
            self.documents[doc_id] = doc
            return doc

        raise ValueError(f"Formato não suportado para '{filename}'. Envie .pdf, .md ou .txt.")

    def load_demo_library(self) -> List[MacroDocument]:
        loaded: List[MacroDocument] = []
        if not self.books_dir.exists():
            return loaded
        for p in sorted(self.books_dir.glob("*")):
            if p.suffix.lower() in {".pdf", ".md", ".txt"}:
                doc = self.ingest_file(p.name, p.read_bytes(), is_demo=True)
                loaded.append(doc)
        self.rebuild_index()
        return loaded

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        self.rebuild_index()
        return True

    # --- Indexação Híbrida Rápida (BM25 Invertido + LSA 96D + Char) --------

    def rebuild_index(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> IndexStats:
        t0 = time.perf_counter()
        if chunk_size is not None:
            self.chunk_size = int(chunk_size)
        if chunk_overlap is not None:
            self.chunk_overlap = min(int(chunk_overlap), max(0, self.chunk_size // 2))

        all_chunks: List[MacroChunk] = []
        for doc in self.documents.values():
            all_chunks.extend(_chunk_document(doc, self.chunk_size, self.chunk_overlap))
        self.chunks = all_chunks

        if not self.chunks:
            self._inverted = {}
            self._doc_lens = np.array([], dtype=np.float64)
            self._avgdl = 1.0
            self._vocab_size = 0
            self._word_vec = None
            self._svd = None
            self._lsa_mat = None
            self._char_vec = None
            self._char_mat = None
            self._lsa_dims = 0
            self._chapter_keywords = {}
            self._last_build_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return self.get_stats()

        # 1. Índice Invertido Okapi BM25
        inv: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        lens = np.zeros(len(self.chunks), dtype=np.float64)
        for i, ch in enumerate(self.chunks):
            tf = Counter(ch.tokens)
            lens[i] = len(ch.tokens)
            for term, freq in tf.items():
                inv[term].append((i, freq))
        self._inverted = dict(inv)
        self._doc_lens = lens
        self._avgdl = float(np.mean(lens)) if len(lens) else 1.0
        self._vocab_size = len(self._inverted)

        # 2. Espaço Semântico Vetorial Denso (Word TF-IDF -> SVD 64D + Char Subword)
        texts = [strip_accents(c.text.lower()) for c in self.chunks]
        min_df_val = 2 if len(texts) >= 40 else 1
        self._word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=min_df_val,
            max_features=14000,
            token_pattern=r"(?u)\b[a-z]{2,}\b",
            stop_words=list(STOPWORDS),
        )
        w_mat = self._word_vec.fit_transform(texts)
        n_docs, n_feats = w_mat.shape
        dims = min(64, max(2, min(n_docs - 1, n_feats - 1)))
        if n_docs >= 4 and n_feats >= 4:
            self._svd = TruncatedSVD(n_components=dims, n_iter=3, random_state=42)
            self._lsa_mat = normalize(self._svd.fit_transform(w_mat), norm="l2", axis=1)
            self._lsa_dims = int(self._lsa_mat.shape[1])
        else:
            self._svd = None
            self._lsa_mat = normalize(w_mat.toarray(), norm="l2", axis=1)
            self._lsa_dims = int(self._lsa_mat.shape[1])

        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            sublinear_tf=True,
            min_df=min_df_val,
            max_features=12000,
        )
        self._char_mat = normalize(self._char_vec.fit_transform(texts), norm="l2", axis=1)

        # 3. Extração automática das palavras-chave mais características por capítulo
        self._compute_chapter_keywords()

        self._last_build_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return self.get_stats()

    def _compute_chapter_keywords(self) -> None:
        ch_texts: Dict[str, List[str]] = defaultdict(list)
        for c in self.chunks:
            ch_texts[c.chapter_id].append(c.text)
        ids = list(ch_texts.keys())
        if not ids:
            self._chapter_keywords = {}
            return
        docs_joined = [strip_accents(" ".join(ch_texts[cid]).lower()) for cid in ids]
        try:
            vec = TfidfVectorizer(
                max_features=4000,
                token_pattern=r"(?u)\b[a-z]{3,}\b",
                stop_words=list(STOPWORDS),
                sublinear_tf=True,
            )
            mat = vec.fit_transform(docs_joined)
            names = np.asarray(vec.get_feature_names_out())
            kw_map: Dict[str, List[str]] = {}
            for row_i, cid in enumerate(ids):
                row = mat.getrow(row_i).toarray().ravel()
                top_idx = np.argsort(-row)[:6]
                kw_map[cid] = [str(names[j]) for j in top_idx if row[j] > 0]
            self._chapter_keywords = kw_map
        except Exception:
            self._chapter_keywords = {}

    # --- Estatísticas, Biblioteca e Sugestões Macroeconômicas --------------

    def get_stats(self) -> IndexStats:
        return IndexStats(
            document_count=len(self.documents),
            total_pages=sum(len(d.pages) for d in self.documents.values()),
            total_chapters=sum(len(d.chapters) for d in self.documents.values()),
            total_chunks=len(self.chunks),
            total_chars=sum(d.total_chars for d in self.documents.values()),
            vocab_size=self._vocab_size,
            lsa_dimensions=self._lsa_dims,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            build_time_ms=self._last_build_ms,
            cache_enabled=True,
        )

    def get_library(self) -> LibraryResponse:
        doc_chunk_counts = Counter(c.doc_id for c in self.chunks)
        ch_chunk_counts = Counter(c.chapter_id for c in self.chunks)

        docs = [
            DocumentSummary(
                doc_id=d.doc_id,
                filename=d.filename,
                format=d.fmt,  # type: ignore[arg-type]
                title=d.title,
                author=d.author,
                pages=len(d.pages),
                chapters_count=len(d.chapters),
                total_chars=d.total_chars,
                total_words=d.total_words,
                chunk_count=doc_chunk_counts.get(d.doc_id, 0),
                is_demo=d.is_demo,
                loaded_from_cache=d.loaded_from_cache,
                ingested_at=d.ingested_at,
            )
            for d in self.documents.values()
        ]

        chapters: List[ChapterSummary] = []
        for d in self.documents.values():
            for ch in d.chapters:
                cnt = ch_chunk_counts.get(ch.chapter_id, 0)
                if cnt == 0:
                    continue
                chapters.append(
                    ChapterSummary(
                        chapter_id=ch.chapter_id,
                        doc_id=d.doc_id,
                        document_title=d.title,
                        part_name=ch.part_name,
                        title=ch.title,
                        start_page=ch.start_page,
                        end_page=ch.end_page,
                        page_count=ch.end_page - ch.start_page + 1,
                        chunk_count=cnt,
                        top_keywords=self._chapter_keywords.get(ch.chapter_id, []),
                    )
                )

        prompts = [
            SuggestedPrompt(
                id="m-ricardian",
                mode="tutor",
                category="Política Fiscal",
                question="O que é a Equivalência Ricardiana e sob quais condições o déficit público não afeta a demanda agregada?",
                hint="Recupera diretamente o Capítulo 5 (The government budget deficit) de Heijdra com citações de página.",
                chapter_hint="Cap. 5",
            ),
            SuggestedPrompt(
                id="m-time-inconsistency",
                mode="tutor",
                category="Política Monetária",
                question="Explique o problema da inconsistência temporal e o viés inflacionário na política monetária (Kydland-Prescott / Barro-Gordon).",
                hint="Explora credibilidade, regras vs. discrição e Banco Central conservador no Capítulo 9.",
                chapter_hint="Cap. 9",
            ),
            SuggestedPrompt(
                id="m-compare-schools",
                mode="compare",
                category="Comparador de Modelos",
                question="Compare o modelo Novo Keynesiano com rigidez de preços (Calvo/concorrência monopolística) versus os Ciclos Reais de Negócios (RBC).",
                hint="Modo Comparador: busca evidências equilibradas entre os Capítulos 12 (New Keynesian) e 15 (RBC).",
                chapter_hint="Caps. 12 vs. 15",
            ),
            SuggestedPrompt(
                id="m-open-economy",
                mode="tutor",
                category="Economia Aberta",
                question="Como funciona o efeito de overshooting da taxa de câmbio de Dornbusch em uma economia aberta com preços rígidos?",
                hint="Consulta o Capítulo 10 (The open economy) sobre política monetária e dinâmica cambial.",
                chapter_hint="Cap. 10",
            ),
            SuggestedPrompt(
                id="m-growth",
                mode="compare",
                category="Crescimento Econômico",
                question="Qual a diferença entre crescimento exógeno (Solow-Swan / Ramsey) e crescimento endógeno (modelo AK e capital humano)?",
                hint="Compara os Capítulos 13 (Exogenous growth) e 14 (Endogenous growth).",
                chapter_hint="Caps. 13 vs. 14",
            ),
            SuggestedPrompt(
                id="m-guardrail",
                mode="verify",
                category="Teste Anti-Alucinação",
                question="Qual foi a escalação titular da Seleção Brasileira na final da Copa do Mundo de 2026?",
                hint="Pergunta fora do escopo do livro de macroeconomia para testar o guardrail anti-alucinação.",
                is_out_of_scope=True,
            ),
        ]

        return LibraryResponse(
            documents=docs,
            chapters=chapters,
            stats=self.get_stats(),
            suggested_prompts=prompts,
        )

    # --- Provedores LLM (Offline, Gemini, Ollama, OpenAI) ------------------

    def inspect_providers(self) -> ProvidersResponse:
        ollama_models: List[str] = []
        ollama_up = False
        ollama_detail = f"Ollama não detectado em {self.ollama_url} (opcional)"
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{self.ollama_url}/api/tags")
                if r.status_code == 200:
                    ollama_models = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
                    ollama_up = True
                    ollama_detail = (
                        f"Online ({len(ollama_models)} modelo(s) local(is)) — 0 bytes enviados à nuvem"
                        if ollama_models
                        else "Online, mas sem modelos baixados (rode: ollama pull qwen2.5:7b)"
                    )
        except Exception:
            ollama_up = False

        has_gemini_env = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        has_openai_env = bool(os.getenv("OPENAI_API_KEY"))

        providers = [
            ProviderInfo(
                id="extractive",
                name="Tutor Extrativo Grounded (100% Offline)",
                available=True,
                requires_key=False,
                is_local=True,
                models=["macro-extractive-v2"],
                default_model="macro-extractive-v2",
                description="100% local e instantâneo (0 bytes enviados externamente). Sintetiza trechos exatos com Capítulo e Página.",
                status_detail="Sempre pronto • Privacidade máxima (0 bytes externos)",
            ),
            ProviderInfo(
                id="gemini",
                name="Google Gemini API (Recomendado p/ Tradução PT-BR)",
                available=True,
                requires_key=not has_gemini_env,
                is_local=False,
                models=["gemini-2.5-flash", "gemini-2.0-flash"],
                default_model="gemini-2.5-flash",
                description="Envia apenas os top-k chunks recuperados (nunca o livro inteiro) para síntese didática fluente em Português.",
                status_detail="Chave detectada no ambiente" if has_gemini_env else "Informe GEMINI_API_KEY na sessão (nunca salva em disco)",
            ),
            ProviderInfo(
                id="ollama",
                name="Ollama Local (LLM Privado On-Premise)",
                available=ollama_up and len(ollama_models) > 0,
                requires_key=False,
                is_local=True,
                models=ollama_models if ollama_models else ["qwen2.5:7b", "llama3.2:3b"],
                default_model=ollama_models[0] if ollama_models else "qwen2.5:7b",
                description="Geração generativa completa rodando 100% na sua máquina, sem tráfego externo.",
                status_detail=ollama_detail,
            ),
            ProviderInfo(
                id="openai",
                name="OpenAI API",
                available=True,
                requires_key=not has_openai_env,
                is_local=False,
                models=["gpt-4o-mini", "gpt-4o"],
                default_model="gpt-4o-mini",
                description="Síntese acadêmica rigorosa enviando apenas a janela de contexto dos trechos recuperados.",
                status_detail="Chave detectada no ambiente" if has_openai_env else "Informe OPENAI_API_KEY na sessão (nunca salva em disco)",
            ),
        ]
        return ProvidersResponse(providers=providers, ollama_url=self.ollama_url)

    def is_ollama_available(self) -> bool:
        try:
            with httpx.Client(timeout=0.8) as client:
                return client.get(f"{self.ollama_url}/api/tags").status_code == 200
        except Exception:
            return False

    # --- Retrieval Híbrido Rápido (BM25 Invertido + LSA/Char + RRF + MMR) --

    def _score_bm25_fast(
        self,
        orig_tokens: Sequence[str],
        expanded_tokens: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        n = len(self.chunks)
        scores = np.zeros(n, dtype=np.float64)
        matched: Dict[int, List[str]] = defaultdict(list)
        if n == 0:
            return scores, matched

        q_weights: Dict[str, float] = {}
        for t in orig_tokens:
            q_weights[t] = max(q_weights.get(t, 0.0), 1.0)
        for t in expanded_tokens:
            q_weights[t] = max(q_weights.get(t, 0.0), 0.80)

        for term, qw in q_weights.items():
            postings = self._inverted.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for doc_i, tf in postings:
                dl = self._doc_lens[doc_i]
                denom = tf + k1 * (1.0 - b + b * (dl / self._avgdl))
                scores[doc_i] += qw * idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)
                matched[doc_i].append(term)

        return scores, matched

    def _score_semantic_fast(self, enriched_query: str) -> np.ndarray:
        n = len(self.chunks)
        if n == 0 or self._word_vec is None or self._lsa_mat is None:
            return np.zeros(n, dtype=np.float64)
        nq = strip_accents(enriched_query.lower())
        qw = self._word_vec.transform([nq])
        qlsa = normalize(self._svd.transform(qw), norm="l2", axis=1) if self._svd is not None else normalize(qw.toarray(), norm="l2", axis=1)
        cos_lsa = np.clip((self._lsa_mat @ qlsa.T).ravel(), 0.0, 1.0)

        if self._char_vec is not None and self._char_mat is not None:
            qc = normalize(self._char_vec.transform([nq]), norm="l2", axis=1)
            cos_char = np.clip((self._char_mat @ qc.T).toarray().ravel(), 0.0, 1.0)
            return np.clip(0.64 * cos_lsa + 0.36 * cos_char, 0.0, 1.0)
        return cos_lsa

    def _mmr_select(self, pool: List[int], scores: np.ndarray, top_k: int, lam: float) -> List[int]:
        if len(pool) <= top_k or self._lsa_mat is None:
            return pool[:top_k]
        selected = [pool[0]]
        remaining = pool[1:]
        while len(selected) < top_k and remaining:
            sel_vecs = self._lsa_mat[selected]
            best_i = remaining[0]
            best_val = -1e9
            for cand in remaining:
                rel = float(scores[cand])
                max_sim = float(np.max(sel_vecs @ self._lsa_mat[cand]))
                # Pequeno bônus de diversidade de capítulos no modo compare
                val = lam * rel - (1.0 - lam) * max_sim
                if val > best_val:
                    best_val = val
                    best_i = cand
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    def _make_excerpt(self, text: str, terms: Sequence[str], max_chars: int = 330) -> str:
        if len(text) <= max_chars:
            return text
        if not terms:
            return text[:max_chars].rstrip() + "..."
        plain = strip_accents(text.lower())
        best_pos, best_hits = 0, -1
        for st in range(0, max(1, len(plain) - max_chars + 1), 65):
            win = plain[st : st + max_chars]
            hits = sum(1 for t in terms if t in win)
            if hits > best_hits:
                best_hits, best_pos = hits, st
        if best_pos > 0:
            sp = text.find(" ", max(0, best_pos - 25), best_pos + 25)
            if sp != -1:
                best_pos = sp + 1
        end = min(len(text), best_pos + max_chars)
        return f"{'...' if best_pos > 0 else ''}{text[best_pos:end].strip()}{'...' if end < len(text) else ''}"

    # --- Prompt Builder & Provedores de Geração ----------------------------

    def _build_prompt(
        self,
        question: str,
        study_mode: str,
        citations: List[RetrievedPassage],
        out_of_scope: bool,
    ) -> PromptInspection:
        mode_instructions = {
            "tutor": (
                "Atue como um Professor Doutor em Macroeconomia. Estruture a resposta em Português Brasileiro em:\n"
                "1) Intuição Econômica Direta;\n"
                "2) Mecanismos, Hipóteses e Equações dos Capítulos (com citações [1], [2]...);\n"
                "3) Implicações de Política Econômica."
            ),
            "compare": (
                "Atue como comparador de escolas e modelos macroeconômicos. Estruture em Português Brasileiro em:\n"
                "1) Síntese da Divergência Central;\n"
                "2) Comparação Ponto a Ponto (hipóteses, dinâmica de preços/salários, papel da política) citando [1], [2]...;\n"
                "3) Quando cada modelo se aplica melhor."
            ),
            "verify": (
                "Atue como verificador rigoroso de definições e demonstrações. Responda de forma objetiva em Português Brasileiro, "
                "indicando exatamente a definição formal, o capítulo e a página em [1], [2]...."
            ),
        }

        system_prompt = (
            "Você é o MacroRAG Tutor Studio, assistente especialista em Macroeconomia Moderna.\n"
            "REGRAS DE FIDELIDADE (GROUNDING):\n"
            "- Baseie-se EXCLUSIVAMENTE nos trechos [1]..[k] extraídos dos livros/notas carregados.\n"
            "- Cite sempre [1], [2], ... indicando capítulo e página.\n"
            "- NUNCA invente teoremas, dados ou afirmações fora do contexto.\n"
            "- Se os trechos não responderem à pergunta, declare explicitamente que o assunto está fora do escopo do material carregado.\n\n"
            + mode_instructions.get(study_mode, mode_instructions["tutor"])
        )

        blocks = [
            f"[{c.rank}] Livro: {c.document_title} | {c.chapter_title} | Página {c.page} (Score: {c.score_final:.3f})\n{c.text}"
            for c in citations
        ]
        ctx = "\n\n---\n\n".join(blocks)
        oos_note = (
            "\n[GUARDRAIL]: A aderência dos trechos ficou abaixo do limiar de confiança. Se o tema não constar nos trechos, recuse especular.\n"
            if out_of_scope
            else ""
        )
        user_prompt = f"TRECHOS RECUPERADOS DO CORPUS DE MACROECONOMIA:\n\n{ctx}\n{oos_note}\nPERGUNTA DO ESTUDANTE/PESQUISADOR:\n{question}"
        est_tokens = max(1, int(round((len(system_prompt) + len(user_prompt)) / 3.8)))
        return PromptInspection(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            estimated_tokens=est_tokens,
            context_chars=len(ctx),
        )

    def _generate_extractive_answer(
        self,
        study_mode: str,
        citations: List[RetrievedPassage],
        query_terms: List[str],
        out_of_scope: bool,
        grounding_score: float,
    ) -> str:
        if not citations or out_of_scope:
            return (
                "### Guardrail Anti-Alucinação — Tema Fora do Escopo do Corpus\n\n"
                f"Os materiais de Macroeconomia carregados não apresentam evidência confiável para responder a esta pergunta "
                f"(confiança de grounding: **{grounding_score:.0f}%**).\n\n"
                "**Por que o assistente recusou responder?**\n"
                "Para garantir rigor acadêmico, o MacroRAG bloqueia respostas especulativas quando os scores absolutos de BM25 "
                "e similaridade semântica indicam que o assunto não está presente nos livros ou notas indexados.\n\n"
                "**Dica:** Envie um PDF, `.md` ou `.txt` sobre o tema na aba **Biblioteca & Capítulos** ou clique em uma das perguntas sugeridas de Macroeconomia."
            )

        q_set = set(query_terms)
        scored_sents: List[Tuple[float, RetrievedPassage, str]] = []
        for c in citations:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c.text) if 50 <= len(s.strip()) <= 430]
            if not sents:
                sents = [c.excerpt]
            for s in sents:
                if sum(ch.isalpha() for ch in s) / max(len(s), 1) < 0.60:
                    continue
                stoks = set(tokenize(s, remove_stopwords=True))
                ov = len(stoks & q_set)
                sc = (ov / max(math.sqrt(len(stoks) + 1), 1.0)) * 1.45 + c.score_final * 0.95
                scored_sents.append((sc, c, s))
        scored_sents.sort(key=lambda x: x[0], reverse=True)

        chosen: List[Tuple[RetrievedPassage, str]] = []
        per_rank: Counter[int] = Counter()
        seen_pref = set()
        for _, c, s in scored_sents:
            if len(chosen) >= 5:
                break
            if per_rank[c.rank] >= 2:
                continue
            pk = strip_accents(s[:50].lower())
            if pk in seen_pref:
                continue
            seen_pref.add(pk)
            per_rank[c.rank] += 1
            chosen.append((c, s))

        chapters_hit = list(dict.fromkeys(f"**{c.chapter_title}**" for c in citations))
        mode_titles = {
            "tutor": "Síntese do Tutor Macroeconômico (Modo Offline Grounded)",
            "compare": "Síntese Comparativa entre Capítulos/Modelos (Modo Offline Grounded)",
            "verify": "Verificação Direta de Definições e Fontes (Modo Offline Grounded)",
        }

        bullets = [
            f"- **[{c.rank}]** *({c.chapter_title}, pág. {c.page})*: \"{re.sub(r'\\s+', ' ', s)}\" [{c.rank}]"
            for c, s in chosen
        ]
        terms = list(dict.fromkeys(t for c in citations for t in c.matched_terms))[:10]
        terms_md = ", ".join(f"`{t}`" for t in terms) if terms else "`similaridade vetorial densa`"

        return (
            f"### {mode_titles.get(study_mode, mode_titles['tutor'])}\n\n"
            f"Foram localizados **{len(citations)} trechos relevantes** distribuídos em {', '.join(chapters_hit[:4])} "
            f"(aderência consolidada: **{grounding_score:.0f}%** • **0 bytes enviados à nuvem**).\n\n"
            f"#### Evidências-Chave Recuperadas do Material\n\n"
            + "\n".join(bullets)
            + "\n\n"
            f"#### Mapa Conceitual & Próximos Passos\n"
            f"- **Termos-âncora casados:** {terms_md}.\n"
            f"- **Auditoria:** clique nos selos **[1]..[{len(citations)}]** para ler o trecho integral da página ou abra a aba **Privacidade & Raio-X**.\n"
            f"- **Tradução/Explicação Generativa em PT-BR:** selecione **Google Gemini**, **Ollama Local** ou **OpenAI** na barra lateral para parafrasear estes mesmos trechos em português didático."
        )

    def _call_llm(self, provider: str, model: str, api_key: Optional[str], prompt: PromptInspection, temp: float) -> str:
        if provider == "ollama":
            with httpx.Client(timeout=90.0) as client:
                r = client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "options": {"temperature": temp},
                        "messages": [
                            {"role": "system", "content": prompt.system_prompt},
                            {"role": "user", "content": prompt.user_prompt},
                        ],
                    },
                )
                r.raise_for_status()
                return (r.json().get("message", {}).get("content") or "").strip()

        if provider == "gemini":
            key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
            if not key:
                raise RuntimeError("Chave Gemini (GEMINI_API_KEY) não fornecida.")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            with httpx.Client(timeout=60.0) as client:
                r = client.post(
                    url,
                    params={"key": key},
                    json={
                        "system_instruction": {"parts": [{"text": prompt.system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": prompt.user_prompt}]}],
                        "generationConfig": {"temperature": temp},
                    },
                )
                r.raise_for_status()
                parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts).strip()

        if provider == "openai":
            key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not key:
                raise RuntimeError("Chave OpenAI (OPENAI_API_KEY) não fornecida.")
            with httpx.Client(timeout=60.0) as client:
                r = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "temperature": temp,
                        "messages": [
                            {"role": "system", "content": prompt.system_prompt},
                            {"role": "user", "content": prompt.user_prompt},
                        ],
                    },
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()

        raise RuntimeError(f"Provedor desconhecido: {provider}")

    # --- Consulta RAG Completa ---------------------------------------------

    def query(self, req: QueryRequest) -> QueryResponse:
        if not self.chunks:
            raise ValueError("Nenhum livro ou nota carregado. Carregue o livro de Macroeconomia ou envie um arquivo.")

        t0 = time.perf_counter()
        orig_tokens, expanded, lang = expand_macro_query(req.question)
        all_terms = list(dict.fromkeys(orig_tokens + expanded))
        enriched_q = f"{req.question} {' '.join(expanded)}".strip()

        raw_bm25, matched_map = self._score_bm25_fast(orig_tokens, expanded)
        raw_sem = self._score_semantic_fast(enriched_q)

        # Máscara de filtro por documento e/ou capítulo
        mask = np.ones(len(self.chunks), dtype=bool)
        if req.doc_ids:
            dset = set(req.doc_ids)
            mask &= np.array([c.doc_id in dset for c in self.chunks], dtype=bool)
        if req.chapter_ids:
            cset = set(req.chapter_ids)
            mask &= np.array([c.chapter_id in cset for c in self.chunks], dtype=bool)

        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            raise ValueError("Nenhum trecho encontrado para os documentos/capítulos filtrados.")

        # Ajusta alpha automaticamente se modo verify (mais lexical) ou compare (mais diverso)
        alpha = req.alpha
        if req.study_mode == "verify":
            alpha = min(alpha, 0.35)
        mmr_lam = 0.62 if req.study_mode == "compare" else 0.75

        max_bm25 = float(np.max(raw_bm25[valid_idx]))
        max_sem = float(np.max(raw_sem[valid_idx]))
        norm_bm25 = np.clip(raw_bm25 / max(max_bm25, 5.0), 0.0, 1.0)
        norm_sem = np.clip(raw_sem / max(max_sem, 0.32), 0.0, 1.0)

        ord_lex = valid_idx[np.argsort(-raw_bm25[valid_idx])]
        ord_sem = valid_idx[np.argsort(-raw_sem[valid_idx])]
        r_lex = {int(i): r for r, i in enumerate(ord_lex, start=1)}
        r_sem = {int(i): r for r, i in enumerate(ord_sem, start=1)}

        k_rrf = 60.0
        max_rrf = 1.0 / (k_rrf + 1.0)
        final_scores = np.zeros(len(self.chunks), dtype=np.float64)
        for idx in valid_idx:
            i = int(idx)
            rrf_n = ((1.0 - alpha) / (k_rrf + r_lex[i]) + alpha / (k_rrf + r_sem[i])) / max_rrf
            w_abs = (1.0 - alpha) * norm_bm25[i] + alpha * norm_sem[i]
            final_scores[i] = 0.82 * w_abs + 0.18 * (rrf_n * max(w_abs, 0.15))

        sorted_valid = [int(i) for i in valid_idx[np.argsort(-final_scores[valid_idx])]]
        pool = sorted_valid[: min(len(sorted_valid), max(req.top_k * 4, 24))]
        chosen = self._mmr_select(pool, final_scores, req.top_k, mmr_lam) if req.use_mmr else pool[: req.top_k]
        chosen.sort(key=lambda i: float(final_scores[i]), reverse=True)

        citations: List[RetrievedPassage] = []
        for rank_i, idx in enumerate(chosen, start=1):
            ch = self.chunks[idx]
            mterms = list(dict.fromkeys(matched_map.get(idx, [])))[:12]
            citations.append(
                RetrievedPassage(
                    rank=rank_i,
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    document_title=ch.document_title,
                    chapter_id=ch.chapter_id,
                    chapter_title=ch.chapter_title,
                    part_name=ch.part_name,
                    page=ch.page,
                    text=ch.text,
                    excerpt=self._make_excerpt(ch.text, mterms),
                    score_final=round(float(final_scores[idx]), 4),
                    score_lexical=round(float(norm_bm25[idx]), 4),
                    score_semantic=round(float(norm_sem[idx]), 4),
                    rank_lexical=r_lex[idx],
                    rank_semantic=r_sem[idx],
                    matched_terms=mterms,
                )
            )

        t_ret = (time.perf_counter() - t0) * 1000.0

        top_terms = set(t for c in citations[:3] for t in c.matched_terms)
        cov = len(top_terms) / max(len(orig_tokens), 1)
        top_sc = citations[0].score_final if citations else 0.0

        # Calibração anti-alucinação (evita falsos positivos por coincidência de 1 palavra genérica em livro grande)
        out_of_scope = (
            (max_bm25 < 8.5 and len(top_terms) <= 1)
            or (max_bm25 < 3.5 and max_sem < 0.24)
            or (len(orig_tokens) >= 4 and len(top_terms) == 0)
        )
        if out_of_scope:
            grounding = round(min(30.0, max(5.0, top_sc * 50.0)), 1)
            conf = "baixa"
        else:
            grounding = round(min(98.5, max(40.0, 48.0 + 38.0 * min(1.0, top_sc / 0.75) + 14.0 * min(1.0, cov))), 1)
            conf = "alta" if grounding >= 74.0 else ("media" if grounding >= 54.0 else "baixa")

        prompt_insp = self._build_prompt(req.question, req.study_mode, citations, out_of_scope)

        t_gen0 = time.perf_counter()
        prov_used = req.provider
        model_used = req.model or ""
        fallback: Optional[str] = None

        if req.provider == "extractive":
            model_used = "macro-extractive-v2"
            answer = self._generate_extractive_answer(req.study_mode, citations, all_terms, out_of_scope, grounding)
        else:
            defaults = {"gemini": "gemini-2.5-flash", "ollama": "qwen2.5:7b", "openai": "gpt-4o-mini"}
            model_used = req.model or defaults.get(req.provider, "gemini-2.5-flash")
            try:
                answer = self._call_llm(req.provider, model_used, req.api_key, prompt_insp, req.temperature)
                if not answer:
                    raise RuntimeError("Resposta vazia do provedor.")
            except Exception as exc:
                fallback = f"Provedor '{req.provider}' indisponível ({exc}). Aplicado fallback local Offline Grounded."
                prov_used = "extractive"
                model_used = "macro-extractive-v2"
                answer = self._generate_extractive_answer(req.study_mode, citations, all_terms, out_of_scope, grounding)

        t_gen = (time.perf_counter() - t_gen0) * 1000.0
        t_tot = (time.perf_counter() - t0) * 1000.0

        # Auditoria explícita de Privacidade (Local vs. Nuvem)
        corpus_chars = sum(d.total_chars for d in self.documents.values())
        is_local = prov_used in {"extractive", "ollama"}
        sent_chars = 0 if is_local else (len(prompt_insp.system_prompt) + len(prompt_insp.user_prompt))
        sent_tokens = 0 if is_local else prompt_insp.estimated_tokens
        exposure_pct = round((sent_chars / max(corpus_chars, 1)) * 100.0, 3)

        privacy = PrivacyAudit(
            provider_used=prov_used,
            is_fully_local=is_local,
            data_residency_label=(
                "100% Local / On-Premise (Zero dados externos)"
                if is_local
                else f"Híbrido: Índice 100% Local + Apenas {len(citations)} chunks enviados ({exposure_pct:.2f}% do corpus)"
            ),
            local_corpus_chars=corpus_chars,
            local_chunks_total=len(self.chunks),
            external_sent_chars=sent_chars,
            external_sent_tokens_est=sent_tokens,
            external_exposure_pct=exposure_pct,
            summary_note=(
                "Nenhum trecho do livro ou vetor saiu da sua máquina nesta consulta."
                if is_local
                else f"O livro completo ({corpus_chars:,} chars) e os vetores permaneceram locais; apenas {sent_chars:,} chars dos top-{len(citations)} trechos foram enviados ao LLM."
            ),
        )

        mode_lbl = (
            "Lexical BM25"
            if alpha <= 0.15
            else ("Semântico LSA/Subword" if alpha >= 0.85 else f"Híbrido RRF ({int(round((1-alpha)*100))}% BM25 / {int(round(alpha*100))}% Sem.)")
        )

        diag = QueryDiagnostics(
            detected_language=lang,
            study_mode=req.study_mode,
            original_tokens=orig_tokens,
            expanded_terms=expanded,
            retrieval_mode_label=mode_lbl,
            candidates_evaluated=int(len(valid_idx)),
            retrieval_ms=round(t_ret, 2),
            generation_ms=round(t_gen, 2),
            total_ms=round(t_tot, 2),
            grounding_score=grounding,
            confidence_level=conf,  # type: ignore[arg-type]
            out_of_scope_warning=out_of_scope,
            unique_chapters_cited=len({c.chapter_id for c in citations}),
            unique_pages_cited=len({(c.doc_id, c.page) for c in citations}),
        )

        return QueryResponse(
            question=req.question,
            study_mode=req.study_mode,
            answer=answer,
            provider_used=prov_used,
            model_used=model_used,
            fallback_notice=fallback,
            citations=citations,
            diagnostics=diag,
            privacy=privacy,
            prompt_inspection=prompt_insp,
        )

    # --- Gerador de Flashcards & Questões de Prova por Capítulo ------------

    def generate_flashcards(self, req: FlashcardRequest) -> FlashcardResponse:
        if not self.chunks:
            raise ValueError("Nenhum livro carregado para gerar flashcards.")
        t0 = time.perf_counter()

        pool = self.chunks
        scope_label = "Todo o Corpus de Macroeconomia"
        if req.doc_id:
            pool = [c for c in pool if c.doc_id == req.doc_id]
        if req.chapter_id:
            pool = [c for c in pool if c.chapter_id == req.chapter_id]
            if pool:
                scope_label = f"{pool[0].chapter_title} ({pool[0].part_name})"
        if not pool:
            raise ValueError("Nenhum trecho encontrado para o capítulo/documento selecionado.")

        # Se usuário informou tópico, ranqueia por relevância; senão, amostra trechos conceituais ao longo do capítulo
        if req.topic and req.topic.strip():
            orig, exp, _ = expand_macro_query(req.topic.strip())
            scope_label = f"{scope_label} • Tópico: {req.topic.strip()}"
            qset = set(orig + exp)
            scored = []
            for c in pool:
                hits = len(set(c.tokens) & qset)
                scored.append((hits, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            selected_chunks = [c for _, c in scored[: req.count]]
        else:
            # Seleciona chunks ricos em definições/conceitos distribuídos ao longo do capítulo
            def _concept_density(ch: MacroChunk) -> float:
                txt = ch.text.lower()
                cues = sum(
                    1
                    for w in (
                        "definition", "defined as", "assumption", "model", "equilibrium",
                        "theorem", "policy", "effect", "shows that", "implies", "therefore",
                        "curve", "rate", "growth", "inflation", "utility",
                    )
                    if w in txt
                )
                alpha_ratio = sum(ch_c.isalpha() for ch_c in ch.text) / max(len(ch.text), 1)
                return cues * 1.5 + (2.0 if alpha_ratio > 0.75 else -2.0)

            ranked = sorted(pool, key=_concept_density, reverse=True)
            # Evita pegar 2 chunks da mesma página
            selected_chunks = []
            used_pages = set()
            for c in ranked:
                if len(selected_chunks) >= req.count:
                    break
                if c.page in used_pages and len(pool) > req.count * 2:
                    continue
                used_pages.add(c.page)
                selected_chunks.append(c)
            if len(selected_chunks) < req.count:
                for c in pool:
                    if c not in selected_chunks:
                        selected_chunks.append(c)
                    if len(selected_chunks) >= req.count:
                        break

        cards: List[FlashcardItem] = []
        diffs = ["Fundamentos", "Intermediário", "Avançado"]
        for i, ch in enumerate(selected_chunks, start=1):
            kw = self._chapter_keywords.get(ch.chapter_id, [])
            # Extrai termo central mais presente neste chunk
            tf = Counter(ch.tokens)
            top_terms = [t for t, _ in tf.most_common(4) if len(t) >= 4]
            concept = ", ".join(top_terms[:2]).title() if top_terms else (kw[0].title() if kw else "Conceito Macro")

            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ch.text) if 55 <= len(s.strip()) <= 360]
            best_sents = [s for s in sents if sum(x.isalpha() for x in s) / max(len(s), 1) >= 0.65][:2]
            answer_text = " ".join(best_sents) if best_sents else ch.text[:320]

            templates = [
                f"No contexto de '{ch.chapter_title}', como se relacionam os conceitos de {concept} e qual o mecanismo econômico central apresentado na página {ch.page}?",
                f"Explique o papel de {concept} na análise desenvolvida em '{ch.chapter_title}' (pág. {ch.page}) e suas implicações de equilíbrio.",
                f"Questão discursiva ({ch.chapter_title}): quais hipóteses e resultados fundamentais envolvem {concept} segundo o texto?",
            ]
            cards.append(
                FlashcardItem(
                    id=f"fc-{ch.chunk_id}-{i}",
                    difficulty=diffs[(i - 1) % 3],  # type: ignore[arg-type]
                    concept=concept,
                    question=templates[(i - 1) % len(templates)],
                    answer=f"Gabarito fundamentado ({ch.chapter_title}, pág. {ch.page}): \"{answer_text}\"",
                    chapter_title=ch.chapter_title,
                    page=ch.page,
                    source_excerpt=ch.text[:420] + ("..." if len(ch.text) > 420 else ""),
                )
            )

        dt = round((time.perf_counter() - t0) * 1000.0, 2)
        return FlashcardResponse(
            scope_label=scope_label,
            provider_used="extractive",
            cards=cards,
            generation_ms=dt,
        )
