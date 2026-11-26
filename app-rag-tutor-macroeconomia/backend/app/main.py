"""API REST FastAPI do MacroRAG Tutor Studio (porta 8003)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .macro_engine import MacroRAGEngine
from .schemas import (
    ChunkConfig,
    FlashcardRequest,
    FlashcardResponse,
    HealthResponse,
    LibraryResponse,
    ProvidersResponse,
    QueryRequest,
    QueryResponse,
)

APP_NAME = "MacroRAG Tutor Studio API"
APP_VERSION = "2.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOKS_DIR = PROJECT_ROOT / "data" / "books"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"

engine = MacroRAGEngine(books_dir=BOOKS_DIR, cache_dir=CACHE_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Carrega automaticamente o livro-texto demo (usando cache comprimido .json.gz quando disponível!)
    if not engine.documents and BOOKS_DIR.exists():
        engine.load_demo_library()
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Backend Python (FastAPI) para RAG especializado em Livros-Texto e Notas de Macroeconomia "
        "(Heijdra — 816 páginas indexadas por capítulo com cache local, busca híbrida BM25 + LSA/Subword + RRF + MMR, "
        "modos Tutor/Comparador/Verificador, Gerador de Flashcards por Capítulo e Auditoria de Privacidade)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["Sistema"])
def health_check() -> HealthResponse:
    demo_books = len([p for p in BOOKS_DIR.glob("*") if p.suffix.lower() in {".pdf", ".md", ".txt"}]) if BOOKS_DIR.exists() else 0
    return HealthResponse(
        status="ok",
        app_name=APP_NAME,
        version=APP_VERSION,
        documents_loaded=len(engine.documents),
        chapters_indexed=sum(len(d.chapters) for d in engine.documents.values()),
        chunks_indexed=len(engine.chunks),
        ollama_available=engine.is_ollama_available(),
        demo_books_available=demo_books,
    )


@app.get("/api/providers", response_model=ProvidersResponse, tags=["Provedores LLM"])
def get_providers() -> ProvidersResponse:
    return engine.inspect_providers()


@app.get("/api/library", response_model=LibraryResponse, tags=["Biblioteca & Capítulos"])
def get_library() -> LibraryResponse:
    return engine.get_library()


@app.post("/api/library/load-demo", response_model=LibraryResponse, tags=["Biblioteca & Capítulos"])
def load_demo_library() -> LibraryResponse:
    loaded = engine.load_demo_library()
    if not loaded:
        raise HTTPException(status_code=404, detail=f"Nenhum livro demo encontrado em {BOOKS_DIR}.")
    return engine.get_library()


@app.post("/api/documents/upload", response_model=LibraryResponse, tags=["Biblioteca & Capítulos"])
async def upload_documents(files: List[UploadFile] = File(...)) -> LibraryResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Envie ao menos um arquivo (.pdf, .md ou .txt).")
    ingested = False
    errors: List[str] = []
    for f in files:
        fname = f.filename or "notas.txt"
        raw = await f.read()
        if not raw:
            errors.append(f"'{fname}': arquivo vazio.")
            continue
        if len(raw) > 50 * 1024 * 1024:
            errors.append(f"'{fname}': excede 50 MB.")
            continue
        try:
            engine.ingest_file(fname, raw, is_demo=False)
            ingested = True
        except Exception as exc:
            errors.append(f"'{fname}': {exc}")
    if not ingested:
        raise HTTPException(status_code=400, detail=" | ".join(errors) or "Falha ao processar arquivos.")
    engine.rebuild_index()
    return engine.get_library()


@app.delete("/api/documents/{doc_id}", response_model=LibraryResponse, tags=["Biblioteca & Capítulos"])
def delete_document(doc_id: str) -> LibraryResponse:
    if not engine.remove_document(doc_id):
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado.")
    return engine.get_library()


@app.post("/api/index/rebuild", response_model=LibraryResponse, tags=["Biblioteca & Capítulos"])
def rebuild_index(cfg: ChunkConfig) -> LibraryResponse:
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap deve ser menor que chunk_size.")
    engine.rebuild_index(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    return engine.get_library()


@app.post("/api/query", response_model=QueryResponse, tags=["Tutor MacroRAG"])
def run_query(req: QueryRequest) -> QueryResponse:
    try:
        return engine.query(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/flashcards", response_model=FlashcardResponse, tags=["Simulado & Flashcards"])
def create_flashcards(req: FlashcardRequest) -> FlashcardResponse:
    try:
        return engine.generate_flashcards(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
