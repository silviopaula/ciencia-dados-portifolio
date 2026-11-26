"""API REST FastAPI do PaperRAG Studio — RAG Científico Híbrido e Didático."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .rag_engine import PaperRAGEngine
from .schemas import (
    ChunkConfig,
    ChunksPageResponse,
    HealthResponse,
    LibraryResponse,
    ProvidersResponse,
    QueryRequest,
    QueryResponse,
)

APP_NAME = "PaperRAG Studio API"
APP_VERSION = "2.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

engine = PaperRAGEngine(demo_papers_dir=DEMO_PAPERS_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Carrega automaticamente os artigos demo (Callaway & Sant'Anna) no startup
    if not engine.documents and DEMO_PAPERS_DIR.exists():
        engine.load_demo_papers()
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Backend Python (FastAPI) para Retrieval-Augmented Generation (RAG) sobre artigos científicos em PDF, "
        "com busca híbrida (Okapi BM25 + Vetores Densos LSA/Subword + RRF + MMR), ponte bilíngue PT↔EN, "
        "inspeção transparente de chunks/prompt e geração multi-provedor (Offline Extrativo, Ollama, OpenAI, Gemini)."
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
    demo_count = len(list(DEMO_PAPERS_DIR.glob("*.pdf"))) if DEMO_PAPERS_DIR.exists() else 0
    return HealthResponse(
        status="ok",
        app_name=APP_NAME,
        version=APP_VERSION,
        documents_loaded=len(engine.documents),
        chunks_indexed=len(engine.chunks),
        ollama_available=engine.is_ollama_available(),
        demo_papers_available=demo_count,
    )


@app.get("/api/providers", response_model=ProvidersResponse, tags=["Provedores LLM"])
def get_providers() -> ProvidersResponse:
    return engine.inspect_providers()


@app.get("/api/library", response_model=LibraryResponse, tags=["Biblioteca & Índice"])
def get_library() -> LibraryResponse:
    return engine.get_library_state()


@app.post("/api/documents/load-demo", response_model=LibraryResponse, tags=["Biblioteca & Índice"])
def load_demo_documents() -> LibraryResponse:
    loaded = engine.load_demo_papers()
    if not loaded:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhum PDF de demonstração encontrado em {DEMO_PAPERS_DIR}.",
        )
    return engine.get_library_state()


@app.post("/api/documents/upload", response_model=LibraryResponse, tags=["Biblioteca & Índice"])
async def upload_documents(files: List[UploadFile] = File(...)) -> LibraryResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Envie ao menos um arquivo PDF.")

    ingested_any = False
    errors: List[str] = []

    for f in files:
        fname = f.filename or "artigo.pdf"
        if not fname.lower().endswith(".pdf"):
            errors.append(f"'{fname}': apenas arquivos .pdf são aceitos.")
            continue
        raw_bytes = await f.read()
        if not raw_bytes:
            errors.append(f"'{fname}': arquivo vazio.")
            continue
        if len(raw_bytes) > 40 * 1024 * 1024:
            errors.append(f"'{fname}': excede o limite de 40 MB.")
            continue
        try:
            engine.ingest_pdf_bytes(filename=fname, content=raw_bytes, is_demo=False)
            ingested_any = True
        except Exception as exc:
            errors.append(f"'{fname}': {exc}")

    if not ingested_any:
        raise HTTPException(status_code=400, detail=" | ".join(errors) or "Falha ao processar PDFs.")

    engine.rebuild_index()
    return engine.get_library_state()


@app.delete("/api/documents/{doc_id}", response_model=LibraryResponse, tags=["Biblioteca & Índice"])
def delete_document(doc_id: str) -> LibraryResponse:
    removed = engine.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado.")
    return engine.get_library_state()


@app.post("/api/index/rebuild", response_model=LibraryResponse, tags=["Biblioteca & Índice"])
def rebuild_index(config: ChunkConfig) -> LibraryResponse:
    if config.chunk_overlap >= config.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="O chunk_overlap deve ser estritamente menor que o chunk_size.",
        )
    engine.rebuild_index(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    return engine.get_library_state()


@app.get("/api/chunks", response_model=ChunksPageResponse, tags=["Laboratório de Chunking"])
def list_chunks(
    doc_id: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=100),
) -> ChunksPageResponse:
    return engine.list_chunks(doc_id=doc_id, search=search, offset=offset, limit=limit)


@app.post("/api/query", response_model=QueryResponse, tags=["Consulta RAG"])
def query_rag(payload: QueryRequest) -> QueryResponse:
    try:
        return engine.query(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
