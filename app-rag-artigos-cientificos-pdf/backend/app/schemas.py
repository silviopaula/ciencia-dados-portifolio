"""Modelos Pydantic v2 (contratos de entrada e saída) da API PaperRAG Studio."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    chunk_size: int = Field(default=1000, ge=350, le=2400, description="Tamanho alvo do chunk em caracteres.")
    chunk_overlap: int = Field(default=180, ge=0, le=600, description="Sobreposição entre chunks consecutivos em caracteres.")


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    title: str
    authors: str
    year: Optional[str] = None
    pages: int
    total_chars: int
    total_words: int
    chunk_count: int
    is_demo: bool = False
    preview: str
    ingested_at: str


class ChunkHistogramBin(BaseModel):
    range_label: str
    min_chars: int
    max_chars: int
    count: int


class IndexStats(BaseModel):
    document_count: int
    total_pages: int
    total_chunks: int
    total_chars: int
    avg_chunk_chars: float
    median_chunk_chars: float
    min_chunk_chars: int
    max_chunk_chars: int
    vocab_size: int
    lsa_dimensions: int
    chunk_size: int
    chunk_overlap: int
    overlap_ratio_pct: float
    build_time_ms: float
    histogram: List[ChunkHistogramBin]


class SuggestedQuestion(BaseModel):
    id: str
    category: str
    question: str
    hint: str
    is_out_of_scope: bool = False


class LibraryResponse(BaseModel):
    documents: List[DocumentSummary]
    stats: IndexStats
    suggested_questions: List[SuggestedQuestion]


class ChunkItem(BaseModel):
    chunk_id: str
    doc_id: str
    document_name: str
    document_title: str
    page: int
    chunk_index: int
    char_count: int
    word_count: int
    overlap_prefix_chars: int
    text: str
    preview: str


class ChunksPageResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: List[ChunkItem]


class ProviderInfo(BaseModel):
    id: Literal["extractive", "ollama", "openai", "gemini"]
    name: str
    available: bool
    requires_key: bool
    models: List[str]
    default_model: str
    description: str
    status_detail: str


class ProvidersResponse(BaseModel):
    providers: List[ProviderInfo]
    ollama_url: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=12)
    alpha: float = Field(default=0.55, ge=0.0, le=1.0, description="0.0 = 100% Lexical BM25; 1.0 = 100% Semântico Vetorial.")
    use_mmr: bool = Field(default=True, description="Ativa Maximal Marginal Relevance para diversificar chunks.")
    mmr_lambda: float = Field(default=0.72, ge=0.3, le=1.0)
    doc_ids: Optional[List[str]] = Field(default=None, description="Filtra busca a documentos específicos.")
    provider: Literal["extractive", "ollama", "openai", "gemini"] = "extractive"
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = Field(default=0.15, ge=0.0, le=1.0)
    response_language: Literal["pt-BR", "en"] = "pt-BR"


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    document_name: str
    document_title: str
    page: int
    chunk_index: int
    text: str
    highlighted_excerpt: str
    score_final: float
    score_lexical: float
    score_semantic: float
    score_rrf: float
    rank_lexical: int
    rank_semantic: int
    matched_terms: List[str]


class QueryDiagnostics(BaseModel):
    detected_language: str
    original_tokens: List[str]
    expanded_terms: List[str]
    alpha: float
    retrieval_mode_label: str
    use_mmr: bool
    candidates_evaluated: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    grounding_score: float
    confidence_level: Literal["alta", "media", "baixa"]
    out_of_scope_warning: bool
    prompt_tokens_estimate: int
    unique_docs_cited: int
    unique_pages_cited: int


class PromptInspection(BaseModel):
    system_prompt: str
    user_prompt: str
    estimated_tokens: int
    context_chars: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    provider_used: str
    model_used: str
    fallback_notice: Optional[str] = None
    citations: List[RetrievedChunk]
    diagnostics: QueryDiagnostics
    prompt_inspection: PromptInspection


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    documents_loaded: int
    chunks_indexed: int
    ollama_available: bool
    demo_papers_available: int
