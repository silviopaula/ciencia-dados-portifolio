"""Contratos tipados Pydantic v2 da API MacroRAG Tutor Studio."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChapterSummary(BaseModel):
    chapter_id: str
    doc_id: str
    document_title: str
    part_name: str
    title: str
    start_page: int
    end_page: int
    page_count: int
    chunk_count: int
    top_keywords: List[str]


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    format: Literal["pdf", "md", "txt"]
    title: str
    author: str
    pages: int
    chapters_count: int
    total_chars: int
    total_words: int
    chunk_count: int
    is_demo: bool
    loaded_from_cache: bool
    ingested_at: str


class IndexStats(BaseModel):
    document_count: int
    total_pages: int
    total_chapters: int
    total_chunks: int
    total_chars: int
    vocab_size: int
    lsa_dimensions: int
    chunk_size: int
    chunk_overlap: int
    build_time_ms: float
    cache_enabled: bool


class SuggestedPrompt(BaseModel):
    id: str
    mode: Literal["tutor", "compare", "verify"]
    category: str
    question: str
    hint: str
    chapter_hint: Optional[str] = None
    is_out_of_scope: bool = False


class LibraryResponse(BaseModel):
    documents: List[DocumentSummary]
    chapters: List[ChapterSummary]
    stats: IndexStats
    suggested_prompts: List[SuggestedPrompt]


class ProviderInfo(BaseModel):
    id: Literal["extractive", "gemini", "ollama", "openai"]
    name: str
    available: bool
    requires_key: bool
    is_local: bool
    models: List[str]
    default_model: str
    description: str
    status_detail: str


class ProvidersResponse(BaseModel):
    providers: List[ProviderInfo]
    ollama_url: str


class ChunkConfig(BaseModel):
    chunk_size: int = Field(default=1050, ge=400, le=2400)
    chunk_overlap: int = Field(default=160, ge=0, le=500)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    study_mode: Literal["tutor", "compare", "verify"] = "tutor"
    top_k: int = Field(default=6, ge=2, le=12)
    alpha: float = Field(default=0.50, ge=0.0, le=1.0)
    use_mmr: bool = True
    doc_ids: Optional[List[str]] = None
    chapter_ids: Optional[List[str]] = None
    provider: Literal["extractive", "gemini", "ollama", "openai"] = "extractive"
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = Field(default=0.15, ge=0.0, le=1.0)


class RetrievedPassage(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    document_title: str
    chapter_id: str
    chapter_title: str
    part_name: str
    page: int
    text: str
    excerpt: str
    score_final: float
    score_lexical: float
    score_semantic: float
    rank_lexical: int
    rank_semantic: int
    matched_terms: List[str]


class PrivacyAudit(BaseModel):
    provider_used: str
    is_fully_local: bool
    data_residency_label: str
    local_corpus_chars: int
    local_chunks_total: int
    external_sent_chars: int
    external_sent_tokens_est: int
    external_exposure_pct: float
    summary_note: str


class QueryDiagnostics(BaseModel):
    detected_language: str
    study_mode: str
    original_tokens: List[str]
    expanded_terms: List[str]
    retrieval_mode_label: str
    candidates_evaluated: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    grounding_score: float
    confidence_level: Literal["alta", "media", "baixa"]
    out_of_scope_warning: bool
    unique_chapters_cited: int
    unique_pages_cited: int


class PromptInspection(BaseModel):
    system_prompt: str
    user_prompt: str
    estimated_tokens: int
    context_chars: int


class QueryResponse(BaseModel):
    question: str
    study_mode: str
    answer: str
    provider_used: str
    model_used: str
    fallback_notice: Optional[str] = None
    citations: List[RetrievedPassage]
    diagnostics: QueryDiagnostics
    privacy: PrivacyAudit
    prompt_inspection: PromptInspection


class FlashcardRequest(BaseModel):
    chapter_id: Optional[str] = None
    doc_id: Optional[str] = None
    topic: Optional[str] = None
    count: int = Field(default=5, ge=2, le=8)
    provider: Literal["extractive", "gemini", "ollama", "openai"] = "extractive"
    model: Optional[str] = None
    api_key: Optional[str] = None


class FlashcardItem(BaseModel):
    id: str
    difficulty: Literal["Fundamentos", "Intermediário", "Avançado"]
    concept: str
    question: str
    answer: str
    chapter_title: str
    page: int
    source_excerpt: str


class FlashcardResponse(BaseModel):
    scope_label: str
    provider_used: str
    cards: List[FlashcardItem]
    generation_ms: float


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    documents_loaded: int
    chapters_indexed: int
    chunks_indexed: int
    ollama_available: bool
    demo_books_available: int
