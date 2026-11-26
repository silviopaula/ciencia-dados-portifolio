export type ProviderId = "extractive" | "gemini" | "ollama" | "openai";
export type StudyMode = "tutor" | "compare" | "verify";

export interface ChapterSummary {
  chapter_id: string;
  doc_id: string;
  document_title: string;
  part_name: string;
  title: string;
  start_page: number;
  end_page: number;
  page_count: number;
  chunk_count: number;
  top_keywords: string[];
}

export interface DocumentSummary {
  doc_id: string;
  filename: string;
  format: "pdf" | "md" | "txt";
  title: string;
  author: string;
  pages: number;
  chapters_count: number;
  total_chars: number;
  total_words: number;
  chunk_count: number;
  is_demo: boolean;
  loaded_from_cache: boolean;
  ingested_at: string;
}

export interface IndexStats {
  document_count: number;
  total_pages: number;
  total_chapters: number;
  total_chunks: number;
  total_chars: number;
  vocab_size: number;
  lsa_dimensions: number;
  chunk_size: number;
  chunk_overlap: number;
  build_time_ms: number;
  cache_enabled: boolean;
}

export interface SuggestedPrompt {
  id: string;
  mode: StudyMode;
  category: string;
  question: string;
  hint: string;
  chapter_hint: string | null;
  is_out_of_scope: boolean;
}

export interface LibraryResponse {
  documents: DocumentSummary[];
  chapters: ChapterSummary[];
  stats: IndexStats;
  suggested_prompts: SuggestedPrompt[];
}

export interface ProviderInfo {
  id: ProviderId;
  name: string;
  available: boolean;
  requires_key: boolean;
  is_local: boolean;
  models: string[];
  default_model: string;
  description: string;
  status_detail: string;
}

export interface ProvidersResponse {
  providers: ProviderInfo[];
  ollama_url: string;
}

export interface QueryRequestPayload {
  question: string;
  study_mode: StudyMode;
  top_k: number;
  alpha: number;
  use_mmr: boolean;
  doc_ids: string[] | null;
  chapter_ids: string[] | null;
  provider: ProviderId;
  model: string | null;
  api_key: string | null;
  temperature: number;
}

export interface RetrievedPassage {
  rank: number;
  chunk_id: string;
  doc_id: string;
  document_title: string;
  chapter_id: string;
  chapter_title: string;
  part_name: string;
  page: number;
  text: string;
  excerpt: string;
  score_final: number;
  score_lexical: number;
  score_semantic: number;
  rank_lexical: intLike;
  rank_semantic: intLike;
  matched_terms: string[];
}

type intLike = number;

export interface PrivacyAudit {
  provider_used: string;
  is_fully_local: boolean;
  data_residency_label: string;
  local_corpus_chars: number;
  local_chunks_total: number;
  external_sent_chars: number;
  external_sent_tokens_est: number;
  external_exposure_pct: number;
  summary_note: string;
}

export interface QueryDiagnostics {
  detected_language: string;
  study_mode: string;
  original_tokens: string[];
  expanded_terms: string[];
  retrieval_mode_label: string;
  candidates_evaluated: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  grounding_score: number;
  confidence_level: "alta" | "media" | "baixa";
  out_of_scope_warning: boolean;
  unique_chapters_cited: number;
  unique_pages_cited: number;
}

export interface PromptInspection {
  system_prompt: string;
  user_prompt: string;
  estimated_tokens: number;
  context_chars: number;
}

export interface QueryResponse {
  question: string;
  study_mode: string;
  answer: string;
  provider_used: string;
  model_used: string;
  fallback_notice: string | null;
  citations: RetrievedPassage[];
  diagnostics: QueryDiagnostics;
  privacy: PrivacyAudit;
  prompt_inspection: PromptInspection;
}

export interface FlashcardItem {
  id: string;
  difficulty: "Fundamentos" | "Intermediário" | "Avançado";
  concept: string;
  question: string;
  answer: string;
  chapter_title: string;
  page: number;
  source_excerpt: string;
}

export interface FlashcardResponse {
  scope_label: string;
  provider_used: string;
  cards: FlashcardItem[];
  generation_ms: number;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  documents_loaded: number;
  chapters_indexed: number;
  chunks_indexed: number;
  ollama_available: boolean;
  demo_books_available: number;
}

export interface ChatTurn {
  id: string;
  timestamp: string;
  response: QueryResponse;
}
