export type ProviderId = "extractive" | "ollama" | "openai" | "gemini";

export interface DocumentSummary {
  doc_id: string;
  filename: string;
  title: string;
  authors: string;
  year: string | null;
  pages: number;
  total_chars: number;
  total_words: number;
  chunk_count: number;
  is_demo: boolean;
  preview: string;
  ingested_at: string;
}

export interface ChunkHistogramBin {
  range_label: string;
  min_chars: number;
  max_chars: number;
  count: number;
}

export interface IndexStats {
  document_count: number;
  total_pages: number;
  total_chunks: number;
  total_chars: number;
  avg_chunk_chars: number;
  median_chunk_chars: number;
  min_chunk_chars: number;
  max_chunk_chars: number;
  vocab_size: number;
  lsa_dimensions: number;
  chunk_size: number;
  chunk_overlap: number;
  overlap_ratio_pct: number;
  build_time_ms: number;
  histogram: ChunkHistogramBin[];
}

export interface SuggestedQuestion {
  id: string;
  category: string;
  question: string;
  hint: string;
  is_out_of_scope: boolean;
}

export interface LibraryResponse {
  documents: DocumentSummary[];
  stats: IndexStats;
  suggested_questions: SuggestedQuestion[];
}

export interface ChunkItem {
  chunk_id: string;
  doc_id: string;
  document_name: string;
  document_title: string;
  page: number;
  chunk_index: number;
  char_count: number;
  word_count: number;
  overlap_prefix_chars: number;
  text: string;
  preview: string;
}

export interface ChunksPageResponse {
  total: number;
  offset: number;
  limit: number;
  items: ChunkItem[];
}

export interface ProviderInfo {
  id: ProviderId;
  name: string;
  available: boolean;
  requires_key: boolean;
  models: string[];
  default_model: string;
  description: string;
  status_detail: string;
}

export interface ProvidersResponse {
  providers: ProviderInfo[];
  ollama_url: string;
}

export interface RetrievedChunk {
  rank: number;
  chunk_id: string;
  doc_id: string;
  document_name: string;
  document_title: string;
  page: number;
  chunk_index: number;
  text: string;
  highlighted_excerpt: string;
  score_final: number;
  score_lexical: number;
  score_semantic: number;
  score_rrf: number;
  rank_lexical: number;
  rank_semantic: number;
  matched_terms: string[];
}

export interface QueryDiagnostics {
  detected_language: string;
  original_tokens: string[];
  expanded_terms: string[];
  alpha: number;
  retrieval_mode_label: string;
  use_mmr: boolean;
  candidates_evaluated: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  grounding_score: number;
  confidence_level: "alta" | "media" | "baixa";
  out_of_scope_warning: boolean;
  prompt_tokens_estimate: number;
  unique_docs_cited: number;
  unique_pages_cited: number;
}

export interface PromptInspection {
  system_prompt: string;
  user_prompt: string;
  estimated_tokens: number;
  context_chars: number;
}

export interface QueryRequestPayload {
  question: string;
  top_k: number;
  alpha: number;
  use_mmr: boolean;
  mmr_lambda: number;
  doc_ids: string[] | null;
  provider: ProviderId;
  model: string | null;
  api_key: string | null;
  temperature: number;
  response_language: "pt-BR" | "en";
}

export interface QueryResponse {
  question: string;
  answer: string;
  provider_used: string;
  model_used: string;
  fallback_notice: string | null;
  citations: RetrievedChunk[];
  diagnostics: QueryDiagnostics;
  prompt_inspection: PromptInspection;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  documents_loaded: number;
  chunks_indexed: number;
  ollama_available: boolean;
  demo_papers_available: number;
}

export interface ChatTurn {
  id: string;
  timestamp: string;
  response: QueryResponse;
}
