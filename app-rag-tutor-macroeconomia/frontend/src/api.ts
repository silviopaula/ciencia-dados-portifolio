import type {
  FlashcardResponse,
  HealthResponse,
  LibraryResponse,
  ProviderId,
  ProvidersResponse,
  QueryRequestPayload,
  QueryResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = `Erro HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // mantém fallback
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function fetchLibrary(): Promise<LibraryResponse> {
  return request<LibraryResponse>("/api/library");
}

export function fetchProviders(): Promise<ProvidersResponse> {
  return request<ProvidersResponse>("/api/providers");
}

export function loadDemoLibrary(): Promise<LibraryResponse> {
  return request<LibraryResponse>("/api/library/load-demo", { method: "POST" });
}

export function uploadMaterials(files: File[]): Promise<LibraryResponse> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  return request<LibraryResponse>("/api/documents/upload", { method: "POST", body: fd });
}

export function deleteDocument(docId: string): Promise<LibraryResponse> {
  return request<LibraryResponse>(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
}

export function rebuildIndex(chunkSize: number, chunkOverlap: number): Promise<LibraryResponse> {
  return request<LibraryResponse>("/api/index/rebuild", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
  });
}

export function runMacroQuery(payload: QueryRequestPayload): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function generateFlashcards(payload: {
  chapter_id?: string | null;
  doc_id?: string | null;
  topic?: string | null;
  count: number;
  provider: ProviderId;
}): Promise<FlashcardResponse> {
  return request<FlashcardResponse>("/api/flashcards", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
