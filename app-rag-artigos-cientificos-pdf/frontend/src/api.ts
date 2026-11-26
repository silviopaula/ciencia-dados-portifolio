import type {
  ChunksPageResponse,
  HealthResponse,
  LibraryResponse,
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
      const errBody = await res.json();
      if (errBody?.detail) {
        detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      }
    } catch {
      // mantém mensagem padrão
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

export function loadDemoDocuments(): Promise<LibraryResponse> {
  return request<LibraryResponse>("/api/documents/load-demo", { method: "POST" });
}

export function uploadDocuments(files: File[]): Promise<LibraryResponse> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  return request<LibraryResponse>("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export function deleteDocument(docId: string): Promise<LibraryResponse> {
  return request<LibraryResponse>(`/api/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
}

export function rebuildIndex(chunkSize: number, chunkOverlap: number): Promise<LibraryResponse> {
  return request<LibraryResponse>("/api/index/rebuild", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
  });
}

export function fetchChunks(params: {
  doc_id?: string;
  search?: string;
  offset?: number;
  limit?: number;
}): Promise<ChunksPageResponse> {
  const sp = new URLSearchParams();
  if (params.doc_id) sp.set("doc_id", params.doc_id);
  if (params.search) sp.set("search", params.search);
  if (params.offset !== undefined) sp.set("offset", String(params.offset));
  if (params.limit !== undefined) sp.set("limit", String(params.limit));
  return request<ChunksPageResponse>(`/api/chunks?${sp.toString()}`);
}

export function runRagQuery(payload: QueryRequestPayload): Promise<QueryResponse> {
  return request<QueryResponse>("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
