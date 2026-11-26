import type {
  ConjunturaResponse,
  ExecuteStudyResponse,
  HealthResponse,
  StudiesListResponse,
} from "./types";

const BASE = "/api";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignora parse de erro não-JSON
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>(`${BASE}/health`);
}

export function fetchConjuntura(): Promise<ConjunturaResponse> {
  return requestJson<ConjunturaResponse>(`${BASE}/conjuntura`);
}

export function syncConjunturaLive(): Promise<ConjunturaResponse> {
  return requestJson<ConjunturaResponse>(`${BASE}/conjuntura/sync`, {
    method: "POST",
  });
}

export function fetchStudies(): Promise<StudiesListResponse> {
  return requestJson<StudiesListResponse>(`${BASE}/studies`);
}

export function runStudy(
  code: string,
  params: Record<string, unknown> = {},
): Promise<ExecuteStudyResponse> {
  return requestJson<ExecuteStudyResponse>(`${BASE}/studies/${encodeURIComponent(code)}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params }),
  });
}

export function getExportCsvUrl(code: string): string {
  return `${BASE}/studies/${encodeURIComponent(code)}/export-csv`;
}
