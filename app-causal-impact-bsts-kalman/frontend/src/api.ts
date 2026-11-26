import type {
  AnalyzeRequestPayload,
  AnalyzeResponse,
  DatasetSummary,
  DemoDatasetMeta,
  HealthResponse,
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

export function fetchDemos(): Promise<DemoDatasetMeta[]> {
  return request<DemoDatasetMeta[]>("/api/demos");
}

export function fetchCurrentDataset(): Promise<DatasetSummary> {
  return request<DatasetSummary>("/api/dataset/current");
}

export function loadDemoDataset(demoId: string): Promise<DatasetSummary> {
  return request<DatasetSummary>(`/api/dataset/demo/${encodeURIComponent(demoId)}`, {
    method: "POST",
  });
}

export function uploadDatasetFile(file: File): Promise<DatasetSummary> {
  const fd = new FormData();
  fd.append("file", file);
  return request<DatasetSummary>("/api/dataset/upload", {
    method: "POST",
    body: fd,
  });
}

export function runCausalAnalysis(payload: AnalyzeRequestPayload): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getExportExcelUrl(): string {
  return `${BASE}/api/export/excel`;
}

export function getExportCsvUrl(freq: string): string {
  return `${BASE}/api/export/csv?freq=${encodeURIComponent(freq)}`;
}

export function getExportReportUrl(): string {
  return `${BASE}/api/export/report`;
}

export function getExportTemplateUrl(): string {
  return `${BASE}/api/export/template`;
}
