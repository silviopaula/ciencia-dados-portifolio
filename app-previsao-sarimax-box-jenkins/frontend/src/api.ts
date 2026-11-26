import type { DatasetSummary, DiagnosticsResponse, ForecastResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `Erro HTTP ${res.status}`;
    let parsedDetail = false;
    try {
      const err = await res.json();
      if (err?.detail) {
        msg = String(err.detail);
        parsedDetail = true;
      }
    } catch {
      // ignora parse
    }
    if (!parsedDetail && res.status >= 500) {
      msg =
        "Não foi possível comunicar com o backend Python (porta 8000). Aguarde alguns segundos para a API iniciar ou reinicie pelo iniciar_app.bat.";
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function loadDemoDataset(): Promise<DatasetSummary> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/dataset/demo`, { method: "POST" });
  } catch {
    throw new Error(
      "Backend Python indisponível na porta 8000. Verifique se o servidor FastAPI está em execução."
    );
  }
  return handleResponse<DatasetSummary>(res);
}

export async function uploadDataset(file: File): Promise<DatasetSummary> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/dataset/upload`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw new Error(
      "Backend Python indisponível na porta 8000. Verifique se o servidor FastAPI está em execução."
    );
  }
  return handleResponse<DatasetSummary>(res);
}

export async function runDiagnostics(payload: {
  dataset_id: string;
  date_col: string;
  target_col: string;
  use_log: boolean;
  diff_order: number;
  seasonal_diff: number;
  seasonal_period: number;
  nlags?: number;
}): Promise<DiagnosticsResponse> {
  const res = await fetch(`${API_BASE}/api/diagnostics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<DiagnosticsResponse>(res);
}

export async function runForecast(payload: {
  dataset_id: string;
  date_col: string;
  target_col: string;
  exog_cols: string[];
  mode: "auto" | "manual";
  p: number;
  d: number;
  q: number;
  P: number;
  D: number;
  Q: number;
  s: number;
  use_log: boolean;
  test_periods: number;
  forecast_horizon: number;
  confidence_level: number;
}): Promise<ForecastResponse> {
  const res = await fetch(`${API_BASE}/api/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<ForecastResponse>(res);
}

export function getExportExcelUrl(datasetId: string): string {
  return `${API_BASE}/api/export/${encodeURIComponent(datasetId)}`;
}
