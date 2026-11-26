import type { AnalysisConfig, AnalysisPayload, DemoDatasetMeta, UploadInspectResponse } from "./types";

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body?.detail) return String(body.detail);
  } catch {
    // ignore
  }
  return `Erro HTTP ${res.status}`;
}

export async function fetchDemos(): Promise<DemoDatasetMeta[]> {
  const res = await fetch("/api/demos");
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function runDemoAnalysis(cfg: AnalysisConfig): Promise<AnalysisPayload> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: cfg.datasetId,
      transform: cfg.transform,
      alpha: cfg.alpha,
      regression: cfg.regression,
      max_lags: cfg.maxLags,
      seasonal_period: cfg.seasonalPeriod,
      auto_arima: cfg.autoArima,
      order_p: cfg.orderP,
      order_d: cfg.orderD,
      order_q: cfg.orderQ,
      seasonal_P: cfg.seasonalP,
      seasonal_D: cfg.seasonalD,
      seasonal_Q: cfg.seasonalQ,
      forecast_horizon: cfg.forecastHorizon,
    }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function inspectUploadedFile(file: File, sheetName?: string): Promise<UploadInspectResponse> {
  const fd = new FormData();
  fd.append("file", file);
  if (sheetName) fd.append("sheet_name", sheetName);
  const res = await fetch("/api/upload/inspect", { method: "POST", body: fd });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

function buildUploadFormData(file: File, cfg: AnalysisConfig): FormData {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("date_col", cfg.dateCol);
  fd.append("value_col", cfg.valueCol);
  if (cfg.sheetName) fd.append("sheet_name", cfg.sheetName);
  fd.append("date_format", cfg.dateFormat);
  fd.append("aggregation", cfg.aggregation);
  fd.append("transform", cfg.transform);
  fd.append("alpha", String(cfg.alpha));
  fd.append("regression", cfg.regression);
  fd.append("max_lags", String(cfg.maxLags));
  fd.append("seasonal_period", String(cfg.seasonalPeriod));
  fd.append("auto_arima", String(cfg.autoArima));
  fd.append("order_p", String(cfg.orderP));
  fd.append("order_d", String(cfg.orderD));
  fd.append("order_q", String(cfg.orderQ));
  fd.append("seasonal_P", String(cfg.seasonalP));
  fd.append("seasonal_D", String(cfg.seasonalD));
  fd.append("seasonal_Q", String(cfg.seasonalQ));
  fd.append("forecast_horizon", String(cfg.forecastHorizon));
  return fd;
}

export async function runUploadAnalysis(file: File, cfg: AnalysisConfig): Promise<AnalysisPayload> {
  const fd = buildUploadFormData(file, cfg);
  const res = await fetch("/api/upload/analyze", { method: "POST", body: fd });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function downloadExcelReport(cfg: AnalysisConfig, uploadedFile: File | null): Promise<void> {
  let res: Response;
  if (cfg.sourceMode === "upload" && uploadedFile) {
    res = await fetch("/api/upload/export/excel", {
      method: "POST",
      body: buildUploadFormData(uploadedFile, cfg),
    });
  } else {
    res = await fetch("/api/export/excel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: cfg.datasetId,
        transform: cfg.transform,
        alpha: cfg.alpha,
        regression: cfg.regression,
        max_lags: cfg.maxLags,
        seasonal_period: cfg.seasonalPeriod,
        auto_arima: cfg.autoArima,
        order_p: cfg.orderP,
        order_d: cfg.orderD,
        order_q: cfg.orderQ,
        seasonal_P: cfg.seasonalP,
        seasonal_D: cfg.seasonalD,
        seasonal_Q: cfg.seasonalQ,
        forecast_horizon: cfg.forecastHorizon,
      }),
    });
  }
  if (!res.ok) throw new Error(await parseError(res));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download =
    cfg.sourceMode === "upload"
      ? "diagnostico_ts_upload.xlsx"
      : `diagnostico_ts_${cfg.datasetId}_${cfg.transform}.xlsx`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
