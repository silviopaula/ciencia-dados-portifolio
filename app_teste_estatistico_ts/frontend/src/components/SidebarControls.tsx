import React from "react";
import {
  Database,
  Download,
  FileSpreadsheet,
  Play,
  Sliders,
  Sparkles,
  Upload,
} from "lucide-react";
import type {
  AggregationType,
  AnalysisConfig,
  DemoDatasetMeta,
  RegressionType,
  TransformType,
  UploadInspectResponse,
} from "../types";

interface SidebarControlsProps {
  demos: DemoDatasetMeta[];
  config: AnalysisConfig;
  setConfig: React.Dispatch<React.SetStateAction<AnalysisConfig>>;
  uploadedFile: File | null;
  uploadMeta: UploadInspectResponse | null;
  onFileSelected: (file: File) => void;
  onSheetChange: (sheet: string) => void;
  onRun: () => void;
  onExportExcel: () => void;
  loading: boolean;
  exporting: boolean;
}

const TRANSFORMS: { value: TransformType; label: string }[] = [
  { value: "level", label: "Nível Original (yₜ)" },
  { value: "log", label: "Logaritmo Natural (ln yₜ)" },
  { value: "boxcox", label: "Box-Cox Ótimo (λ auto)" },
  { value: "diff1", label: "1ª Diferença (Δyₜ)" },
  { value: "diff_seasonal", label: "Diferença Sazonal (Δₛyₜ)" },
  { value: "diff1_seasonal", label: "Δ + Δₛ (Dupla)" },
  { value: "log_diff", label: "Log-Retorno (Δ ln yₜ)" },
];

export const SidebarControls: React.FC<SidebarControlsProps> = ({
  demos,
  config,
  setConfig,
  uploadedFile,
  uploadMeta,
  onFileSelected,
  onSheetChange,
  onRun,
  onExportExcel,
  loading,
  exporting,
}) => {
  const activeDemo = demos.find((d) => d.id === config.datasetId);

  return (
    <aside className="w-full lg:w-80 xl:w-[340px] shrink-0 bg-white border-r border-slate-200 flex flex-col">
      <div className="p-4 border-b border-slate-200 bg-slate-900 text-white">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-sky-300">
          <Sliders className="w-3.5 h-3.5" />
          Painel Econométrico
        </div>
        <p className="text-xs text-slate-300 mt-1">
          Configure a série, transformação e bateria de testes estatísticos.
        </p>
      </div>

      <div className="p-4 space-y-5 overflow-y-auto flex-1 text-sm">
        {/* Fonte de Dados */}
        <section className="space-y-2.5">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-indigo-600" />
            1. Fonte de Dados
          </label>
          <div className="grid grid-cols-2 gap-1.5 bg-slate-100 p-1 rounded-lg">
            <button
              type="button"
              onClick={() => setConfig((c) => ({ ...c, sourceMode: "demo" }))}
              className={`py-1.5 px-2 rounded-md text-xs font-semibold transition ${
                config.sourceMode === "demo"
                  ? "bg-white text-indigo-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              5 Cenários Demo
            </button>
            <button
              type="button"
              onClick={() => setConfig((c) => ({ ...c, sourceMode: "upload" }))}
              className={`py-1.5 px-2 rounded-md text-xs font-semibold transition ${
                config.sourceMode === "upload"
                  ? "bg-white text-indigo-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Upload CSV / XLSX
            </button>
          </div>

          {config.sourceMode === "demo" ? (
            <div className="space-y-2">
              <select
                value={config.datasetId}
                onChange={(e) => {
                  const next = demos.find((d) => d.id === e.target.value);
                  setConfig((c) => ({
                    ...c,
                    datasetId: e.target.value,
                    seasonalPeriod: next?.seasonal_period ?? 12,
                    transform: "level",
                  }));
                }}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              >
                {demos.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.title}
                  </option>
                ))}
              </select>

              {activeDemo && (
                <div className="rounded-lg border border-indigo-100 bg-indigo-50/70 p-2.5 text-xs space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-600 text-white">
                      {activeDemo.regime_badge}
                    </span>
                    <span className="text-[11px] font-medium text-slate-500">
                      N = {activeDemo.n_obs}
                    </span>
                  </div>
                  <p className="text-slate-700 leading-relaxed">{activeDemo.subtitle}</p>
                  <p className="text-[11px] text-indigo-900 bg-white/80 rounded p-1.5 border border-indigo-100">
                    <strong>Diagnóstico esperado:</strong> {activeDemo.expected_diagnosis}
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-2.5">
              <label className="flex flex-col items-center justify-center border-2 border-dashed border-indigo-300 rounded-lg p-3.5 cursor-pointer bg-indigo-50/40 hover:bg-indigo-50 transition text-center">
                <Upload className="w-5 h-5 text-indigo-600 mb-1" />
                <span className="text-xs font-semibold text-indigo-800">
                  {uploadedFile ? uploadedFile.name : "Selecionar .csv ou .xlsx"}
                </span>
                <span className="text-[11px] text-slate-500 mt-0.5">
                  Separador (`,` / `;`) e abas detectados automaticamente
                </span>
                <input
                  type="file"
                  accept=".csv,.txt,.xlsx,.xls"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) onFileSelected(f);
                  }}
                />
              </label>

              {uploadMeta && (
                <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs">
                  <div className="flex items-center justify-between text-[11px] text-slate-600">
                    <span className="font-semibold flex items-center gap-1">
                      <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600" />
                      {uploadMeta.total_rows} linhas
                    </span>
                    {uploadMeta.detected_separator && (
                      <span className="px-1.5 py-0.5 bg-white border rounded font-mono">
                        sep: "{uploadMeta.detected_separator}"
                      </span>
                    )}
                  </div>

                  {uploadMeta.sheets.length > 0 && (
                    <div>
                      <label className="block text-[11px] font-medium text-slate-600 mb-0.5">
                        Aba do Excel
                      </label>
                      <select
                        value={config.sheetName}
                        onChange={(e) => onSheetChange(e.target.value)}
                        className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                      >
                        {uploadMeta.sheets.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[11px] font-medium text-slate-600 mb-0.5">
                        Coluna Data
                      </label>
                      <select
                        value={config.dateCol}
                        onChange={(e) => setConfig((c) => ({ ...c, dateCol: e.target.value }))}
                        className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                      >
                        {uploadMeta.columns.map((col) => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-600 mb-0.5">
                        Coluna Valor
                      </label>
                      <select
                        value={config.valueCol}
                        onChange={(e) => setConfig((c) => ({ ...c, valueCol: e.target.value }))}
                        className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                      >
                        {uploadMeta.columns.map((col) => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="block text-[11px] font-medium text-slate-600 mb-0.5">
                        Formato Data
                      </label>
                      <select
                        value={config.dateFormat}
                        onChange={(e) => setConfig((c) => ({ ...c, dateFormat: e.target.value }))}
                        className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                      >
                        <option value="auto">Automático</option>
                        <option value="ymd">YYYY-MM-DD</option>
                        <option value="dmy">DD/MM/YYYY</option>
                        <option value="mdy">MM/DD/YYYY</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] font-medium text-slate-600 mb-0.5">
                        Agregação
                      </label>
                      <select
                        value={config.aggregation}
                        onChange={(e) =>
                          setConfig((c) => ({ ...c, aggregation: e.target.value as AggregationType }))
                        }
                        className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                      >
                        <option value="none">Manter original</option>
                        <option value="month_mean">Média Mensal</option>
                        <option value="month_sum">Soma Mensal</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Transformação & Testes */}
        <section className="space-y-3 border-t border-slate-200 pt-4">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            2. Transformação & Testes
          </label>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1">
              Transformação da Série
            </label>
            <select
              value={config.transform}
              onChange={(e) =>
                setConfig((c) => ({ ...c, transform: e.target.value as TransformType }))
              }
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-800"
            >
              {TRANSFORMS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Significância (α)
              </label>
              <select
                value={config.alpha}
                onChange={(e) => setConfig((c) => ({ ...c, alpha: Number(e.target.value) }))}
                className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
              >
                <option value={0.01}>1% (α = 0,01)</option>
                <option value={0.05}>5% (α = 0,05)</option>
                <option value={0.1}>10% (α = 0,10)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Termos Determ.
              </label>
              <select
                value={config.regression}
                onChange={(e) =>
                  setConfig((c) => ({ ...c, regression: e.target.value as RegressionType }))
                }
                className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
              >
                <option value="ct">Const. + Tend. (ct)</option>
                <option value="c">Apenas Const. (c)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Período Sazonal (s)
              </label>
              <select
                value={config.seasonalPeriod}
                onChange={(e) =>
                  setConfig((c) => ({ ...c, seasonalPeriod: Number(e.target.value) }))
                }
                className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
              >
                <option value={12}>Mensal (s = 12)</option>
                <option value={4}>Trimestral (s = 4)</option>
                <option value={7}>Diária (s = 7)</option>
                <option value={52}>Semanal (s = 52)</option>
                <option value={1}>Sem Sazonalidade (s = 1)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Lags LB / ARCH
              </label>
              <input
                type="number"
                min={4}
                max={24}
                value={config.maxLags}
                onChange={(e) =>
                  setConfig((c) => ({
                    ...c,
                    maxLags: Math.min(24, Math.max(4, Number(e.target.value) || 12)),
                  }))
                }
                className="w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
              />
            </div>
          </div>
        </section>

        {/* Modelagem SARIMA */}
        <section className="space-y-2.5 border-t border-slate-200 pt-4">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              3. Modelo SARIMA & Previsão
            </label>
            <label className="flex items-center gap-1.5 text-xs font-medium text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={config.autoArima}
                onChange={(e) => setConfig((c) => ({ ...c, autoArima: e.target.checked }))}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              Auto (AICc)
            </label>
          </div>

          {!config.autoArima && (
            <div className="space-y-2 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
              <div className="grid grid-cols-3 gap-1.5 text-xs">
                {(["orderP", "orderD", "orderQ"] as const).map((k, idx) => (
                  <div key={k}>
                    <span className="text-[11px] text-slate-500">{["p", "d", "q"][idx]}</span>
                    <input
                      type="number"
                      min={0}
                      max={idx === 1 ? 2 : 5}
                      value={config[k]}
                      onChange={(e) => setConfig((c) => ({ ...c, [k]: Number(e.target.value) || 0 }))}
                      className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                    />
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-3 gap-1.5 text-xs">
                {(["seasonalP", "seasonalD", "seasonalQ"] as const).map((k, idx) => (
                  <div key={k}>
                    <span className="text-[11px] text-slate-500">{["P", "D", "Q"][idx]}</span>
                    <input
                      type="number"
                      min={0}
                      max={idx === 1 ? 1 : 2}
                      value={config[k]}
                      onChange={(e) => setConfig((c) => ({ ...c, [k]: Number(e.target.value) || 0 }))}
                      className="w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          <div>
            <div className="flex justify-between text-xs text-slate-700 mb-1">
              <span className="font-medium">Horizonte de Previsão (h)</span>
              <span className="font-bold text-indigo-700">{config.forecastHorizon} períodos</span>
            </div>
            <input
              type="range"
              min={3}
              max={36}
              value={config.forecastHorizon}
              onChange={(e) =>
                setConfig((c) => ({ ...c, forecastHorizon: Number(e.target.value) }))
              }
              className="w-full accent-indigo-600"
            />
          </div>
        </section>
      </div>

      {/* Botões de Ação */}
      <div className="p-4 border-t border-slate-200 bg-slate-50 space-y-2">
        <button
          type="button"
          onClick={onRun}
          disabled={loading || (config.sourceMode === "upload" && !uploadedFile)}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-semibold py-2.5 px-4 text-xs shadow-sm transition"
        >
          <Play className="w-4 h-4" />
          {loading ? "Executando Bateria..." : "Executar Diagnóstico Completo"}
        </button>

        <button
          type="button"
          onClick={onExportExcel}
          disabled={exporting || loading || (config.sourceMode === "upload" && !uploadedFile)}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 text-white font-semibold py-2 px-4 text-xs shadow-sm transition"
        >
          <Download className="w-4 h-4" />
          {exporting ? "Gerando .xlsx..." : "Exportar Relatório Excel (5 Abas)"}
        </button>
      </div>
    </aside>
  );
};
