import React, { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  Database,
  FileSpreadsheet,
  Play,
  Sliders,
  Sparkles,
  Upload,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { checkApiHealth, loadDemoDataset, runDiagnostics, runForecast, uploadDataset } from "./api";
import type { DatasetSummary, DiagnosticsResponse, ForecastResponse } from "./types";
import { DiagnosticsPanel } from "./components/DiagnosticsPanel";
import { ForecastChart } from "./components/ForecastChart";
import { ResidualsPanel } from "./components/ResidualsPanel";
import { GrowthTable } from "./components/GrowthTable";

type TabId = "overview" | "diagnostics" | "forecast" | "residuals" | "growth";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [dateCol, setDateCol] = useState("");
  const [targetCol, setTargetCol] = useState("");
  const [exogCols, setExogCols] = useState<string[]>([]);

  // Parâmetros de modelagem SARIMAX
  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const [p, setP] = useState(1);
  const [d, setD] = useState(1);
  const [q, setQ] = useState(1);
  const [P, setSeasonalP] = useState(1);
  const [D, setSeasonalD] = useState(1);
  const [Q, setSeasonalQ] = useState(1);
  const [s, setS] = useState(12);
  const [useLog, setUseLog] = useState(true);
  const [testPeriods, setTestPeriods] = useState(12);
  const [forecastHorizon, setForecastHorizon] = useState(24);
  const [confidenceLevel, setConfidenceLevel] = useState(0.95);

  // Diagnóstico interativo
  const [diagDiff, setDiagDiff] = useState(0);
  const [diagSeasonalDiff, setDiagSeasonalDiff] = useState(0);
  const [diagnostics, setDiagnostics] = useState<DiagnosticsResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);

  // Estados de carregamento, conexão e erro
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [loadingData, setLoadingData] = useState(false);
  const [loadingDiag, setLoadingDiag] = useState(false);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      const ok = await checkApiHealth();
      if (active) setApiStatus(ok ? "online" : "offline");
    };
    poll();
    const timer = window.setInterval(poll, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const availableExogCols =
    dataset?.numeric_columns.filter((c) => c !== targetCol && c !== dateCol) ?? [];

  const handleLoadDemo = async () => {
    setErrorMsg(null);
    setLoadingData(true);
    try {
      const meta = await loadDemoDataset();
      setDataset(meta);
      setDateCol(meta.default_date_col);
      setTargetCol(meta.default_target_col);
      // Seleciona exógenas reais do dataset demo por padrão para demonstrar SARIMAX
      const defaults = ["Temperatura_media", "dias_uteis"].filter((c) =>
        meta.numeric_columns.includes(c)
      );
      setExogCols(defaults);
      setForecast(null);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Erro ao carregar dataset demo.");
    } finally {
      setLoadingData(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    setLoadingData(true);
    try {
      const meta = await uploadDataset(file);
      setDataset(meta);
      setDateCol(meta.default_date_col);
      setTargetCol(meta.default_target_col);
      setExogCols([]);
      setForecast(null);
      setDiagnostics(null);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Erro no upload do arquivo.");
    } finally {
      setLoadingData(false);
      e.target.value = "";
    }
  };

  const fetchDiagnostics = async (customD = diagDiff, customSD = diagSeasonalDiff) => {
    if (!dataset || !dateCol || !targetCol) return;
    setErrorMsg(null);
    setLoadingDiag(true);
    try {
      const res = await runDiagnostics({
        dataset_id: dataset.dataset_id,
        date_col: dateCol,
        target_col: targetCol,
        use_log: useLog,
        diff_order: customD,
        seasonal_diff: customSD,
        seasonal_period: s,
        nlags: 24,
      });
      setDiagnostics(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Erro no diagnóstico Box-Jenkins.");
    } finally {
      setLoadingDiag(false);
    }
  };

  const handleRunForecast = async () => {
    if (!dataset || !dateCol || !targetCol) return;
    setErrorMsg(null);
    setLoadingForecast(true);
    try {
      const res = await runForecast({
        dataset_id: dataset.dataset_id,
        date_col: dateCol,
        target_col: targetCol,
        exog_cols: exogCols,
        mode,
        p,
        d,
        q,
        P,
        D,
        Q,
        s,
        use_log: useLog,
        test_periods: testPeriods,
        forecast_horizon: forecastHorizon,
        confidence_level: confidenceLevel,
      });
      setForecast(res);
      setActiveTab("forecast");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Erro ao executar SARIMAX.");
    } finally {
      setLoadingForecast(false);
    }
  };

  useEffect(() => {
    if (dataset && dateCol && targetCol && activeTab === "diagnostics") {
      fetchDiagnostics(diagDiff, diagSeasonalDiff);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset?.dataset_id, dateCol, targetCol, useLog, s, activeTab]);

  const toggleExog = (col: string) => {
    setExogCols((prev) =>
      prev.includes(col) ? prev.filter((item) => item !== col) : [...prev, col]
    );
  };

  const tabs: { id: TabId; label: string; icon: React.ReactNode; disabled?: boolean }[] = [
    { id: "overview", label: "1. Guia & Dados", icon: <BookOpen className="h-4 w-4" /> },
    { id: "diagnostics", label: "2. Box-Jenkins (ACF/ADF)", icon: <Activity className="h-4 w-4" />, disabled: !dataset },
    { id: "forecast", label: "3. Previsão SARIMAX", icon: <BarChart3 className="h-4 w-4" />, disabled: !forecast },
    { id: "residuals", label: "4. Resíduos & Coeficientes", icon: <Sliders className="h-4 w-4" />, disabled: !forecast },
    { id: "growth", label: "5. Crescimento & Excel", icon: <FileSpreadsheet className="h-4 w-4" />, disabled: !forecast },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Navbar */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-5 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 text-white shadow-lg shadow-indigo-950">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-extrabold tracking-tight text-white">
                SARIMAX Forecast Studio
              </h1>
              <p className="text-xs text-slate-400">
                FastAPI + Statsmodels + React — Metodologia Box-Jenkins Interativa
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-3 sm:flex">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ring-1 ${
                apiStatus === "online"
                  ? "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30"
                  : apiStatus === "checking"
                  ? "bg-amber-500/15 text-amber-300 ring-amber-500/30"
                  : "bg-rose-500/15 text-rose-300 ring-rose-500/30"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "online"
                    ? "bg-emerald-400"
                    : apiStatus === "checking"
                    ? "animate-pulse bg-amber-400"
                    : "bg-rose-400"
                }`}
              />
              {apiStatus === "online"
                ? "API Python Online"
                : apiStatus === "checking"
                ? "Conectando API..."
                : "API Offline (porta 8000)"}
            </span>
            {dataset && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-500/30">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {dataset.filename} ({dataset.observed_rows} obs)
              </span>
            )}
            <span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1 text-xs text-slate-400">
              Dr. Silvio da Rosa Paula
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1600px] grid-cols-1 gap-6 p-5 lg:grid-cols-12">
        {/* Sidebar de Configuração */}
        <aside className="space-y-5 lg:col-span-3">
          {/* Card 1: Fonte de Dados */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-lg">
            <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-300">
              <Database className="h-4 w-4" />
              1. Fonte de Dados
            </div>

            <div className="space-y-2.5">
              <button
                onClick={handleLoadDemo}
                disabled={loadingData}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-600 px-3.5 py-2.5 text-xs font-semibold text-white transition hover:bg-indigo-500 disabled:opacity-50"
              >
                <Sparkles className="h-4 w-4" />
                {loadingData ? "Carregando..." : "Carregar Dataset Demo (Energia)"}
              </button>

              <label className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-950/60 px-3.5 py-2.5 text-xs font-semibold text-slate-200 transition hover:border-indigo-500 hover:text-white">
                <Upload className="h-4 w-4 text-slate-400" />
                Enviar CSV ou Excel (.xlsx)
                <input
                  type="file"
                  accept=".xlsx,.xls,.csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            {dataset && (
              <div className="mt-4 space-y-3 border-t border-slate-800 pt-4">
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-400">Coluna de Data</label>
                  <select
                    value={dateCol}
                    onChange={(e) => setDateCol(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-white focus:border-indigo-500 focus:outline-none"
                  >
                    {dataset.columns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-400">Série Alvo (Y)</label>
                  <select
                    value={targetCol}
                    onChange={(e) => setTargetCol(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-white focus:border-indigo-500 focus:outline-none"
                  >
                    {dataset.numeric_columns.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                {availableExogCols.length > 0 && (
                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-slate-400">
                      Variáveis Exógenas (X — Opcional / SARIMAX)
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                      {availableExogCols.map((col) => {
                        const active = exogCols.includes(col);
                        return (
                          <button
                            key={col}
                            type="button"
                            onClick={() => toggleExog(col)}
                            className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition ${
                              active
                                ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-500/50"
                                : "bg-slate-950 text-slate-400 ring-1 ring-slate-800 hover:text-slate-200"
                            }`}
                          >
                            {col}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card 2: Configuração SARIMA / SARIMAX */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-lg">
            <div className="mb-3 flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-indigo-300">
              <Sliders className="h-4 w-4" />
              2. Configuração do Modelo
            </div>

            <div className="space-y-3.5 text-xs">
              <div className="grid grid-cols-2 gap-1.5 rounded-xl bg-slate-950 p-1">
                <button
                  type="button"
                  onClick={() => setMode("auto")}
                  className={`rounded-lg py-1.5 font-semibold transition ${
                    mode === "auto" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Auto-SARIMAX
                </button>
                <button
                  type="button"
                  onClick={() => setMode("manual")}
                  className={`rounded-lg py-1.5 font-semibold transition ${
                    mode === "manual" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
                  }`}
                >
                  Manual (p,d,q)
                </button>
              </div>

              {mode === "manual" && (
                <div className="space-y-2.5 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-[11px] font-semibold text-slate-300">Ordens Não-Sazonais (p, d, q)</div>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { lbl: "p (AR)", val: p, set: setP, max: 4 },
                      { lbl: "d (Diff)", val: d, set: setD, max: 2 },
                      { lbl: "q (MA)", val: q, set: setQ, max: 4 },
                    ].map((item) => (
                      <div key={item.lbl}>
                        <label className="text-[10px] text-slate-400">{item.lbl}</label>
                        <input
                          type="number"
                          min={0}
                          max={item.max}
                          value={item.val}
                          onChange={(e) => item.set(Number(e.target.value))}
                          className="mt-0.5 w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-center font-mono text-white"
                        />
                      </div>
                    ))}
                  </div>

                  <div className="pt-1 text-[11px] font-semibold text-slate-300">Ordens Sazonais (P, D, Q)</div>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { lbl: "P", val: P, set: setSeasonalP, max: 2 },
                      { lbl: "D", val: D, set: setSeasonalD, max: 1 },
                      { lbl: "Q", val: Q, set: setSeasonalQ, max: 2 },
                    ].map((item) => (
                      <div key={item.lbl}>
                        <label className="text-[10px] text-slate-400">{item.lbl}</label>
                        <input
                          type="number"
                          min={0}
                          max={item.max}
                          value={item.val}
                          onChange={(e) => item.set(Number(e.target.value))}
                          className="mt-0.5 w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-center font-mono text-white"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="mb-1 block text-slate-400">Período Sazonal (s)</label>
                  <input
                    type="number"
                    min={1}
                    max={52}
                    value={s}
                    onChange={(e) => setS(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-slate-400">Backtest (meses)</label>
                  <input
                    type="number"
                    min={3}
                    max={36}
                    value={testPeriods}
                    onChange={(e) => setTestPeriods(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div>
                  <label className="mb-1 block text-slate-400">Horizonte Futuro</label>
                  <input
                    type="number"
                    min={1}
                    max={120}
                    value={forecastHorizon}
                    onChange={(e) => setForecastHorizon(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 font-mono text-white"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-slate-400">Confiança IC</label>
                  <select
                    value={confidenceLevel}
                    onChange={(e) => setConfidenceLevel(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-white"
                  >
                    <option value={0.8}>80%</option>
                    <option value={0.9}>90%</option>
                    <option value={0.95}>95%</option>
                  </select>
                </div>
              </div>

              <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-2">
                <span className="text-slate-300">Transformação Log (log y)</span>
                <input
                  type="checkbox"
                  checked={useLog}
                  onChange={(e) => setUseLog(e.target.checked)}
                  className="h-4 w-4 accent-indigo-500"
                />
              </label>

              <button
                onClick={handleRunForecast}
                disabled={!dataset || loadingForecast}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-4 py-3 text-xs font-bold text-white shadow-lg shadow-emerald-950/60 transition hover:from-emerald-500 hover:to-teal-500 disabled:opacity-40"
              >
                {loadingForecast ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Estimando SARIMAX...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-current" />
                    Executar Modelagem &amp; Previsão
                  </>
                )}
              </button>
            </div>
          </div>
        </aside>

        {/* Área Principal com Abas */}
        <main className="space-y-5 lg:col-span-9">
          {/* Navegação por Abas */}
          <div className="flex flex-wrap gap-1.5 rounded-2xl border border-slate-800 bg-slate-900/70 p-1.5">
            {tabs.map((t) => (
              <button
                key={t.id}
                disabled={t.disabled}
                onClick={() => setActiveTab(t.id)}
                className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-2 text-xs font-semibold transition ${
                  activeTab === t.id
                    ? "bg-indigo-600 text-white shadow"
                    : t.disabled
                    ? "cursor-not-allowed text-slate-600"
                    : "text-slate-300 hover:bg-slate-800 hover:text-white"
                }`}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </div>

          {errorMsg && (
            <div className="flex items-center gap-2.5 rounded-xl border border-rose-500/40 bg-rose-950/50 px-4 py-3 text-xs text-rose-200">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* ABA 1: Guia Didático Box-Jenkins + Preview dos Dados */}
          {activeTab === "overview" && (
            <div className="space-y-6">
              <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/50 p-6">
                <h2 className="text-lg font-extrabold text-white">
                  Guia Didático: Como Funciona a Metodologia Box-Jenkins (SARIMA / SARIMAX)?
                </h2>
                <p className="mt-2 text-sm leading-relaxed text-slate-300">
                  Modelos <strong className="text-indigo-300">ARIMA(p, d, q)</strong> e sua extensão sazonal com
                  variáveis exógenas <strong className="text-cyan-300">SARIMAX(p, d, q)×(P, D, Q, s)</strong> combinam
                  três blocos intuitivos para prever séries temporais:
                </p>

                <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-3">
                  <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                    <div className="text-xs font-bold text-indigo-400">AR (p) — Autorregressivo</div>
                    <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                      Usa valores passados da própria série (memória temporal) para prever o próximo período. O gráfico{" "}
                      <strong className="text-slate-200">PACF</strong> ajuda a identificar a ordem <code>p</code>.
                    </p>
                  </div>
                  <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                    <div className="text-xs font-bold text-emerald-400">I (d) — Integração / Diferenciação</div>
                    <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                      Remove tendências para tornar a série estacionária. Os testes{" "}
                      <strong className="text-slate-200">ADF e KPSS</strong> (na Aba 2) indicam exatamente qual{" "}
                      <code>d</code> utilizar.
                    </p>
                  </div>
                  <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                    <div className="text-xs font-bold text-amber-400">MA (q) + Exógenas (X)</div>
                    <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                      Corrige o futuro com base nos erros de previsão anteriores (identificado via{" "}
                      <strong className="text-slate-200">ACF</strong>) e incorpora choques externos como clima e tarifas.
                    </p>
                  </div>
                </div>

                {!dataset && (
                  <div className="mt-5 flex flex-wrap items-center gap-3">
                    <button
                      onClick={handleLoadDemo}
                      disabled={loadingData}
                      className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-bold text-white transition hover:bg-indigo-500"
                    >
                      <Sparkles className="h-4 w-4" />
                      Começar Agora com o Dataset Demo (Consumo de Energia)
                    </button>
                  </div>
                )}
              </div>

              {dataset && (
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
                  <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-white">Prévia dos Dados Carregados — {dataset.filename}</h3>
                      <p className="text-xs text-slate-400">
                        Período: {dataset.date_min} até {dataset.date_max} • {dataset.observed_rows} observações reais
                        {dataset.future_exog_rows > 0
                          ? ` + ${dataset.future_exog_rows} períodos futuros com variáveis exógenas preenchidas`
                          : ""}
                      </p>
                    </div>
                    <button
                      onClick={() => setActiveTab("diagnostics")}
                      className="rounded-xl bg-indigo-500/20 px-3.5 py-2 text-xs font-semibold text-indigo-300 ring-1 ring-indigo-500/40 hover:bg-indigo-500/30"
                    >
                      Ir para Diagnóstico Box-Jenkins →
                    </button>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400">
                          {dataset.columns.map((c) => (
                            <th key={c} className="py-2 pr-4 font-semibold">{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {dataset.preview.map((row, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/40">
                            {dataset.columns.map((c) => (
                              <td key={c} className="py-2 pr-4 font-mono text-slate-300">
                                {row[c] == null ? "—" : String(row[c])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ABA 2: Diagnóstico Box-Jenkins */}
          {activeTab === "diagnostics" && (
            <DiagnosticsPanel
              data={diagnostics}
              loading={loadingDiag}
              diffOrder={diagDiff}
              seasonalDiff={diagSeasonalDiff}
              onChangeDiff={(nd, nsd) => {
                setDiagDiff(nd);
                setDiagSeasonalDiff(nsd);
                fetchDiagnostics(nd, nsd);
              }}
              onApplyRecommendation={(rd, rsd) => {
                setD(rd);
                setSeasonalD(rsd);
                setDiagDiff(rd);
                setDiagSeasonalDiff(rsd);
                fetchDiagnostics(rd, rsd);
              }}
            />
          )}

          {/* ABA 3: Previsão SARIMAX */}
          {activeTab === "forecast" && forecast && <ForecastChart data={forecast} />}

          {/* ABA 4: Resíduos & Coeficientes */}
          {activeTab === "residuals" && forecast && <ResidualsPanel data={forecast} />}

          {/* ABA 5: Crescimento Anual & Excel */}
          {activeTab === "growth" && forecast && dataset && (
            <GrowthTable datasetId={dataset.dataset_id} rows={forecast.annual_growth} />
          )}
        </main>
      </div>
    </div>
  );
}
