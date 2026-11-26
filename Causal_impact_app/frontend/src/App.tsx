import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BookOpen,
  Calendar,
  Database,
  FileUp,
  FlaskConical,
  LineChart,
  Play,
  SlidersHorizontal,
  Sparkles,
  Table2,
  UploadCloud,
} from "lucide-react";
import {
  fetchDemos,
  fetchHealth,
  loadDemoDataset,
  runCausalAnalysis,
  uploadDatasetFile,
} from "./api";
import { AuditPlaceboPanel } from "./components/AuditPlaceboPanel";
import { CausalChartsPanel } from "./components/CausalChartsPanel";
import { DataUploadPanel } from "./components/DataUploadPanel";
import { ExploratoryPanel } from "./components/ExploratoryPanel";
import { ImpactTableExportPanel } from "./components/ImpactTableExportPanel";
import { MethodologyGuide } from "./components/MethodologyGuide";
import type {
  AnalyzeResponse,
  DatasetSummary,
  DemoDatasetMeta,
  DeseasonalizeMode,
  ModelType,
} from "./types";

type TabId = "data" | "impact" | "audit" | "table" | "explore" | "guide";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("data");
  const [apiOnline, setApiOnline] = useState(false);
  const [demos, setDemos] = useState<DemoDatasetMeta[]>([]);
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);

  const [isLoadingData, setIsLoadingData] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Parâmetros do formulário BSTS
  const [dateCol, setDateCol] = useState("data");
  const [yCol, setYCol] = useState("");
  const [xCols, setXCols] = useState<string[]>([]);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [interventionDate, setInterventionDate] = useState("");
  const [alpha, setAlpha] = useState(0.05);
  const [modelType, setModelType] = useState<ModelType>("local_level");
  const [seasonalPeriod, setSeasonalPeriod] = useState(12);
  const [deseasonalize, setDeseasonalize] = useState<DeseasonalizeMode>("none");
  const [stlPeriod, setStlPeriod] = useState(12);
  const [nSims, setNSims] = useState(1000);
  const [runPlacebo, setRunPlacebo] = useState(true);

  const syncFormFromDataset = (ds: DatasetSummary) => {
    setDataset(ds);
    setDateCol(ds.date_column);
    setYCol(ds.suggested_y);
    setXCols(ds.suggested_xs);
    setStartDate(ds.min_date);
    setEndDate(ds.max_date);
    setInterventionDate(ds.suggested_intervention_date);
    setAnalysis(null);
  };

  const executeAnalysis = useCallback(
    async (overrides?: {
      date_col: string;
      y_col: string;
      x_cols: string[];
      start_date: string;
      end_date: string;
      intervention_date: string;
    }) => {
      setIsAnalyzing(true);
      setErrorMsg(null);
      try {
        const res = await runCausalAnalysis({
          date_col: overrides?.date_col ?? dateCol,
          y_col: overrides?.y_col ?? yCol,
          x_cols: overrides?.x_cols ?? xCols,
          start_date: (overrides?.start_date ?? startDate) || null,
          end_date: (overrides?.end_date ?? endDate) || null,
          intervention_date: overrides?.intervention_date ?? interventionDate,
          alpha,
          model_type: modelType,
          seasonal_period: seasonalPeriod,
          deseasonalize,
          stl_period: stlPeriod,
          n_sims: nSims,
          run_placebo: runPlacebo,
        });
        setAnalysis(res);
        setActiveTab("impact");
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "Erro ao executar análise causal.");
      } finally {
        setIsAnalyzing(false);
      }
    },
    [
      alpha,
      dateCol,
      deseasonalize,
      endDate,
      interventionDate,
      modelType,
      nSims,
      runPlacebo,
      seasonalPeriod,
      startDate,
      stlPeriod,
      xCols,
      yCol,
    ]
  );

  const bootstrap = useCallback(async () => {
    try {
      const h = await fetchHealth();
      if (h.app_name !== "CausalImpact Studio API") {
        throw new Error(
          `Outro serviço está respondendo na porta da API (${h.app_name || "desconhecido"}). ` +
            "Reinicie via iniciar_app.bat ou suba o CausalImpact na porta 8002."
        );
      }
      const demoList = await fetchDemos();
      setApiOnline(true);
      setDemos(demoList);
      setErrorMsg(null);
    } catch (err) {
      setApiOnline(false);
      setErrorMsg(
        err instanceof Error
          ? `Backend FastAPI indisponível: ${err.message}. Inicie o servidor Python na porta 8002.`
          : "Backend FastAPI indisponível na porta 8002."
      );
    }
  }, []);

  useEffect(() => {
    bootstrap();
  }, [bootstrap]);

  const handleSelectDemo = async (demoId: string, autoRun = true) => {
    setIsLoadingData(true);
    setErrorMsg(null);
    try {
      const ds = await loadDemoDataset(demoId);
      syncFormFromDataset(ds);
      if (autoRun) {
        await executeAnalysis({
          date_col: ds.date_column,
          y_col: ds.suggested_y,
          x_cols: ds.suggested_xs,
          start_date: ds.min_date,
          end_date: ds.max_date,
          intervention_date: ds.suggested_intervention_date,
        });
      } else {
        setActiveTab("data");
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao carregar dataset demo.");
    } finally {
      setIsLoadingData(false);
    }
  };

  const handleUploadFile = async (file: File) => {
    setIsLoadingData(true);
    setErrorMsg(null);
    try {
      const ds = await uploadDatasetFile(file);
      syncFormFromDataset(ds);
      // Mantém na tela de Dados & Upload para o usuário revisar colunas e data da intervenção antes de rodar!
      setActiveTab("data");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao processar arquivo enviado.");
    } finally {
      setIsLoadingData(false);
    }
  };

  const toggleXCol = (col: string) => {
    setXCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  };

  const candidateXs = (dataset?.numeric_columns ?? []).filter((c) => c !== yCol && c !== dateCol);

  const tabs: {
    id: TabId;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    requiresAnalysis?: boolean;
  }[] = [
    { id: "data", label: "Dados & Upload", icon: UploadCloud },
    { id: "impact", label: "Impacto Causal (3 Painéis)", icon: Activity, requiresAnalysis: true },
    { id: "audit", label: "Auditoria & Placebo", icon: FlaskConical, requiresAnalysis: true },
    { id: "table", label: "Tabela & Exportação", icon: Table2, requiresAnalysis: true },
    { id: "explore", label: "Exploração & Descritivas", icon: LineChart, requiresAnalysis: true },
    { id: "guide", label: "Guia Metodológico", icon: BookOpen },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header Fixo */}
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-5 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 text-slate-950 shadow-lg shadow-emerald-500/20">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold tracking-tight text-white">CausalImpact Studio</h1>
                <span className="rounded-md bg-emerald-500/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-emerald-300 ring-1 ring-emerald-500/30">
                  BSTS Python v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Inferência Causal Contrafactual em Séries Temporais • State-Space Kalman + Monte Carlo + Placebo Audit
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-cyan-500/40 bg-cyan-500/15 px-3.5 py-1.5 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/25">
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                disabled={isLoadingData}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  e.target.value = "";
                  if (file) handleUploadFile(file);
                }}
                className="hidden"
              />
              <FileUp className="h-3.5 w-3.5" />
              Enviar CSV / XLSX
            </label>

            {dataset && (
              <div className="hidden items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 md:flex">
                <Database className="h-3.5 w-3.5 text-emerald-400" />
                <span className="max-w-[240px] truncate">
                  <strong className="text-white">{dataset.name}</strong> ({dataset.rows} obs.)
                </span>
              </div>
            )}

            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${
                apiOnline
                  ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-300"
                  : "border-rose-500/35 bg-rose-500/10 text-rose-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${apiOnline ? "animate-pulse bg-emerald-400" : "bg-rose-400"}`}
              />
              {apiOnline ? "API Python Online (:8002)" : "API Offline"}
            </div>
          </div>
        </div>

        {/* Barra de Abas */}
        <div className="mx-auto flex max-w-[1600px] gap-1 overflow-x-auto px-5 pb-2.5">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            const disabled = Boolean(tab.requiresAnalysis && !analysis);
            return (
              <button
                key={tab.id}
                type="button"
                disabled={disabled}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex shrink-0 items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition ${
                  active
                    ? "bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-500/40"
                    : disabled
                      ? "cursor-not-allowed text-slate-600"
                      : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
                title={disabled ? "Carregue uma base e execute a análise para liberar esta aba" : undefined}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* Corpo Principal */}
      <div className="mx-auto grid max-w-[1600px] grid-cols-1 gap-6 px-5 py-6 lg:grid-cols-12">
        {/* Sidebar Esquerda de Configuração */}
        <aside className="space-y-4 lg:col-span-3">
          {/* Card 1: Fonte de Dados (Upload em destaque + Demos) */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2">
              <Database className="h-4 w-4 text-cyan-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                1. Base de Dados
              </h2>
            </div>

            <div className="space-y-2.5">
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl bg-cyan-500/15 px-3 py-2.5 text-xs font-bold text-cyan-200 ring-1 ring-cyan-500/40 transition hover:bg-cyan-500/25">
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  disabled={isLoadingData}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    e.target.value = "";
                    if (file) handleUploadFile(file);
                  }}
                  className="hidden"
                />
                <FileUp className="h-4 w-4" />
                {isLoadingData ? "Carregando..." : "Enviar meu CSV / XLSX"}
              </label>

              <p className="pt-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                Ou usar cenário de demonstração:
              </p>

              {demos.map((d) => {
                const active = dataset?.dataset_id === d.id;
                return (
                  <button
                    key={d.id}
                    type="button"
                    disabled={isLoadingData || isAnalyzing}
                    onClick={() => handleSelectDemo(d.id, true)}
                    className={`w-full rounded-xl border p-2.5 text-left transition ${
                      active
                        ? "border-emerald-400/60 bg-emerald-950/25"
                        : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
                    }`}
                  >
                    <span className="block text-xs font-bold text-white">{d.name}</span>
                    <span className="mt-0.5 block text-[10px] text-cyan-300">{d.frequency_label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Card 2: Variáveis & Janela da Intervenção */}
          {dataset && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
              <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2">
                <Calendar className="h-4 w-4 text-emerald-400" />
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                  2. Variáveis &amp; Intervenção
                </h2>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="mb-1 block text-slate-400">Coluna de Data</label>
                  <select
                    value={dateCol}
                    onChange={(e) => setDateCol(e.target.value)}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-slate-100 focus:border-cyan-400 focus:outline-none"
                  >
                    {dataset.columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="mb-1 block text-slate-400">Variável Resposta (Y — Tratada)</label>
                  <select
                    value={yCol}
                    onChange={(e) => {
                      const nextY = e.target.value;
                      setYCol(nextY);
                      setXCols((prev) => prev.filter((c) => c !== nextY));
                    }}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-slate-100 focus:border-emerald-400 focus:outline-none"
                  >
                    {dataset.numeric_columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <div className="mb-1 flex items-center justify-between">
                    <label className="text-slate-400">Controles (X — Não Afetados)</label>
                    <button
                      type="button"
                      onClick={() => setXCols(xCols.length > 0 ? [] : candidateXs)}
                      className="text-[10px] font-semibold text-cyan-300 hover:underline"
                    >
                      {xCols.length > 0 ? "Limpar (Univariado)" : "Selecionar todos"}
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {candidateXs.map((c) => {
                      const selected = xCols.includes(c);
                      return (
                        <button
                          key={c}
                          type="button"
                          onClick={() => toggleXCol(c)}
                          className={`rounded-lg px-2 py-1 font-mono text-[11px] transition ${
                            selected
                              ? "bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-500/40"
                              : "bg-slate-950 text-slate-400 hover:text-slate-200"
                          }`}
                        >
                          {c}
                        </button>
                      );
                    })}
                    {candidateXs.length === 0 && (
                      <span className="text-[11px] italic text-slate-500">Nenhuma outra coluna numérica.</span>
                    )}
                  </div>
                </div>

                <div>
                  <label className="mb-1 block font-semibold text-amber-300">Data de Início da Intervenção</label>
                  <input
                    type="date"
                    value={interventionDate}
                    min={startDate}
                    max={endDate}
                    onChange={(e) => setInterventionDate(e.target.value)}
                    className="w-full rounded-xl border border-amber-500/40 bg-slate-950 px-3 py-2 font-mono text-slate-100 focus:border-amber-400 focus:outline-none"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="mb-1 block text-[11px] text-slate-400">Início</label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-[11px] text-slate-200"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-[11px] text-slate-400">Fim</label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-[11px] text-slate-200"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Card 3: Especificação do Modelo BSTS */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2">
              <SlidersHorizontal className="h-4 w-4 text-violet-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                3. Modelo Estrutural (BSTS)
              </h2>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="mb-1 block text-slate-400">Especificação de Estado</label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value as ModelType)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-violet-400 focus:outline-none"
                >
                  <option value="local_level">Local Level (BSTS Clássico)</option>
                  <option value="local_linear_trend">Local Linear Trend (Tendência)</option>
                  <option value="seasonal_local_level">Local Level + Sazonalidade Estocástica</option>
                </select>
              </div>

              {modelType === "seasonal_local_level" && (
                <div>
                  <label className="mb-1 block text-slate-400">Período Sazonal (ex.: 12 mensal, 7 diário)</label>
                  <input
                    type="number"
                    min={2}
                    max={52}
                    value={seasonalPeriod}
                    onChange={(e) => setSeasonalPeriod(Number(e.target.value))}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-slate-100"
                  />
                </div>
              )}

              <div>
                <label className="mb-1 block text-slate-400">Pré-Desazonalização STL (Opcional)</label>
                <select
                  value={deseasonalize}
                  onChange={(e) => setDeseasonalize(e.target.value as DeseasonalizeMode)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"
                >
                  <option value="none">Não (Série Original)</option>
                  <option value="stl_deseasonalize">STL: Tendência + Resíduo</option>
                  <option value="stl_trend_only">STL: Somente Tendência</option>
                </select>
              </div>

              {deseasonalize !== "none" && (
                <div>
                  <label className="mb-1 block text-slate-400">Frequência STL</label>
                  <input
                    type="number"
                    min={2}
                    max={52}
                    value={stlPeriod}
                    onChange={(e) => setStlPeriod(Number(e.target.value))}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-slate-100"
                  />
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="mb-1 block text-slate-400">Significância (α)</label>
                  <select
                    value={alpha}
                    onChange={(e) => setAlpha(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-slate-100"
                  >
                    <option value={0.01}>0.01 (IC 99%)</option>
                    <option value={0.05}>0.05 (IC 95%)</option>
                    <option value={0.1}>0.10 (IC 90%)</option>
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-slate-400">Simulações MC</label>
                  <select
                    value={nSims}
                    onChange={(e) => setNSims(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 font-mono text-slate-100"
                  >
                    <option value={500}>500</option>
                    <option value={1000}>1.000</option>
                    <option value={2500}>2.500</option>
                  </select>
                </div>
              </div>

              <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-2">
                <span className="text-slate-300">Executar Teste Placebo no Tempo</span>
                <input
                  type="checkbox"
                  checked={runPlacebo}
                  onChange={(e) => setRunPlacebo(e.target.checked)}
                  className="h-4 w-4 accent-emerald-400"
                />
              </label>

              <button
                type="button"
                disabled={!dataset || isAnalyzing}
                onClick={() => executeAnalysis()}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-4 py-2.5 text-xs font-extrabold text-slate-950 shadow-lg shadow-emerald-500/20 transition hover:brightness-110 disabled:opacity-45"
              >
                <Play className="h-4 w-4" />
                {isAnalyzing ? "Estimando BSTS..." : "Executar Análise Causal"}
              </button>
            </div>
          </div>
        </aside>

        {/* Workspace Principal */}
        <main className="space-y-4 lg:col-span-9">
          {errorMsg && (
            <div className="flex items-start gap-3 rounded-2xl border border-rose-500/40 bg-rose-950/35 px-4 py-3 text-sm text-rose-200">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              <span className="flex-1">{errorMsg}</span>
              <button
                type="button"
                onClick={bootstrap}
                className="rounded-lg bg-rose-500/20 px-2.5 py-1 text-xs font-semibold text-rose-200 hover:bg-rose-500/30"
              >
                Recarregar
              </button>
            </div>
          )}

          {activeTab === "data" && (
            <DataUploadPanel
              demos={demos}
              dataset={dataset}
              isLoadingData={isLoadingData}
              isAnalyzing={isAnalyzing}
              onUploadFile={handleUploadFile}
              onSelectDemo={handleSelectDemo}
              onRunAnalysis={() => executeAnalysis()}
            />
          )}

          {analysis && activeTab === "impact" && <CausalChartsPanel analysis={analysis} />}
          {analysis && activeTab === "audit" && <AuditPlaceboPanel analysis={analysis} />}
          {analysis && activeTab === "table" && <ImpactTableExportPanel analysis={analysis} />}
          {analysis && dataset && activeTab === "explore" && (
            <ExploratoryPanel dataset={dataset} analysis={analysis} />
          )}
          {activeTab === "guide" && <MethodologyGuide />}
        </main>
      </div>
    </div>
  );
}
