import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart2,
  BookOpen,
  Compass,
  Layers,
  Loader2,
  Table2,
} from "lucide-react";
import {
  downloadExcelReport,
  fetchDemos,
  inspectUploadedFile,
  runDemoAnalysis,
  runUploadAnalysis,
} from "./api";
import { DataTableTab } from "./components/DataTableTab";
import { DecompForecastTab } from "./components/DecompForecastTab";
import { EconometricGuideTab } from "./components/EconometricGuideTab";
import { SidebarControls } from "./components/SidebarControls";
import { StatTestsTab } from "./components/StatTestsTab";
import { TransformsTab } from "./components/TransformsTab";
import { VisualsDistTab } from "./components/VisualsDistTab";
import type {
  AnalysisConfig,
  AnalysisPayload,
  DemoDatasetMeta,
  TransformType,
  UploadInspectResponse,
} from "./types";

type TabId = "tests" | "transforms" | "visuals" | "decomp" | "data" | "guide";

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "tests", label: "Testes Estatísticos", icon: <Activity className="w-4 h-4" /> },
  { id: "transforms", label: "Lab. Transformações", icon: <Layers className="w-4 h-4" /> },
  { id: "visuals", label: "Visualizações & Outliers", icon: <BarChart2 className="w-4 h-4" /> },
  { id: "decomp", label: "STL & Previsão SARIMA", icon: <Compass className="w-4 h-4" /> },
  { id: "data", label: "Dados Processados", icon: <Table2 className="w-4 h-4" /> },
  { id: "guide", label: "Guia Econométrico", icon: <BookOpen className="w-4 h-4" /> },
];

const INITIAL_CONFIG: AnalysisConfig = {
  sourceMode: "demo",
  datasetId: "varejo_sazonal_i1",
  transform: "level",
  alpha: 0.05,
  regression: "ct",
  maxLags: 12,
  seasonalPeriod: 12,
  autoArima: true,
  orderP: 1,
  orderD: 1,
  orderQ: 1,
  seasonalP: 0,
  seasonalD: 0,
  seasonalQ: 0,
  forecastHorizon: 12,
  sheetName: "",
  dateCol: "",
  valueCol: "",
  dateFormat: "auto",
  aggregation: "none",
};

export default function App() {
  const [demos, setDemos] = useState<DemoDatasetMeta[]>([]);
  const [config, setConfig] = useState<AnalysisConfig>(INITIAL_CONFIG);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadMeta, setUploadMeta] = useState<UploadInspectResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("tests");
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeAnalysis = useCallback(
    async (cfgToRun: AnalysisConfig, fileToRun: File | null) => {
      setLoading(true);
      setError(null);
      try {
        const res =
          cfgToRun.sourceMode === "upload" && fileToRun
            ? await runUploadAnalysis(fileToRun, cfgToRun)
            : await runDemoAnalysis(cfgToRun);
        setAnalysis(res);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Erro inesperado ao executar análise.");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const list = await fetchDemos();
        if (!active) return;
        setDemos(list);
        await executeAnalysis(INITIAL_CONFIG, null);
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : "Não foi possível conectar ao backend FastAPI na porta 8005."
          );
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [executeAnalysis]);

  const handleFileSelected = async (file: File) => {
    setUploadedFile(file);
    setError(null);
    try {
      const meta = await inspectUploadedFile(file);
      setUploadMeta(meta);
      const nextCfg: AnalysisConfig = {
        ...config,
        sourceMode: "upload",
        sheetName: meta.selected_sheet ?? "",
        dateCol: meta.suggested_date_col,
        valueCol: meta.suggested_value_col,
        seasonalPeriod: meta.suggested_period,
        transform: "level",
      };
      setConfig(nextCfg);
      await executeAnalysis(nextCfg, file);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro ao inspecionar arquivo.");
    }
  };

  const handleSheetChange = async (sheet: string) => {
    if (!uploadedFile) return;
    setError(null);
    try {
      const meta = await inspectUploadedFile(uploadedFile, sheet);
      setUploadMeta(meta);
      setConfig((c) => ({
        ...c,
        sheetName: sheet,
        dateCol: meta.suggested_date_col,
        valueCol: meta.suggested_value_col,
        seasonalPeriod: meta.suggested_period,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro ao trocar aba do Excel.");
    }
  };

  const handleApplyTransform = (nextTransform: TransformType) => {
    const nextCfg: AnalysisConfig = { ...config, transform: nextTransform };
    setConfig(nextCfg);
    void executeAnalysis(nextCfg, uploadedFile);
  };

  const handleExportExcel = async () => {
    setExporting(true);
    setError(null);
    try {
      await downloadExcelReport(config, uploadedFile);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro ao exportar arquivo Excel.");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-slate-50">
      <SidebarControls
        demos={demos}
        config={config}
        setConfig={setConfig}
        uploadedFile={uploadedFile}
        uploadMeta={uploadMeta}
        onFileSelected={handleFileSelected}
        onSheetChange={handleSheetChange}
        onRun={() => void executeAnalysis(config, uploadedFile)}
        onExportExcel={() => void handleExportExcel()}
        loading={loading}
        exporting={exporting}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 px-6 py-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white font-bold text-sm">
                TS
              </span>
              <div>
                <h1 className="text-base sm:text-lg font-extrabold text-slate-900 tracking-tight">
                  TimeSeries Diagnostics Studio
                </h1>
                <p className="text-xs text-slate-500">
                  Laboratório Econométrico: Raiz Unitária, Quebra Estrutural, Autocorrelação, ARCH-LM, Normalidade, STL &amp; SARIMA
                </p>
              </div>
            </div>
          </div>

          {analysis && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium">
                {analysis.meta.date_start} → {analysis.meta.date_end} ({analysis.meta.n_analyzed} obs.)
              </span>
              <span className="px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-800 font-semibold border border-indigo-200">
                {analysis.meta.transform_label}
              </span>
            </div>
          )}
        </header>

        {/* Tabs */}
        <nav className="bg-white border-b border-slate-200 px-4 sm:px-6 overflow-x-auto">
          <div className="flex items-center gap-1 min-w-max py-2">
            {TABS.map((t) => {
              const active = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setActiveTab(t.id)}
                  className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition ${
                    active
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              );
            })}
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-4 sm:p-6 max-w-[1520px] w-full mx-auto space-y-5">
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-900 flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {loading && !analysis ? (
            <div className="bg-white rounded-xl border border-slate-200 p-16 flex flex-col items-center justify-center text-slate-500 gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
              <p className="text-sm font-medium">
                Executando bateria de 10 testes estatísticos, STL e SARIMA...
              </p>
            </div>
          ) : analysis ? (
            <div className={loading ? "opacity-60 pointer-events-none transition" : "transition"}>
              {activeTab === "tests" && (
                <StatTestsTab data={analysis} onApplyTransform={handleApplyTransform} />
              )}
              {activeTab === "transforms" && (
                <TransformsTab data={analysis} onApplyTransform={handleApplyTransform} />
              )}
              {activeTab === "visuals" && <VisualsDistTab data={analysis} />}
              {activeTab === "decomp" && <DecompForecastTab data={analysis} />}
              {activeTab === "data" && <DataTableTab data={analysis} />}
              {activeTab === "guide" && <EconometricGuideTab />}
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}
