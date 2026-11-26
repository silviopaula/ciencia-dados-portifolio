import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Cpu,
  FlaskConical,
  RefreshCw,
} from "lucide-react";
import { fetchConjuntura, fetchHealth, fetchStudies, runStudy, syncConjunturaLive } from "./api";
import { ArchitecturePanel } from "./components/ArchitecturePanel";
import { CatalogPanel } from "./components/CatalogPanel";
import { ConjunturaPanel } from "./components/ConjunturaPanel";
import { StudyStudioPanel } from "./components/StudyStudioPanel";
import type {
  ConjunturaResponse,
  HealthResponse,
  StudiesListResponse,
  StudyRunResult,
} from "./types";

type ActiveTab = "conjuntura" | "catalog" | "studio" | "architecture";

const EXPECTED_APP_NAME = "Observatório Macro Brasil API";

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("conjuntura");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  const [conjuntura, setConjuntura] = useState<ConjunturaResponse | null>(null);
  const [catalog, setCatalog] = useState<StudiesListResponse | null>(null);
  const [loadingBoot, setLoadingBoot] = useState(true);
  const [syncingLive, setSyncingLive] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  const [selectedCode, setSelectedCode] = useState<string>("M01");
  const [studyResult, setStudyResult] = useState<StudyRunResult | null>(null);
  const [runningStudy, setRunningStudy] = useState(false);
  const [studyError, setStudyError] = useState<string | null>(null);

  const loadInitialData = useCallback(async () => {
    setLoadingBoot(true);
    setBackendError(null);
    try {
      const h = await fetchHealth();
      if (h.app_name !== EXPECTED_APP_NAME) {
        throw new Error(
          `Serviço inesperado na porta 8004 ("${h.app_name || "desconhecido"}"). Encerre processos antigos e inicie a API do Observatório Macro Brasil.`,
        );
      }
      setHealth(h);
      const [conj, cat] = await Promise.all([fetchConjuntura(), fetchStudies()]);
      setConjuntura(conj);
      setCatalog(cat);
    } catch (err) {
      setBackendError(
        err instanceof Error
          ? err.message
          : "Backend FastAPI indisponível na porta 8004. Inicie o servidor Python.",
      );
    } finally {
      setLoadingBoot(false);
    }
  }, []);

  useEffect(() => {
    void loadInitialData();
  }, [loadInitialData]);

  const executeSelectedStudy = useCallback(async (code: string, params: Record<string, unknown> = {}) => {
    setRunningStudy(true);
    setStudyError(null);
    try {
      const resp = await runStudy(code, params);
      setStudyResult(resp.result);
    } catch (err) {
      setStudyError(err instanceof Error ? err.message : "Erro ao executar estudo econométrico.");
    } finally {
      setRunningStudy(false);
    }
  }, []);

  const handleSelectStudy = useCallback(
    (code: string) => {
      setSelectedCode(code);
      setStudyResult(null);
      setActiveTab("studio");
      void executeSelectedStudy(code, {});
    },
    [executeSelectedStudy],
  );

  const handleSyncLive = useCallback(async () => {
    setSyncingLive(true);
    setSyncMessage(null);
    try {
      const updated = await syncConjunturaLive();
      setConjuntura(updated);
      const liveCount = updated.sync_details?.filter((d) => d.live).length ?? 0;
      const total = updated.sync_details?.length ?? 0;
      setSyncMessage(
        liveCount > 0
          ? `Sincronização concluída: ${liveCount}/${total} séries atualizadas ao vivo via Banco Central e salvas no DuckDB + Parquet.`
          : "API do Banco Central indisponível no momento — exibindo a versão mais recente em cache local DuckDB/Parquet.",
      );
    } catch (err) {
      setSyncMessage(err instanceof Error ? err.message : "Falha ao sincronizar com a API do BCB.");
    } finally {
      setSyncingLive(false);
    }
  }, []);

  const selectedStudy = useMemo(
    () => catalog?.items.find((s) => s.code === selectedCode) ?? catalog?.items[0] ?? null,
    [catalog, selectedCode],
  );

  const specializedStudies = useMemo(
    () => (catalog?.items ?? []).filter((s) => s.specialized),
    [catalog],
  );

  return (
    <div className="flex min-h-screen flex-col">
      {/* Top Navbar */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#060b16]/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <button
            onClick={() => setActiveTab("conjuntura")}
            className="flex items-center gap-3 text-left"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-sky-500 text-slate-950 shadow-lg shadow-emerald-500/20">
              <BarChart3 className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-extrabold tracking-tight text-white">
                  Observatório Macro Brasil
                </h1>
                <span className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] font-bold text-emerald-300">
                  STUDIO
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Macroeconomia Aplicada, Conjuntura BCB &amp; Econometria em Python
              </p>
            </div>
          </button>

          {/* Nav Tabs */}
          <nav className="flex flex-wrap items-center gap-1 rounded-2xl border border-slate-800/90 bg-slate-900/70 p-1">
            {[
              { id: "conjuntura" as const, label: "Conjuntura Brasil", icon: Activity },
              {
                id: "catalog" as const,
                label: `Catálogo (${catalog?.total ?? 60})`,
                icon: BookOpen,
              },
              {
                id: "studio" as const,
                label: selectedStudy ? `Estúdio (${selectedStudy.code})` : "Estúdio",
                icon: FlaskConical,
              },
              { id: "architecture" as const, label: "Arquitetura & Dados", icon: Cpu },
            ].map((tab) => {
              const Icon = tab.icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id);
                    if (tab.id === "studio" && !studyResult && selectedStudy && !runningStudy) {
                      void executeSelectedStudy(selectedStudy.code, {});
                    }
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
                    active
                      ? "bg-gradient-to-r from-emerald-500/20 to-sky-500/20 text-white shadow-inner ring-1 ring-emerald-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Backend Health Pill */}
          <div className="flex items-center gap-2">
            {health ? (
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-[11px] font-medium text-emerald-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
                API Python Online (:8004)
              </div>
            ) : (
              <button
                onClick={() => void loadInitialData()}
                className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/40 bg-rose-500/15 px-3 py-1 text-[11px] font-medium text-rose-200"
              >
                <RefreshCw className="h-3 w-3" />
                Reconectar API (:8004)
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        {backendError && (
          <div className="mb-6 flex items-start gap-3 rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-400" />
            <div>
              <div className="font-bold">Backend FastAPI indisponível</div>
              <p className="mt-0.5 text-xs text-rose-200/90">{backendError}</p>
            </div>
          </div>
        )}

        {activeTab === "conjuntura" && (
          <ConjunturaPanel
            data={conjuntura}
            loading={loadingBoot}
            syncing={syncingLive}
            syncMessage={syncMessage}
            specializedStudies={specializedStudies}
            onSyncLive={() => void handleSyncLive()}
            onSelectStudy={handleSelectStudy}
            onGoCatalog={() => setActiveTab("catalog")}
          />
        )}

        {activeTab === "catalog" && catalog && (
          <CatalogPanel
            studies={catalog.items}
            themes={catalog.themes}
            onSelectStudy={handleSelectStudy}
          />
        )}

        {activeTab === "studio" && selectedStudy && catalog && (
          <StudyStudioPanel
            study={selectedStudy}
            allStudies={catalog.items}
            result={studyResult}
            loading={runningStudy}
            error={studyError}
            onBack={() => setActiveTab("catalog")}
            onSelectStudy={handleSelectStudy}
            onExecute={(code, params) => void executeSelectedStudy(code, params)}
          />
        )}

        {activeTab === "architecture" && (
          <ArchitecturePanel health={health} dbStats={conjuntura?.db_stats ?? catalog?.db_stats ?? null} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/70 bg-slate-950/60 py-4">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-4 text-xs text-slate-500 sm:px-6">
          <span>
            Observatório Macro Brasil Studio • Portfólio de Ciência de Dados &amp; Econometria (Dr. Silvio da Rosa Paula)
          </span>
          <span className="font-mono">FastAPI (:8004) • DuckDB + Parquet • React + TypeScript (:5177)</span>
        </div>
      </footer>
    </div>
  );
}
