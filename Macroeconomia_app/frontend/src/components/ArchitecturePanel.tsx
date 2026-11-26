import React from "react";
import { Cpu, Database, Globe, Layers, Server, ShieldCheck } from "lucide-react";
import type { DbStats, HealthResponse } from "../types";

interface ArchitecturePanelProps {
  health: HealthResponse | null;
  dbStats: DbStats | null;
}

const ENDPOINTS = [
  { method: "GET", path: "/api/health", desc: "Identidade da API, contagem de estudos, tabelas DuckDB e séries BCB." },
  { method: "GET", path: "/api/conjuntura", desc: "Consolida os 6 KPIs e 3 séries executivas do Painel de Conjuntura Brasil." },
  { method: "POST", path: "/api/conjuntura/sync", desc: "Atualiza ao vivo séries centrais nas APIs SGS e Olinda Focus do BCB." },
  { method: "GET", path: "/api/studies", desc: "Lista os 60 estudos macroeconométricos autorais agrupados em 6 pilares (10 por pilar)." },
  { method: "POST", path: "/api/studies/{code}/run", desc: "Executa o motor econométrico Python com parâmetros interativos customizados." },
  { method: "GET", path: "/api/studies/{code}/export-csv", desc: "Exporta a base tratada/projetada do estudo selecionado em formato CSV." },
];

export const ArchitecturePanel: React.FC<ArchitecturePanelProps> = ({ health, dbStats }) => {
  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800/90 bg-gradient-to-br from-slate-900/90 to-slate-950 p-6 shadow-xl">
        <div className="inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-3 py-1 text-xs font-medium text-violet-300">
          <Layers className="h-3.5 w-3.5" />
          Engenharia de Dados &amp; Econometria Full-Stack
        </div>
        <h2 className="mt-2.5 text-2xl font-extrabold text-white">Arquitetura do Observatório Macro Brasil Studio</h2>
        <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-slate-300">
          Plataforma desacoplada construída 100% em Python + React/TypeScript com armazenamento colunar DuckDB +
          Parquet, conectores oficiais para o Banco Central do Brasil (SGS e Olinda Focus) e catálogo integralmente
          autoral em português.
        </p>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/75 p-4">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Database className="h-4 w-4 text-emerald-400" /> Banco DuckDB
            </div>
            <div className="mt-2 font-mono text-2xl font-extrabold text-white">
              {dbStats ? `${dbStats.size_mb.toFixed(2)} MB` : "—"}
            </div>
            <div className="mt-1 text-[11px] text-slate-400">
              {dbStats ? `${dbStats.tables_count} tabelas históricas + ${dbStats.series_count} séries BCB` : "—"}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/75 p-4">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Server className="h-4 w-4 text-sky-400" /> Registros Analíticos
            </div>
            <div className="mt-2 font-mono text-2xl font-extrabold text-white">
              {dbStats ? dbStats.total_rows.toLocaleString("pt-BR") : "—"}
            </div>
            <div className="mt-1 text-[11px] text-slate-400">Observações reais prontas em milissegundos</div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/75 p-4">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Cpu className="h-4 w-4 text-amber-400" /> Motor Econométrico
            </div>
            <div className="mt-2 font-mono text-2xl font-extrabold text-white">
              {health?.studies_count ?? 60} Estudos
            </div>
            <div className="mt-1 text-[11px] text-slate-400">Statsmodels • VAR, ECM, HP, SARIMA, HAC, EWMA</div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/75 p-4">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <ShieldCheck className="h-4 w-4 text-violet-400" /> Portas Dedicadas
            </div>
            <div className="mt-2 font-mono text-2xl font-extrabold text-white">8004 / 5177</div>
            <div className="mt-1 text-[11px] text-slate-400">
              Última sync BCB: {dbStats?.last_updated ?? "Cache local"}
            </div>
          </div>
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-bold text-white">
            <Globe className="h-4 w-4 text-emerald-400" /> Endpoints REST (FastAPI — Porta 8004)
          </h3>
          <div className="space-y-2.5">
            {ENDPOINTS.map((ep) => (
              <div
                key={ep.path}
                className="rounded-xl border border-slate-800/80 bg-slate-950/70 p-3"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold ${
                      ep.method === "GET"
                        ? "bg-sky-500/20 text-sky-300"
                        : "bg-emerald-500/20 text-emerald-300"
                    }`}
                  >
                    {ep.method}
                  </span>
                  <code className="font-mono text-xs text-white">{ep.path}</code>
                </div>
                <p className="mt-1 text-xs text-slate-400">{ep.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5 space-y-4">
          <h3 className="text-sm font-bold text-white">Camadas da Solução</h3>
          {[
            {
              title: "1. Ingestão & Cache Analítico (DuckDB + Parquet)",
              body: "As séries oficiais do Banco Central (SGS e Olinda Focus) são persistidas em observatorio_macro.duckdb e espelhadas em bcb_series.parquet, garantindo inicialização instantânea offline e atualização sob demanda.",
            },
            {
              title: "2. Motor Econométrico Reprodutível (Python + Statsmodels)",
              body: "Execução nativa em Python com validação de raiz unitária (ADF), vetores autorregressivos (VAR + IRF + Causalidade de Granger), cointegração de Engle-Granger + ECM, Filtro Hodrick-Prescott, projeções SARIMA, volatilidade EWMA e regressões MQO com erros robustos Newey-West HAC.",
            },
            {
              title: "3. Interface Executiva & Didática (React + TypeScript + Recharts)",
              body: "SPA moderna construída com Vite, Tailwind CSS e Recharts, apresentando intuição econômica, método e guia de interpretação originais lado a lado com os gráficos e tabelas estatísticas.",
            },
          ].map((item) => (
            <div key={item.title} className="rounded-xl border border-slate-800/80 bg-slate-950/70 p-4">
              <div className="text-xs font-bold text-emerald-300">{item.title}</div>
              <p className="mt-1 text-xs leading-relaxed text-slate-300">{item.body}</p>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
};
