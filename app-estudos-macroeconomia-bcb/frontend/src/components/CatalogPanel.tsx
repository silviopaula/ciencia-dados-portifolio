import React, { useMemo, useState } from "react";
import { BookOpen, Database, Filter, Search, Sparkles } from "lucide-react";
import type { StudyMeta, ThemeMeta } from "../types";

interface CatalogPanelProps {
  studies: StudyMeta[];
  themes: ThemeMeta[];
  onSelectStudy: (code: string) => void;
}

const THEME_BADGE: Record<string, string> = {
  fiscal: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  politica_monetaria: "border-sky-500/30 bg-sky-500/10 text-sky-300",
  inflacao: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  atividade_pib: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  trabalho: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
  cambio_externo: "border-violet-500/30 bg-violet-500/10 text-violet-300",
};

export const CatalogPanel: React.FC<CatalogPanelProps> = ({ studies, themes, onSelectStudy }) => {
  const [query, setQuery] = useState("");
  const [selectedTheme, setSelectedTheme] = useState<string>("all");
  const [onlySpecialized, setOnlySpecialized] = useState(false);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return studies.filter((s) => {
      if (selectedTheme !== "all" && s.theme !== selectedTheme) return false;
      if (onlySpecialized && !s.specialized) return false;
      if (!q) return true;
      return (
        s.code.toLowerCase().includes(q) ||
        s.title.toLowerCase().includes(q) ||
        s.subtitle.toLowerCase().includes(q) ||
        s.methodology_label.toLowerCase().includes(q) ||
        s.theme_label.toLowerCase().includes(q)
      );
    });
  }, [studies, query, selectedTheme, onlySpecialized]);

  return (
    <div className="space-y-6">
      {/* Filter Header */}
      <section className="rounded-3xl border border-slate-800/90 bg-slate-900/65 p-5 shadow-xl md:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-300">
              <BookOpen className="h-3.5 w-3.5" />
              Biblioteca Autoral de Macroeconomia Brasileira (M01–M60)
            </div>
            <h2 className="mt-2 text-2xl font-extrabold text-white">Catálogo de {studies.length} Estudos Econométricos</h2>
            <p className="mt-1 text-sm text-slate-400">
              Cada estudo possui fundamentação econômica inédita, guia de interpretação, base própria em DuckDB e
              integração opcional com séries ao vivo do Banco Central.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative min-w-[260px] flex-1 sm:w-80 sm:flex-none">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Buscar por tema, modelo (VAR, Taylor, SARIMA, HP)..."
                className="w-full rounded-xl border border-slate-700/80 bg-slate-950 py-2.5 pl-9 pr-3 text-sm text-white placeholder-slate-500 outline-none transition focus:border-emerald-500/60"
              />
            </div>

            <button
              onClick={() => setOnlySpecialized((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-2.5 text-xs font-semibold transition ${
                onlySpecialized
                  ? "border-amber-400/60 bg-amber-500/20 text-amber-200"
                  : "border-slate-700/80 bg-slate-950 text-slate-300 hover:border-slate-600"
              }`}
            >
              <Sparkles className="h-3.5 w-3.5" />
              Modelos Dedicados (6)
            </button>
          </div>
        </div>

        {/* Theme Pills */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="mr-1 inline-flex items-center gap-1 text-xs font-medium text-slate-400">
            <Filter className="h-3.5 w-3.5" /> Pilares:
          </span>
          <button
            onClick={() => setSelectedTheme("all")}
            className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
              selectedTheme === "all"
                ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
                : "border-slate-800 bg-slate-950/70 text-slate-400 hover:text-slate-200"
            }`}
          >
            Todos ({studies.length})
          </button>
          {themes.map((t) => (
            <button
              key={t.key}
              onClick={() => setSelectedTheme(t.key)}
              className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                selectedTheme === t.key
                  ? "border-emerald-400/60 bg-emerald-500/20 text-emerald-200"
                  : "border-slate-800 bg-slate-950/70 text-slate-400 hover:text-slate-200"
              }`}
            >
              {t.label} ({t.count})
            </button>
          ))}
        </div>
      </section>

      {/* Results Count */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>
          Exibindo <strong className="text-white">{filtered.length}</strong> de{" "}
          <strong className="text-white">{studies.length}</strong> estudos macroeconométricos
        </span>
        {(query || selectedTheme !== "all" || onlySpecialized) && (
          <button
            onClick={() => {
              setQuery("");
              setSelectedTheme("all");
              setOnlySpecialized(false);
            }}
            className="font-semibold text-emerald-300 hover:text-emerald-200"
          >
            Limpar filtros
          </button>
        )}
      </div>

      {/* Study Cards Grid */}
      {filtered.length === 0 ? (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-10 text-center text-sm text-slate-400">
          Nenhum estudo encontrado para o filtro atual. Tente outro termo ou limpe os filtros.
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((study) => {
            const badgeCls = THEME_BADGE[study.theme] || "border-slate-700 bg-slate-800 text-slate-300";
            return (
              <button
                key={study.code}
                onClick={() => onSelectStudy(study.code)}
                className="group flex flex-col justify-between rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5 text-left transition duration-200 hover:-translate-y-0.5 hover:border-emerald-500/50 hover:bg-slate-900/95 hover:shadow-xl hover:shadow-emerald-950/30"
              >
                <div>
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="rounded-lg bg-slate-800 px-2.5 py-0.5 font-mono text-xs font-bold text-white group-hover:bg-emerald-500/20 group-hover:text-emerald-300">
                        {study.code}
                      </span>
                      <span className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${badgeCls}`}>
                        {study.theme_label}
                      </span>
                    </div>
                    {study.specialized && (
                      <span className="inline-flex items-center gap-1 rounded-full border border-amber-400/40 bg-amber-500/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-300">
                        <Sparkles className="h-2.5 w-2.5" /> Dedicado
                      </span>
                    )}
                  </div>

                  <h3 className="text-base font-bold leading-snug text-white transition group-hover:text-emerald-200">
                    {study.title}
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{study.subtitle}</p>
                </div>

                <div className="mt-4 space-y-2.5 border-t border-slate-800/80 pt-3">
                  <div className="rounded-lg bg-slate-950/80 px-2.5 py-1.5 text-[11px] font-medium text-slate-300">
                    {study.methodology_label}
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span className="inline-flex items-center gap-1">
                      <Database className="h-3 w-3 text-slate-500" />
                      {study.dataset_keys.length > 0
                        ? `${study.dataset_keys.length} base(s) DuckDB • ${study.bcb_series.length} BCB`
                        : `${study.bcb_series.length} séries oficiais BCB (DuckDB)`}
                    </span>
                    <span className="font-semibold text-emerald-300 group-hover:translate-x-0.5 transition">
                      Executar no Estúdio →
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};
