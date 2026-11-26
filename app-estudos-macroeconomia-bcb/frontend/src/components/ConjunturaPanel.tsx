import React from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Landmark,
  RefreshCw,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ConjunturaResponse, StudyMeta } from "../types";

interface ConjunturaPanelProps {
  data: ConjunturaResponse | null;
  loading: boolean;
  syncing: boolean;
  syncMessage: string | null;
  specializedStudies: StudyMeta[];
  onSyncLive: () => void;
  onSelectStudy: (code: string) => void;
  onGoCatalog: () => void;
}

interface TooltipEntry {
  name?: string;
  value?: unknown;
  color?: string;
}

function formatTooltipVal(val: unknown): string {
  if (Array.isArray(val) && val.length === 2) {
    return `${Number(val[0]).toFixed(2)} – ${Number(val[1]).toFixed(2)}`;
  }
  if (typeof val === "number") return val.toFixed(2);
  return String(val ?? "—");
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-xl border border-slate-700/80 bg-slate-950/95 px-3.5 py-2.5 text-xs shadow-2xl backdrop-blur">
      <div className="mb-1.5 font-mono text-[11px] font-semibold text-slate-300">{label}</div>
      <div className="space-y-1">
        {payload.map((entry, idx) => (
          <div key={idx} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: entry.color || "#34d399" }}
              />
              {entry.name}
            </span>
            <span className="font-mono font-semibold text-white">
              {formatTooltipVal(entry.value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

const KPI_ACCENTS: Record<string, string> = {
  selic: "from-sky-500/20 to-sky-500/5 border-sky-500/25 text-sky-300",
  ipca12: "from-amber-500/20 to-amber-500/5 border-amber-500/25 text-amber-300",
  dbgg: "from-rose-500/20 to-rose-500/5 border-rose-500/25 text-rose-300",
  cambio: "from-violet-500/20 to-violet-500/5 border-violet-500/25 text-violet-300",
  ibcbr: "from-emerald-500/20 to-emerald-500/5 border-emerald-500/25 text-emerald-300",
  desocupacao: "from-cyan-500/20 to-cyan-500/5 border-cyan-500/25 text-cyan-300",
};

export const ConjunturaPanel: React.FC<ConjunturaPanelProps> = ({
  data,
  loading,
  syncing,
  syncMessage,
  specializedStudies,
  onSyncLive,
  onSelectStudy,
  onGoCatalog,
}) => {
  if (loading && !data) {
    return (
      <div className="grid gap-4 md:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="h-32 animate-pulse rounded-2xl border border-slate-800 bg-slate-900/50"
          />
        ))}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-6 text-sm text-rose-200">
        Não foi possível carregar o Painel de Conjuntura Brasil. Verifique a conexão com o backend Python na porta 8004.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-800/90 bg-gradient-to-br from-slate-900/90 via-[#0d1629] to-slate-950 p-6 shadow-2xl md:p-7">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-3xl space-y-2.5">
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
              <Landmark className="h-3.5 w-3.5" />
              Séries Oficiais do Banco Central do Brasil (SGS + Olinda Focus) • Cache Analítico DuckDB/Parquet
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight text-white md:text-3xl">
              Painel Executivo de Conjuntura Macroeconômica
            </h2>
            <p className="text-sm leading-relaxed text-slate-300">
              Acompanhe em tempo real os seis pilares da economia brasileira — juros, inflação em 12 meses, dívida
              bruta, câmbio, atividade econômica e mercado de trabalho — e aprofunde cada fenômeno nos{" "}
              <span className="font-semibold text-emerald-300">60 estudos econométricos autorais</span> executados em
              Python.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={onSyncLive}
              disabled={syncing}
              className="inline-flex items-center gap-2 rounded-xl border border-emerald-400/40 bg-emerald-500/15 px-4 py-2.5 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-500/25 disabled:opacity-60"
            >
              <RefreshCw className={`h-4 w-4 ${syncing ? "animate-spin" : ""}`} />
              {syncing ? "Sincronizando API BCB..." : "Sincronizar BCB ao Vivo"}
            </button>
            <button
              onClick={onGoCatalog}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-sky-500 px-4 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/20 transition hover:brightness-110"
            >
              <Sparkles className="h-4 w-4" />
              Explorar 60 Estudos
            </button>
          </div>
        </div>

        {syncMessage && (
          <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2.5 text-xs text-emerald-200">
            {syncMessage}
          </div>
        )}
      </section>

      {/* 6 KPI Cards */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {data.kpis.map((kpi) => {
          const accent = KPI_ACCENTS[kpi.key] || "from-slate-800 to-slate-900 border-slate-700 text-slate-200";
          const delta = kpi.delta_12m ?? 0;
          const up = delta > 0;
          return (
            <div
              key={kpi.key}
              className={`rounded-2xl border bg-gradient-to-b p-4 transition hover:border-slate-600 ${accent}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-300">
                  {kpi.label}
                </span>
              </div>
              <div className="mt-2.5 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-extrabold tracking-tight text-white">
                  {kpi.value !== null ? kpi.value.toFixed(kpi.key === "ibcbr" || kpi.key === "desocupacao" ? 1 : 2) : "—"}
                </span>
                <span className="text-xs font-medium text-slate-400">{kpi.unit}</span>
              </div>
              <div className="mt-2.5 flex items-center justify-between text-[11px]">
                <span
                  className={`inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 font-mono font-medium ${
                    up
                      ? "bg-amber-500/15 text-amber-300"
                      : "bg-emerald-500/15 text-emerald-300"
                  }`}
                >
                  {up ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                  {delta >= 0 ? "+" : ""}
                  {delta.toFixed(kpi.key === "ibcbr" || kpi.key === "desocupacao" ? 1 : 2)} (12m)
                </span>
                <span className="font-mono text-slate-400">{kpi.last_date ?? "—"}</span>
              </div>
            </div>
          );
        })}
      </section>

      {/* 3 Executive Charts */}
      <section className="grid gap-5 lg:grid-cols-2">
        {/* Chart 1: Selic vs IPCA vs Meta (full width) */}
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5 lg:col-span-2">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="flex items-center gap-2 text-base font-bold text-white">
                <TrendingUp className="h-4 w-4 text-sky-400" />
                Política Monetária vs. Inflação: Selic, IPCA 12m e Meta do CMN
              </h3>
              <p className="text-xs text-slate-400">
                Comparação histórica mensal (últimos 9 anos) — avalie a distância entre inflação realizada, meta oficial
                e taxa básica de juros.
              </p>
            </div>
            <span className="rounded-lg border border-slate-700/80 bg-slate-950 px-2.5 py-1 font-mono text-[11px] text-slate-400">
              BCB SGS 4189 • 13522 • 13521
            </span>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.charts.juros_inflacao} margin={{ top: 8, right: 16, left: -12, bottom: 0 }}>
                <defs>
                  <linearGradient id="ipcaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={28} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} unit="%" />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
                <Area
                  type="monotone"
                  dataKey="ipca12"
                  name="IPCA 12m (%)"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  fill="url(#ipcaGrad)"
                />
                <Line
                  type="monotone"
                  dataKey="selic"
                  name="Selic (% a.a.)"
                  stroke="#38bdf8"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="meta"
                  name="Meta CMN (%)"
                  stroke="#34d399"
                  strokeWidth={1.8}
                  strokeDasharray="6 4"
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Fiscal vs Câmbio */}
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <div className="mb-4 flex items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-bold text-white">Dívida Bruta (DBGG) &amp; Taxa de Câmbio</h3>
              <p className="text-xs text-slate-400">Eixo esquerdo: DBGG (% PIB) • Eixo direito: Dólar (R$/US$)</p>
            </div>
            <span className="rounded-lg border border-slate-700/80 bg-slate-950 px-2 py-1 font-mono text-[10px] text-slate-400">
              13762 • 3698
            </span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.charts.fiscal_cambio} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={28} />
                <YAxis yAxisId="left" tick={{ fill: "#fb7185", fontSize: 11 }} domain={["auto", "auto"]} unit="%" />
                <YAxis yAxisId="right" orientation="right" tick={{ fill: "#a78bfa", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "6px" }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="dbgg"
                  name="DBGG (% PIB)"
                  stroke="#fb7185"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="cambio"
                  name="Câmbio (R$/US$)"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: IBC-Br vs Desocupação */}
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <div className="mb-4 flex items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-bold text-white">Atividade Econômica (IBC-Br) &amp; Desocupação</h3>
              <p className="text-xs text-slate-400">Dinâmica cíclica entre nível de atividade e taxa de desemprego PNAD</p>
            </div>
            <span className="rounded-lg border border-slate-700/80 bg-slate-950 px-2 py-1 font-mono text-[10px] text-slate-400">
              24364 • 24369
            </span>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.charts.atividade_trabalho} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={28} />
                <YAxis yAxisId="left" tick={{ fill: "#34d399", fontSize: 11 }} domain={["auto", "auto"]} />
                <YAxis yAxisId="right" orientation="right" tick={{ fill: "#22d3ee", fontSize: 11 }} domain={["auto", "auto"]} unit="%" />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "6px" }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="ibcbr"
                  name="IBC-Br (índice)"
                  stroke="#34d399"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="desoc"
                  name="Desocupação (%)"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Quick Launch to the 6 Dedicated Models */}
      <section className="rounded-2xl border border-slate-800/90 bg-slate-900/50 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="flex items-center gap-2 text-base font-bold text-white">
              <Activity className="h-4 w-4 text-emerald-400" />
              Acesso Rápido — 6 Laboratórios Econométricos Dedicados
            </h3>
            <p className="text-xs text-slate-400">
              Modelos interativos com controle de parâmetros em tempo real (VAR/Granger, ARIMA, Regra de Taylor HAC,
              Inércia Rolling AR(1), SARIMA e Curva de Phillips).
            </p>
          </div>
          <button
            onClick={onGoCatalog}
            className="text-xs font-semibold text-emerald-300 transition hover:text-emerald-200"
          >
            Ver todos os 60 estudos →
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {specializedStudies.map((s) => (
            <button
              key={s.code}
              onClick={() => onSelectStudy(s.code)}
              className="group flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-left transition hover:border-emerald-500/50 hover:bg-slate-900/90"
            >
              <div>
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="rounded-md bg-emerald-500/15 px-2 py-0.5 font-mono text-xs font-bold text-emerald-300">
                    {s.code}
                  </span>
                  <span className="text-[11px] font-medium text-slate-400">{s.theme_label}</span>
                </div>
                <h4 className="text-sm font-bold text-white group-hover:text-emerald-200">{s.title}</h4>
                <p className="mt-1 line-clamp-2 text-xs text-slate-400">{s.subtitle}</p>
              </div>
              <div className="mt-3 flex items-center justify-between border-t border-slate-800/80 pt-2.5 text-[11px] text-slate-400">
                <span className="truncate pr-2">{s.methodology_label}</span>
                <span className="shrink-0 font-semibold text-emerald-300">Abrir →</span>
              </div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
};
