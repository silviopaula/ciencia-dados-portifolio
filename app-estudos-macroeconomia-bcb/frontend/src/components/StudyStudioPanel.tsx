import React, { useEffect, useState } from "react";
import {
  ArrowLeft,
  BookMarked,
  CheckCircle2,
  Compass,
  Download,
  Lightbulb,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getExportCsvUrl } from "../api";
import type {
  AdfRow,
  CoefRow,
  CorrMatrix,
  StudyMeta,
  StudyRunResult,
  SummaryStatRow,
  ThematicModelResult,
} from "../types";

interface StudyStudioPanelProps {
  study: StudyMeta;
  allStudies: StudyMeta[];
  result: StudyRunResult | null;
  loading: boolean;
  error: string | null;
  onBack: () => void;
  onSelectStudy: (code: string) => void;
  onExecute: (code: string, params: Record<string, unknown>) => void;
}

const SERIES_COLORS = ["#34d399", "#38bdf8", "#f59e0b", "#a78bfa", "#fb7185", "#22d3ee"];

interface TEntry {
  name?: string;
  value?: unknown;
  color?: string;
}

function fmtVal(val: unknown): string {
  if (Array.isArray(val) && val.length === 2) {
    return `${Number(val[0]).toFixed(2)} a ${Number(val[1]).toFixed(2)}`;
  }
  if (typeof val === "number") return Number.isInteger(val) ? String(val) : val.toFixed(3);
  return String(val ?? "—");
}

function StudioTooltip({ active, payload, label }: { active?: boolean; payload?: TEntry[]; label?: string }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-xl border border-slate-700/80 bg-slate-950/95 px-3.5 py-2.5 text-xs shadow-2xl">
      <div className="mb-1.5 font-mono text-[11px] font-semibold text-slate-300">{label}</div>
      <div className="space-y-1">
        {payload.map((entry, idx) => (
          <div key={idx} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: entry.color || "#34d399" }} />
              {entry.name}
            </span>
            <span className="font-mono font-semibold text-white">{fmtVal(entry.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AdfTable({ rows }: { rows: AdfRow[] }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
      <h4 className="mb-3 text-sm font-bold text-white">Testes de Raiz Unitária — Augmented Dickey-Fuller (ADF)</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-[11px] uppercase tracking-wider text-slate-400">
              <th className="py-2 pr-3">Série</th>
              <th className="px-3 py-2 text-right">Estatística t</th>
              <th className="px-3 py-2 text-right">p-valor</th>
              <th className="px-3 py-2 text-right">Lags</th>
              <th className="px-3 py-2 text-right">Obs.</th>
              <th className="pl-3 py-2 text-right">Diagnóstico (5%)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-slate-800/30">
                <td className="py-2 pr-3 font-medium text-slate-200">{r.series}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-300">{r.statistic.toFixed(4)}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-300">{r.pvalue.toFixed(4)}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-400">{r.lags_used}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-400">{r.nobs}</td>
                <td className="pl-3 py-2 text-right">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                      r.stationary_5pct
                        ? "bg-emerald-500/15 text-emerald-300"
                        : "bg-amber-500/15 text-amber-300"
                    }`}
                  >
                    {r.stationary_5pct ? "Estacionária I(0)" : "Raiz Unitária I(1)"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CoefTable({ rows, title }: { rows: CoefRow[]; title: string }) {
  return (
    <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
      <h4 className="mb-3 text-sm font-bold text-white">{title}</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-[11px] uppercase tracking-wider text-slate-400">
              <th className="py-2 pr-3">Termo</th>
              <th className="px-3 py-2 text-right">Coeficiente</th>
              <th className="px-3 py-2 text-right">Erro-Padrão HAC</th>
              <th className="px-3 py-2 text-right">Estatística t</th>
              <th className="px-3 py-2 text-right">p-valor</th>
              <th className="pl-3 py-2 text-right">Sig. 5%</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-slate-800/30">
                <td className="py-2 pr-3 font-medium text-slate-200">{r.term}</td>
                <td className="px-3 py-2 text-right font-mono font-semibold text-white">{r.coef.toFixed(4)}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-400">{r.std_err.toFixed(4)}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-300">{r.t_stat.toFixed(3)}</td>
                <td className="px-3 py-2 text-right font-mono text-slate-300">{r.pvalue.toFixed(4)}</td>
                <td className="pl-3 py-2 text-right">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                      r.sig ? "bg-emerald-500/15 text-emerald-300" : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {r.sig ? "Significante ✓" : "Não sig."}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SummaryAndCorr({ stats, corr }: { stats: SummaryStatRow[]; corr: CorrMatrix }) {
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
        <h4 className="mb-3 text-sm font-bold text-white">Estatísticas Descritivas das Séries</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                <th className="py-2 pr-2">Variável</th>
                <th className="px-2 py-2 text-right">Média</th>
                <th className="px-2 py-2 text-right">Desvio</th>
                <th className="px-2 py-2 text-right">Mín</th>
                <th className="px-2 py-2 text-right">Mediana</th>
                <th className="pl-2 py-2 text-right">Máx</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {stats.map((s, i) => (
                <tr key={i}>
                  <td className="py-2 pr-2 font-medium text-slate-200">{s.column}</td>
                  <td className="px-2 py-2 text-right font-mono text-slate-300">{s.mean.toFixed(2)}</td>
                  <td className="px-2 py-2 text-right font-mono text-slate-400">{s.std.toFixed(2)}</td>
                  <td className="px-2 py-2 text-right font-mono text-slate-400">{s.min.toFixed(2)}</td>
                  <td className="px-2 py-2 text-right font-mono text-slate-300">{s.median.toFixed(2)}</td>
                  <td className="pl-2 py-2 text-right font-mono text-slate-300">{s.max.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
        <h4 className="mb-3 text-sm font-bold text-white">Matriz de Correlação de Pearson</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-center text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                <th className="py-2 text-left">Variável</th>
                {corr.cols.map((c) => (
                  <th key={c} className="px-2 py-2 font-mono">
                    {c.slice(0, 10)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {corr.cols.map((rowName, ri) => (
                <tr key={rowName}>
                  <td className="py-2 pr-2 text-left font-medium text-slate-200">{rowName.slice(0, 14)}</td>
                  {corr.rows[ri].map((val, ci) => {
                    const bg =
                      ri === ci
                        ? "bg-slate-800 text-slate-300"
                        : val >= 0.5
                          ? "bg-emerald-500/25 text-emerald-200"
                          : val <= -0.5
                            ? "bg-rose-500/25 text-rose-200"
                            : "bg-slate-950/60 text-slate-400";
                    return (
                      <td key={ci} className="p-1">
                        <span className={`inline-block w-14 rounded-md py-1 font-mono text-[11px] font-semibold ${bg}`}>
                          {val.toFixed(2)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ThematicModelVisual({ mr }: { mr: ThematicModelResult }) {
  if (mr.model_type === "var_granger") {
    const [v1, v2] = mr.var_names;
    return (
      <div className="grid gap-5 xl:grid-cols-2">
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-1 text-sm font-bold text-white">
            Função Impulso-Resposta (IRF) — VAR({mr.used_lag}) • AIC {mr.aic.toFixed(2)}
          </h4>
          <p className="mb-3 text-xs text-slate-400">Propagação dinâmica em 10 passos entre Δ({v1}) e Δ({v2}).</p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mr.irf_points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="h" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
                <Tooltip content={<StudioTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line type="monotone" dataKey="shock_2to1" name={`Choque ${v2} → ${v1}`} stroke="#34d399" strokeWidth={2.2} />
                <Line type="monotone" dataKey="shock_1to2" name={`Choque ${v1} → ${v2}`} stroke="#38bdf8" strokeWidth={2.2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-3 text-sm font-bold text-white">Testes de Causalidade de Granger (1ª Diferença)</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                  <th className="py-2">Lag</th>
                  <th className="px-2 py-2 text-right">F ({v2}→{v1})</th>
                  <th className="px-2 py-2 text-right">p-valor</th>
                  <th className="px-2 py-2 text-right">F ({v1}→{v2})</th>
                  <th className="pl-2 py-2 text-right">p-valor</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {mr.granger_rows.map((r) => (
                  <tr key={r.lag}>
                    <td className="py-2 font-mono font-bold text-white">{r.lag}</td>
                    <td className="px-2 py-2 text-right font-mono text-slate-300">{r.f_2to1.toFixed(2)}</td>
                    <td className="px-2 py-2 text-right">
                      <span
                        className={`rounded-md px-1.5 py-0.5 font-mono text-[11px] font-semibold ${
                          r.sig_2to1 ? "bg-emerald-500/20 text-emerald-300" : "text-slate-400"
                        }`}
                      >
                        {r.p_2to1.toFixed(4)}
                      </span>
                    </td>
                    <td className="px-2 py-2 text-right font-mono text-slate-300">{r.f_1to2.toFixed(2)}</td>
                    <td className="pl-2 py-2 text-right">
                      <span
                        className={`rounded-md px-1.5 py-0.5 font-mono text-[11px] font-semibold ${
                          r.sig_1to2 ? "bg-sky-500/20 text-sky-300" : "text-slate-400"
                        }`}
                      >
                        {r.p_1to2.toFixed(4)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }

  if (mr.model_type === "coint_vecm") {
    const [v1, v2] = mr.var_names;
    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Engle-Granger (t / p)</div>
            <div className="mt-1 font-mono text-base font-bold text-white">
              {mr.coint_stat.toFixed(2)} / p={mr.coint_pval.toFixed(4)}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Cointegração (5%)</div>
            <div className={`mt-1 text-base font-bold ${mr.is_cointegrated ? "text-emerald-300" : "text-amber-300"}`}>
              {mr.is_cointegrated ? "Confirmada ✓" : "Não rejeita H0"}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Elasticidade Longo Prazo (β)</div>
            <div className="mt-1 font-mono text-base font-bold text-sky-300">{mr.beta_lr.toFixed(4)}</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Ajuste ECM (α_ECT)</div>
            <div className="mt-1 font-mono text-base font-bold text-violet-300">
              {mr.alpha_ect.toFixed(4)} (p={mr.p_ect.toFixed(3)})
            </div>
          </div>
        </div>

        <div className="grid gap-5 xl:grid-cols-2">
          <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
            <h4 className="mb-3 text-sm font-bold text-white">
              {v1}: Observado vs. Equilíbrio de Longo Prazo (em função de {v2})
            </h4>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mr.points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={26} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={["auto", "auto"]} />
                  <Tooltip content={<StudioTooltip />} />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Line type="monotone" dataKey="actual" name="Observado" stroke="#38bdf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="equilibrium" name="Equilíbrio Longo Prazo" stroke="#34d399" strokeWidth={2} strokeDasharray="5 3" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
            <h4 className="mb-3 text-sm font-bold text-white">Termo de Correção de Erros (Desvio do Equilíbrio — ECT)</h4>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={mr.points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={26} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />
                  <Tooltip content={<StudioTooltip />} />
                  <Area type="monotone" dataKey="ect" name="Resíduo ECT" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (mr.model_type === "hp_neutral") {
    return (
      <div className="grid gap-5 xl:grid-cols-2">
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-3 text-sm font-bold text-white">
            Decomposição Hodrick-Prescott (λ = {mr.hp_lambda}) — {mr.target}
          </h4>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mr.points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={26} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip content={<StudioTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line type="monotone" dataKey="observed" name="Série Observada" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="trend" name="Tendência Estrutural (HP)" stroke="#f59e0b" strokeWidth={2.4} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-3 text-sm font-bold text-white">
            Componente Cíclico / Hiato (Último: {mr.latest_cycle >= 0 ? "+" : ""}{mr.latest_cycle.toFixed(2)})
          </h4>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={mr.points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={26} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#64748b" strokeDasharray="4 4" />
                <Tooltip content={<StudioTooltip />} />
                <Area type="monotone" dataKey="cycle" name="Ciclo / Hiato" stroke="#34d399" fill="#34d399" fillOpacity={0.22} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  }

  if (mr.model_type === "volatility") {
    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Volatilidade EWMA Atual</div>
            <div className="mt-1 font-mono text-base font-bold text-rose-300">{mr.latest_vol.toFixed(3)}</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Pico Histórico EWMA</div>
            <div className="mt-1 font-mono text-base font-bold text-amber-300">{mr.max_vol.toFixed(3)}</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Curtose (Excess)</div>
            <div className="mt-1 font-mono text-base font-bold text-white">{mr.kurtosis.toFixed(2)}</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
            <div className="text-[11px] text-slate-400">Assimetria (Skewness)</div>
            <div className="mt-1 font-mono text-base font-bold text-sky-300">{mr.skewness.toFixed(2)}</div>
          </div>
        </div>
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-3 text-sm font-bold text-white">
            Retornos ({mr.ret_label}) &amp; Volatilidade Condicional EWMA (λ = 0,94)
          </h4>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={mr.points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={28} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#475569" />
                <Tooltip content={<StudioTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Bar dataKey="ret" name="Retorno / Variação" fill="#38bdf8" opacity={0.45} />
                <Line type="monotone" dataKey="vol" name="Volatilidade EWMA" stroke="#fb7185" strokeWidth={2.2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  }

  if (mr.model_type === "sarima_forecast") {
    return (
      <div className="grid gap-5 xl:grid-cols-2">
        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-3 text-sm font-bold text-white">
            Projeção ARIMA(1,1,1) — {mr.target} (AIC {mr.aic.toFixed(1)} • BIC {mr.bic.toFixed(1)})
          </h4>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={mr.forecast_points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={22} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip content={<StudioTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Area type="monotone" dataKey="ci95" name="IC 95%" stroke="none" fill="#34d399" fillOpacity={0.2} />
                <Line type="monotone" dataKey="observed" name="Histórico" stroke="#38bdf8" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="forecast" name="Previsão (12 passos)" stroke="#34d399" strokeWidth={2.4} dot={{ r: 2 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
          <h4 className="mb-3 text-sm font-bold text-white">Correlograma (ACF &amp; PACF)</h4>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mr.correlogram} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="lag" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={[-1, 1]} />
                <ReferenceLine y={0} stroke="#64748b" />
                <Tooltip content={<StudioTooltip />} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Bar dataKey="acf" name="ACF" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="pacf" name="PACF" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    );
  }

  // ols_regression
  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
        <h4 className="mb-3 text-sm font-bold text-white">
          Regressão MQO-HAC ({mr.dep_var}) — R² = {(mr.r2 * 100).toFixed(1)}% • F = {mr.f_stat.toFixed(1)}
        </h4>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mr.points} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={26} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={["auto", "auto"]} />
              <Tooltip content={<StudioTooltip />} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line type="monotone" dataKey="actual" name="Observado" stroke="#38bdf8" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="fitted" name="Ajustado (MQO)" stroke="#34d399" strokeWidth={2} strokeDasharray="5 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <CoefTable rows={mr.coefs} title="Coeficientes Estimados — Erros-Padrão Robustos Newey-West (HAC)" />
    </div>
  );
}

export const StudyStudioPanel: React.FC<StudyStudioPanelProps> = ({
  study,
  allStudies,
  result,
  loading,
  error,
  onBack,
  onSelectStudy,
  onExecute,
}) => {
  // Parâmetros locais específicos de cada modelo
  const [subsampleStart, setSubsampleStart] = useState("2014-09-05");
  const [maxLags, setMaxLags] = useState(12);
  const [horizon, setHorizon] = useState(12);
  const [hpLambda, setHpLambda] = useState(14400);
  const [rollingWindow, setRollingWindow] = useState(48);
  const [hacLags, setHacLags] = useState(4);
  const [useLive, setUseLive] = useState(false);

  // Parâmetros SARIMA (M13)
  const [p, setP] = useState(1);
  const [d, setD] = useState(0);
  const [q, setQ] = useState(1);
  const [P, setPS] = useState(1);
  const [D, setDS] = useState(0);
  const [Q, setQS] = useState(1);

  useEffect(() => {
    setSubsampleStart("2014-09-05");
    setMaxLags(12);
    setHorizon(12);
    setHpLambda(14400);
    setRollingWindow(48);
    setHacLags(4);
    setUseLive(study.code === "M02" || study.code === "M06" || study.code === "M12");
    setP(1);
    setD(0);
    setQ(1);
    setPS(1);
    setDS(0);
    setQS(1);
  }, [study.code]);

  const handleRun = () => {
    const code = study.code;
    if (code === "M01") {
      onExecute(code, { subsample_start: subsampleStart, max_lags: maxLags, use_live: useLive });
    } else if (code === "M02") {
      onExecute(code, { horizon, use_live: useLive });
    } else if (code === "M06") {
      onExecute(code, { hp_lambda: hpLambda, use_live: useLive });
    } else if (code === "M12") {
      onExecute(code, { window: rollingWindow, use_live: useLive });
    } else if (code === "M13") {
      onExecute(code, { p, d, q, P, D, Q, horizon });
    } else if (code === "M14") {
      onExecute(code, { hac_lags: hacLags });
    } else {
      onExecute(code, { use_live: useLive });
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Navigation & Quick Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/80 px-3.5 py-2 text-xs font-semibold text-slate-200 transition hover:border-slate-700 hover:bg-slate-800"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Voltar ao Catálogo
        </button>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            value={study.code}
            onChange={(e) => onSelectStudy(e.target.value)}
            className="max-w-xs rounded-xl border border-slate-700/80 bg-slate-950 px-3 py-2 text-xs font-medium text-slate-200 outline-none focus:border-emerald-500"
          >
            {allStudies.map((s) => (
              <option key={s.code} value={s.code}>
                {s.code} — {s.title}
              </option>
            ))}
          </select>

          <a
            href={getExportCsvUrl(study.code)}
            download
            className="inline-flex items-center gap-1.5 rounded-xl border border-emerald-500/40 bg-emerald-500/15 px-3.5 py-2 text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/25"
          >
            <Download className="h-3.5 w-3.5" />
            Exportar CSV
          </a>
        </div>
      </div>

      {/* Authorial Header & Didactic 3-Card Guide */}
      <section className="rounded-3xl border border-slate-800/90 bg-gradient-to-br from-slate-900/90 to-slate-950 p-6 shadow-xl">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-lg bg-emerald-500/20 px-2.5 py-0.5 font-mono text-xs font-bold text-emerald-300">
            {study.code}
          </span>
          <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-0.5 text-xs font-medium text-slate-300">
            {study.theme_label}
          </span>
          {study.specialized && (
            <span className="inline-flex items-center gap-1 rounded-full border border-amber-400/40 bg-amber-500/15 px-2.5 py-0.5 text-[11px] font-bold text-amber-300">
              <Sparkles className="h-3 w-3" /> Modelo Dedicado Interativo
            </span>
          )}
          <span className="ml-auto rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-0.5 text-[11px] font-medium text-slate-400">
            {study.methodology_label}
          </span>
        </div>

        <h2 className="mt-3 text-2xl font-extrabold tracking-tight text-white">{study.title}</h2>
        <p className="mt-1 text-sm text-slate-300">{study.subtitle}</p>

        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.04] p-4">
            <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-emerald-300">
              <Lightbulb className="h-3.5 w-3.5" /> Intuição Econômica
            </h3>
            <p className="text-xs leading-relaxed text-slate-300">{study.economic_intuition}</p>
          </div>
          <div className="rounded-2xl border border-sky-500/20 bg-sky-500/[0.04] p-4">
            <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-sky-300">
              <SlidersHorizontal className="h-3.5 w-3.5" /> Metodologia Econométrica
            </h3>
            <p className="text-xs leading-relaxed text-slate-300">{study.methodology_explanation}</p>
          </div>
          <div className="rounded-2xl border border-amber-500/20 bg-amber-500/[0.04] p-4">
            <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-amber-300">
              <Compass className="h-3.5 w-3.5" /> Como Interpretar
            </h3>
            <p className="text-xs leading-relaxed text-slate-300">{study.interpretation_guide}</p>
          </div>
        </div>

        {study.references.length > 0 && (
          <div className="mt-4 flex flex-wrap items-start gap-2 border-t border-slate-800/80 pt-3 text-[11px] text-slate-400">
            <BookMarked className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-500" />
            <div className="space-y-0.5">
              {study.references.map((ref, idx) => (
                <div key={idx}>{ref}</div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Parameter Control Bar */}
      <section className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-4">
        <div className="flex flex-wrap items-end gap-4">
          {study.code === "M01" && (
            <>
              <label className="flex flex-col gap-1 text-xs text-slate-300">
                <span>Início Subamostra</span>
                <input
                  type="date"
                  value={subsampleStart}
                  onChange={(e) => setSubsampleStart(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs text-white"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-slate-300">
                <span>Defasagens Máx. (VAR/Granger)</span>
                <input
                  type="number"
                  min={2}
                  max={20}
                  value={maxLags}
                  onChange={(e) => setMaxLags(Number(e.target.value))}
                  className="w-24 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs text-white"
                />
              </label>
            </>
          )}

          {study.code === "M02" && (
            <label className="flex flex-col gap-1 text-xs text-slate-300">
              <span>Horizonte de Projeção (meses)</span>
              <input
                type="number"
                min={3}
                max={36}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-28 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs text-white"
              />
            </label>
          )}

          {study.code === "M06" && (
            <label className="flex flex-col gap-1 text-xs text-slate-300">
              <span>Parâmetro λ Filtro HP</span>
              <select
                value={hpLambda}
                onChange={(e) => setHpLambda(Number(e.target.value))}
                className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs text-white"
              >
                <option value={1600}>1.600 (Trimestral)</option>
                <option value={14400}>14.400 (Mensal Padrão)</option>
                <option value={50000}>50.000 (Mais Suave)</option>
              </select>
            </label>
          )}

          {study.code === "M12" && (
            <label className="flex flex-col gap-1 text-xs text-slate-300">
              <span>Janela Móvel AR(1) (meses)</span>
              <input
                type="number"
                min={18}
                max={96}
                value={rollingWindow}
                onChange={(e) => setRollingWindow(Number(e.target.value))}
                className="w-28 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs text-white"
              />
            </label>
          )}

          {study.code === "M13" && (
            <div className="flex flex-wrap items-end gap-2">
              {[
                { lbl: "p", v: p, set: setP, max: 4 },
                { lbl: "d", v: d, set: setD, max: 2 },
                { lbl: "q", v: q, set: setQ, max: 4 },
                { lbl: "P", v: P, set: setPS, max: 2 },
                { lbl: "D", v: D, set: setDS, max: 1 },
                { lbl: "Q", v: Q, set: setQS, max: 2 },
              ].map((ord) => (
                <label key={ord.lbl} className="flex flex-col gap-1 text-xs text-slate-300">
                  <span className="text-center font-mono">{ord.lbl}</span>
                  <input
                    type="number"
                    min={0}
                    max={ord.max}
                    value={ord.v}
                    onChange={(e) => ord.set(Number(e.target.value))}
                    className="w-14 rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 text-center font-mono text-xs text-white"
                  />
                </label>
              ))}
              <label className="flex flex-col gap-1 text-xs text-slate-300">
                <span>Horizonte (m)</span>
                <input
                  type="number"
                  min={3}
                  max={24}
                  value={horizon}
                  onChange={(e) => setHorizon(Number(e.target.value))}
                  className="w-20 rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 text-center font-mono text-xs text-white"
                />
              </label>
            </div>
          )}

          {study.code === "M14" && (
            <label className="flex flex-col gap-1 text-xs text-slate-300">
              <span>Defasagens Newey-West HAC</span>
              <input
                type="number"
                min={1}
                max={12}
                value={hacLags}
                onChange={(e) => setHacLags(Number(e.target.value))}
                className="w-24 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs text-white"
              />
            </label>
          )}

          {study.code !== "M13" && study.code !== "M14" && study.bcb_series.length > 0 && (
            <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700/80 bg-slate-950 px-3 py-2 text-xs text-slate-200">
              <input
                type="checkbox"
                checked={useLive}
                onChange={(e) => setUseLive(e.target.checked)}
                className="h-3.5 w-3.5 rounded accent-emerald-500"
              />
              Usar séries oficiais BCB (SGS/Olinda)
            </label>
          )}

          <button
            onClick={handleRun}
            disabled={loading}
            className="ml-auto inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-sky-500 px-4 py-2 text-xs font-bold text-slate-950 shadow-lg shadow-emerald-500/20 transition hover:brightness-110 disabled:opacity-60"
          >
            {loading ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            {loading ? "Estimando em Python..." : "Executar / Recalcular"}
          </button>
        </div>
      </section>

      {error && (
        <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">{error}</div>
      )}

      {loading && !result && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-12 text-center text-sm text-slate-300">
          <RefreshCw className="mx-auto mb-3 h-6 w-6 animate-spin text-emerald-400" />
          Estimando modelos econométricos no backend Python (Statsmodels + DuckDB)...
        </div>
      )}

      {result && (
        <div className="space-y-6">
          {/* KPIs */}
          <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5">
            {result.kpis.map((k, idx) => (
              <div key={idx} className="rounded-2xl border border-slate-800/90 bg-slate-900/70 p-4">
                <div className="text-[11px] font-medium text-slate-400">{k.label}</div>
                <div className="mt-1.5 font-mono text-xl font-extrabold text-white">{k.value}</div>
                <div className="mt-1 text-[11px] text-slate-400">{k.sub}</div>
              </div>
            ))}
          </div>

          {/* Conclusion Banner */}
          <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
              <div className="space-y-1">
                <div className="text-xs font-bold uppercase tracking-wider text-emerald-300">
                  Diagnóstico Econométrico ({result.elapsed_ms.toFixed(0)} ms)
                </div>
                <p className="text-sm leading-relaxed text-emerald-50">{result.conclusion}</p>
                <p className="text-[11px] text-emerald-200/75">{result.source_note}</p>
              </div>
            </div>
          </div>

          {/* DEDICATED M01 */}
          {result.kind === "dedicated_m01" && (
            <div className="space-y-5">
              <div className="grid gap-5 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">CDS Soberano 5 Anos (bps) vs. Taxa Selic (% a.a.)</h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={result.level_series} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={30} />
                        <YAxis yAxisId="l" tick={{ fill: "#fb7185", fontSize: 11 }} />
                        <YAxis yAxisId="r" orientation="right" tick={{ fill: "#38bdf8", fontSize: 11 }} unit="%" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Line yAxisId="l" type="monotone" dataKey="cds5y" name="CDS 5Y (bps)" stroke="#fb7185" strokeWidth={2} dot={false} />
                        <Line yAxisId="r" type="monotone" dataKey="selic" name="Selic (%)" stroke="#38bdf8" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                    <h4 className="text-sm font-bold text-white">Função de Correlação Cruzada (CCF) — ΔSelic vs. ΔCDS</h4>
                    <span className="font-mono text-[11px] text-slate-400">
                      AIC:{result.lag_criteria.AIC} • BIC:{result.lag_criteria.BIC} • HQIC:{result.lag_criteria.HQIC}
                    </span>
                  </div>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={result.ccf} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="lag" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <ReferenceLine y={0} stroke="#64748b" />
                        <Tooltip content={<StudioTooltip />} />
                        <Bar dataKey="corr" name="Correlação" fill="#34d399" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="grid gap-5 xl:grid-cols-2">
                {[
                  { title: "Causalidade de Granger — Subamostra de Estresse Fiscal", rows: result.granger_sub },
                  { title: "Causalidade de Granger — Amostra Completa", rows: result.granger_full },
                ].map((tbl) => (
                  <div key={tbl.title} className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                    <h4 className="mb-3 text-sm font-bold text-white">{tbl.title}</h4>
                    <div className="max-h-72 overflow-y-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                            <th className="py-2">Lag</th>
                            <th className="px-2 py-2 text-right">F (Selic→CDS)</th>
                            <th className="px-2 py-2 text-right">p-valor</th>
                            <th className="px-2 py-2 text-right">F (CDS→Selic)</th>
                            <th className="pl-2 py-2 text-right">p-valor</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/70">
                          {tbl.rows.map((r) => (
                            <tr key={r.lag}>
                              <td className="py-1.5 font-mono font-bold text-white">{r.lag}</td>
                              <td className="px-2 py-1.5 text-right font-mono text-slate-300">{r.f_selic_to_cds.toFixed(2)}</td>
                              <td className="px-2 py-1.5 text-right">
                                <span
                                  className={`rounded-md px-1.5 py-0.5 font-mono text-[11px] font-semibold ${
                                    r.sig_selic_to_cds ? "bg-emerald-500/20 text-emerald-300" : "text-slate-400"
                                  }`}
                                >
                                  {r.p_selic_to_cds.toFixed(4)}
                                </span>
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono text-slate-300">{r.f_cds_to_selic.toFixed(2)}</td>
                              <td className="pl-2 py-1.5 text-right">
                                <span
                                  className={`rounded-md px-1.5 py-0.5 font-mono text-[11px] font-semibold ${
                                    r.sig_cds_to_selic ? "bg-rose-500/20 text-rose-300" : "text-slate-400"
                                  }`}
                                >
                                  {r.p_cds_to_selic.toFixed(4)}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </div>

              <AdfTable rows={result.adf} />
            </div>
          )}

          {/* DEDICATED M02 */}
          {result.kind === "dedicated_m02" && (
            <div className="space-y-5">
              <div className="grid gap-5 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Trajetória e Projeção ARIMA(1,1,1) da DBGG (% PIB)</h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={result.forecast_series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={["auto", "auto"]} unit="%" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Area type="monotone" dataKey="ci95" name="IC 95%" stroke="none" fill="#fb7185" fillOpacity={0.14} />
                        <Area type="monotone" dataKey="ci80" name="IC 80%" stroke="none" fill="#fb7185" fillOpacity={0.22} />
                        <Line type="monotone" dataKey="observed" name="DBGG Observada" stroke="#38bdf8" strokeWidth={2.2} dot={false} />
                        <Line type="monotone" dataKey="forecast" name="Projeção ARIMA" stroke="#fb7185" strokeWidth={2.4} dot={{ r: 2 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">
                    Painel Fiscal Histórico: Superávit Primário Efetivo vs. Requerido (% PIB)
                  </h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={result.fiscal_panel} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} unit="%" />
                        <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Line type="monotone" dataKey="s_primario" name="Primário Efetivo" stroke="#34d399" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="s_req" name="Primário Requerido" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 3" dot={false} />
                        <Line type="monotone" dataKey="juro_real" name="Juro Real" stroke="#a78bfa" strokeWidth={1.6} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
              <AdfTable rows={result.adf} />
            </div>
          )}

          {/* DEDICATED M06 */}
          {result.kind === "dedicated_m06" && (
            <div className="space-y-5">
              <div className="grid gap-5 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Taxa Selic Observada vs. Ajustada pela Regra de Taylor</h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={result.series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} unit="%" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Line type="monotone" dataKey="selic_obs" name="Selic Observada" stroke="#38bdf8" strokeWidth={2.4} dot={false} />
                        <Line type="monotone" dataKey="selic_fit" name="Regra de Taylor (Ajustado)" stroke="#34d399" strokeWidth={2} strokeDasharray="5 3" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Hiato do Produto (Filtro HP) &amp; Desvio de Expectativa</h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={result.series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Area type="monotone" dataKey="hiato" name="Hiato do Produto" stroke="#34d399" fill="#34d399" fillOpacity={0.18} />
                        <Line type="monotone" dataKey="desvio_inf" name="Desvio Inflação (E[π]−π*)" stroke="#f59e0b" strokeWidth={2} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
              <CoefTable rows={result.coefs} title="Função de Reação do Banco Central (MQO com Erros Newey-West HAC)" />
            </div>
          )}

          {/* DEDICATED M12 */}
          {result.kind === "dedicated_m12" && (
            <div className="grid gap-5 xl:grid-cols-2">
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                <h4 className="mb-3 text-sm font-bold text-white">IPCA Acumulado em 12 Meses vs. Meta de Inflação</h4>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={result.ipca_series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                      <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} unit="%" />
                      <Tooltip content={<StudioTooltip />} />
                      <Legend wrapperStyle={{ fontSize: "12px" }} />
                      <Line type="monotone" dataKey="ipca12" name="IPCA 12m (%)" stroke="#f59e0b" strokeWidth={2.2} dot={false} />
                      <Line type="stepAfter" dataKey="meta" name="Meta (%)" stroke="#34d399" strokeWidth={2} strokeDasharray="5 3" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                <h4 className="mb-3 text-sm font-bold text-white">
                  Persistência Inflacionária Dinâmica — γ(t) Rolling AR(1) com IC 95%
                </h4>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={result.rolling_series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                      <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={[-0.4, 1]} />
                      <ReferenceLine y={0} stroke="#64748b" />
                      <ReferenceLine y={0.5} stroke="#f59e0b" strokeDasharray="4 4" />
                      <Tooltip content={<StudioTooltip />} />
                      <Legend wrapperStyle={{ fontSize: "12px" }} />
                      <Area type="monotone" dataKey="band" name="IC 95%" stroke="none" fill="#a78bfa" fillOpacity={0.22} />
                      <Line type="monotone" dataKey="gamma" name="Inércia γ(t)" stroke="#a78bfa" strokeWidth={2.4} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* DEDICATED M13 */}
          {result.kind === "dedicated_m13" && (
            <div className="space-y-5">
              <div className="grid gap-5 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Previsão Mensal do IPCA — SARIMA com Banda IC 95%</h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={result.forecast_series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={20} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} unit="%" />
                        <ReferenceLine y={0} stroke="#475569" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Area type="monotone" dataKey="ci95" name="IC 95%" stroke="none" fill="#34d399" fillOpacity={0.2} />
                        <Line type="monotone" dataKey="observed" name="IPCA Observado" stroke="#38bdf8" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="forecast" name="Previsão SARIMA" stroke="#34d399" strokeWidth={2.5} dot={{ r: 2.5 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Diagnóstico de Memória: Correlograma ACF &amp; PACF</h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={result.correlogram} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="lag" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={[-0.6, 0.8]} />
                        <ReferenceLine y={0} stroke="#64748b" />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Bar dataKey="acf" name="ACF" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                        <Bar dataKey="pacf" name="PACF" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
              <AdfTable rows={result.adf} />
            </div>
          )}

          {/* DEDICATED M14 */}
          {result.kind === "dedicated_m14" && (
            <div className="space-y-5">
              <div className="grid gap-5 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Curva de Phillips ({result.dep_var}): Observado vs. Ajustado</h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={result.series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <Tooltip content={<StudioTooltip />} />
                        <Legend wrapperStyle={{ fontSize: "12px" }} />
                        <Line type="monotone" dataKey="actual" name="Observado" stroke="#f59e0b" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="fitted" name="Curva de Phillips Ajustada" stroke="#34d399" strokeWidth={2} strokeDasharray="5 3" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                  <h4 className="mb-3 text-sm font-bold text-white">Resíduos da Curva de Phillips (Choques Não Explicados)</h4>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={result.series} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={24} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <ReferenceLine y={0} stroke="#64748b" />
                        <Tooltip content={<StudioTooltip />} />
                        <Bar dataKey="resid" name="Resíduo" fill="#a78bfa" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
              <CoefTable rows={result.coefs} title="Coeficientes da Curva de Phillips (MQO Newey-West HAC)" />
              <AdfTable rows={result.adf} />
            </div>
          )}

          {/* THEMATIC STUDIES (M03..M60) */}
          {result.kind === "thematic" && (
            <div className="space-y-5">
              <ThematicModelVisual mr={result.model_result} />

              {/* Exploratory Series Chart */}
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5">
                <h4 className="mb-3 text-sm font-bold text-white">Trajetória Histórica das Variáveis da Base ({study.code})</h4>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={result.raw_chart} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={28} />
                      <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={["auto", "auto"]} />
                      <Tooltip content={<StudioTooltip />} />
                      <Legend wrapperStyle={{ fontSize: "12px" }} />
                      {result.series_keys.map((key, idx) => (
                        <Line
                          key={key}
                          type="monotone"
                          dataKey={key}
                          name={key}
                          stroke={SERIES_COLORS[idx % SERIES_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <AdfTable rows={result.adf} />
              <SummaryAndCorr stats={result.summary_stats} corr={result.corr_matrix} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
