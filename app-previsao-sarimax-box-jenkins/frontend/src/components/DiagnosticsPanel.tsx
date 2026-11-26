import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CheckCircle2, AlertTriangle, Lightbulb, Sparkles } from "lucide-react";
import type { DiagnosticsResponse } from "../types";

interface Props {
  data: DiagnosticsResponse | null;
  loading: boolean;
  diffOrder: number;
  seasonalDiff: number;
  onChangeDiff: (d: number, D: number) => void;
  onApplyRecommendation: (d: number, D: number) => void;
}

export const DiagnosticsPanel: React.FC<Props> = ({
  data,
  loading,
  diffOrder,
  seasonalDiff,
  onChangeDiff,
  onApplyRecommendation,
}) => {
  if (loading) {
    return (
      <div className="flex h-72 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900/60">
        <div className="flex items-center gap-3 text-slate-300">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-400 border-t-transparent" />
          <span>Calculando testes ADF, KPSS, ACF e PACF...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/40 p-10 text-center text-slate-400">
        Carregue um dataset na barra lateral para iniciar o diagnóstico Box-Jenkins.
      </div>
    );
  }

  const bound = data.correlations[0]?.conf_bound ?? 0.15;

  return (
    <div className="space-y-6">
      {/* Banner Didático + Controles de Diferenciação Interativa */}
      <div className="rounded-2xl border border-indigo-500/30 bg-gradient-to-r from-indigo-950/70 via-slate-900 to-slate-900 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-indigo-300">
              <Lightbulb className="h-5 w-5" />
              <span className="text-xs font-semibold uppercase tracking-wider">
                Etapa 1 de Box-Jenkins — Identificação &amp; Estacionariedade
              </span>
            </div>
            <p className="text-sm leading-relaxed text-slate-200">{data.didactic_summary}</p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <div className="flex items-center gap-1 rounded-xl border border-slate-700 bg-slate-950/70 p-1 text-xs">
              <span className="px-2 text-slate-400">d:</span>
              {[0, 1, 2].map((val) => (
                <button
                  key={val}
                  onClick={() => onChangeDiff(val, seasonalDiff)}
                  className={`rounded-lg px-2.5 py-1 font-semibold transition ${
                    diffOrder === val
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {val}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1 rounded-xl border border-slate-700 bg-slate-950/70 p-1 text-xs">
              <span className="px-2 text-slate-400">D (sazonal):</span>
              {[0, 1].map((val) => (
                <button
                  key={val}
                  onClick={() => onChangeDiff(diffOrder, val)}
                  className={`rounded-lg px-2.5 py-1 font-semibold transition ${
                    seasonalDiff === val
                      ? "bg-emerald-600 text-white"
                      : "text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {val}
                </button>
              ))}
            </div>
            <button
              onClick={() => onApplyRecommendation(data.recommended_d, data.recommended_D)}
              className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-500/20 px-3 py-2 text-xs font-semibold text-indigo-300 ring-1 ring-indigo-500/40 transition hover:bg-indigo-500/30"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Aplicar Sugerido (d={data.recommended_d}, D={data.recommended_D})
            </button>
          </div>
        </div>
      </div>

      {/* Cards de Testes ADF e KPSS */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[data.adf, data.kpss].map((t) => (
          <div
            key={t.test_name}
            className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">{t.test_name}</h3>
                <div className="mt-2 flex items-baseline gap-4">
                  <div>
                    <span className="text-xs text-slate-400">Estatística: </span>
                    <span className="font-mono text-lg font-bold text-white">{t.statistic}</span>
                  </div>
                  <div>
                    <span className="text-xs text-slate-400">P-Valor: </span>
                    <span className="font-mono text-lg font-bold text-indigo-300">{t.p_value}</span>
                  </div>
                </div>
              </div>
              <span
                className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
                  t.is_stationary
                    ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                    : "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30"
                }`}
              >
                {t.is_stationary ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <AlertTriangle className="h-3.5 w-3.5" />
                )}
                {t.is_stationary ? "Estacionária" : "Não Estacionária"}
              </span>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-slate-400">{t.interpretation}</p>
          </div>
        ))}
      </div>

      {/* Série Transformada no Tempo */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Série Analisada (d = {diffOrder}, D = {seasonalDiff})
            </h3>
            <p className="text-xs text-slate-400">
              Uma série estacionária oscila em torno de uma média constante, sem tendência crescente/decrescente.
            </p>
          </div>
          <span className="rounded-lg bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
            N = {data.n_obs} observações
          </span>
        </div>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.transformed_series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={35} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} width={65} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: 12 }}
                labelStyle={{ color: "#e2e8f0", fontWeight: 600 }}
              />
              <Line type="monotone" dataKey="value" name="Valor" stroke="#6366f1" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Gráficos ACF e PACF lado a lado */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <h3 className="text-sm font-semibold text-white">ACF — Autocorrelação (Sugere ordem MA: q)</h3>
          <p className="mb-3 text-xs text-slate-400">
            Picos significativos fora da faixa ±{bound.toFixed(2)} indicam termos de Média Móvel (q / Q).
          </p>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.correlations}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="lag" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis domain={[-1, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: 12 }}
                />
                <ReferenceLine y={bound} stroke="#f59e0b" strokeDasharray="4 4" />
                <ReferenceLine y={-bound} stroke="#f59e0b" strokeDasharray="4 4" />
                <ReferenceLine y={0} stroke="#475569" />
                <Bar dataKey="acf" name="ACF" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <h3 className="text-sm font-semibold text-white">PACF — Autocorrelação Parcial (Sugere ordem AR: p)</h3>
          <p className="mb-3 text-xs text-slate-400">
            Corte abrupto após o lag k fora da faixa ±{bound.toFixed(2)} sugere ordem Autorregressiva (p / P).
          </p>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.correlations}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="lag" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis domain={[-1, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: 12 }}
                />
                <ReferenceLine y={bound} stroke="#f59e0b" strokeDasharray="4 4" />
                <ReferenceLine y={-bound} stroke="#f59e0b" strokeDasharray="4 4" />
                <ReferenceLine y={0} stroke="#475569" />
                <Bar dataKey="pacf" name="PACF" fill="#a78bfa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
