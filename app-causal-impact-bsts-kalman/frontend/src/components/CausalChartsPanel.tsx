import React from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  MinusCircle,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";
import type { AnalyzeResponse } from "../types";

interface CausalChartsPanelProps {
  analysis: AnalyzeResponse;
}

function fmtNum(val: number, decimals = 2): string {
  return val.toLocaleString("pt-BR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtSigned(val: number, decimals = 2): string {
  const sign = val > 0 ? "+" : "";
  return `${sign}${fmtNum(val, decimals)}`;
}

function renderBoldText(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-white">
          {p.slice(2, -2)}
        </strong>
      );
    }
    if (p.startsWith("`") && p.endsWith("`")) {
      return (
        <code key={i} className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-xs text-cyan-300">
          {p.slice(1, -1)}
        </code>
      );
    }
    return <React.Fragment key={i}>{p}</React.Fragment>;
  });
}

export const CausalChartsPanel: React.FC<CausalChartsPanelProps> = ({ analysis }) => {
  const { metrics: m, pre_fit, series } = analysis;

  const chartSeries = series.map((pt) => ({
    ...pt,
    predBand: [pt.lower, pt.upper] as [number, number],
    pointBand: [pt.point_effect_lower, pt.point_effect_upper] as [number, number],
    cumBand: [pt.cum_effect_lower, pt.cum_effect_upper] as [number, number],
  }));

  const isPos = m.is_significant && m.direction === "positivo";
  const isNeg = m.is_significant && m.direction === "negativo";

  const statusTone = isPos
    ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-200"
    : isNeg
      ? "border-rose-500/35 bg-rose-500/10 text-rose-200"
      : "border-amber-500/35 bg-amber-500/10 text-amber-200";

  return (
    <div className="space-y-5">
      {/* Faixa de KPIs Executivos */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Efeito Médio / Período
          </p>
          <p
            className={`mt-1.5 font-mono text-2xl font-extrabold ${
              isPos ? "text-emerald-300" : isNeg ? "text-rose-300" : "text-amber-300"
            }`}
          >
            {fmtSigned(m.avg_abs_effect, 2)}
          </p>
          <p className="mt-1 text-[11px] text-slate-400">
            IC {m.confidence_level_pct.toFixed(0)}%: [{fmtSigned(m.avg_abs_effect_lower, 1)};{" "}
            {fmtSigned(m.avg_abs_effect_upper, 1)}]
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Impacto Relativo (%)
          </p>
          <div className="mt-1.5 flex items-center gap-1.5">
            {isPos ? (
              <ArrowUpRight className="h-5 w-5 text-emerald-400" />
            ) : isNeg ? (
              <ArrowDownRight className="h-5 w-5 text-rose-400" />
            ) : (
              <MinusCircle className="h-5 w-5 text-amber-400" />
            )}
            <span
              className={`font-mono text-2xl font-extrabold ${
                isPos ? "text-emerald-300" : isNeg ? "text-rose-300" : "text-amber-300"
              }`}
            >
              {fmtSigned(m.avg_rel_effect_pct, 2)}%
            </span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            IC: [{fmtSigned(m.avg_rel_effect_lower_pct, 1)}%; {fmtSigned(m.avg_rel_effect_upper_pct, 1)}%]
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Impacto Acumulado Total
          </p>
          <p className="mt-1.5 font-mono text-2xl font-extrabold text-cyan-300">
            {fmtSigned(m.cum_abs_effect, 1)}
          </p>
          <p className="mt-1 text-[11px] text-slate-400">
            Em {m.n_post} períodos pós ({fmtNum(m.cum_actual, 0)} vs. {fmtNum(m.cum_predicted, 0)})
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Probabilidade Causal (1 - p)
          </p>
          <p className="mt-1.5 font-mono text-2xl font-extrabold text-violet-300">
            {fmtNum(m.causal_probability_pct, 1)}%
          </p>
          <p className="mt-1 text-[11px] text-slate-400">
            p-valor caudal posterior: <span className="font-mono text-slate-200">{m.p_value.toFixed(4)}</span>
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Validade do Contrafactual
          </p>
          <p className="mt-1.5 font-mono text-2xl font-extrabold text-white">
            R² {(pre_fit.r2_pre * 100).toFixed(1)}%
          </p>
          <p className="mt-1 text-[11px] text-slate-400">
            MAPE pré: <span className="font-mono text-cyan-300">{pre_fit.mape_pre_pct.toFixed(2)}%</span> •{" "}
            {analysis.placebo.executed
              ? analysis.placebo.passed
                ? "Placebo OK"
                : "Placebo Alerta"
              : "Sem placebo"}
          </p>
        </div>
      </div>

      {/* Veredito Executivo em Português */}
      <div className={`rounded-2xl border p-4 ${statusTone}`}>
        <div className="flex items-start gap-3">
          {m.is_significant ? (
            <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0" />
          ) : (
            <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0" />
          )}
          <div className="space-y-1">
            <p className="text-xs font-bold uppercase tracking-wider">
              {m.is_significant
                ? `Conclusão Causal: Impacto ${m.direction.toUpperCase()} Estatisticamente Significativo (α = ${m.alpha})`
                : `Conclusão Causal: Sem Evidência de Efeito Estatisticamente Significativo (α = ${m.alpha})`}
            </p>
            <p className="text-sm leading-relaxed text-slate-200">
              {renderBoldText(analysis.executive_summary)}
            </p>
          </div>
        </div>
      </div>

      {/* Avisos Metodológicos (Univariado, Spillover, Ajuste) */}
      {analysis.warnings.length > 0 && (
        <div className="space-y-2">
          {analysis.warnings.map((w, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2.5 rounded-xl border border-amber-500/35 bg-amber-950/25 px-4 py-2.5 text-xs text-amber-200"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* PAINEL 1: Realizado vs Contrafactual + Banda de Credibilidade */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">
              1. Série Observada vs. Contrafactual Estimado — {analysis.y_col}
            </h3>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" /> Realizado
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-400" /> Contrafactual BSTS
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded bg-cyan-400/30" /> IC {m.confidence_level_pct.toFixed(0)}%
            </span>
            <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-amber-300">
              Evento: {m.post_start}
            </span>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartSeries} margin={{ top: 10, right: 20, left: 4, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={28} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  borderRadius: "12px",
                  fontSize: "12px",
                  color: "#f8fafc",
                }}
                formatter={(value: unknown, name: string) => {
                  if (name === "predBand" && Array.isArray(value)) {
                    return [`[${fmtNum(Number(value[0]))}; ${fmtNum(Number(value[1]))}]`, `IC ${m.confidence_level_pct.toFixed(0)}%`];
                  }
                  const label = name === "actual" ? "Realizado" : name === "predicted" ? "Contrafactual" : name;
                  return [fmtNum(Number(value)), label];
                }}
              />
              <ReferenceLine
                x={m.post_start}
                stroke="#f59e0b"
                strokeDasharray="5 5"
                strokeWidth={2}
                label={{ value: "Intervenção", fill: "#fbbf24", fontSize: 11, position: "insideTopRight" }}
              />
              <Area
                type="monotone"
                dataKey="predBand"
                stroke="none"
                fill="#22d3ee"
                fillOpacity={0.18}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#22d3ee"
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#34d399"
                strokeWidth={2.4}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* PAINÉIS 2 & 3 Lado a Lado: Efeito Pontual e Efeito Acumulado */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {/* Painel 2: Efeito Pontual */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
          <div className="mb-3">
            <h3 className="text-sm font-bold text-white">2. Efeito Pontual (Realizado − Contrafactual)</h3>
            <p className="text-xs text-slate-400">
              No período pré-intervenção deve oscilar ao redor de zero; no pós-evento revela a dinâmica temporal do impacto.
            </p>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartSeries} margin={{ top: 8, right: 16, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={34} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                  formatter={(value: unknown, name: string) => {
                    if (name === "pointBand" && Array.isArray(value)) {
                      return [`[${fmtSigned(Number(value[0]))}; ${fmtSigned(Number(value[1]))}]`, "IC Efeito"];
                    }
                    return [fmtSigned(Number(value)), "Efeito Pontual"];
                  }}
                />
                <ReferenceLine y={0} stroke="#64748b" strokeWidth={1.5} />
                <ReferenceLine x={m.post_start} stroke="#f59e0b" strokeDasharray="5 5" strokeWidth={1.8} />
                <Area
                  type="monotone"
                  dataKey="pointBand"
                  stroke="none"
                  fill="#a78bfa"
                  fillOpacity={0.2}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="point_effect"
                  stroke="#a78bfa"
                  strokeWidth={2.2}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Painel 3: Efeito Acumulado */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
          <div className="mb-3">
            <h3 className="text-sm font-bold text-white">3. Efeito Acumulado Pós-Intervenção (Soma Incremental)</h3>
            <p className="text-xs text-slate-400">
              Soma acumulada do impacto causal com banda posterior conjunta (capturando a autocorrelação do estado BSTS).
            </p>
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartSeries} margin={{ top: 8, right: 16, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={34} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                  formatter={(value: unknown, name: string) => {
                    if (name === "cumBand" && Array.isArray(value)) {
                      return [`[${fmtSigned(Number(value[0]), 1)}; ${fmtSigned(Number(value[1]), 1)}]`, "IC Acumulado"];
                    }
                    return [fmtSigned(Number(value), 1), "Efeito Acumulado"];
                  }}
                />
                <ReferenceLine y={0} stroke="#64748b" strokeWidth={1.5} />
                <ReferenceLine x={m.post_start} stroke="#f59e0b" strokeDasharray="5 5" strokeWidth={1.8} />
                <Area
                  type="monotone"
                  dataKey="cumBand"
                  stroke="none"
                  fill="#34d399"
                  fillOpacity={0.2}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="cum_effect"
                  stroke="#34d399"
                  strokeWidth={2.4}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
