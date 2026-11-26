import React from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
import { AlertTriangle, CheckCircle2, FlaskConical, Gauge, Layers, ShieldAlert } from "lucide-react";
import type { AnalyzeResponse } from "../types";

interface AuditPlaceboPanelProps {
  analysis: AnalyzeResponse;
}

export const AuditPlaceboPanel: React.FC<AuditPlaceboPanelProps> = ({ analysis }) => {
  const { pre_fit, placebo, covariates, series } = analysis;

  const placeboChartData = placebo.series.map((pt) => ({
    ...pt,
    band: [pt.lower, pt.upper] as [number, number],
  }));

  const covChartData = covariates.map((c) => ({
    name: c.name,
    peso: c.importance_pct,
    betaStd: c.beta_std,
    spillover: c.spillover_warning,
  }));

  return (
    <div className="space-y-5">
      {/* 1. Diagnóstico do Ajuste Pré-Intervenção */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Gauge className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">1. Auditoria do Ajuste Pré-Intervenção (Validade Interna)</h3>
          </div>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-bold uppercase ${
              pre_fit.fit_quality_label === "excelente"
                ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                : pre_fit.fit_quality_label === "bom"
                  ? "bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30"
                  : "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30"
            }`}
          >
            Ajuste {pre_fit.fit_quality_label}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] uppercase text-slate-400">R² Pré</p>
            <p className="mt-1 font-mono text-lg font-extrabold text-emerald-300">
              {(pre_fit.r2_pre * 100).toFixed(2)}%
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] uppercase text-slate-400">RMSE Pré</p>
            <p className="mt-1 font-mono text-lg font-extrabold text-white">{pre_fit.rmse_pre.toFixed(3)}</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] uppercase text-slate-400">MAPE Pré</p>
            <p className="mt-1 font-mono text-lg font-extrabold text-cyan-300">{pre_fit.mape_pre_pct.toFixed(2)}%</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] uppercase text-slate-400">Ljung-Box p</p>
            <p
              className={`mt-1 font-mono text-lg font-extrabold ${
                pre_fit.residuals_white_noise ? "text-emerald-300" : "text-amber-300"
              }`}
            >
              {pre_fit.ljung_box_pvalue.toFixed(3)}
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] uppercase text-slate-400">Resíduos</p>
            <p className="mt-1 text-xs font-bold text-slate-200">
              {pre_fit.residuals_white_noise ? "Ruído Branco ✓" : "Autocorrelação"}
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] uppercase text-slate-400">AIC / BIC</p>
            <p className="mt-1 font-mono text-xs font-bold text-violet-300">
              {pre_fit.aic.toFixed(1)} / {pre_fit.bic.toFixed(1)}
            </p>
          </div>
        </div>
      </div>

      {/* 2. Teste Placebo no Tempo (In-Time Falsification) */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-violet-400" />
            <h3 className="text-sm font-bold text-white">
              2. Teste Placebo no Tempo (Falsificação no Período Pré-Intervenção)
            </h3>
          </div>
          {placebo.executed && (
            <span
              className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-bold ${
                placebo.passed
                  ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                  : "bg-rose-500/15 text-rose-300 ring-1 ring-rose-500/30"
              }`}
            >
              {placebo.passed ? <CheckCircle2 className="h-3.5 w-3.5" /> : <ShieldAlert className="h-3.5 w-3.5" />}
              {placebo.passed ? "Placebo Aprovado (Sem Falso Positivo)" : "Placebo em Alerta"}
            </span>
          )}
        </div>

        {!placebo.executed ? (
          <p className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-xs text-slate-400">
            {placebo.reason_skipped || "Teste placebo não executado."}
          </p>
        ) : (
          <div className="space-y-4">
            <div
              className={`rounded-xl border px-4 py-3 text-xs leading-relaxed ${
                placebo.passed
                  ? "border-emerald-500/30 bg-emerald-950/20 text-emerald-200"
                  : "border-rose-500/30 bg-rose-950/20 text-rose-200"
              }`}
            >
              {placebo.verdict}
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={placeboChartData} margin={{ top: 8, right: 16, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={26} />
                  <YAxis stroke="#94a3b8" fontSize={11} domain={["auto", "auto"]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "12px",
                      fontSize: "12px",
                    }}
                  />
                  {placebo.placebo_start && (
                    <ReferenceLine
                      x={placebo.placebo_start}
                      stroke="#a78bfa"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      label={{ value: "Intervenção Fictícia", fill: "#c4b5fd", fontSize: 11, position: "insideTopRight" }}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="band"
                    stroke="none"
                    fill="#a78bfa"
                    fillOpacity={0.18}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="Contrafactual Placebo"
                    stroke="#a78bfa"
                    strokeDasharray="6 4"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Realizado Pré"
                    stroke="#34d399"
                    strokeWidth={2.2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* 3. Coeficientes & Auditoria de Contaminação (No-Spillover) */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-400" />
          <h3 className="text-sm font-bold text-white">
            3. Coeficientes BSTS das Covariáveis & Auditoria No-Spillover
          </h3>
        </div>

        {covariates.length === 0 ? (
          <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-5 text-xs leading-relaxed text-slate-400">
            <p className="font-semibold text-slate-200">Modelo Univariado Ativo (0 covariáveis de controle)</p>
            <p className="mt-1">
              Sem covariáveis, o contrafactual é projetado apenas pelo componente estrutural da própria variável{" "}
              <code className="text-cyan-300">{analysis.y_col}</code>. Selecione uma ou mais séries de controle na barra
              lateral para habilitar a tabela de coeficientes e o teste de estabilidade No-Spillover.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
            <div className="h-60 lg:col-span-5">
              <p className="mb-2 text-xs text-slate-400">Peso Relativo Padronizado (% de contribuição no controle sintético)</p>
              <ResponsiveContainer width="100%" height="90%">
                <BarChart data={covChartData} layout="vertical" margin={{ top: 4, right: 20, left: 16, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" stroke="#94a3b8" fontSize={11} domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="name" stroke="#cbd5e1" fontSize={11} width={120} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "12px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="peso" name="Peso Relativo (%)" radius={[0, 6, 6, 0]}>
                    {covChartData.map((entry, index) => (
                      <Cell key={index} fill={entry.spillover ? "#fb7185" : "#22d3ee"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-x-auto lg:col-span-7">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                    <th className="py-2 pr-3">Covariável</th>
                    <th className="py-2 pr-3 text-right">β Real</th>
                    <th className="py-2 pr-3 text-right">β Pad.</th>
                    <th className="py-2 pr-3 text-right">p-valor</th>
                    <th className="py-2 pr-3 text-right">Corr. Pré</th>
                    <th className="py-2 pr-3 text-right">Desvio Pós</th>
                    <th className="py-2">Auditoria No-Spillover</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {covariates.map((c) => (
                    <tr key={c.name} className="hover:bg-slate-800/30">
                      <td className="py-2.5 pr-3 font-mono font-semibold text-slate-200">{c.name}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-cyan-300">{c.beta_raw.toFixed(3)}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-slate-300">{c.beta_std.toFixed(3)}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-slate-300">{c.p_value.toFixed(4)}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-emerald-300">{c.pre_correlation.toFixed(3)}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-slate-300">
                        {c.post_z_shift >= 0 ? "+" : ""}
                        {c.post_z_shift.toFixed(2)}σ
                      </td>
                      <td className="py-2.5">
                        <span
                          className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-bold ${
                            c.spillover_warning
                              ? "bg-rose-500/20 text-rose-300"
                              : "bg-emerald-500/15 text-emerald-300"
                          }`}
                        >
                          {c.status_label}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* 4. Decomposição Estrutural do Contrafactual */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex items-center gap-2">
          <Layers className="h-4 w-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white">
            4. Decomposição Estrutural: O que compõe o Contrafactual?
          </h3>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 8, right: 16, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={30} />
              <YAxis stroke="#94a3b8" fontSize={11} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  borderRadius: "12px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <ReferenceLine x={analysis.metrics.post_start} stroke="#f59e0b" strokeDasharray="5 5" />
              <Line type="monotone" dataKey="level_component" name="Nível Estrutural" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="covariates_component" name="Contribuição Covariáveis (Xβ)" stroke="#34d399" strokeWidth={2} dot={false} isAnimationActive={false} />
              {analysis.model_type === "seasonal_local_level" && (
                <Line type="monotone" dataKey="seasonal_component" name="Componente Sazonal" stroke="#fbbf24" strokeWidth={2} dot={false} isAnimationActive={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
