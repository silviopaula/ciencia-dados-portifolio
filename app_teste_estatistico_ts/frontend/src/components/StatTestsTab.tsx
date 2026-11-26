import React from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  GitBranch,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisPayload, TransformType } from "../types";

interface StatTestsTabProps {
  data: AnalysisPayload;
  onApplyTransform: (t: TransformType) => void;
}

const BANNER_STYLES = {
  emerald: "bg-emerald-50 border-emerald-200 text-emerald-950",
  amber: "bg-amber-50 border-amber-200 text-amber-950",
  rose: "bg-rose-50 border-rose-200 text-rose-950",
  indigo: "bg-indigo-50 border-indigo-200 text-indigo-950",
};

const BADGE_STYLES = {
  emerald: "bg-emerald-600 text-white",
  amber: "bg-amber-600 text-white",
  rose: "bg-rose-600 text-white",
  indigo: "bg-indigo-600 text-white",
};

export const StatTestsTab: React.FC<StatTestsTabProps> = ({ data, onApplyTransform }) => {
  const { stationarity_consensus: cons, tests_summary: tests, meta } = data;
  const recommendedRow = data.transform_comparison.find((r) => r.recommended);
  const canJump =
    recommendedRow && recommendedRow.available && recommendedRow.key !== meta.transform;

  const passedCount = tests.filter((t) => t.passed).length;

  return (
    <div className="space-y-6">
      {/* Banner Consenso Cruzado 4-Quadrantes */}
      <div className={`rounded-xl border p-5 ${BANNER_STYLES[cons.badge_color]}`}>
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${BADGE_STYLES[cons.badge_color]}`}
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                Veredito Cruzado (ADF × PP × KPSS × Zivot-Andrews)
              </span>
              <span className="text-xs font-semibold opacity-75">
                {passedCount}/{tests.length} diagnósticos aprovados (α = {(meta.alpha * 100).toFixed(0)}%)
              </span>
            </div>
            <h2 className="text-lg font-bold">{cons.regime_title}</h2>
            <p className="text-sm opacity-90 max-w-3xl leading-relaxed">{cons.summary}</p>
            <p className="text-xs font-semibold pt-1">
              Recomendação: <span className="font-normal">{cons.recommendation}</span>
            </p>
          </div>

          {canJump && recommendedRow && (
            <button
              type="button"
              onClick={() => onApplyTransform(recommendedRow.key)}
              className="shrink-0 inline-flex items-center gap-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white px-3.5 py-2 text-xs font-semibold shadow-sm transition self-start"
            >
              Aplicar {recommendedRow.label}
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Grid dos 10 Cards de Testes */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-3">
        {tests.map((t) => (
          <div
            key={t.id}
            className={`rounded-xl border bg-white p-3.5 shadow-sm transition hover:shadow-md ${
              t.passed ? "border-emerald-200/80" : "border-rose-200/80"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                {t.category}
              </span>
              {t.passed ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              ) : (
                <XCircle className="w-4 h-4 text-rose-500 shrink-0" />
              )}
            </div>
            <h3 className="text-xs font-bold text-slate-900 mt-1 line-clamp-1" title={t.name}>
              {t.name}
            </h3>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="text-lg font-extrabold text-slate-900 font-mono">
                p = {t.p_formatted}
              </span>
              <span className="text-[11px] font-mono text-slate-500">
                stat {t.statistic ?? "—"}
              </span>
            </div>
            <div className="mt-2 flex items-center justify-between gap-1">
              <span
                className={`inline-flex px-2 py-0.5 rounded text-[11px] font-semibold ${
                  t.passed ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
                }`}
              >
                {t.status_label}
              </span>
            </div>
            {t.extra_note && (
              <p className="text-[10px] text-slate-500 mt-1.5 truncate" title={t.extra_note}>
                {t.extra_note}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Tabela Completa + Especificações ADF/KPSS */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Tabela Completa da Bateria Econométrica (10 Testes)
            </h3>
            <p className="text-xs text-slate-500">
              Modelo diagnóstico dos resíduos: <strong>{data.arima.spec_label}</strong> •
              Durbin-Watson: <strong>{data.durbin_watson ?? "—"}</strong>
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px]">
            {(["c", "ct", "n"] as const).map((spec) => {
              const s = data.adf_specifications[spec];
              return (
                <span
                  key={spec}
                  className="px-2 py-1 rounded bg-slate-100 text-slate-700 font-mono"
                >
                  ADF({spec}): p={s.p_value !== null ? s.p_value.toFixed(4) : "—"}
                </span>
              );
            })}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-4">Categoria</th>
                <th className="py-2.5 px-3">Teste</th>
                <th className="py-2.5 px-3">Alvo</th>
                <th className="py-2.5 px-3 text-right">Estatística</th>
                <th className="py-2.5 px-3 text-right">p-valor</th>
                <th className="py-2.5 px-3 text-right">Crítico 5%</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-4">Interpretação</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {tests.map((t) => (
                <tr key={t.id} className="hover:bg-slate-50/70">
                  <td className="py-2.5 px-4 font-medium text-slate-500">{t.category}</td>
                  <td className="py-2.5 px-3 font-semibold text-slate-900">{t.name}</td>
                  <td className="py-2.5 px-3 text-slate-600">{t.target_series}</td>
                  <td className="py-2.5 px-3 text-right font-mono">{t.statistic ?? "—"}</td>
                  <td className="py-2.5 px-3 text-right font-mono font-semibold">
                    {t.p_formatted}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-500">
                    {t.critical_5pct ?? "—"}
                  </td>
                  <td className="py-2.5 px-3">
                    <span
                      className={`inline-flex px-2 py-0.5 rounded-full text-[11px] font-semibold ${
                        t.passed ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
                      }`}
                    >
                      {t.status_label}
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-slate-600 max-w-md">{t.interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ACF e PACF + Box-Jenkins */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
              <GitBranch className="w-4 h-4 text-indigo-600" />
              Correlogramas ACF & PACF ({meta.transform_label})
            </h3>
            <p className="text-xs text-slate-500">
              Bandas de Bartlett 95%: ±{data.acf_pacf[0]?.upper?.toFixed(3) ?? "0.15"} (barras em destaque são estatisticamente significativas).
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="border border-slate-100 rounded-lg p-3 bg-slate-50/40">
            <h4 className="text-xs font-bold text-slate-700 mb-2">
              Função de Autocorrelação (ACF)
            </h4>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.acf_pacf}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="lag" tick={{ fontSize: 11 }} />
                  <YAxis domain={[-1, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    formatter={(v: number) => [v.toFixed(4), "ACF"]}
                    labelFormatter={(l) => `Lag ${l}`}
                  />
                  <ReferenceLine y={0} stroke="#64748b" />
                  {data.acf_pacf[0]?.upper && (
                    <>
                      <ReferenceLine y={data.acf_pacf[0].upper} stroke="#ef4444" strokeDasharray="4 4" />
                      <ReferenceLine y={data.acf_pacf[0].lower!} stroke="#ef4444" strokeDasharray="4 4" />
                    </>
                  )}
                  <Bar dataKey="acf" radius={[3, 3, 0, 0]}>
                    {data.acf_pacf.map((entry) => (
                      <Cell
                        key={`acf-${entry.lag}`}
                        fill={entry.acf_sig ? "#2563eb" : "#93c5fd"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="border border-slate-100 rounded-lg p-3 bg-slate-50/40">
            <h4 className="text-xs font-bold text-slate-700 mb-2">
              Autocorrelação Parcial (PACF)
            </h4>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.acf_pacf}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="lag" tick={{ fontSize: 11 }} />
                  <YAxis domain={[-1, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    formatter={(v: number) => [v.toFixed(4), "PACF"]}
                    labelFormatter={(l) => `Lag ${l}`}
                  />
                  <ReferenceLine y={0} stroke="#64748b" />
                  {data.acf_pacf[0]?.upper && (
                    <>
                      <ReferenceLine y={data.acf_pacf[0].upper} stroke="#ef4444" strokeDasharray="4 4" />
                      <ReferenceLine y={data.acf_pacf[0].lower!} stroke="#ef4444" strokeDasharray="4 4" />
                    </>
                  )}
                  <Bar dataKey="pacf" radius={[3, 3, 0, 0]}>
                    {data.acf_pacf.map((entry) => (
                      <Cell
                        key={`pacf-${entry.lag}`}
                        fill={entry.pacf_sig ? "#059669" : "#6ee7b7"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-sky-200 bg-sky-50/70 p-3.5 text-xs text-sky-950 flex items-start gap-2.5">
          <Activity className="w-4 h-4 text-sky-700 shrink-0 mt-0.5" />
          <div>
            <strong className="font-semibold">Leitura Automática Box-Jenkins: </strong>
            {data.box_jenkins_hint}
          </div>
        </div>
      </div>

      {/* Curva de p-valores Ljung-Box por Lag */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Estabilidade de Ruído Branco — Curva de p-valores Ljung-Box (Resíduos {data.arima.spec_label})
            </h3>
            <p className="text-xs text-slate-500">
              Pontos <strong>acima</strong> da linha vermelha (α = {meta.alpha}) confirmam ausência de autocorrelação residual naquele horizonte de lags.
            </p>
          </div>
          {data.ljung_box_curve.some((p) => (p.lb_p ?? 1) < meta.alpha) && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-800">
              <AlertTriangle className="w-3.5 h-3.5" />
              Algum lag abaixo de α
            </span>
          )}
        </div>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.ljung_box_curve}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="lag" tick={{ fontSize: 11 }} label={{ value: "Lag", position: "insideBottomRight", offset: -4, fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(v: number) => [v.toFixed(4), "Ljung-Box p-valor"]}
                labelFormatter={(l) => `Lag ${l}`}
              />
              <ReferenceLine
                y={meta.alpha}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{ value: `α = ${meta.alpha}`, position: "right", fill: "#dc2626", fontSize: 11 }}
              />
              <Line
                type="monotone"
                dataKey="lb_p"
                stroke="#4f46e5"
                strokeWidth={2.5}
                dot={{ r: 3.5, fill: "#4f46e5" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
