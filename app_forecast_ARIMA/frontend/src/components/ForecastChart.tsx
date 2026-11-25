import React, { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrendingUp, Award, Layers } from "lucide-react";
import type { ForecastResponse } from "../types";

interface Props {
  data: ForecastResponse;
}

const fmtNum = (n: number | null | undefined) =>
  n == null ? "—" : n.toLocaleString("pt-BR", { maximumFractionDigits: 2 });

export const ForecastChart: React.FC<Props> = ({ data }) => {
  const [windowMode, setWindowMode] = useState<"recent" | "all">("recent");

  const chartData = useMemo(() => {
    const raw = data.series.map((pt) => ({
      ...pt,
      ciRange:
        pt.lower_ci != null && pt.upper_ci != null ? [pt.lower_ci, pt.upper_ci] : null,
      backtestLine: pt.split === "test" ? pt.fitted_or_backtest : null,
      trainFitted: pt.split === "train" ? pt.fitted_or_backtest : null,
    }));
    if (windowMode === "recent" && raw.length > 96) {
      return raw.slice(-96);
    }
    return raw;
  }, [data.series, windowMode]);

  const firstTestDate = data.series.find((p) => p.split === "test")?.date;
  const firstFutureDate = data.series.find((p) => p.split === "future")?.date;

  return (
    <div className="space-y-6">
      {/* Resumo Executivo */}
      <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-r from-emerald-950/60 via-slate-900 to-slate-900 p-5">
        <div className="flex items-start gap-3">
          <div className="rounded-xl bg-emerald-500/20 p-2.5 text-emerald-300">
            <Award className="h-5 w-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-bold text-white">{data.model_label}</h2>
              {data.use_log && (
                <span className="rounded-full bg-indigo-500/20 px-2.5 py-0.5 text-[11px] font-semibold text-indigo-300">
                  Escala Log (Duan Smearing)
                </span>
              )}
              {data.exog_cols.length > 0 && (
                <span className="rounded-full bg-cyan-500/20 px-2.5 py-0.5 text-[11px] font-semibold text-cyan-300">
                  {data.exog_cols.length} Exógena(s)
                </span>
              )}
            </div>
            <p className="mt-1.5 text-sm text-slate-200">{data.executive_takeaway}</p>
          </div>
        </div>
      </div>

      {/* Cards KPI de Performance Out-of-Sample */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
          <span className="text-xs font-medium text-slate-400">MAPE (Teste)</span>
          <div className="mt-1 text-2xl font-extrabold text-emerald-400">
            {data.test_metrics.mape.toFixed(2)}%
          </div>
          <span className="mt-1 block text-[11px] text-slate-400">
            Naive: {data.baseline_metrics.mape.toFixed(2)}% ({data.mape_improvement_pct >= 0 ? "-" : "+"}
            {Math.abs(data.mape_improvement_pct).toFixed(1)}%)
          </span>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
          <span className="text-xs font-medium text-slate-400">RMSE (Teste)</span>
          <div className="mt-1 text-2xl font-extrabold text-white">
            {fmtNum(data.test_metrics.rmse)}
          </div>
          <span className="mt-1 block text-[11px] text-slate-400">
            Naive: {fmtNum(data.baseline_metrics.rmse)}
          </span>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
          <span className="text-xs font-medium text-slate-400">MAE (Teste)</span>
          <div className="mt-1 text-2xl font-extrabold text-white">
            {fmtNum(data.test_metrics.mae)}
          </div>
          <span className="mt-1 block text-[11px] text-slate-400">
            Naive: {fmtNum(data.baseline_metrics.mae)}
          </span>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
          <span className="text-xs font-medium text-slate-400">R² (Out-of-Sample)</span>
          <div className="mt-1 text-2xl font-extrabold text-indigo-300">
            {data.test_metrics.r2.toFixed(3)}
          </div>
          <span className="mt-1 block text-[11px] text-slate-400">
            Naive R²: {data.baseline_metrics.r2.toFixed(3)}
          </span>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 col-span-2 lg:col-span-1">
          <span className="text-xs font-medium text-slate-400">AIC / BIC</span>
          <div className="mt-1 text-xl font-bold text-slate-100">
            {data.test_metrics.aic?.toFixed(1) ?? "—"}
          </div>
          <span className="mt-1 block text-[11px] text-slate-400">
            BIC: {data.test_metrics.bic?.toFixed(1) ?? "—"}
          </span>
        </div>
      </div>

      {/* Gráfico Principal Interativo */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-indigo-400" />
            <div>
              <h3 className="text-sm font-bold text-white">
                Série Histórica, Backtest (Teste) e Projeção Futura com Intervalo de Confiança
              </h3>
              <p className="text-xs text-slate-400">
                Passe o mouse sobre qualquer ponto para comparar realizado, backtest, baseline e bandas de confiança.
              </p>
            </div>
          </div>
          <div className="flex rounded-xl border border-slate-700 bg-slate-950 p-1 text-xs">
            <button
              onClick={() => setWindowMode("recent")}
              className={`rounded-lg px-3 py-1 font-semibold transition ${
                windowMode === "recent" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              Zoom Recente
            </button>
            <button
              onClick={() => setWindowMode("all")}
              className={`rounded-lg px-3 py-1 font-semibold transition ${
                windowMode === "all" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-white"
              }`}
            >
              Série Completa
            </button>
          </div>
        </div>

        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={30} />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                width={72}
                tickFormatter={(v) => Number(v).toLocaleString("pt-BR")}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  borderRadius: 12,
                  fontSize: 12,
                }}
                formatter={(value: unknown, name: string) => {
                  if (Array.isArray(value)) {
                    return [`${fmtNum(Number(value[0]))} – ${fmtNum(Number(value[1]))}`, name];
                  }
                  return [fmtNum(Number(value)), name];
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />

              {firstTestDate && (
                <ReferenceLine
                  x={firstTestDate}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  label={{ value: "Início Teste", fill: "#f59e0b", fontSize: 11, position: "insideTopLeft" }}
                />
              )}
              {firstFutureDate && (
                <ReferenceLine
                  x={firstFutureDate}
                  stroke="#10b981"
                  strokeDasharray="4 4"
                  label={{ value: "Projeção Futura", fill: "#10b981", fontSize: 11, position: "insideTopRight" }}
                />
              )}

              <Area
                type="monotone"
                dataKey="ciRange"
                name="Intervalo de Confiança"
                stroke="none"
                fill="#10b981"
                fillOpacity={0.18}
              />
              <Line
                type="monotone"
                dataKey="actual"
                name="Realizado"
                stroke="#38bdf8"
                strokeWidth={2.3}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="backtestLine"
                name="Backtest SARIMAX"
                stroke="#f59e0b"
                strokeWidth={2.4}
                strokeDasharray="5 3"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="baseline_naive"
                name="Baseline Naive"
                stroke="#94a3b8"
                strokeWidth={1.6}
                strokeDasharray="2 3"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="forecast"
                name="Projeção Futura"
                stroke="#10b981"
                strokeWidth={2.6}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Ranking dos Candidatos Testados no Auto-SARIMAX */}
      {data.candidates_tested.length > 1 && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="mb-3 flex items-center gap-2">
            <Layers className="h-4 w-4 text-indigo-400" />
            <h3 className="text-sm font-semibold text-white">
              Seleção Automática de Ordem — Modelos Candidatos Comparados (Critério AIC)
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-4">Ordem (p,d,q)×(P,D,Q,s)</th>
                  <th className="py-2 pr-4">AIC (Menor é Melhor)</th>
                  <th className="py-2 pr-4">BIC</th>
                  <th className="py-2">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {data.candidates_tested.map((c) => (
                  <tr key={c.order} className={c.selected ? "bg-indigo-500/10" : ""}>
                    <td className="py-2 pr-4 font-mono font-semibold text-slate-100">{c.order}</td>
                    <td className="py-2 pr-4 font-mono text-slate-300">{c.aic.toFixed(2)}</td>
                    <td className="py-2 pr-4 font-mono text-slate-300">{c.bic.toFixed(2)}</td>
                    <td className="py-2">
                      {c.selected ? (
                        <span className="rounded-full bg-emerald-500/20 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300">
                          Selecionado
                        </span>
                      ) : (
                        <span className="text-slate-500">Avaliado</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
