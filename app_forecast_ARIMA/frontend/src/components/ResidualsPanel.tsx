import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CheckCircle2, AlertTriangle, Sigma } from "lucide-react";
import type { ForecastResponse } from "../types";

interface Props {
  data: ForecastResponse;
}

export const ResidualsPanel: React.FC<Props> = ({ data }) => {
  const rd = data.residuals_diagnostic;

  return (
    <div className="space-y-6">
      {/* Diagnóstico de Ruído Branco (Ljung-Box) */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="text-sm font-bold text-white">
              Teste de Ljung-Box (Diagnóstico de Ruído Branco nos Resíduos — Lag {rd.ljung_box_lag})
            </h3>
            <p className="mt-1 text-xs leading-relaxed text-slate-300">{rd.interpretation}</p>
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <div className="text-right">
              <div className="text-[11px] text-slate-400">Estatística Q / P-Valor</div>
              <div className="font-mono text-sm font-bold text-white">
                {rd.ljung_box_stat} <span className="text-indigo-300">(p = {rd.ljung_box_pvalue})</span>
              </div>
            </div>
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
                rd.is_white_noise
                  ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                  : "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30"
              }`}
            >
              {rd.is_white_noise ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : (
                <AlertTriangle className="h-3.5 w-3.5" />
              )}
              {rd.is_white_noise ? "Ruído Branco" : "Autocorrelação Residual"}
            </span>
          </div>
        </div>

        <div className="mt-5 h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.residuals_series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={35} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} width={60} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: 12 }}
              />
              <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="residual" name="Resíduo" stroke="#38bdf8" strokeWidth={1.6} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tabela de Coeficientes Estimados */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
        <div className="mb-4 flex items-center gap-2">
          <Sigma className="h-4 w-4 text-indigo-400" />
          <div>
            <h3 className="text-sm font-bold text-white">Parâmetros Estimados (Máxima Verossimilhança — MLE)</h3>
            <p className="text-xs text-slate-400">
              Coeficientes com p-valor &lt; 0,05 são estatisticamente significativos ao nível de 5%.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 pr-4">Parâmetro</th>
                <th className="py-2.5 pr-4">Estimativa</th>
                <th className="py-2.5 pr-4">Erro-Padrão</th>
                <th className="py-2.5 pr-4">Estatística z</th>
                <th className="py-2.5 pr-4">P-Valor</th>
                <th className="py-2.5 pr-4">IC 95%</th>
                <th className="py-2.5">Significância</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {data.coefficients.map((c) => (
                <tr key={c.name} className="hover:bg-slate-800/40">
                  <td className="py-2.5 pr-4 font-mono font-semibold text-slate-100">{c.name}</td>
                  <td className="py-2.5 pr-4 font-mono text-white">{c.estimate.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-400">{c.std_err.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-300">{c.z_stat.toFixed(2)}</td>
                  <td className="py-2.5 pr-4 font-mono text-indigo-300">{c.p_value.toFixed(4)}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-400">
                    [{c.ci_lower.toFixed(3)}, {c.ci_upper.toFixed(3)}]
                  </td>
                  <td className="py-2.5">
                    {c.significant ? (
                      <span className="rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300">
                        Significativo (5%)
                      </span>
                    ) : (
                      <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[11px] text-slate-400">
                        Não Sig.
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
