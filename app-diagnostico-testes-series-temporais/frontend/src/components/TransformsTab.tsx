import React from "react";
import { Check, Layers, Sparkles } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisPayload, TransformType } from "../types";

interface TransformsTabProps {
  data: AnalysisPayload;
  onApplyTransform: (t: TransformType) => void;
}

const fmtP = (v: number | null) => {
  if (v === null) return "—";
  if (v < 0.0001) return "< 0,0001";
  return v.toFixed(4).replace(".", ",");
};

export const TransformsTab: React.FC<TransformsTabProps> = ({ data, onApplyTransform }) => {
  const { transform_comparison: rows, meta, series_table: rowsTs } = data;

  return (
    <div className="space-y-6">
      {/* Matriz Comparativa das 7 Transformações */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              Laboratório Comparativo de Transformações (Estacionariedade & Variância)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Todas as 7 transformações avaliadas simultaneamente sobre a série bruta. Clique em{" "}
              <strong>Testar Agora</strong> para alternar o painel inteiro instantaneamente.
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-800 border border-indigo-200">
            Ativa: {meta.transform_label}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-4">Transformação</th>
                <th className="py-2.5 px-3 text-right">N</th>
                <th className="py-2.5 px-3 text-right">ADF p</th>
                <th className="py-2.5 px-3 text-right">PP p</th>
                <th className="py-2.5 px-3 text-right">KPSS p</th>
                <th className="py-2.5 px-3 text-right">ARCH p</th>
                <th className="py-2.5 px-3 text-right">Desvio-Padrão</th>
                <th className="py-2.5 px-3">Veredito</th>
                <th className="py-2.5 px-4 text-right">Ação</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((r) => {
                const isActive = r.key === meta.transform;
                const badgeCls =
                  r.status === "ok"
                    ? "bg-emerald-100 text-emerald-800"
                    : r.status === "warn"
                    ? "bg-amber-100 text-amber-800"
                    : r.status === "fail"
                    ? "bg-rose-100 text-rose-800"
                    : "bg-slate-100 text-slate-500";
                return (
                  <tr
                    key={r.key}
                    className={`${isActive ? "bg-indigo-50/70" : "hover:bg-slate-50"}`}
                  >
                    <td className="py-2.5 px-4 font-semibold text-slate-900">
                      <div className="flex items-center gap-2">
                        <span>{r.label}</span>
                        {r.recommended && (
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-600 text-white text-[10px] font-bold">
                            <Sparkles className="w-2.5 h-2.5" />
                            Recomendada
                          </span>
                        )}
                      </div>
                      {!r.available && r.reason && (
                        <p className="text-[10px] font-normal text-slate-400 mt-0.5">{r.reason}</p>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono">{r.available ? r.n_obs : "—"}</td>
                    <td className="py-2.5 px-3 text-right font-mono">{fmtP(r.adf_p)}</td>
                    <td className="py-2.5 px-3 text-right font-mono">{fmtP(r.pp_p)}</td>
                    <td className="py-2.5 px-3 text-right font-mono">{fmtP(r.kpss_p)}</td>
                    <td className="py-2.5 px-3 text-right font-mono">{fmtP(r.arch_p)}</td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      {r.std !== null ? r.std.toFixed(3) : "—"}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`inline-flex px-2 py-0.5 rounded-full text-[11px] font-semibold ${badgeCls}`}>
                        {r.consensus}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      {isActive ? (
                        <span className="inline-flex items-center gap-1 text-indigo-700 font-bold text-[11px]">
                          <Check className="w-3.5 h-3.5" /> Em uso
                        </span>
                      ) : (
                        <button
                          type="button"
                          disabled={!r.available}
                          onClick={() => onApplyTransform(r.key)}
                          className="px-2.5 py-1 rounded-md border border-indigo-300 bg-white hover:bg-indigo-600 hover:text-white disabled:opacity-40 text-indigo-700 font-semibold text-[11px] transition"
                        >
                          Testar Agora
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Gráficos: Série Original vs Transformada + Rolling Mean / Rolling Std */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-sm font-bold text-slate-900">
            Série Analisada & Média Móvel ({meta.rolling_window} períodos)
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Em séries estacionárias, a média móvel permanece aproximadamente horizontal ao longo do tempo.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rowsTs}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={30} />
                <YAxis tick={{ fontSize: 11 }} width={58} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="value"
                  name={meta.transform_label}
                  stroke="#3b82f6"
                  strokeWidth={1.8}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="rolling_mean"
                  name={`Média Móvel (${meta.rolling_window})`}
                  stroke="#f59e0b"
                  strokeWidth={2.4}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
          <h3 className="text-sm font-bold text-slate-900">
            Estabilidade de Variância — Desvio-Padrão Móvel ({meta.rolling_window} períodos)
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Tendência crescente indica heterocedasticidade multiplicativa (aplique Log/Box-Cox); picos agrupados revelam clusters GARCH.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rowsTs}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={30} />
                <YAxis tick={{ fontSize: 11 }} width={58} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="rolling_std"
                  name="Desvio-Padrão Móvel"
                  stroke="#e11d48"
                  strokeWidth={2.2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
