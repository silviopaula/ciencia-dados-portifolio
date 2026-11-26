import React from "react";
import { AlertOctagon, BarChart3 } from "lucide-react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisPayload } from "../types";

interface VisualsDistTabProps {
  data: AnalysisPayload;
}

export const VisualsDistTab: React.FC<VisualsDistTabProps> = ({ data }) => {
  const { descriptive_stats: d, outliers, distributions, series_table, zivot_andrews: za, meta } = data;

  const statCards = [
    { label: "Observações (N)", value: String(d.n) },
    { label: "Média", value: d.mean?.toFixed(3) ?? "—" },
    { label: "Mediana (Q2)", value: d.median?.toFixed(3) ?? "—" },
    { label: "Desvio-Padrão", value: d.std?.toFixed(3) ?? "—" },
    { label: "Assimetria (Skew)", value: d.skewness?.toFixed(3) ?? "—" },
    { label: "Curtose Excesso", value: d.kurtosis?.toFixed(3) ?? "—" },
    { label: "Outliers (IQR / MAD)", value: `${outliers.iqr_count} / ${outliers.mad_count}` },
  ];

  return (
    <div className="space-y-6">
      {/* Estatísticas Descritivas */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {statCards.map((c) => (
          <div key={c.label} className="bg-white rounded-xl border border-slate-200 p-3 shadow-sm">
            <div className="text-[11px] font-medium text-slate-500">{c.label}</div>
            <div className="text-base font-extrabold text-slate-900 font-mono mt-1">{c.value}</div>
          </div>
        ))}
      </div>

      {/* Série Completa + Outliers + Linha de Quebra Estrutural ZA */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Série Temporal ({meta.transform_label}), Outliers (IQR/MAD) & Quebra Zivot-Andrews
            </h3>
            <p className="text-xs text-slate-500">
              Pontos vermelhos destacam anomalias (IQR 1,5× ou Z-MAD &gt; 3,5). Linha tracejada âmbar sinaliza a principal quebra candidata de Zivot-Andrews ({za.break_date ?? "—"}).
            </p>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={series_table}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={30} />
              <YAxis tick={{ fontSize: 11 }} width={58} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {za.break_date && (
                <ReferenceLine
                  x={za.break_date}
                  stroke="#d97706"
                  strokeDasharray="5 5"
                  label={{ value: `ZA: ${za.break_date}`, position: "top", fill: "#b45309", fontSize: 10 }}
                />
              )}
              <Line
                type="monotone"
                dataKey="value"
                name={meta.transform_label}
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
              />
              <Scatter dataKey="outlier_value" name="Outlier Detectado" fill="#ef4444" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Histogramas (Série e Resíduos) + Normal Q-Q Plot */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5 mb-2">
            <BarChart3 className="w-3.5 h-3.5 text-indigo-600" />
            Distribuição dos Valores vs Normal
          </h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={distributions.values_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="bin_label" tick={{ fontSize: 9 }} interval={2} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Frequência" fill="#818cf8" radius={[3, 3, 0, 0]} />
                <Line
                  type="monotone"
                  dataKey="normal_expected"
                  name="Normal Teórica"
                  stroke="#1e293b"
                  strokeWidth={2}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <h4 className="text-xs font-bold text-slate-900 mb-2">
            Resíduos ({data.arima.spec_label}) vs Normal
          </h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={distributions.residuals_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="bin_label" tick={{ fontSize: 9 }} interval={2} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Resíduos" fill="#34d399" radius={[3, 3, 0, 0]} />
                <Line
                  type="monotone"
                  dataKey="normal_expected"
                  name="Normal Teórica"
                  stroke="#dc2626"
                  strokeWidth={2}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
          <h4 className="text-xs font-bold text-slate-900 mb-2">
            Normal Q-Q Plot (Resíduos Padronizados)
          </h4>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={distributions.residuals_qq}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="theoretical" type="number" domain={["auto", "auto"]} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Line
                  type="linear"
                  dataKey="ref_line"
                  name="Referência 45°"
                  stroke="#ef4444"
                  strokeDasharray="4 4"
                  strokeWidth={1.8}
                  dot={false}
                />
                <Scatter dataKey="sample" name="Quantis Amostrais" fill="#4f46e5" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Tabela de Outliers */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <AlertOctagon className="w-4 h-4 text-rose-600" />
            Auditoria de Outliers ({outliers.total_flagged} pontos sinalizados)
          </h3>
          <span className="text-xs text-slate-500 font-mono">
            Limites IQR: [{outliers.iqr_low ?? "—"}, {outliers.iqr_high ?? "—"}]
          </span>
        </div>
        {outliers.top_outliers.length === 0 ? (
          <div className="p-5 text-xs text-emerald-800 bg-emerald-50/50">
            Nenhum outlier extremo detectado pelos critérios IQR (1,5×) ou Z-Score Robusto MAD (&gt; 3,5).
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-2 px-4">Data</th>
                  <th className="py-2 px-3 text-right">Valor</th>
                  <th className="py-2 px-3 text-right">Z-Score Robusto (MAD)</th>
                  <th className="py-2 px-4">Critério Disparado</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {outliers.top_outliers.map((o) => (
                  <tr key={o.date} className="hover:bg-slate-50">
                    <td className="py-2 px-4 font-mono font-semibold text-slate-800">{o.date}</td>
                    <td className="py-2 px-3 text-right font-mono">{o.value}</td>
                    <td className="py-2 px-3 text-right font-mono font-bold text-rose-600">
                      {o.z_mad}
                    </td>
                    <td className="py-2 px-4">
                      <span className="px-2 py-0.5 rounded bg-rose-100 text-rose-800 text-[11px] font-semibold">
                        {o.methods}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
