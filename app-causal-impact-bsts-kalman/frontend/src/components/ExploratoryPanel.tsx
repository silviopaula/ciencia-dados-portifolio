import React, { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3, LineChart as LineIcon } from "lucide-react";
import type { AnalyzeResponse, DatasetSummary } from "../types";

interface ExploratoryPanelProps {
  dataset: DatasetSummary;
  analysis: AnalyzeResponse;
}

const PALETTE = ["#34d399", "#22d3ee", "#a78bfa", "#fbbf24", "#fb7185", "#60a5fa"];

export const ExploratoryPanel: React.FC<ExploratoryPanelProps> = ({ dataset, analysis }) => {
  const allVars = useMemo(() => [analysis.y_col, ...analysis.x_cols], [analysis.y_col, analysis.x_cols]);
  const [selectedVars, setSelectedVars] = useState<string[]>(allVars);
  const [normalizeBase100, setNormalizeBase100] = useState(true);

  const visibleVars = selectedVars.filter((v) => allVars.includes(v));
  const activeVars = visibleVars.length > 0 ? visibleVars : [analysis.y_col];

  const chartData = useMemo(() => {
    const firstRow = analysis.raw_series[0];
    return analysis.raw_series.map((row) => {
      const point: Record<string, string | number> = { date: row.date };
      for (const v of activeVars) {
        const val = row.values[v] ?? 0;
        if (normalizeBase100 && firstRow && Math.abs(firstRow.values[v] ?? 0) > 1e-9) {
          point[v] = Number(((val / firstRow.values[v]) * 100).toFixed(2));
        } else {
          point[v] = val;
        }
      }
      return point;
    });
  }, [analysis.raw_series, activeVars, normalizeBase100]);

  const toggleVar = (v: string) => {
    setSelectedVars((prev) =>
      prev.includes(v) ? (prev.length > 1 ? prev.filter((item) => item !== v) : prev) : [...prev, v]
    );
  };

  return (
    <div className="space-y-5">
      {/* Gráfico Comparativo Multi-Séries */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <LineIcon className="h-4 w-4 text-cyan-400" />
            <div>
              <h3 className="text-sm font-bold text-white">
                Comparação Visual: Variável Tratada (Y) vs. Controles (X)
              </h3>
              <p className="text-xs text-slate-400">
                No período pré-intervenção, bons controles acompanham o movimento de Y (tendências paralelas / co-movimento).
              </p>
            </div>
          </div>

          <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-200">
            <input
              type="checkbox"
              checked={normalizeBase100}
              onChange={(e) => setNormalizeBase100(e.target.checked)}
              className="accent-cyan-400"
            />
            Normalizar em Base 100
          </label>
        </div>

        <div className="mb-3 flex flex-wrap gap-2">
          {allVars.map((v, i) => {
            const on = activeVars.includes(v);
            return (
              <button
                key={v}
                type="button"
                onClick={() => toggleVar(v)}
                className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                  on
                    ? "bg-slate-800 text-white ring-1 ring-slate-600"
                    : "bg-slate-950 text-slate-500 hover:text-slate-300"
                }`}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                />
                {v} {v === analysis.y_col ? "(Y)" : "(X)"}
              </button>
            );
          })}
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={28} />
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
              <ReferenceLine
                x={analysis.metrics.post_start}
                stroke="#f59e0b"
                strokeDasharray="5 5"
                strokeWidth={2}
              />
              {activeVars.map((v, idx) => (
                <Line
                  key={v}
                  type="monotone"
                  dataKey={v}
                  stroke={PALETTE[idx % PALETTE.length]}
                  strokeWidth={v === analysis.y_col ? 2.6 : 1.7}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Estatísticas Descritivas Pré vs Pós */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white">Estatísticas Descritivas por Período (Pré vs. Pós vs. Total)</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                <th className="py-2 pr-3">Variável</th>
                <th className="py-2 pr-3">Período</th>
                <th className="py-2 pr-3 text-right">N</th>
                <th className="py-2 pr-3 text-right">Média</th>
                <th className="py-2 pr-3 text-right">Mediana</th>
                <th className="py-2 pr-3 text-right">Desvio-Padrão</th>
                <th className="py-2 pr-3 text-right">Mínimo</th>
                <th className="py-2 text-right">Máximo</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {analysis.descriptive_stats.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30">
                  <td className="py-2 pr-3 font-mono font-semibold text-slate-200">{row.variable}</td>
                  <td className="py-2 pr-3">
                    <span
                      className={`rounded px-2 py-0.5 text-[10px] font-semibold ${
                        row.period === "Pré-Intervenção"
                          ? "bg-cyan-500/15 text-cyan-300"
                          : row.period === "Pós-Intervenção"
                            ? "bg-emerald-500/15 text-emerald-300"
                            : "bg-slate-800 text-slate-300"
                      }`}
                    >
                      {row.period}
                    </span>
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-slate-400">{row.n}</td>
                  <td className="py-2 pr-3 text-right font-mono text-white">{row.mean.toFixed(2)}</td>
                  <td className="py-2 pr-3 text-right font-mono text-slate-300">{row.median.toFixed(2)}</td>
                  <td className="py-2 pr-3 text-right font-mono text-slate-300">{row.std.toFixed(2)}</td>
                  <td className="py-2 pr-3 text-right font-mono text-slate-400">{row.min_val.toFixed(2)}</td>
                  <td className="py-2 text-right font-mono text-slate-400">{row.max_val.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Prévia da Base */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <h3 className="mb-3 text-sm font-bold text-white">
          Prévia da Base Carregada ({dataset.rows} linhas • Frequência inferida: {dataset.inferred_frequency})
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                {dataset.columns.map((col) => (
                  <th key={col} className="py-2 pr-4">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {dataset.preview.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30">
                  {dataset.columns.map((col) => (
                    <td key={col} className="py-2 pr-4 text-slate-300">
                      {row[col] === null || row[col] === undefined ? "—" : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
