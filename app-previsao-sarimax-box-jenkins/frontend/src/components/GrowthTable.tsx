import React from "react";
import { Download, ArrowUpRight, ArrowDownRight } from "lucide-react";
import type { AnnualGrowthItem } from "../types";
import { getExportExcelUrl } from "../api";

interface Props {
  datasetId: string;
  rows: AnnualGrowthItem[];
}

const fmt = (n: number) => n.toLocaleString("pt-BR", { maximumFractionDigits: 1 });

export const GrowthTable: React.FC<Props> = ({ datasetId, rows }) => {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-sm font-bold text-white">Análise de Crescimento Anual (YoY) &amp; Exportação</h3>
          <p className="mt-1 text-xs text-slate-400">
            Consolidação anual (média e acumulado) integrando o histórico recente e o horizonte projetado pelo modelo.
          </p>
        </div>
        <a
          href={getExportExcelUrl(datasetId)}
          download="relatorio_sarimax_forecast.xlsx"
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-emerald-950/50 transition hover:bg-emerald-500"
        >
          <Download className="h-4 w-4" />
          Exportar Relatório Completo (.xlsx)
        </a>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 pr-4">Ano</th>
                <th className="py-2.5 pr-4">Origem</th>
                <th className="py-2.5 pr-4">Meses</th>
                <th className="py-2.5 pr-4">Média Mensal</th>
                <th className="py-2.5 pr-4">Acumulado Anual</th>
                <th className="py-2.5">Crescimento YoY (Média)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {rows.map((r) => (
                <tr key={r.year} className="hover:bg-slate-800/40">
                  <td className="py-2.5 pr-4 font-mono text-sm font-bold text-white">{r.year}</td>
                  <td className="py-2.5 pr-4">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                        r.source === "Projeção"
                          ? "bg-emerald-500/15 text-emerald-300"
                          : r.source === "Misto"
                          ? "bg-amber-500/15 text-amber-300"
                          : "bg-slate-800 text-slate-300"
                      }`}
                    >
                      {r.source}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 text-slate-300">{r.months_count}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-200">{fmt(r.mean_value)}</td>
                  <td className="py-2.5 pr-4 font-mono text-slate-300">{fmt(r.total_value)}</td>
                  <td className="py-2.5">
                    {r.yoy_growth_pct == null ? (
                      <span className="text-slate-500">—</span>
                    ) : (
                      <span
                        className={`inline-flex items-center gap-0.5 font-mono font-semibold ${
                          r.yoy_growth_pct >= 0 ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {r.yoy_growth_pct >= 0 ? (
                          <ArrowUpRight className="h-3.5 w-3.5" />
                        ) : (
                          <ArrowDownRight className="h-3.5 w-3.5" />
                        )}
                        {r.yoy_growth_pct >= 0 ? "+" : ""}
                        {r.yoy_growth_pct.toFixed(2)}%
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
