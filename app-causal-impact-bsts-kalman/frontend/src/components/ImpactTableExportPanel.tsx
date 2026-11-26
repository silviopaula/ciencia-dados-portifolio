import React, { useState } from "react";
import { Check, Copy, Download, FileSpreadsheet, FileText, Table2 } from "lucide-react";
import { getExportCsvUrl, getExportExcelUrl, getExportReportUrl } from "../api";
import type { AnalyzeResponse } from "../types";

interface ImpactTableExportPanelProps {
  analysis: AnalyzeResponse;
}

const FREQ_OPTIONS = [
  { id: "original", label: "Período Original" },
  { id: "month", label: "Mensal" },
  { id: "quarter", label: "Trimestral" },
  { id: "semester", label: "Semestral" },
  { id: "year", label: "Anual" },
];

function fmt(val: number, dec = 2): string {
  return val.toLocaleString("pt-BR", { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

export const ImpactTableExportPanel: React.FC<ImpactTableExportPanelProps> = ({ analysis }) => {
  const [freq, setFreq] = useState("month");
  const [copied, setCopied] = useState(false);

  const rows = analysis.aggregated_tables[freq] ?? [];

  const handleCopyReport = async () => {
    try {
      await navigator.clipboard.writeText(analysis.detailed_report_md);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignora
    }
  };

  return (
    <div className="space-y-5">
      {/* Barra de Frequência + Botões de Exportação */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 inline-flex items-center gap-1.5 text-xs font-semibold text-slate-400">
            <Table2 className="h-4 w-4 text-cyan-400" /> Agrupar por:
          </span>
          {FREQ_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => setFreq(opt.id)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                freq === opt.id
                  ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-400/40"
                  : "bg-slate-950 text-slate-400 hover:text-slate-200"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href={getExportExcelUrl()}
            download="relatorio_causal_impact.xlsx"
            className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 px-3.5 py-2 text-xs font-bold text-slate-950 transition hover:bg-emerald-400"
          >
            <FileSpreadsheet className="h-4 w-4" /> Exportar Excel (.xlsx)
          </a>
          <a
            href={getExportCsvUrl(freq)}
            download={`impacto_causal_${freq}.csv`}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-cyan-500/50 hover:text-cyan-200"
          >
            <Download className="h-3.5 w-3.5" /> CSV ({freq})
          </a>
          <a
            href={getExportReportUrl()}
            download="relatorio_causal_impact.md"
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-violet-500/50 hover:text-violet-200"
          >
            <FileText className="h-3.5 w-3.5" /> Relatório (.md)
          </a>
        </div>
      </div>

      {/* Tabela Agregada de Impactos Pós-Intervenção */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white">
            Tabela de Impacto Pós-Intervenção ({rows.length} período(s))
          </h3>
          <span className="text-xs text-slate-400">
            Variável: <code className="text-cyan-300">{analysis.y_col}</code>
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                <th className="py-2.5 pr-4">Período</th>
                <th className="py-2.5 pr-4 text-right">Obs.</th>
                <th className="py-2.5 pr-4 text-right">Realizado</th>
                <th className="py-2.5 pr-4 text-right">Contrafactual</th>
                <th className="py-2.5 pr-4 text-right">Impacto Abs.</th>
                <th className="py-2.5 pr-4 text-right">Impacto %</th>
                <th className="py-2.5 pr-4 text-right">IC (Inf / Sup)</th>
                <th className="py-2.5 text-right">Acumulado</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {rows.map((row) => {
                const pos = row.abs_effect_sum >= 0;
                return (
                  <tr key={row.period_label} className="hover:bg-slate-800/30">
                    <td className="py-2.5 pr-4 font-mono font-semibold text-white">{row.period_label}</td>
                    <td className="py-2.5 pr-4 text-right font-mono text-slate-400">{row.n_obs}</td>
                    <td className="py-2.5 pr-4 text-right font-mono text-slate-200">{fmt(row.actual_sum)}</td>
                    <td className="py-2.5 pr-4 text-right font-mono text-cyan-300">{fmt(row.predicted_sum)}</td>
                    <td
                      className={`py-2.5 pr-4 text-right font-mono font-bold ${
                        pos ? "text-emerald-300" : "text-rose-300"
                      }`}
                    >
                      {pos ? "+" : ""}
                      {fmt(row.abs_effect_sum)}
                    </td>
                    <td
                      className={`py-2.5 pr-4 text-right font-mono font-semibold ${
                        pos ? "text-emerald-300" : "text-rose-300"
                      }`}
                    >
                      {pos ? "+" : ""}
                      {fmt(row.rel_effect_pct)}%
                    </td>
                    <td className="py-2.5 pr-4 text-right font-mono text-[11px] text-slate-400">
                      [{fmt(row.lower_effect_sum, 1)}; {fmt(row.upper_effect_sum, 1)}]
                    </td>
                    <td className="py-2.5 text-right font-mono font-bold text-violet-300">
                      {row.cum_effect >= 0 ? "+" : ""}
                      {fmt(row.cum_effect)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Relatório Técnico Markdown */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white">Relatório Técnico Completo (Pronto para Documentação)</h3>
          <button
            type="button"
            onClick={handleCopyReport}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-cyan-500/50"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copiado!" : "Copiar Markdown"}
          </button>
        </div>
        <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-300">
          {analysis.detailed_report_md}
        </pre>
      </div>
    </div>
  );
};
