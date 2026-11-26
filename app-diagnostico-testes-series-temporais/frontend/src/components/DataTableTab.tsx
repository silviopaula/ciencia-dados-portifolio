import React, { useMemo, useState } from "react";
import { Download, Search, Table2 } from "lucide-react";
import type { AnalysisPayload } from "../types";

interface DataTableTabProps {
  data: AnalysisPayload;
}

const PAGE_SIZE = 20;

export const DataTableTab: React.FC<DataTableTabProps> = ({ data }) => {
  const [query, setQuery] = useState("");
  const [onlyOutliers, setOnlyOutliers] = useState(false);
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    return data.series_table.filter((r) => {
      if (onlyOutliers && !r.is_outlier) return false;
      if (query.trim() && !r.date.includes(query.trim())) return false;
      return true;
    });
  }, [data.series_table, query, onlyOutliers]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const rows = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const handleDownloadCsv = () => {
    const header = [
      "date",
      "raw_value",
      "value",
      "stl_trend",
      "stl_seasonal",
      "stl_remainder",
      "fitted",
      "residual",
      "is_outlier",
    ];
    const lines = [
      header.join(";"),
      ...data.series_table.map((r) =>
        [
          r.date,
          r.raw_value ?? "",
          r.value ?? "",
          r.stl_trend ?? "",
          r.stl_seasonal ?? "",
          r.stl_remainder ?? "",
          r.fitted ?? "",
          r.residual ?? "",
          r.is_outlier ? "SIM" : "NAO",
        ].join(";")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `serie_processada_${data.meta.transform}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="p-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Table2 className="w-4 h-4 text-indigo-600" />
            Dataset Processado, Componentes STL & Resíduos ({filtered.length} linhas)
          </h3>
          <p className="text-xs text-slate-500">
            Inspecione cada período com valores brutos, transformados, STL e diagnóstico de anomalias.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Filtrar data (ex: 2020)"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(1);
              }}
              className="pl-8 pr-3 py-1.5 text-xs rounded-lg border border-slate-300 focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <label className="flex items-center gap-1.5 text-xs font-medium text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={onlyOutliers}
              onChange={(e) => {
                setOnlyOutliers(e.target.checked);
                setPage(1);
              }}
              className="rounded border-slate-300 text-rose-600"
            />
            Somente Outliers
          </label>

          <button
            type="button"
            onClick={handleDownloadCsv}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold transition"
          >
            <Download className="w-3.5 h-3.5" />
            Baixar CSV
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
            <tr>
              <th className="py-2.5 px-4">Data</th>
              <th className="py-2.5 px-3 text-right">Bruto</th>
              <th className="py-2.5 px-3 text-right">Analisado</th>
              <th className="py-2.5 px-3 text-right">STL Tend.</th>
              <th className="py-2.5 px-3 text-right">STL Saz.</th>
              <th className="py-2.5 px-3 text-right">Ajustado</th>
              <th className="py-2.5 px-3 text-right">Resíduo</th>
              <th className="py-2.5 px-4 text-center">Outlier</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono">
            {rows.map((r) => (
              <tr key={r.date} className={r.is_outlier ? "bg-rose-50/60" : "hover:bg-slate-50"}>
                <td className="py-2 px-4 font-semibold text-slate-800">{r.date}</td>
                <td className="py-2 px-3 text-right text-slate-500">{r.raw_value ?? "—"}</td>
                <td className="py-2 px-3 text-right font-semibold text-slate-900">{r.value ?? "—"}</td>
                <td className="py-2 px-3 text-right text-indigo-700">{r.stl_trend ?? "—"}</td>
                <td className="py-2 px-3 text-right text-emerald-700">{r.stl_seasonal ?? "—"}</td>
                <td className="py-2 px-3 text-right text-slate-700">{r.fitted ?? "—"}</td>
                <td className="py-2 px-3 text-right text-slate-600">{r.residual ?? "—"}</td>
                <td className="py-2 px-4 text-center font-sans">
                  {r.is_outlier ? (
                    <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 text-[10px] font-bold">
                      Sim
                    </span>
                  ) : (
                    <span className="text-slate-400 text-[10px]">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-between text-xs text-slate-600 bg-slate-50">
        <span>
          Página <strong>{safePage}</strong> de <strong>{totalPages}</strong>
        </span>
        <div className="flex gap-1.5">
          <button
            type="button"
            disabled={safePage <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="px-3 py-1 rounded border border-slate-300 bg-white disabled:opacity-40 hover:bg-slate-100 font-medium"
          >
            Anterior
          </button>
          <button
            type="button"
            disabled={safePage >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="px-3 py-1 rounded border border-slate-300 bg-white disabled:opacity-40 hover:bg-slate-100 font-medium"
          >
            Próxima
          </button>
        </div>
      </div>
    </div>
  );
};
