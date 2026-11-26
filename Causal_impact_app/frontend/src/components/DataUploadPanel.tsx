import React, { useState } from "react";
import {
  CheckCircle2,
  Database,
  FileSpreadsheet,
  FileUp,
  Play,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { getExportTemplateUrl } from "../api";
import type { DatasetSummary, DemoDatasetMeta } from "../types";

interface DataUploadPanelProps {
  demos: DemoDatasetMeta[];
  dataset: DatasetSummary | null;
  isLoadingData: boolean;
  isAnalyzing: boolean;
  onUploadFile: (file: File) => Promise<void>;
  onSelectDemo: (demoId: string, autoRun: boolean) => Promise<void>;
  onRunAnalysis: () => Promise<void>;
}

export const DataUploadPanel: React.FC<DataUploadPanelProps> = ({
  demos,
  dataset,
  isLoadingData,
  isAnalyzing,
  onUploadFile,
  onSelectDemo,
  onRunAnalysis,
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await onUploadFile(file);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner de Boas-Vindas & Upload Drag-and-Drop */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        {/* Card Principal: Upload dos Próprios Dados */}
        <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-slate-900 via-slate-900/90 to-cyan-950/25 p-6 xl:col-span-7">
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30">
                <UploadCloud className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-base font-extrabold text-white">
                  Carregue Seus Próprios Dados (CSV ou Excel)
                </h2>
                <p className="mt-1 text-xs leading-relaxed text-slate-400">
                  Envie uma série temporal com uma coluna de <strong className="text-slate-200">Data</strong>, sua{" "}
                  <strong className="text-slate-200">variável alvo (Y)</strong> e, opcionalmente,{" "}
                  <strong className="text-slate-200">variáveis de controle (X)</strong>. Suporta vírgula ou ponto-e-vírgula e padrão decimal brasileiro.
                </p>
              </div>
            </div>

            <a
              href={getExportTemplateUrl()}
              download="modelo_dados_causalimpact.xlsx"
              className="inline-flex shrink-0 items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800/90 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-emerald-500/50 hover:text-emerald-200"
            >
              <FileSpreadsheet className="h-4 w-4 text-emerald-400" />
              Baixar Planilha Modelo (.xlsx)
            </a>
          </div>

          <label
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-9 text-center transition ${
              isDragging
                ? "border-cyan-400 bg-cyan-500/10"
                : "border-slate-700 bg-slate-950/70 hover:border-cyan-400/60 hover:bg-slate-950"
            }`}
          >
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              disabled={isLoadingData}
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) onUploadFile(file);
              }}
              className="hidden"
            />
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 text-cyan-300 ring-1 ring-cyan-500/30">
              <FileUp className="h-6 w-6" />
            </div>
            <p className="mt-3 text-sm font-bold text-white">
              {isLoadingData
                ? "Lendo e validando arquivo..."
                : "Arraste seu arquivo .CSV ou .XLSX aqui, ou clique para escolher"}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Detecção automática da coluna de datas, colunas numéricas e sugestão inicial da data de intervenção.
            </p>
          </label>
        </div>

        {/* Card Lateral: Passo a Passo Rápido */}
        <div className="flex flex-col justify-between rounded-2xl border border-slate-800 bg-slate-900/75 p-6 xl:col-span-5">
          <div>
            <div className="mb-3 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white">Fluxo Guiado em 3 Passos</h3>
            </div>
            <ol className="space-y-3 text-xs leading-relaxed text-slate-300">
              <li className="flex items-start gap-2.5">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-cyan-500/20 font-mono text-[11px] font-bold text-cyan-300">
                  1
                </span>
                <span>
                  <strong className="text-white">Carregue sua base</strong> (CSV/XLSX) acima ou escolha um dos{" "}
                  <strong className="text-white">3 cenários demo</strong> abaixo.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 font-mono text-[11px] font-bold text-emerald-300">
                  2
                </span>
                <span>
                  Na barra lateral esquerda, confirme a <strong className="text-white">Coluna de Data</strong>, a{" "}
                  <strong className="text-white">Variável Y</strong>, os{" "}
                  <strong className="text-white">Controles X</strong> e a{" "}
                  <strong className="text-amber-300">Data da Intervenção</strong>.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-violet-500/20 font-mono text-[11px] font-bold text-violet-300">
                  3
                </span>
                <span>
                  Clique em <strong className="text-white">Executar Análise Causal</strong> para estimar o contrafactual
                  BSTS, auditar o Teste Placebo e exportar o relatório completo em Excel.
                </span>
              </li>
            </ol>
          </div>

          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-[11px] text-slate-400">
            <strong className="text-cyan-300">Dica econométrica:</strong> inclua covariáveis de controle que acompanhem
            o histórico de Y mas <strong className="text-slate-200">não tenham sofrido impacto da intervenção</strong>{" "}
            (ex.: região controle, buscas do setor, índice macro).
          </div>
        </div>
      </div>

      {/* Base Atualmente Carregada (Pronta para Configurar & Executar) */}
      {dataset && (
        <div className="rounded-2xl border border-emerald-500/35 bg-slate-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-bold text-white">Base Pronta para Modelagem: {dataset.name}</h3>
                  <span
                    className={`rounded-md px-2 py-0.5 text-[10px] font-bold uppercase ${
                      dataset.source === "upload"
                        ? "bg-cyan-500/20 text-cyan-200"
                        : "bg-emerald-500/20 text-emerald-200"
                    }`}
                  >
                    {dataset.source === "upload" ? "Arquivo do Usuário" : "Cenário Demo"}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-400">
                  <strong className="text-slate-200">{dataset.rows}</strong> observações • Frequência:{" "}
                  <strong className="text-slate-200">{dataset.inferred_frequency}</strong> • Período:{" "}
                  <span className="font-mono text-cyan-300">
                    {dataset.min_date} a {dataset.max_date}
                  </span>{" "}
                  • Colunas numéricas detectadas:{" "}
                  <span className="font-mono text-slate-200">{dataset.numeric_columns.join(", ")}</span>
                </p>
              </div>
            </div>

            <button
              type="button"
              disabled={isAnalyzing}
              onClick={onRunAnalysis}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 px-5 py-2.5 text-xs font-extrabold text-slate-950 shadow-lg shadow-emerald-500/20 transition hover:brightness-110 disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {isAnalyzing ? "Estimando BSTS..." : "Executar Análise Causal Agora"}
            </button>
          </div>

          {/* Prévia das primeiras linhas */}
          <div className="mt-4 overflow-x-auto">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
              Prévia da Base (primeiras {dataset.preview.length} linhas) — confira as colunas na barra lateral antes de rodar:
            </p>
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
                {dataset.preview.slice(0, 6).map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30">
                    {dataset.columns.map((col) => (
                      <td key={col} className="py-1.5 pr-4 text-slate-300">
                        {row[col] === null || row[col] === undefined ? "—" : String(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Galeria dos 3 Cenários Demo */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-4 flex items-center gap-2">
          <Database className="h-4 w-4 text-violet-400" />
          <div>
            <h3 className="text-sm font-bold text-white">Ou Teste Rapidamente com 1 dos 3 Cenários de Demonstração</h3>
            <p className="text-xs text-slate-400">
              Clique em <strong>Carregar &amp; Analisar</strong> para explorar um caso positivo real, o benchmark clássico de Brodersen ou um cenário placebo nulo.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {demos.map((d) => (
            <div
              key={d.id}
              className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-950/70 p-4 transition hover:border-slate-700"
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <span className="rounded-md bg-violet-500/15 px-2 py-0.5 text-[10px] font-bold uppercase text-violet-300">
                    {d.frequency_label}
                  </span>
                  <span className="font-mono text-[10px] text-slate-500">{d.default_intervention_date}</span>
                </div>
                <h4 className="mt-2.5 text-sm font-bold text-white">{d.name}</h4>
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{d.description}</p>
                <p className="mt-2 text-[11px] font-medium text-emerald-300/90">{d.expected_insight}</p>
              </div>

              <div className="mt-4 flex gap-2">
                <button
                  type="button"
                  disabled={isLoadingData || isAnalyzing}
                  onClick={() => onSelectDemo(d.id, true)}
                  className="flex-1 rounded-lg bg-slate-800 px-3 py-2 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-500/30 transition hover:bg-cyan-500/20 disabled:opacity-45"
                >
                  Carregar &amp; Analisar
                </button>
                <button
                  type="button"
                  disabled={isLoadingData || isAnalyzing}
                  onClick={() => onSelectDemo(d.id, false)}
                  className="rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-2 text-xs text-slate-400 transition hover:border-slate-600 hover:text-slate-200 disabled:opacity-45"
                  title="Apenas carregar a base para configurar manualmente"
                >
                  Só Carregar
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
