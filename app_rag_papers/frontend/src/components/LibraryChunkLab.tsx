import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BookMarked,
  Database,
  FileUp,
  RefreshCw,
  Scissors,
  Search,
  Sliders,
  Trash2,
} from "lucide-react";
import { fetchChunks } from "../api";
import type { ChunkItem, DocumentSummary, IndexStats } from "../types";

interface LibraryChunkLabProps {
  documents: DocumentSummary[];
  stats: IndexStats;
  isBusy: boolean;
  onUploadFiles: (files: File[]) => Promise<void>;
  onLoadDemo: () => Promise<void>;
  onDeleteDoc: (docId: string) => Promise<void>;
  onRebuildIndex: (chunkSize: number, chunkOverlap: number) => Promise<void>;
}

export const LibraryChunkLab: React.FC<LibraryChunkLabProps> = ({
  documents,
  stats,
  isBusy,
  onUploadFiles,
  onLoadDemo,
  onDeleteDoc,
  onRebuildIndex,
}) => {
  const [chunkSize, setChunkSize] = useState(stats.chunk_size);
  const [chunkOverlap, setChunkOverlap] = useState(stats.chunk_overlap);
  const [isDragging, setIsDragging] = useState(false);

  // Explorador de chunks
  const [filterDocId, setFilterDocId] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [offset, setOffset] = useState(0);
  const [chunkItems, setChunkItems] = useState<ChunkItem[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const PAGE_LIMIT = 8;

  useEffect(() => {
    setChunkSize(stats.chunk_size);
    setChunkOverlap(stats.chunk_overlap);
  }, [stats.chunk_size, stats.chunk_overlap]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingChunks(true);
      try {
        const res = await fetchChunks({
          doc_id: filterDocId || undefined,
          search: searchTerm.trim() || undefined,
          offset,
          limit: PAGE_LIMIT,
        });
        if (!cancelled) {
          setChunkItems(res.items);
          setChunkTotal(res.total);
        }
      } catch {
        if (!cancelled) {
          setChunkItems([]);
          setChunkTotal(0);
        }
      } finally {
        if (!cancelled) setLoadingChunks(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [filterDocId, searchTerm, offset, stats.total_chunks, stats.build_time_ms]);

  const handleFileInput = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const pdfs = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) return;
    await onUploadFiles(pdfs);
    setOffset(0);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    await handleFileInput(e.dataTransfer.files);
  };

  const maxAllowedOverlap = Math.min(600, Math.floor(chunkSize / 2));

  return (
    <div className="space-y-5">
      {/* Seção 1: Upload + Biblioteca de PDFs */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        {/* Dropzone */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileUp className="h-4 w-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">Ingestão de Artigos (PDF)</h3>
            </div>
            <button
              type="button"
              disabled={isBusy}
              onClick={onLoadDemo}
              className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/35 bg-cyan-500/10 px-2.5 py-1 text-xs font-semibold text-cyan-300 transition hover:bg-cyan-500/20 disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isBusy ? "animate-spin" : ""}`} />
              Recarregar Demo
            </button>
          </div>

          <label
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition ${
              isDragging
                ? "border-cyan-400 bg-cyan-500/10"
                : "border-slate-700 bg-slate-950/60 hover:border-cyan-500/50"
            }`}
          >
            <input
              type="file"
              accept=".pdf,application/pdf"
              multiple
              disabled={isBusy}
              onChange={(e) => {
                handleFileInput(e.target.files);
                e.target.value = "";
              }}
              className="hidden"
            />
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300">
              <FileUp className="h-5 w-5" />
            </div>
            <p className="mt-3 text-sm font-semibold text-slate-100">
              Arraste artigos PDF aqui ou clique para selecionar
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Limpeza automática de ligaduras LaTeX (<code className="text-cyan-300">ﬁ/ﬀ/ﬂ</code>) e preservação do número exato da página.
            </p>
          </label>
        </div>

        {/* Lista de Documentos Carregados */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-7">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BookMarked className="h-4 w-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white">
                Biblioteca Indexada ({documents.length} documento(s) • {stats.total_pages} páginas)
              </h3>
            </div>
          </div>

          {documents.length === 0 ? (
            <p className="py-8 text-center text-sm text-slate-500">
              Nenhum documento carregado. Envie um PDF ou clique em "Recarregar Demo".
            </p>
          ) : (
            <div className="space-y-3">
              {documents.map((doc) => (
                <div
                  key={doc.doc_id}
                  className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5 transition hover:border-slate-700"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h4 className="text-sm font-bold text-white">{doc.title}</h4>
                        {doc.is_demo && (
                          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-emerald-300">
                            Demo Pré-Carregado
                          </span>
                        )}
                        {doc.year && (
                          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-mono text-slate-300">
                            {doc.year}
                          </span>
                        )}
                      </div>
                      <p className="mt-0.5 text-xs text-cyan-300/90">{doc.authors}</p>
                      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
                        <span>{doc.pages} páginas</span>
                        <span>•</span>
                        <span>{doc.chunk_count} chunks</span>
                        <span>•</span>
                        <span>{doc.total_words.toLocaleString("pt-BR")} palavras</span>
                      </div>
                    </div>

                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => onDeleteDoc(doc.doc_id)}
                      className="rounded-lg border border-slate-800 bg-slate-900 p-2 text-slate-400 transition hover:border-rose-500/50 hover:text-rose-300 disabled:opacity-40"
                      title="Remover documento da biblioteca"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Seção 2: Laboratório Interativo de Chunking & Histograma */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-5">
          <div className="mb-3 flex items-center gap-2">
            <Sliders className="h-4 w-4 text-violet-400" />
            <h3 className="text-sm font-bold text-white">Laboratório de Chunking & Reindexação</h3>
          </div>
          <p className="mb-4 text-xs leading-relaxed text-slate-400">
            Ajuste o tamanho dos blocos e a sobreposição entre chunks consecutivos para observar em tempo real o
            trade-off clássico de RAG: <strong className="text-slate-200">chunks menores</strong> aumentam precisão local,
            enquanto <strong className="text-slate-200">chunks maiores + overlap</strong> preservam contexto de teoremas e fórmulas.
          </p>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-300">Tamanho do Chunk (chunk_size)</span>
                <span className="font-mono font-bold text-cyan-300">{chunkSize} chars</span>
              </div>
              <input
                type="range"
                min={350}
                max={2400}
                step={50}
                value={chunkSize}
                onChange={(e) => {
                  const next = Number(e.target.value);
                  setChunkSize(next);
                  if (chunkOverlap > Math.floor(next / 2)) {
                    setChunkOverlap(Math.floor(next / 2));
                  }
                }}
                className="mt-1.5 w-full accent-cyan-400"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-300">Sobreposição (chunk_overlap)</span>
                <span className="font-mono font-bold text-violet-300">{chunkOverlap} chars</span>
              </div>
              <input
                type="range"
                min={0}
                max={maxAllowedOverlap}
                step={20}
                value={Math.min(chunkOverlap, maxAllowedOverlap)}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                className="mt-1.5 w-full accent-violet-400"
              />
            </div>

            <button
              type="button"
              disabled={isBusy || documents.length === 0}
              onClick={async () => {
                await onRebuildIndex(chunkSize, Math.min(chunkOverlap, maxAllowedOverlap));
                setOffset(0);
              }}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-500 to-cyan-500 px-4 py-2.5 text-xs font-bold text-slate-950 transition hover:brightness-110 disabled:opacity-40"
            >
              <Scissors className="h-4 w-4" />
              {isBusy ? "Reindexando..." : "Aplicar Chunking & Reconstruir Índices"}
            </button>

            {/* Métricas do Índice */}
            <div className="grid grid-cols-3 gap-2 pt-1">
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
                <p className="text-[10px] uppercase text-slate-400">Chunks</p>
                <p className="mt-0.5 font-mono text-base font-bold text-white">{stats.total_chunks}</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
                <p className="text-[10px] uppercase text-slate-400">Média / Med.</p>
                <p className="mt-0.5 font-mono text-xs font-bold text-cyan-300">
                  {Math.round(stats.avg_chunk_chars)} / {Math.round(stats.median_chunk_chars)}
                </p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
                <p className="text-[10px] uppercase text-slate-400">Overlap Real</p>
                <p className="mt-0.5 font-mono text-base font-bold text-violet-300">
                  {stats.overlap_ratio_pct.toFixed(1)}%
                </p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
                <p className="text-[10px] uppercase text-slate-400">Vocab BM25</p>
                <p className="mt-0.5 font-mono text-sm font-bold text-emerald-300">
                  {stats.vocab_size.toLocaleString("pt-BR")}
                </p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
                <p className="text-[10px] uppercase text-slate-400">Dim. LSA</p>
                <p className="mt-0.5 font-mono text-sm font-bold text-amber-300">{stats.lsa_dimensions}D</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
                <p className="text-[10px] uppercase text-slate-400">Build</p>
                <p className="mt-0.5 font-mono text-sm font-bold text-slate-200">
                  {stats.build_time_ms.toFixed(0)} ms
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Histograma Recharts */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-7">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">Distribuição de Tamanho dos Chunks (Caracteres)</h3>
            </div>
            <span className="text-xs text-slate-400">
              Min {stats.min_chunk_chars} — Max {stats.max_chunk_chars} chars
            </span>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.histogram} margin={{ top: 10, right: 16, left: 0, bottom: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="range_label"
                  stroke="#94a3b8"
                  fontSize={11}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "12px",
                    color: "#f8fafc",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="count" name="Quantidade de Chunks" fill="#38bdf8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Seção 3: Explorador de Chunks com Destaque Visual da Zona de Overlap */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-white">
              Explorador de Chunks & Visualização da Zona de Overlap
            </h3>
            <p className="text-xs text-slate-400">
              Trechos destacados em <span className="rounded bg-cyan-500/20 px-1 text-cyan-200">ciano</span> mostram o
              prefixo herdado do chunk anterior (overlap) para evitar cortar definições e teoremas ao meio.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={filterDocId}
              onChange={(e) => {
                setFilterDocId(e.target.value);
                setOffset(0);
              }}
              className="rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
            >
              <option value="">Todos os artigos</option>
              {documents.map((d) => (
                <option key={d.doc_id} value={d.doc_id}>
                  {d.title}
                </option>
              ))}
            </select>

            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setOffset(0);
                }}
                placeholder="Filtrar chunks por palavra..."
                className="w-56 rounded-xl border border-slate-700 bg-slate-950 py-2 pl-8 pr-3 text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {loadingChunks ? (
          <p className="py-6 text-center text-xs text-slate-400">Carregando chunks...</p>
        ) : chunkItems.length === 0 ? (
          <p className="py-6 text-center text-xs text-slate-500">Nenhum chunk encontrado para este filtro.</p>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {chunkItems.map((ch) => {
                const overlapLen = Math.min(ch.overlap_prefix_chars, ch.text.length);
                const overlapPart = overlapLen > 0 ? ch.text.slice(0, overlapLen) : "";
                const freshPart = ch.text.slice(overlapLen);
                return (
                  <div
                    key={ch.chunk_id}
                    className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5"
                  >
                    <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-[11px]">
                      <span className="font-semibold text-slate-200">
                        {ch.document_title.slice(0, 42)}
                        {ch.document_title.length > 42 ? "..." : ""} • Pág. {ch.page}
                      </span>
                      <div className="flex items-center gap-1.5">
                        {overlapLen > 0 && (
                          <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
                            Overlap {overlapLen}c
                          </span>
                        )}
                        <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
                          #{ch.chunk_index} • {ch.char_count}c
                        </span>
                      </div>
                    </div>
                    <p className="max-h-36 overflow-y-auto text-xs leading-relaxed text-slate-300">
                      {overlapPart && (
                        <span
                          className="rounded bg-cyan-500/20 px-0.5 text-cyan-200 ring-1 ring-cyan-500/25"
                          title="Zona de Overlap herdada do chunk anterior"
                        >
                          {overlapPart}
                        </span>
                      )}
                      <span>{freshPart}</span>
                    </p>
                  </div>
                );
              })}
            </div>

            {/* Paginação */}
            <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
              <span>
                Mostrando {offset + 1}–{Math.min(offset + PAGE_LIMIT, chunkTotal)} de {chunkTotal} chunks
              </span>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - PAGE_LIMIT))}
                  className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-slate-200 disabled:opacity-40"
                >
                  Anterior
                </button>
                <button
                  type="button"
                  disabled={offset + PAGE_LIMIT >= chunkTotal}
                  onClick={() => setOffset(offset + PAGE_LIMIT)}
                  className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-slate-200 disabled:opacity-40"
                >
                  Próximo
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
