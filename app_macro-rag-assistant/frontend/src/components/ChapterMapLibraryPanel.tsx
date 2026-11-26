import React, { useMemo, useState } from "react";
import {
  BookMarked,
  FileUp,
  GraduationCap,
  Layers,
  MessageSquareText,
  RefreshCw,
  Trash2,
  Zap,
} from "lucide-react";
import type { ChapterSummary, DocumentSummary, IndexStats } from "../types";

interface ChapterMapLibraryPanelProps {
  documents: DocumentSummary[];
  chapters: ChapterSummary[];
  stats: IndexStats;
  isBusy: boolean;
  onUploadFiles: (files: File[]) => Promise<void>;
  onReloadDemo: () => Promise<void>;
  onDeleteDoc: (docId: string) => Promise<void>;
  onStudyChapter: (chapterId: string) => void;
  onFlashcardsChapter: (chapterId: string) => void;
}

export const ChapterMapLibraryPanel: React.FC<ChapterMapLibraryPanelProps> = ({
  documents,
  chapters,
  stats,
  isBusy,
  onUploadFiles,
  onReloadDemo,
  onDeleteDoc,
  onStudyChapter,
  onFlashcardsChapter,
}) => {
  const [dragging, setDragging] = useState(false);
  const [chapterSearch, setChapterSearch] = useState("");

  const handleFiles = async (fl: FileList | null) => {
    if (!fl || fl.length === 0) return;
    const valid = Array.from(fl).filter((f) => /\.(pdf|md|txt)$/i.test(f.name));
    if (valid.length === 0) return;
    await onUploadFiles(valid);
  };

  const groupedChapters = useMemo(() => {
    const q = chapterSearch.trim().toLowerCase();
    const filtered = q
      ? chapters.filter(
          (ch) =>
            ch.title.toLowerCase().includes(q) ||
            ch.part_name.toLowerCase().includes(q) ||
            ch.top_keywords.some((k) => k.toLowerCase().includes(q))
        )
      : chapters;

    const groups: Record<string, ChapterSummary[]> = {};
    for (const ch of filtered) {
      const key = `${ch.document_title} — ${ch.part_name}`;
      if (!groups[key]) groups[key] = [];
      groups[key].push(ch);
    }
    return groups;
  }, [chapters, chapterSearch]);

  return (
    <div className="space-y-5">
      {/* Linha Superior: Upload Multi-Formato + Livros/Notas Carregados */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-2xl border border-amber-500/30 bg-slate-900/80 p-5 lg:col-span-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileUp className="h-4 w-4 text-amber-400" />
              <h3 className="text-sm font-bold text-white">Enviar Livros &amp; Notas de Aula</h3>
            </div>
            <button
              type="button"
              disabled={isBusy}
              onClick={onReloadDemo}
              className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/35 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-200 hover:bg-amber-500/20 disabled:opacity-45"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isBusy ? "animate-spin" : ""}`} />
              Recarregar Heijdra
            </button>
          </div>

          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFiles(e.dataTransfer.files);
            }}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition ${
              dragging
                ? "border-amber-400 bg-amber-500/10"
                : "border-slate-700 bg-slate-950/65 hover:border-amber-400/60"
            }`}
          >
            <input
              type="file"
              accept=".pdf,.md,.txt"
              multiple
              disabled={isBusy}
              onChange={(e) => {
                handleFiles(e.target.files);
                e.target.value = "";
              }}
              className="hidden"
            />
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-amber-500/15 text-amber-300">
              <FileUp className="h-5 w-5" />
            </div>
            <p className="mt-3 text-sm font-bold text-white">
              Arraste arquivos .PDF, .MD ou .TXT aqui
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Extrai automaticamente capítulos do sumário PDF ou cabeçalhos Markdown e salva em cache local comprimido.
            </p>
          </label>
        </div>

        {/* Lista de Documentos Indexados */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-7">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <BookMarked className="h-4 w-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white">
                Biblioteca Ativa ({stats.document_count} obra(s) • {stats.total_pages} págs. • {stats.total_chunks} chunks)
              </h3>
            </div>
            <span className="font-mono text-xs text-slate-400">Build: {stats.build_time_ms.toFixed(0)} ms</span>
          </div>

          <div className="space-y-3">
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="flex items-start justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/70 p-3.5"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded bg-amber-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase text-amber-200">
                      {doc.format}
                    </span>
                    <h4 className="text-sm font-bold text-white">{doc.title}</h4>
                    {doc.loaded_from_cache && (
                      <span className="inline-flex items-center gap-1 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-300">
                        <Zap className="h-3 w-3" /> Cache Local
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-xs text-slate-400">{doc.author}</p>
                  <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
                    <span>{doc.pages} páginas</span>
                    <span>•</span>
                    <span>{doc.chapters_count} capítulos/seções</span>
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
                  className="rounded-lg border border-slate-800 bg-slate-900 p-2 text-slate-400 hover:border-rose-500/50 hover:text-rose-300"
                  title="Remover da biblioteca"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Mapa Hierárquico de Capítulos */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-cyan-400" />
            <div>
              <h3 className="text-sm font-bold text-white">
                Mapa de Capítulos &amp; Tópicos Extraídos Automaticamente ({chapters.length} capítulos)
              </h3>
              <p className="text-xs text-slate-400">
                Clique em qualquer capítulo para filtrar o Tutor exclusivamente naquele tema ou gerar flashcards de prova.
              </p>
            </div>
          </div>

          <input
            type="text"
            value={chapterSearch}
            onChange={(e) => setChapterSearch(e.target.value)}
            placeholder="Filtrar capítulos ou palavras-chave..."
            className="w-64 rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:border-amber-400 focus:outline-none"
          />
        </div>

        <div className="space-y-5">
          {Object.entries(groupedChapters).map(([groupTitle, list]) => (
            <div key={groupTitle}>
              <h4 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-amber-300">
                {groupTitle}
              </h4>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                {list.map((ch) => (
                  <div
                    key={ch.chapter_id}
                    className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-950/75 p-3.5 transition hover:border-slate-700"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 text-[11px] text-slate-400">
                        <span className="font-mono text-cyan-300">
                          pp. {ch.start_page}–{ch.end_page} ({ch.page_count}p)
                        </span>
                        <span className="font-mono">{ch.chunk_count} chunks</span>
                      </div>
                      <h5 className="mt-1.5 text-sm font-bold text-white">{ch.title}</h5>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {ch.top_keywords.map((kw) => (
                          <span
                            key={kw}
                            className="rounded bg-slate-800/90 px-1.5 py-0.5 font-mono text-[10px] text-slate-300"
                          >
                            {kw}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="mt-3.5 flex gap-2">
                      <button
                        type="button"
                        onClick={() => onStudyChapter(ch.chapter_id)}
                        className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-amber-500/15 px-2.5 py-1.5 text-[11px] font-semibold text-amber-200 ring-1 ring-amber-500/30 hover:bg-amber-500/25"
                      >
                        <MessageSquareText className="h-3.5 w-3.5" /> Estudar
                      </button>
                      <button
                        type="button"
                        onClick={() => onFlashcardsChapter(ch.chapter_id)}
                        className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-cyan-500/15 px-2.5 py-1.5 text-[11px] font-semibold text-cyan-200 ring-1 ring-cyan-500/30 hover:bg-cyan-500/25"
                      >
                        <GraduationCap className="h-3.5 w-3.5" /> Flashcards
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
