import React, { useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Clock,
  Download,
  FileText,
  GraduationCap,
  Lock,
  Microscope,
  Scale,
  SearchCheck,
  Send,
  ShieldAlert,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import type {
  ChapterSummary,
  ChatTurn,
  RetrievedPassage,
  StudyMode,
  SuggestedPrompt,
} from "../types";

interface TutorChatPanelProps {
  suggestedPrompts: SuggestedPrompt[];
  chapters: ChapterSummary[];
  activeChapterIds: string[];
  studyMode: StudyMode;
  turns: ChatTurn[];
  isQuerying: boolean;
  hasDocuments: boolean;
  onChangeStudyMode: (m: StudyMode) => void;
  onClearChapterFilter: () => void;
  onAsk: (question: string, modeOverride?: StudyMode) => void;
  onInspectTurn: (turnId: string) => void;
  onClearChat: () => void;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTerms(text: string, terms: string[]): React.ReactNode {
  const clean = Array.from(new Set(terms.filter((t) => t.length >= 3))).slice(0, 14);
  if (!clean.length) return text;
  const rx = new RegExp(`(${clean.map(escapeRegExp).join("|")})`, "gi");
  return text.split(rx).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="rounded bg-amber-500/25 px-1 py-0.5 text-amber-200 ring-1 ring-amber-400/30">
        {part}
      </mark>
    ) : (
      <React.Fragment key={i}>{part}</React.Fragment>
    )
  );
}

function renderInline(line: string, onCite: (rank: number) => void): React.ReactNode[] {
  const rx = /(\[\d+\]|\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  return line.split(rx).map((part, idx) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const r = Number(m[1]);
      return (
        <button
          key={idx}
          type="button"
          onClick={() => onCite(r)}
          className="mx-0.5 inline-flex items-center rounded-md bg-amber-500/20 px-1.5 py-0.5 text-xs font-bold text-amber-300 ring-1 ring-amber-400/40 transition hover:bg-amber-500/35 hover:text-white"
        >
          [{r}]
        </button>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={idx} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={idx} className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-xs text-cyan-300">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return (
        <em key={idx} className="italic text-slate-300">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

function FormattedAnswer({ text, onCite }: { text: string; onCite: (rank: number) => void }) {
  return (
    <div className="space-y-2 text-sm leading-relaxed text-slate-200">
      {text.split("\n").map((raw, i) => {
        const ln = raw.trim();
        if (!ln) return <div key={i} className="h-1" />;
        if (ln.startsWith("### ")) {
          return (
            <h3 key={i} className="pt-1 text-base font-bold text-white">
              {renderInline(ln.slice(4), onCite)}
            </h3>
          );
        }
        if (ln.startsWith("#### ")) {
          return (
            <h4 key={i} className="pt-2 text-xs font-bold uppercase tracking-wider text-amber-300">
              {renderInline(ln.slice(5), onCite)}
            </h4>
          );
        }
        if (ln.startsWith("- ") || ln.startsWith("* ")) {
          return (
            <div key={i} className="flex items-start gap-2 pl-1">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400" />
              <p className="flex-1">{renderInline(ln.slice(2), onCite)}</p>
            </div>
          );
        }
        return <p key={i}>{renderInline(ln, onCite)}</p>;
      })}
    </div>
  );
}

const MODE_OPTIONS: {
  id: StudyMode;
  label: string;
  desc: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { id: "tutor", label: "Tutor Didático", desc: "Intuição + mecanismos + política", icon: GraduationCap },
  { id: "compare", label: "Comparador", desc: "Contrasta escolas e capítulos", icon: Scale },
  { id: "verify", label: "Verificador", desc: "Definições exatas e páginas", icon: SearchCheck },
];

export const TutorChatPanel: React.FC<TutorChatPanelProps> = ({
  suggestedPrompts,
  chapters,
  activeChapterIds,
  studyMode,
  turns,
  isQuerying,
  hasDocuments,
  onChangeStudyMode,
  onClearChapterFilter,
  onAsk,
  onInspectTurn,
  onClearChat,
}) => {
  const [input, setInput] = useState("");
  const [selectedPassage, setSelectedPassage] = useState<{
    turnId: string;
    passage: RetrievedPassage;
  } | null>(null);

  const activeChapterObjs = chapters.filter((c) => activeChapterIds.includes(c.chapter_id));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim().length < 3 || isQuerying || !hasDocuments) return;
    onAsk(input.trim());
    setInput("");
  };

  const handleExportMd = () => {
    if (!turns.length) return;
    const body = turns
      .map((t, i) => {
        const r = t.response;
        const refs = r.citations
          .map(
            (c) =>
              `- [${c.rank}] **${c.chapter_title}** (pág. ${c.page}, ${c.document_title}) — Score: \`${c.score_final.toFixed(3)}\``
          )
          .join("\n");
        return `## ${i + 1}. ${r.question}\n*Modo: ${r.study_mode} | Provedor: ${r.model_used} | Grounding: ${r.diagnostics.grounding_score}% | Exposição Externa: ${r.privacy.external_sent_chars} chars*\n\n${r.answer}\n\n### Referências\n${refs}`;
      })
      .join("\n\n---\n\n");
    const blob = new Blob([`# MacroRAG Tutor Studio — Caderno de Estudos\n\n${body}`], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `caderno-macroeconomia-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      {/* Coluna do Chat */}
      <div className="space-y-4 xl:col-span-8">
        {/* Seletor de Modo de Estudo + Filtro Ativo de Capítulos */}
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/75 px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            {MODE_OPTIONS.map((m) => {
              const Icon = m.icon;
              const active = studyMode === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => onChangeStudyMode(m.id)}
                  className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
                    active
                      ? "bg-amber-500/20 text-amber-200 ring-1 ring-amber-400/45"
                      : "bg-slate-950 text-slate-400 hover:text-slate-200"
                  }`}
                  title={m.desc}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {m.label}
                </button>
              );
            })}
          </div>

          {turns.length > 0 && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleExportMd}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-200 hover:border-amber-400/50"
              >
                <Download className="h-3.5 w-3.5" /> Caderno .md
              </button>
              <button
                type="button"
                onClick={() => {
                  setSelectedPassage(null);
                  onClearChat();
                }}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-400 hover:border-rose-500/50 hover:text-rose-300"
              >
                <Trash2 className="h-3.5 w-3.5" /> Limpar
              </button>
            </div>
          )}
        </div>

        {/* Banner de Filtro por Capítulo (se ativo) */}
        {activeChapterObjs.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-amber-500/35 bg-amber-950/20 px-4 py-2.5 text-xs text-amber-200">
            <span>
              <strong>Escopo restrito a {activeChapterObjs.length} capítulo(s):</strong>{" "}
              {activeChapterObjs.map((c) => c.title).join(" • ")}
            </span>
            <button
              type="button"
              onClick={onClearChapterFilter}
              className="inline-flex items-center gap-1 rounded-lg bg-amber-500/20 px-2 py-1 text-[11px] font-semibold hover:bg-amber-500/30"
            >
              <X className="h-3 w-3" /> Pesquisar livro inteiro
            </button>
          </div>
        )}

        {/* Estado inicial: Perguntas Guiadas */}
        {turns.length === 0 && (
          <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900/80 to-amber-950/20 p-6">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-extrabold text-white">
                  Consulte 816 Páginas de Macroeconomia Moderna em Segundos
                </h2>
                <p className="mt-1 text-sm text-slate-400">
                  Indexação hierárquica por capítulo sobre <strong className="text-slate-200">Heijdra (Oxford)</strong> +
                  suas próprias notas em <code className="text-amber-300">.pdf / .md / .txt</code>. Pergunte em{" "}
                  <strong className="text-slate-200">Português</strong> e receba citações exatas de{" "}
                  <strong className="text-amber-300">Capítulo e Página</strong> com auditoria de privacidade.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {suggestedPrompts.map((sp) => (
                <button
                  key={sp.id}
                  type="button"
                  disabled={isQuerying || !hasDocuments}
                  onClick={() => onAsk(sp.question, sp.mode)}
                  className={`flex flex-col justify-between rounded-xl border p-3.5 text-left transition ${
                    sp.is_out_of_scope
                      ? "border-rose-500/30 bg-rose-950/15 hover:border-rose-400/60"
                      : "border-slate-800 bg-slate-950/60 hover:border-amber-500/50 hover:bg-slate-900"
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`rounded-md px-2 py-0.5 text-[10px] font-bold uppercase ${
                          sp.is_out_of_scope
                            ? "bg-rose-500/20 text-rose-300"
                            : "bg-amber-500/15 text-amber-300"
                        }`}
                      >
                        {sp.category}
                      </span>
                      {sp.chapter_hint && (
                        <span className="font-mono text-[10px] text-slate-500">{sp.chapter_hint}</span>
                      )}
                    </div>
                    <p className="mt-2 text-sm font-semibold text-slate-100">{sp.question}</p>
                  </div>
                  <p className="mt-2 text-xs text-slate-400">{sp.hint}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Histórico */}
        <div className="space-y-5">
          {turns.map((turn) => {
            const r = turn.response;
            return (
              <div
                key={turn.id}
                className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/75 shadow-lg shadow-black/20"
              >
                <div className="border-b border-slate-800 bg-slate-950/65 px-5 py-3.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold uppercase tracking-wider text-amber-400">
                      Consulta ({r.study_mode})
                    </span>
                    <span className="text-slate-500">{turn.timestamp}</span>
                  </div>
                  <p className="mt-1 text-base font-bold text-white">{r.question}</p>
                </div>

                <div className="space-y-4 p-5">
                  {r.fallback_notice && (
                    <div className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3.5 py-2.5 text-xs text-amber-200">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                      <span>{r.fallback_notice}</span>
                    </div>
                  )}

                  {/* Badges de Grounding, Latência e Privacidade */}
                  <div className="flex flex-wrap items-center gap-2">
                    {r.diagnostics.confidence_level === "alta" ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-semibold text-emerald-300 ring-1 ring-emerald-500/30">
                        <CheckCircle2 className="h-3.5 w-3.5" /> Grounding {r.diagnostics.grounding_score.toFixed(0)}%
                      </span>
                    ) : r.diagnostics.confidence_level === "media" ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-0.5 text-xs font-semibold text-amber-300 ring-1 ring-amber-500/30">
                        <AlertTriangle className="h-3.5 w-3.5" /> Grounding {r.diagnostics.grounding_score.toFixed(0)}%
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2.5 py-0.5 text-xs font-semibold text-rose-300 ring-1 ring-rose-500/30">
                        <ShieldAlert className="h-3.5 w-3.5" /> Fora do Escopo ({r.diagnostics.grounding_score.toFixed(0)}%)
                      </span>
                    )}

                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ${
                        r.privacy.is_fully_local
                          ? "bg-cyan-500/15 text-cyan-200 ring-cyan-500/30"
                          : "bg-violet-500/15 text-violet-200 ring-violet-500/30"
                      }`}
                    >
                      <Lock className="h-3 w-3" />
                      {r.privacy.is_fully_local
                        ? "100% Local (0 bytes externos)"
                        : `${r.privacy.external_exposure_pct.toFixed(2)}% do corpus enviado`}
                    </span>

                    <span className="inline-flex items-center gap-1 rounded-full bg-slate-800 px-2.5 py-0.5 text-xs text-slate-300">
                      <Clock className="h-3 w-3 text-amber-400" /> {r.diagnostics.total_ms.toFixed(0)} ms
                    </span>

                    <button
                      type="button"
                      onClick={() => onInspectTurn(turn.id)}
                      className="ml-auto inline-flex items-center gap-1.5 rounded-lg bg-amber-500/15 px-3 py-1 text-xs font-semibold text-amber-200 ring-1 ring-amber-500/30 hover:bg-amber-500/25"
                    >
                      <Microscope className="h-3.5 w-3.5" /> Privacidade &amp; Raio-X
                    </button>
                  </div>

                  <div className="rounded-xl border border-slate-800/80 bg-slate-950/55 p-4">
                    <FormattedAnswer
                      text={r.answer}
                      onCite={(rank) => {
                        const p = r.citations.find((c) => c.rank === rank);
                        if (p) setSelectedPassage({ turnId: turn.id, passage: p });
                      }}
                    />
                  </div>

                  {/* Fontes Recuperadas por Capítulo & Página */}
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Passagens Recuperadas ({r.citations.length} trechos em {r.diagnostics.unique_chapters_cited}{" "}
                      capítulo(s), {r.diagnostics.unique_pages_cited} página(s)):
                    </p>
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {r.citations.map((c) => {
                        const active =
                          selectedPassage?.turnId === turn.id && selectedPassage?.passage.chunk_id === c.chunk_id;
                        return (
                          <button
                            key={c.chunk_id}
                            type="button"
                            onClick={() => setSelectedPassage({ turnId: turn.id, passage: c })}
                            className={`rounded-xl border p-3 text-left transition ${
                              active
                                ? "border-amber-400 bg-amber-950/25 ring-1 ring-amber-400/40"
                                : "border-slate-800 bg-slate-950/60 hover:border-slate-600"
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="inline-flex items-center gap-1.5 text-xs font-bold text-amber-300">
                                <span className="rounded bg-amber-500/20 px-1.5 py-0.5">[{c.rank}]</span>
                                Pág. {c.page}
                              </span>
                              <span className="font-mono text-[11px] text-emerald-300">
                                {c.score_final.toFixed(3)}
                              </span>
                            </div>
                            <p className="mt-1 truncate text-xs font-bold text-slate-200">{c.chapter_title}</p>
                            <p className="mt-0.5 line-clamp-2 text-[11px] text-slate-400">{c.excerpt}</p>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Input */}
        <form
          onSubmit={handleSubmit}
          className="sticky bottom-3 rounded-2xl border border-slate-700/80 bg-slate-900/95 p-3 shadow-2xl shadow-black/50 backdrop-blur"
        >
          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isQuerying || !hasDocuments}
              placeholder="Pergunte sobre Macroeconomia (ex.: Explique a Equivalência Ricardiana; Compare Solow vs. Romer)..."
              className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:border-amber-400 focus:outline-none focus:ring-1 focus:ring-amber-400"
            />
            <button
              type="submit"
              disabled={input.trim().length < 3 || isQuerying || !hasDocuments}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 px-5 py-3 text-sm font-bold text-slate-950 transition hover:brightness-110 disabled:opacity-45"
            >
              <Send className="h-4 w-4" />
              {isQuerying ? "Consultando..." : "Consultar"}
            </button>
          </div>
        </form>
      </div>

      {/* Inspetor Lateral Direito de Capítulo & Página */}
      <div className="xl:col-span-4">
        <div className="sticky top-20 rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
          <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-3">
            <BookOpen className="h-4 w-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">Inspetor de Capítulo &amp; Página</h3>
          </div>

          {!selectedPassage ? (
            <div className="space-y-3 py-8 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-800/80 text-slate-400">
                <FileText className="h-6 w-6" />
              </div>
              <p className="text-sm font-medium text-slate-300">
                Clique em qualquer citação <span className="text-amber-300">[1]</span>,{" "}
                <span className="text-amber-300">[2]</span>...
              </p>
              <p className="text-xs leading-relaxed text-slate-500">
                Veja o trecho literal exato do livro-texto ou nota de aula, o capítulo completo, a página original e os
                termos macroeconômicos casados pela busca híbrida.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-3.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="rounded bg-amber-500/20 px-2 py-0.5 font-bold text-amber-200">
                    Fonte [{selectedPassage.passage.rank}]
                  </span>
                  <span className="font-mono font-semibold text-slate-200">
                    Pág. {selectedPassage.passage.page}
                  </span>
                </div>
                <p className="mt-2 text-sm font-bold text-white">{selectedPassage.passage.chapter_title}</p>
                <p className="mt-0.5 text-[11px] text-amber-300/80">
                  {selectedPassage.passage.part_name} • {selectedPassage.passage.document_title}
                </p>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-center">
                  <p className="text-[10px] uppercase text-slate-400">Híbrido</p>
                  <p className="mt-0.5 font-mono text-sm font-bold text-emerald-300">
                    {selectedPassage.passage.score_final.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-center">
                  <p className="text-[10px] uppercase text-slate-400">
                    BM25 (#{selectedPassage.passage.rank_lexical})
                  </p>
                  <p className="mt-0.5 font-mono text-sm font-bold text-cyan-300">
                    {selectedPassage.passage.score_lexical.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-center">
                  <p className="text-[10px] uppercase text-slate-400">
                    Semânt. (#{selectedPassage.passage.rank_semantic})
                  </p>
                  <p className="mt-0.5 font-mono text-sm font-bold text-violet-300">
                    {selectedPassage.passage.score_semantic.toFixed(3)}
                  </p>
                </div>
              </div>

              {selectedPassage.passage.matched_terms.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {selectedPassage.passage.matched_terms.map((t) => (
                    <span
                      key={t}
                      className="rounded bg-amber-500/15 px-2 py-0.5 font-mono text-[11px] text-amber-200 ring-1 ring-amber-500/30"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              )}

              <div className="max-h-[420px] overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/80 p-3.5 text-xs leading-relaxed text-slate-200">
                {highlightTerms(selectedPassage.passage.text, selectedPassage.passage.matched_terms)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
