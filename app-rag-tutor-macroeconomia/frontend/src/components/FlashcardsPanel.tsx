import React, { useState } from "react";
import { CheckCircle2, Eye, EyeOff, GraduationCap, RotateCcw, Sparkles } from "lucide-react";
import type { ChapterSummary, FlashcardResponse } from "../types";

interface FlashcardsPanelProps {
  chapters: ChapterSummary[];
  selectedChapterId: string;
  flashcards: FlashcardResponse | null;
  isGenerating: boolean;
  onSelectChapterId: (id: string) => void;
  onGenerate: (chapterId: string, topic: string, count: number) => Promise<void>;
}

export const FlashcardsPanel: React.FC<FlashcardsPanelProps> = ({
  chapters,
  selectedChapterId,
  flashcards,
  isGenerating,
  onSelectChapterId,
  onGenerate,
}) => {
  const [topic, setTopic] = useState("");
  const [count, setCount] = useState(5);
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [mastered, setMastered] = useState<Record<string, boolean>>({});

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    setRevealed({});
    setMastered({});
    await onGenerate(selectedChapterId, topic, count);
  };

  const masteredCount = flashcards
    ? flashcards.cards.filter((c) => mastered[c.id]).length
    : 0;

  return (
    <div className="space-y-5">
      {/* Formulário de Geração de Simulado / Flashcards */}
      <form
        onSubmit={handleRun}
        className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5"
      >
        <div className="mb-4 flex items-center gap-2">
          <GraduationCap className="h-5 w-5 text-amber-400" />
          <div>
            <h2 className="text-base font-bold text-white">
              Gerador de Flashcards &amp; Questões de Prova (Baseado nos Capítulos)
            </h2>
            <p className="text-xs text-slate-400">
              Escolha um capítulo de Macroeconomia (ou todo o livro) e gere cartões de estudo com gabarito ancorado na página exata.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
          <div className="md:col-span-5">
            <label className="mb-1 block text-xs text-slate-400">Capítulo / Seção</label>
            <select
              value={selectedChapterId}
              onChange={(e) => onSelectChapterId(e.target.value)}
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-xs text-slate-100 focus:border-amber-400 focus:outline-none"
            >
              <option value="">Todo o Corpus (Todos os Capítulos)</option>
              {chapters.map((ch) => (
                <option key={ch.chapter_id} value={ch.chapter_id}>
                  {ch.title} (pp. {ch.start_page}–{ch.end_page})
                </option>
              ))}
            </select>
          </div>

          <div className="md:col-span-4">
            <label className="mb-1 block text-xs text-slate-400">Tópico Foco (Opcional)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Ex.: Phillips curve, Ricardian, Calvo..."
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:border-amber-400 focus:outline-none"
            />
          </div>

          <div className="md:col-span-1">
            <label className="mb-1 block text-xs text-slate-400">Qtd.</label>
            <select
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-2 py-2.5 font-mono text-xs text-slate-100"
            >
              {[3, 4, 5, 6, 8].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end md:col-span-2">
            <button
              type="submit"
              disabled={isGenerating}
              className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 px-4 py-2.5 text-xs font-extrabold text-slate-950 transition hover:brightness-110 disabled:opacity-45"
            >
              <Sparkles className="h-4 w-4" />
              {isGenerating ? "Gerando..." : "Gerar Cards"}
            </button>
          </div>
        </div>
      </form>

      {/* Placar de Estudo + Cards */}
      {flashcards && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-800 bg-slate-900/75 px-4 py-3 text-xs">
            <div>
              <span className="font-bold text-white">{flashcards.scope_label}</span>
              <span className="ml-2 text-slate-400">({flashcards.generation_ms.toFixed(1)} ms)</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="font-semibold text-emerald-300">
                Progresso: {masteredCount} / {flashcards.cards.length} dominado(s)
              </span>
              <button
                type="button"
                onClick={() => {
                  setRevealed({});
                  setMastered({});
                }}
                className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1 text-slate-300 hover:text-white"
              >
                <RotateCcw className="h-3 w-3" /> Reiniciar
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {flashcards.cards.map((card, idx) => {
              const isOpen = Boolean(revealed[card.id]);
              const isDone = Boolean(mastered[card.id]);
              return (
                <div
                  key={card.id}
                  className={`flex flex-col justify-between rounded-2xl border p-5 transition ${
                    isDone
                      ? "border-emerald-500/40 bg-emerald-950/15"
                      : "border-slate-800 bg-slate-900/80"
                  }`}
                >
                  <div>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="rounded-md bg-amber-500/15 px-2 py-0.5 text-[10px] font-bold uppercase text-amber-300">
                        Card #{idx + 1} • {card.difficulty}
                      </span>
                      <span className="font-mono text-[11px] text-cyan-300">
                        {card.chapter_title} • pág. {card.page}
                      </span>
                    </div>

                    <p className="mt-2 font-mono text-xs font-semibold text-violet-300">
                      Conceito-chave: {card.concept}
                    </p>
                    <p className="mt-2 text-sm font-bold leading-relaxed text-white">{card.question}</p>

                    {isOpen && (
                      <div className="mt-4 space-y-2.5 rounded-xl border border-emerald-500/25 bg-slate-950/80 p-3.5">
                        <p className="text-xs leading-relaxed text-emerald-200">{card.answer}</p>
                        <details className="text-[11px] text-slate-400">
                          <summary className="cursor-pointer font-semibold text-cyan-300">
                            Ver trecho literal da página {card.page}
                          </summary>
                          <p className="mt-2 leading-relaxed text-slate-300">{card.source_excerpt}</p>
                        </details>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 flex items-center justify-between gap-2 border-t border-slate-800 pt-3">
                    <button
                      type="button"
                      onClick={() => setRevealed((prev) => ({ ...prev, [card.id]: !prev[card.id] }))}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700"
                    >
                      {isOpen ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      {isOpen ? "Ocultar Gabarito" : "Revelar Gabarito"}
                    </button>

                    <button
                      type="button"
                      onClick={() => setMastered((prev) => ({ ...prev, [card.id]: !prev[card.id] }))}
                      className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                        isDone
                          ? "bg-emerald-500/25 text-emerald-200 ring-1 ring-emerald-400/40"
                          : "border border-slate-700 bg-slate-950 text-slate-400 hover:text-emerald-300"
                      }`}
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      {isDone ? "Dominado!" : "Marcar como Dominado"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
