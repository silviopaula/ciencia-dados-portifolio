import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Clock,
  Download,
  FileText,
  Filter,
  Microscope,
  Send,
  ShieldAlert,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { ChatTurn, DocumentSummary, RetrievedChunk, SuggestedQuestion } from "../types";

interface ChatWorkspaceProps {
  documents: DocumentSummary[];
  suggestedQuestions: SuggestedQuestion[];
  turns: ChatTurn[];
  isQuerying: boolean;
  selectedDocIds: string[];
  onToggleDocFilter: (docId: string) => void;
  onClearDocFilter: () => void;
  onAsk: (question: string) => void;
  onInspectTurn: (turnId: string) => void;
  onClearChat: () => void;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTermsInText(text: string, terms: string[]): React.ReactNode {
  const cleanTerms = Array.from(new Set(terms.filter((t) => t.length >= 3))).slice(0, 14);
  if (!cleanTerms.length) return text;
  const regex = new RegExp(`(${cleanTerms.map(escapeRegExp).join("|")})`, "gi");
  const parts = text.split(regex);
  return parts.map((part, idx) =>
    idx % 2 === 1 ? (
      <mark
        key={idx}
        className="rounded bg-cyan-500/25 px-1 py-0.5 text-cyan-200 ring-1 ring-cyan-400/30"
      >
        {part}
      </mark>
    ) : (
      <React.Fragment key={idx}>{part}</React.Fragment>
    )
  );
}

function renderInlineFormatting(
  line: string,
  onClickCitation: (rank: number) => void
): React.ReactNode[] {
  // Divide por citações [1], negrito **...**, itálico *...* ou código `...`
  const tokenRegex = /(\[\d+\]|\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  const parts = line.split(tokenRegex);
  return parts.map((part, idx) => {
    const citeMatch = part.match(/^\[(\d+)\]$/);
    if (citeMatch) {
      const rank = Number(citeMatch[1]);
      return (
        <button
          key={idx}
          type="button"
          onClick={() => onClickCitation(rank)}
          className="mx-0.5 inline-flex items-center rounded-md bg-cyan-500/20 px-1.5 py-0.5 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-400/40 transition hover:bg-cyan-500/35 hover:text-white"
          title={`Abrir evidência literal da fonte [${rank}]`}
        >
          [{rank}]
        </button>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={idx} className="font-semibold text-slate-100">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={idx}
          className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-xs text-emerald-300"
        >
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

function AnswerMarkdown({
  text,
  onClickCitation,
}: {
  text: string;
  onClickCitation: (rank: number) => void;
}) {
  const lines = text.split("\n");
  return (
    <div className="space-y-2 text-sm leading-relaxed text-slate-200">
      {lines.map((raw, i) => {
        const line = raw.trim();
        if (!line) return <div key={i} className="h-1" />;
        if (line.startsWith("### ")) {
          return (
            <h3 key={i} className="pt-1 text-base font-bold text-white">
              {renderInlineFormatting(line.slice(4), onClickCitation)}
            </h3>
          );
        }
        if (line.startsWith("#### ")) {
          return (
            <h4 key={i} className="pt-2 text-sm font-semibold uppercase tracking-wide text-cyan-300">
              {renderInlineFormatting(line.slice(5), onClickCitation)}
            </h4>
          );
        }
        if (line.startsWith("- ") || line.startsWith("* ")) {
          return (
            <div key={i} className="flex items-start gap-2 pl-1">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-400" />
              <p className="flex-1">{renderInlineFormatting(line.slice(2), onClickCitation)}</p>
            </div>
          );
        }
        return <p key={i}>{renderInlineFormatting(line, onClickCitation)}</p>;
      })}
    </div>
  );
}

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  documents,
  suggestedQuestions,
  turns,
  isQuerying,
  selectedDocIds,
  onToggleDocFilter,
  onClearDocFilter,
  onAsk,
  onInspectTurn,
  onClearChat,
}) => {
  const [input, setInput] = useState("");
  const [activeEvidence, setActiveEvidence] = useState<{
    turnId: string;
    chunk: RetrievedChunk;
  } | null>(null);

  const canSubmit = input.trim().length >= 3 && !isQuerying && documents.length > 0;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onAsk(input.trim());
    setInput("");
  };

  const handleExportMarkdown = () => {
    if (!turns.length) return;
    const sections = turns.map((t, idx) => {
      const r = t.response;
      const cites = r.citations
        .map(
          (c) =>
            `  - [${c.rank}] **${c.document_title}** (pág. ${c.page}) — Score Híbrido: \`${c.score_final.toFixed(3)}\` (BM25: \`${c.score_lexical.toFixed(3)}\`, Semântico: \`${c.score_semantic.toFixed(3)}\`)`
        )
        .join("\n");
      return [
        `## Consulta ${idx + 1}: ${r.question}`,
        `- **Provedor / Modelo:** ${r.provider_used} (\`${r.model_used}\`)`,
        `- **Modo de Busca:** ${r.diagnostics.retrieval_mode_label} | **Grounding:** ${r.diagnostics.grounding_score}% | **Tempo Total:** ${r.diagnostics.total_ms} ms`,
        "",
        r.answer,
        "",
        "### Fontes Citadas",
        cites,
      ].join("\n");
    });

    const md = [
      "# PaperRAG Studio — Relatório de Pesquisa Científica",
      "",
      `Gerado em: ${new Date().toLocaleString("pt-BR")}`,
      "",
      "---",
      "",
      sections.join("\n\n---\n\n"),
    ].join("\n");

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `paperrag-sessao-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const confidenceBadge = useMemo(
    () => (level: "alta" | "media" | "baixa", score: number) => {
      if (level === "alta") {
        return (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-semibold text-emerald-300 ring-1 ring-emerald-500/30">
            <CheckCircle2 className="h-3.5 w-3.5" /> Grounding {score.toFixed(0)}% (Alta)
          </span>
        );
      }
      if (level === "media") {
        return (
          <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-0.5 text-xs font-semibold text-amber-300 ring-1 ring-amber-500/30">
            <AlertTriangle className="h-3.5 w-3.5" /> Grounding {score.toFixed(0)}% (Média)
          </span>
        );
      }
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-rose-500/15 px-2.5 py-0.5 text-xs font-semibold text-rose-300 ring-1 ring-rose-500/30">
          <ShieldAlert className="h-3.5 w-3.5" /> Fora do Escopo ({score.toFixed(0)}%)
        </span>
      );
    },
    []
  );

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      {/* Coluna principal do Chat */}
      <div className="space-y-4 xl:col-span-8">
        {/* Barra de escopo documental + ações */}
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/70 px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
              <Filter className="h-3.5 w-3.5 text-cyan-400" /> Escopo:
            </span>
            <button
              type="button"
              onClick={onClearDocFilter}
              className={`rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                selectedDocIds.length === 0
                  ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-400/40"
                  : "bg-slate-800/80 text-slate-400 hover:text-slate-200"
              }`}
            >
              Todos os Artigos ({documents.length})
            </button>
            {documents.map((doc) => {
              const active = selectedDocIds.includes(doc.doc_id);
              return (
                <button
                  key={doc.doc_id}
                  type="button"
                  onClick={() => onToggleDocFilter(doc.doc_id)}
                  className={`max-w-[230px] truncate rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                    active
                      ? "bg-violet-500/20 text-violet-200 ring-1 ring-violet-400/40"
                      : "bg-slate-800/80 text-slate-400 hover:text-slate-200"
                  }`}
                  title={doc.title}
                >
                  {doc.title}
                </button>
              );
            })}
          </div>

          {turns.length > 0 && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleExportMarkdown}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/90 px-2.5 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-500/50 hover:text-cyan-200"
              >
                <Download className="h-3.5 w-3.5" /> Exportar .md
              </button>
              <button
                type="button"
                onClick={() => {
                  setActiveEvidence(null);
                  onClearChat();
                }}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/90 px-2.5 py-1.5 text-xs font-medium text-slate-400 transition hover:border-rose-500/40 hover:text-rose-300"
              >
                <Trash2 className="h-3.5 w-3.5" /> Limpar
              </button>
            </div>
          )}
        </div>

        {/* Estado inicial: Perguntas Guiadas */}
        {turns.length === 0 && (
          <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900/90 via-slate-900/70 to-cyan-950/20 p-6">
            <div className="mb-4 flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">
                  Pergunte aos Artigos Científicos com Transparência Total
                </h2>
                <p className="mt-1 text-sm text-slate-400">
                  Faça perguntas em <strong className="text-slate-200">Português</strong> mesmo sobre artigos em{" "}
                  <strong className="text-slate-200">Inglês</strong>. Nossa ponte bilíngue PT↔EN + busca híbrida (BM25 + LSA/Subword + RRF)
                  localiza os trechos exatos e exibe citações auditáveis página a página.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {suggestedQuestions.map((sq) => (
                <button
                  key={sq.id}
                  type="button"
                  disabled={isQuerying || documents.length === 0}
                  onClick={() => onAsk(sq.question)}
                  className={`group flex flex-col justify-between rounded-xl border p-3.5 text-left transition ${
                    sq.is_out_of_scope
                      ? "border-rose-500/30 bg-rose-950/15 hover:border-rose-400/60 hover:bg-rose-950/30"
                      : "border-slate-800 bg-slate-950/60 hover:border-cyan-500/50 hover:bg-slate-900"
                  }`}
                >
                  <div>
                    <span
                      className={`inline-block rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${
                        sq.is_out_of_scope
                          ? "bg-rose-500/20 text-rose-300"
                          : "bg-cyan-500/15 text-cyan-300"
                      }`}
                    >
                      {sq.category}
                    </span>
                    <p className="mt-2 text-sm font-medium text-slate-100 group-hover:text-white">
                      {sq.question}
                    </p>
                  </div>
                  <p className="mt-2.5 text-xs text-slate-400">{sq.hint}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Histórico de Consultas */}
        <div className="space-y-5">
          {turns.map((turn) => {
            const r = turn.response;
            return (
              <div
                key={turn.id}
                className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/75 shadow-lg shadow-black/20"
              >
                {/* Pergunta do Usuário */}
                <div className="border-b border-slate-800 bg-slate-950/60 px-5 py-3.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
                      Pergunta do Pesquisador
                    </span>
                    <span className="text-xs text-slate-500">{turn.timestamp}</span>
                  </div>
                  <p className="mt-1 text-base font-semibold text-white">{r.question}</p>
                </div>

                {/* Corpo da Resposta */}
                <div className="space-y-4 p-5">
                  {/* Aviso de fallback se houver */}
                  {r.fallback_notice && (
                    <div className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3.5 py-2.5 text-xs text-amber-200">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                      <span>{r.fallback_notice}</span>
                    </div>
                  )}

                  {/* Telemetria da Resposta */}
                  <div className="flex flex-wrap items-center gap-2">
                    {confidenceBadge(r.diagnostics.confidence_level, r.diagnostics.grounding_score)}
                    <span className="inline-flex items-center gap-1 rounded-full bg-slate-800 px-2.5 py-0.5 text-xs text-slate-300">
                      <Clock className="h-3 w-3 text-cyan-400" /> {r.diagnostics.total_ms.toFixed(0)} ms
                      <span className="text-slate-500">
                        (retrieval {r.diagnostics.retrieval_ms.toFixed(0)} ms)
                      </span>
                    </span>
                    <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs text-slate-300">
                      {r.diagnostics.retrieval_mode_label}
                    </span>
                    <span className="rounded-full bg-violet-500/15 px-2.5 py-0.5 text-xs text-violet-300 ring-1 ring-violet-500/30">
                      {r.model_used}
                    </span>
                    <button
                      type="button"
                      onClick={() => onInspectTurn(turn.id)}
                      className="ml-auto inline-flex items-center gap-1.5 rounded-lg bg-cyan-500/15 px-3 py-1 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-500/30 transition hover:bg-cyan-500/25 hover:text-white"
                    >
                      <Microscope className="h-3.5 w-3.5" /> Abrir no Raio-X
                    </button>
                  </div>

                  {/* Texto da Resposta com citações clicáveis */}
                  <div className="rounded-xl border border-slate-800/80 bg-slate-950/50 p-4">
                    <AnswerMarkdown
                      text={r.answer}
                      onClickCitation={(rank) => {
                        const found = r.citations.find((c) => c.rank === rank);
                        if (found) setActiveEvidence({ turnId: turn.id, chunk: found });
                      }}
                    />
                  </div>

                  {/* Cartões Compactos das Fontes Citadas */}
                  <div>
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Evidências Recuperadas ({r.citations.length} chunks em {r.diagnostics.unique_docs_cited} artigo(s),{" "}
                      {r.diagnostics.unique_pages_cited} página(s)) — clique para auditar o trecho completo:
                    </p>
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {r.citations.map((c) => {
                        const isSelected =
                          activeEvidence?.turnId === turn.id && activeEvidence?.chunk.chunk_id === c.chunk_id;
                        return (
                          <button
                            key={c.chunk_id}
                            type="button"
                            onClick={() => setActiveEvidence({ turnId: turn.id, chunk: c })}
                            className={`rounded-xl border p-3 text-left transition ${
                              isSelected
                                ? "border-cyan-400 bg-cyan-950/30 ring-1 ring-cyan-400/40"
                                : "border-slate-800 bg-slate-950/60 hover:border-slate-600"
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="inline-flex items-center gap-1.5 text-xs font-bold text-cyan-300">
                                <span className="rounded bg-cyan-500/20 px-1.5 py-0.5">[{c.rank}]</span>
                                Pág. {c.page}
                              </span>
                              <span className="font-mono text-[11px] text-emerald-300">
                                Score {c.score_final.toFixed(3)}
                              </span>
                            </div>
                            <p className="mt-1 truncate text-xs font-semibold text-slate-200">
                              {c.document_title}
                            </p>
                            <p className="mt-1 line-clamp-2 text-xs text-slate-400">
                              {c.highlighted_excerpt}
                            </p>
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

        {/* Formulário de Entrada */}
        <form
          onSubmit={handleSubmit}
          className="sticky bottom-3 rounded-2xl border border-slate-700/80 bg-slate-900/95 p-3 shadow-2xl shadow-black/50 backdrop-blur"
        >
          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isQuerying || documents.length === 0}
              placeholder={
                documents.length === 0
                  ? "Carregue PDFs na aba Biblioteca & Chunking para iniciar..."
                  : "Digite sua pergunta científica em Português ou Inglês (ex.: Como o TWFE falha com tratamento contínuo?)"
              }
              className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
            />
            <button
              type="submit"
              disabled={!canSubmit}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-emerald-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
            >
              <Send className="h-4 w-4" />
              {isQuerying ? "Buscando..." : "Perguntar"}
            </button>
          </div>
          {turns.length > 0 && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] text-slate-500">Sugestões rápidas:</span>
              {suggestedQuestions.slice(0, 3).map((sq) => (
                <button
                  key={sq.id}
                  type="button"
                  disabled={isQuerying}
                  onClick={() => onAsk(sq.question)}
                  className="max-w-[260px] truncate rounded-md border border-slate-800 bg-slate-950 px-2 py-0.5 text-[11px] text-slate-400 transition hover:border-cyan-500/40 hover:text-cyan-200"
                  title={sq.question}
                >
                  {sq.category}
                </button>
              ))}
            </div>
          )}
        </form>
      </div>

      {/* Painel Lateral Direito — Inspetor de Evidência Literal */}
      <div className="xl:col-span-4">
        <div className="sticky top-20 rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
          <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-3">
            <BookOpen className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">Inspetor de Evidência Literal</h3>
          </div>

          {!activeEvidence ? (
            <div className="space-y-3 py-8 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-800/80 text-slate-400">
                <FileText className="h-6 w-6" />
              </div>
              <p className="text-sm font-medium text-slate-300">
                Clique em qualquer citação <span className="text-cyan-300">[1]</span>,{" "}
                <span className="text-cyan-300">[2]</span>...
              </p>
              <p className="text-xs leading-relaxed text-slate-500">
                Aqui você audita o texto integral do chunk recuperado no PDF original, com destaque automático dos termos
                lexicais e bilíngues casados, número exato da página e decomposição dos scores BM25 vs. Semântico.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="rounded-xl border border-cyan-500/30 bg-cyan-950/20 p-3.5">
                <div className="flex items-center justify-between">
                  <span className="rounded-md bg-cyan-500/20 px-2 py-0.5 text-xs font-bold text-cyan-200">
                    Fonte [{activeEvidence.chunk.rank}]
                  </span>
                  <span className="text-xs font-semibold text-slate-300">
                    Página {activeEvidence.chunk.page} • Chunk #{activeEvidence.chunk.chunk_index}
                  </span>
                </div>
                <p className="mt-2 text-sm font-bold text-white">{activeEvidence.chunk.document_title}</p>
                <p className="mt-0.5 truncate text-[11px] text-slate-400">
                  {activeEvidence.chunk.document_name}
                </p>
              </div>

              {/* Mini-placar de Scores */}
              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-center">
                  <p className="text-[10px] uppercase text-slate-400">Final RRF</p>
                  <p className="mt-0.5 font-mono text-sm font-bold text-emerald-300">
                    {activeEvidence.chunk.score_final.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-center">
                  <p className="text-[10px] uppercase text-slate-400">BM25 (#{activeEvidence.chunk.rank_lexical})</p>
                  <p className="mt-0.5 font-mono text-sm font-bold text-cyan-300">
                    {activeEvidence.chunk.score_lexical.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-center">
                  <p className="text-[10px] uppercase text-slate-400">
                    Semânt. (#{activeEvidence.chunk.rank_semantic})
                  </p>
                  <p className="mt-0.5 font-mono text-sm font-bold text-violet-300">
                    {activeEvidence.chunk.score_semantic.toFixed(3)}
                  </p>
                </div>
              </div>

              {/* Termos Casados */}
              {activeEvidence.chunk.matched_terms.length > 0 && (
                <div>
                  <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                    Termos Casados neste Chunk
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {activeEvidence.chunk.matched_terms.map((term) => (
                      <span
                        key={term}
                        className="rounded-md bg-cyan-500/15 px-2 py-0.5 font-mono text-[11px] text-cyan-200 ring-1 ring-cyan-500/30"
                      >
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Texto Completo do Chunk */}
              <div>
                <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Trecho Literal Extraído do PDF
                </p>
                <div className="max-h-[420px] overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/80 p-3.5 text-xs leading-relaxed text-slate-200">
                  {highlightTermsInText(
                    activeEvidence.chunk.text,
                    activeEvidence.chunk.matched_terms
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
