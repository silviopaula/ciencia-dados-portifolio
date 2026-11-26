import React, { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Check, Code2, Copy, Languages, Layers, Microscope, Sparkles } from "lucide-react";
import type { ChatTurn } from "../types";

interface RetrievalXRayProps {
  turns: ChatTurn[];
  selectedTurnId: string | null;
  onSelectTurn: (turnId: string) => void;
  onGoToChat: () => void;
}

export const RetrievalXRay: React.FC<RetrievalXRayProps> = ({
  turns,
  selectedTurnId,
  onSelectTurn,
  onGoToChat,
}) => {
  const [copied, setCopied] = useState(false);

  if (!turns.length) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-10 text-center">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30">
          <Microscope className="h-7 w-7" />
        </div>
        <h2 className="mt-4 text-lg font-bold text-white">Raio-X do Retrieval aguardando sua primeira consulta</h2>
        <p className="mx-auto mt-2 max-w-xl text-sm text-slate-400">
          Envie uma pergunta na aba <strong className="text-slate-200">Chat Científico</strong> para inspecionar em tempo real a
          expansão bilíngue PT↔EN, o placar comparativo <strong className="text-cyan-300">BM25 vs. Semântico vs. RRF</strong> e o{" "}
          <strong className="text-emerald-300">Prompt Grounded exato</strong> montado para o LLM.
        </p>
        <button
          type="button"
          onClick={onGoToChat}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400"
        >
          <Sparkles className="h-4 w-4" /> Ir para o Chat Científico
        </button>
      </div>
    );
  }

  const activeTurn = turns.find((t) => t.id === selectedTurnId) ?? turns[turns.length - 1];
  const r = activeTurn.response;
  const diag = r.diagnostics;

  const chartData = r.citations.map((c) => ({
    label: `[${c.rank}] p.${c.page}`,
    BM25: Number((c.score_lexical * 100).toFixed(1)),
    Semantico: Number((c.score_semantic * 100).toFixed(1)),
    HibridoRRF: Number((c.score_final * 100).toFixed(1)),
  }));

  const fullPromptText = `=== SYSTEM PROMPT ===\n${r.prompt_inspection.system_prompt}\n\n=== USER PROMPT (CONTEXTO + PERGUNTA) ===\n${r.prompt_inspection.user_prompt}`;

  const handleCopyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(fullPromptText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignora restrição de clipboard
    }
  };

  return (
    <div className="space-y-5">
      {/* Seletor de Consulta do Histórico */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-cyan-400">
            Auditoria Transparente do Pipeline RAG
          </p>
          <h2 className="mt-0.5 text-base font-bold text-white">{r.question}</h2>
        </div>
        {turns.length > 1 && (
          <select
            value={activeTurn.id}
            onChange={(e) => onSelectTurn(e.target.value)}
            className="max-w-xs rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
          >
            {turns.map((t, i) => (
              <option key={t.id} value={t.id}>
                #{i + 1}: {t.response.question.slice(0, 58)}...
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Etapa 1: Ponte Bilíngue PT <-> EN & Diagnóstico da Consulta */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-7">
          <div className="mb-3 flex items-center gap-2">
            <Languages className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">
              1. Análise da Consulta & Ponte Bilíngue (PT ↔ EN)
            </h3>
            <span className="ml-auto rounded-md bg-slate-800 px-2 py-0.5 text-[11px] font-semibold text-slate-300">
              Idioma detectado: {diag.detected_language}
            </span>
          </div>
          <p className="mb-3 text-xs leading-relaxed text-slate-400">
            Artigos científicos costumam estar em inglês enquanto o pesquisador pergunta em português. Para evitar queda
            de revocação (<em>vocabulary mismatch</em>), o motor normaliza os tokens da pergunta e expande conceitos
            técnicos equivalentes em inglês com peso calibrado (0.78×).
          </p>

          <div className="space-y-3">
            <div>
              <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                Tokens Normalizados da Pergunta ({diag.original_tokens.length})
              </p>
              <div className="flex flex-wrap gap-1.5">
                {diag.original_tokens.map((tok) => (
                  <span
                    key={tok}
                    className="rounded-md bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-200 ring-1 ring-slate-700"
                  >
                    {tok}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-cyan-400">
                Expansão Semântica / Bilíngue Aplicada ({diag.expanded_terms.length})
              </p>
              {diag.expanded_terms.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {diag.expanded_terms.map((tok) => (
                    <span
                      key={tok}
                      className="rounded-md bg-cyan-500/15 px-2 py-0.5 font-mono text-xs text-cyan-200 ring-1 ring-cyan-500/30"
                    >
                      +{tok}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-xs italic text-slate-500">
                  Nenhuma expansão PT→EN adicional necessária (consulta já em inglês ou termos literais).
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Mini KPIs do Retrieval */}
        <div className="grid grid-cols-2 gap-3 lg:col-span-5">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <p className="text-[11px] uppercase tracking-wide text-slate-400">Modo de Fusão</p>
            <p className="mt-1 text-sm font-bold text-cyan-300">{diag.retrieval_mode_label}</p>
            <p className="mt-1 text-[11px] text-slate-500">
              α = {diag.alpha.toFixed(2)} • {diag.use_mmr ? "MMR Ativo (diversidade)" : "Ranking Direto"}
            </p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <p className="text-[11px] uppercase tracking-wide text-slate-400">Grounding Score</p>
            <p
              className={`mt-1 text-2xl font-extrabold ${
                diag.confidence_level === "alta"
                  ? "text-emerald-300"
                  : diag.confidence_level === "media"
                    ? "text-amber-300"
                    : "text-rose-300"
              }`}
            >
              {diag.grounding_score.toFixed(1)}%
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              {diag.out_of_scope_warning ? "Guardrail anti-alucinação ativado" : `Confiança ${diag.confidence_level}`}
            </p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <p className="text-[11px] uppercase tracking-wide text-slate-400">Chunks Avaliados</p>
            <p className="mt-1 text-2xl font-extrabold text-white">{diag.candidates_evaluated}</p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              Latência busca: {diag.retrieval_ms.toFixed(1)} ms
            </p>
          </div>
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <p className="text-[11px] uppercase tracking-wide text-slate-400">Janela de Prompt</p>
            <p className="mt-1 text-2xl font-extrabold text-violet-300">
              ~{r.prompt_inspection.estimated_tokens}
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              tokens est. ({r.prompt_inspection.context_chars.toLocaleString("pt-BR")} chars)
            </p>
          </div>
        </div>
      </div>

      {/* Etapa 2: Gráfico Comparativo Recharts BM25 vs Semântico vs Híbrido */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-white">
              2. Decomposição dos Scores por Chunk: Lexical (BM25) vs. Semântico (LSA/Char) vs. Híbrido RRF
            </h3>
          </div>
          <span className="text-xs text-slate-400">Escala normalizada (0–100)</span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="label" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  borderRadius: "12px",
                  color: "#f8fafc",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar dataKey="BM25" name="BM25 Lexical" fill="#22d3ee" radius={[6, 6, 0, 0]} />
              <Bar dataKey="Semantico" name="Semântico Vetorial" fill="#a78bfa" radius={[6, 6, 0, 0]} />
              <Bar dataKey="HibridoRRF" name="Score Híbrido Final" fill="#34d399" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Tabela de Ranks */}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] uppercase text-slate-400">
                <th className="py-2 pr-3">Cite</th>
                <th className="py-2 pr-3">Artigo & Página</th>
                <th className="py-2 pr-3 text-right">Rank BM25</th>
                <th className="py-2 pr-3 text-right">Rank Semânt.</th>
                <th className="py-2 pr-3 text-right">RRF</th>
                <th className="py-2 pr-3 text-right">Score Final</th>
                <th className="py-2">Termos Casados</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {r.citations.map((c) => (
                <tr key={c.chunk_id} className="hover:bg-slate-800/30">
                  <td className="py-2.5 pr-3 font-bold text-cyan-300">[{c.rank}]</td>
                  <td className="max-w-[260px] py-2.5 pr-3">
                    <p className="truncate font-medium text-slate-200">{c.document_title}</p>
                    <p className="text-[11px] text-slate-500">
                      Pág. {c.page} • Chunk #{c.chunk_index}
                    </p>
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-cyan-300">#{c.rank_lexical}</td>
                  <td className="py-2.5 pr-3 text-right font-mono text-violet-300">#{c.rank_semantic}</td>
                  <td className="py-2.5 pr-3 text-right font-mono text-slate-300">{c.score_rrf.toFixed(4)}</td>
                  <td className="py-2.5 pr-3 text-right font-mono font-bold text-emerald-300">
                    {c.score_final.toFixed(3)}
                  </td>
                  <td className="py-2.5">
                    <div className="flex flex-wrap gap-1">
                      {c.matched_terms.slice(0, 6).map((t) => (
                        <span key={t} className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
                          {t}
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Etapa 3: Inspetor de Prompt Grounded ("Por Baixo dos Panos") */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Code2 className="h-4 w-4 text-violet-400" />
            <h3 className="text-sm font-bold text-white">
              3. Por Baixo dos Panos: Prompt Grounded Exato Injetado no LLM
            </h3>
          </div>
          <button
            type="button"
            onClick={handleCopyPrompt}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-500/50 hover:text-cyan-200"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copiado!" : "Copiar Prompt Completo"}
          </button>
        </div>

        <p className="mb-3 text-xs text-slate-400">
          O segredo do RAG não é "treinar o LLM" nos PDFs, mas sim <strong className="text-slate-200">injetar os chunks mais relevantes na janela de contexto</strong> com instruções estritas de citação e recusa de alucinação:
        </p>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-violet-300">
              System Prompt (Regras de Grounding)
            </p>
            <pre className="h-72 overflow-y-auto whitespace-pre-wrap rounded-xl border border-violet-500/25 bg-slate-950 p-3.5 font-mono text-[11px] leading-relaxed text-violet-100">
              {r.prompt_inspection.system_prompt}
            </pre>
          </div>
          <div className="lg:col-span-8">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-300">
              User Prompt (Contexto Recuperado [1..{r.citations.length}] + Pergunta)
            </p>
            <pre className="h-72 overflow-y-auto whitespace-pre-wrap rounded-xl border border-emerald-500/25 bg-slate-950 p-3.5 font-mono text-[11px] leading-relaxed text-slate-200">
              {r.prompt_inspection.user_prompt}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
