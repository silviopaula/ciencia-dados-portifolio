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
import { Check, Code2, Copy, Languages, Lock, ShieldCheck, Sparkles } from "lucide-react";
import type { ChatTurn } from "../types";

interface PrivacyXRayPanelProps {
  turns: ChatTurn[];
  selectedTurnId: string | null;
  onSelectTurn: (id: string) => void;
  onGoToTutor: () => void;
}

export const PrivacyXRayPanel: React.FC<PrivacyXRayPanelProps> = ({
  turns,
  selectedTurnId,
  onSelectTurn,
  onGoToTutor,
}) => {
  const [copied, setCopied] = useState(false);

  if (!turns.length) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-10 text-center">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30">
          <ShieldCheck className="h-7 w-7" />
        </div>
        <h2 className="mt-4 text-lg font-bold text-white">
          Auditoria de Privacidade &amp; Raio-X aguardando sua primeira consulta
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm text-slate-400">
          Envie uma pergunta no <strong className="text-slate-200">Tutor de Macroeconomia</strong> para auditar em tempo
          real quantos bytes ficaram locais vs. enviados a LLMs externos, ver a ponte bilíngue PT↔EN e o placar BM25 vs. Semântico.
        </p>
        <button
          type="button"
          onClick={onGoToTutor}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-amber-500 px-4 py-2.5 text-sm font-bold text-slate-950 hover:bg-amber-400"
        >
          <Sparkles className="h-4 w-4" /> Ir para o Tutor MacroRAG
        </button>
      </div>
    );
  }

  const turn = turns.find((t) => t.id === selectedTurnId) ?? turns[turns.length - 1];
  const r = turn.response;
  const p = r.privacy;
  const d = r.diagnostics;

  const chartData = r.citations.map((c) => ({
    label: `[${c.rank}] p.${c.page}`,
    BM25: Number((c.score_lexical * 100).toFixed(1)),
    Semantico: Number((c.score_semantic * 100).toFixed(1)),
    Hibrido: Number((c.score_final * 100).toFixed(1)),
  }));

  const fullPrompt = `=== SYSTEM PROMPT ===\n${r.prompt_inspection.system_prompt}\n\n=== USER PROMPT ===\n${r.prompt_inspection.user_prompt}`;

  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(fullPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignora
    }
  };

  return (
    <div className="space-y-5">
      {/* Cabeçalho + Seletor */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-400">
            Governança de Dados &amp; Transparência RAG
          </p>
          <h2 className="mt-0.5 text-base font-bold text-white">{r.question}</h2>
        </div>
        {turns.length > 1 && (
          <select
            value={turn.id}
            onChange={(e) => onSelectTurn(e.target.value)}
            className="max-w-xs rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-200"
          >
            {turns.map((t, idx) => (
              <option key={t.id} value={t.id}>
                #{idx + 1}: {t.response.question.slice(0, 56)}...
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Painel de Privacidade & Residência de Dados */}
      <div className="rounded-2xl border border-cyan-500/30 bg-slate-900/80 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Lock className="h-4 w-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">1. Auditoria de Privacidade (Local vs. Nuvem)</h3>
          </div>
          <span
            className={`rounded-full px-3 py-0.5 text-xs font-bold ${
              p.is_fully_local
                ? "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/35"
                : "bg-amber-500/20 text-amber-300 ring-1 ring-amber-500/35"
            }`}
          >
            {p.data_residency_label}
          </span>
        </div>

        <p className="mb-4 text-xs leading-relaxed text-slate-300">{p.summary_note}</p>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5">
            <p className="text-[10px] uppercase text-slate-400">Mantido 100% Local</p>
            <p className="mt-1 font-mono text-xl font-extrabold text-emerald-300">
              {p.local_corpus_chars.toLocaleString("pt-BR")} chars
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              {p.local_chunks_total.toLocaleString("pt-BR")} chunks + índices BM25/LSA locais
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5">
            <p className="text-[10px] uppercase text-slate-400">Enviado para API Externa</p>
            <p className="mt-1 font-mono text-xl font-extrabold text-cyan-300">
              {p.external_sent_chars.toLocaleString("pt-BR")} chars
            </p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              ~{p.external_sent_tokens_est} tokens ({p.external_exposure_pct.toFixed(3)}% do corpus)
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-3.5">
            <p className="text-[10px] uppercase text-slate-400">Provedor Executado</p>
            <p className="mt-1 font-mono text-base font-extrabold text-white">{r.model_used}</p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              {p.is_fully_local ? "Execução On-Premise / Offline" : "Apenas janela top-k enviada"}
            </p>
          </div>
        </div>
      </div>

      {/* Ponte Bilíngue + Gráfico Comparativo */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-5">
          <div className="mb-3 flex items-center gap-2">
            <Languages className="h-4 w-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">2. Ponte Bilíngue Macro (PT ↔ EN)</h3>
          </div>
          <div className="space-y-3">
            <div>
              <p className="mb-1.5 text-[11px] uppercase text-slate-400">Tokens Originais ({d.original_tokens.length})</p>
              <div className="flex flex-wrap gap-1.5">
                {d.original_tokens.map((t) => (
                  <span key={t} className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-200">
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="mb-1.5 text-[11px] uppercase text-amber-400">
                Termos Macro Expandidos ({d.expanded_terms.length})
              </p>
              <div className="flex flex-wrap gap-1.5">
                {d.expanded_terms.length > 0 ? (
                  d.expanded_terms.map((t) => (
                    <span
                      key={t}
                      className="rounded bg-amber-500/15 px-2 py-0.5 font-mono text-xs text-amber-200 ring-1 ring-amber-500/30"
                    >
                      +{t}
                    </span>
                  ))
                ) : (
                  <span className="text-xs italic text-slate-500">Sem expansão adicional necessária.</span>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5 lg:col-span-7">
          <h3 className="mb-3 text-sm font-bold text-white">
            3. Decomposição de Retrieval: BM25 vs. Semântico vs. Híbrido
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Bar dataKey="BM25" name="BM25 Lexical" fill="#22d3ee" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Semantico" name="Semântico LSA" fill="#a78bfa" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Hibrido" name="Final Híbrido" fill="#f59e0b" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Prompt Grounded Exato */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Code2 className="h-4 w-4 text-violet-400" />
            <h3 className="text-sm font-bold text-white">
              4. Prompt Grounded Completo (~{r.prompt_inspection.estimated_tokens} tokens)
            </h3>
          </div>
          <button
            type="button"
            onClick={copyPrompt}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-200 hover:border-amber-400/50"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copiado!" : "Copiar Prompt"}
          </button>
        </div>
        <pre className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-950 p-4 font-mono text-[11px] leading-relaxed text-slate-300">
          {fullPrompt}
        </pre>
      </div>
    </div>
  );
};
