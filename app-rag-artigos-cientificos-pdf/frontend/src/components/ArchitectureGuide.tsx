import React, { useState } from "react";
import {
  BookOpenCheck,
  CheckCircle2,
  Cpu,
  Database,
  FileSearch,
  FileText,
  RefreshCw,
  Scissors,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import type { ProviderInfo } from "../types";

interface ArchitectureGuideProps {
  providers: ProviderInfo[];
  ollamaUrl: string;
  onRefreshProviders: () => void;
}

const PIPELINE_STEPS = [
  {
    step: 1,
    title: "Ingestão & Limpeza Tipográfica",
    icon: FileText,
    color: "text-cyan-300 bg-cyan-500/15 ring-cyan-500/30",
    summary: "Extração página a página com normalização NFKC para PDFs científicos em LaTeX.",
    detail:
      "PDFs acadêmicos compilados em LaTeX contêm ligaduras Unicode (como ﬁ, ﬀ, ﬂ) e palavras hifenizadas em quebras de linha. Sem normalização NFKC e des-hifenização, buscas por 'difference' ou 'identification' falham silenciosamente. Cada página mantém seu índice exato para citações auditáveis.",
  },
  {
    step: 2,
    title: "Chunking Consciente de Sentenças + Overlap",
    icon: Scissors,
    color: "text-violet-300 bg-violet-500/15 ring-violet-500/30",
    summary: "Segmentação em blocos configuráveis rastreando a faixa exata de sobreposição.",
    detail:
      "Dividir artigos em chunks muito grandes dilui o sinal de relevância e encarece o contexto; chunks muito pequenos quebram demonstrações e definições. O overlap copia uma janela final do chunk anterior (destacada em ciano no Laboratório de Chunking) para garantir continuidade semântica.",
  },
  {
    step: 3,
    title: "Indexação Dupla (BM25 + LSA/Subword)",
    icon: Database,
    color: "text-emerald-300 bg-emerald-500/15 ring-emerald-500/30",
    summary: "Combina busca lexical exata (Okapi BM25) com vetores semânticos densos (SVD/LSA + Char N-grams).",
    detail:
      "Busca vetorial pura costuma errar siglas exatas como 'ATT(g,t)', 'ACRT' ou 'TWFE'. Já a busca lexical pura perde sinônimos e paráfrases. Nossa arquitetura indexa ambos os espaços: Okapi BM25 (k1=1.5, b=0.75) + Embeddings Densos via TruncatedSVD (LSA) combinados com n-gramas de caracteres (3–5) para cognatos PT↔EN.",
  },
  {
    step: 4,
    title: "Fusão Híbrida (RRF) & Diversidade (MMR)",
    icon: FileSearch,
    color: "text-amber-300 bg-amber-500/15 ring-amber-500/30",
    summary: "Reciprocal Rank Fusion (RRF) ponderado por α + Maximal Marginal Relevance.",
    detail:
      "O parâmetro α permite explorar todo o espectro (0 = 100% BM25, 0.55 = Híbrido, 1 = 100% Semântico). O RRF combina rankings independentes de escala, enquanto o MMR penaliza candidatos redundantes já cobertos por chunks selecionados — trazendo evidências de páginas e artigos diferentes.",
  },
  {
    step: 5,
    title: "Guardrail Anti-Alucinação & Prompt Grounded",
    icon: ShieldCheck,
    color: "text-rose-300 bg-rose-500/15 ring-rose-500/30",
    summary: "Calibração absoluta de relevância impede respostas inventadas fora do escopo.",
    detail:
      "Antes de gerar a resposta, o motor verifica limiares absolutos de BM25 e similaridade semântica (sem inflar ruído por normalização min-max ingênua). Se a pergunta fugir dos PDFs carregados, o guardrail alerta 'baixa confiança' e recusa especular. Todo prompt montado fica 100% visível na aba Raio-X.",
  },
  {
    step: 6,
    title: "Geração Multi-Provedor com Citações [1..k]",
    icon: Sparkles,
    color: "text-sky-300 bg-sky-500/15 ring-sky-500/30",
    summary: "Funciona 100% offline no modo Extrativo Grounded ou com Ollama, OpenAI e Gemini.",
    detail:
      "Diferente de protótipos que travam sem GPU ou chave paga, o modo Extrativo Grounded sintetiza evidências literais com citações clicáveis instantaneamente. Quando conectado ao Ollama local, OpenAI ou Gemini, o mesmo contexto curado alimenta o LLM para síntese e tradução fluente em português.",
  },
];

const GLOSSARY = [
  {
    term: "Okapi BM25",
    desc: "Algoritmo probabilístico de recuperação lexical que pondera frequência do termo (TF) com saturação não-linear (k1), raridade no corpus (IDF) e normalização pelo comprimento do chunk (b).",
  },
  {
    term: "LSA (TruncatedSVD)",
    desc: "Latent Semantic Analysis: projeta a matriz esparsa TF-IDF em um espaço vetorial denso de menor dimensão onde termos que co-ocorrem em contextos semelhantes ficam próximos em cosseno.",
  },
  {
    term: "Reciprocal Rank Fusion (RRF)",
    desc: "Método robusto de fusão de rankings (1 / (60 + rank)) que combina sinais de busca lexical e semântica sem exigir que ambos estejam na mesma distribuição estatística.",
  },
  {
    term: "MMR (Maximal Marginal Relevance)",
    desc: "Algoritmo guloso que equilibra relevância para a pergunta (λ) contra redundância em relação aos chunks já escolhidos (1 - λ), aumentando diversidade documental.",
  },
  {
    term: "Grounding / Fidelidade",
    desc: "Grau em que cada afirmação da resposta está ancorada em trechos recuperados verificáveis ([1], [2]), combatendo alucinações paramétricas dos LLMs.",
  },
  {
    term: "Ponte Bilíngue PT ↔ EN",
    desc: "Expansão controlada de termos científicos em português para seus equivalentes em inglês, somada a subword n-grams que reconhecem cognatos automaticamente.",
  },
];

export const ArchitectureGuide: React.FC<ArchitectureGuideProps> = ({
  providers,
  ollamaUrl,
  onRefreshProviders,
}) => {
  const [selectedStep, setSelectedStep] = useState(1);
  const current = PIPELINE_STEPS.find((s) => s.step === selectedStep) ?? PIPELINE_STEPS[0];

  return (
    <div className="space-y-5">
      {/* Fluxo Interativo de 6 Etapas */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-6">
        <div className="mb-4 flex items-center gap-2">
          <BookOpenCheck className="h-5 w-5 text-cyan-400" />
          <div>
            <h2 className="text-base font-bold text-white">
              Arquitetura RAG em 6 Etapas Observáveis (Clique em cada etapa)
            </h2>
            <p className="text-xs text-slate-400">
              Cada bloco abaixo corresponde diretamente ao código em execução no backend Python (FastAPI).
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3 lg:grid-cols-6">
          {PIPELINE_STEPS.map((st) => {
            const Icon = st.icon;
            const active = st.step === selectedStep;
            return (
              <button
                key={st.step}
                type="button"
                onClick={() => setSelectedStep(st.step)}
                className={`flex flex-col items-start rounded-xl border p-3.5 text-left transition ${
                  active
                    ? "border-cyan-400 bg-slate-950 ring-1 ring-cyan-400/40"
                    : "border-slate-800 bg-slate-950/50 hover:border-slate-700"
                }`}
              >
                <div className="flex w-full items-center justify-between">
                  <span className={`flex h-8 w-8 items-center justify-center rounded-lg ring-1 ${st.color}`}>
                    <Icon className="h-4 w-4" />
                  </span>
                  <span className="font-mono text-xs font-bold text-slate-500">0{st.step}</span>
                </div>
                <p className="mt-3 text-xs font-bold text-white">{st.title}</p>
                <p className="mt-1 line-clamp-2 text-[11px] text-slate-400">{st.summary}</p>
              </button>
            );
          })}
        </div>

        {/* Detalhe expandido da etapa selecionada */}
        <div className="mt-4 rounded-xl border border-cyan-500/25 bg-slate-950/80 p-5">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-cyan-500/20 px-2 py-0.5 font-mono text-xs font-bold text-cyan-300">
              Etapa 0{current.step}
            </span>
            <h3 className="text-sm font-bold text-white">{current.title}</h3>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-slate-300">{current.detail}</p>
        </div>
      </div>

      {/* Status dos Provedores de Geração */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-emerald-400" />
            <div>
              <h2 className="text-base font-bold text-white">Provedores de Geração (Offline + LLMs Opcionais)</h2>
              <p className="text-xs text-slate-400">
                Endpoint Ollama monitorado: <code className="text-cyan-300">{ollamaUrl}</code>
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onRefreshProviders}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-500/50 hover:text-cyan-200"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Revalidar Provedores
          </button>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {providers.map((p) => (
            <div key={p.id} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-bold text-white">{p.name}</h3>
                {p.available ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-semibold text-emerald-300">
                    <CheckCircle2 className="h-3 w-3" /> Disponível
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded-full bg-slate-800 px-2 py-0.5 text-[11px] font-semibold text-slate-400">
                    <XCircle className="h-3 w-3" /> Inativo
                  </span>
                )}
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{p.description}</p>
              <p className="mt-2 text-[11px] font-medium text-cyan-300">{p.status_detail}</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {p.models.slice(0, 4).map((m) => (
                  <span key={m} className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[10px] text-slate-300">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Glossário Técnico */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-6">
        <h2 className="mb-3 text-base font-bold text-white">Glossário Didático de Engenharia RAG</h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
          {GLOSSARY.map((item) => (
            <div key={item.term} className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <h3 className="font-mono text-xs font-bold text-cyan-300">{item.term}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
