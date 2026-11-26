import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Cpu,
  Database,
  KeyRound,
  Layers,
  MessageSquareText,
  Microscope,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import {
  deleteDocument,
  fetchHealth,
  fetchLibrary,
  fetchProviders,
  loadDemoDocuments,
  rebuildIndex,
  runRagQuery,
  uploadDocuments,
} from "./api";
import { ArchitectureGuide } from "./components/ArchitectureGuide";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { LibraryChunkLab } from "./components/LibraryChunkLab";
import { RetrievalXRay } from "./components/RetrievalXRay";
import type {
  ChatTurn,
  HealthResponse,
  LibraryResponse,
  ProviderId,
  ProviderInfo,
} from "./types";

type TabId = "chat" | "xray" | "library" | "guide";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("chat");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [library, setLibrary] = useState<LibraryResponse | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [ollamaUrl, setOllamaUrl] = useState("http://127.0.0.1:11434");

  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);

  // Parâmetros do Pipeline RAG (Sidebar)
  const [alpha, setAlpha] = useState(0.55);
  const [topK, setTopK] = useState(5);
  const [useMmr, setUseMmr] = useState(true);
  const [responseLang, setResponseLang] = useState<"pt-BR" | "en">("pt-BR");
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // Provedor de Geração
  const [provider, setProvider] = useState<ProviderId>("extractive");
  const [model, setModel] = useState<string>("extractive-grounded-v2");
  const [apiKey, setApiKey] = useState<string>("");
  const [temperature, setTemperature] = useState<number>(0.15);

  // Histórico do Chat & Raio-X
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [inspectedTurnId, setInspectedTurnId] = useState<string | null>(null);

  const bootstrap = useCallback(async () => {
    try {
      const h = await fetchHealth();
      if (h.app_name !== "PaperRAG Studio API") {
        throw new Error(
          `Outro serviço está respondendo na porta da API (${h.app_name || "serviço desconhecido"}). ` +
            "Reinicie via iniciar_app.bat ou suba o PaperRAG na porta 8001."
        );
      }
      const [lib, prov] = await Promise.all([fetchLibrary(), fetchProviders()]);
      setHealth(h);
      setApiOnline(true);
      setLibrary(lib);
      setProviders(prov.providers);
      setOllamaUrl(prov.ollama_url);
      setErrorMsg(null);
    } catch (err) {
      setApiOnline(false);
      setErrorMsg(
        err instanceof Error
          ? `Backend FastAPI indisponível: ${err.message}. Inicie o servidor Python na porta 8001.`
          : "Backend FastAPI indisponível na porta 8001."
      );
    }
  }, []);

  useEffect(() => {
    bootstrap();
  }, [bootstrap]);

  const currentProviderInfo = useMemo(
    () => providers.find((p) => p.id === provider),
    [providers, provider]
  );

  const handleSelectProvider = (next: ProviderId) => {
    setProvider(next);
    const info = providers.find((p) => p.id === next);
    if (info) {
      setModel(info.default_model);
    }
  };

  const handleToggleDocFilter = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
    );
  };

  const handleAsk = async (question: string) => {
    setIsQuerying(true);
    setErrorMsg(null);
    try {
      const res = await runRagQuery({
        question,
        top_k: topK,
        alpha,
        use_mmr: useMmr,
        mmr_lambda: 0.72,
        doc_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
        provider,
        model: model || null,
        api_key: provider === "openai" || provider === "gemini" ? apiKey.trim() || null : null,
        temperature,
        response_language: responseLang,
      });
      const turn: ChatTurn = {
        id: `turn-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        response: res,
      };
      setTurns((prev) => [...prev, turn]);
      setInspectedTurnId(turn.id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Erro ao executar consulta RAG.");
    } finally {
      setIsQuerying(false);
    }
  };

  const handleUploadFiles = async (files: File[]) => {
    setIsBusy(true);
    setErrorMsg(null);
    try {
      const updated = await uploadDocuments(files);
      setLibrary(updated);
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha no upload de PDFs.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleLoadDemo = async () => {
    setIsBusy(true);
    setErrorMsg(null);
    try {
      const updated = await loadDemoDocuments();
      setLibrary(updated);
      setSelectedDocIds([]);
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao recarregar artigos demo.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleDeleteDoc = async (docId: string) => {
    setIsBusy(true);
    setErrorMsg(null);
    try {
      const updated = await deleteDocument(docId);
      setLibrary(updated);
      setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao remover documento.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleRebuildIndex = async (chunkSize: number, chunkOverlap: number) => {
    setIsBusy(true);
    setErrorMsg(null);
    try {
      const updated = await rebuildIndex(chunkSize, chunkOverlap);
      setLibrary(updated);
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao reindexar chunks.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleRefreshProviders = async () => {
    try {
      const prov = await fetchProviders();
      setProviders(prov.providers);
      setOllamaUrl(prov.ollama_url);
    } catch {
      // ignora
    }
  };

  const alphaLabel =
    alpha <= 0.15
      ? "100% Lexical (BM25)"
      : alpha >= 0.85
        ? "100% Semântico (LSA)"
        : `Híbrido (${Math.round((1 - alpha) * 100)}% BM25 / ${Math.round(alpha * 100)}% Sem.)`;

  const tabs: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: "chat", label: "Chat Científico & Citações", icon: MessageSquareText },
    { id: "xray", label: "Raio-X do Retrieval", icon: Microscope },
    { id: "library", label: "Biblioteca & Chunking", icon: Database },
    { id: "guide", label: "Arquitetura RAG", icon: BookOpen },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header Fixo */}
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-5 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-emerald-500 text-slate-950 shadow-lg shadow-cyan-500/20">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold tracking-tight text-white">PaperRAG Studio</h1>
                <span className="rounded-md bg-cyan-500/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-cyan-300 ring-1 ring-cyan-500/30">
                  v2.0 Full-Stack
                </span>
              </div>
              <p className="text-xs text-slate-400">
                RAG Científico Híbrido & Didático • BM25 + Vetores Densos LSA/Subword + RRF + MMR
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            {library && (
              <div className="hidden items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 md:flex">
                <Layers className="h-3.5 w-3.5 text-cyan-400" />
                <span>
                  <strong className="text-white">{library.stats.document_count}</strong> PDFs •{" "}
                  <strong className="text-white">{library.stats.total_pages}</strong> págs •{" "}
                  <strong className="text-cyan-300">{library.stats.total_chunks}</strong> chunks
                </span>
              </div>
            )}

            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${
                apiOnline
                  ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-300"
                  : "border-rose-500/35 bg-rose-500/10 text-rose-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  apiOnline ? "animate-pulse bg-emerald-400" : "bg-rose-400"
                }`}
              />
              {apiOnline ? "API Python Online" : "API Offline"}
              {health?.ollama_available && (
                <span className="ml-1 rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] text-emerald-200">
                  +Ollama
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Barra de Abas */}
        <div className="mx-auto flex max-w-[1600px] gap-1 overflow-x-auto px-5 pb-2.5">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex shrink-0 items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold transition ${
                  active
                    ? "bg-cyan-500/15 text-cyan-200 ring-1 ring-cyan-500/40"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
                {tab.id === "xray" && turns.length > 0 && (
                  <span className="rounded-full bg-cyan-500/25 px-1.5 py-0.2 font-mono text-[10px] text-cyan-200">
                    {turns.length}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </header>

      {/* Corpo Principal: Sidebar de Controles + Workspace */}
      <div className="mx-auto grid max-w-[1600px] grid-cols-1 gap-6 px-5 py-6 lg:grid-cols-12">
        {/* Sidebar Esquerda de Parâmetros RAG */}
        <aside className="space-y-4 lg:col-span-3">
          {/* Card 1: Parâmetros de Recuperação Híbrida */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2.5">
              <SlidersHorizontal className="h-4 w-4 text-cyan-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Motor de Recuperação (Retrieval)
              </h2>
            </div>

            <div className="space-y-4">
              {/* Slider Alpha */}
              <div>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-300">Peso Semântico (α)</span>
                  <span className="font-mono font-bold text-cyan-300">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="mt-1.5 w-full accent-cyan-400"
                />
                <div className="mt-1 flex justify-between text-[10px] text-slate-500">
                  <span>BM25 Exato</span>
                  <span className="font-medium text-cyan-300">{alphaLabel}</span>
                  <span>Semântico</span>
                </div>
              </div>

              {/* Slider Top-K */}
              <div>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-300">Top-K Chunks Recuperados</span>
                  <span className="font-mono font-bold text-emerald-300">{topK}</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={10}
                  step={1}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="mt-1.5 w-full accent-emerald-400"
                />
              </div>

              {/* Toggle MMR */}
              <label className="flex cursor-pointer items-start justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div>
                  <p className="text-xs font-semibold text-slate-200">Diversidade MMR</p>
                  <p className="mt-0.5 text-[11px] text-slate-400">
                    Evita chunks redundantes da mesma página/parágrafo.
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={useMmr}
                  onChange={(e) => setUseMmr(e.target.checked)}
                  className="mt-1 h-4 w-4 accent-cyan-400"
                />
              </label>

              {/* Idioma da Resposta */}
              <div>
                <p className="mb-1.5 text-xs font-medium text-slate-300">Idioma do Prompt / Resposta</p>
                <div className="grid grid-cols-2 gap-2">
                  {(["pt-BR", "en"] as const).map((lang) => (
                    <button
                      key={lang}
                      type="button"
                      onClick={() => setResponseLang(lang)}
                      className={`rounded-lg py-1.5 text-xs font-semibold transition ${
                        responseLang === lang
                          ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-500/40"
                          : "bg-slate-950 text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {lang === "pt-BR" ? "Português (BR)" : "English"}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Card 2: Provedor de Geração */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2.5">
              <Cpu className="h-4 w-4 text-violet-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Camada de Geração (LLM)
              </h2>
            </div>

            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-300">Provedor</label>
                <select
                  value={provider}
                  onChange={(e) => handleSelectProvider(e.target.value as ProviderId)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 focus:border-violet-400 focus:outline-none"
                >
                  <option value="extractive">Extrativo Grounded (Offline — Padrão)</option>
                  <option value="ollama">Ollama Local (LLM Privado)</option>
                  <option value="openai">OpenAI API (Chave opcional)</option>
                  <option value="gemini">Google Gemini API (Chave opcional)</option>
                </select>
                {currentProviderInfo && (
                  <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
                    {currentProviderInfo.status_detail}
                  </p>
                )}
              </div>

              {provider !== "extractive" && currentProviderInfo && (
                <>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-slate-300">Modelo</label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs text-slate-100 focus:border-violet-400 focus:outline-none"
                    >
                      {currentProviderInfo.models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>

                  {(provider === "openai" || provider === "gemini") && (
                    <div>
                      <label className="mb-1 flex items-center gap-1.5 text-xs font-medium text-slate-300">
                        <KeyRound className="h-3.5 w-3.5 text-amber-400" />
                        Chave de Sessão ({provider === "openai" ? "OpenAI" : "Gemini"})
                      </label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="Opcional se já configurada em variável de ambiente"
                        className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:border-amber-400 focus:outline-none"
                      />
                      <p className="mt-1 text-[10px] text-slate-500">
                        Mantida apenas em memória no navegador; nunca gravada em disco ou Git.
                      </p>
                    </div>
                  )}

                  <div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-300">Temperatura</span>
                      <span className="font-mono font-bold text-violet-300">{temperature.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={0.8}
                      step={0.05}
                      value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                      className="mt-1 w-full accent-violet-400"
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Card 3: Resumo do Índice Vetorial */}
          {library && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
              <p className="mb-2.5 text-xs font-bold uppercase tracking-wider text-slate-400">
                Telemetria do Índice Ativo
              </p>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Chunking</span>
                  <span className="font-mono text-slate-200">
                    {library.stats.chunk_size}c / {library.stats.chunk_overlap}c
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Vocabulário BM25</span>
                  <span className="font-mono text-cyan-300">
                    {library.stats.vocab_size.toLocaleString("pt-BR")} termos
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Espaço Semântico LSA</span>
                  <span className="font-mono text-violet-300">{library.stats.lsa_dimensions} dimensões</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Tempo de Indexação</span>
                  <span className="font-mono text-emerald-300">{library.stats.build_time_ms.toFixed(0)} ms</span>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* Área Principal da Aba Selecionada */}
        <main className="space-y-4 lg:col-span-9">
          {errorMsg && (
            <div className="flex items-start gap-3 rounded-2xl border border-rose-500/40 bg-rose-950/35 px-4 py-3 text-sm text-rose-200">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              <span className="flex-1">{errorMsg}</span>
              <button
                type="button"
                onClick={bootstrap}
                className="rounded-lg bg-rose-500/20 px-2.5 py-1 text-xs font-semibold text-rose-200 hover:bg-rose-500/30"
              >
                Tentar novamente
              </button>
            </div>
          )}

          {activeTab === "chat" && (
            <ChatWorkspace
              documents={library?.documents ?? []}
              suggestedQuestions={library?.suggested_questions ?? []}
              turns={turns}
              isQuerying={isQuerying}
              selectedDocIds={selectedDocIds}
              onToggleDocFilter={handleToggleDocFilter}
              onClearDocFilter={() => setSelectedDocIds([])}
              onAsk={handleAsk}
              onInspectTurn={(turnId) => {
                setInspectedTurnId(turnId);
                setActiveTab("xray");
              }}
              onClearChat={() => {
                setTurns([]);
                setInspectedTurnId(null);
              }}
            />
          )}

          {activeTab === "xray" && (
            <RetrievalXRay
              turns={turns}
              selectedTurnId={inspectedTurnId}
              onSelectTurn={setInspectedTurnId}
              onGoToChat={() => setActiveTab("chat")}
            />
          )}

          {activeTab === "library" && library && (
            <LibraryChunkLab
              documents={library.documents}
              stats={library.stats}
              isBusy={isBusy}
              onUploadFiles={handleUploadFiles}
              onLoadDemo={handleLoadDemo}
              onDeleteDoc={handleDeleteDoc}
              onRebuildIndex={handleRebuildIndex}
            />
          )}

          {activeTab === "guide" && (
            <ArchitectureGuide
              providers={providers}
              ollamaUrl={ollamaUrl}
              onRefreshProviders={handleRefreshProviders}
            />
          )}
        </main>
      </div>
    </div>
  );
}
