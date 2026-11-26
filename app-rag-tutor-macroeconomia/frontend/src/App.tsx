import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Cpu,
  FileUp,
  GraduationCap,
  KeyRound,
  Layers,
  Lock,
  MessageSquareText,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import {
  deleteDocument,
  fetchHealth,
  fetchLibrary,
  fetchProviders,
  generateFlashcards,
  loadDemoLibrary,
  runMacroQuery,
  uploadMaterials,
} from "./api";
import { ChapterMapLibraryPanel } from "./components/ChapterMapLibraryPanel";
import { FlashcardsPanel } from "./components/FlashcardsPanel";
import { PrivacyXRayPanel } from "./components/PrivacyXRayPanel";
import { TutorChatPanel } from "./components/TutorChatPanel";
import type {
  ChatTurn,
  FlashcardResponse,
  HealthResponse,
  LibraryResponse,
  ProviderId,
  ProviderInfo,
  StudyMode,
} from "./types";

type TabId = "tutor" | "chapters" | "flashcards" | "privacy";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("tutor");
  const [apiOnline, setApiOnline] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [library, setLibrary] = useState<LibraryResponse | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);

  // Controles do RAG Macroeconômico
  const [studyMode, setStudyMode] = useState<StudyMode>("tutor");
  const [alpha, setAlpha] = useState(0.5);
  const [topK, setTopK] = useState(6);
  const [useMmr, setUseMmr] = useState(true);
  const [activeChapterIds, setActiveChapterIds] = useState<string[]>([]);

  // Provedor LLM
  const [provider, setProvider] = useState<ProviderId>("extractive");
  const [model, setModel] = useState("macro-extractive-v2");
  const [apiKey, setApiKey] = useState("");
  const [temperature, setTemperature] = useState(0.15);

  // Histórico e Flashcards
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [inspectedTurnId, setInspectedTurnId] = useState<string | null>(null);
  const [fcChapterId, setFcChapterId] = useState("");
  const [flashcards, setFlashcards] = useState<FlashcardResponse | null>(null);
  const [isGeneratingFc, setIsGeneratingFc] = useState(false);

  const bootstrap = useCallback(async () => {
    try {
      const h = await fetchHealth();
      if (h.app_name !== "MacroRAG Tutor Studio API") {
        throw new Error(
          `Outro serviço está respondendo na porta da API (${h.app_name || "desconhecido"}). ` +
            "Reinicie via iniciar_app.bat ou suba o MacroRAG na porta 8003."
        );
      }
      const [lib, prov] = await Promise.all([fetchLibrary(), fetchProviders()]);
      setHealth(h);
      setApiOnline(true);
      setLibrary(lib);
      setProviders(prov.providers);
      setErrorMsg(null);
    } catch (err) {
      setApiOnline(false);
      setErrorMsg(
        err instanceof Error
          ? `Backend FastAPI indisponível: ${err.message}. Inicie o servidor Python na porta 8003.`
          : "Backend FastAPI indisponível na porta 8003."
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
    if (info) setModel(info.default_model);
  };

  const handleAsk = async (question: string, modeOverride?: StudyMode) => {
    const modeToUse = modeOverride ?? studyMode;
    if (modeOverride) setStudyMode(modeOverride);
    setIsQuerying(true);
    setErrorMsg(null);
    try {
      const res = await runMacroQuery({
        question,
        study_mode: modeToUse,
        top_k: topK,
        alpha,
        use_mmr: useMmr,
        doc_ids: null,
        chapter_ids: activeChapterIds.length > 0 ? activeChapterIds : null,
        provider,
        model: model || null,
        api_key: provider === "gemini" || provider === "openai" ? apiKey.trim() || null : null,
        temperature,
      });
      const turn: ChatTurn = {
        id: `mturn-${Date.now()}`,
        timestamp: new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        response: res,
      };
      setTurns((prev) => [...prev, turn]);
      setInspectedTurnId(turn.id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Erro ao consultar MacroRAG.");
    } finally {
      setIsQuerying(false);
    }
  };

  const handleUpload = async (files: File[]) => {
    setIsBusy(true);
    setErrorMsg(null);
    try {
      const updated = await uploadMaterials(files);
      setLibrary(updated);
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao processar arquivos.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleReloadDemo = async () => {
    setIsBusy(true);
    setErrorMsg(null);
    try {
      const updated = await loadDemoLibrary();
      setLibrary(updated);
      setActiveChapterIds([]);
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao recarregar livro demo.");
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
      setActiveChapterIds([]);
      setHealth(await fetchHealth());
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao remover documento.");
    } finally {
      setIsBusy(false);
    }
  };

  const handleGenerateFlashcards = async (chapterId: string, topic: string, count: number) => {
    setIsGeneratingFc(true);
    setErrorMsg(null);
    try {
      const res = await generateFlashcards({
        chapter_id: chapterId || null,
        topic: topic.trim() || null,
        count,
        provider,
      });
      setFlashcards(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Falha ao gerar flashcards.");
    } finally {
      setIsGeneratingFc(false);
    }
  };

  const tabs: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { id: "tutor", label: "Tutor de Macroeconomia", icon: MessageSquareText },
    { id: "chapters", label: "Mapa de Capítulos & Biblioteca", icon: BookOpen },
    { id: "flashcards", label: "Simulado & Flashcards", icon: GraduationCap },
    { id: "privacy", label: "Privacidade & Raio-X", icon: Lock },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header Fixo */}
      <header className="sticky top-0 z-30 border-b border-slate-800/90 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-5 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 text-slate-950 shadow-lg shadow-amber-500/20">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-extrabold tracking-tight text-white">MacroRAG Tutor Studio</h1>
                <span className="rounded-md bg-amber-500/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-amber-300 ring-1 ring-amber-500/30">
                  v2.0 Full-Stack
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Assistente RAG por Capítulos para Livros &amp; Notas de Macroeconomia • Cache Local + Flashcards + Privacidade
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-500/15 px-3.5 py-1.5 text-xs font-semibold text-amber-200 transition hover:bg-amber-500/25">
              <input
                type="file"
                accept=".pdf,.md,.txt"
                multiple
                disabled={isBusy}
                onChange={(e) => {
                  const files = e.target.files ? Array.from(e.target.files) : [];
                  e.target.value = "";
                  if (files.length) handleUpload(files);
                }}
                className="hidden"
              />
              <FileUp className="h-3.5 w-3.5" />
              Enviar PDF / MD / TXT
            </label>

            {library && (
              <div className="hidden items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 md:flex">
                <Layers className="h-3.5 w-3.5 text-amber-400" />
                <span>
                  <strong className="text-white">{library.stats.total_pages}</strong> págs •{" "}
                  <strong className="text-amber-300">{library.chapters.length}</strong> caps •{" "}
                  <strong className="text-white">{library.stats.total_chunks}</strong> chunks
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
              <span className={`h-2 w-2 rounded-full ${apiOnline ? "animate-pulse bg-emerald-400" : "bg-rose-400"}`} />
              {apiOnline ? "API Python Online (:8003)" : "API Offline"}
              {health?.ollama_available && (
                <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px]">+Ollama</span>
              )}
            </div>
          </div>
        </div>

        {/* Abas */}
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
                    ? "bg-amber-500/15 text-amber-200 ring-1 ring-amber-500/40"
                    : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </header>

      {/* Corpo */}
      <div className="mx-auto grid max-w-[1600px] grid-cols-1 gap-6 px-5 py-6 lg:grid-cols-12">
        {/* Sidebar Esquerda */}
        <aside className="space-y-4 lg:col-span-3">
          {/* Card 1: Retrieval Híbrido */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2">
              <SlidersHorizontal className="h-4 w-4 text-amber-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Recuperação Híbrida
              </h2>
            </div>

            <div className="space-y-4 text-xs">
              <div>
                <div className="flex justify-between">
                  <span className="text-slate-300">Peso Semântico (α)</span>
                  <span className="font-mono font-bold text-amber-300">{alpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="mt-1.5 w-full accent-amber-400"
                />
                <div className="mt-0.5 flex justify-between text-[10px] text-slate-500">
                  <span>BM25 Exato</span>
                  <span>Híbrido</span>
                  <span>Semântico LSA</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between">
                  <span className="text-slate-300">Top-K Passagens</span>
                  <span className="font-mono font-bold text-cyan-300">{topK}</span>
                </div>
                <input
                  type="range"
                  min={2}
                  max={10}
                  step={1}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="mt-1.5 w-full accent-cyan-400"
                />
              </div>

              <label className="flex cursor-pointer items-center justify-between rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-2">
                <span className="text-slate-300">Diversidade MMR entre páginas</span>
                <input
                  type="checkbox"
                  checked={useMmr}
                  onChange={(e) => setUseMmr(e.target.checked)}
                  className="h-4 w-4 accent-amber-400"
                />
              </label>
            </div>
          </div>

          {/* Card 2: Provedor LLM & Privacidade */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4">
            <div className="mb-3 flex items-center gap-2 border-b border-slate-800 pb-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Geração &amp; Privacidade
              </h2>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="mb-1 block text-slate-400">Provedor</label>
                <select
                  value={provider}
                  onChange={(e) => handleSelectProvider(e.target.value as ProviderId)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-amber-400 focus:outline-none"
                >
                  <option value="extractive">Extrativo Offline (100% Local — Padrão)</option>
                  <option value="gemini">Google Gemini API (Síntese PT-BR)</option>
                  <option value="ollama">Ollama Local (LLM On-Premise)</option>
                  <option value="openai">OpenAI API</option>
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
                    <label className="mb-1 block text-slate-400">Modelo</label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-slate-100"
                    >
                      {currentProviderInfo.models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>

                  {(provider === "gemini" || provider === "openai") && (
                    <div>
                      <label className="mb-1 flex items-center gap-1.5 text-slate-300">
                        <KeyRound className="h-3.5 w-3.5 text-amber-400" />
                        Chave de Sessão ({provider === "gemini" ? "Gemini" : "OpenAI"})
                      </label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="Opcional se já no ambiente"
                        className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-xs text-slate-100 placeholder-slate-500"
                      />
                    </div>
                  )}

                  <div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Temperatura</span>
                      <span className="font-mono font-bold text-amber-300">{temperature.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={0.8}
                      step={0.05}
                      value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                      className="mt-1 w-full accent-amber-400"
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Card 3: Telemetria do Corpus */}
          {library && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-4 text-xs">
              <p className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-400">
                Índice Local Persistente
              </p>
              <div className="space-y-1.5">
                <div className="flex justify-between">
                  <span className="text-slate-400">Páginas Indexadas</span>
                  <span className="font-mono font-bold text-white">{library.stats.total_pages}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Vocabulário BM25</span>
                  <span className="font-mono text-cyan-300">
                    {library.stats.vocab_size.toLocaleString("pt-BR")}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Vetores LSA</span>
                  <span className="font-mono text-violet-300">{library.stats.lsa_dimensions}D</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Build do Índice</span>
                  <span className="font-mono text-emerald-300">{library.stats.build_time_ms.toFixed(0)} ms</span>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* Workspace */}
        <main className="space-y-4 lg:col-span-9">
          {errorMsg && (
            <div className="flex items-start gap-3 rounded-2xl border border-rose-500/40 bg-rose-950/35 px-4 py-3 text-sm text-rose-200">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
              <span className="flex-1">{errorMsg}</span>
              <button
                type="button"
                onClick={bootstrap}
                className="rounded-lg bg-rose-500/20 px-2.5 py-1 text-xs font-semibold hover:bg-rose-500/30"
              >
                Tentar novamente
              </button>
            </div>
          )}

          {activeTab === "tutor" && (
            <TutorChatPanel
              suggestedPrompts={library?.suggested_prompts ?? []}
              chapters={library?.chapters ?? []}
              activeChapterIds={activeChapterIds}
              studyMode={studyMode}
              turns={turns}
              isQuerying={isQuerying}
              hasDocuments={(library?.documents.length ?? 0) > 0}
              onChangeStudyMode={setStudyMode}
              onClearChapterFilter={() => setActiveChapterIds([])}
              onAsk={handleAsk}
              onInspectTurn={(id) => {
                setInspectedTurnId(id);
                setActiveTab("privacy");
              }}
              onClearChat={() => {
                setTurns([]);
                setInspectedTurnId(null);
              }}
            />
          )}

          {activeTab === "chapters" && library && (
            <ChapterMapLibraryPanel
              documents={library.documents}
              chapters={library.chapters}
              stats={library.stats}
              isBusy={isBusy}
              onUploadFiles={handleUpload}
              onReloadDemo={handleReloadDemo}
              onDeleteDoc={handleDeleteDoc}
              onStudyChapter={(cid) => {
                setActiveChapterIds([cid]);
                setActiveTab("tutor");
              }}
              onFlashcardsChapter={(cid) => {
                setFcChapterId(cid);
                setActiveTab("flashcards");
                handleGenerateFlashcards(cid, "", 5);
              }}
            />
          )}

          {activeTab === "flashcards" && (
            <FlashcardsPanel
              chapters={library?.chapters ?? []}
              selectedChapterId={fcChapterId}
              flashcards={flashcards}
              isGenerating={isGeneratingFc}
              onSelectChapterId={setFcChapterId}
              onGenerate={handleGenerateFlashcards}
            />
          )}

          {activeTab === "privacy" && (
            <PrivacyXRayPanel
              turns={turns}
              selectedTurnId={inspectedTurnId}
              onSelectTurn={setInspectedTurnId}
              onGoToTutor={() => setActiveTab("tutor")}
            />
          )}
        </main>
      </div>
    </div>
  );
}
