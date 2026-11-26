# 📚 PaperRAG Studio — RAG Científico Híbrido, Transparente e Didático

Aplicação web **full-stack** (*FastAPI + Python* no backend e *React + TypeScript + Tailwind CSS + Recharts* no frontend) para **Retrieval-Augmented Generation (RAG)** sobre artigos científicos em PDF.

Projetado para ir muito além de uma "caixa-preta" de chat com PDFs, o **PaperRAG Studio** expõe de forma interativa e auditável **todas as 6 etapas de uma arquitetura RAG moderna**: limpeza tipográfica de PDFs LaTeX, chunking com visualização da zona de *overlap*, busca híbrida (**Okapi BM25 + Vetores Densos LSA/Subword + Reciprocal Rank Fusion + MMR**), ponte bilíngue **PT ↔ EN**, **guardrail anti-alucinação** e inspeção integral do *prompt grounded* montado para o LLM.

---

## 🎯 Principais Diferenciais Arquiteturais e Didáticos

1. **Zero Dependência Obrigatória de GPU, Docker ou Chave Paga (Funciona 100% Offline no 1º Clique)**:
   - O modo padrão **Extrativo Grounded Offline** já inicia com 2 artigos clássicos de econometria (*Callaway & Sant'Anna*) indexados em `~1,2 s` e responde em **`< 20 ms`** com citações clicáveis `[1]`, `[2]` vinculadas à página exata do PDF.
   - Suporta ainda conexão opcional com **Ollama Local** (detecção automática em `http://127.0.0.1:11434`), **OpenAI API** ou **Google Gemini API** (chave informada apenas em memória na sessão ou via variável de ambiente — nunca gravada em disco).
2. **Limpeza Tipográfica para PDFs Científicos (`NFKC` + Des-hifenização)**:
   - PDFs acadêmicos gerados em LaTeX frequentemente contêm ligaduras Unicode (`ﬁ`, `ﬀ`, `ﬂ`) e palavras quebradas por hífen no fim da linha (`identi-\nfication`). O pipeline normaliza todo o texto preservando o **número exato da página** de cada trecho.
3. **Busca Híbrida Real com Controle Interativo (`α`) e Diversidade (`MMR`)**:
   - **Canal Lexical — Okapi BM25** ($k_1 = 1{,}5$, $b = 0{,}75$): captura siglas e termos exatos (`ATT(g,t)`, `TWFE`, `ACRT`, `Strong Parallel Trends`).
   - **Canal Semântico — Vetores Densos LSA (`TruncatedSVD` 96D) + Subword Char N-grams (3–5)**: captura relações conceituais latentes e cognatos entre português e inglês.
   - **Ponte Bilíngue PT ↔ EN**: permite consultar em **Português** artigos escritos em **Inglês**, exibindo na aba **Raio-X do Retrieval** exatamente quais termos foram expandidos.
   - **Fusão via Reciprocal Rank Fusion (RRF) + Maximal Marginal Relevance (MMR)**: evita retornar múltiplos chunks redundantes da mesma página.
4. **Guardrail Anti-Alucinação Calibrado**:
   - Avalia sinais absolutos de relevância (sem inflar ruído com normalização *min-max* ingênua). Perguntas fora do escopo dos PDFs acionam o alerta de **evidência insuficiente**, demonstrando na prática como sistemas RAG evitam alucinações.
5. **Quatro Módulos Interativos na Interface React**:
   - **Chat Científico & Citações**: perguntas guiadas (incluindo teste anti-alucinação), filtro por artigo, citações clicáveis `[1]..[k]` que abrem o **Inspetor de Evidência Literal** com termos destacados e exportação da sessão em Markdown (`.md`).
   - **Raio-X do Retrieval**: gráfico comparativo `BM25 vs. Semântico vs. Híbrido RRF`, tabela de ranks individuais e **Inspetor do Prompt Real** (*System Prompt* + *Context Window* + estimativa de tokens).
   - **Biblioteca & Laboratório de Chunking**: upload *drag-and-drop* de múltiplos PDFs, sliders ao vivo de `chunk_size` e `chunk_overlap` com reindexação instantânea, histograma de tamanhos e **Explorador de Chunks com destaque visual em ciano da zona de overlap**.
   - **Arquitetura RAG**: guia interativo das 6 etapas, status ao vivo dos 4 provedores LLM e glossário de engenharia RAG.

---

## 📊 Corpus Demo Pré-Carregado & Métricas Validadas

Os dois artigos científicos reais mantidos em [`data/papers/`](./data/papers/) são carregados automaticamente na inicialização da API:

| Artigo Científico | Autores | Páginas | Palavras | Chunks (`1000c / 180c`) |
| :--- | :--- | :---: | :---: | :---: |
| *Difference-in-Differences with a Continuous Treatment* (2021) | Brantly Callaway, Andrew Goodman-Bacon & Pedro H. C. Sant'Anna | 40 | 16.902 | 151 |
| *Difference-in-Differences with Multiple Time Periods* (2019) | Brantly Callaway & Pedro H. C. Sant'Anna | 45 | 22.452 | 201 |
| **Total do Índice Híbrido** | **Vocab BM25: 2.912 termos • Espaço LSA: 96D • Build: `1.179 ms`** | **85** | **39.354** | **352** |

### Benchmark Validado nas Perguntas Guiadas (Português → PDFs em Inglês)

| Categoria da Consulta | Grounding Score | Guardrail | Latência Total | Top-1 Recuperado |
| :--- | :---: | :---: | :---: | :--- |
| Identificação Causal (*Strong Parallel Trends*) | **96,2%** (`alta`) | Aprovado | `15,7 ms` | *Continuous Treatment* — pág. 3 |
| Múltiplos Períodos (`ATT(g,t)` & agregação) | **96,5%** (`alta`) | Aprovado | `13,2 ms` | *Multiple Time Periods* — pág. 3 |
| Diagnóstico TWFE (viés de seleção e pesos) | **98,5%** (`alta`) | Aprovado | `15,7 ms` | *Continuous Treatment* — pág. 19 |
| Aplicação Empírica (salário mínimo / Medicare) | **95,3%** (`alta`) | Aprovado | `12,5 ms` | *Multiple Time Periods* — pág. 32 |
| **Teste Anti-Alucinação** (Petrobras 3T25 — fora do escopo) | **6,0%** (`baixa`) | **Bloqueado** | `12,0 ms` | *Recusa especular com transparência* |

---

## 🗂️ Estrutura do Projeto

```text
app-rag-artigos-cientificos-pdf/
├── data/
│   └── papers/                           # 2 PDFs científicos de demonstração (Callaway & Sant'Anna)
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # API REST FastAPI (endpoints + lifespan auto-ingest)
│   │   ├── rag_engine.py                 # Motor RAG Híbrido (NFKC, Chunking, BM25, LSA, RRF, MMR, LLMs)
│   │   └── schemas.py                    # Contratos tipados Pydantic v2
│   ├── tests/
│   │   └── test_api.py                   # Suíte automatizada (9/9 testes end-to-end)
│   └── requirements.txt                  # Dependências Python enxutas
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWorkspace.tsx         # Chat com citações clicáveis e Inspetor de Evidência Literal
│   │   │   ├── RetrievalXRay.tsx         # Raio-X (Ponte PT↔EN, Gráfico BM25 vs Semântico, Prompt)
│   │   │   ├── LibraryChunkLab.tsx       # Upload PDF, Sliders de Chunking, Histograma e Overlap
│   │   │   └── ArchitectureGuide.tsx     # Guia interativo 6 etapas, Provedores LLM e Glossário
│   │   ├── api.ts                        # Cliente HTTP tipado para /api/*
│   │   ├── types.ts                      # Interfaces TypeScript espelhadas com Pydantic v2
│   │   ├── App.tsx                       # Layout principal, sidebar de controle RAG e abas
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts                    # Proxy /api -> http://127.0.0.1:8001
│   └── tailwind.config.js
└── README.md
```

---

## 🔌 Endpoints da API FastAPI

Documentação interativa automática (Swagger UI) disponível em `http://127.0.0.1:8001/docs`.

| Método | Endpoint | Função |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Status da API, total de PDFs/chunks indexados e detecção do Ollama local |
| `GET` | `/api/providers` | Lista os 4 provedores (`extractive`, `ollama`, `openai`, `gemini`) e modelos disponíveis |
| `GET` | `/api/library` | Metadados dos artigos, estatísticas/histograma de chunks e perguntas guiadas |
| `POST` | `/api/documents/load-demo` | Recarrega os 2 artigos científicos de demonstração em `data/papers/` |
| `POST` | `/api/documents/upload` | Upload multipart de um ou mais PDFs com indexação imediata |
| `DELETE` | `/api/documents/{doc_id}` | Remove um artigo específico e reconstrói os índices |
| `POST` | `/api/index/rebuild` | Reaplica `chunk_size` e `chunk_overlap` e reconstrói BM25 + LSA |
| `GET` | `/api/chunks` | Lista paginada e filtrável de chunks com tamanho exato da zona de overlap |
| `POST` | `/api/query` | Executa o pipeline RAG completo retornando resposta, citações, diagnósticos e prompt |

---

## 🚀 Como Executar Localmente

### 1. Backend (Python + FastAPI)

```bash
cd app-rag-artigos-cientificos-pdf/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

### 2. Frontend (React + Vite + TypeScript)

Em outro terminal:

```bash
cd app-rag-artigos-cientificos-pdf/frontend
npm install
npm run dev
```

Acesse **`http://localhost:5174`** no navegador.

> **Opcional — Usando LLM Generativo Local via Ollama:** se quiser respostas generativas traduzidas/sintetizadas localmente com privacidade total, basta ter o [Ollama](https://ollama.com/) rodando (`ollama pull qwen2.5:7b`) e selecionar **Ollama Local** na barra lateral.

---

## ✅ Validação e Testes Automatizados

Execute a suíte de 9 testes *end-to-end* do backend (`TestClient` + PDFs reais + guardrail anti-alucinação + upload/delete):

```bash
cd app-rag-artigos-cientificos-pdf/backend
python -m unittest discover -s tests -v
```

E valide tipagem TypeScript estrita + *bundle* de produção do frontend:

```bash
cd app-rag-artigos-cientificos-pdf/frontend
npm run build
```

---

## 👤 Autor

**Silvio da Rosa Paula** — Doutor em Economia Aplicada | Cientista de Dados  
[LinkedIn](https://www.linkedin.com/in/silvio-paula/) • [Portfólio Web](https://silviopaula.github.io/)
