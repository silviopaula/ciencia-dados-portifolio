# 🏛️ MacroRAG Tutor Studio — Assistente RAG por Capítulos para Macroeconomia

Aplicação web **full-stack** (*Python + FastAPI* no backend e *React + TypeScript + Tailwind CSS + Recharts* no frontend) especializada em **estudo, pesquisa e revisão de livros-texto extensos e notas de aula de Macroeconomia** (`.pdf`, `.md` e `.txt`).

Refatorado a partir do protótipo legado em Streamlit + Chroma temporário para uma arquitetura rápida, persistente, transparente em privacidade e pronta para escalar sobre obras de **800+ páginas** — pré-carregado com ***Foundations of Modern Macroeconomics*** (Ben J. Heijdra, Oxford University Press — **816 páginas**).

---

## 🎯 Principais Diferenciais Arquiteturais e Didáticos

1. **Indexação Hierárquica por Capítulos + Cache Local Comprimido (`.json.gz`):**
   - Extrai automaticamente a árvore de **Partes e Capítulos** do sumário PDF (*bookmarks*) — ou cabeçalhos `#`/`##` de notas em Markdown — rotulando cada chunk com **`Capítulo`, `Parte` e `Página`**.
   - Persiste a extração limpa (`NFKC`, sem ligaduras LaTeX/OUP e ignorando sumário/índice remissivo) em cache local comprimido (`backend/.cache/`), reduzindo o carregamento das 816 páginas de `~17 s` para **`< 0,05 s`** nas execuções seguintes.
2. **Busca Híbrida de Baixa Latência (~30 ms em 2.468 Chunks):**
   - **Canal Lexical:** Okapi BM25 com **índice invertido** (`9.855` termos únicos) + **Ponte Bilíngue Macroeconômica PT ↔ EN** (*Equivalência Ricardiana, Inconsistência Temporal, Curva de Phillips, Overshooting de Dornbusch, Novo Keynesiano, RBC, Solow/Ramsey, OLG*).
   - **Canal Semântico:** vetores densos **LSA (`TruncatedSVD` 64D)** + *character subword n-grams* (3–4).
   - **Fusão RRF + Diversidade MMR + Filtro por Capítulo/Documento** + **Guardrail Anti-Alucinação**.
3. **Quatro Modos Didáticos & Estudo Ativo:**
   - **Tutor Didático:** explica intuição, mecanismos e implicações de política econômica com citações clicáveis `[Capítulo, pág.]`.
   - **Comparador de Escolas & Modelos:** contrasta tradições (ex.: *Novo Keynesiano vs. RBC*, *Solow vs. Crescimento Endógeno*) diversificando passagens entre capítulos via MMR.
   - **Verificador Técnico:** reforça precisão lexical para localizar definições e páginas exatas.
   - **Gerador de Simulado & Flashcards por Capítulo:** cria cartões de estudo e questões discursivas de pós-graduação a partir de qualquer capítulo do livro ou nota enviada, com gabarito ancorado na página original e controle de progresso (*Dominado*).
4. **Auditoria Explícita de Privacidade (Local vs. Nuvem):**
   - Funciona **100% offline** no modo **Extrativo Grounded** (ou com **Ollama Local**) enviando **`0 bytes`** à nuvem.
   - Ao conectar **Google Gemini (`gemini-2.5-flash`)** ou **OpenAI (`gpt-4o-mini`)** com chave de sessão, a aba **Privacidade & Raio-X** exibe exatamente quantos caracteres do corpus permaneceram locais (`1.873.607 chars`) vs. quantos foram enviados na janela *top-k* (`~0,3%` do livro).
5. **Upload Multi-Formato (`.pdf`, `.md`, `.txt`):**
   - Permite enviar seus próprios PDFs, resumos de aula em Markdown ou textos (ex.: notas do COPOM, capítulos extras), consultá-los em conjunto ou isoladamente e exportar todo o histórico como **Caderno de Estudos (`.md`)**.

---

## 📊 Corpus Demo Pré-Carregado & Benchmark Validado

Livro pré-carregado em [`data/books/`](./data/books/): ***Foundations of Modern Macroeconomics*** — Ben J. Heijdra (Oxford University Press):
- **815 páginas úteis** (`1.873.607` caracteres) • **19 capítulos indexados** (Parte I: *Intermediate Macroeconomics* Caps. 1–10; Parte II: *Towards Advanced Macroeconomics* Caps. 11–17; Epílogo & Apêndice Matemático) • **2.468 chunks** • **9.855 termos BM25** • **LSA 64D** • **Boot com cache: `6,42 s`**.

| Consulta em Português (sobre livro em Inglês) | Modo | Grounding | Guardrail | Latência Retrieval | Capítulo & Página Top-1 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Equivalência Ricardiana e déficit público | `tutor` | **96,9%** (`alta`) | Aprovado | `30,3 ms` | **Cap. 5** (*The government budget deficit*) — pág. 162 |
| Inconsistência temporal e viés inflacionário | `tutor` | **95,7%** (`alta`) | Aprovado | `30,0 ms` | **Cap. 9** (*Macroeconomic policy, credibility*) — pág. 263 |
| Novo Keynesiano (Calvo) vs. Ciclos Reais (RBC) | `compare` | **98,5%** (`alta`) | Aprovado | `32,2 ms` | **Cap. 12** (*New Keynesian economics*) — pág. 418 |
| *Overshooting* cambial de Dornbusch | `tutor` | **98,5%** (`alta`) | Aprovado | `34,0 ms` | **Cap. 10** (*The open economy*) — pág. 338 |
| Crescimento exógeno (Solow/Ramsey) vs. endógeno | `compare` | **98,5%** (`alta`) | Aprovado | `32,5 ms` | **Cap. 13** (*Exogenous economic growth*) — pág. 432 |
| **Teste Anti-Alucinação** (Copa do Mundo 2026) | `verify` | **30,0%** (`baixa`) | **Bloqueado** | `24,4 ms` | *Recusa especular fora do escopo* |

---

## 🗂️ Estrutura do Projeto

```text
app-rag-tutor-macroeconomia/
├── data/
│   └── books/
│       └── Foundations of Modern Macroecon - Ben J. Heijdra.pdf   # 816 páginas (Oxford)
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                        # API REST FastAPI (porta 8003)
│   │   ├── macro_engine.py                # Motor Macro-RAG por capítulos, cache .json.gz, Flashcards e Privacidade
│   │   └── schemas.py                     # Contratos tipados Pydantic v2
│   ├── tests/
│   │   └── test_api.py                    # Suíte automatizada (7/7 testes end-to-end)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TutorChatPanel.tsx         # Modos Tutor/Comparador/Verificador + Inspetor de Capítulo/Página
│   │   │   ├── ChapterMapLibraryPanel.tsx # Upload PDF/MD/TXT + Mapa de Capítulos com keywords TF-IDF
│   │   │   ├── FlashcardsPanel.tsx        # Simulado & Flashcards interativos por capítulo
│   │   │   └── PrivacyXRayPanel.tsx       # Auditoria de Privacidade + Ponte PT↔EN + Recharts + Prompt
│   │   ├── api.ts
│   │   ├── types.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts                     # Proxy /api -> http://127.0.0.1:8003 (porta 5176)
│   └── tailwind.config.js
└── README.md
```

---

## 🔌 Endpoints da API FastAPI

Swagger UI interativo em `http://127.0.0.1:8003/docs`.

| Método | Endpoint | Função |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Status da API, páginas/capítulos/chunks indexados e detecção Ollama |
| `GET` | `/api/providers` | Lista provedores (`extractive`, `gemini`, `ollama`, `openai`) e residência de dados |
| `GET` | `/api/library` | Livros carregados, árvore completa de capítulos com keywords e perguntas sugeridas |
| `POST` | `/api/library/load-demo` | Recarrega a obra de demonstração em `data/books/` |
| `POST` | `/api/documents/upload` | Upload multipart de arquivos `.pdf`, `.md` ou `.txt` |
| `DELETE` | `/api/documents/{doc_id}` | Remove um livro/nota e reconstrói o índice |
| `POST` | `/api/query` | Executa RAG híbrido com resposta, citações, auditoria de privacidade e prompt |
| `POST` | `/api/flashcards` | Gera flashcards/questões fundamentadas por capítulo ou tópico |

---

## 🚀 Como Executar Localmente

### 1. Backend (Python + FastAPI)

```bash
cd app-rag-tutor-macroeconomia/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8003 --reload
```

### 2. Frontend (React + Vite + TypeScript)

```bash
cd app-rag-tutor-macroeconomia/frontend
npm install
npm run dev
```

Acesse **`http://localhost:5176`** no navegador.

---

## ✅ Validação e Testes Automatizados

```bash
cd app-rag-tutor-macroeconomia/backend
python -m unittest discover -s tests -v
```

```bash
cd app-rag-tutor-macroeconomia/frontend
npm run build
```

---

## 👤 Autor

**Silvio da Rosa Paula** — Doutor em Economia Aplicada | Cientista de Dados  
[LinkedIn](https://www.linkedin.com/in/silvio-paula/) • [Portfólio Web](https://silviopaula.github.io/)
