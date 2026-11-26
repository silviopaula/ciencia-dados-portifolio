# 📈 CausalImpact Studio — Inferência Causal Contrafactual em Séries Temporais (BSTS)

Aplicação web **full-stack** (*Python + FastAPI + Statsmodels* no backend e *React + TypeScript + Tailwind CSS + Recharts* no frontend) para estimação rigorosa de **impacto causal em séries temporais** sem experimento A/B randomizado, baseada na metodologia **Bayesian Structural Time Series (BSTS / CausalImpact)** de Brodersen et al. (2015).

Refatorado a partir de um aplicativo legado em R/Shiny para uma arquitetura Python/React limpa, rápida, sem *data leakage*, com **auditoria automática das hipóteses causais**, **Teste Placebo no Tempo (*In-Time Falsification*)** e exportação executiva em Excel multi-abas.

---

## 🎯 Principais Evoluções e Diferenciais Didáticos

1. **Motor Python Nativo em Espaço de Estados (Zero Deprecation Warnings):**
   - Construído diretamente sobre `statsmodels.tsa.statespace.structural.UnobservedComponents` + Filtro/Suavizador de Kalman + **Simulação Preditiva Posterior Conjunta (Monte Carlo)**, propagando tanto a autocorrelação do estado latente quanto a incerteza dos coeficientes $\hat{\beta}$ para intervalos de credibilidade acumulados rigorosos.
   - Padronização calculada **estritamente no período pré-intervenção** (zero *data leakage*).
   - Suporta 3 especificações estruturais: **Local Level** (padrão Google CausalImpact), **Local Linear Trend** e **Local Level + Sazonalidade Estocástica nativa**, além de pré-desazonalização opcional via **STL** e modo **univariado** (com alerta didático).
2. **Auditoria Econométrica Automática das Hipóteses Causais:**
   - **Qualidade do Ajuste Pré-Intervenção:** $R^2_{\text{pré}}$, RMSE, MAPE, AIC/BIC e teste de autocorrelação residual **Ljung-Box**.
   - **Auditoria *No-Spillover* nos Controles:** verifica se alguma covariável explicativa $X_j$ sofreu quebra estrutural anômala no pós-intervenção (alerta de contaminação do grupo de controle).
   - **Teste Placebo no Tempo (*In-Time Falsification*):** divide automaticamente o período pré-intervenção simulando um evento fictício (efeito real = 0) e exibe gráfico + veredito de aprovação.
   - **Decomposição Estrutural:** separa visualmente a contribuição do **Nível**, das **Covariáveis ($X\hat{\beta}$)** e da **Sazonalidade** na formação do contrafactual.
3. **Seis Módulos Interativos na Interface React:**
   - **Dados & Upload (Tela Inicial):** área *drag-and-drop* em destaque para enviar `.csv` (`, ` ou `;`, decimal BR) ou `.xlsx`, botão para **Baixar Planilha Modelo (`.xlsx`)**, prévia interativa para conferência de colunas/datas antes da estimação e acesso em 1 clique aos 3 cenários demo.
   - **Impacto Causal (3 Painéis Clássicos):** KPIs executivos, narrativa automática em português e os 3 gráficos de Brodersen et al. (*Observado vs. Contrafactual + IC*, *Efeito Pontual + IC* e *Efeito Acumulado + IC*).
   - **Auditoria & Placebo:** métricas pré-intervenção, gráfico do teste placebo, ranking de pesos padronizados das covariáveis, auditoria *No-Spillover* e decomposição estrutural.
   - **Tabela & Exportação:** agregação dinâmica (*Original, Mensal, Trimestral, Semestral, Anual*) e download em 1 clique de **Excel multi-abas (`.xlsx` — 6 abas)**, **CSV** e **Relatório Markdown (`.md`)**.
   - **Exploração & Descritivas:** gráfico comparativo com opção **Base 100**, estatísticas descritivas (Pré vs. Pós vs. Total) e prévia da base (aceita upload de `.csv` com padrão BR `;`/`,` ou `.xlsx`).
   - **Guia Metodológico:** explicação visual do contrafactual BSTS, hipóteses de identificação e referências acadêmicas.

---

## 📊 Cenários Demo Pré-Carregados & Resultados Validados

Três conjuntos de dados em [`data/`](./data/) permitem explorar em segundos um impacto positivo real, a replicação exata do benchmark clássico e um cenário placebo de efeito nulo:

| Cenário Demo (`data/`) | Obs. (Pré / Pós) | Efeito Médio | Impacto Acumulado (IC 95%) | $p$-valor / Prob. Causal | $R^2_{\text{pré}}$ / MAPE | Teste Placebo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Campanha de Marketing Digital** (`campanha_marketing_semanal.csv`) | 80 (56 / 24) | **`+24,78`** (`+10,22%`) | **`+594,68`** `[+479,75; +701,92]` | `0,0000` (**100,0%**) | `0,9289` / `0,83%` | **Aprovado** (`+0,35%`, `p=0,425`) |
| **Clássico Brodersen / Google** (`exemplo_classico_brodersen.csv`, choque real `+10,0`) | 52 (40 / 12) | **`+9,95`** (`+7,69%`) | **`+119,37`** `[+106,61; +131,80]` | `0,0000` (**100,0%**) | `0,7890` / `0,63%` | **Aprovado** (`+0,01%`, `p=0,488`) |
| **Cenário Placebo / Efeito Nulo** (`placebo_efeito_nulo_mensal.csv`, efeito real `0,0`) | 60 (45 / 15) | `-1,03` (`-0,54%`) | `-15,51` `[-36,36; +4,79]` *(cruza 0)* | `0,0671` (**Não sig.**) | `0,9180` / `0,75%` | **Aprovado** (`+0,18%`, `p=0,320`) |

---

## 🗂️ Estrutura do Projeto

```text
app-causal-impact-bsts-kalman/
├── data/
│   ├── campanha_marketing_semanal.csv     # Demo 1: E-commerce (80 semanas, impacto positivo)
│   ├── exemplo_classico_brodersen.csv     # Demo 2: Benchmark clássico (+10 nas 12 últimas semanas)
│   └── placebo_efeito_nulo_mensal.csv     # Demo 3: Intervenção sem efeito (60 meses)
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                        # API REST FastAPI (porta 8002)
│   │   ├── causal_engine.py               # Motor BSTS / Kalman / Monte Carlo / Placebo / Excel
│   │   └── schemas.py                     # Contratos Pydantic v2
│   ├── tests/
│   │   └── test_api.py                    # Suíte automatizada (7/7 testes end-to-end)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CausalChartsPanel.tsx      # KPIs + 3 painéis clássicos Brodersen et al.
│   │   │   ├── AuditPlaceboPanel.tsx      # Ajuste pré, Teste Placebo, Coeficientes & No-Spillover
│   │   │   ├── ImpactTableExportPanel.tsx # Tabela agregada + downloads Excel/CSV/Markdown
│   │   │   ├── ExploratoryPanel.tsx       # Gráfico multi-séries Base 100 + Descritivas
│   │   │   └── MethodologyGuide.tsx       # Guia didático BSTS e referências
│   │   ├── api.ts
│   │   ├── types.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts                     # Proxy /api -> http://127.0.0.1:8002 (porta 5175)
│   └── tailwind.config.js
└── README.md
```

---

## 🔌 Endpoints da API FastAPI

Documentação interativa (Swagger UI) em `http://127.0.0.1:8002/docs`.

| Método | Endpoint | Função |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Status da API, dataset carregado e quantidade de demos |
| `GET` | `/api/demos` | Catálogo dos 3 cenários de demonstração |
| `POST` | `/api/dataset/demo/{demo_id}` | Carrega um cenário demo em memória |
| `POST` | `/api/dataset/upload` | Upload de arquivo `.csv` (`, ` ou `;`, decimal BR) ou `.xlsx` |
| `GET` | `/api/dataset/current` | Metadados, colunas e prévia da base ativa |
| `POST` | `/api/analyze` | Estima o modelo BSTS, executa Placebo, *No-Spillover* e tabelas agregadas |
| `GET` | `/api/export/excel` | Exporta relatório completo em Excel (`.xlsx`, 6 abas formatadas) |
| `GET` | `/api/export/csv?freq=...` | Exporta tabela de impacto agregada (`original`, `month`, `quarter`, `semester`, `year`) |
| `GET` | `/api/export/report` | Exporta relatório técnico executivo em Markdown (`.md`) |
| `GET` | `/api/export/template` | Baixa planilha modelo em Excel (`.xlsx`) com dados de exemplo e instruções |

---

## 🚀 Como Executar Localmente

### 1. Backend (Python + FastAPI)

```bash
cd app-causal-impact-bsts-kalman/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload
```

### 2. Frontend (React + Vite + TypeScript)

Em outro terminal:

```bash
cd app-causal-impact-bsts-kalman/frontend
npm install
npm run dev
```

Acesse **`http://localhost:5175`** no navegador.

---

## ✅ Validação e Testes Automatizados

```bash
cd app-causal-impact-bsts-kalman/backend
python -m unittest discover -s tests -v
```

```bash
cd app-causal-impact-bsts-kalman/frontend
npm run build
```

---

## 📚 Referências

- Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). *Inferring causal impact using Bayesian structural time-series models*. **The Annals of Applied Statistics**, 9(1), 247–274.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.

---

## 👤 Autor

**Silvio da Rosa Paula** — Doutor em Economia Aplicada | Cientista de Dados  
[LinkedIn](https://www.linkedin.com/in/silvio-paula/) • [Portfólio Web](https://silviopaula.github.io/)
