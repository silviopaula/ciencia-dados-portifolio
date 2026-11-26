# 📊 Observatório Macro Brasil Studio

Aplicação web *full-stack* de **Macroeconomia Aplicada e Econometria do Brasil**, estruturada em arquitetura moderna e desacoplada (**Python + FastAPI + Statsmodels + DuckDB/Parquet** no backend e **React + TypeScript + Tailwind CSS + Recharts** no frontend).

O projeto integra um **Painel Executivo de Conjuntura Brasil** conectado às APIs oficiais do **Banco Central do Brasil (SGS e Olinda Focus)** com uma biblioteca interativa de **60 estudos macroeconométricos 100% autorais** (`M01` a `M60`), perfeitamente distribuídos em **6 pilares temáticos (10 estudos por pilar)** — incluindo **6 laboratórios econométricos dedicados** com ajuste de parâmetros em tempo real.

---

## 🎯 Principais Funcionalidades

1. **Painel de Conjuntura Brasil (BCB SGS + Olinda Focus)**
   - Monitoramento em tempo real de **6 termômetros macroeconômicos**: Taxa Selic (`SGS 4189`), IPCA Acumulado 12m (`SGS 13522`) vs. Meta CMN (`SGS 13521`), Dívida Bruta DBGG/PIB (`SGS 13762`), Câmbio Comercial (`SGS 3698`), Atividade IBC-Br (`SGS 24364`) e Desocupação PNAD (`SGS 24369`).
   - Botão **Sincronizar BCB ao Vivo** (`POST /api/conjuntura/sync`), que atualiza as séries diretamente do Banco Central e persiste o snapshot em **DuckDB + Parquet** para inicialização instantânea offline.
2. **Estúdio Econométrico Interativo (60 Estudos Autorais `M01`–`M60`)**
   - Cada estudo oferece fundamentação **Intuição Econômica**, **Metodologia Econométrica**, **Guia de Interpretação** e **Referências Acadêmicas** redigidos do zero em português.
   - **6 Modelos Dedicados Interativos** (`M01`, `M02`, `M06`, `M12`, `M13`, `M14`) com controles paramétricos (subamostra, defasagens VAR/Granger, horizonte ARIMA/SARIMA, $\lambda$ do Filtro HP, janela móvel Rolling AR(1) e defasagens Newey-West HAC).
   - **54 Estudos Temáticos** executados por motor unificado (`VAR + IRF + Granger`, `Engle-Granger + ECM`, `Filtro Hodrick-Prescott`, `Volatilidade EWMA`, `ARIMA(1,1,1)` e `MQO Newey-West HAC`), acompanhados de testes de raiz unitária (ADF), estatísticas descritivas, matriz de correlação e opção de atualização via API BCB.
   - Exportação de qualquer estudo tratado/projetado em **CSV** (`/api/studies/{code}/export-csv`).
3. **Armazenamento Analítico Colunar (DuckDB + Parquet)**
   - Banco local `data/observatorio_macro.duckdb` (`10,51 MB`) consolidando **37 tabelas históricas** e **76 séries oficiais do BCB (SGS + Olinda Focus)** (**43.109 registros analíticos**, atualizados até nov/2026), espelhado em `data/bcb_series.parquet`.

---

## 📈 Termômetros de Conjuntura & Resultados Verificados

Snapshot verificado diretamente pelo motor Python sobre o banco `data/observatorio_macro.duckdb`:

| Indicador Oficial (BCB) | Último Valor | Variação (12m) | Referência |
| :--- | :---: | :---: | :---: |
| **Taxa Selic Mensal (`SGS 4189`)** | **13,65% a.a.** | `-1,25 p.p.` | `2026-11` |
| **IPCA Acumulado 12m (`SGS 13522`)** | **4,63%** *(Meta: 3,00%)* | `-0,05 p.p.` | `2026-10` |
| **Dívida Bruta DBGG (`SGS 13762`)** | **82,95% PIB** | `+5,11 p.p.` | `2026-09` |
| **Câmbio Comercial (`SGS 3698`)** | **R\$ 5,13 / US\$** | `-0,26` | `2026-10` |
| **Atividade IBC-Br (`SGS 24364`)** | **109,8** | `+0,9` | `2026-09` |
| **Desocupação PNAD (`SGS 24369`)** | **5,2%** | `-0,4 p.p.` | `2026-09` |

### Destaques dos 6 Laboratórios Econométricos Dedicados

- **`M01` — Dominância Fiscal (Selic vs. CDS 5 Anos)**: `3.492` observações diárias pareadas; na subamostra de estresse fiscal a partir de `2014-09-05` (`N = 349`), $\Delta\text{Selic}$ Granger-causa $\Delta\text{CDS 5Y}$ a 5% nas defasagens `[9, 10, 11, 12]` sem evidência de causalidade reversa.
- **`M02` — Sustentabilidade e Projeção da DBGG/PIB**: partindo de `82,95%` do PIB (`09/2026`), o modelo `ARIMA(1,1,1)` com drift (`N = 238`, `AIC = 473,8`) projeta cenário inercial de `84,65%` do PIB em 12 meses (`+1,70 p.p.`), acompanhado do painel histórico *Primário Efetivo vs. Requerido*.
- **`M06` — Função de Reação do Banco Central (Regra de Taylor HAC)**: `R² = 99,7%` (`N = 128` meses), suavização $\rho = 0{,}973$ e resposta de longo prazo $\beta_1 / (1 - \rho) = 4{,}49 > 1{,}0$ — confirmando o **Princípio de Taylor**.
- **`M12` — Persistência e Inércia Inflacionária (Rolling AR(1))**: com janela móvel de 48 meses, a persistência atual do IPCA situa-se em $\gamma(t) = 0{,}198$ frente à média histórica de `0,450` (pico de `0,604`).
- **`M13` — Modelagem Sazonal e Previsão do IPCA (SARIMA)**: `SARIMA(1,0,1)(1,0,1)[12]` sobre o IPCA mensal (`0,14%` em `10/2026`), com `AIC = 103,7`, `RMSE = 0,372 p.p.` e projeção acumulada de `3,62%` em 12 meses.
- **`M14` — Curva de Phillips Brasileira (MQO-HAC)**: `R² = 41,8%` (`R² ajustado = 40,4%`, `F = 7,2`, `N = 168`), resíduos estacionários (`ADF p = 0,0000`) e significância estatística das expectativas de inflação a 5%.

---

## 📚 Catálogo Completo dos 60 Estudos Autorais (10 por Pilar)

Os estudos marcados com ⭐ possuem laboratório paramétrico dedicado:

| Código | Estudo Macroeconométrico | Pilar Temático | Metodologia Principal |
| :---: | :--- | :--- | :--- |
| `M01` | ⭐ **Dominância Fiscal no Brasil: Taxa Selic e Risco Soberano (CDS 5 Anos)** | Política Fiscal & Dívida | VAR, Teste ADF, Correlação Cruzada (CCF) & Causalidade de Granger |
| `M02` | ⭐ **Sustentabilidade e Projeção da Dívida Bruta do Governo Geral (DBGG/PIB)** | Política Fiscal & Dívida | Painel Fiscal (r − g), Teste ADF & Projeção ARIMA(1,1,1) com Bandas 80%/95% |
| `M03` | **Dívida Pública e Incerteza Econômica: Equilíbrio de Longo Prazo** | Política Fiscal & Dívida | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M04` | **Econometria do Ajuste Fiscal: Receitas Líquidas vs. Despesas Primárias** | Política Fiscal & Dívida | Cointegração de Engle-Granger & Dinâmica de Ajuste Fiscal (ECM) |
| `M05` | **Âncora Fiscal, Regras de Gasto e Taxa de Juros de Equilíbrio** | Política Fiscal & Dívida | Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP) |
| `M06` | ⭐ **Função de Reação do Banco Central: Regra de Taylor e Hiato do Produto** | Política Monetária & Juros | Regra de Taylor Dinâmica (MQO Newey-West HAC) + Hiato via Filtro HP |
| `M07` | **Estimação da Taxa de Juros Neutra na Economia Brasileira** | Política Monetária & Juros | Filtro Hodrick-Prescott (HP) & Comparação Juro Real vs. Neutro |
| `M08` | **Juro Real Ex-Ante vs. Ex-Post e Postura da Política Monetária** | Política Monetária & Juros | Decomposição Tendência-Ciclo (HP) sobre Séries de Juro Real |
| `M09` | **Crédito Direcionado e Potência dos Canais de Transmissão Monetária** | Política Monetária & Juros | Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos |
| `M10` | **Intervenções Cambiais via Swaps do Banco Central e Volatilidade** | Política Monetária & Juros | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M11` | **Transmissão Monetária: Selic, Juros à Pessoa Física e Spread Bancário** | Política Monetária & Juros | Regressão Econométrica (MQO Newey-West HAC) & Matriz de Correlação |
| `M12` | ⭐ **Persistência e Inércia Inflacionária no Brasil (Rolling AR(1))** | Inflação & Expectativas | Regressão em Janela Móvel (Rolling AR(1)) + IPCA 12m vs. Meta CMN |
| `M13` | ⭐ **Modelagem Sazonal e Previsão do IPCA (SARIMA Box-Jenkins)** | Inflação & Expectativas | SARIMA(p,d,q)(P,D,Q)[12], Teste ADF & Correlogramas ACF/PACF |
| `M14` | ⭐ **Curva de Phillips Brasileira: Inflação, Expectativas e Ciclo Econômico** | Inflação & Expectativas | Curva de Phillips Híbrida (MQO Newey-West HAC) & Diagnóstico de Resíduos |
| `M15` | **Expectativas de Inflação: Especialistas (Focus) vs. Consumidores** | Inflação & Expectativas | Regressão Econométrica (MQO HAC) & Análise de Acurácia de Expectativas |
| `M16` | **Repasse Cambial e de Commodities (IC-Br) para a Inflação** | Inflação & Expectativas | VAR, Função Impulso-Resposta (IRF) & Causalidade de Granger |
| `M17` | **Dinâmica do IGP-M e Núcleos de Inflação (Tendência Subjacente)** | Inflação & Expectativas | Modelagem ARIMA, Correlogramas ACF/PACF & Projeção 12 Meses |
| `M18` | **Processo Gerador do PIB Brasileiro: Raiz Unitária e Persistência de Choques** | Atividade Econômica & PIB | Teste ADF, Estrutura ACF/PACF & Modelagem ARIMA do Produto |
| `M19` | **Hiato do Produto e Produto Potencial via Filtro Hodrick-Prescott** | Atividade Econômica & PIB | Filtro Hodrick-Prescott (HP) — Separação de Tendência Potencial e Hiato |
| `M20` | **Incerteza Econômica, Nível de Atividade (PIB) e Capacidade Instalada** | Atividade Econômica & PIB | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M21` | **Contas Nacionais, IBC-Br e Dinâmica Setorial da Atividade** | Atividade Econômica & PIB | Decomposição Tendência-Ciclo (Filtro HP) & Correlação Setorial |
| `M22` | **Crédito de Fomento (BNDES), Energia e Taxa de Investimento** | Atividade Econômica & PIB | Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos |
| `M23` | **Retropolação da Taxa de Desemprego: PNAD Contínua vs. Antiga PME** | Mercado de Trabalho | Regressão de Encadeamento (MQO HAC) & Diagnóstico de Pareamento |
| `M24` | **Emprego Formal (CAGED) vs. PNAD Contínua: Dinâmica e Cointegração** | Mercado de Trabalho | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M25` | **Projeção da Taxa de Desemprego com Equilíbrio de Longo Prazo (ECM)** | Mercado de Trabalho | Cointegração (Engle-Granger) & Correção de Erros (ECM/VECM) |
| `M26` | **Emprego Formal Privado (CLT) e Sensibilidade ao Ciclo Econômico** | Mercado de Trabalho | Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP) |
| `M27` | **Determinantes Macroeconômicos da Taxa de Câmbio (R\$/US\$)** | Câmbio & Finanças | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M28` | **Volatilidade Condicional do Câmbio e Risco Global (EWMA)** | Câmbio & Finanças | Econometria Financeira: Retornos Log, Volatilidade EWMA ($\lambda=0{,}94$) & Curtose |
| `M29` | **Mercado de Capitais (IFIX & Ibovespa): Retornos, Risco e Juros** | Câmbio & Finanças | Econometria Financeira: Retornos, Volatilidade Condicional EWMA & Correlações |
| `M30` | **Poupança Doméstica e Investimento: Causalidade e Restrição Externa** | Câmbio & Finanças | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M31` | **Dívida Bruta (DBGG) vs. Dívida Líquida (DLSP): O Papel das Reservas Internacionais** | Política Fiscal & Dívida | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M32` | **Resultado Primário, Juros e Acumulação da Dívida Mobiliária Federal** | Política Fiscal & Dívida | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M33` | **NFSP Nominal, Custo da Dívida e Dinâmica do Produto Mensal** | Política Fiscal & Dívida | Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos |
| `M34` | **Ciclo Fiscal Brasileiro: Componente Estrutural vs. Cíclico do Resultado Primário** | Política Fiscal & Dívida | Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP) |
| `M35` | **Transbordamento Fiscal: Impacto de Choques na DBGG sobre Câmbio e Inflação** | Política Fiscal & Dívida | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M36` | **Meios de Pagamento (M1 & M2), Base Monetária e PIB no Longo Prazo** | Política Monetária & Juros | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M37` | **Spread Bancário, Taxa de Juros de Mercado e Inadimplência no Brasil** | Política Monetária & Juros | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M38` | **Canal do Crédito: Aprofundamento Financeiro (Crédito/PIB) e Política Monetária** | Política Monetária & Juros | Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos |
| `M39` | **Assimetria da Transmissão Monetária: Crédito Pessoa Física vs. Pessoa Jurídica** | Política Monetária & Juros | Regressão Econométrica (MQO Newey-West HAC) & Matriz de Correlação |
| `M40` | **Ancoragem das Expectativas: Boletim Focus 12m vs. Meta CMN e IPCA Realizado** | Inflação & Expectativas | Regressão Econométrica (MQO Newey-West HAC) sobre Dados Focus Olinda BCB |
| `M41` | **Preços Administrados vs. Preços Livres: Propagação de Choques de Oferta** | Inflação & Expectativas | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M42` | **Inflação de Serviços, Alimentação e Núcleos Subjacentes do Banco Central** | Inflação & Expectativas | Regressão Econométrica (MQO Newey-West HAC) & Decomposição de Grupos |
| `M43` | **Transmissão de Atacado ao Varejo: IPA-DI, IPCA-15 e INPC** | Inflação & Expectativas | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M44` | **O IBC-Br como Termômetro Mensal da Indústria (PIM-PF) e do Varejo (PMC)** | Atividade Econômica & PIB | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M45` | **Ciclo Industrial Brasileiro: Bens de Capital, Intermediários e de Consumo** | Atividade Econômica & PIB | Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP) |
| `M46` | **Utilização da Capacidade Instalada (NUCI), Energia Industrial e Produção Física** | Atividade Econômica & PIB | Regressão Econométrica (MQO Newey-West HAC) para Nowcasting Industrial |
| `M47` | **Varejo Restrito vs. Varejo Ampliado, Crédito e Confiança do Consumidor (ICC)** | Atividade Econômica & PIB | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M48` | **Previsão de Curto Prazo do Consumo Total de Energia Elétrica (Termômetro do PIB)** | Atividade Econômica & PIB | Modelagem ARIMA(1,1,1), Correlogramas ACF/PACF & Projeção 12 Meses |
| `M49` | **Lei de Okun no Brasil: Elasticidade do Desemprego ao Ciclo de Atividade** | Mercado de Trabalho | Lei de Okun Empírica (MQO Newey-West HAC) & Matriz de Correlação |
| `M50` | **Rendimento Médio Real (PNAD) e Pressões na Inflação de Serviços** | Mercado de Trabalho | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M51` | **Demografia da Força de Trabalho: População Ocupada vs. Inatividade (PNAD)** | Mercado de Trabalho | Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP) |
| `M52` | **Novo CAGED vs. População Desocupada: Velocidade de Absorção Formal** | Mercado de Trabalho | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M53` | **Salário Mínimo Nominal, Cesta Básica (DIEESE) e Poder de Compra (INPC)** | Mercado de Trabalho | Regressão Econométrica (MQO Newey-West HAC) & Análise de Poder de Compra |
| `M54` | **Projeção da Taxa de Desocupação PNAD Contínua via ARIMA** | Mercado de Trabalho | Modelagem ARIMA(1,1,1), Correlogramas ACF/PACF & Projeção 12 Meses |
| `M55` | **Balança Comercial Brasileira: Exportações, Importações e Câmbio Real Efetivo** | Câmbio & Finanças | Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM) |
| `M56` | **Ciclo de Commodities (IC-Br Agro, Metal e Energia) e Exportações Brasileiras** | Câmbio & Finanças | Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger |
| `M57` | **Contas Externas: Transações Correntes e Financiamento por Investimento Direto (IDP)** | Câmbio & Finanças | Regressão Econométrica (MQO Newey-West HAC) sobre Balanço de Pagamentos |
| `M58` | **Desalinhamento Cambial: Tendência de Equilíbrio da Taxa de Câmbio Real Efetiva** | Câmbio & Finanças | Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP) |
| `M59` | **Volatilidade Condicional do Índice de Commodities Brasil (IC-Br) e Choques Globais** | Câmbio & Finanças | Econometria Financeira: Retornos Log, Volatilidade EWMA ($\lambda=0{,}94$) & Curtose |
| `M60` | **Colchão de Liquidez Externa: Reservas Internacionais e Projeção ARIMA** | Câmbio & Finanças | Modelagem ARIMA(1,1,1), Correlogramas ACF/PACF & Projeção 12 Meses |

---

## 🏗️ Arquitetura & Estrutura de Pastas

```text
app-estudos-macroeconomia-bcb/
├── backend/
│   ├── app/
│   │   ├── main.py           # API REST FastAPI (porta dedicada 8004)
│   │   ├── catalog.py        # Catálogo 100% autoral dos estudos M01–M30 (unificado com M31–M60)
│   │   ├── catalog_extra.py  # Extensão autoral M31–M60 alimentada por séries BCB (SGS + Olinda)
│   │   ├── data_store.py     # Persistência DuckDB + Parquet & cliente BCB SGS/Olinda
│   │   └── macro_models.py   # Motor econométrico Statsmodels (6 dedicados + 6 temáticos)
│   ├── tests/
│   │   └── test_api.py       # Suíte automatizada cobrindo conjuntura e todos os 60 estudos
│   └── requirements.txt
├── frontend/                 # SPA React 18 + Vite 6 + TypeScript + Tailwind + Recharts (:5177)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── types.ts
│   │   └── components/
│   │       ├── ConjunturaPanel.tsx
│   │       ├── CatalogPanel.tsx
│   │       ├── StudyStudioPanel.tsx
│   │       └── ArchitecturePanel.tsx
│   ├── package.json
│   └── vite.config.ts        # Proxy /api -> http://127.0.0.1:8004
├── data/
│   ├── observatorio_macro.duckdb  # 37 tabelas históricas + 76 séries BCB (10,51 MB)
│   └── bcb_series.parquet         # Snapshot colunar das séries oficiais do BCB
└── README.md
```

### Endpoints REST (`http://127.0.0.1:8004`)

| Método | Rota | Descrição |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Identidade da API (`Observatório Macro Brasil API`), tabelas DuckDB e séries BCB |
| `GET` | `/api/conjuntura` | 6 KPIs executivos + 3 séries históricas do Painel de Conjuntura Brasil |
| `POST` | `/api/conjuntura/sync` | Sincroniza séries oficiais nas APIs SGS e Olinda Focus do Banco Central |
| `GET` | `/api/studies` | Catálogo completo dos 60 estudos e contagem por pilar temático (10 por pilar) |
| `GET` | `/api/studies/{code}` | Metadados didáticos, intuição e referências do estudo (`M01`–`M60`) |
| `POST` | `/api/studies/{code}/run` | Executa o modelo econométrico Python com parâmetros interativos |
| `GET` | `/api/studies/{code}/export-csv` | Exporta os resultados / séries tratadas do estudo em formato CSV |

---

## 🚀 Como Executar Localmente

### 1. Backend Python (FastAPI — porta `8004`)

```bash
cd app-estudos-macroeconomia-bcb/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8004 --reload
```

- Documentação interativa Swagger UI: `http://127.0.0.1:8004/docs`

### 2. Frontend React + Vite (porta `5177`)

```bash
cd app-estudos-macroeconomia-bcb/frontend
npm install
npm run dev
```

- Acesse a interface em: `http://localhost:5177`

### 3. Testes Automatizados e Build

```bash
# Validação do backend (executa Conjuntura + todos os 60 estudos M01–M60 + CSV)
cd app-estudos-macroeconomia-bcb/backend
python -m unittest discover -s tests -v

# Checagem TypeScript + build de produção do frontend
cd app-estudos-macroeconomia-bcb/frontend
npm run build
```

---

## 🛠️ Stack Tecnológico

- **Backend & Econometria**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Statsmodels (`SARIMAX`, `VAR`, `hpfilter`, `OLS HAC`, `adfuller`, `coint`, `grangercausalitytests`), Pandas, NumPy, SciPy, HTTPX.
- **Armazenamento Analítico**: DuckDB, Apache Parquet (`pyarrow`), conectores oficiais **BCB SGS** e **BCB Olinda Focus**.
- **Frontend**: React 18, TypeScript 5, Vite 6, Tailwind CSS 3, Recharts, Lucide React.
