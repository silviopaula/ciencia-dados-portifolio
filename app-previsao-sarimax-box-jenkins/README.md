# 📈 SARIMAX Forecast Studio — Aplicação Full-Stack (Python FastAPI + React)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMAX-150458?style=for-the-badge)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)

</div>

Aplicação web **full-stack** moderna e didática para diagnóstico, modelagem, backtesting e projeção de séries temporais com modelos **ARIMA, SARIMA e SARIMAX** (com variáveis exógenas), implementada com **Backend analítico em Python (FastAPI + Statsmodels)** e **Frontend interativo em React + TypeScript + Tailwind CSS + Recharts**.

> **Evolução arquitetural:** este projeto foi completamente refatorado de um aplicativo legado em R/Shiny para uma arquitetura desacoplada **REST API (Python) + SPA (React)**, incorporando a metodologia **Box-Jenkins passo a passo**, prevenção contra *data leakage*, comparação obrigatória com *baseline* **Seasonal Naive**, teste de ruído branco (**Ljung-Box**) e exportação automatizada em Excel multi-aba.

---

## 🎯 Por que este projeto é didático e funcional?

Em vez de tratar modelos ARIMA como uma "caixa-preta", o **SARIMAX Forecast Studio** organiza o fluxo em **5 abas guiadas** que acompanham exatamente o raciocínio estatístico da metodologia **Box-Jenkins**:

| Aba na Interface | O que o usuário aprende e faz |
| :--- | :--- |
| **1. Guia & Dados** | Explica intuitivamente o significado de `(p, d, q) × (P, D, Q, s)` e variáveis exógenas (`X`). Permite carregar o dataset demonstrativo real de **Consumo de Energia Elétrica** com 1 clique ou fazer upload de qualquer arquivo `.xlsx` / `.csv`. |
| **2. Box-Jenkins (ACF/ADF)** | Testa interativamente a estacionariedade combinando **ADF** (*H₀*: raiz unitária) e **KPSS** (*H₀*: estacionária), sugere automaticamente `d` e `D`, e exibe a série transformada + correlogramas **ACF e PACF** com bandas de confiança de 95% ($\pm 1{,}96 / \sqrt{n}$). |
| **3. Previsão SARIMAX** | Executa **Auto-SARIMAX** (seleção inteligente por AIC/BIC com tabela de candidatos) ou **Modo Manual** (`p, d, q, P, D, Q, s`). Entrega gráfico interativo unificado (Treino + Backtest *Out-of-Sample* + Baseline *Seasonal Naive* + Projeção Futura com Intervalo de Confiança ajustável). |
| **4. Resíduos & Coeficientes** | Valida se o modelo capturou toda a dinâmica temporal via teste **Ljung-Box** (respeitando o *burn-in* difuso do filtro de Kalman — `loglikelihood_burn`), gráfico de resíduos ao longo do tempo e tabela de coeficientes com erro-padrão, estatística *z*, *p-valor* e IC 95%. |
| **5. Crescimento & Excel** | Consolida a projeção em visão executiva anual (média mensal, total anual e variação **YoY %** diferenciando anos *Histórico*, *Misto* e *Projeção*) e gera relatório `.xlsx` multi-aba pronto para diretoria. |

---

## 📊 Resultados Verificados no Dataset Demonstrativo (`data/dados.xlsx`)

O projeto inclui uma base real mensal de **Consumo de Energia Elétrica (MWh)** com variáveis exógenas climáticas, de calendário e tarifárias (`360` linhas de `2005-01` a `2034-12`: `235` meses históricos observados + `125` meses de cenários futuros de exógenas).

### 1. Diagnóstico de Estacionariedade (Escala Log)

| Transformação | Estatística ADF (*p-valor*) | Estatística KPSS (*p-valor*) | Conclusão Box-Jenkins |
| :--- | :---: | :---: | :--- |
| Série em Nível (`d=0, D=0`) | `-1,2897` (`p = 0,6339`) | `2,2755` (`p = 0,0100`) | **Não estacionária** — recomenda `d = 1, D = 1` |
| Diferenciada (`d=1, D=1, s=12`) | `-8,0285` (`p < 0,0001`) | `0,0435` (`p = 0,1000`) | **Estacionária** (consenso ADF + KPSS) |

### 2. Backtest *Out-of-Sample* (Últimos 12 Meses) vs. *Seasonal Naive*

| Modelo | Especificação Selecionada (Auto AIC) | MAPE Teste | Redução vs. Naive | MAE (MWh) | AIC | Ljung-Box(12) *p-valor* | Desvio-Padrão Resíduos (log) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Seasonal Naive ($y_{t-12}$) | `9,42%` | — | `93.475` | — | — | — |
| **Auto-SARIMA (Univariado)** | `SARIMA(0,1,1)×(0,1,1,12)` | `4,38%` | **-53,5%** | `43.396` | `-948,26` | `0,3573` (Ruído Branco) | `0,0209` |
| ⭐ **Auto-SARIMAX (Temp + Dias Úteis)** | `SARIMAX(1,1,0)×(1,1,1,12)` | **`4,26%`** | **-54,8%** | **`42.025`** | **`-1071,06`** | **`0,3494` (Ruído Branco)** | **`0,0152`** |
| **Auto-SARIMAX (+ Tarifa)** | `SARIMAX(1,1,0)×(1,1,1,12)` | `4,37%` | `-53,6%` | `43.155` | `-1069,97` | `0,3663` (Ruído Branco) | `0,0152` |

> **Insight econométrico:** no modelo campeão (`Auto-SARIMAX` com temperatura e dias úteis), ambas as exógenas padronizadas são altamente significativas ($p < 0{,}0001$): +1 desvio-padrão na **Temperatura Média** eleva o consumo elétrico em ~**+3,5%**, e +1 desvio-padrão em **Dias Úteis** eleva o consumo em ~**+0,5%**, reduzindo o erro residual (*in-sample* em log) de `2,09%` para `1,52%` e melhorando o AIC em mais de **122 pontos**. Já a inclusão da tarifa no curto prazo exibe sinal negativo esperado (`-0,0057`), mas sem significância estatística em primeira diferença ($p = 0{,}445$), ilustrando na prática o princípio da parcimônia.

---

## 🏗️ Arquitetura e Estrutura do Projeto

```text
app-previsao-sarimax-box-jenkins/
├── data/
│   └── dados.xlsx                    # Dataset demonstrativo (Consumo de Energia + Exógenas)
├── backend/                          # API REST Analítica em Python (FastAPI)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # Rotas REST, CORS, upload e download Excel
│   │   ├── schemas.py                # Contratos tipados com Pydantic v2
│   │   └── forecasting.py            # Motor Box-Jenkins, Auto-SARIMAX, backtest e Excel
│   ├── tests/
│   │   └── test_api.py               # Testes automatizados ponta a ponta (unittest + TestClient)
│   └── requirements.txt              # Dependências Python enxutas
├── frontend/                         # Interface Web Moderna em React + TypeScript
│   ├── src/
│   │   ├── components/
│   │   │   ├── DiagnosticsPanel.tsx  # Cards ADF/KPSS, série transformada e ACF/PACF
│   │   │   ├── ForecastChart.tsx     # Gráfico interativo Treino + Backtest + Projeção + IC
│   │   │   ├── ResidualsPanel.tsx    # Ljung-Box, gráfico de resíduos e tabela de coeficientes
│   │   │   └── GrowthTable.tsx       # Consolidado anual (YoY %) e download do relatório .xlsx
│   │   ├── App.tsx                   # Orquestração do layout, sidebar e 5 abas guiadas
│   │   ├── api.ts                    # Cliente HTTP tipado para a API FastAPI
│   │   ├── types.ts                  # Interfaces TypeScript espelhando Pydantic v2
│   │   ├── index.css                 # Estilização Tailwind CSS (tema escuro analítico)
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts                # Proxy automático /api -> http://127.0.0.1:8000
│   ├── tailwind.config.js
│   └── tsconfig.json
└── README.md
```

### Principais Endpoints da API (`FastAPI`)

| Método | Rota | Descrição |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Verifica disponibilidade da API e versão do serviço |
| `POST` | `/api/dataset/demo` | Carrega `data/dados.xlsx` e retorna metadados, tipos e prévia |
| `POST` | `/api/dataset/upload` | Recebe upload `.xlsx`, `.xls` ou `.csv` com detecção automática de datas e variáveis numéricas |
| `POST` | `/api/diagnostics` | Executa testes ADF/KPSS, recomenda `d`/`D` e retorna série transformada + ACF/PACF |
| `POST` | `/api/forecast` | Ajusta Auto ou Manual SARIMA/SARIMAX, executa backtest vs. Seasonal Naive, Ljung-Box e projeção futura |
| `GET` | `/api/export/{dataset_id}` | Exporta relatório `.xlsx` multi-aba (`Resumo_Executivo`, `Serie_e_Previsao`, `Coeficientes`, `Crescimento_Anual`) |

> A documentação interativa Swagger/OpenAPI fica disponível automaticamente em `http://127.0.0.1:8000/docs`.

---

## ⚙️ Boas Práticas Estatísticas Implementadas no Motor Python

- **Zero *Data Leakage*:** padronização *Z-score* das variáveis exógenas ajustada exclusivamente no conjunto de treino e reaplicada no teste/futuro.
- **Reajuste Honesto:** métricas de erro (`MAPE`, `RMSE`, `MAE`, `R²`) avaliadas rigorosamente fora da amostra (*out-of-sample*), seguidas de reajuste em 100% do histórico observado para gerar a projeção futura.
- **Correção de Viés de Retransformação (Duan Smearing):** quando a escala logarítmica está ativa, aplica fator de *smearing* de Duan sobre os resíduos pós-*burn-in* ao voltar para a escala original.
- **Respeito ao *Burn-In* Difuso do Filtro de Kalman:** exclui corretamente os primeiros `loglikelihood_burn` passos difusos ao calcular desvio-padrão residual, teste de Ljung-Box e valores ajustados no gráfico.
- **Exógenas Futuras Inteligentes:** utiliza os cenários futuros presentes na planilha quando fornecidos; caso o horizonte solicitado exceda as linhas disponíveis, completa com segurança repetindo o último ciclo sazonal.

---

## 🚀 Como Executar Localmente

### Pré-requisitos
- **Python 3.10+**
- **Node.js 18+** (com `npm`)

### 1. Subir o Backend (Python / FastAPI)

```bash
cd app-previsao-sarimax-box-jenkins/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

A API responderá em `http://127.0.0.1:8000` (Swagger UI em `http://127.0.0.1:8000/docs`).

### 2. Subir o Frontend (React / Vite)

Em um segundo terminal:

```bash
cd app-previsao-sarimax-box-jenkins/frontend
npm install
npm run dev
```

Abra no navegador o endereço exibido pelo Vite (padrão `http://localhost:5173`). O Vite encaminha automaticamente todas as requisições `/api/*` para `http://127.0.0.1:8000` (para apontar para outro host em produção, defina `VITE_API_URL`).

### 3. Executar Testes Automatizados e Build de Produção

```bash
# Testes de integração da API (FastAPI TestClient + unittest)
cd app-previsao-sarimax-box-jenkins/backend
python -m unittest discover -s tests -v

# Verificação de tipos TypeScript + build de produção do React
cd app-previsao-sarimax-box-jenkins/frontend
npm run build
```

---

## 👨‍💻 Autor

**Silvio da Rosa Paula** — Doutor em Economia Aplicada | Cientista de Dados  
[LinkedIn](https://www.linkedin.com/in/silvio-paula/) • [Portfólio Web](https://silviopaula.github.io/)
