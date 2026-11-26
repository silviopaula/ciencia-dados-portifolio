# 📊 TimeSeries Diagnostics Studio — Laboratório Econométrico de Testes Estatísticos

<div align="center">

![FastAPI](https://img.shields.io/badge/Backend-FastAPI_Python-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Frontend-React_18_+_TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Statsmodels](https://img.shields.io/badge/Econometria-Statsmodels_+_SciPy-1D4ED8?style=for-the-badge)
![Tests](https://img.shields.io/badge/Unittest-5%2F5_Passed-10B981?style=for-the-badge)

</div>

Aplicação web *full-stack* (**FastAPI + React/TypeScript + Tailwind CSS + Recharts**) dedicada ao diagnóstico econométrico rigoroso de séries temporais — evoluindo a versão legada em R Shiny para uma plataforma modular, interativa e 100% testada.

Reúne em um único estúdio **10 testes estatísticos clássicos e modernos**, **Veredito Cruzado 4-Quadrantes (ADF × Phillips-Perron × KPSS × Zivot-Andrews)**, **Laboratório Comparativo de 7 Transformações**, **Decomposição STL Robusta ($F_T, F_S$)**, **Modelagem SARIMA Auto/Manual com Backtest *Out-of-Sample* vs. *Seasonal Naive*** e **Exportação em Excel Multi-Abas (5 abas formatadas)**.

---

## ✨ Funcionalidades Principais

### 1. Bateria Completa de 10 Testes Econométricos + Veredito Cruzado
- **Estacionariedade & Raiz Unitária:**
  - **ADF (*Augmented Dickey-Fuller*)**: especificações `n`, `c` e `ct` com seleção de lags por AIC e valores críticos `1% / 5% / 10%`.
  - **Phillips-Perron (PP)**: implementação econométrica exata da estatística $Z_t$ com variância de longo prazo de **Newey-West (núcleo de Bartlett)** e p-valores de MacKinnon.
  - **KPSS**: hipótese nula invertida ($H_0$: estacionária) em nível (`c`) e tendência (`ct`).
  - **Zivot-Andrews (1992)**: teste de raiz unitária com **quebra estrutural endógena**, identificando a data exata da ruptura (corrige o viés de Perron).
- **Autocorrelação & Ruído Branco:**
  - **Ljung-Box ($Q^*$)** e **Box-Pierce ($Q$)** nos resíduos com correção de graus de liberdade (`model_df = p + q + P + Q`), curva interativa de p-valores por lag (`1..24`) e **Durbin-Watson**.
  - **Correlogramas ACF & PACF** com bandas de Bartlett 95% ($\pm 1,96/\sqrt{N}$) e leitura automática **Box-Jenkins**.
- **Heterocedasticidade & Volatilidade:**
  - **Engle ARCH-LM** (clusters de volatilidade condicional → diagnóstico para modelos GARCH).
  - **Breusch-Pagan Temporal** (variância crescente ao longo do tempo → diagnóstico para Log/Box-Cox).
  - Gráficos de **Média Móvel** e **Desvio-Padrão Móvel** (*Rolling Mean / Rolling Std*).
- **Normalidade, Caudas Pesadas & Outliers:**
  - **Anderson-Darling**, **Jarque-Bera** (Assimetria e Curtose) e **Shapiro-Wilk**.
  - Histogramas com curva Normal teórica, **Normal Q-Q Plot** e auditoria automática de outliers via **IQR (1,5×)** e **Z-Score Robusto (MAD > 3,5)**.

### 2. Laboratório de Transformações (Antes vs. Depois em 1 Clique)
Avalia simultaneamente **7 transformações** (`Nível`, `Log Natural`, `Box-Cox Ótimo`, `1ª Diferença Δ`, `Diferença Sazonal Δₛ`, `Δ + Δₛ` e `Log-Retorno Δ ln`) com ADF, PP, KPSS e ARCH-LM lado a lado, destacando a transformação estacionária mais parcimoniosa e permitindo aplicá-la em todo o dashboard com 1 clique.

### 3. 5 Cenários Econométricos Prontos + Upload Inteligente CSV / Excel
1. **Vendas Varejo — Tendência Estocástica `I(1)` + Sazonalidade Anual** (`252` meses)
2. **Retornos Financeiros — Clusters de Volatilidade GARCH** (estacionária na média, falha em `ARCH-LM` e normalidade)
3. **Atividade Econômica — Quebra Estrutural Estilo 2020** (viés de Perron: engana o ADF, mas **Zivot-Andrews** identifica a data da quebra!)
4. **Tráfego & Receita — Variância Multiplicativa Crescente** (dispara `Breusch-Pagan Temporal` no nível e estabiliza com Log/Box-Cox)
5. **Processo AR(1) Gaussiano — Controle Estacionário Limpo** (referência didática)

- **Upload CSV / XLSX:** detecção automática de separador (`,` / `;` / `\t`), suporte a decimais brasileiros (`,`), escolha de aba do Excel, formato de data, período sazonal (`s = 1, 4, 7, 12, 52`) e agregação opcional.

### 4. Decomposição STL Robusta, SARIMA & Exportação Excel (5 Abas)
- **STL Robusta** com índices de força de tendência ($F_T$) e sazonalidade ($F_S$) de Wang, Smith & Hyndman (2006).
- **SARIMA Auto (AICc) ou Manual** `(p,d,q)(P,D,Q,s)` com **Backtest *Out-of-Sample* vs. *Seasonal Naive*** (`RMSE`, `MAE`, `MAPE`, `% Ganho`) e projeção com intervalos **IC 80%** e **IC 95%**.
- **Relatório Excel `.xlsx`** estilizado com 5 abas (`Resumo_Executivo`, `Testes_Estatisticos`, `Matriz_Transformacoes`, `Serie_e_STL`, `Previsao_SARIMA`) + download CSV filtrado.

---

## 🏗️ Arquitetura do Projeto

```text
app_teste_estatistico_ts/
├── backend/
│   ├── app/
│   │   ├── datasets.py          # 5 cenários econométricos + parser inteligente CSV/XLSX
│   │   ├── diagnostics.py       # Motor de 10 testes, PP Newey-West, ZA, STL, matriz 7 transformações e SARIMA
│   │   ├── exporter.py          # Gerador de workbook Excel multi-abas (openpyxl)
│   │   ├── schemas.py           # Contratos Pydantic v2
│   │   └── main.py              # API REST FastAPI (porta 8005)
│   ├── tests/
│   │   └── test_diagnostics.py  # 5 testes automatizados cobrindo os 5 regimes, PP, upload BR e Excel
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SidebarControls.tsx
│   │   │   ├── StatTestsTab.tsx
│   │   │   ├── TransformsTab.tsx
│   │   │   ├── VisualsDistTab.tsx
│   │   │   ├── DecompForecastTab.tsx
│   │   │   ├── DataTableTab.tsx
│   │   │   └── EconometricGuideTab.tsx
│   │   ├── api.ts
│   │   ├── types.ts
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts           # Porta 5178 com proxy /api -> 127.0.0.1:8005
└── README.md
```

---

## 🚀 Como Executar Localmente

> **Portas dedicadas (sem conflito com os demais apps do portfólio):** Backend FastAPI `8005` | Frontend React/Vite `5178`.

### 1. Backend (Python + FastAPI)
```bash
cd backend
pip install -r requirements.txt

# Executar testes automatizados (5/5 OK)
python -m unittest discover -s tests -v

# Subir API na porta 8005 (Swagger em http://127.0.0.1:8005/docs)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8005 --reload
```

### 2. Frontend (React + TypeScript + Vite)
```bash
cd frontend
npm install
npm run dev
# Acesse: http://localhost:5178
```
