# App Forecasts — Machine Learning & Séries Temporais (Python + Modern Web UI)

Plataforma web completa para importação multi-formato, engenharia de variáveis temporais, decomposição sazonal e projeções avançadas de séries temporais com **Machine Learning (arquitetura PyCaret / Caret Recipes)**, **ARIMA / SARIMAX** e **AutoTS com Análise de Crescimento**.

---

## Arquitetura do Projeto

O aplicativo foi construído com separação total entre **Backend (Python FastAPI)** e **Frontend Web Moderno (HTML5 + CSS3 + JavaScript ES6 + Plotly.js — zero Python no frontend)**:

```plaintext
app-previsao-automl-series-temporais/
├── backend/                        # Backend Python (FastAPI + Engines de Modelagem)
│   ├── __init__.py
│   ├── main.py                     # Servidor FastAPI, rotas REST (/api/...) e exportação
│   ├── state.py                    # Gerenciamento de estado da sessão e logs
│   ├── data_import.py              # Importação multi-formato, conversão date_ok, dummies, transformações e sazonalidade
│   ├── forecast_caret.py           # Engine ML (PyCaret / Caret Recipes, 165 modelos, Waterfall Importance, Elasticidades, Backtest)
│   ├── forecast_arima.py           # Engine ARIMA / SARIMAX (Auto.ARIMA, variações, exógenas, IC 95% e dashboard)
│   └── forecast_autots.py          # Engine AutoTS (Prophet, ETS, SARIMA, TBATS, BATS, STLM, Bagged) e Visões de Crescimento
├── frontend/                       # Frontend Web Moderno (SPA)
│   ├── index.html                  # Estrutura das 5 abas e painéis interativos
│   ├── styles.css                  # Design system moderno com suporte a Tema Claro e Escuro
│   └── app.js                      # Cliente REST ES6 + gráficos interativos Plotly.js
├── data/                           # Conjuntos de dados de exemplo (CSV, Excel, RDS, Stata, SQLite)
├── tests/                          # Suíte automatizada de testes unitários e de integração da API
│   ├── __init__.py
│   └── test_app.py                 # Testes end-to-end dos 5 módulos, 6 formatos e exportações
├── www/                            # Recursos visuais estáticos (logo e imagens de exemplo)
├── iniciar_app.bat                 # Inicializador Windows (porta dedicada 8060 + healthcheck /api/health)
├── run_app.py                      # Script Python de inicialização do Uvicorn (porta 8060)
├── requirements.txt                # Dependências Python enxutas e pinadas do projeto
├── DOCUMENTACAO_APP.md             # Documentação técnica detalhada de todos os módulos e endpoints
├── README.md                       # Guia rápido do projeto
└── .gitignore                      # Configuração de arquivos ignorados pelo Git
```

---

## Como Instalar e Executar

### 1. Pré-requisitos
Instale as dependências em Python **3.10+**:

```bash
pip install -r requirements.txt
```

### 2. Opção 1: Duplo clique no `.bat` (Windows)
Dê duplo clique em `iniciar_app.bat`. O script irá:
1. Liberar exclusivamente a porta dedicada `8060` (caso esteja ocupada por uma sessão anterior).
2. Iniciar o servidor FastAPI (`.venv\Scripts\python.exe` ou `python` do PATH).
3. Aguardar `GET /api/health` responder `HTTP 200` antes de abrir **http://127.0.0.1:8060** no navegador.

### 3. Opção 2: Via Terminal
```bash
python run_app.py
```
Acesse no navegador: **http://127.0.0.1:8060** (documentação OpenAPI interativa em `/docs`).

### 4. Testes Automatizados
Execute a suíte de testes unitários e de integração (`FastAPI TestClient`, cobrindo importação multi-formato, preparação em 8 etapas, ML Recipes/Backtest/Cascata/Elasticidades, ARIMA/SARIMAX com exógenas futuras, AutoTS 7 modelos e visões de crescimento):

```bash
python -m unittest discover -s tests -v
```

---

## Funcionalidades Principais

### 1. Importação e Preparação de Dados (8 Etapas)
- **Múltiplos Formatos**: Upload local ou carregamento rápido da pasta `data/` para arquivos `.csv`, `.txt`, `.xlsx`, `.xls`, `.rds` (R), `.dta` (Stata), `.sav` (SPSS) e `.sqlite` / `.db`.
- **Conversão Inteligente de Datas**: Inspeção automática e conversão para a coluna padronizada `date_ok`.
- **Engenharia de Variáveis Temporais e Dummies**:
  - Dummies de Ano (`dummy_ano_YYYY`) e Mês (`dummy_mes_1..12`)
  - Tendência Linear (`trend`) e Funções Trigonométricas (`sin1`, `cos1`, `sin2`, `cos2`)
  - Estações do Hemisfério Sul (`verao`, `outono`, `inverno`, `primavera`) e Período **"B-ró B-ró"** (Set–Dez)
  - 8 Crises Pré-definidas (COVID Global/Brasil, Crise 2008, Bolha .com, Crise 2014–2016, etc.) e Dummy Personalizada.
- **Transformações Numéricas**: Log Natural (`ln(x)` com tratamento automático de valores $\le 0$), Box-Cox ($\lambda$ automático ou manual), Padronização (`Scale`) e Centralização (`Center`).
- **Decomposição de Sazonalidade e Tendência**: Extrai `<var>_sazonalidade` e `<var>_tendencia` com cálculo de importância relativa (% da variância).
- **Tabela Paginada, Gráficos Exploratórios (Linha, Barra, Dispersão, Boxplot Mensal) e Exportação (`.xlsx`, `.csv`, `.pkl`)**.

### 2. Projeções Machine Learning (PyCaret / Caret Recipes)
- **Catálogo de 165 Modelos**: Regressões Lineares, Robustas (Huber), Penalizadas (Ridge, Lasso, ElasticNet, LARS, OMP, Bayesian Ridge, ARD, PLS, GAM Splines), SVM (Linear, Polinomial, Radial), Processos Gaussianos, Árvores e Ensembles (Random Forest, ExtraTrees, Bagging, AdaBoost, Gradient Boosting, XGBoost), KNN e Redes Neurais (MLP), nas versões base, `_scale`, `_boxcox` e `_tune`.
- **Pré-processamento Avançado (Recipes)**: Imputação (KNN, Iterative/Bagged, Mediana, Média, Moda), Transformações Numéricas, Engenharia de Features (Interações, Polinômios, Lags, Médias Móveis, Features de Data) e Redução de Dimensionalidade (PCA, ICA, PLS).
- **Dashboard de Aderência (Backtest Externo)**: Avalia `RMSE`, `MAPE`, `MPE`, `MedAE`, `RMSLE`, `R²`, `Explained_Var`, `Desv.Máx (%)`, Score Composto e Status (`Ótimo`, `Recomendado`, `Aceitável`, `Avaliar`, `Reprojetar`).
- **Interpretabilidade**: Gráfico em Cascata (*Waterfall*) de Importância Cumulativa das Variáveis e Gráfico/Tabela de Elasticidades ($\Delta\% Y / \Delta\% X$).

### 3. Projeções ARIMA / SARIMAX
- **Auto.ARIMA + Variações** ou **Modo Manual** $(p, d, q)(P, D, Q)_m$.
- Ajuste simultâneo de modelos **Univariados** e **Multivariados (ARIMAX com variáveis exógenas)**, intervalos de confiança de 95%, análise de desvios no período de teste e exportação para Excel.

### 4. Projeções AutoTS & Análise de Crescimento
- Comparação automática entre **Prophet**, **ETS (Holt-Winters)**, **SARIMA**, **TBATS**, **BATS**, **STLM (STL + ETS)** e **Bagged Ensemble**.
- **Análise de Crescimento Multi-Granularidade**: Tabelas e gráficos interativos nas visões **Mensal**, **Trimestral**, **Semestral** e **Anual** com **Valores Absolutos**, **Taxas de Crescimento YoY (%)** e **Variações Absolutas YoY**.
