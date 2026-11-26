# DOCUMENTAÇÃO TÉCNICA COMPLETA — APP FORECASTS (PYTHON + MODERN FRONTEND)

## 1. Visão Geral da Arquitetura

O **App Forecasts** foi integralmente reescrito em **Python** no backend utilizando **FastAPI** e um **Frontend Web Moderno (SPA)** em **HTML5, CSS3 e JavaScript ES6 puro com Plotly.js**, sem qualquer dependência de frameworks de UI em Python.

### Fluxo de Comunicação
- O frontend ([`frontend/index.html`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/frontend/index.html), [`frontend/styles.css`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/frontend/styles.css), [`frontend/app.js`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/frontend/app.js)) consome a API REST JSON exposta pelo FastAPI ([`backend/main.py`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/backend/main.py)) na porta `8060`.
- O estado da sessão ([`backend/state.py`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/backend/state.py)) mantém o `DataFrame` bruto (`raw_df`), o `DataFrame` processado (`processed_df`), o status de validação da coluna temporal (`date_ok`), o histórico cronológico de operações (`operation_logs`) e os objetos/resultados dos modelos treinados.

---

## 2. Detalhamento dos Módulos e Funcionalidades

### ABA 1: INTRODUÇÃO
- **Hero Banner & Quick Start**: Permite iniciar o fluxo de 4 etapas ou carregar instantaneamente o dataset de exemplo (`dados_virgula.csv`) com um clique no botão **"Carregar Exemplo Rápido"**.
- **Status Global**: Exibe no topo da aplicação o arquivo ativo, número de linhas e quantidade de colunas disponíveis para todos os módulos.
- **Tema Claro / Escuro**: Alternância dinâmica que adapta tanto a interface quanto o layout de todos os gráficos Plotly.js.

---

### ABA 2: IMPORTAÇÃO E PREPARAÇÃO DE DADOS
Implementada no backend em [`backend/data_import.py`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/backend/data_import.py), estruturada em **8 etapas**:

1. **Importar Dados (`importar_dados`)**:
   - **CSV / TXT**: Detecta automaticamente o separador (`;` ou `,`) e converte decimais com vírgula quando aplicável.
   - **Excel (`.xlsx`, `.xls`, `.xlsm`)**: Lista todas as planilhas (*sheets*) disponíveis via `/api/import/inspect` e importa a planilha selecionada.
   - **RDS (`.rds`)**: Leitura nativa de objetos R via `pyreadr`.
   - **Stata (`.dta`) e SPSS (`.sav`)**: Leitura nativa via `pandas` / `pyreadstat`.
   - **SQLite (`.sqlite`, `.db`, `.sqlite3`)**: Inspeciona o catálogo `sqlite_master`, lista as tabelas disponíveis e importa a tabela escolhida.
2. **Converter Coluna de Data (`inspecionar_coluna_data` & `converter_coluna_para_date_ok`)**:
   - Identifica colunas candidatas a data por tipo ou nome (`Data`, `date`, `mes`, `ano`, `periodo`, etc.).
   - Reconhece formatos ISO (`YYYY-MM-DD`), brasileiros (`DD/MM/YYYY`), americanos (`MM/DD/YYYY`), anos (`YYYY`), ano-mês (`YYYYMM`), datas seriais do Excel e timestamps Unix.
   - Cria a coluna padronizada `date_ok` posicionada na primeira coluna do `DataFrame`.
3. **Gerar Variáveis e Dummies (`gerar_dummies`)**:
   - `dummy_ano_YYYY` e `dummy_mes_1` a `dummy_mes_12`
   - `trend`: sequência linear ordenada cronologicamente ($1 \dots n$)
   - `sin1`, `cos1`, `sin2`, `cos2`: harmônicos trigonométricos anuais ($\sin(2\pi m/12)$, $\cos(2\pi m/12)$, $\sin(4\pi m/12)$, $\cos(4\pi m/12)$)
   - Estações do Hemisfério Sul com corte exato por dia/mês (`verao`: 21/Dez–20/Mar, `outono`: 21/Mar–20/Jun, `inverno`: 21/Jun–20/Set, `primavera`: 21/Set–20/Dez)
   - `bro_bro`: dummy para o período quente do Nordeste (Setembro a Dezembro)
   - Crises históricas pré-configuradas (`pandemia_covid`, `crise_covid_brasil`, `crise_2008`, `crise_dot_com`, `crise_dilma`, `crise_divida_80`, `hiperinflacao_80_90`, `crise_cambio_1999`)
   - Dummy personalizada com validação de nome e intervalo de datas.
4. **Transformar Variáveis Numéricas (`transformar_dados`)**:
   - **Log Natural (`log`)**: Cria `ln_<col>`. Caso existam valores $\le 0$, aplica automaticamente $\ln(x + |\min(x)| + 10^{-9})$ e registra a constante no log.
   - **Box-Cox (`boxcox`)**: Cria `<col>_boxcox`. Corrige valores $\le 0$ e estima $\lambda$ ótimo via `scipy.stats.boxcox` (ou aplica o $\lambda$ informado pelo usuário).
   - **Padronizar (`scale`)**: Cria `<col>_scaled` com média $0$ e desvio padrão $1$.
   - **Centralizar (`center`)**: Cria `<col>_centered` com média $0$.
5. **Extração de Sazonalidade e Tendência (`extrair_sazonalidade`)**:
   - Interpola linearmente eventuais valores faltantes no período selecionado (mínimo de 12 meses).
   - Executa `statsmodels.tsa.seasonal.seasonal_decompose(period=12)`.
   - Mapeia o efeito sazonal médio de cada mês para toda a extensão da série (`<col>_sazonalidade`).
   - Ajusta regressão linear sobre o componente de tendência para interpolar e extrapolar `<col>_tendencia` em todas as datas da base.
   - Calcula estatísticas descritivas (Mínimo, Quartis, Média, Mediana, Máximo) e a **Importância Relativa (%)** da variância sazonal e da tendência (classificada em *Fraca*, *Moderada* ou *Forte*).
6. **Exportar Dados Processados (`/api/data/export`)**:
   - Exporta em `.xlsx`, `.csv` (UTF-8 com BOM) ou `.pkl`.
7. **Visualizar Tabela de Dados**:
   - Paginação configurável (10, 25, 50, 100 linhas) e busca textual instantânea em todas as colunas.
8. **Análise Gráfica Exploratória**:
   - Gráficos Plotly multi-variáveis em **Linhas**, **Barras**, **Dispersão** e **Boxplot Sazonal agrupado por mês**.

---

### ABA 3: PROJEÇÕES ML (PYCARET / CARET RECIPES)
Implementada em [`backend/forecast_caret.py`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/backend/forecast_caret.py):

- **Catálogo de Modelos (`LISTA_MODELOS_PADRAO`)**:
  - 45 algoritmos base + variantes `_scale`, `_boxcox` e `_tune` (totalizando **165 configurações**), além de suporte a upload de arquivo `.txt` customizado (`/api/caret/custom-models`).
  - Famílias suportadas:
    - *Regressão Linear & Robusta*: `lm` (`LinearRegression`), `rlm` (`HuberRegressor`), `bayesglm` (`BayesianRidge`), `ard` (`ARDRegression`), `lar` (`Lars`), `omp` (`OrthogonalMatchingPursuit`), `leapBackward` / `leapForward` / `leapSeq` / `lmStepAIC` (`RFE` + `LinearRegression`/`Ridge`), `BstLm` / `glmboost`.
    - *Regressão Penalizada & Aditiva*: `glmnet` / `enet` / `penalized` (`ElasticNet`), `ridge` (`Ridge`), `lasso` (`Lasso`), `llar` (`LassoLars`), `plsRglm` (`PLSRegression`), `gamSpline` / `gamLoess` (`SplineTransformer` + `Ridge`).
    - *SVM & Processos Gaussianos*: `svmLinear`, `svmLinear2`, `svmLinear3`, `svmPoly`, `svmRadial` (`SVR`), `gaussprLinear`, `gaussprPoly` (`GaussianProcessRegressor`).
    - *Árvores, Ensembles & Boosting*: `rf`, `ranger`, `Rborist` (`RandomForestRegressor`), `et` (`ExtraTreesRegressor`), `gbm` (`GradientBoostingRegressor`), `xgboost` (`XGBRegressor`), `AdaBag` (`AdaBoostRegressor`), `treebag` (`BaggingRegressor`), `dt` (`DecisionTreeRegressor`), `knn` (`KNeighborsRegressor`).
    - *Redes Neurais*: `brnn` / `mlp` (`MLPRegressor`).
- **Ordem Estrita de Featurização**:
  - Os dados são divididos cronologicamente no período de treino `[data_inicio, data_teste]` **antes** de ajustar qualquer etapa da receita (`Pipeline`), evitando *data leakage*.
- **Dashboard de Aderência / Backtest Externo (`gerar_dashboard_recomendacao_adaptado`)**:
  - Compara as projeções com os valores realizados nos últimos $N$ meses (`periodo_analise`):
  - Calcula `RMSE`, `MAPE (%)`, `MPE (%)`, `MedAE`, `RMSLE`, `R²`, `Explained_Var`, `Desv.Máx (%)` e $\text{Score} = \text{MAPE} + 0.4 \times \text{Desv.Máx}$.
  - Classifica automaticamente cada modelo:
    - `Ótimo`: $\text{Score} \le 1.5$ e nenhum desvio mensal $> 5\%$
    - `Recomendado`: $\text{Score} \le 2.5$ e até 1 desvio $> 5\%$
    - `Aceitável`: $\text{Score} \le 3.5$ e até 1 desvio $> 5\%$
    - `Avaliar`: $\text{Score} \le 5.5$
    - `Reprojetar`: $\text{Score} > 5.5$
- **Importância em Cascata (`extrair_importancia_modelo`)**:
  - Extrai coeficientes, `feature_importances_` ou importância por permutação, normaliza para $100\%$ e constrói o gráfico Waterfall cumulativo (`Start` $\to$ `End`).
- **Elasticidades (`calcular_elasticidades_modelo`)**:
  - Aplica perturbações de $+1\%, +5\%, +10\%, +15\%$ em cada variável explicativa $X_j$ e calcula a elasticidade mediana (% de variação em $\hat{Y}$ por $1\%$ de variação em $X_j$), classificando o impacto em **Positivo** (verde) ou **Negativo** (vermelho).
- **Exportação**:
  - Planilha Excel multi-abas (`Projecoes`, `Backtest_Aderencia`, `Metricas_Internas`, `Configuracoes`, `Dados_Treino_Usados`), CSV de importância e pacote `.joblib` com os modelos treinados.

---

### ABA 4: PROJEÇÕES ARIMA / SARIMAX
Implementada em [`backend/forecast_arima.py`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/backend/forecast_arima.py):

- Padroniza datas para o primeiro dia do mês e separa o conjunto de treino e os `n_teste` meses finais para validação.
- **Modo Automático**: Realiza teste ADF (`adfuller`) para estimar a ordem de integração $d$, seleciona a ordem base $(p,d,q)(P,D,Q)_m$ via critério de informação de Akaike (AIC) e gera `num_models` variações adjacentes.
- **Modo Manual**: Ajusta a ordem exata informada pelo usuário.
- **Univariado + Multivariado (ARIMAX)**: Quando variáveis exógenas são selecionadas, verifica colinearidade (`check_collinearity`), imputa eventuais NAs futuros das exógenas pela média de treino e ajusta tanto os modelos `arima_uni_i` quanto `arima_multi_i`.
- Gera previsões com intervalos de confiança de 95% (`Lo95`, `Hi95`), tabela de erros (`RMSE`, `MAE`, `MAPE`), estatísticas de desvios no teste e Dashboard de Recomendação ponderado por `peso_recente` e `peso_desvio_max`.

---

### ABA 5: PROJEÇÕES AUTOTS & ANÁLISE DE CRESCIMENTO
Implementada em [`backend/forecast_autots.py`](file:///d:/OneDrive/Documentos/GitHub/app_forecasts_R/backend/forecast_autots.py):

1. **Sub-aba Modelos Univariados AutoTS**:
   - Ajusta até 7 algoritmos univariados com intervalos de previsão de 95%:
     - `prophet`: Tendência linear segmentada (*changepoints*) + série de Fourier sazonal
     - `ets`: Suavização Exponencial de Holt-Winters (`ExponentialSmoothing`)
     - `sarima`: SARIMAX sazonal
     - `tbats`: Transformação Box-Cox + harmônicos trigonométricos + erros ARMA
     - `bats`: Box-Cox + Suavização Exponencial com tendência amortecida (*damped trend*)
     - `stlm`: Decomposição STL + Suavização Exponencial (`STLForecast`)
     - `bagged`: Ensemble combinando as projeções médias e limites inferiores/superiores dos modelos ativos.
   - Calcula o ranking de erros e o **Dashboard de Desvios AutoTS (`gerar_dashboard_autots`)** com pesos configuráveis.
2. **Sub-aba Análise de Crescimento (`generate_df_visoes_ATS`)**:
   - Consolida a série histórica realizada + projeção futura do modelo selecionado em 4 granularidades:
     - **Mensal** (colunas `Jan` a `Dez` por `Ano`)
     - **Trimestral** (`Q1` a `Q4` por `Ano`)
     - **Semestral** (`S1` e `S2` por `Ano`)
     - **Anual** (`Valor_Anual` por `Ano`)
   - Para cada granularidade, gera 3 tabelas e gráficos interativos (Linhas, Barras coloridas por sinal e Combinado):
     - **Valores Absolutos**
     - **Taxas de Crescimento YoY (%)** em relação ao mesmo período do ano anterior
     - **Variações Absolutas YoY** em relação ao mesmo período do ano anterior
   - Exporta todas as 16 planilhas consolidadas em um único arquivo Excel (`/api/autots/export-growth`).

---

## 3. Resumo dos Endpoints da API REST

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `GET` | `/api/state` | Retorna o estado global da sessão, colunas, datas e logs |
| `POST` | `/api/reset` | Limpa os dados e reinicia o estado da sessão |
| `GET` | `/api/samples` | Lista os arquivos de exemplo disponíveis na pasta `data/` |
| `POST` | `/api/import/inspect` | Inspeciona tipo, planilhas Excel ou tabelas SQLite de um arquivo |
| `POST` | `/api/import/upload` | Faz upload e importa arquivo local |
| `POST` | `/api/import/sample` | Importa um arquivo de exemplo da pasta `data/` |
| `GET` | `/api/date/inspect` | Diagnostica a coluna de data selecionada |
| `POST` | `/api/date/convert` | Converte a coluna selecionada para `date_ok` |
| `POST` | `/api/dummies/generate` | Gera variáveis temporais, estações, B-ró B-ró e dummies de crises |
| `POST` | `/api/transform/apply` | Aplica transformações Log, Box-Cox, Scale ou Center |
| `POST` | `/api/seasonality/extract` | Extrai colunas de sazonalidade e tendência e gera relatório |
| `GET` | `/api/data/table` | Retorna linhas paginadas/filtradas do dataset ativo |
| `GET` | `/api/data/export` | Exporta os dados processados em `.xlsx`, `.csv` ou `.pkl` |
| `GET` | `/api/caret/models` | Lista as 165 configurações de modelos PyCaret / Caret |
| `POST` | `/api/caret/custom-models` | Carrega lista customizada de modelos via arquivo `.txt` |
| `POST` | `/api/caret/train` | Treina pipelines ML, calcula backtest, importância em cascata e elasticidades |
| `POST` | `/api/caret/backtest` | Recalcula o backtest externo para uma nova janela de meses |
| `GET` | `/api/caret/export-projections` | Exporta planilha Excel multi-abas com resultados do módulo ML |
| `GET` | `/api/caret/export-models` | Exporta os objetos dos modelos treinados em `.joblib` |
| `POST` | `/api/arima/run` | Executa modelagem ARIMA e SARIMAX (univariada e multivariada) |
| `POST` | `/api/arima/dashboard` | Atualiza os pesos do score de recomendação ARIMA |
| `GET` | `/api/arima/export` | Exporta projeções ARIMA selecionadas em `.xlsx` |
| `POST` | `/api/autots/run` | Executa a suíte AutoTS e pré-calcula as visões de crescimento |
| `POST` | `/api/autots/desvios` | Atualiza o dashboard de desvios do AutoTS |
| `POST` | `/api/autots/growth` | Gera as tabelas e séries de crescimento Mensal, Trimestral, Semestral e Anual |
| `GET` | `/api/autots/export` | Exporta todas as projeções AutoTS ou apenas o modelo selecionado em `.xlsx` |
| `GET` | `/api/autots/export-growth` | Exporta o relatório completo de crescimento em Excel multi-abas |
