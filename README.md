# 📊 Portfólio de Ciência de Dados, Econometria & Engenharia de Dados

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/silvio-paula/)
[![Portfólio Web](https://img.shields.io/badge/Portfólio_Web-181717?style=for-the-badge&logo=github&logoColor=white)](https://silviopaula.github.io/)
[![Artigos Científicos](https://img.shields.io/badge/Artigos_Científicos-2C5BB4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://github.com/silviopaula/Portifolio-Artigos-Publicados-Economia)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:silvio.rosa.paula@gmail.com)

</div>

---

## 👋 Sobre Mim

Sou **Silvio da Rosa Paula**, **Doutor em Economia Aplicada** pela Universidade Federal de Pelotas (UFPel), com especialização em **Inteligência Artificial e Ciência de Dados** pela Universidade Anhanguera.

Atuo como Cientista de Dados combinando **inferência causal, econometria aplicada, machine learning, séries temporais, NLP/LLMs e engenharia de dados (Apache Spark / APIs)** para resolver problemas reais de negócio e tomada de decisão baseada em evidências.

---

## 🗂️ Catálogo de Projetos

> Projetos sinalizados com ⭐ foram revisados recentemente com documentação completa, execução validada e foco didático/reprodutível.

### 1. 🧪 Inferência Causal, Econometria & Experimentação

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**teste-ab-retencao-jogadores**](./teste-ab-retencao-jogadores/) | Avaliação estatística rigorosa do posicionamento de *gates* na retenção de 90.189 jogadores (*Cookie Cats*), com diagnóstico SRM, Bootstrap (10k), análise de *compliance* e *Cohen's h*. | Python, SciPy, Statsmodels, Bootstrap |
| ⭐ [**dimensionamento-experimentos-mde**](./dimensionamento-experimentos-mde/) | Guia didático sobre a "armadilha do p-valor": cálculo de Poder Estatístico (*Power*) e *Minimum Detectable Effect* (MDE) em painel *Difference-in-Differences* (TWFE). | Python, Statsmodels, SciPy, DiD TWFE |
| ⭐ [**endogeneidade-mqo-vs-2sls**](./endogeneidade-mqo-vs-2sls/) | Demonstração prática do viés de endogeneidade na elasticidade-preço, correção via Variáveis Instrumentais (IV / 2SLS) e o paradoxo entre RMSE histórico e decisão de *Pricing*. | Python, Linearmodels (`IV2SLS`), Statsmodels |
| ⭐ [**painel-dinamico-system-gmm**](./painel-dinamico-system-gmm/) | Guia completo de Painel Dinâmico (*Two-Step System-GMM* + Windmeijer): simulação Monte Carlo provando o Viés de Nickell, replicação do artigo na RBE (27 UFs, 1996–2015), proliferação de instrumentos, Efeitos de Longo Prazo (Método Delta) e *Specification Curve*. | Python, `pydynpd`, Linearmodels, SciPy, Stata (`xtabond2`) |
| ⭐ [**app-causal-impact-bsts-kalman**](./app-causal-impact-bsts-kalman/) | Aplicação web *full-stack* (**CausalImpact Studio**) para inferência causal contrafactual em séries temporais via modelos estruturais em espaço de estados (**BSTS / Kalman + Simulação Posterior Monte Carlo**), com os 3 painéis clássicos de Brodersen et al., **Teste Placebo no Tempo**, auditoria *No-Spillover* das covariáveis, 3 cenários demo (incluindo placebo nulo), upload CSV/XLSX e exportação Excel multi-abas. | Python, FastAPI, Statsmodels (`UnobservedComponents`), React, TypeScript, Tailwind, Recharts |
| ⭐ [**databricks-impacto-causal-lei-cotas-genero-eleicoes**](./databricks-impacto-causal-lei-cotas-genero-eleicoes/) | Engenharia de dados distribuída em **Databricks (PySpark + Unity Catalog / Medallion Bronze→Silver→Gold)** sobre microdados do **TSE/BigQuery (`1998–2022`)** para avaliação causal da **Lei nº 12.034/2009** (cota de 30%): imputação probabilística de gênero (`12.537 → 0` ausentes em 1996), painel balanceado LOCF (`4.000` linhas × `27` UFs × `7` cargos × `25` anos), avanço de `16,23% → 29,10%` (`35,25%` em 2022) nos cargos proporcionais vs. controles majoritários, com **Databricks Asset Bundle (`databricks.yml`)**, SQL BigQuery e scripts modulares `CausalImpact` / DiD. | PySpark, Databricks, Unity Catalog, Delta Lake, BigQuery, CausalImpact, DiD |
| ⭐ [**marketing-digital-vies-selecao-causalidade**](./marketing-digital-vies-selecao-causalidade/) | Laboratório empírico de **Inferência Causal em Marketing Digital** sob **viés de seleção observável e tendências não-paralelas** (`250` lojas × `24` meses = `6.000` obs, ATT verdadeiro = `R$ 6.012,28/mês`) comparando **24 especificações reprodutíveis em Python**: *Entropy Balancing* (`SMD = 0,0000`, `-0,51%`), *AIPW*, *PSM Abadie-Imbens*, *ETWFE de Wooldridge* e *Event Study* (`+R$ 78/mês, p = 0,024` → `+R$ 1,64/mês, p = 0,950`), *Callaway & Sant'Anna / DR-DiD*, *Synthetic Control / ASCM / SDID*, *BSTS*, *Meta-Learners* (`CATE r = 0,95`, PEHE `R$ 162`) e **Auditoria de *Panel Cross-Fitting Leakage* no DoubleML** (`KFold` por linha `+96,6%` → `GroupKFold` por loja `-1,16%`). | Python, DoubleML, Statsmodels, SciPy, Scikit-Learn, Unittest |
| [**causal-ml-uplift-marketing**](./causal-ml-uplift-marketing/) | Aprendizado causal (*Causal ML / Uplift Modeling*, Meta-learners e Double Machine Learning) aplicado à otimização de promoções e descontos. | Python, Causal ML, Uplift, DML |

---

### 2. 🤖 Machine Learning, Análise de Sobrevivência & MLOps

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**analise-sobrevivencia-churn-clientes**](./analise-sobrevivencia-churn-clientes/) | Estudo completo de *Time-to-Churn* com censura à direita: *Swimmer Plot*, Kaplan-Meier, Log-Rank, Nelson-Aalen, Regressão de Cox (*Hazard Ratios*) e XGBoost `survival:cox`. | Python, Lifelines, XGBoost, Scikit-Learn |
| ⭐ [**gestao-estoques-cluster-classificacao**](./gestao-estoques-cluster-classificacao/) | Pipeline *end-to-end* sem *Data Leakage* sobre 198.917 SKUs combinando Clusterização (`PCA`, `K-Means`, `GMM`, `KNN`) + Classificação (`LightGBM`, `XGBoost`) e otimização de *threshold* para decisão de inventário. | Python, Scikit-Learn, LightGBM, XGBoost |
| ⭐ [**mlops-churn-mlflow-registry**](./mlops-churn-mlflow-registry/) | Pipeline MLOps *end-to-end* com **MLflow 3.3** (SQLite Tracking Store + **Model Registry** + Aliases `@baseline` / `@challenger` / `@champion` + **Quality Gate automatizado**) sobre ABT temporal de churn (**4.055 registros, 1.506 clientes, Abr–Nov/2024**): split cronológico *Out-of-Time* sem vazamento, `ChurnFeatureEngineer` embutido no artefato (**Zero *Training-Serving Skew*** `1,1e-16`), benchmark de **9 modelos** (`@champion Soft-Voting`: `OOT ROC-AUC 0,8339`, `PR-AUC 0,8069`, `Brier 0,1707`), monitoramento de **Data Drift (PSI)**, otimização de **ROI de retenção** (`τ = 0,29`, `+36%` lucro sobre abordar todos), *Slice Error Analysis* e CLI de inferência via `models:/churn-predictor@champion`. | Python, MLflow 3.3, Scikit-Learn, SQLite, Unittest |

---

### 3. ⚡ Engenharia de Dados, Big Data (Spark) & APIs

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**api-fastapi-consulta-usuarios**](./api-fastapi-consulta-usuarios/) | API REST modular de alta performance para ingestão e consultas analíticas sobre 100.000 usuários em JSON aninhado (67 MB) com latência `< 2 ms` e suíte de testes automatizados. | Python, FastAPI, Pydantic v2, Uvicorn, Unittest |
| ⭐ [**spark-performance-big-data-joins**](./spark-performance-big-data-joins/) | Benchmark real no Apache Spark 4.0 (15M linhas, 95% *Data Skew*) comparando `SortMergeJoin`, `Repartition`, `Salting Manual`, `AQE Skew Join` e `Broadcast Hash Join` (20,7x *speedup*). | PySpark 4.0, Catalyst Explain, AQE, Salting |
| ⭐ [**spark-comercio-exterior-tarifas-eua**](./spark-comercio-exterior-tarifas-eua/) | Engenharia distribuída em **Apache Spark 4.0** sobre **36,5M de registros oficiais SECEX (`2010–Jun/2025`, US\$ 7,09 tri)** com auditoria *Star-Schema* (eliminação de *fan-out* de `10,7×`), `BroadcastHashJoin` (`2,07×` *speedup*), evolução do **HHI**, exposição EUA/China/Rússia, econometria da demanda de suco de laranja (`MQO`, `HAC`, `ARDL` e teste crítico `2SLS/IV`) e **Monte Carlo (10.000 cenários tarifários EUA: 10%, 25% e 50% + US\$ 415/t)**. | PySpark 4.0, AQE, Broadcast Join, Statsmodels, Monte Carlo, Parquet |

---

### 4. 📈 Séries Temporais, Macroeconomia & *Foundation Models*

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**previsao-energia-ia-chronos-amazon**](./previsao-energia-ia-chronos-amazon/) | Benchmark *Zero-Shot* dos *Foundation Models* Amazon Chronos (Bolt & T5) vs. Holt-Winters e Seasonal Naive sobre 21 anos de consumo elétrico (EPE), projeção de 36 meses (P10–P90) e app Streamlit. | Python, Chronos 2.x, PyTorch, Statsmodels, Streamlit |
| ⭐ [**previsao-energia-deep-learning-lstm**](./previsao-energia-deep-learning-lstm/) | Jornada vertical de Deep Learning em séries temporais (**AutoKeras NAS → Keras Functional `ResidualLSTM` → TensorFlow `GradientTape` `ProbResidualAttentionLSTM`** com NLL Gaussiana, Atenção Temporal e bandas P10–P90) vs. Holt-Winters/SARIMAX (`MAPE 2,81%` em 24m OOS) e projeção 2025–2030. | Python, TensorFlow 2.20, Keras 3, AutoKeras 2, Statsmodels |
| ⭐ [**cointegracao-johansen-vecm-desemprego**](./cointegracao-johansen-vecm-desemprego/) | Cointegração de Johansen e Modelo Vetorial de Correção de Erros (VECM) para a taxa de desocupação brasileira (`IBC-Br`, `Selic`, `IAEmp`, `IIE-Br`), com Granger, IRF/FEVD, backtest 12m OOS (`MAPE 1,91%` vs. SARIMA `3,62%` e VAR-diff `23,12%`) e auditoria real Abr–Mai/2025. | Python, Statsmodels (`VECM`, `VAR`, `SARIMAX`) |
| ⭐ [**previsao-energia-modelo-estrutural-SEM**](./previsao-energia-modelo-estrutural-SEM/) | Sistema de 5 equações estruturais simultâneas (SEM/SUR) integrando PIB, Renda, Indústria, Comércio, Clima e Tarifas com decomposição causal (direto vs. indireto), backtest *out-of-sample* (MAPE `1,82%`) e cenários até 2034. | Python, Linearmodels (`SUR`), Statsmodels |
| ⭐ [**app-previsao-sarimax-box-jenkins**](./app-previsao-sarimax-box-jenkins/) | Aplicação web *full-stack* interativa e didática (*SARIMAX Forecast Studio*) com fluxo Box-Jenkins guiado (ADF/KPSS, ACF/PACF), Auto/Manual SARIMA & SARIMAX com variáveis exógenas, backtest vs. Seasonal Naive, diagnóstico Ljung-Box e exportação Excel. | Python, FastAPI, Statsmodels, React, TypeScript, Tailwind, Recharts |
| ⭐ [**app-previsao-automl-series-temporais**](./app-previsao-automl-series-temporais/) | Aplicação web *full-stack* (**App Forecasts**) em FastAPI + SPA Vanilla JS/Plotly (Dark/Light Mode) com **5 módulos integrados**: Importação multi-formato (`.csv` BR/US, `.xlsx`, `.rds`, `.dta`, `.sav`, `.sqlite`) + preparação em **8 etapas** (`date_ok`, dummies de calendário/estações/crises, Log/Box-Cox/Scale/Center, decomposição sazonal), **AutoML / Caret Recipes** (**165 configurações** entre Lineares, Penalizados, SVM, Processos Gaussianos, RF/ET/GBM/XGBoost e MLP com imputação, lags, janelas, PCA/ICA/PLS, `TimeSeriesSplit`, Backtest externo, **Importância em Cascata** e **Elasticidades**), **ARIMA/SARIMAX** univariado & multivariado com cenários exógenos futuros e checagem de colinearidade, e **AutoTS (7 modelos + Bagged Ensemble)** com **Análise de Crescimento YoY** (Mensal/Trimestral/Semestral/Anual, Excel 16 abas e exportação `.joblib`). | Python, FastAPI, Scikit-Learn, XGBoost, Statsmodels, SciPy, OpenPyXL, Joblib, Plotly.js, HTML5/CSS3/JS |
| ⭐ [**app-diagnostico-testes-series-temporais**](./app-diagnostico-testes-series-temporais/) | Aplicação web *full-stack* (**TimeSeries Diagnostics Studio**) com bateria completa de **10 testes econométricos** (**ADF `n/c/ct`**, **Phillips-Perron Newey-West**, **KPSS**, **Zivot-Andrews** com quebra estrutural endógena, **Ljung-Box / Box-Pierce** com `model_df`, **Engle ARCH-LM**, **Breusch-Pagan Temporal**, **Anderson-Darling**, **Jarque-Bera** e **Shapiro-Wilk**), Veredito Cruzado 4-Quadrantes, Laboratório de 7 Transformações, **5 regimes demo + upload CSV/XLSX**, STL Robusta ($F_T, F_S$), SARIMA + Backtest OOS e Excel 5 abas. | Python, FastAPI, Statsmodels, SciPy, OpenPyXL, React, TypeScript, Tailwind, Recharts |
| ⭐ [**app-exploracao-anomalias-previsao-ts**](./app-exploracao-anomalias-previsao-ts/) | Aplicação web *full-stack* (**TimeSeries Studio Pro**) unificando **7 módulos** em FastAPI + SPA Vanilla JS/Plotly (Dark/Light Mode): Central de Dados (`.xlsx`/`.xls`/`.csv` BR com detecção automática de separador e decimal), **Tratamento de Outliers** (**6 detectores** — Z-Score, IQR, Mahalanobis multivariado, Isolation Forest, LOF, One-Class SVM + **7 corretores** incluindo Kalman/SARIMAX e STL, Undo e Commit), **Sazonalidade & Controle XmR** (participação mensal ±1 DP, cartas $X$/$MR$, anomalias STL, ANOVA/Kruskal), **Editor Visual Interativo** (seleção Laço/Caixa, `Ctrl+Z`, cenários absolutos/percentuais), **ARIMA/SARIMAX** univariado & multivariado com checagem de colinearidade, **AutoTS (7 modelos + Ensemble Bagged)** e **Análise Multirresolução de Crescimento** (Mensal/Trimestral/Semestral/Anual com exportação Excel). | Python, FastAPI, Statsmodels, Scikit-Learn, SciPy, OpenPyXL, Plotly.js, HTML5/CSS3/JS |
| ⭐ [**app-estudos-macroeconomia-bcb**](./app-estudos-macroeconomia-bcb/) | Aplicação web *full-stack* (**Observatório Macro Brasil Studio**) conectada às APIs oficiais do Banco Central (**SGS + Olinda Focus**) com cache colunar **DuckDB/Parquet** (`76` séries BCB + `37` tabelas = `43.109` registros) e **60 estudos macroeconométricos autorais** (`M01`–`M60`, 10 por pilar), incluindo 6 laboratórios interativos (**Dominância Fiscal VAR/Granger**, **Projeção DBGG ARIMA**, **Regra de Taylor HAC**, **Inércia Rolling AR(1)**, **SARIMA IPCA** e **Curva de Phillips**). | Python, FastAPI, DuckDB, Parquet, Statsmodels, React, TypeScript, Tailwind, Recharts |

---

### 5. 🧠 NLP, LLMs & RAG

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**analise-discurso-banco-central-selic-NLP**](./analise-discurso-banco-central-selic-NLP/) | NLP financeiro e de política monetária sobre **111 atas oficiais do COPOM (2012–2026, até a 282ª Reunião)**: Índice *Hawkish–Dovish* (`r = +0,44` com a decisão seguinte da Selic), sentimento *Loughran–McDonald*, 5 tópicos latentes (TF-IDF + NMF), validação *Walk-Forward* e comparação entre 4 gestões do BCB. | Python, NLP, TF-IDF/NMF, Scikit-Learn, Statsmodels |
| ⭐ [**avaliacoes-ecommerce-olist-NLP**](./avaliacoes-ecommerce-olist-NLP/) | NLP em PT-BR & Modelagem Multimodal (**Texto + Logística**) sobre **98.673 pedidos únicos da Olist** (`42.173` avaliações escritas) com split cronológico *Out-of-Time*, preservação de negações (`+2,03 p.p.` Macro-F1), benchmark de **10 modelos** (**XGBoost Multimodal** `Macro-F1 0,6873` / `ROC-AUC 0,9718` e **Calibrated LinearSVC** `Acc 88,65%` / `PR-AUC 0,8936`), *Slice Error Analysis*, ABSA + NMF + Grafo `NetworkX` e **Copiloto RAG de SAC**. | Python, NLP (PT-BR), TF-IDF, LightGBM, XGBoost, Scikit-Learn, NetworkX |
| ⭐ [**classificacao-reviews-spark-NLP**](./classificacao-reviews-spark-NLP/) | NLP & Classificação Distribuída em **Apache Spark 4.0.1 (`pyspark.ml`)** sobre **50.000 avaliações reais do Yelp (`1–5★`)** em Parquet (`zstd`): *ablation* de colisão de hash (`HashingTF` `50 → 16.384`, `+21,3 p.p.` ROC-AUC), split estratificado *Zero-Leakage*, preservação de negações, **`Word2Vec` 64D nativo distribuído** e benchmark de **10 pipelines MLlib** (**Campeão Híbrido `TF-IDF 16k + Word2Vec 64D + Meta`**: `ROC-AUC 0,9155`, `PR-AUC 0,8778`, `F1 0,8090` em `τ = 0,37`, `Brier 0,1128`), com calibração e *Slice Error Analysis*. | PySpark 4.0, MLlib (`HashingTF`, `IDF`, `Word2Vec`, `LR`, `SVC`, `NB`, `RF`, `GBT`), Parquet |
| ⭐ [**app-rag-tutor-macroeconomia**](./app-rag-tutor-macroeconomia/) | Aplicação web *full-stack* (**MacroRAG Tutor Studio**) para estudo e pesquisa em livros-texto e notas de Macroeconomia (*Foundations of Modern Macroeconomics*, Heijdra — **816 páginas** indexadas por **19 capítulos** com cache local comprimido): busca híbrida (**BM25 invertido + LSA 64D + Char Subword + RRF/MMR**), ponte bilíngue PT↔EN macroeconômica, modos **Tutor / Comparador / Verificador**, **Gerador de Flashcards por Capítulo**, upload `.pdf/.md/.txt` e **Auditoria de Privacidade** (0 bytes externos offline/Ollama). | Python, FastAPI, Scikit-Learn, PyPDF, React, TypeScript, Tailwind, Recharts |
| ⭐ [**app-rag-artigos-cientificos-pdf**](./app-rag-artigos-cientificos-pdf/) | Aplicação web *full-stack* (**PaperRAG Studio**) para RAG científico híbrido e didático sobre PDFs acadêmicos: limpeza NFKC de ligaduras LaTeX, **Okapi BM25 + Vetores Densos LSA/Subword + RRF + MMR**, ponte bilíngue PT↔EN, guardrail anti-alucinação, Laboratório de Chunking com visualização de *overlap*, Raio-X do Retrieval/Prompt e suporte offline + Ollama/OpenAI/Gemini. | Python, FastAPI, Scikit-Learn, PyPDF, React, TypeScript, Tailwind, Recharts |

---

## 🧠 Competências Principais

### Análise Estatística, Econometria & Inferência Causal
- **Experimentação & Testes de Hipóteses**: A/B Testing, Diagnóstico de SRM, Power & MDE, Bootstrap, Mann-Whitney U, Welch t-test, Qui-Quadrado, Log-Rank.
- **Inferência Causal**: Difference-in-Differences (TWFE, Callaway & Sant'Anna), Variáveis Instrumentais (IV / 2SLS), RDD, Synthetic Control, Propensity Score Matching (PSM), CausalImpact / BSTS, SEM, Painel Dinâmico (System GMM) e Econometria Espacial (SAR, SEM, SDM, Moran).
- **Séries Temporais**: ARIMA, SARIMA, SARIMAX, Holt-Winters, VAR, VECM (Cointegração), Prophet, LSTM e *Foundation Models* (Chronos, TimeGPT, TimesFM).
- **Análise de Sobrevivência**: Kaplan-Meier, Nelson-Aalen, Regressão de Riscos Proporcionais de Cox e XGBoost Survival (`survival:cox`).

### Machine Learning & Engenharia de Dados
- **Modelagem Supervisionada & Não-Supervisionada**: LightGBM, XGBoost, CatBoost, Random Forest, Regressão Linear/Ridge/Lasso/Logística, K-Means, GMM, DBSCAN, PCA, UMAP, Densidade KNN e otimização de *thresholds* por impacto financeiro.
- **Boas Práticas & Interpretabilidade**: Pipelines sem *Data Leakage*, Validação Cruzada Estratificada, PR-AUC / ROC-AUC / C-Index, Feature Importance, SHAP, MLflow.
- **Big Data, APIs & LLMs**: PySpark (AQE, Salting, Broadcast, MLlib), Databricks, FastAPI, Pydantic v2, DuckDB, Polars, RAG e LangChain.

---

## 🛠️ Stack Tecnológico

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=postgresql&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Stata](https://img.shields.io/badge/Stata-1A4B8C?style=for-the-badge)

| Ecossistema | Principais Bibliotecas e Ferramentas |
| :--- | :--- |
| **Python — Dados & Big Data** | `pandas`, `numpy`, `pyspark`, `polars`, `duckdb`, `pyarrow` |
| **Python — ML, Econometria & Sobrevivência** | `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `statsmodels`, `linearmodels`, `pydynpd`, `scipy`, `lifelines`, `tensorflow`, `torch`, `mlflow`, `chronos-forecasting` |
| **Python — Visualização, APIs & LLMs** | `matplotlib`, `seaborn`, `plotly`, `fastapi`, `pydantic`, `uvicorn`, `streamlit`, `langchain`, `huggingface_hub` |
| **R — Econometria, Séries & Relatórios** | `tidyverse`, `data.table`, `fixest`, `did`, `CausalImpact`, `bsts`, `vars`, `urca`, `forecast`, `fable`, `tidymodels`, `survival`, `ggplot2`, `shiny`, `quarto` |
| **Ambientes & Versionamento** | Git, GitHub, Databricks, Jupyter, VS Code, Positron, RStudio, MLflow |

---

## 📬 Contato

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/silvio-paula/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/silviopaula)
[![Portfólio Web](https://img.shields.io/badge/Site_Pessoal-2C5BB4?style=flat&logo=googlechrome&logoColor=white)](https://silviopaula.github.io/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:silvio.rosa.paula@gmail.com)

---

## 📄 Licença

Este repositório está licenciado sob a [Licença MIT](./LICENSE).
