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
| ⭐ [**AB_testing**](./AB_testing/) | Avaliação estatística rigorosa do posicionamento de *gates* na retenção de 90.189 jogadores (*Cookie Cats*), com diagnóstico SRM, Bootstrap (10k), análise de *compliance* e *Cohen's h*. | Python, SciPy, Statsmodels, Bootstrap |
| ⭐ [**Power_MDE**](./Power_MDE/) | Guia didático sobre a "armadilha do p-valor": cálculo de Poder Estatístico (*Power*) e *Minimum Detectable Effect* (MDE) em painel *Difference-in-Differences* (TWFE). | Python, Statsmodels, SciPy, DiD TWFE |
| ⭐ [**MQO_vs_IV**](./MQO_vs_IV/) | Demonstração prática do viés de endogeneidade na elasticidade-preço, correção via Variáveis Instrumentais (IV / 2SLS) e o paradoxo entre RMSE histórico e decisão de *Pricing*. | Python, Linearmodels (`IV2SLS`), Statsmodels |
| ⭐ [**Painel_Dinamico_GMM**](./Painel_Dinamico_GMM/) | Guia completo de Painel Dinâmico (*Two-Step System-GMM* + Windmeijer): simulação Monte Carlo provando o Viés de Nickell, replicação do artigo na RBE (27 UFs, 1996–2015), proliferação de instrumentos, Efeitos de Longo Prazo (Método Delta) e *Specification Curve*. | Python, `pydynpd`, Linearmodels, SciPy, Stata (`xtabond2`) |
| ⭐ [**Causal_impact_app**](./Causal_impact_app/) | Aplicação web *full-stack* (**CausalImpact Studio**) para inferência causal contrafactual em séries temporais via modelos estruturais em espaço de estados (**BSTS / Kalman + Simulação Posterior Monte Carlo**), com os 3 painéis clássicos de Brodersen et al., **Teste Placebo no Tempo**, auditoria *No-Spillover* das covariáveis, 3 cenários demo (incluindo placebo nulo), upload CSV/XLSX e exportação Excel multi-abas. | Python, FastAPI, Statsmodels (`UnobservedComponents`), React, TypeScript, Tailwind, Recharts |
| [**Databricks_spark_causal_impact_Lei_12034-2009**](./Databricks_spark_causal_impact_Lei_12034-2009/) | Avaliação de impacto causal da Lei 12.034/2009 processada em larga escala com Apache Spark / Databricks. | PySpark, Databricks, CausalImpact |
| [**Marketing_causal_inference**](./Marketing_causal_inference/) | Inferência causal aplicada a campanhas de marketing e avaliação de incrementalidade. | Econometria, Inferência Causal |
| [**Marketing_causal_learning**](./Marketing_causal_learning/) | Aprendizado causal (*Causal ML / Uplift Modeling*) aplicado à otimização de ações de marketing. | Causal ML, Uplift, Python |

---

### 2. 🤖 Machine Learning, Análise de Sobrevivência & MLOps

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**Aplicando Survival Analysis**](./Aplicando%20Survival%20Analysis/) | Estudo completo de *Time-to-Churn* com censura à direita: *Swimmer Plot*, Kaplan-Meier, Log-Rank, Nelson-Aalen, Regressão de Cox (*Hazard Ratios*) e XGBoost `survival:cox`. | Python, Lifelines, XGBoost, Scikit-Learn |
| ⭐ [**Gestao_Estoques_Classif_Cluster**](./Gestao_Estoques_Classif_Cluster/) | Pipeline *end-to-end* sem *Data Leakage* sobre 198.917 SKUs combinando Clusterização (`PCA`, `K-Means`, `GMM`, `KNN`) + Classificação (`LightGBM`, `XGBoost`) e otimização de *threshold* para decisão de inventário. | Python, Scikit-Learn, LightGBM, XGBoost |
| [**aplicacao_mlflow**](./aplicacao_mlflow/) | Rastreamento de experimentos, versionamento de métricas, artefatos e registro de modelos com MLflow. | Python, MLflow, Scikit-Learn |

---

### 3. ⚡ Engenharia de Dados, Big Data (Spark) & APIs

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**API_python**](./API_python/) | API REST modular de alta performance para ingestão e consultas analíticas sobre 100.000 usuários em JSON aninhado (67 MB) com latência `< 2 ms` e suíte de testes automatizados. | Python, FastAPI, Pydantic v2, Uvicorn, Unittest |
| ⭐ [**Otimização_joins_spark**](./Otimização_joins_spark/) | Benchmark real no Apache Spark 4.0 (15M linhas, 95% *Data Skew*) comparando `SortMergeJoin`, `Repartition`, `Salting Manual`, `AQE Skew Join` e `Broadcast Hash Join` (20,7x *speedup*). | PySpark 4.0, Catalyst Explain, AQE, Salting |
| ⭐ [**Comercio_exterior_spark_análise**](./Comercio_exterior_spark_análise/) | Engenharia distribuída em **Apache Spark 4.0** sobre **36,5M de registros oficiais SECEX (`2010–Jun/2025`, US\$ 7,09 tri)** com auditoria *Star-Schema* (eliminação de *fan-out* de `10,7×`), `BroadcastHashJoin` (`2,07×` *speedup*), evolução do **HHI**, exposição EUA/China/Rússia, econometria da demanda de suco de laranja (`MQO`, `HAC`, `ARDL` e teste crítico `2SLS/IV`) e **Monte Carlo (10.000 cenários tarifários EUA: 10%, 25% e 50% + US\$ 415/t)**. | PySpark 4.0, AQE, Broadcast Join, Statsmodels, Monte Carlo, Parquet |
| [**projeto-plataforma-dados**](./projeto-plataforma-dados/) | Arquitetura e implementação de plataforma de dados end-to-end. | Engenharia de Dados, Pipelines |

---

### 4. 📈 Séries Temporais, Macroeconomia & *Foundation Models*

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**Time_Series_Forecast_Chronos_Amazon**](./Time_Series_Forecast_Chronos_Amazon/) | Benchmark *Zero-Shot* dos *Foundation Models* Amazon Chronos (Bolt & T5) vs. Holt-Winters e Seasonal Naive sobre 21 anos de consumo elétrico (EPE), projeção de 36 meses (P10–P90) e app Streamlit. | Python, Chronos 2.x, PyTorch, Statsmodels, Streamlit |
| ⭐ [**LSTM_forecast_Keras_Tensor_Flow**](./LSTM_forecast_Keras_Tensor_Flow/) | Jornada vertical de Deep Learning em séries temporais (**AutoKeras NAS → Keras Functional `ResidualLSTM` → TensorFlow `GradientTape` `ProbResidualAttentionLSTM`** com NLL Gaussiana, Atenção Temporal e bandas P10–P90) vs. Holt-Winters/SARIMAX (`MAPE 2,81%` em 24m OOS) e projeção 2025–2030. | Python, TensorFlow 2.20, Keras 3, AutoKeras 2, Statsmodels |
| ⭐ [**VECM_cointegracao_forecast**](./VECM_cointegracao_forecast/) | Cointegração de Johansen e Modelo Vetorial de Correção de Erros (VECM) para a taxa de desocupação brasileira (`IBC-Br`, `Selic`, `IAEmp`, `IIE-Br`), com Granger, IRF/FEVD, backtest 12m OOS (`MAPE 1,91%` vs. SARIMA `3,62%` e VAR-diff `23,12%`) e auditoria real Abr–Mai/2025. | Python, Statsmodels (`VECM`, `VAR`, `SARIMAX`) |
| ⭐ [**SEM-energy-forecasting**](./SEM-energy-forecasting/) | Sistema de 5 equações estruturais simultâneas (SEM/SUR) integrando PIB, Renda, Indústria, Comércio, Clima e Tarifas com decomposição causal (direto vs. indireto), backtest *out-of-sample* (MAPE `1,82%`) e cenários até 2034. | Python, Linearmodels (`SUR`), Statsmodels |
| ⭐ [**app_forecast_ARIMA**](./app_forecast_ARIMA/) | Aplicação web *full-stack* interativa e didática (*SARIMAX Forecast Studio*) com fluxo Box-Jenkins guiado (ADF/KPSS, ACF/PACF), Auto/Manual SARIMA & SARIMAX com variáveis exógenas, backtest vs. Seasonal Naive, diagnóstico Ljung-Box e exportação Excel. | Python, FastAPI, Statsmodels, React, TypeScript, Tailwind, Recharts |
| [**app_forecasts**](./app_forecasts/) | Aplicação interativa para previsão de séries temporais com múltiplos modelos estatísticos. | Python, Streamlit, Statsmodels |
| ⭐ [**app_teste_estatistico_ts**](./app_teste_estatistico_ts/) | Aplicação web *full-stack* (**TimeSeries Diagnostics Studio**) com bateria completa de **10 testes econométricos** (**ADF `n/c/ct`**, **Phillips-Perron Newey-West**, **KPSS**, **Zivot-Andrews** com quebra estrutural endógena, **Ljung-Box / Box-Pierce** com `model_df`, **Engle ARCH-LM**, **Breusch-Pagan Temporal**, **Anderson-Darling**, **Jarque-Bera** e **Shapiro-Wilk**), Veredito Cruzado 4-Quadrantes, Laboratório de 7 Transformações, **5 regimes demo + upload CSV/XLSX**, STL Robusta ($F_T, F_S$), SARIMA + Backtest OOS e Excel 5 abas. | Python, FastAPI, Statsmodels, SciPy, OpenPyXL, React, TypeScript, Tailwind, Recharts |
| [**app_time_series**](./app_time_series/) | Ferramenta interativa de exploração e decomposição de séries temporais. | Python, Streamlit, Statsmodels |
| ⭐ [**Macroeconomia_app**](./Macroeconomia_app/) | Aplicação web *full-stack* (**Observatório Macro Brasil Studio**) conectada às APIs oficiais do Banco Central (**SGS + Olinda Focus**) com cache colunar **DuckDB/Parquet** (`76` séries BCB + `37` tabelas = `43.109` registros) e **60 estudos macroeconométricos autorais** (`M01`–`M60`, 10 por pilar), incluindo 6 laboratórios interativos (**Dominância Fiscal VAR/Granger**, **Projeção DBGG ARIMA**, **Regra de Taylor HAC**, **Inércia Rolling AR(1)**, **SARIMA IPCA** e **Curva de Phillips**). | Python, FastAPI, DuckDB, Parquet, Statsmodels, React, TypeScript, Tailwind, Recharts |

---

### 5. 🧠 NLP, LLMs & RAG

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**Analise_sentimento_COPOM**](./Analise_sentimento_COPOM/) | NLP financeiro e de política monetária sobre **111 atas oficiais do COPOM (2012–2026, até a 282ª Reunião)**: Índice *Hawkish–Dovish* (`r = +0,44` com a decisão seguinte da Selic), sentimento *Loughran–McDonald*, 5 tópicos latentes (TF-IDF + NMF), validação *Walk-Forward* e comparação entre 4 gestões do BCB. | Python, NLP, TF-IDF/NMF, Scikit-Learn, Statsmodels |
| ⭐ [**NLP_ecommerce_Brasil**](./NLP_ecommerce_Brasil/) | NLP em PT-BR & Modelagem Multimodal (**Texto + Logística**) sobre **98.673 pedidos únicos da Olist** (`42.173` avaliações escritas) com split cronológico *Out-of-Time*, preservação de negações (`+2,03 p.p.` Macro-F1), benchmark de **10 modelos** (**XGBoost Multimodal** `Macro-F1 0,6873` / `ROC-AUC 0,9718` e **Calibrated LinearSVC** `Acc 88,65%` / `PR-AUC 0,8936`), *Slice Error Analysis*, ABSA + NMF + Grafo `NetworkX` e **Copiloto RAG de SAC**. | Python, NLP (PT-BR), TF-IDF, LightGBM, XGBoost, Scikit-Learn, NetworkX |
| ⭐ [**NLP_Spark_classificação**](./NLP_Spark_classificação/) | NLP & Classificação Distribuída em **Apache Spark 4.0.1 (`pyspark.ml`)** sobre **50.000 avaliações reais do Yelp (`1–5★`)** em Parquet (`zstd`): *ablation* de colisão de hash (`HashingTF` `50 → 16.384`, `+21,3 p.p.` ROC-AUC), split estratificado *Zero-Leakage*, preservação de negações, **`Word2Vec` 64D nativo distribuído** e benchmark de **10 pipelines MLlib** (**Campeão Híbrido `TF-IDF 16k + Word2Vec 64D + Meta`**: `ROC-AUC 0,9155`, `PR-AUC 0,8778`, `F1 0,8090` em `τ = 0,37`, `Brier 0,1128`), com calibração e *Slice Error Analysis*. | PySpark 4.0, MLlib (`HashingTF`, `IDF`, `Word2Vec`, `LR`, `SVC`, `NB`, `RF`, `GBT`), Parquet |
| ⭐ [**app_macro-rag-assistant**](./app_macro-rag-assistant/) | Aplicação web *full-stack* (**MacroRAG Tutor Studio**) para estudo e pesquisa em livros-texto e notas de Macroeconomia (*Foundations of Modern Macroeconomics*, Heijdra — **816 páginas** indexadas por **19 capítulos** com cache local comprimido): busca híbrida (**BM25 invertido + LSA 64D + Char Subword + RRF/MMR**), ponte bilíngue PT↔EN macroeconômica, modos **Tutor / Comparador / Verificador**, **Gerador de Flashcards por Capítulo**, upload `.pdf/.md/.txt` e **Auditoria de Privacidade** (0 bytes externos offline/Ollama). | Python, FastAPI, Scikit-Learn, PyPDF, React, TypeScript, Tailwind, Recharts |
| ⭐ [**app_rag_papers**](./app_rag_papers/) | Aplicação web *full-stack* (**PaperRAG Studio**) para RAG científico híbrido e didático sobre PDFs acadêmicos: limpeza NFKC de ligaduras LaTeX, **Okapi BM25 + Vetores Densos LSA/Subword + RRF + MMR**, ponte bilíngue PT↔EN, guardrail anti-alucinação, Laboratório de Chunking com visualização de *overlap*, Raio-X do Retrieval/Prompt e suporte offline + Ollama/OpenAI/Gemini. | Python, FastAPI, Scikit-Learn, PyPDF, React, TypeScript, Tailwind, Recharts |

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
