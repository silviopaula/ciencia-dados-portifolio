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
| [**Causal_impact_app**](./Causal_impact_app/) | Aplicação interativa para estimação contrafactual de impacto causal em séries temporais (*Bayesian Structural Time Series*). | Python / R, CausalImpact, BSTS |
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
| [**Comercio_exterior_spark_análise**](./Comercio_exterior_spark_análise/) | Processamento e análise exploratória de dados massivos de comércio exterior brasileiro com Apache Spark. | PySpark, Big Data |
| [**projeto-plataforma-dados**](./projeto-plataforma-dados/) | Arquitetura e implementação de plataforma de dados end-to-end. | Engenharia de Dados, Pipelines |

---

### 4. 📈 Séries Temporais, Macroeconomia & *Foundation Models*

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| ⭐ [**Time_Series_Forecast_Chronos_Amazon**](./Time_Series_Forecast_Chronos_Amazon/) | Benchmark *Zero-Shot* dos *Foundation Models* Amazon Chronos (Bolt & T5) vs. Holt-Winters e Seasonal Naive sobre 21 anos de consumo elétrico (EPE), projeção de 36 meses (P10–P90) e app Streamlit. | Python, Chronos 2.x, PyTorch, Statsmodels, Streamlit |
| [**LSTM_forecast_Keras_Tensor_Flow**](./LSTM_forecast_Keras_Tensor_Flow/) | Modelagem preditiva de séries temporais com redes neurais recorrentes LSTM. | Python, TensorFlow, Keras |
| [**VECM_cointegracao_forecast**](./VECM_cointegracao_forecast/) | Projeção da taxa de desocupação utilizando cointegração e Modelo Vetorial de Correção de Erros (VECM). | R, Econometria, Cointegração, VECM |
| [**SEM-energy-forecasting**](./SEM-energy-forecasting/) | Modelagem de equações estruturais (SEM) aplicada à previsão de demanda de energia. | R, SEM, Econometria |
| [**app_forecasts**](./app_forecasts/) / [**app_forecast_ARIMA**](./app_forecast_ARIMA/) | Aplicações interativas para previsão de séries temporais (ARIMA, SARIMA e modelos estatísticos). | Python, Streamlit, Statsmodels |
| [**app_time_series**](./app_time_series/) / [**app_teste_estatistico_ts**](./app_teste_estatistico_ts/) | Ferramentas interativas de exploração, decomposição e testes de estacionariedade/raiz unitária em séries temporais. | Python, Streamlit, SciPy, Statsmodels |
| [**Macroeconomia_app**](./Macroeconomia_app/) | Aplicação analítica para acompanhamento e modelagem de indicadores macroeconômicos. | Python, Macroeconomia |

---

### 5. 🧠 NLP, LLMs & RAG

| Projeto | Descrição | Técnicas & Stack |
| :--- | :--- | :--- |
| [**Analise_sentimento_COPOM**](./Analise_sentimento_COPOM/) | Análise de sentimento e mineração de texto sobre as atas do Comitê de Política Monetária (COPOM / Banco Central). | Python, NLP, Dicionário Loughran-McDonald |
| [**NLP_ecommerce_Brasil**](./NLP_ecommerce_Brasil/) | Classificação e análise de avaliações de clientes no e-commerce brasileiro. | Python, NLP, Machine Learning |
| [**NLP_Spark_classificação**](./NLP_Spark_classificação/) | Pipeline distribuído de classificação de textos com Apache Spark MLlib. | PySpark, MLlib, NLP |
| [**app_macro-rag-assistant**](./app_macro-rag-assistant/) | Assistente de pesquisa macroeconômica baseado em *Retrieval-Augmented Generation* (RAG). | Python, LLMs, LangChain, RAG |
| [**app_rag_papers**](./app_rag_papers/) | Aplicação RAG para consulta semântica e síntese de artigos científicos em PDF. | Python, LLMs, RAG, Embeddings |

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
| **Python — ML, Econometria & Sobrevivência** | `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `statsmodels`, `linearmodels`, `scipy`, `lifelines`, `tensorflow`, `torch`, `mlflow`, `chronos-forecasting` |
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
