# 📈 TimeSeries Studio Pro — Suite Unificada de Análise e Previsão de Séries Temporais

Aplicação web completa para diagnóstico estatístico, tratamento de anomalias, edição interativa de cenários e previsão multivariada/univariada de séries temporais.

Construída com arquitetura desacoplada:
- **Backend 100% em Python (`FastAPI`)**: Motores estatísticos, econométricos e de Machine Learning (`statsmodels`, `scikit-learn`, `scipy`, `pandas`, `numpy`).
- **Frontend Web Responsivo (`HTML5` + `CSS3` + `Vanilla JS` + `Plotly.js`)**: Interface Single-Page Application (SPA) moderna com suporte a **Tema Escuro/Claro**, sem dependências de frameworks Python no front (zero Streamlit/Dash/Shiny).

---

## 🗂️ Estrutura do Projeto

```text
app_time_series/
├── backend/
│   ├── __init__.py
│   ├── main.py             # API REST FastAPI, gerenciamento de sessão e exportação Excel
│   ├── analytics.py        # Outliers (6+7 métodos), Sazonalidade, Controle XmR, STL e Visões YoY/MoM
│   └── forecasting.py      # Motores ARIMA/SARIMAX (Uni & Multivariado) e AutoTS (7 modelos)
├── frontend/
│   ├── index.html          # Interface Single-Page Application (7 módulos integrados)
│   ├── styles.css          # Design System responsivo (Dark & Light Mode)
│   └── app.js              # Controlador SPA, gráficos interativos Plotly e eventos
├── data/
│   └── ei_total_(PA).xlsx  # Base histórica e de variáveis exógenas para demonstração imediata
├── iniciar_app.bat         # Inicializador automático para Windows (encerra sessões antigas e abre o app)
├── run.py                  # Script de inicialização do servidor Uvicorn (porta 8055)
├── requirements.txt        # Dependências Python do projeto
├── .gitignore              # Regras de exclusão para controle de versão Git
└── README.md               # Documentação técnica e guia de uso
```

*(Nota: Os antigos aplicativos separados em R/Shiny foram movidos para a pasta `arquivos_para_excluir/`, que já está listada no `.gitignore` e pode ser deletada com segurança).*

---

## 🚀 Como Instalar e Executar

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.10+** instalado. Para instalar as dependências (caso ainda não estejam no seu ambiente):

```bash
pip install -r requirements.txt
```

### 2. Inicialização Rápida (Windows)
Dê um duplo clique no arquivo:
```text
iniciar_app.bat
```
O script irá automaticamente:
1. Verificar e encerrar qualquer sessão anterior que esteja ocupando a porta;
2. Iniciar o servidor FastAPI em `http://127.0.0.1:8055`;
3. Abrir o navegador diretamente na interface do **TimeSeries Studio Pro**.

### 3. Inicialização Manual via Terminal
```bash
python run.py
```
Acesse no navegador: **http://127.0.0.1:8055**

---

## 🧩 Guia Completo dos Módulos

### 1. 📁 Central de Dados & Configuração
- **Upload Drag-and-Drop**: Suporta planilhas Excel (`.xlsx`, `.xls`) com múltiplas abas e arquivos `.csv` com detecção automática de delimitador (`,` ou `;`).
- **Indexação Temporal Inteligente**: Detecta automaticamente colunas candidatas a data e converte formatos `YMD`, `DMY`, `MDY`, `YDM` ou datas seriais do Excel, padronizando a série temporal para todos os módulos.
- **Diagnóstico de Variáveis**: Exibe contagem de observações válidas, valores ausentes (`NA`), média, mínimo, máximo e cobertura temporal de cada coluna numérica.

---

### 2. ⚠️ Tratamento Avançado de Outliers
Módulo dedicado à detecção e correção de valores discrepantes com preservação da estrutura temporal:

- **6 Algoritmos de Identificação**:
  | Método | Descrição Técnica |
  | :--- | :--- |
  | **Z-Score** | Identifica pontos cujo desvio padronizado $|z|$ excede o limiar configurável (padrão: `3.0`). |
  | **IQR (Box Plot)** | Identifica valores abaixo de $Q_1 - k \cdot \text{IQR}$ ou acima de $Q_3 + k \cdot \text{IQR}$ (padrão: $k = 1.5$). |
  | **Distância de Mahalanobis** | Abordagem **multivariada** considerando a matriz de covariância de todas as variáveis numéricas ($\chi^2_{0.95}$). |
  | **Isolation Forest** | Algoritmo de árvores de isolamento (`scikit-learn`) com limiar ajustável sobre o score de anomalia. |
  | **LOF (Local Outlier Factor)** | Mede o desvio de densidade local de cada observação em relação aos seus $k$ vizinhos mais próximos. |
  | **One-Class SVM** | Máquina de vetores de suporte não-supervisionada com kernel RBF e parâmetro $\nu$ ajustável. |

- **7 Métodos de Correção**:
  - **Interpolação Linear**, **Média Móvel Centralizada** (com janela ajustável), **Média**, **Mediana**, **Winsorização (P5/P95)**, **Auto ARIMA (Kalman Filter via SARIMAX)** e **Decomposição Sazonal STL (substituição pela tendência)**.
- **Recursos de Produtividade**:
  - **Desfazer (`Ctrl+Z`)** em múltiplos níveis;
  - Comparativo completo de estatísticas (*Média, Mediana, Desvio Padrão, Mínimo, Máximo, Skewness, Kurtosis e Variação %*);
  - Botão **"Salvar Série Tratada no Workspace"** (disponibiliza a coluna `<variavel>_tratado` imediatamente para os módulos de previsão ARIMA e AutoTS);
  - Exportação `.xlsx` contendo os dados tratados e aba de **Metadados** da operação.

---

### 3. 🌊 Sazonalidade, Participação Mensal & Controle XmR
Combina três ferramentas de diagnóstico estrutural e controle estatístico de processo (CEP):

1. **Participação Mensal e Banda de Limites ($\pm 1 \text{ DP}$)**:
   - Calcula o peso percentual de cada mês no acumulado anual ($P_{m,a} = \frac{Y_{m,a}}{\sum_{m=1}^{12} Y_{m,a}}$);
   - Permite filtrar o período histórico de referência (Ano Inicial e Ano Final) para construir a banda de controle ($\mu_m \pm \sigma_m$) e comparar o perfil sazonal de cada ano.
2. **Gráfico de Controle Estatístico XmR**:
   - **Gráfico $X$ (Valores Individuais)**: Limites superior e inferior dados por $\bar{X} \pm 2.66 \times \overline{MR}$;
   - **Gráfico $MR$ (Amplitude Móvel)**: Limite superior dado por $UCL_{MR} = 3.0 \times \overline{MR}$ ($MR_t = |X_t - X_{t-1}|$), destacando violações em vermelho.
3. **Anomalias por Resíduos STL & Testes de Sazonalidade**:
   - Decompõe a série via **STL Robusto** ($Y_t = T_t + S_t + R_t$), calcula a **Força Sazonal** ($F_S = \max(0, 1 - \frac{\text{Var}(R)}{\text{Var}(S+R)})$), executa testes **ANOVA ($F$)** e **Kruskal-Wallis ($H$)** entre os meses e lista todas as observações cujo resíduo $|R_t|$ ultrapassa $\pm k\sigma$.

---

### 4. ✏️ Editor Avançado de Séries Temporais
Ferramenta visual para ajuste fino de séries históricas ou construção de cenários *what-if*:
- **Seleção Direta no Gráfico**: Clique em um ponto individual ou utilize a ferramenta **Laço (`Lasso`) / Caixa (`Box Select`)** no gráfico Plotly para selecionar múltiplos períodos simultaneamente.
- **3 Modos de Modificação**:
  - **Definir Novo Valor Exato**;
  - **Acréscimo / Decréscimo Absoluto ($+/-$)**;
  - **Variação Percentual ($\%$)**.
- **Histórico Auditável & `Ctrl+Z`**: Pilha completa de *Undo* e tabela de Log registrando índice, data, valor original, valor anterior, valor ajustado e tipo de operação.
- **Impacto Instantâneo**: Atualiza em tempo real os gráficos de taxas de crescimento e acréscimo Mês/Mês (MoM) e Mês/Ano (YoY), além das visões agregadas Mensal, Trimestral, Semestral e Anual.

---

### 5. 📊 Previsão ARIMA / SARIMAX (Univariado & Multivariado)
Motor econométrico completo para projeções com ou sem variáveis explicativas:
- **Modelagem Simultânea Univariada e Multivariada (`ARIMAX` / `SARIMAX`)**: Quando variáveis exógenas são selecionadas, o motor ajusta tanto as versões univariadas quanto as multivariadas para comparação direta de ganho preditivo.
- **Diagnóstico Automático de Colinearidade**: Verifica a matriz de correlação das variáveis exógenas ($|r| > 0.99$) antes da estimação.
- **Modo Automático vs. Manual**:
  - *Automático*: Seleciona a ordem base ótima via critério de informação (AIC) e teste ADF de estacionariedade, gerando $N$ variações vizinhas $(p,d,q)(P,D,Q)_s$.
  - *Manual*: Permite especificar diretamente os parâmetros não-sazonais $(p,d,q)$ e sazonais $(P,D,Q)_s$.
- **Backtesting (Treino / Teste) & Dashboard de Recomendação**: Reserva os últimos $N$ períodos realizados para validação fora da amostra, calcula `RMSE`, `MAE`, `MAPE` e classifica automaticamente cada modelo (*Ótimo*, *Regular*, *Ruim*, *Reprojetar*) com base no score ponderado de desvios recentes.

---

### 6. 🤖 Previsão AutoTS Multi-Modelos Univariados
Executa um benchmark automatizado entre **7 algoritmos de séries temporais**:
1. **Prophet** (Tendência linear por partes com *changepoints* + harmônicos sazonais de Fourier)
2. **ETS** (Suavização Exponencial em Espaço de Estados com tendência amortecida)
3. **SARIMA** (Auto-SARIMAX sazonal)
4. **TBATS** (Transformação Box-Cox + Harmônicos Trigonométricos + ARMA)
5. **BATS** (Transformação Box-Cox + Suavização Exponencial Sazonal)
6. **STLM** (Decomposição STL + SARIMAX na série dessazonalizada)
7. **Bagged (Ensemble)** (Combinação por média dos modelos ativos)

- Gera **intervalos de confiança de 95% (`mean`, `lower`, `upper`)** para todos os modelos, ranking de erros e o **Dashboard de Análise de Desvios AutoTS** com pesos configuráveis para meses recentes e desvio máximo.

---

### 7. 📈 Análise Multirresolução de Taxas de Crescimento
Consolida qualquer variável da base de dados ou qualquer série projetada pelos módulos **ARIMA** e **AutoTS** em quatro recortes temporais:
- **Mensal (`Jan` a `Dez`)**
- **Trimestral (`Q1` a `Q4`)**
- **Semestral (`S1` e `S2`)**
- **Anual (`Valor_Anual`)**

Oferece gráficos interativos (Combinado Barras + Linha, Linhas ou Barras Coloridas por sinal $+/-$), filtro dinâmico de anos e matrizes completas de **Valores**, **Taxas de Crescimento YoY (%)** e **Variações Absolutas YoY**, com exportação direta para Excel (`.xlsx`) em múltiplas abas.

---

## 🔌 Principais Endpoints da API REST

Com o servidor rodando, a documentação interativa OpenAPI (Swagger UI) também fica disponível em `http://127.0.0.1:8055/docs`:

- `GET  /api/dataset/info` — Metadados, estatísticas e preview do dataset ativo
- `POST /api/dataset/upload` — Upload de arquivo `.xlsx`, `.xls` ou `.csv`
- `POST /api/dataset/configure` — Seleção de aba, coluna de data e formato
- `POST /api/outliers/identify` — Detecção de outliers (6 métodos)
- `POST /api/outliers/correct` — Correção de outliers (7 métodos) + Undo/Commit/Export
- `POST /api/seasonality/analyze` — Participação sazonal, limites, gráfico XmR e anomalias STL
- `POST /api/editor/init` & `/modify` — Edição interativa de pontos, Undo e Commit
- `POST /api/forecast/arima` — Execução de modelos ARIMA/SARIMAX univariados e multivariados
- `POST /api/forecast/autots` — Execução do benchmark AutoTS (7 modelos)
- `POST /api/growth/analyze` — Matrizes e séries de crescimento Mensal, Trimestral, Semestral e Anual
