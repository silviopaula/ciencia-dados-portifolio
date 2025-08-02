# 📊 Sistema de Análise de Séries Temporais ARIMA

[![R](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-1.7%2B-green.svg)](https://shiny.rstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)]()

Um aplicativo web interativo desenvolvido em R Shiny para análise avançada de séries temporais usando modelos ARIMA (AutoRegressive Integrated Moving Average). O sistema oferece uma interface intuitiva para carregar dados, ajustar múltiplos modelos, comparar resultados e gerar previsões com intervalos de confiança.

## 🚀 Características Principais

### ✨ Funcionalidades Principais
- **Análise Automática**: Detecção automática dos melhores parâmetros ARIMA
- **Configuração Manual**: Controle total sobre parâmetros p, d, q, P, D, Q
- **Modelos Multivariados**: Suporte a variáveis explicativas (exógenas)
- **Comparação de Modelos**: Avaliação automática com múltiplas métricas
- **Visualizações Interativas**: Gráficos dinâmicos com Plotly
- **Análise de Coeficientes**: Detalhamento completo dos parâmetros estimados
- **Análise de Crescimento**: Visualização de taxas em diferentes períodos
- **Exportação Excel**: Relatórios completos em formato XLSX

### 🎨 Interface Moderna
- Design responsivo
- Navegação intuitiva por abas
- Feedback visual em tempo real
- Tabelas interativas com busca e filtros
- Gráficos interativos com zoom e pan

## 📋 Como Usar

### 1. 📤 Upload de Dados
- Acesse a aba **"Upload e Configuração"**
- Faça upload de arquivo Excel (.xlsx) ou CSV
- Selecione a aba do Excel (se aplicável)
- Escolha a coluna que contém as datas
- Clique em **"Processar Dados"**

### 2. ⚙️ Configuração do Modelo
- Vá para a aba **"Modelos ARIMA"**
- Selecione a variável para previsão
- (Opcional) Adicione variáveis explicativas
- Configure o período de análise
- Escolha entre modo automático ou manual
- Clique em **"Executar Análise ARIMA"**

### 3. 📊 Análise dos Resultados
- Visualize as métricas de erro (RMSE, MAPE, MAE)
- Examine os gráficos de previsão interativos
- Exporte as projeções em Excel

### 4. 🔍 Análise Detalhada
- **Coeficientes**: Veja parâmetros estimados e significância
- **Crescimento**: Analise taxas de crescimento por período
- **Exportação**: Baixe relatórios completos

## 📊 Estrutura dos Dados

### Formato Requerido

Seus dados devem ter a seguinte estrutura:

| Data       | Variavel_1 | Variavel_2 | Variavel_3 |
|------------|------------|------------|------------|
| 2020-01-01 | 1500.50    | 2300.75    | 850.25     |
| 2020-02-01 | 1620.30    | 2410.80    | 920.15     |
| 2020-03-01 | 1580.90    | 2350.60    | 890.45     |

### Requisitos:
- **Coluna de Data**: Formato reconhecível (YYYY-MM-DD, DD/MM/YYYY, etc.)
- **Variáveis Numéricas**: Valores numéricos para análise
- **Periodicidade Regular**: Dados mensais são ideais
- **Minimum de Observações**: Pelo menos 24 pontos para análise robusta
- **Sem Lacunas Grandes**: Evite intervalos longos sem dados

### Formatos de Data Suportados:
- `2023-01-15` (ISO 8601)
- `15/01/2023` (DD/MM/YYYY)
- `01/15/2023` (MM/DD/YYYY)
- `15-Jan-2023` (DD-MMM-YYYY)

## 🔧 Funcionalidades Técnicas

### Modelos ARIMA
- **Automático**: Utiliza algoritmo Hyndman-Khandakar
- **Manual**: Controle total sobre parâmetros (p,d,q,P,D,Q)
- **Sazonalidade**: Detecção e modelagem automática
- **Validação**: Divisão automática treino/teste

### Métricas de Avaliação
- **RMSE**: Raiz do Erro Quadrático Médio
- **MAPE**: Erro Percentual Absoluto Médio
- **MAE**: Erro Absoluto Médio
- **AIC/BIC**: Critérios de informação

### Análise de Coeficientes
- Valores estimados e erros padrão
- Testes de significância (t-test)
- Intervalos de confiança
- Diagnósticos de modelo
```

## 🛠️ Tecnologias Utilizadas

### Backend
- **R**: Linguagem de programação estatística
- **Shiny**: Framework web reativo
- **forecast**: Pacote para modelos de séries temporais
- **tidyverse**: Ecossistema de manipulação de dados

### Frontend
- **shinydashboard**: Layout e componentes UI
- **Plotly**: Visualizações interativas
- **DT**: Tabelas dinâmicas
- **shinyWidgets**: Componentes UI avançados

### Algoritmos
- **Auto.ARIMA**: Seleção automática de modelos
- **Box-Jenkins**: Metodologia clássica ARIMA
- **Hyndman-Khandakar**: Algoritmo de busca eficiente
- **Maximum Likelihood**: Estimação de parâmetros
```

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!

📊 **Happy Forecasting!** 🚀
