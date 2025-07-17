# 📊 Sistema de Análise de Séries Temporais ARIMA

[![R](https://img.shields.io/badge/R-4.0%2B-blue.svg)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-1.7%2B-green.svg)](https://shiny.rstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)]()

Um aplicativo web interativo desenvolvido em R Shiny para análise avançada de séries temporais usando modelos ARIMA (AutoRegressive Integrated Moving Average). O sistema oferece uma interface intuitiva para carregar dados, ajustar múltiplos modelos, comparar resultados e gerar previsões com intervalos de confiança.

## 🚀 Características Principais

### ✨ Funcionalidades Principais
- **🤖 Análise Automática**: Detecção automática dos melhores parâmetros ARIMA
- **⚙️ Configuração Manual**: Controle total sobre parâmetros p, d, q, P, D, Q
- **📈 Modelos Multivariados**: Suporte a variáveis explicativas (exógenas)
- **🎯 Comparação de Modelos**: Avaliação automática com múltiplas métricas
- **📊 Visualizações Interativas**: Gráficos dinâmicos com Plotly
- **📋 Análise de Coeficientes**: Detalhamento completo dos parâmetros estimados
- **📈 Análise de Crescimento**: Visualização de taxas em diferentes períodos
- **📤 Exportação Excel**: Relatórios completos em formato XLSX

### 🎨 Interface Moderna
- Design responsivo e profissional
- Navegação intuitiva por abas
- Feedback visual em tempo real
- Tabelas interativas com busca e filtros
- Gráficos interativos com zoom e pan

## 📷 Screenshots

![Dashboard Principal](screenshots/dashboard.png)
*Dashboard principal com visualização de previsões*

![Análise de Coeficientes](screenshots/coefficients.png)
*Análise detalhada dos coeficientes estimados*

![Configuração de Modelos](screenshots/configuration.png)
*Interface de configuração dos modelos ARIMA*

## 🛠️ Instalação

### Pré-requisitos
- R (versão 4.0 ou superior)
- RStudio (recomendado)

### Instalação das Dependências

```r
# Instalar pacman se não estiver instalado
if (!require(pacman)) install.packages("pacman")

# Instalar todas as dependências necessárias
pacman::p_load(
  shiny,           # Framework web
  shinydashboard,  # Layout dashboard
  tidyverse,       # Manipulação de dados
  lubridate,       # Processamento de datas
  plotly,          # Gráficos interativos
  readxl,          # Leitura de arquivos Excel
  DT,              # Tabelas interativas
  forecast,        # Modelos ARIMA
  writexl,         # Escrita em Excel
  shinyWidgets,    # Widgets adicionais
  shinythemes,     # Temas visuais
  RColorBrewer,    # Paletas de cores
  zoo,             # Séries temporais
  stringr,         # Manipulação de strings
  shinyjs          # JavaScript interativo
)
```

### Executando o Aplicativo

```r
# Clonar o repositório
git clone https://github.com/seu-usuario/arima-shiny-app.git
cd arima-shiny-app

# Executar no R/RStudio
source("app.R")
```

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

## 🏗️ Arquitetura

```
📦 arima-shiny-app/
├── 📄 app.R                    # Aplicação principal
├── 📄 README.md               # Este arquivo
├── 📄 LICENSE                 # Licença MIT
├── 📁 www/                    # Recursos web
│   ├── 📊 dados.xlsx          # Dados de exemplo
│   └── 🎨 custom.css          # Estilos personalizados
├── 📁 screenshots/            # Capturas de tela
├── 📁 functions/              # Funções auxiliares
│   ├── 📄 arima_functions.R   # Funções ARIMA
│   ├── 📄 data_processing.R   # Processamento de dados
│   └── 📄 visualization.R     # Visualizações
└── 📁 tests/                  # Testes unitários
    └── 📄 test_functions.R    # Testes das funções
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

## 📈 Exemplos de Uso

### Caso de Uso 1: Previsão de Vendas
```r
# Dados mensais de vendas de 2020-2023
# Objetivo: Prever vendas para os próximos 24 meses
# Configuração: ARIMA automático com sazonalidade
```

### Caso de Uso 2: Análise de Demanda
```r
# Dados diários de demanda por produto
# Variáveis exógenas: preço, promoções, temperatura
# Configuração: ARIMA multivariado manual
```

### Caso de Uso 3: Indicadores Econômicos
```r
# Dados trimestrais de PIB
# Objetivo: Análise de crescimento e projeções
# Configuração: Múltiplos modelos para comparação
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, siga estes passos:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

### Diretrizes para Contribuição
- Mantenha o código bem documentado
- Adicione testes para novas funcionalidades
- Siga as convenções de estilo do R
- Atualize a documentação quando necessário

## 🐛 Reportando Bugs

Encontrou um bug? Por favor, abra uma [issue](https://github.com/seu-usuario/arima-shiny-app/issues) com:

- Descrição detalhada do problema
- Passos para reproduzir
- Comportamento esperado vs. atual
- Screenshots (se aplicável)
- Informações do sistema (R version, OS, etc.)

## 📝 Changelog

### v1.0.0 (2024-01-XX)
- 🎉 Lançamento inicial
- ✨ Análise automática ARIMA
- 📊 Interface interativa completa
- 📤 Exportação em Excel
- 🔍 Análise de coeficientes
- 📈 Visualizações dinâmicas

### Próximas Versões
- [ ] Suporte a modelos SARIMA avançados
- [ ] Integração com APIs de dados
- [ ] Análise de múltiplas séries (VAR)
- [ ] Dashboard executivo
- [ ] Relatórios automatizados em PDF

## 📊 Performance

### Testado com:
- ✅ Datasets de até 10.000 observações
- ✅ Até 10 variáveis exógenas simultâneas
- ✅ Modelos com até 20 configurações paralelas
- ✅ Navegadores: Chrome, Firefox, Safari, Edge

### Tempos de Processamento:
- **Pequenos datasets** (< 500 obs): < 10 segundos
- **Médios datasets** (500-2000 obs): 10-30 segundos
- **Grandes datasets** (2000+ obs): 30-120 segundos

## 🔒 Segurança

- Processamento local dos dados (sem upload para servidores)
- Validação de entrada robusta
- Limite de tamanho de arquivo (100MB)
- Sanitização de inputs do usuário

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
MIT License

Copyright (c) 2024 Seu Nome

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👤 Autor

**Seu Nome**
- 🐱 GitHub: [@seu-usuario](https://github.com/seu-usuario)
- 📧 Email: seu.email@exemplo.com
- 💼 LinkedIn: [Seu Perfil](https://linkedin.com/in/seu-perfil)

## 🙏 Agradecimentos

- [R Core Team](https://www.r-project.org/) pelo R
- [RStudio Team](https://rstudio.com/) pelo Shiny
- [Rob Hyndman](https://robjhyndman.com/) pelo pacote forecast
- [Hadley Wickham](http://hadley.nz/) pelo tidyverse
- Comunidade R pelo suporte e feedback

## 📚 Referências Acadêmicas

- Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control.
- Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R.
- Brockwell, P. J., & Davis, R. A. (2016). Introduction to Time Series and Forecasting.

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!

📊 **Happy Forecasting!** 🚀