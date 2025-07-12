# ====================================================================
# DASHBOARD DE ANÁLISE DE SÉRIES TEMPORAIS 
# ====================================================================
# 
# Dashboard profissional para análise estatística completa de séries temporais
# 
# FUNCIONALIDADES PRINCIPAIS:
# ✓ Suporte a múltiplos formatos: Excel (.xlsx), CSV (vírgula/ponto e vírgula)
# ✓ Dados simulados para demonstração
# ✓ Testes de estacionariedade (ADF, Phillips-Perron, KPSS)
# ✓ Testes de autocorrelação (Box-Pierce, Ljung-Box)
# ✓ Testes de heterocedasticidade (ARCH)
# ✓ Testes de normalidade (Anderson-Darling)
# ✓ Visualizações interativas e responsivas
# ✓ Decomposição STL de séries temporais
# ✓ Previsão usando modelos ARIMA
# ✓ Exportação completa para Excel
# ✓ Interface moderna e intuitiva
#
# Autor: [Seu Nome]
# Data: [Data Atual]
# Versão: 2.0 Premium
# ====================================================================

# ====================================================================
# 1. CONFIGURAÇÃO E CARREGAMENTO DE PACOTES
# ====================================================================

# Instalar e carregar pacotes necessários
if(!require(pacman)) { install.packages("pacman") }
pacman::p_load(
  shiny,          # Framework web para R
  shinydashboard, # Interface de dashboard para Shiny
  readxl,         # Leitura de arquivos Excel
  readr,          # Leitura eficiente de arquivos CSV
  tidyverse,      # Conjunto de pacotes para manipulação de dados
  lubridate,      # Manipulação de datas
  plotly,         # Gráficos interativos
  forecast,       # Análise e previsão de séries temporais
  tseries,        # Testes estatísticos para séries temporais
  lmtest,         # Testes de diagnóstico para modelos lineares
  nortest,        # Testes de normalidade
  DT,             # Tabelas interativas
  writexl,        # Exportação para Excel
  shinycssloaders # Loading spinners
)

# ====================================================================
# 2. FUNÇÃO PARA GERAR DADOS SIMULADOS
# ====================================================================

generate_simulated_data <- function() {
  # Criar sequência de datas mensais de 2005 a 2025
  dates <- seq(from = as.Date("2005-01-01"), 
               to = as.Date("2025-12-01"), 
               by = "month")
  
  n <- length(dates)
  t <- 1:n  # Índice temporal
  
  # Componentes da série temporal
  # 1. Tendência crescente (drift positivo)
  trend <- 100 + 0.8 * t  # Crescimento de 0.8 unidades por mês
  
  # 2. Sazonalidade anual bem marcada
  # Pico no meio do ano (junho/julho) e vale no inverno (dezembro/janeiro)
  seasonal <- 25 * sin(2 * pi * t / 12) + 10 * cos(2 * pi * t / 12)
  
  # 3. Ciclo de longo prazo (5 anos)
  long_cycle <- 15 * sin(2 * pi * t / 60)
  
  # 4. Ruído aleatório
  set.seed(123)  # Para reprodutibilidade
  noise <- rnorm(n, mean = 0, sd = 8)
  
  # Combinar todos os componentes
  values <- trend + seasonal + long_cycle + noise
  
  # Garantir que não há valores negativos
  values <- pmax(values, 10)
  
  # Retornar dataframe
  data.frame(
    Date = dates,
    Value = round(values, 2)
  )
}

# ====================================================================
# 3. FUNÇÃO PARA DETECTAR SEPARADOR CSV
# ====================================================================

detect_csv_separator <- function(file_path, sample_lines = 5) {
  # Ler primeiras linhas do arquivo
  sample_text <- readLines(file_path, n = sample_lines)
  sample_text <- paste(sample_text, collapse = "\n")
  
  # Contar vírgulas e ponto e vírgulas
  comma_count <- str_count(sample_text, ",")
  semicolon_count <- str_count(sample_text, ";")
  
  # Retornar separador mais provável
  if (semicolon_count > comma_count) {
    return(";")
  } else {
    return(",")
  }
}

# ====================================================================
# 4. CSS CUSTOMIZADO PARA INTERFACE PREMIUM
# ====================================================================

custom_css <- "
  /* Importar Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  
  /* Variáveis CSS para cores consistentes */
  :root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #f093fb;
    --success-color: #4facfe;
    --warning-color: #43e97b;
    --danger-color: #fa709a;
    --dark-color: #2c3e50;
    --light-color: #ecf0f1;
    --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-accent: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --shadow-soft: 0 10px 25px rgba(0,0,0,0.1);
    --shadow-medium: 0 15px 35px rgba(0,0,0,0.15);
  }
  
  /* Estilos globais */
  body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: var(--dark-color);
  }
  
  /* Header customizado */
  .main-header {
    background: var(--gradient-primary) !important;
    box-shadow: var(--shadow-soft);
    border: none !important;
  }
  
  .main-header .navbar {
    background: transparent !important;
  }
  
  .main-header .navbar-custom-menu > .navbar-nav > li > .dropdown-menu {
    background: white;
    box-shadow: var(--shadow-medium);
    border: none;
    border-radius: 10px;
  }
  
  /* Sidebar premium */
  .main-sidebar {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
    box-shadow: var(--shadow-medium);
  }
  
  .sidebar-menu > li > a {
    color: white !important;
    transition: all 0.3s ease;
    border-radius: 8px;
    margin: 2px 10px;
  }
  
  .sidebar-menu > li > a:hover {
    background: rgba(255,255,255,0.1) !important;
    transform: translateX(5px);
  }
  
  /* Content wrapper moderno */
  .content-wrapper {
    background: transparent !important;
    padding: 20px;
  }
  
  /* Boxes com design premium */
  .box {
    border: none !important;
    border-radius: 15px !important;
    box-shadow: var(--shadow-soft) !important;
    background: white !important;
    overflow: hidden;
    transition: all 0.3s ease;
  }
  
  .box:hover {
    box-shadow: var(--shadow-medium) !important;
    transform: translateY(-2px);
  }
  
  .box-header {
    background: var(--gradient-primary) !important;
    color: white !important;
    border-radius: 15px 15px 0 0 !important;
    border: none !important;
    padding: 20px !important;
  }
  
  .box-header .box-title {
    font-weight: 600 !important;
    font-size: 18px !important;
  }
  
  .box-body {
    padding: 25px !important;
  }
  
  /* Info boxes estilizados */
  .info-box {
    border-radius: 15px !important;
    box-shadow: var(--shadow-soft) !important;
    border: none !important;
    transition: all 0.3s ease;
    overflow: hidden;
  }
  
  .info-box:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-medium) !important;
  }
  
  .info-box-icon {
    border-radius: 15px 0 0 15px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
  
  .info-box-content {
    padding: 15px !important;
  }
  
  .info-box-number {
    font-weight: 700 !important;
    font-size: 20px !important;
  }
  
  .info-box-text {
    font-weight: 500 !important;
    font-size: 14px !important;
  }
  
  /* Botões modernos */
  .btn {
    border-radius: 25px !important;
    font-weight: 500 !important;
    padding: 10px 25px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
  }
  
  .btn-primary {
    background: var(--gradient-primary) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
  }
  
  .btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
  }
  
  .btn-success {
    background: var(--gradient-success) !important;
    box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4) !important;
  }
  
  .btn-success:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6) !important;
  }
  
  .btn-info {
    background: var(--gradient-accent) !important;
    box-shadow: 0 5px 15px rgba(240, 147, 251, 0.4) !important;
  }
  
  .btn-info:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(240, 147, 251, 0.6) !important;
  }
  
  /* Inputs estilizados */
  .form-control {
    border-radius: 10px !important;
    border: 2px solid #e9ecef !important;
    padding: 12px 15px !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
  }
  
  .form-control:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    transform: translateY(-1px) !important;
  }
  
  /* File input customizado */
  .form-group input[type='file'] {
    padding: 15px !important;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    border: 2px dashed var(--primary-color) !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
  }
  
  .form-group input[type='file']:hover {
    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%) !important;
    transform: translateY(-1px) !important;
  }
  
  /* Radio buttons estilizados */
  .radio {
    margin: 10px 0 !important;
  }
  
  .radio label {
    font-weight: 500 !important;
    color: var(--dark-color) !important;
    padding-left: 25px !important;
  }
  
  /* Tabs modernas */
  .nav-tabs {
    border: none !important;
    background: white !important;
    border-radius: 15px !important;
    padding: 5px !important;
    box-shadow: var(--shadow-soft) !important;
    margin-bottom: 20px !important;
  }
  
  .nav-tabs > li > a {
    border: none !important;
    border-radius: 10px !important;
    margin: 0 2px !important;
    font-weight: 500 !important;
    color: var(--dark-color) !important;
    transition: all 0.3s ease !important;
  }
  
  .nav-tabs > li.active > a {
    background: var(--gradient-primary) !important;
    color: white !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
  }
  
  .nav-tabs > li > a:hover {
    background: rgba(102, 126, 234, 0.1) !important;
    transform: translateY(-1px) !important;
  }
  
  /* Tabelas estilizadas */
  .dataTables_wrapper {
    padding: 20px !important;
  }
  
  .table {
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-soft) !important;
  }
  
  .table thead th {
    background: var(--gradient-primary) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 15px !important;
  }
  
  .table tbody tr {
    transition: all 0.3s ease !important;
  }
  
  .table tbody tr:hover {
    background: rgba(102, 126, 234, 0.05) !important;
    transform: scale(1.01) !important;
  }
  
  /* Cards informativos */
  .info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
    margin: 10px 0;
    box-shadow: var(--shadow-soft);
    transition: all 0.3s ease;
  }
  
  .info-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-medium);
  }
  
  .info-card h5 {
    font-weight: 600;
    margin-bottom: 10px;
  }
  
  .info-card p, .info-card li {
    font-size: 13px;
    opacity: 0.9;
  }
  
  /* Loading spinners */
  .load-container {
    position: relative;
  }
  
  /* Animações suaves */
  .fade-in {
    animation: fadeIn 0.5s ease-in;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  /* Scrollbar customizada */
  ::-webkit-scrollbar {
    width: 8px;
  }
  
  ::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: var(--gradient-primary);
    border-radius: 10px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-color);
  }
  
  /* Responsive design melhorado */
  @media (max-width: 768px) {
    .content-wrapper {
      padding: 10px;
    }
    
    .box {
      margin: 10px 0;
    }
    
    .info-box {
      margin: 5px 0;
    }
  }
  
  /* Guias explicativos estilizados */
  .guide-box {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid var(--primary-color);
    border-radius: 0 10px 10px 0;
    padding: 20px;
    margin: 20px 0;
    box-shadow: var(--shadow-soft);
    transition: all 0.3s ease;
  }
  
  .guide-box:hover {
    transform: translateX(5px);
    box-shadow: var(--shadow-medium);
  }
  
  .guide-box h4 {
    color: var(--primary-color);
    font-weight: 600;
    margin-bottom: 15px;
  }
  
  .guide-box p {
    line-height: 1.6;
    margin-bottom: 10px;
  }
  
  .guide-box ul li {
    margin: 8px 0;
    line-height: 1.5;
  }
"

# ====================================================================
# 5. INTERFACE DO USUÁRIO (UI)
# ====================================================================

ui <- dashboardPage(
  # Cabeçalho do dashboard
  dashboardHeader(title = "📊 Análise de Séries Temporais Premium"),
  
  # Barra lateral com controles
  dashboardSidebar(
    # CSS customizado
    tags$head(
      tags$style(HTML(custom_css))
    ),
    
    # Opção para escolher fonte de dados
    div(class = "info-card",
        h5("🔧 Fonte de Dados"),
        radioButtons("data_source", NULL,
                     choices = list(
                       "📁 Carregar arquivo Excel (.xlsx)" = "excel",
                       "📄 Carregar arquivo CSV" = "csv", 
                       "🎲 Usar dados simulados" = "simulated"
                     ),
                     selected = "simulated")
    ),
    
    # Controles condicionais para upload de arquivo Excel
    conditionalPanel(
      condition = "input.data_source == 'excel'",
      fileInput("file_excel", "Selecionar arquivo Excel", 
                accept = c(".xlsx", ".xls"),
                placeholder = "Nenhum arquivo selecionado"),
      uiOutput("sheet_select_ui"),
      uiOutput("date_column_ui"),
      uiOutput("value_column_ui"),
      selectInput("date_format", "🗓️ Formato de data:",
                  choices = c("Automático" = "auto", 
                              "YYYY-MM-DD" = "ymd", 
                              "MM/DD/YYYY" = "mdy", 
                              "DD/MM/YYYY" = "dmy"),
                  selected = "auto")
    ),
    
    # Controles condicionais para upload de arquivo CSV
    conditionalPanel(
      condition = "input.data_source == 'csv'",
      fileInput("file_csv", "Selecionar arquivo CSV", 
                accept = c(".csv", ".txt"),
                placeholder = "Nenhum arquivo selecionado"),
      div(class = "info-card",
          h5("ℹ️ Detecção Automática"),
          p("O separador (vírgula ou ponto e vírgula) será detectado automaticamente.")
      ),
      uiOutput("csv_date_column_ui"),
      uiOutput("csv_value_column_ui"),
      selectInput("csv_date_format", "🗓️ Formato de data:",
                  choices = c("Automático" = "auto", 
                              "YYYY-MM-DD" = "ymd", 
                              "MM/DD/YYYY" = "mdy", 
                              "DD/MM/YYYY" = "dmy"),
                  selected = "auto")
    ),
    
    # Informação sobre dados simulados
    conditionalPanel(
      condition = "input.data_source == 'simulated'",
      div(class = "info-card",
          h5("🎯 Dados Simulados"),
          p("Série temporal mensal (2005-2025) com:"),
          tags$ul(
            tags$li("📈 Tendência crescente"),
            tags$li("🔄 Sazonalidade anual marcada"),
            tags$li("🌊 Ciclo de longo prazo"),
            tags$li("🎲 Ruído aleatório controlado")
          )
      )
    ),
    
    # Botão para executar análise
    br(),
    actionButton("run_tests", "🚀 Executar Análise", 
                 class = "btn btn-primary btn-block"),
    br(), br(),
    
    # Botão para exportar resultados
    downloadButton("download_excel", "📊 Exportar Resultados", 
                   class = "btn btn-success btn-block"),
    br(), br(),
    
    # Informações adicionais
    div(class = "info-card",
        h5("💡 Dica"),
        p("Use os dados simulados para uma demonstração rápida de todas as funcionalidades do dashboard.")
    )
  ),
  
  # Corpo principal do dashboard
  dashboardBody(
    # Loading CSS
    tags$head(
      tags$link(rel = "stylesheet", type = "text/css", 
                href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css")
    ),
    
    tabsetPanel(
      # ====================================================================
      # ABA 1: TESTES ESTATÍSTICOS
      # ====================================================================
      tabPanel("📊 Testes Estatísticos",
               fluidRow(
                 box(title = "📈 Resultados dos Testes Estatísticos", 
                     width = 12, status = "primary",
                     # Primeira linha de info boxes
                     fluidRow(
                       column(width = 4, withSpinner(infoBoxOutput("adf_info", width = NULL), type = 4, color = "#667eea")),
                       column(width = 4, withSpinner(infoBoxOutput("pp_info", width = NULL), type = 4, color = "#667eea")),
                       column(width = 4, withSpinner(infoBoxOutput("kpss_info", width = NULL), type = 4, color = "#667eea"))
                     ),
                     # Segunda linha de info boxes
                     fluidRow(
                       column(width = 4, withSpinner(infoBoxOutput("box_pierce_info", width = NULL), type = 4, color = "#667eea")),
                       column(width = 4, withSpinner(infoBoxOutput("ljung_box_info", width = NULL), type = 4, color = "#667eea")),
                       column(width = 4, withSpinner(infoBoxOutput("arch_info", width = NULL), type = 4, color = "#667eea"))
                     ),
                     # Terceira linha de info boxes
                     fluidRow(
                       column(width = 12, withSpinner(infoBoxOutput("anderson_info", width = NULL), type = 4, color = "#667eea"))
                     ),
                     hr(),
                     # Tabela resumo dos testes
                     withSpinner(dataTableOutput("tests_summary"), type = 4, color = "#667eea")
                 )
               ),
               # Gráficos de autocorrelação
               fluidRow(
                 box(title = "🔗 Funções de Autocorrelação", width = 12,
                     withSpinner(plotlyOutput("acf_pacf_plot", height = "700px"), type = 4, color = "#667eea"),
                     hr(),
                     # Guia de interpretação melhorado
                     div(class = "guide-box",
                         h4("📚 Como Interpretar os Gráficos ACF e PACF"),
                         p(strong("Autocorrelação (ACF):"), "Mostra a correlação entre a série e suas defasagens (lags). Barras que ultrapassam as linhas tracejadas vermelhas são estatisticamente significativas."),
                         p(strong("Autocorrelação Parcial (PACF):"), "Mostra a correlação entre a série e uma defasagem específica, removendo o efeito das defasagens intermediárias."),
                         p(strong("🎯 Identificação de modelos ARIMA:")),
                         tags$ul(
                           tags$li(strong("AR(p):"), "PACF 'corta' após o lag p; ACF 'decai' gradualmente"),
                           tags$li(strong("MA(q):"), "ACF 'corta' após o lag q; PACF 'decai' gradualmente"),
                           tags$li(strong("ARMA(p,q):"), "Ambos ACF e PACF decaem gradualmente")
                         ),
                         p(strong("💡 Significado prático:"), "Autocorrelações significativas indicam que valores passados influenciam valores futuros, fundamental para modelagem e previsão.")
                     )
                 )
               )
      ),
      
      # ====================================================================
      # ABA 2: VISUALIZAÇÕES
      # ====================================================================
      tabPanel("📈 Visualizações",
               # Gráfico da série temporal
               fluidRow(
                 box(title = "📊 Série Temporal Completa", width = 12, 
                     withSpinner(plotlyOutput("time_series_plot", height = "400px"), type = 4, color = "#667eea"))
               ),
               # Histograma e boxplot
               fluidRow(
                 column(width = 6,
                        box(title = "📊 Distribuição dos Valores", width = NULL,
                            withSpinner(plotlyOutput("histogram_plot", height = "350px"), type = 4, color = "#667eea"))
                 ),
                 column(width = 6,
                        box(title = "📦 Análise de Outliers", width = NULL,
                            withSpinner(plotlyOutput("boxplot_plot", height = "350px"), type = 4, color = "#667eea"))
                 )
               ),
               # Histograma dos resíduos
               fluidRow(
                 box(title = "🔍 Análise dos Resíduos do Modelo", width = 12,
                     withSpinner(plotlyOutput("residuals_histogram_plot", height = "350px"), type = 4, color = "#667eea"))
               )
      ),
      
      # ====================================================================
      # ABA 3: DECOMPOSIÇÃO E PREVISÃO
      # ====================================================================
      tabPanel("🔍 Decomposição & Previsão",
               # Decomposição STL
               fluidRow(
                 box(title = "🧩 Decomposição STL da Série Temporal", width = 12,
                     withSpinner(plotlyOutput("decomposition_plot", height = "700px"), type = 4, color = "#667eea"),
                     div(class = "guide-box",
                         h4("📖 Sobre a Decomposição STL"),
                         p("A decomposição STL (Seasonal and Trend decomposition using Loess) separa a série temporal em:"),
                         tags$ul(
                           tags$li(strong("Tendência:"), "Movimento de longo prazo dos dados"),
                           tags$li(strong("Sazonalidade:"), "Padrões que se repetem em períodos fixos"),
                           tags$li(strong("Resíduo:"), "Variações aleatórias não explicadas pelos componentes anteriores")
                         )
                     )
                 )
               ),
               # Previsão ARIMA
               fluidRow(
                 box(title = "🔮 Previsão com Modelo ARIMA", width = 12,
                     fluidRow(
                       column(width = 6,
                              numericInput("forecast_periods", "📅 Períodos para previsão:", 
                                           12, min = 1, max = 36, step = 1)
                       ),
                       column(width = 6,
                              checkboxInput("auto_arima", "🤖 Auto ARIMA (Recomendado)", TRUE)
                       )
                     ),
                     withSpinner(plotlyOutput("forecast_plot", height = "500px"), type = 4, color = "#667eea"),
                     div(class = "guide-box",
                         h4("🎯 Interpretação da Previsão"),
                         p("O gráfico mostra:"),
                         tags$ul(
                           tags$li(strong("Linha azul:"), "Dados históricos observados"),
                           tags$li(strong("Linha vermelha:"), "Valores previstos pelo modelo"),
                           tags$li(strong("Área clara:"), "Intervalo de confiança de 80%"),
                           tags$li(strong("Área mais clara:"), "Intervalo de confiança de 95%")
                         ),
                         p("Quanto maior o horizonte de previsão, maior a incerteza (bandas mais largas).")
                     )
                 )
               )
      ),
      
      # ====================================================================
      # ABA 4: DADOS
      # ====================================================================
      tabPanel("💾 Dados",
               fluidRow(
                 box(title = "📋 Dataset Processado", width = 12,
                     downloadButton("download_data", "📥 Exportar Dados Processados", 
                                    class = "btn btn-info"),
                     br(), br(),
                     withSpinner(dataTableOutput("data_table"), type = 4, color = "#667eea"))
               )
      ),
      
      # ====================================================================
      # ABA 5: EXPLICAÇÃO DOS TESTES
      # ====================================================================
      tabPanel("📚 Guia Completo",
               fluidRow(
                 box(title = "📖 Guia Completo de Interpretação dos Testes", 
                     width = 12, status = "primary",
                     HTML('<div style="padding: 15px;">
                           <h3 style="color: #667eea; margin-bottom: 25px;">🔬 1. Testes de Estacionariedade</h3>
                           
                           <div class="guide-box">
                             <h4>📊 Teste de Dickey-Fuller Aumentado (ADF)</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> A série tem uma raiz unitária (série NÃO é estacionária)</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> A série NÃO tem raiz unitária (série É estacionária)</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor < 0.05, rejeitamos H0 e concluímos que a série é estacionária.</p>
                             <p><strong>💡 Implicação prática:</strong> Série estacionária é ideal para modelagem ARMA. Pode prosseguir sem diferenciação.</p>
                           </div>
                           
                           <div class="guide-box">
                             <h4>📊 Teste de Phillips-Perron (PP)</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> A série tem uma raiz unitária (série NÃO é estacionária)</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> A série NÃO tem raiz unitária (série É estacionária)</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor < 0.05, rejeitamos H0 e concluímos que a série é estacionária.</p>
                             <p><strong>💡 Vantagem:</strong> Mais robusto que o ADF para heterocedasticidade e autocorrelações de ordem superior.</p>
                           </div>
                           
                           <div class="guide-box">
                             <h4>📊 Teste KPSS</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> A série É estacionária</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> A série NÃO é estacionária</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor > 0.05, NÃO rejeitamos H0 (série é estacionária).</p>
                             <p><strong>⚠️ Importante:</strong> Hipótese nula oposta aos outros testes. Excelente para confirmação cruzada.</p>
                           </div>
                           
                           <h3 style="color: #667eea; margin: 30px 0 25px 0;">🔗 2. Testes de Autocorrelação</h3>
                           
                           <div class="guide-box">
                             <h4>📊 Teste Box-Pierce</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> Não há autocorrelação nos resíduos</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> Existe autocorrelação nos resíduos</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor > 0.05, modelo adequado (sem autocorrelação residual).</p>
                             <p><strong>🔧 Se rejeitado:</strong> Modelo precisa ser respecificado (ajustar parâmetros AR/MA).</p>
                           </div>
                           
                           <div class="guide-box">
                             <h4>📊 Teste Ljung-Box</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> Não há autocorrelação nos resíduos</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> Existe autocorrelação nos resíduos</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor > 0.05, modelo adequado (sem autocorrelação residual).</p>
                             <p><strong>⭐ Vantagem:</strong> Versão aprimorada do Box-Pierce, melhor para amostras pequenas.</p>
                           </div>
                           
                           <h3 style="color: #667eea; margin: 30px 0 25px 0;">📊 3. Teste de Heterocedasticidade</h3>
                           
                           <div class="guide-box">
                             <h4>📊 Teste ARCH</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> Não há efeitos ARCH (variância constante)</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> Existem efeitos ARCH (variância condicional)</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor > 0.05, variância é estável ao longo do tempo.</p>
                             <p><strong>🔧 Se rejeitado:</strong> Considerar modelos GARCH para capturar variância condicional.</p>
                           </div>
                           
                           <h3 style="color: #667eea; margin: 30px 0 25px 0;">📈 4. Teste de Normalidade</h3>
                           
                           <div class="guide-box">
                             <h4>📊 Teste Anderson-Darling</h4>
                             <p><strong>🎯 Hipótese Nula (H0):</strong> Os resíduos seguem distribuição normal</p>
                             <p><strong>🎯 Hipótese Alternativa (H1):</strong> Os resíduos não são normalmente distribuídos</p>
                             <p><strong>📏 Interpretação:</strong> Se p-valor > 0.05, resíduos são normais (ideal).</p>
                             <p><strong>⚠️ Se rejeitado:</strong> Intervalos de confiança podem não ser confiáveis, mas modelo ainda útil para previsão.</p>
                           </div>
                           
                           <h3 style="color: #667eea; margin: 30px 0 25px 0;">🤔 Lidando com Resultados Conflitantes</h3>
                           <div class="guide-box">
                             <p>É comum testes de estacionariedade apresentarem resultados contraditórios:</p>
                             <ul>
                               <li>🔍 <strong>Examine visualmente:</strong> Gráfico da série e ACF/PACF são fundamentais</li>
                               <li>📊 <strong>Considere o contexto:</strong> Natureza econômica/física dos dados</li>
                               <li>🧪 <strong>Teste transformações:</strong> Diferenciação, logaritmo, Box-Cox</li>
                               <li>⚖️ <strong>Seja conservador:</strong> Na dúvida, assuma não-estacionariedade e diferencie</li>
                             </ul>
                           </div>
                           
                           <h3 style="color: #667eea; margin: 30px 0 25px 0;">🎯 Implicações Práticas para Modelagem</h3>
                           
                           <div class="guide-box">
                             <h4>🔧 Se a série NÃO for estacionária:</h4>
                             <ul>
                               <li>Aplicar diferenciação (ARIMA com d > 0)</li>
                               <li>Considerar transformações logarítmicas</li>
                               <li>Verificar cointegração se múltiplas séries</li>
                             </ul>
                           </div>
                           
                           <div class="guide-box">
                             <h4>🔧 Se houver autocorrelação residual:</h4>
                             <ul>
                               <li>Aumentar ordem AR ou MA no modelo</li>
                               <li>Considerar componentes sazonais (SARIMA)</li>
                               <li>Verificar se há quebras estruturais</li>
                             </ul>
                           </div>
                           
                           <div class="guide-box">
                             <h4>🔧 Se houver heterocedasticidade:</h4>
                             <ul>
                               <li>Modelos da família GARCH</li>
                               <li>Transformações estabilizadoras de variância</li>
                               <li>Robust standard errors</li>
                             </ul>
                           </div>
                           
                           <div class="guide-box">
                             <h4>🔧 Se resíduos não forem normais:</h4>
                             <ul>
                               <li>Investigar e tratar outliers</li>
                               <li>Modelo ainda válido para previsão</li>
                               <li>Usar bootstrap para intervalos de confiança</li>
                             </ul>
                           </div>
                           </div>')
                 )
               )
      )
    )
  )
)

# ====================================================================
# 6. LÓGICA DO SERVIDOR (SERVER)
# ====================================================================

server <- function(input, output, session) {
  
  # ====================================================================
  # 6.1 FUNÇÕES AUXILIARES
  # ====================================================================
  
  # Função para formatar valores-p de forma consistente
  format_p_value <- function(p) {
    if (is.na(p)) return("NA")
    if (p < 0.0001) return("< 0.0001")
    return(format(p, digits = 4))
  }
  
  # ====================================================================
  # 6.2 CARREGAMENTO DE DADOS
  # ====================================================================
  
  # Detectar sheets disponíveis no arquivo Excel
  sheets_available <- reactive({
    req(input$file_excel)
    excel_sheets(input$file_excel$datapath)
  })
  
  # Interface dinâmica para seleção de sheet (Excel)
  output$sheet_select_ui <- renderUI({
    req(sheets_available())
    selectInput("sheet", "📋 Selecionar Aba:", choices = sheets_available())
  })
  
  # Carregar dataset do Excel
  dataset_excel <- reactive({
    req(input$file_excel, input$sheet)
    read_xlsx(input$file_excel$datapath, sheet = input$sheet)
  })
  
  # Carregar dataset do CSV
  dataset_csv <- reactive({
    req(input$file_csv)
    
    # Detectar separador automaticamente
    separator <- detect_csv_separator(input$file_csv$datapath)
    
    # Ler arquivo CSV
    if (separator == ";") {
      read_delim(input$file_csv$datapath, delim = ";", locale = locale(encoding = "UTF-8"))
    } else {
      read_csv(input$file_csv$datapath, locale = locale(encoding = "UTF-8"))
    }
  })
  
  # Interface dinâmica para seleção de colunas (Excel)
  output$date_column_ui <- renderUI({
    req(dataset_excel())
    selectInput("date_column", "📅 Coluna de Data:", choices = names(dataset_excel()))
  })
  
  output$value_column_ui <- renderUI({
    req(dataset_excel())
    cols <- names(dataset_excel())
    numeric_cols <- cols[sapply(dataset_excel(), is.numeric)]
    selectInput("value_column", "📊 Coluna de Valores:", 
                choices = cols, selected = if(length(numeric_cols) > 0) numeric_cols[1] else cols[1])
  })
  
  # Interface dinâmica para seleção de colunas (CSV)
  output$csv_date_column_ui <- renderUI({
    req(dataset_csv())
    selectInput("csv_date_column", "📅 Coluna de Data:", choices = names(dataset_csv()))
  })
  
  output$csv_value_column_ui <- renderUI({
    req(dataset_csv())
    cols <- names(dataset_csv())
    numeric_cols <- cols[sapply(dataset_csv(), is.numeric)]
    selectInput("csv_value_column", "📊 Coluna de Valores:", 
                choices = cols, selected = if(length(numeric_cols) > 0) numeric_cols[1] else cols[1])
  })
  
  # ====================================================================
  # 6.3 PROCESSAMENTO DE DADOS
  # ====================================================================
  
  # Valores reativos para armazenar resultados da análise
  results <- reactiveValues(
    data = NULL,          # Dados mensais processados
    ts_data = NULL,       # Série temporal
    tests = NULL,         # Resultados dos testes estatísticos
    arima_model = NULL,   # Modelo ARIMA ajustado
    residuals = NULL      # Resíduos do modelo
  )
  
  # Evento principal: executar análise quando botão é pressionado
  observeEvent(input$run_tests, {
    
    # Determinar fonte de dados e carregar
    if (input$data_source == "simulated") {
      # Usar dados simulados
      monthly_data <- generate_simulated_data()
      showNotification("✅ Dados simulados carregados com sucesso!", 
                       type = "message", duration = 3)
      
    } else if (input$data_source == "excel") {
      # Usar dados do arquivo Excel
      req(input$file_excel, input$sheet, input$date_column, input$value_column)
      
      # Carregar e processar dados do Excel
      data <- dataset_excel()
      
      # Converter coluna de data baseado no formato selecionado
      if (input$date_format == "auto") {
        date_col <- suppressWarnings(ymd(data[[input$date_column]]))
        if (all(is.na(date_col))) date_col <- suppressWarnings(mdy(data[[input$date_column]]))
        if (all(is.na(date_col))) date_col <- suppressWarnings(dmy(data[[input$date_column]]))
        data[[input$date_column]] <- date_col
      } else if (input$date_format == "ymd") {
        data[[input$date_column]] <- ymd(data[[input$date_column]])
      } else if (input$date_format == "mdy") {
        data[[input$date_column]] <- mdy(data[[input$date_column]])
      } else if (input$date_format == "dmy") {
        data[[input$date_column]] <- dmy(data[[input$date_column]])
      }
      
      # Verificar se a conversão de data foi bem-sucedida
      if (all(is.na(data[[input$date_column]]))) {
        showNotification("❌ Erro na conversão de datas. Verifique o formato.", 
                         type = "error", duration = 5)
        return(NULL)
      }
      
      # Agregar dados por mês (médias mensais)
      monthly_data <- data %>% 
        mutate(Date = floor_date(get(input$date_column), "month")) %>%
        group_by(Date) %>%
        summarize(Value = mean(get(input$value_column), na.rm = TRUE), .groups = 'drop') %>%
        na.omit()
      
      showNotification("✅ Arquivo Excel processado com sucesso!", 
                       type = "message", duration = 3)
      
    } else if (input$data_source == "csv") {
      # Usar dados do arquivo CSV
      req(input$file_csv, input$csv_date_column, input$csv_value_column)
      
      # Carregar e processar dados do CSV
      data <- dataset_csv()
      
      # Converter coluna de data baseado no formato selecionado
      if (input$csv_date_format == "auto") {
        date_col <- suppressWarnings(ymd(data[[input$csv_date_column]]))
        if (all(is.na(date_col))) date_col <- suppressWarnings(mdy(data[[input$csv_date_column]]))
        if (all(is.na(date_col))) date_col <- suppressWarnings(dmy(data[[input$csv_date_column]]))
        data[[input$csv_date_column]] <- date_col
      } else if (input$csv_date_format == "ymd") {
        data[[input$csv_date_column]] <- ymd(data[[input$csv_date_column]])
      } else if (input$csv_date_format == "mdy") {
        data[[input$csv_date_column]] <- mdy(data[[input$csv_date_column]])
      } else if (input$csv_date_format == "dmy") {
        data[[input$csv_date_column]] <- dmy(data[[input$csv_date_column]])
      }
      
      # Verificar se a conversão de data foi bem-sucedida
      if (all(is.na(data[[input$csv_date_column]]))) {
        showNotification("❌ Erro na conversão de datas. Verifique o formato.", 
                         type = "error", duration = 5)
        return(NULL)
      }
      
      # Agregar dados por mês (médias mensais)
      monthly_data <- data %>% 
        mutate(Date = floor_date(get(input$csv_date_column), "month")) %>%
        group_by(Date) %>%
        summarize(Value = mean(get(input$csv_value_column), na.rm = TRUE), .groups = 'drop') %>%
        na.omit()
      
      showNotification("✅ Arquivo CSV processado com sucesso!", 
                       type = "message", duration = 3)
    }
    
    # ====================================================================
    # 6.4 CRIAÇÃO DA SÉRIE TEMPORAL
    # ====================================================================
    
    # Criar objeto de série temporal com frequência mensal
    ts_data <- ts(
      monthly_data$Value,
      frequency = 12,  # 12 observações por ano (mensal)
      start = c(year(min(monthly_data$Date)), month(min(monthly_data$Date)))
    )
    
    # ====================================================================
    # 6.5 EXECUÇÃO DOS TESTES ESTATÍSTICOS
    # ====================================================================
    
    # Mostrar progresso
    showNotification("🔬 Executando testes estatísticos...", 
                     type = "message", duration = 2)
    
    test_results <- list()
    
    # Teste ADF (Augmented Dickey-Fuller)
    test_results$adf <- tryCatch({
      adf.test(ts_data)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # Teste Phillips-Perron
    test_results$pp <- tryCatch({
      pp.test(ts_data)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # Teste KPSS
    test_results$kpss <- tryCatch({
      kpss.test(ts_data)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # ====================================================================
    # 6.6 AJUSTE DO MODELO ARIMA
    # ====================================================================
    
    showNotification("🤖 Ajustando modelo ARIMA...", 
                     type = "message", duration = 2)
    
    # Ajustar modelo ARIMA
    arima_model <- tryCatch({
      if (input$auto_arima) {
        auto.arima(ts_data)
      } else {
        arima(ts_data, order = c(1, 0, 1))
      }
    }, error = function(e) {
      showNotification("⚠️ Erro ao ajustar ARIMA. Tentando modelo mais simples.", 
                       type = "warning", duration = 3)
      arima(ts_data, order = c(1, 0, 0))
    })
    
    # Extrair resíduos do modelo
    residuals_data <- residuals(arima_model)
    
    # ====================================================================
    # 6.7 TESTES NOS RESÍDUOS
    # ====================================================================
    
    # Teste Box-Pierce
    test_results$box_pierce <- tryCatch({
      Box.test(residuals_data, type = "Box-Pierce", lag = 12)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # Teste Ljung-Box
    test_results$ljung_box <- tryCatch({
      Box.test(residuals_data, type = "Ljung-Box", lag = 12)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # Teste ARCH
    test_results$arch <- tryCatch({
      ArchTest(residuals_data, lags = 12)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # Teste Anderson-Darling
    test_results$anderson <- tryCatch({
      ad.test(residuals_data)
    }, error = function(e) {
      list(p.value = NA)
    })
    
    # ====================================================================
    # 6.8 ARMAZENAMENTO DOS RESULTADOS
    # ====================================================================
    
    # Armazenar todos os resultados
    results$data <- monthly_data
    results$ts_data <- ts_data
    results$tests <- test_results
    results$arima_model <- arima_model
    results$residuals <- residuals_data
    
    showNotification("🎉 Análise completa finalizada com sucesso!", 
                     type = "message", duration = 4)
  })
  
  # ====================================================================
  # 7. OUTPUTS - TABELAS E VISUALIZAÇÕES
  # ====================================================================
  
  # ====================================================================
  # 7.1 TABELAS
  # ====================================================================
  
  # Tabela de dados processados
  output$data_table <- renderDataTable({
    req(results$data)
    results$data
  }, options = list(
    pageLength = 15, 
    scrollX = TRUE,
    dom = 'Bfrtip',
    buttons = c('copy', 'csv', 'excel'),
    language = list(
      url = '//cdn.datatables.net/plug-ins/1.10.11/i18n/Portuguese-Brasil.json'
    )
  ))
  
  # Tabela resumo dos testes estatísticos
  output$tests_summary <- renderDataTable({
    req(results$tests)
    tests <- results$tests
    
    data.frame(
      `🧪 Teste` = c("ADF", "Phillips-Perron", "KPSS", "Box-Pierce", "Ljung-Box", "ARCH", "Anderson-Darling"),
      `📊 p-valor` = c(
        format_p_value(tests$adf$p.value),
        format_p_value(tests$pp$p.value),
        format_p_value(tests$kpss$p.value),
        format_p_value(tests$box_pierce$p.value),
        format_p_value(tests$ljung_box$p.value),
        format_p_value(tests$arch$p.value),
        format_p_value(tests$anderson$p.value)
      ),
      `📈 Resultado` = c(
        ifelse(tests$adf$p.value < 0.05, "✅ Série estacionária", "❌ Série não-estacionária"),
        ifelse(tests$pp$p.value < 0.05, "✅ Série estacionária", "❌ Série não-estacionária"),
        ifelse(tests$kpss$p.value < 0.05, "❌ Série não-estacionária", "✅ Série estacionária"),
        ifelse(tests$box_pierce$p.value < 0.05, "⚠️ Há autocorrelação", "✅ Sem autocorrelação"),
        ifelse(tests$ljung_box$p.value < 0.05, "⚠️ Há autocorrelação", "✅ Sem autocorrelação"),
        ifelse(!is.na(tests$arch$p.value) && tests$arch$p.value < 0.05, "⚠️ Há heterocedasticidade", "✅ Sem heterocedasticidade"),
        ifelse(tests$anderson$p.value < 0.05, "⚠️ Resíduos não normais", "✅ Resíduos normais")
      ),
      `⭐ Status` = c(
        ifelse(tests$adf$p.value < 0.05, "🟢 Positivo", "🔴 Negativo"),
        ifelse(tests$pp$p.value < 0.05, "🟢 Positivo", "🔴 Negativo"),
        ifelse(tests$kpss$p.value > 0.05, "🟢 Positivo", "🔴 Negativo"),
        ifelse(tests$box_pierce$p.value > 0.05, "🟢 Positivo", "🔴 Negativo"),
        ifelse(tests$ljung_box$p.value > 0.05, "🟢 Positivo", "🔴 Negativo"),
        ifelse(!is.na(tests$arch$p.value) && tests$arch$p.value < 0.05, "🔴 Negativo", "🟢 Positivo"),
        ifelse(tests$anderson$p.value > 0.05, "🟢 Positivo", "🔴 Negativo")
      ),
      check.names = FALSE
    )
  }, options = list(
    dom = 't', 
    ordering = FALSE, 
    paging = FALSE,
    columnDefs = list(list(className = 'dt-center', targets = "_all"))
  ))
  
  # ====================================================================
  # 7.2 INFO BOXES DOS TESTES
  # ====================================================================
  
  # Info Box: Teste ADF
  output$adf_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$adf$p.value)
    status <- results$tests$adf$p.value < 0.05
    
    infoBox(
      "ADF Test", 
      p_val,
      subtitle = ifelse(status, "✅ Estacionária", "❌ Não-estacionária"),
      icon = icon(ifelse(status, "check-circle", "times-circle")),
      color = ifelse(status, "green", "red"),
      fill = TRUE
    )
  })
  
  # Info Box: Teste Phillips-Perron
  output$pp_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$pp$p.value)
    status <- results$tests$pp$p.value < 0.05
    
    infoBox(
      "Phillips-Perron", 
      p_val,
      subtitle = ifelse(status, "✅ Estacionária", "❌ Não-estacionária"),
      icon = icon(ifelse(status, "check-circle", "times-circle")),
      color = ifelse(status, "green", "red"),
      fill = TRUE
    )
  })
  
  # Info Box: Teste KPSS
  output$kpss_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$kpss$p.value)
    status <- results$tests$kpss$p.value > 0.05
    
    infoBox(
      "KPSS Test", 
      p_val,
      subtitle = ifelse(status, "✅ Estacionária", "❌ Não-estacionária"),
      icon = icon(ifelse(status, "check-circle", "times-circle")),
      color = ifelse(status, "green", "red"),
      fill = TRUE
    )
  })
  
  # Info Box: Teste Box-Pierce
  output$box_pierce_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$box_pierce$p.value)
    status <- results$tests$box_pierce$p.value > 0.05
    
    infoBox(
      "Box-Pierce", 
      p_val,
      subtitle = ifelse(status, "✅ Adequado", "⚠️ Autocorrelação"),
      icon = icon(ifelse(status, "thumbs-up", "exclamation-triangle")),
      color = ifelse(status, "green", "yellow"),
      fill = TRUE
    )
  })
  
  # Info Box: Teste Ljung-Box
  output$ljung_box_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$ljung_box$p.value)
    status <- results$tests$ljung_box$p.value > 0.05
    
    infoBox(
      "Ljung-Box", 
      p_val,
      subtitle = ifelse(status, "✅ Adequado", "⚠️ Autocorrelação"),
      icon = icon(ifelse(status, "thumbs-up", "exclamation-triangle")),
      color = ifelse(status, "green", "yellow"),
      fill = TRUE
    )
  })
  
  # Info Box: Teste ARCH
  output$arch_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$arch$p.value)
    status <- is.na(results$tests$arch$p.value) || results$tests$arch$p.value > 0.05
    
    infoBox(
      "ARCH Test", 
      p_val,
      subtitle = ifelse(status, "✅ Homocedasticidade", "⚠️ Heterocedasticidade"),
      icon = icon(ifelse(status, "balance-scale", "chart-line")),
      color = ifelse(status, "green", "orange"),
      fill = TRUE
    )
  })
  
  # Info Box: Teste Anderson-Darling
  output$anderson_info <- renderInfoBox({
    req(results$tests)
    p_val <- format_p_value(results$tests$anderson$p.value)
    status <- results$tests$anderson$p.value > 0.05
    
    infoBox(
      "Anderson-Darling (Normalidade)", 
      p_val,
      subtitle = ifelse(status, "✅ Resíduos Normais", "⚠️ Não-Normais"),
      icon = icon(ifelse(status, "bell", "bell-slash")),
      color = ifelse(status, "blue", "purple"),
      fill = TRUE
    )
  })
  
  # ====================================================================
  # 7.3 GRÁFICOS PRINCIPAIS
  # ====================================================================
  
  # Gráfico da série temporal
  output$time_series_plot <- renderPlotly({
    req(results$data)
    
    p <- plot_ly(results$data, x = ~Date, y = ~Value, type = 'scatter', mode = 'lines',
                 line = list(color = '#667eea', width = 3),
                 hovertemplate = '<b>Data:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>') %>%
      layout(
        title = list(text = "📊 Evolução Temporal da Série", font = list(size = 18, color = '#2c3e50')),
        xaxis = list(title = "Data", gridcolor = '#ecf0f1'),
        yaxis = list(title = "Valor", gridcolor = '#ecf0f1'),
        plot_bgcolor = '#f8f9fa',
        paper_bgcolor = 'white',
        hovermode = 'x unified'
      )
    
    p
  })
  
  # Gráficos ACF e PACF
  output$acf_pacf_plot <- renderPlotly({
    req(results$ts_data)
    
    # Calcular ACF e PACF
    acf_data <- acf(results$ts_data, plot = FALSE)
    pacf_data <- pacf(results$ts_data, plot = FALSE)
    
    # Calcular limites de significância
    n <- length(results$ts_data)
    critical_value <- qnorm(0.975) / sqrt(n)
    
    # Gráfico ACF
    acf_plot <- plot_ly() %>%
      add_bars(
        x = seq(0, length(acf_data$acf) - 1),
        y = acf_data$acf,
        name = "ACF",
        marker = list(color = '#4facfe'),
        hovertemplate = '<b>Lag:</b> %{x}<br><b>ACF:</b> %{y:.3f}<extra></extra>'
      ) %>%
      add_segments(
        x = 0, xend = length(acf_data$acf) - 1, 
        y = critical_value, yend = critical_value,
        line = list(color = "red", dash = "dash", width = 2),
        showlegend = FALSE
      ) %>%
      add_segments(
        x = 0, xend = length(acf_data$acf) - 1, 
        y = -critical_value, yend = -critical_value,
        line = list(color = "red", dash = "dash", width = 2),
        showlegend = FALSE
      ) %>%
      layout(
        title = list(text = "🔗 Função de Autocorrelação (ACF)", font = list(size = 16)),
        xaxis = list(title = "Lag"),
        yaxis = list(title = "Autocorrelação", range = c(-1, 1)),
        plot_bgcolor = '#f8f9fa'
      )
    
    # Gráfico PACF
    pacf_plot <- plot_ly() %>%
      add_bars(
        x = seq(1, length(pacf_data$acf)),
        y = pacf_data$acf,
        name = "PACF",
        marker = list(color = '#43e97b'),
        hovertemplate = '<b>Lag:</b> %{x}<br><b>PACF:</b> %{y:.3f}<extra></extra>'
      ) %>%
      add_segments(
        x = 1, xend = length(pacf_data$acf), 
        y = critical_value, yend = critical_value,
        line = list(color = "red", dash = "dash", width = 2),
        showlegend = FALSE
      ) %>%
      add_segments(
        x = 1, xend = length(pacf_data$acf), 
        y = -critical_value, yend = -critical_value,
        line = list(color = "red", dash = "dash", width = 2),
        showlegend = FALSE
      ) %>%
      layout(
        title = list(text = "🎯 Função de Autocorrelação Parcial (PACF)", font = list(size = 16)),
        xaxis = list(title = "Lag"),
        yaxis = list(title = "Autocorrelação Parcial", range = c(-1, 1)),
        plot_bgcolor = '#f8f9fa'
      )
    
    # Combinar os gráficos
    subplot(acf_plot, pacf_plot, nrows = 2, shareX = FALSE) %>%
      layout(
        annotations = list(
          list(
            x = 0.5, y = 1.02, 
            text = "Linhas tracejadas vermelhas: Limites de significância estatística (95%)", 
            showarrow = FALSE,
            xref = "paper", yref = "paper",
            font = list(size = 12, color = '#7f8c8d')
          )
        )
      )
  })
  
  # ====================================================================
  # 7.4 GRÁFICOS AUXILIARES
  # ====================================================================
  
  # Histograma dos valores
  output$histogram_plot <- renderPlotly({
    req(results$data)
    
    plot_ly(x = ~results$data$Value, type = "histogram", nbinsx = 20,
            marker = list(color = '#f093fb', line = list(color = '#764ba2', width = 2))) %>%
      layout(
        title = list(text = "📊 Distribuição dos Valores", font = list(size = 16)),
        xaxis = list(title = "Valor"),
        yaxis = list(title = "Frequência"),
        plot_bgcolor = '#f8f9fa'
      )
  })
  
  # Box plot dos valores
  output$boxplot_plot <- renderPlotly({
    req(results$data)
    
    plot_ly(y = ~results$data$Value, type = "box",
            marker = list(color = '#43e97b'),
            line = list(color = '#2c3e50', width = 2)) %>%
      layout(
        title = list(text = "📦 Análise de Outliers", font = list(size = 16)),
        yaxis = list(title = "Valor"),
        plot_bgcolor = '#f8f9fa'
      )
  })
  
  # Histograma dos resíduos
  output$residuals_histogram_plot <- renderPlotly({
    req(results$residuals)
    
    plot_ly(x = ~results$residuals, type = "histogram", nbinsx = 20,
            marker = list(color = '#fa709a', line = list(color = '#f093fb', width = 2))) %>%
      layout(
        title = list(text = "🔍 Distribuição dos Resíduos", font = list(size = 16)),
        xaxis = list(title = "Resíduos"),
        yaxis = list(title = "Frequência"),
        plot_bgcolor = '#f8f9fa'
      )
  })
  
  # ====================================================================
  # 7.5 DECOMPOSIÇÃO STL
  # ====================================================================
  
  output$decomposition_plot <- renderPlotly({
    req(results$ts_data)
    
    # Decomposição STL
    decomp <- stl(results$ts_data, s.window = "periodic")
    
    # Extrair componentes
    trend <- decomp$time.series[, "trend"]
    seasonal <- decomp$time.series[, "seasonal"]
    remainder <- decomp$time.series[, "remainder"]
    
    # Criar datas para plotagem
    dates <- results$data$Date
    
    # Gráfico da série original
    p1 <- plot_ly(x = ~dates, y = ~as.numeric(results$ts_data), 
                  type = 'scatter', mode = 'lines', name = "Original",
                  line = list(color = '#667eea', width = 2)) %>%
      layout(title = list(text = "📊 Série Original", font = list(size = 14)),
             yaxis = list(title = "Valor"))
    
    # Gráfico da tendência
    p2 <- plot_ly(x = ~dates, y = ~as.numeric(trend), 
                  type = 'scatter', mode = 'lines', name = "Tendência",
                  line = list(color = '#4facfe', width = 2)) %>%
      layout(title = list(text = "📈 Tendência", font = list(size = 14)),
             yaxis = list(title = "Tendência"))
    
    # Gráfico da sazonalidade
    p3 <- plot_ly(x = ~dates, y = ~as.numeric(seasonal), 
                  type = 'scatter', mode = 'lines', name = "Sazonalidade",
                  line = list(color = '#43e97b', width = 2)) %>%
      layout(title = list(text = "🔄 Sazonalidade", font = list(size = 14)),
             yaxis = list(title = "Sazonalidade"))
    
    # Gráfico dos resíduos
    p4 <- plot_ly(x = ~dates, y = ~as.numeric(remainder), 
                  type = 'scatter', mode = 'lines', name = "Resíduo",
                  line = list(color = '#fa709a', width = 2)) %>%
      layout(title = list(text = "🎲 Resíduo", font = list(size = 14)),
             yaxis = list(title = "Resíduo"))
    
    # Combinar todos os gráficos
    subplot(p1, p2, p3, p4, nrows = 4, shareX = TRUE) %>%
      layout(
        title = list(text = "🧩 Decomposição STL Completa", font = list(size = 18)),
        showlegend = FALSE
      )
  })
  
  # ====================================================================
  # 7.6 PREVISÃO ARIMA
  # ====================================================================
  
  output$forecast_plot <- renderPlotly({
    req(results$arima_model, input$forecast_periods)
    
    # Gerar previsão
    forecast_result <- forecast(results$arima_model, h = input$forecast_periods)
    
    # Obter datas para o período de previsão
    last_date <- max(results$data$Date)
    forecast_dates <- seq(last_date, by = "month", length.out = input$forecast_periods + 1)[-1]
    
    # Criar gráfico base
    p <- plot_ly() %>%
      # Dados históricos
      add_trace(
        x = results$data$Date,
        y = results$data$Value,
        type = "scatter",
        mode = "lines",
        name = "📊 Histórico",
        line = list(color = "#667eea", width = 3),
        hovertemplate = '<b>Data:</b> %{x}<br><b>Valor:</b> %{y:.2f}<extra></extra>'
      ) %>%
      # Previsão
      add_trace(
        x = forecast_dates,
        y = forecast_result$mean,
        type = "scatter",
        mode = "lines",
        name = "🔮 Previsão",
        line = list(color = "#fa709a", width = 3, dash = "dot"),
        hovertemplate = '<b>Data:</b> %{x}<br><b>Previsão:</b> %{y:.2f}<extra></extra>'
      ) %>%
      # Intervalo de confiança 80%
      add_ribbons(
        x = forecast_dates,
        ymin = forecast_result$lower[, 1],
        ymax = forecast_result$upper[, 1],
        name = "IC 80%",
        line = list(color = "transparent"),
        fillcolor = "rgba(250, 112, 154, 0.3)",
        hovertemplate = '<b>IC 80%</b><br>Min: %{ymin:.2f}<br>Max: %{ymax:.2f}<extra></extra>'
      ) %>%
      # Intervalo de confiança 95%
      add_ribbons(
        x = forecast_dates,
        ymin = forecast_result$lower[, 2],
        ymax = forecast_result$upper[, 2],
        name = "IC 95%",
        line = list(color = "transparent"),
        fillcolor = "rgba(250, 112, 154, 0.15)",
        hovertemplate = '<b>IC 95%</b><br>Min: %{ymin:.2f}<br>Max: %{ymax:.2f}<extra></extra>'
      ) %>%
      layout(
        title = list(text = "🔮 Previsão ARIMA com Intervalos de Confiança", font = list(size = 18)),
        xaxis = list(title = "Data", gridcolor = '#ecf0f1'),
        yaxis = list(title = "Valor", gridcolor = '#ecf0f1'),
        plot_bgcolor = '#f8f9fa',
        paper_bgcolor = 'white',
        legend = list(x = 0.02, y = 0.98, bgcolor = 'rgba(255,255,255,0.8)'),
        hovermode = 'x unified'
      )
    
    p
  })
  
  # ====================================================================
  # 8. DOWNLOADS E EXPORTAÇÕES
  # ====================================================================
  
  # Download dos resultados completos
  output$download_excel <- downloadHandler(
    filename = function() {
      paste("analise_serie_temporal_completa_", format(Sys.Date(), "%Y%m%d"), ".xlsx", sep = "")
    },
    content = function(file) {
      req(results$tests)
      tests <- results$tests
      
      # Dataframe dos testes
      tests_df <- data.frame(
        Teste = c("ADF", "Phillips-Perron", "KPSS", "Box-Pierce", "Ljung-Box", "ARCH", "Anderson-Darling"),
        Hipotese_Nula = c(
          "Série não-estacionária", 
          "Série não-estacionária", 
          "Série estacionária", 
          "Sem autocorrelação", 
          "Sem autocorrelação", 
          "Sem heterocedasticidade", 
          "Resíduos normais"
        ),
        p_valor = c(
          format_p_value(tests$adf$p.value),
          format_p_value(tests$pp$p.value),
          format_p_value(tests$kpss$p.value),
          format_p_value(tests$box_pierce$p.value),
          format_p_value(tests$ljung_box$p.value),
          format_p_value(tests$arch$p.value),
          format_p_value(tests$anderson$p.value)
        ),
        Resultado = c(
          ifelse(tests$adf$p.value < 0.05, "Rejeitar H0 (Série estacionária)", "Não rejeitar H0 (Série não-estacionária)"),
          ifelse(tests$pp$p.value < 0.05, "Rejeitar H0 (Série estacionária)", "Não rejeitar H0 (Série não-estacionária)"),
          ifelse(tests$kpss$p.value < 0.05, "Rejeitar H0 (Série não-estacionária)", "Não rejeitar H0 (Série estacionária)"),
          ifelse(tests$box_pierce$p.value < 0.05, "Rejeitar H0 (Há autocorrelação)", "Não rejeitar H0 (Sem autocorrelação)"),
          ifelse(tests$ljung_box$p.value < 0.05, "Rejeitar H0 (Há autocorrelação)", "Não rejeitar H0 (Sem autocorrelação)"),
          ifelse(!is.na(tests$arch$p.value) && tests$arch$p.value < 0.05, "Rejeitar H0 (Há heterocedasticidade)", "Não rejeitar H0 (Sem heterocedasticidade)"),
          ifelse(tests$anderson$p.value < 0.05, "Rejeitar H0 (Resíduos não normais)", "Não rejeitar H0 (Resíduos normais)")
        ),
        Status = c(
          ifelse(tests$adf$p.value < 0.05, "Positivo", "Negativo"),
          ifelse(tests$pp$p.value < 0.05, "Positivo", "Negativo"),
          ifelse(tests$kpss$p.value > 0.05, "Positivo", "Negativo"),
          ifelse(tests$box_pierce$p.value > 0.05, "Positivo", "Negativo"),
          ifelse(tests$ljung_box$p.value > 0.05, "Positivo", "Negativo"),
          ifelse(!is.na(tests$arch$p.value) && tests$arch$p.value < 0.05, "Negativo", "Positivo"),
          ifelse(tests$anderson$p.value > 0.05, "Positivo", "Negativo")
        )
      )
      
      # Informações do modelo
      if (!is.null(results$arima_model)) {
        model_info <- data.frame(
          Metrica = c("AIC", "BIC", "Log Likelihood", "Ordem (p,d,q)", "Sigma²"),
          Valor = c(
            round(results$arima_model$aic, 4),
            round(BIC(results$arima_model), 4),
            round(results$arima_model$loglik, 4),
            paste0("(", results$arima_model$arma[1], ",", results$arima_model$arma[6], ",", results$arima_model$arma[2], ")"),
            round(results$arima_model$sigma2, 6)
          )
        )
      } else {
        model_info <- data.frame(
          Metrica = "Status",
          Valor = "Modelo ARIMA não disponível"
        )
      }
      
      # Lista para exportação
      export_list <- list(
        "Resumo_Executivo" = tests_df,
        "Modelo_ARIMA" = model_info,
        "Dados_Processados" = if(!is.null(results$data)) results$data else data.frame(Nota = "Dados não disponíveis"),
        "Residuos_Modelo" = if(!is.null(results$residuals)) data.frame(Residuos = as.numeric(results$residuals)) else data.frame(Nota = "Resíduos não disponíveis")
      )
      
      # Exportar
      write_xlsx(export_list, file)
    }
  )
  
  # Download dos dados processados
  output$download_data <- downloadHandler(
    filename = function() {
      paste("dados_processados_", format(Sys.Date(), "%Y%m%d"), ".xlsx", sep = "")
    },
    content = function(file) {
      req(results$data)
      write_xlsx(results$data, file)
    }
  )
}

# ====================================================================
# 9. EXECUÇÃO DO APLICATIVO
# ====================================================================

# Executar o dashboard premium
shinyApp(ui, server)