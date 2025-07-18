# app.R - Versão Completa com Desazonalização, Opções Avançadas (BSTS manual),


if (!require(pacman)) install.packages("pacman")
p_load(shiny, tidyverse, lubridate, plotly, zoo, CausalImpact, DT, htmltools, readxl, shinythemes,
       shinyjs, shinydashboard, shinyWidgets, data.table, openxlsx) 

# Aumentar limite máximo de upload de arquivos para 100 MB
options(shiny.maxRequestSize = 100 * 1024^2)


########################################################################
#        FUNÇÃO PARA APLICAR O PACOTE CAUSAL IMPACT
########################################################################
if(!require(pacman)){install.packages("pacman")}
pacman::p_load(tidyverse, lubridate, plotly, zoo, CausalImpact, ggplot2, data.table)

## Função para gerar a base 
# O período pré-intervenção é de data_inicio até (data_inicio_evento - 1 dia)
generate_df_causal_impact <- function(data_frame, data_inicio, data_fim, data_inicio_evento, 
                                      var_Y, var_Xs, deseasonalize = "none", freq_sazonal = 12) {
  # Validações
  if (!("Data" %in% names(data_frame))) {
    stop("Coluna 'Data' não encontrada no dataframe.")
  }
  if (!(var_Y %in% names(data_frame))) {
    stop(paste("Variável resposta", var_Y, "não encontrada no dataframe."))
  }
  for (var in var_Xs) {
    if (!(var %in% names(data_frame))) {
      stop(paste("Variável explicativa", var, "não encontrada no dataframe."))
    }
  }
  
  df_A <- data_frame %>% 
    arrange(Data) %>% 
    filter(Data <= as.Date(data_fim)) %>% 
    select(Data, all_of(c(var_Y, var_Xs)))
  
  # Verifica se há dados suficientes
  if (nrow(df_A) < 10) {
    stop("Dados insuficientes para análise. Mínimo de 10 observações necessárias.")
  }
  
  # Desazonalizar a série se solicitado
  if (deseasonalize != "none") {
    for (col in c(var_Y, var_Xs)) {
      series <- ts(df_A[[col]], frequency = freq_sazonal)
      deseasoned <- deseasonalize_series(series, method = deseasonalize, frequency = freq_sazonal)
      df_A[[col]] <- as.numeric(deseasoned)
    }
  }
  
  df_zoo <- zoo(df_A[, !(names(df_A) == "Data")], df_A$Data)
  
  pre.period <- as.Date(c(data_inicio, as.Date(data_inicio_evento) - 1))
  post.period <- as.Date(c(data_inicio_evento, data_fim))
  
  # Validação dos períodos
  pre_data_count <- sum(index(df_zoo) >= pre.period[1] & index(df_zoo) <= pre.period[2])
  post_data_count <- sum(index(df_zoo) >= post.period[1] & index(df_zoo) <= post.period[2])
  
  if (pre_data_count < 5) {
    stop("Período pré-intervenção muito curto. Mínimo de 5 observações necessárias.")
  }
  if (post_data_count < 1) {
    stop("Período pós-intervenção deve conter pelo menos 1 observação.")
  }
  
  list(df_zoo = df_zoo, pre_period = pre.period, post_period = post.period, 
       deseasonalized = deseasonalize != "none", deseasonalize_method = deseasonalize)
}

#--------------------------------------------------
#  Função para desazonalizar a série temporal
#--------------------------------------------------
deseasonalize_series <- function(time_series, method = "stl", frequency = 12) {
  if (!is.ts(time_series)) {
    time_series <- ts(time_series, frequency = frequency)
  }
  if (length(time_series) < 2 * frequency) {
    warning("Série temporal muito curta para desazonalização confiável.")
    return(time_series)
  }
  # "stl" e "t+res" realizam a decomposição e removem o componente sazonal
  if (method == "stl" || method == "t+res") {
    decomp <- stats::stl(time_series, s.window = "periodic")
    deseasoned <- time_series - decomp$time.series[, "seasonal"]
    return(deseasoned)
  } else if (method == "trend_only") {
    decomp <- stats::stl(time_series, s.window = "periodic")
    trend_only <- decomp$time.series[, "trend"]
    return(trend_only)
  } else {
    return(time_series)
  }
}

#--------------------------------------------------
#  Função para modificar os gráficos gerados
#--------------------------------------------------
modify_plot <- function(p, title = "title", title_y = "title_y", data_inicio_evento = NULL) {
  if (is.null(data_inicio_evento)) {
    data_inicio_evento <- as.Date("2020-03-11")
  } else {
    data_inicio_evento <- as.Date(data_inicio_evento)
  }
  
  contrafactual_color <- "#000080"  # Azul marinho
  realizado_color <- "#B22222"      # Vermelho escuro
  sd_color <- "#4682B4"             # Azul aço
  
  if("lower" %in% names(p$data) && "upper" %in% names(p$data)){
    p <- p + geom_ribbon(aes(ymin = lower, ymax = upper, fill = "Intervalo de Confiança"), alpha = 0.15)
  }
  
  p <- p +
    geom_line(aes(y = point.pred, colour = "Contrafactual"), linewidth = 1.2) +
    geom_line(aes(y = response, colour = "Realizado"), linewidth = 1.2) +
    geom_vline(xintercept = as.numeric(data_inicio_evento), linetype = "dashed", colour = "darkgrey", linewidth = 1) +
    scale_colour_manual(values = c("Contrafactual" = contrafactual_color, "Realizado" = realizado_color), name = "") +
    scale_fill_manual(values = c("Intervalo de Confiança" = sd_color), name = "") +
    theme_minimal(base_size = 16) +
    labs(title = title) +
    scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
    theme(
      axis.text.x = element_text(size = 11, color = "black", angle = 45, hjust = 1),
      axis.text.y = element_text(size = 11, color = "black", hjust = 1),
      plot.title = element_text(size = 16, face = "bold", color = "black", hjust = 0.5),
      axis.title.y = element_text(size = 13, color = "black", vjust = -0.5),
      axis.title.x = element_text(size = 13, color = "black", margin = margin(t = 10)),
      panel.grid.major = element_line(color = "#ECECEC"),
      panel.grid.minor = element_line(color = "#F5F5F5"),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "bottom",
      legend.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
    ) +
    labs(y = title_y, x = "Data")
  
  p_plotly <- ggplotly(p, dynamicTicks = TRUE) %>%
    layout(
      autosize = TRUE,
      hovermode = "x unified",
      hoverlabel = list(
        bgcolor = "white",
        bordercolor = "gray80",
        font = list(family = "Arial", size = 12)
      ),
      legend = list(
        orientation = "h",
        x = 0.5, y = -0.15,
        xanchor = "center",
        yanchor = "top",
        bgcolor = "rgba(255,255,255,0.8)",
        bordercolor = "rgba(0,0,0,0.2)",
        font = list(family = "Arial", size = 12)
      ),
      shapes = list(
        list(
          type = "line",
          x0 = data_inicio_evento,
          x1 = data_inicio_evento,
          y0 = 0,
          y1 = 1,
          yref = "paper",
          line = list(color = "gray", dash = "dash", width = 1.5)
        )
      ),
      annotations = list(
        list(x = data_inicio_evento, y = 1, yref = "paper", 
             text = "Início do Evento", showarrow = TRUE, 
             arrowhead = 2, ax = 0, ay = -30,
             font = list(family = "Arial", size = 12, color = "black"))
      ),
      margin = list(l = 60, r = 20, t = 50, b = 60)
    ) %>%
    config(
      displayModeBar = TRUE,
      modeBarButtonsToRemove = list(
        "lasso2d", "select2d", "autoScale2d", "hoverClosestCartesian",
        "hoverCompareCartesian", "toggleSpikelines"
      ),
      toImageButtonOptions = list(
        format = "png",
        filename = "analise_causal",
        width = 1200,
        height = 700,
        scale = 2
      ),
      displaylogo = FALSE
    )
  
  return(p_plotly)
}

#------------------------------------------------------------------------
#    Função para traduzir o report (retorna string)
#------------------------------------------------------------------------
summary_pt_br <- function(ci_result, type = "summary") {
  if(type == "summary") {
    output_english <- capture.output(summary(ci_result))
    output_portuguese <- gsub("Posterior inference", "Inferência Posterior", output_english)
    output_portuguese <- gsub("Average", "Média", output_portuguese)
    output_portuguese <- gsub("Cumulative", "Acumulado", output_portuguese)
    output_portuguese <- gsub("Actual", "Real", output_portuguese)
    output_portuguese <- gsub("Prediction", "Projetado", output_portuguese)
    output_portuguese <- gsub("Absolute effect", "Efeito Absoluto", output_portuguese)
    output_portuguese <- gsub("Relative effect", "Efeito Relativo", output_portuguese)
    output_portuguese <- gsub("Posterior tail-area probability p", "Probabilidade do efeito ser de origem aleatória", output_portuguese)
    output_portuguese <- gsub("Posterior prob. of a causal effect", "Probabilidade do efeito ser causal", output_portuguese)
    output_portuguese <- gsub("For more details, type", "Para mais detalhes, digite", output_portuguese)
    paste(output_portuguese, collapse = "\n")
  } else if(type == "report") {
    output_english <- capture.output(summary(ci_result, "report"))
    output_formatted <- c()
    in_table <- FALSE
    table_content <- c()
    
    for (line in output_english) {
      if (grepl("^\\s*[A-Za-z]+\\s+[A-Za-z]+\\s+[A-Za-z]+\\s+|^\\s*-----", line)) {
        in_table <- TRUE
        table_content <- c(table_content, line)
        next
      }
      if (in_table && (line == "" || grepl("^[A-Za-z]", line) && !grepl("^\\s+[0-9]", line))) {
        in_table <- FALSE
        if (length(table_content) > 0) {
          table_formatted <- format_table(table_content)
          output_formatted <- c(output_formatted, table_formatted, "")
          table_content <- c()
        }
      }
      if (in_table) {
        table_content <- c(table_content, line)
        next
      }
      if (nchar(line) > 80) {
        words <- unlist(strsplit(line, " "))
        current_line <- ""
        for (word in words) {
          if (nchar(current_line) + nchar(word) + 1 > 80) {
            output_formatted <- c(output_formatted, current_line)
            current_line <- word
          } else {
            if (current_line == "") {
              current_line <- word
            } else {
              current_line <- paste(current_line, word)
            }
          }
        }
        if (current_line != "") {
          output_formatted <- c(output_formatted, current_line)
        }
      } else {
        output_formatted <- c(output_formatted, line)
      }
    }
    if (length(table_content) > 0) {
      table_formatted <- format_table(table_content)
      output_formatted <- c(output_formatted, table_formatted)
    }
    output_portuguese <- gsub("Analysis report", "RELATÓRIO DE ANÁLISE", output_formatted)
    output_portuguese <- gsub("Posterior inference", "INFERÊNCIA POSTERIOR", output_portuguese)
    output_portuguese <- gsub("Actual response", "Resposta Real", output_portuguese)
    output_portuguese <- gsub("Predicted response", "Resposta Projetada", output_portuguese)
    output_portuguese <- gsub("Absolute effect", "Efeito Absoluto", output_portuguese)
    output_portuguese <- gsub("Relative effect", "Efeito Relativo", output_portuguese)
    output_portuguese <- gsub("Posterior tail-area probability p", "Probabilidade do efeito ser de origem aleatória", output_portuguese)
    output_portuguese <- gsub("Posterior prob. of a causal effect", "Probabilidade do efeito ser causal", output_portuguese)
    output_portuguese <- gsub("Prior standard deviation", "Desvio padrão prior", output_portuguese)
    output_portuguese <- gsub("Posterior mean", "Média posterior", output_portuguese)
    output_portuguese <- gsub("Posterior standard deviation", "Desvio padrão posterior", output_portuguese)
    output_portuguese <- gsub("Posterior quantiles", "Quantis posteriores", output_portuguese)
    output_portuguese <- gsub("MCMC iterations", "Iterações MCMC", output_portuguese)
    output_portuguese <- gsub("Number of data points", "Número de pontos de dados", output_portuguese)
    output_portuguese <- gsub("during the post-intervention period", "durante o período pós-intervenção", output_portuguese)
    output_portuguese <- gsub("average", "média", output_portuguese)
    output_portuguese <- gsub("cumulative", "acumulado", output_portuguese)
    output_portuguese <- gsub("confidence interval", "intervalo de confiança", output_portuguese)
    output_portuguese <- gsub("Average", "Média", output_portuguese)
    output_portuguese <- gsub("Cumulative", "Acumulado", output_portuguese)
    
    formatted_report <- c(
      "============================================================",
      "               RELATÓRIO COMPLETO CAUSALIMPACT               ",
      "============================================================",
      "",
      output_portuguese
    )
    paste(formatted_report, collapse = "\n")
  }
}

# Função auxiliar para formatar tabelas
format_table <- function(table_lines) {
  if (length(table_lines) < 2) return(table_lines)
  headers <- table_lines[!grepl("^-+", table_lines) & !grepl("^\\s*$", table_lines)]
  formatted_lines <- c()
  for (line in headers) {
    cols <- unlist(strsplit(trimws(line), "\\s+"))
    if (length(cols) <= 1) {
      formatted_lines <- c(formatted_lines, line)
      next
    }
    if (length(cols) >= 4) {
      col1_width <- 20
      col2_width <- 15
      col3_width <- 15
      if (length(formatted_lines) == 0) {
        formatted_lines <- c(formatted_lines, paste(cols, collapse=" "))
      } else {
        if (length(cols) > 4) {
          first_part <- sprintf("%-*s %-*s %-*s", 
                                col1_width, substr(cols[1], 1, col1_width),
                                col2_width, substr(cols[2], 1, col2_width),
                                col3_width, substr(cols[3], 1, col3_width))
          second_part <- paste(cols[4:length(cols)], collapse=" ")
          formatted_lines <- c(formatted_lines, first_part)
          formatted_lines <- c(formatted_lines, paste("  ", second_part))
        } else {
          formatted_line <- sprintf("%-*s %-*s %-*s %s", 
                                    col1_width, substr(cols[1], 1, col1_width),
                                    col2_width, substr(cols[2], 1, col2_width),
                                    col3_width, substr(cols[3], 1, col3_width),
                                    if(length(cols) > 3) cols[4] else "")
          formatted_lines <- c(formatted_lines, formatted_line)
        }
      }
    } else {
      formatted_lines <- c(formatted_lines, line)
    }
  }
  if (length(formatted_lines) > 1) {
    separator <- paste(rep("-", 60), collapse="")
    formatted_lines <- c(formatted_lines[1], separator, formatted_lines[2:length(formatted_lines)])
  }
  return(formatted_lines)
}

#-----------------------------------------------------------------
#  Função para gerar tabela dos impactos agrupados por frequência
#-----------------------------------------------------------------
table_df_impact <- function(df_response, title, freq, data_inicio_evento) {
  df_response <- df_response %>% 
    mutate(Data = as.Date(rownames(df_response), format = "%Y-%m-%d")) %>%
    filter(Data >= as.Date(data_inicio_evento))
  df_grouped <- switch(freq,
                       "Mês" = df_response %>% mutate(Periodo = floor_date(Data, "month")),
                       "Trimestre" = df_response %>% mutate(Periodo = paste0(year(Data), "-Q", quarter(Data))),
                       "Semestre" = df_response %>% mutate(Periodo = ifelse(month(Data) <= 6, paste0(year(Data), "-S1"), paste0(year(Data), "-S2"))),
                       "Ano" = df_response %>% mutate(Periodo = as.character(year(Data)))
  )
  df_summary <- df_grouped %>% 
    group_by(Periodo) %>%
    summarize(
      Realizado = sum(response, na.rm = TRUE),
      Projetado = sum(point.pred, na.rm = TRUE),
      Impacto = sum(point.effect, na.rm = TRUE),
      `Impacto %` = ifelse(sum(point.pred, na.rm = TRUE) == 0, NA, 
                           (sum(point.effect, na.rm = TRUE) / sum(point.pred, na.rm = TRUE) * 100))
    ) %>%
    mutate(`Impacto Acumulado` = cumsum(Impacto)) %>%
    ungroup()
  df_summary <- df_summary %>%
    mutate(across(c(Realizado, Projetado, Impacto, `Impacto Acumulado`), 
                  ~format(round(.), big.mark = ".", decimal.mark = ",")),
           `Impacto %` = ifelse(is.na(`Impacto %`), "N/A", paste0(format(round(`Impacto %`, 1), nsmall=1), "%")))
  datatable(df_summary, options = list(pageLength = nrow(df_summary), dom = 'Bfrtip'), 
            rownames = FALSE,
            caption = htmltools::tags$caption(style = "caption-side: top; font-size: 1.0em; font-weight: bold;", title))
}

########################################################################
#                  UI do Shiny App
########################################################################
ui <- fluidPage(
  theme = shinytheme("flatly"),
  useShinyjs(),
  tags$head(
    tags$style(HTML("
      .well { background-color: #f8f9fa; }
      .nav-tabs { margin-bottom: 15px; }
      .info-box { padding: 10px; background-color: #e9ecef; border-radius: 5px; margin-bottom: 15px; }
      .bold-text { font-weight: bold; }
      .centered { text-align: center; }
      .footer { text-align: center; padding: 10px; margin-top: 20px; font-size: 0.8em; color: #6c757d; }
      .plot-container { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
      .summary-container { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 4px solid #007bff; 
        margin-top: 15px;
        font-family: 'Courier New', monospace;
      }
      .summary-container h4 {
        color: #007bff;
        margin-top: 0;
        margin-bottom: 15px;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 8px;
      }
      #report_summary, #report_summary_original {
        font-family: 'Courier New', monospace;
        line-height: 1.5;
        background-color: transparent;
        border: none;
        overflow-y: auto;
        max-height: 400px;
      }
      .univariate-notice {
        color: #3366cc;
        margin-bottom: 10px;
        font-weight: bold;
      }
      .developer-info {
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #dee2e6;
        font-size: 0.9em;
        color: #6c757d;
      }
    "))
  ),
  navbarPage(
    "Análise Causal com CausalImpact",
    tabPanel("Análise Principal",
             sidebarLayout(
               sidebarPanel(
                 width = 3,
                 h4("Carregar Dados"),
                 fileInput("datafile", "Carregar arquivo CSV ou XLSX", 
                           accept = c(".csv", ".xlsx")),
                 
                 conditionalPanel(
                   condition = "output.fileLoaded",
                   h4("Configuração das Colunas"),
                   uiOutput("dataColumnUI"),
                   uiOutput("colunasUI"),
                   
                   h4("Período de Análise"),
                   dateRangeInput("date_range", "Período da Série",
                                  start = Sys.Date() - 365, end = Sys.Date()),
                   dateInput("data_inicio_evento", "Data Início do Evento", 
                             value = "2020-07-01"),
                   
                   h4("Avançado"),
                   # Opções avançadas para alpha e configuração manual do BSTS
                   numericInput("alpha", "Nível de Significância (alpha)", 
                                value = 0.05, min = 0, max = 1, step = 0.01),
                   checkboxInput("manual_bsts", "Configurar BSTS manualmente", value = FALSE),
                   conditionalPanel(
                     condition = "input.manual_bsts == true",
                     tagList(
                       numericInput("niter", "Número de Iterações MCMC", 
                                    value = 1000, min = 100, step = 100),
                       numericInput("prior_level_sd", "Desvio Padrão do Prior do Nível", 
                                    value = 0.01, min = 0.001, step = 0.001)
                     )
                   ),
                   selectInput("deseasonalize", "Desazonalizar Série", 
                               choices = c("Não" = "none", 
                                           "Decomposição STL (Tendência + Resíduo)" = "t+res",
                                           "Somente Tendência" = "trend_only"),
                               selected = "none"),
                   conditionalPanel(
                     condition = "input.deseasonalize != 'none'",
                     tagList(
                       numericInput("freq_sazonal", "Frequência Sazonal", 
                                    value = 12, min = 1, max = 52),
                       helpText("12 para mensal (anual), 4 para trimestral, 7 para dias da semana")
                     )
                   ),
                   actionButton("analisar", "Executar Análise", 
                                class = "btn-primary", width = "100%"),
                   br(), br(),
                   conditionalPanel(
                     condition = "output.analysisComplete",
                     downloadButton("download_report", "Baixar Relatório")
                   )
                 ),
                 
                 conditionalPanel(
                   condition = "!output.fileLoaded",
                   div(class = "info-box",
                       h4("Instruções:"),
                       p("1. Faça o upload de um arquivo CSV ou XLSX contendo seus dados."),
                       p("2. Selecione a coluna de data e as variáveis para análise."),
                       p("3. Defina o período de análise e a data do evento."),
                       p("4. Clique em 'Executar Análise' para gerar resultados.")
                   )
                 )
               ),
               mainPanel(
                 width = 9,
                 conditionalPanel(
                   condition = "output.fileLoaded && !output.analysisComplete",
                   div(class = "centered", 
                       h3("Dados carregados. Configure os parâmetros e execute a análise."))
                 ),
                 conditionalPanel(
                   condition = "output.analysisComplete",
                   tabsetPanel(
                     tabPanel("Visualização", 
                              # Exibe resumo e gráfico dos coeficientes antes dos gráficos
                              fluidRow(
                                column(12,
                                       h4("Resumo da Análise"),
                                       verbatimTextOutput("impact_summary"),
                                       h4("Coeficientes do Modelo BSTS"),
                                       uiOutput("modelo_info"),  # Adicionado aqui o aviso de modelo univariado
                                       plotOutput("coefficients_plot")
                                )
                              ),
                              tabsetPanel(
                                tabPanel("Gráfico Interativo",
                                         div(class = "plot-container",
                                             plotlyOutput("plot_causal", height = "600px", width = "100%"),
                                             uiOutput("plot_info"))
                                ),
                                tabPanel("Gráfico Original",
                                         div(class = "plot-container",
                                             plotOutput("plot_original", height = "800px", width = "100%"))
                                )
                              ),
                              br()
                     ),
                     tabPanel("Tabela de Impacto", 
                              fluidRow(
                                column(4, selectInput("impact_freq", "Agrupar Impactos por", 
                                                      choices = c("Mês", "Trimestre", "Semestre", "Ano"), 
                                                      selected = "Trimestre")),
                                column(8, div(style = "text-align: right; padding-top: 20px;",
                                              downloadButton("download_table", "Exportar Tabela")))
                              ),
                              DTOutput("tabela_impacto")
                     ),
                     tabPanel("Relatório Completo", 
                              verbatimTextOutput("report_detailed"),
                              downloadButton("download_text_report", "Baixar Texto")),
                     tabPanel("Análise Adicional", 
                              fluidRow(
                                column(4, 
                                       uiOutput("plotColsUI")),
                                column(8,
                                       plotlyOutput("plot_adicional", height = "600px"))
                              ),
                              br(),
                              h4("Análise Descritiva"),
                              uiOutput("descritiva_periodo_info"),
                              DTOutput("tabela_descritiva"),
                              br(),
                              h4("Tabela de Dados"),
                              DTOutput("tabela_carregada")
                     )
                   )
                 ),
                 conditionalPanel(
                   condition = "!output.fileLoaded",
                   div(class = "centered", 
                       img(src = "https://www.r-project.org/logo/Rlogo.png", height = 100),
                       h3("Bem-vindo à Análise Causal com CausalImpact"),
                       p("Carregue um arquivo para começar sua análise."),
                       # Adicionada a imagem do gato logo abaixo do texto de boas-vindas
                       img(src = "https://i.etsystatic.com/37145909/r/il/a1da3a/4209674353/il_fullxfull.4209674353_87un.jpg", height = 500))
                 )
               )
             )
    ),
    tabPanel("Ajuda",
             fluidRow(
               column(12,
                      h2("Como usar esta ferramenta"),
                      h3("Sobre o CausalImpact"),
                      p("O CausalImpact é uma biblioteca desenvolvida pelo Google para análise de efeitos causais em séries temporais. 
                        Ela permite estimar o impacto de uma intervenção ou evento em uma variável ao longo do tempo."),
                      
                      h3("Requisitos dos dados"),
                      tags$ul(
                        tags$li("Uma coluna de data em formato YYYY-MM-DD"),
                        tags$li("Uma variável resposta (Y) que será analisada"),
                        tags$li("Variáveis explicativas (Xs) que ajudarão a construir o modelo contrafactual"),
                        tags$li("Períodos suficientes antes e depois do evento para análise")
                      ),
                      
                      h3("Interpretação dos resultados"),
                      p("Os principais elementos para análise são:"),
                      tags$ul(
                        tags$li(strong("Gráfico:"), " Mostra a série realizada vs. contrafactual"),
                        tags$li(strong("Efeito Absoluto:"), " A diferença entre o realizado e o contrafactual"),
                        tags$li(strong("Efeito Relativo:"), " O efeito percentual em relação ao contrafactual"),
                        tags$li(strong("Probabilidade causal:"), " A probabilidade de que o efeito observado seja causal e não aleatório")
                      ),
                      
                      h3("Desazonalização"),
                      p("A ferramenta oferece três métodos de transformação de séries temporais:"),
                      tags$ul(
                        tags$li(strong("Decomposição STL (Tendência + Resíduo):"), " Realiza a decomposição e remove a sazonalidade, mantendo a tendência e o resíduo"),
                        tags$li(strong("Somente Tendência:"), " Mantém apenas o componente de tendência, removendo a sazonalidade e o ruído"),
                        tags$li(strong("Não:"), " Sem transformação")
                      ),
                      p("A frequência sazonal deve ser definida de acordo com a periodicidade dos dados:"),
                      tags$ul(
                        tags$li("12 para dados mensais (padrão)"),
                        tags$li("4 para dados trimestrais"),
                        tags$li("7 para dados diários com padrão semanal"),
                        tags$li("52 para dados semanais com padrão anual")
                      ),
                      
                      h3("Configuração Avançada"),
                      p("Você pode configurar o nível de significância (alpha) e, se desejar, ajustar manualmente os parâmetros do modelo BSTS."),
                      
                      h3("Metodologia"),
                      p("A análise utiliza um modelo bayesiano de séries temporais estruturais para estimar o impacto causal da intervenção."),
                      
                      h3("Referências"),
                      tags$ul(
                        tags$li(HTML('<a href="https://google.github.io/CausalImpact/CausalImpact.html" target="_blank">Documentação oficial do CausalImpact</a>')),
                        tags$li(HTML('<a href="https://doi.org/10.1214/14-AOAS788" target="_blank">Brodersen et al. (2015)</a>'))
                      ),
                      
                      # Adicionando informações do desenvolvedor e licença
                      h3("Desenvolvedor"),
                      p("Desenvolvido por Silvio Paula"),
                      p("Email: Silvio.economia@gmail.com"),
                      
                      h3("Licença"),
                      tags$pre(style = "background-color: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap;",
                               "MIT License
Copyright (c) 2025 Silvio da Rosa Paula
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.")
               )
             )
    )
  )
)

########################################################################
#                 Server do Shiny App
########################################################################
server <- function(input, output, session) {
  
  rv <- reactiveValues(
    data = NULL,
    analysis_result = NULL,
    analysis_complete = FALSE,
    ci_metrics = NULL
  )
  
  observeEvent(input$datafile, {
    withProgress(message = 'Carregando dados...', value = 0.5, {
      tryCatch({
        ext <- tools::file_ext(input$datafile$name)
        if(ext == "csv"){
          df <- fread(input$datafile$datapath, data.table = FALSE)
          date_cols <- names(df)[sapply(df, function(x) any(grepl("^\\d{4}-\\d{2}-\\d{2}$", as.character(x))))]
          if(length(date_cols) > 0) {
            df[[date_cols[1]]] <- as.Date(df[[date_cols[1]]])
          }
        } else if(ext == "xlsx"){
          df <- read_excel(input$datafile$datapath)
          date_cols <- names(df)[sapply(df, function(x) lubridate::is.Date(x) || any(grepl("^\\d{4}-\\d{2}-\\d{2}$", as.character(x))))]
          if(length(date_cols) > 0) {
            df[[date_cols[1]]] <- as.Date(df[[date_cols[1]]])
          }
        } else {
          stop("Arquivo inválido. Carregue um CSV ou XLSX.")
        }
        rv$data <- df
        rv$analysis_complete <- FALSE
        showNotification("Dados carregados com sucesso", type = "message")
      }, error = function(e) {
        showNotification(paste("Erro ao carregar arquivo:", e$message), type = "error", duration = 10)
      })
    })
  })
  
  output$fileLoaded <- reactive({ !is.null(rv$data) })
  outputOptions(output, "fileLoaded", suspendWhenHidden = FALSE)
  
  output$analysisComplete <- reactive({ rv$analysis_complete })
  outputOptions(output, "analysisComplete", suspendWhenHidden = FALSE)
  
  output$dataColumnUI <- renderUI({
    req(rv$data)
    df <- rv$data
    date_cols <- names(df)[sapply(df, function(x) lubridate::is.Date(x) || any(grepl("^\\d{4}-\\d{2}-\\d{2}$", as.character(x))))]
    if(length(date_cols) == 0) { date_cols <- names(df) }
    selectInput("data_column", "Coluna de Data", choices = names(df), 
                selected = if(length(date_cols) > 0) date_cols[1] else names(df)[1])
  })
  
  output$colunasUI <- renderUI({
    df <- rv$data
    req(df, input$data_column)
    vars <- setdiff(names(df), input$data_column)
    num_vars <- names(df)[sapply(df, is.numeric) & names(df) != input$data_column]
    if(length(num_vars) == 0) { num_vars <- vars }
    tagList(
      selectInput("var_Y", "Variável Resposta (Y)", choices = vars, 
                  selected = if(length(num_vars) > 0) num_vars[1] else vars[1]),
      pickerInput("var_Xs", "Variáveis Explicativas (Xs)", choices = vars, 
                  multiple = TRUE, options = list(`live-search` = TRUE))
    )
  })
  
  # Adicionado o renderUI para o aviso de modelo univariado
  output$modelo_info <- renderUI({
    req(rv$analysis_complete)
    if(length(input$var_Xs) == 0) {
      div(class = "univariate-notice", "Modelo univariado definido.")
    } else {
      return(NULL)
    }
  })
  
  output$plotColsUI <- renderUI({
    df <- dados_processados()
    req(df, input$data_column)
    num_vars <- names(df)[sapply(df, is.numeric) & names(df) != "Data"]
    pickerInput("plot_cols", "Colunas para Análise", choices = num_vars, 
                multiple = TRUE, options = list(`live-search` = TRUE))
  })
  
  output$tabela_carregada <- renderDT({
    df <- rv$data
    req(df)
    if(!is.null(input$search_data) && input$search_data != "") {
      search_text <- tolower(input$search_data)
      df <- df %>% filter(across(everything(), ~grepl(search_text, tolower(as.character(.)), fixed = TRUE)))
    }
    datatable(df, options = list(pageLength = 10, dom = 'Blfrtip', scrollX = TRUE, buttons = c('copy', 'csv', 'excel')))
  })
  
  dados_processados <- reactive({
    df <- rv$data
    req(df, input$data_column)
    df_processed <- df
    df_processed[[input$data_column]] <- as.Date(df_processed[[input$data_column]])
    names(df_processed)[names(df_processed) == input$data_column] <- "Data"
    df_processed %>% arrange(Data)
  })
  
  observeEvent({dados_processados(); input$var_Y}, {
    df <- dados_processados()
    req(input$var_Y)
    if("Data" %in% names(df) && input$var_Y %in% names(df)){
      min_date <- min(df$Data[!is.na(df[[input$var_Y]])], na.rm = TRUE)
      max_date <- max(df$Data[!is.na(df[[input$var_Y]])], na.rm = TRUE)
      if(!is.infinite(min_date) && !is.infinite(max_date)) {
        updateDateRangeInput(session, "date_range", start = min_date, end = max_date)
      }
    }
  })
  
  observeEvent(input$analisar, {
    withProgress(message = 'Executando análise...', value = 0, {
      tryCatch({
        incProgress(0.2, detail = "Preparando dados")
        df <- dados_processados()
        req(input$var_Y)
        # Removida a exigência de pelo menos uma covariável
        colunas_importantes <- c("Data", input$var_Y, input$var_Xs)
        df_clean <- df %>% drop_na(any_of(colunas_importantes))
        if(nrow(df_clean) < 10) {
          showNotification("Dados insuficientes após remoção de valores ausentes", type = "error")
          return()
        }
        incProgress(0.4, detail = "Processando modelo causal")
        res <- tryCatch({
          generate_df_causal_impact(
            data_frame = df_clean,
            data_inicio = input$date_range[1],
            data_fim = input$date_range[2],
            data_inicio_evento = input$data_inicio_evento,
            var_Y = input$var_Y,
            var_Xs = input$var_Xs,
            deseasonalize = input$deseasonalize,
            freq_sazonal = input$freq_sazonal
          )
        }, error = function(e) {
          showNotification(paste("Erro ao gerar análise:", e$message), type = "error")
          return(NULL)
        })
        if(is.null(res)) return()
        incProgress(0.7, detail = "Finalizando")
        # Configuração manual do BSTS, se selecionada
        if (input$manual_bsts) {
          ci_result <- tryCatch({
            CausalImpact(res$df_zoo, res$pre_period, res$post_period,
                         model.args = list(niter = input$niter, prior.level.sd = input$prior_level_sd))
          }, error = function(e) {
            showNotification(paste("Erro no CausalImpact:", e$message), type = "error")
            return(NULL)
          })
        } else {
          ci_result <- tryCatch({
            CausalImpact(res$df_zoo, res$pre_period, res$post_period)
          }, error = function(e) {
            showNotification(paste("Erro no CausalImpact:", e$message), type = "error")
            return(NULL)
          })
        }
        if(is.null(ci_result)) return()
        rv$analysis_result <- list(
          ci_result = ci_result, 
          df_clean = df_clean, 
          res = res,
          var_Y = input$var_Y,
          data_inicio_evento = input$data_inicio_evento
        )
        tryCatch({
          ci_summary <- summary(ci_result)  # alpha removido, pois não é aceito
          avg_actual <- if(!is.null(ci_summary$average$actual)) ci_summary$average$actual else NA
          avg_pred <- if(!is.null(ci_summary$average$pred)) ci_summary$average$pred else NA
          avg_abs_effect <- if(!is.null(ci_summary$average$abs.effect)) ci_summary$average$abs.effect else NA
          avg_rel_effect <- if(!is.null(ci_summary$average$rel.effect)) ci_summary$average$rel.effect else NA
          p_value <- if(!is.null(ci_summary$p.value)) ci_summary$p.value else NA
          rv$ci_metrics <- list(
            avg_actual = avg_actual,
            avg_pred = avg_pred,
            avg_abs_effect = avg_abs_effect,
            avg_rel_effect = avg_rel_effect,
            p_value = p_value,
            prob_causal = if(!is.na(p_value)) 1 - p_value else NA
          )
        }, error = function(e) {
          showNotification(paste("Erro ao extrair métricas:", e$message), type = "warning")
          rv$ci_metrics <- list(
            avg_actual = NA,
            avg_pred = NA,
            avg_abs_effect = NA,
            avg_rel_effect = NA,
            p_value = NA,
            prob_causal = NA
          )
        })
        rv$analysis_complete <- TRUE
        showNotification("Análise concluída com sucesso", type = "message")
      }, error = function(e) {
        showNotification(paste("Erro na análise:", e$message), type = "error", duration = 10)
      })
    })
  })
  
  output$plot_causal <- renderPlotly({
    req(rv$analysis_complete, rv$analysis_result)
    ci_result <- rv$analysis_result$ci_result
    df_series <- as.data.frame(ci_result$series)
    df_series$Data <- as.Date(index(ci_result$series))
    p <- ggplot(df_series, aes(x = Data)) +
      labs(title = paste("Análise de Impacto Causal -", rv$analysis_result$var_Y), 
           y = rv$analysis_result$var_Y, 
           x = "Data")
    modify_plot(p, 
                title = paste("Análise de Impacto Causal -", rv$analysis_result$var_Y), 
                title_y = rv$analysis_result$var_Y, 
                data_inicio_evento = input$data_inicio_evento)
  })
  
  output$plot_original <- renderPlot({
    req(rv$analysis_complete, rv$analysis_result)
    ci_result <- rv$analysis_result$ci_result
    par(mar = c(5, 4, 4, 2) + 0.1, cex.main = 1.2, cex.lab = 1.1, cex.axis = 1)
    plot(ci_result)
  }, height = 800, width = 1000)
  
  output$plot_info <- renderUI({
    req(rv$analysis_complete, rv$ci_metrics)
    desazonalizado_texto <- ""
    if (!is.null(rv$analysis_result$res$deseasonalized) && rv$analysis_result$res$deseasonalized) {
      metodo <- switch(rv$analysis_result$res$deseasonalize_method,
                       "t+res" = "Decomposição STL (Tendência + Resíduo)",
                       "trend_only" = "Somente Tendência",
                       "Desconhecido")
      desazonalizado_texto <- paste0("<br>Série transformada usando: ", metodo)
    }
    div(
      style = "margin-top: 10px; font-style: italic; text-align: center;",
      HTML(paste0(
        "Período pré-evento: ", format(input$date_range[1], "%d/%m/%Y"), 
        " a ", format(as.Date(input$data_inicio_evento)-1, "%d/%m/%Y"), "<br>",
        "Período pós-evento: ", format(input$data_inicio_evento, "%d/%m/%Y"), 
        " a ", format(input$date_range[2], "%d/%m/%Y"),
        desazonalizado_texto
      ))
    )
  })
  
  output$impact_summary <- renderPrint({
    req(rv$analysis_complete, rv$analysis_result)
    summary(rv$analysis_result$ci_result)
  })
  
  # Função corrigida para não tentar renderizar coeficientes em modelo univariado
  output$coefficients_plot <- renderPlot({
    req(rv$analysis_complete, rv$analysis_result)
    
    # Verificar se há covariáveis
    if(length(input$var_Xs) == 0) {
      # Mostrar um plot vazio com uma mensagem informativa
      plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "", 
           main = "Não há coeficientes para mostrar em modelo univariado")
      text(0, 0, "Modelo sem covariáveis não possui coeficientes para visualizar.", 
           cex = 1.2, col = "darkblue")
    } else {
      # Mostrar os coeficientes normalmente
      plot(rv$analysis_result$ci_result$model$bsts.model, "coefficients")
    }
  })
  
  output$report_summary <- renderText({
    req(rv$analysis_complete, rv$analysis_result)
    summary_pt_br(rv$analysis_result$ci_result, type = "summary")
  })
  
  output$report_summary_original <- renderText({
    req(rv$analysis_complete, rv$analysis_result)
    summary_pt_br(rv$analysis_result$ci_result, type = "summary")
  })
  
  output$report_detailed <- renderText({
    req(rv$analysis_complete, rv$analysis_result)
    summary_pt_br(rv$analysis_result$ci_result, type = "report")
  })
  
  output$tabela_impacto <- renderDT({
    req(rv$analysis_complete, rv$analysis_result)
    ci_result <- rv$analysis_result$ci_result
    df_response <- as.data.frame(ci_result$series)
    table_df_impact(df_response, 
                    title = paste("Impacto em", rv$analysis_result$var_Y, "por", input$impact_freq), 
                    freq = input$impact_freq,
                    data_inicio_evento = rv$analysis_result$data_inicio_evento)
  })
  
  # Plot interativo para a aba Análise Adicional
  output$plot_adicional <- renderPlotly({
    df <- dados_processados()
    req(input$plot_cols, length(input$plot_cols) > 0, input$data_inicio_evento)
    req("Data" %in% names(df))
    
    # Filter data based on date range
    df_filtered <- df %>% 
      filter(Data >= input$date_range[1] & Data <= input$date_range[2]) %>%
      select(Data, all_of(input$plot_cols))
    
    # Criar gráfico de linha básico
    p <- ggplot(df_filtered, aes(x = Data, y = .data[[input$plot_cols[1]]])) +
      geom_line(linewidth = 1.2, color = "#B22222") +
      geom_vline(xintercept = as.Date(input$data_inicio_evento), 
                 linetype = "dashed", colour = "darkgrey", linewidth = 1) +
      theme_minimal(base_size = 16) +
      labs(title = paste("Série Temporal -", input$plot_cols[1]),
           y = input$plot_cols[1], 
           x = "Data") +
      scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
      theme(
        axis.text.x = element_text(size = 11, color = "black", angle = 45, hjust = 1),
        axis.text.y = element_text(size = 11, color = "black", hjust = 1),
        plot.title = element_text(size = 16, face = "bold", color = "black", hjust = 0.5),
        axis.title.y = element_text(size = 13, color = "black", vjust = -0.5),
        axis.title.x = element_text(size = 13, color = "black", margin = margin(t = 10)),
        panel.grid.major = element_line(color = "#ECECEC"),
        panel.grid.minor = element_line(color = "#F5F5F5"),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
      )
    
    # Converter para plotly
    p_plotly <- ggplotly(p, dynamicTicks = TRUE) %>%
      layout(
        autosize = TRUE,
        hovermode = "x unified",
        hoverlabel = list(
          bgcolor = "white",
          bordercolor = "gray80",
          font = list(family = "Arial", size = 12)
        ),
        shapes = list(
          list(
            type = "line",
            x0 = input$data_inicio_evento,
            x1 = input$data_inicio_evento,
            y0 = 0,
            y1 = 1,
            yref = "paper",
            line = list(color = "gray", dash = "dash", width = 1.5)
          )
        ),
        annotations = list(
          list(x = input$data_inicio_evento, y = 1, yref = "paper", 
               text = "Início do Evento", showarrow = TRUE, 
               arrowhead = 2, ax = 0, ay = -30,
               font = list(family = "Arial", size = 12, color = "black"))
        ),
        margin = list(l = 60, r = 20, t = 50, b = 60)
      ) %>%
      config(
        displayModeBar = TRUE,
        modeBarButtonsToRemove = list(
          "lasso2d", "select2d", "autoScale2d", "hoverClosestCartesian",
          "hoverCompareCartesian", "toggleSpikelines"
        ),
        toImageButtonOptions = list(
          format = "png",
          filename = "serie_temporal",
          width = 1200,
          height = 700,
          scale = 2
        ),
        displaylogo = FALSE
      )
    
    return(p_plotly)
  })
  
  # Informações sobre os períodos
  output$descritiva_periodo_info <- renderUI({
    req(input$data_inicio_evento, input$date_range)
    div(
      style = "margin-bottom: 15px; font-style: italic;",
      HTML(paste0(
        "Período pré-intervenção: ", format(input$date_range[1], "%d/%m/%Y"), 
        " a ", format(as.Date(input$data_inicio_evento)-1, "%d/%m/%Y"), "<br>",
        "Período pós-intervenção: ", format(input$data_inicio_evento, "%d/%m/%Y"), 
        " a ", format(input$date_range[2], "%d/%m/%Y")
      ))
    )
  })
  
  # Tabela de análise descritiva por período - VERSÃO SIMPLIFICADA
  output$tabela_descritiva <- renderDT({
    # Obter os dados
    df <- dados_processados()
    req(input$plot_cols, length(input$plot_cols) > 0)
    req("Data" %in% names(df), input$data_inicio_evento)
    
    # Filtrar pelo intervalo de datas
    df_filtered <- df %>% 
      filter(Data >= input$date_range[1] & Data <= input$date_range[2]) %>%
      select(Data, all_of(input$plot_cols))
    
    # Dividir em períodos pré e pós evento
    data_evento <- as.Date(input$data_inicio_evento)
    df_pre <- df_filtered %>% filter(Data < data_evento)
    df_post <- df_filtered %>% filter(Data >= data_evento)
    
    # Função simples para calcular estatísticas básicas
    get_stats <- function(data, varname) {
      if(nrow(data) == 0) {
        return(data.frame(
          Variável = varname,
          N = 0,
          Média = NA,
          Mediana = NA,
          Mínimo = NA,
          Máximo = NA,
          DP = NA
        ))
      }
      
      data.frame(
        Variável = varname,
        N = nrow(data),
        Média = mean(data[[varname]], na.rm = TRUE),
        Mediana = median(data[[varname]], na.rm = TRUE),
        Mínimo = min(data[[varname]], na.rm = TRUE),
        Máximo = max(data[[varname]], na.rm = TRUE),
        DP = sd(data[[varname]], na.rm = TRUE)
      )
    }
    
    # Criar dataframe de resultados
    resultados <- data.frame()
    
    # Para cada variável selecionada
    for (col in input$plot_cols) {
      # Calcular estatísticas para cada período
      stats_pre <- get_stats(df_pre, col) %>% mutate(Período = "Pré-Intervenção")
      stats_post <- get_stats(df_post, col) %>% mutate(Período = "Pós-Intervenção")
      stats_total <- get_stats(df_filtered, col) %>% mutate(Período = "Total")
      
      # Combinar resultados
      resultados <- bind_rows(resultados, stats_pre, stats_post, stats_total)
    }
    
    # Formatação para exibição
    resultados_formatados <- resultados %>%
      mutate(across(c(Média, Mediana, Mínimo, Máximo, DP), 
                    ~ifelse(is.na(.), "N/A", format(round(., 2), nsmall = 2, big.mark = ".", decimal.mark = ","))))
    
    # Exibir tabela
    datatable(resultados_formatados, 
              options = list(pageLength = nrow(resultados_formatados), dom = 't'), 
              rownames = FALSE,
              caption = htmltools::tags$caption(
                style = "caption-side: top; font-size: 1.0em; font-weight: bold;", 
                "Estatísticas Descritivas por Período"
              ))
  })
  
  output$download_data <- downloadHandler(
    filename = function() {
      paste("dados_", format(Sys.time(), "%Y%m%d_%H%M"), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(rv$data, file, row.names = FALSE)
    }
  )
  
  output$download_table <- downloadHandler(
    filename = function() {
      paste("impacto_", rv$analysis_result$var_Y, "_", format(Sys.time(), "%Y%m%d"), ".csv", sep = "")
    },
    content = function(file) {
      ci_result <- rv$analysis_result$ci_result
      df_response <- as.data.frame(ci_result$series)
      df_response <- df_response %>% 
        mutate(Data = as.Date(rownames(df_response), format = "%Y-%m-%d")) %>%
        filter(Data >= as.Date(rv$analysis_result$data_inicio_evento))
      df_grouped <- switch(input$impact_freq,
                           "Mês" = df_response %>% mutate(Periodo = floor_date(Data, "month")),
                           "Trimestre" = df_response %>% mutate(Periodo = paste0(year(Data), "-Q", quarter(Data))),
                           "Semestre" = df_response %>% mutate(Periodo = ifelse(month(Data) <= 6, paste0(year(Data), "-S1"), paste0(year(Data), "-S2"))),
                           "Ano" = df_response %>% mutate(Periodo = as.character(year(Data)))
      )
      df_summary <- df_grouped %>% 
        group_by(Periodo) %>%
        summarize(
          Realizado = sum(response, na.rm = TRUE),
          Projetado = sum(point.pred, na.rm = TRUE),
          Impacto = sum(point.effect, na.rm = TRUE),
          Impacto_Percentual = ifelse(sum(point.pred, na.rm = TRUE) == 0, NA, (sum(point.effect, na.rm = TRUE) / sum(point.pred, na.rm = TRUE) * 100))
        ) %>%
        mutate(Impacto_Acumulado = cumsum(Impacto)) %>%
        ungroup()
      write.csv(df_summary, file, row.names = FALSE)
    }
  )
  
  output$download_text_report <- downloadHandler(
    filename = function() {
      paste("relatorio_causal_impact_", format(Sys.time(), "%Y%m%d"), ".txt", sep = "")
    },
    content = function(file) {
      resumo <- summary_pt_br(rv$analysis_result$ci_result, type = "summary")
      detalhado <- summary_pt_br(rv$analysis_result$ci_result, type = "report")
      conteudo <- paste0(
        "# RESUMO DA ANÁLISE\n",
        "====================\n\n",
        resumo,
        "\n\n\n",
        "# RELATÓRIO DETALHADO\n",
        "=====================\n\n",
        detalhado,
        "\n\n\n",
        "# INFORMAÇÕES ADICIONAIS\n",
        "=====================\n\n",
        "Desenvolvido por Silvio da Rosa Paula (Silvio.economia@gmail.com)\n",
        "MIT License - Copyright (c) 2025 Silvio da Rosa Paula\n",
        "Para detalhes completos da licença, consulte a aba Ajuda no aplicativo."
      )
      cat(conteudo, file = file)
    }
  )
  
  # Botão de download do relatório em XLSX com todas as visões
  output$download_report <- downloadHandler(
    filename = function() {
      paste("relatorio_causal_impact_", format(Sys.time(), "%Y%m%d"), ".xlsx", sep = "")
    },
    content = function(file) {
      wb <- createWorkbook()
      # Aba Resumo
      resumo <- summary_pt_br(rv$analysis_result$ci_result, type = "summary")
      addWorksheet(wb, "Resumo")
      writeData(wb, sheet = "Resumo", x = data.frame(Resumo = resumo))
      
      # Aba Relatório Detalhado
      detalhado <- summary_pt_br(rv$analysis_result$ci_result, type = "report")
      addWorksheet(wb, "Relatorio Detalhado")
      writeData(wb, sheet = "Relatorio Detalhado", x = data.frame(Relatorio = detalhado))
      
      # Função para gerar os dados de impacto por agrupamento
      gerar_impacto <- function(freq) {
        ci_result <- rv$analysis_result$ci_result
        df_response <- as.data.frame(ci_result$series)
        df_response <- df_response %>% 
          mutate(Data = as.Date(rownames(df_response), format = "%Y-%m-%d")) %>%
          filter(Data >= as.Date(rv$analysis_result$data_inicio_evento))
        df_grouped <- switch(freq,
                             "Mês" = df_response %>% mutate(Periodo = floor_date(Data, "month")),
                             "Trimestre" = df_response %>% mutate(Periodo = paste0(year(Data), "-Q", quarter(Data))),
                             "Semestre" = df_response %>% mutate(Periodo = ifelse(month(Data) <= 6, paste0(year(Data), "-S1"), paste0(year(Data), "-S2"))),
                             "Ano" = df_response %>% mutate(Periodo = as.character(year(Data)))
        )
        df_summary <- df_grouped %>% 
          group_by(Periodo) %>%
          summarize(
            Realizado = sum(response, na.rm = TRUE),
            Projetado = sum(point.pred, na.rm = TRUE),
            Impacto = sum(point.effect, na.rm = TRUE),
            `Impacto %` = ifelse(sum(point.pred, na.rm = TRUE) == 0, NA,
                                 (sum(point.effect, na.rm = TRUE) / sum(point.pred, na.rm = TRUE) * 100))
          ) %>%
          mutate(`Impacto Acumulado` = cumsum(Impacto)) %>%
          ungroup()
        df_summary <- df_summary %>%
          mutate(across(c(Realizado, Projetado, Impacto, `Impacto Acumulado`), 
                        ~format(round(.), big.mark = ".", decimal.mark = ",")),
                 `Impacto %` = ifelse(is.na(`Impacto %`), "N/A", paste0(format(round(`Impacto %`, 1), nsmall=1), "%")))
        return(df_summary)
      }
      
      # Aba Impacto Mensal
      impacto_mensal <- gerar_impacto("Mês")
      addWorksheet(wb, "Impacto Mensal")
      writeData(wb, sheet = "Impacto Mensal", x = impacto_mensal)
      
      # Aba Impacto Trimestral
      impacto_trimestral <- gerar_impacto("Trimestre")
      addWorksheet(wb, "Impacto Trimestral")
      writeData(wb, sheet = "Impacto Trimestral", x = impacto_trimestral)
      
      # Aba Impacto Semestral
      impacto_semestral <- gerar_impacto("Semestre")
      addWorksheet(wb, "Impacto Semestral")
      writeData(wb, sheet = "Impacto Semestral", x = impacto_semestral)
      
      # Aba Impacto Anual
      impacto_anual <- gerar_impacto("Ano")
      addWorksheet(wb, "Impacto Anual")
      writeData(wb, sheet = "Impacto Anual", x = impacto_anual)
      
      # Aba Métricas
      metrics <- rv$ci_metrics
      metrics_df <- as.data.frame(t(unlist(metrics)))
      addWorksheet(wb, "Metricas")
      writeData(wb, sheet = "Metricas", x = metrics_df)
      
      # Adicionar informações de autoria
      addWorksheet(wb, "Informações")
      info_df <- data.frame(
        Informação = c("Desenvolvido por", "Email", "Licença"),
        Valor = c("Silvio da Rosa Paula", 
                  "Silvio.economia@gmail.com", 
                  "MIT License - Copyright (c) 2025 Silvio da Rosa Paula")
      )
      writeData(wb, sheet = "Informações", x = info_df)
      
      saveWorkbook(wb, file, overwrite = TRUE)
    }
  )
}

########################################################################
#                 Inicialização do App
########################################################################
shinyApp(ui, server)