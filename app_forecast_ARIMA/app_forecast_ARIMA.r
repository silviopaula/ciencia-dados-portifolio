#======================================================================
###                  SHINNY APP AUTO.ARIMA                          ###
#======================================================================

# Instalar e carregar packages
if (!require(pacman)) install.packages("pacman")
p_load(shiny, shinydashboard, tidyverse, lubridate, plotly, readxl, DT, 
       forecast, writexl, shinyWidgets, shinythemes, RColorBrewer, zoo, stringr, shinyjs)

# Aumentar limite máximo de upload de arquivos para 100 MB
options(shiny.maxRequestSize = 100 * 1024^5)

# Caminho do arquivo de exemplo (corrigido)
EXAMPLE_DATA_PATH <- normalizePath("www/dados.xlsx", mustWork = FALSE)

#======================================================================
#   Funções
#======================================================================
{
  
  #======================================================================
  ###                  Função para checar collineariedade             ###
  #======================================================================
  
  # Função para verificar colinearidade entre variáveis
  check_collinearity <- function(data, variables, threshold = 0.99) {
    if (length(variables) <= 1) {
      return(FALSE)  # Não há colinearidade com uma única variável
    }
    
    # Extrair apenas as variáveis de interesse
    data_subset <- data[, variables, drop = FALSE]
    
    # Remover linhas com NA
    data_subset <- na.omit(data_subset)
    
    # Se não houver dados suficientes após remover NAs, não podemos verificar
    if (nrow(data_subset) < 5) {
      return(TRUE)  # Assumir pior cenário
    }
    
    # Calcular matriz de correlação
    cor_matrix <- cor(data_subset, use = "pairwise.complete.obs")
    
    # Verificar se algum par de variáveis tem correlação acima do threshold
    diag(cor_matrix) <- 0  # Ignorar a diagonal (correlação de uma variável com ela mesma)
    return(any(abs(cor_matrix) > threshold))
  }
  
  #======================================================================
  ###                  Função para ARIMA                             ###
  #======================================================================
  run_arima_models_extended <- function(df, 
                                        target_var, 
                                        exog_vars = NULL, 
                                        data_inicial, 
                                        data_atual, 
                                        end_predict = NULL,
                                        n_teste = 3, 
                                        seasonal = TRUE, 
                                        period = 12, 
                                        num_models = 10, 
                                        manual_mode = FALSE,
                                        arima_params = NULL,
                                        seed = 123) {
    
    set.seed(seed)
    start_time <- Sys.time()
    
    cat("Iniciando análise ARIMA\n")
    
    # Garantir que datas sejam objetos de tipo Date
    df$Data <- as.Date(df$Data)
    data_inicial <- as.Date(data_inicial)
    data_atual <- as.Date(data_atual)
    
    # Padronizar todas as datas para o primeiro dia do mês
    standardize_date <- function(date) {
      if(length(date) == 1) {
        if(is.na(date)) return(NA)
        return(as.Date(paste0(format(date, "%Y-%m"), "-01")))
      } else {
        # Para vetores de datas
        result <- as.Date(paste0(format(date, "%Y-%m"), "-01"))
        result[is.na(date)] <- NA
        return(result)
      }
    }
    
    df$Data <- standardize_date(df$Data)
    data_inicial <- standardize_date(data_inicial)
    data_atual <- standardize_date(data_atual)
    
    # Definir data final de previsão
    if (is.null(end_predict) || !inherits(end_predict, "Date")) {
      end_predict <- data_atual %m+% months(12)
    } else if (end_predict <= data_atual) {
      end_predict <- data_atual %m+% months(12)
    }
    end_predict <- standardize_date(end_predict)
    
    # Filtrar dados relevantes
    df_filtered <- df %>% 
      filter(!is.na(Data)) %>%
      filter(!is.na(!!sym(target_var))) %>%
      # Remover duplicações de datas (manter a primeira ocorrência)
      group_by(Data) %>%
      slice(1) %>%
      ungroup()
    
    # Definir conjuntos de treino e teste
    if (n_teste == 0) {
      cutoff_date <- data_atual
      test <- data.frame()
      test_y <- numeric(0)
      test_dates <- NULL
    } else {
      cutoff_date <- data_atual %m-% months(n_teste)
      cutoff_date <- standardize_date(cutoff_date)
      test <- df_filtered %>% filter(Data > cutoff_date & Data <= data_atual)
      test_y <- test[[target_var]]
      test_dates <- test$Data
    }
    
    # Conjunto de treino: dados até o cutoff_date
    train <- df_filtered %>% filter(Data >= data_inicial & Data <= cutoff_date)
    
    # Verificar dados suficientes
    min_required <- period * 2
    if (nrow(train) < min_required) {
      stop(paste0("Dados de treinamento insuficientes. São necessários pelo menos ", min_required, " pontos."))
    }
    
    # Usar apenas os dados de treino para ajustar os modelos
    train_y <- ts(train[[target_var]], frequency = period)
    
    has_exog_vars <- !is.null(exog_vars) && length(exog_vars) > 0
    
    # Calcular horizontes de previsão
    h_test <- length(test_y)
    h_future <- floor(as.numeric(difftime(end_predict, data_atual, units = "days")) / 30) + 1
    h_total <- h_test + h_future
    
    cat("Horizonte total de previsão:", h_total, "meses (", 
        h_test, "de teste +", h_future, "futuros)\n")
    
    # Gerar datas futuras precisas para o horizonte de previsão
    # CORREÇÃO: Garantir que todas as datas sejam no dia 1 do mês
    future_dates <- seq.Date(
      from = standardize_date(data_atual %m+% months(1)), 
      by = "month", 
      length.out = h_future
    )
    
    # Combinar com datas de teste se houver período de teste
    if (h_test > 0) {
      all_forecast_dates <- c(test_dates, future_dates)
    } else {
      all_forecast_dates <- future_dates
    }
    
    # Função para calcular métricas de erro
    calc_error_metrics <- function(actual, predicted) {
      if (length(actual) == 0 || length(predicted) == 0) {
        return(list(MAPE = NA, RMSE = NA, MAE = NA))
      }
      
      len <- min(length(actual), length(predicted))
      actual <- actual[1:len]
      predicted <- predicted[1:len]
      actual_safe <- ifelse(actual == 0, 1e-10, actual)
      
      MAPE <- mean(abs((actual - predicted) / abs(actual_safe))) * 100
      RMSE <- sqrt(mean((actual - predicted)^2))
      MAE <- mean(abs(actual - predicted))
      
      return(list(MAPE = MAPE, RMSE = RMSE, MAE = MAE))
    }
    
    # Preparação das variáveis exógenas, se aplicável
    if (has_exog_vars) {
      cat("Preparando matriz de variáveis exógenas para previsão...\n")
      
      # Para período de teste, se houver
      if (h_test > 0) {
        test_exog_df <- test[, exog_vars, drop = FALSE]
        for (var in exog_vars) {
          if (any(is.na(test_exog_df[[var]]))) {
            var_mean <- mean(train[[var]], na.rm = TRUE)
            test_exog_df[[var]][is.na(test_exog_df[[var]])] <- var_mean
          }
        }
        test_exog_matrix <- as.matrix(test_exog_df)
      }
      
      # Para período futuro
      future_data <- df_filtered %>% 
        filter(Data > data_atual) %>%
        select(Data, all_of(exog_vars))
      
      future_exog_df <- data.frame(Data = future_dates)
      future_exog_df <- left_join(future_exog_df, future_data, by = "Data")
      
      for (var in exog_vars) {
        if (!var %in% names(future_exog_df)) {
          future_exog_df[[var]] <- NA
        }
        if (any(is.na(future_exog_df[[var]]))) {
          cat("Imputando valores para variável:", var, "\n")
          var_mean <- mean(train[[var]], na.rm = TRUE)
          future_exog_df[[var]][is.na(future_exog_df[[var]])] <- var_mean
        }
      }
      
      future_exog_matrix <- as.matrix(future_exog_df[, exog_vars, drop = FALSE])
      
      # Combinar matrizes para horizonte total (teste + futuro)
      if (h_test > 0) {
        total_exog_matrix <- rbind(test_exog_matrix, future_exog_matrix)
      } else {
        total_exog_matrix <- future_exog_matrix
      }
      
      cat("Matriz de previsão total preparada com dimensões:", dim(total_exog_matrix)[1], "×", dim(total_exog_matrix)[2], "\n")
      
      # Matriz de treinamento
      train_exog_matrix <- as.matrix(train[, exog_vars, drop = FALSE])
      
      if (length(train_y) != nrow(train_exog_matrix)) {
        cat("Ajustando comprimento das séries para garantir compatibilidade\n")
        min_len <- min(length(train_y), nrow(train_exog_matrix))
        train_y <- train_y[1:min_len]
        train_exog_matrix <- train_exog_matrix[1:min_len,, drop = FALSE]
      }
    }
    
    # Configuração dos modelos ARIMA
    if (manual_mode && !is.null(arima_params)) {
      variations <- list(c(
        arima_params$p, arima_params$d, arima_params$q,
        arima_params$P, arima_params$D, arima_params$Q
      ))
    } else {
      cat("Executando auto.arima para encontrar modelo base...\n")
      initial_fit <- auto.arima(train_y, seasonal = seasonal)
      initial_order <- arimaorder(initial_fit)
      cat("Modelo base encontrado:", paste(initial_order, collapse=","), "\n")
      
      generate_variations <- function(order, num_variations) {
        variations <- list()
        variations[[1]] <- order  # Inclui o modelo original
        if (num_variations > 1) {
          for (i in 2:num_variations) {
            new_order <- order
            new_order[1] <- max(0, sample((max(0, order[1]-1)):(order[1]+1), 1))
            new_order[3] <- max(0, sample((max(0, order[3]-1)):(order[3]+1), 1))
            if (seasonal) {
              new_order[4] <- max(0, sample((max(0, order[4]-1)):(order[4]+1), 1))
              new_order[6] <- max(0, sample((max(0, order[6]-1)):(order[6]+1), 1))
            }
            variations[[i]] <- new_order
          }
        }
        return(variations)
      }
      
      variations <- generate_variations(initial_order, num_models)
    }
    
    # Inicializando listas e dataframes para armazenar os resultados
    forecast_results_uni <- list()
    forecast_results_multi <- list()
    rmse_results_uni <- data.frame()
    rmse_results_multi <- data.frame()
    arima_models_uni <- list()
    arima_models_multi <- list()
    
    # Dataframe para armazenar os desvios
    deviation_results <- data.frame(model = character(), 
                                    min_deviation = numeric(), 
                                    max_deviation = numeric(),
                                    mean_deviation = numeric(),
                                    cum_deviation = numeric(),
                                    min_pct_deviation = numeric(),
                                    max_pct_deviation = numeric(),
                                    mean_pct_deviation = numeric(),
                                    cum_pct_deviation = numeric(),
                                    stringsAsFactors = FALSE)
    
    cat("Processando", length(variations), "modelos ARIMA...\n")
    for (i in 1:length(variations)) {
      var_order <- variations[[i]]
      cat("Modelo", i, "de", length(variations), "com ordem:", paste(var_order, collapse=","), "\n")
      
      #----- MODELO UNIVARIADO -----
      fit_uni <- tryCatch({
        if (seasonal) {
          Arima(train_y, order = var_order[1:3], seasonal = list(order = var_order[4:6], period = period))
        } else {
          Arima(train_y, order = var_order[1:3])
        }
      }, error = function(e) {
        cat("Erro ao ajustar modelo univariado:", e$message, "\n")
        NULL
      })
      
      if (!is.null(fit_uni)) {
        # Gerar previsão para horizonte total (teste + futuro)
        fcast_uni <- forecast(fit_uni, h = h_total)
        
        if (h_test > 0) {
          # Extrair previsões para o período de teste
          test_pred_uni <- fcast_uni$mean[1:h_test]
          
          # Calcular desvios e métricas para o período de teste
          desvios_uni <- test_pred_uni - test_y
          desvios_pct_uni <- (desvios_uni / ifelse(test_y == 0, 1e-10, test_y)) * 100
          
          deviation_results <- rbind(deviation_results, data.frame(
            model = paste0("arima_uni_", i),
            min_deviation = round(min(desvios_uni, na.rm = TRUE), 2),
            max_deviation = round(max(desvios_uni, na.rm = TRUE), 2),
            mean_deviation = round(mean(desvios_uni, na.rm = TRUE), 2),
            cum_deviation = round(sum(desvios_uni, na.rm = TRUE), 2),
            min_pct_deviation = round(min(desvios_pct_uni, na.rm = TRUE), 2),
            max_pct_deviation = round(max(desvios_pct_uni, na.rm = TRUE), 2),
            mean_pct_deviation = round(mean(desvios_pct_uni, na.rm = TRUE), 2),
            cum_pct_deviation = round(sum(desvios_pct_uni, na.rm = TRUE), 2)
          ))
          errors_uni <- calc_error_metrics(test_y, test_pred_uni)
        } else {
          errors_uni <- list(MAPE = NA, RMSE = NA, MAE = NA)
        }
        
        # CORREÇÃO CRÍTICA: Usar as datas exatas e padronizadas para o alinhamento perfeito
        df_fcast_uni <- data.frame(
          Data = all_forecast_dates,
          ec_res_predict = as.numeric(fcast_uni$mean),
          Lo95 = as.numeric(fcast_uni$lower[, 2]),
          Hi95 = as.numeric(fcast_uni$upper[, 2]),
          Model = paste0("arima_uni_", i)
        )
        
        forecast_results_uni[[paste0("arima_uni_", i)]] <- df_fcast_uni
        arima_models_uni[[paste0("arima_uni_", i)]] <- fit_uni
        rmse_results_uni <- rbind(rmse_results_uni, data.frame(
          model = paste0("arima_uni_", i),
          MAPE = errors_uni$MAPE,
          RMSE = errors_uni$RMSE,
          MAE = errors_uni$MAE
        ))
        
        cat("Modelo univariado", i, "concluído. RMSE:", round(errors_uni$RMSE, 4), "\n")
      }
      
      #----- MODELO MULTIVARIADO -----
      if (has_exog_vars) {
        cat("Ajustando modelo multivariado", i, "...\n")
        
        fit_multi <- tryCatch({
          if (seasonal) {
            Arima(train_y, order = var_order[1:3],
                  seasonal = list(order = var_order[4:6], period = period),
                  xreg = train_exog_matrix)
          } else {
            Arima(train_y, order = var_order[1:3], xreg = train_exog_matrix)
          }
        }, error = function(e) {
          cat("Erro ao ajustar modelo multivariado:", e$message, "\n")
          NULL
        })
        
        if (!is.null(fit_multi)) {
          # Gerar previsão para horizonte total (teste + futuro)
          fcast_multi <- tryCatch({
            forecast(fit_multi, xreg = total_exog_matrix, h = h_total)
          }, error = function(e) {
            cat("ERRO na previsão multivariada:", e$message, "\n")
            NULL
          })
          
          if (!is.null(fcast_multi)) {
            if (h_test > 0) {
              # Extrair previsões para o período de teste
              test_pred_multi <- fcast_multi$mean[1:h_test]
              
              # Calcular desvios e métricas para o período de teste
              desvios_multi <- test_pred_multi - test_y
              desvios_pct_multi <- (desvios_multi / ifelse(test_y == 0, 1e-10, test_y)) * 100
              
              deviation_results <- rbind(deviation_results, data.frame(
                model = paste0("arima_multi_", i),
                min_deviation = round(min(desvios_multi, na.rm = TRUE), 2),
                max_deviation = round(max(desvios_multi, na.rm = TRUE), 2),
                mean_deviation = round(mean(desvios_multi, na.rm = TRUE), 2),
                cum_deviation = round(sum(desvios_multi, na.rm = TRUE), 2),
                min_pct_deviation = round(min(desvios_pct_multi, na.rm = TRUE), 2),
                max_pct_deviation = round(max(desvios_pct_multi, na.rm = TRUE), 2),
                mean_pct_deviation = round(mean(desvios_pct_multi, na.rm = TRUE), 2),
                cum_pct_deviation = round(sum(desvios_pct_multi, na.rm = TRUE), 2)
              ))
              errors_multi <- calc_error_metrics(test_y, test_pred_multi)
            } else {
              errors_multi <- list(MAPE = NA, RMSE = NA, MAE = NA)
            }
            
            # CORREÇÃO CRÍTICA: Usar as datas exatas e padronizadas para o alinhamento perfeito
            df_fcast_multi <- data.frame(
              Data = all_forecast_dates,
              ec_res_predict = as.numeric(fcast_multi$mean),
              Lo95 = as.numeric(fcast_multi$lower[, 2]),
              Hi95 = as.numeric(fcast_multi$upper[, 2]),
              Model = paste0("arima_multi_", i)
            )
            
            forecast_results_multi[[paste0("arima_multi_", i)]] <- df_fcast_multi
            arima_models_multi[[paste0("arima_multi_", i)]] <- fit_multi
            rmse_results_multi <- rbind(rmse_results_multi, data.frame(
              model = paste0("arima_multi_", i),
              MAPE = errors_multi$MAPE,
              RMSE = errors_multi$RMSE,
              MAE = errors_multi$MAE
            ))
            
            cat("Modelo multivariado", i, "concluído com", nrow(df_fcast_multi), "períodos. RMSE:", round(errors_multi$RMSE, 4), "\n")
          }
        }
      }
    }
    
    # Combinar resultados
    forecast_results <- c(forecast_results_uni, forecast_results_multi)
    rmse_results <- rbind(rmse_results_uni, rmse_results_multi)
    arima_models <- c(arima_models_uni, arima_models_multi)
    
    if (length(forecast_results) == 0) {
      stop("Nenhum modelo ARIMA foi ajustado com sucesso. Tente ajustar os parâmetros.")
    }
    
    # Ordenar modelos por RMSE
    rmse_results_sorted <- rmse_results %>% 
      mutate(RMSE_sort = ifelse(is.na(RMSE), Inf, RMSE)) %>%
      arrange(RMSE_sort)
    
    top_models <- rmse_results_sorted %>% pull(model)
    
    # Preparar dados para visualização
    df_train_out <- data.frame(
      Data = train$Data,
      ec_res = train[[target_var]],
      Period = "Treino"
    )
    
    df_test_out <- data.frame(
      Data = test$Data,
      ec_res = test[[target_var]],
      Period = "Teste"
    )
    
    # Combinar dados de treino e teste
    df_ <- bind_rows(df_train_out, df_test_out)
    
    # Criar o gráfico base com dados reais
    fig <- plot_ly(df_, x = ~Data, y = ~ec_res, name = 'Real', 
                   type = 'scatter', mode = 'lines', line = list(color = '#2c3e50', width = 3))
    
    # Adicionar marcação para períodos
    if (n_teste > 0) {
      fig <- fig %>% layout(
        shapes = list(
          # Marcador para data de corte entre treino e teste
          list(
            type = "line",
            x0 = cutoff_date,
            x1 = cutoff_date,
            y0 = 0,
            y1 = 1,
            yref = "paper",
            line = list(color = "#e74c3c", width = 2, dash = "dash")
          ),
          # Marcador para data atual (início da previsão futura)
          list(
            type = "line",
            x0 = data_atual,
            x1 = data_atual,
            y0 = 0,
            y1 = 1,
            yref = "paper",
            line = list(color = "#3498db", width = 2, dash = "dash")
          )
        )
      )
    }
    
    # Cores neutras para os modelos
    neutral_colors <- c("#7f8c8d", "#34495e", "#95a5a6", "#2c3e50", "#ecf0f1", 
                        "#bdc3c7", "#85929e", "#566573", "#273746", "#1b2631")
    
    uni_models <- grep("uni", top_models, value = TRUE)
    multi_models <- grep("multi", top_models, value = TRUE)
    
    # Plotar modelos univariados
    for (i in seq_along(uni_models)) {
      df_fcast <- forecast_results[[uni_models[i]]]
      if (!is.null(df_fcast) && nrow(df_fcast) > 0) {
        fig <- fig %>% add_lines(
          data = df_fcast, 
          x = ~Data, 
          y = ~ec_res_predict, 
          name = uni_models[i],
          line = list(
            color = neutral_colors[i %% length(neutral_colors) + 1],
            width = 2, 
            dash = 'solid'
          )
        )
      }
    }
    
    # Plotar modelos multivariados
    for (i in seq_along(multi_models)) {
      df_fcast <- forecast_results[[multi_models[i]]]
      if (!is.null(df_fcast) && nrow(df_fcast) > 0) {
        fig <- fig %>% add_lines(
          data = df_fcast, 
          x = ~Data, 
          y = ~ec_res_predict, 
          name = multi_models[i],
          line = list(
            color = neutral_colors[(i + length(uni_models)) %% length(neutral_colors) + 1],
            width = 2, 
            dash = 'solid'
          )
        )
      }
    }
    
    # Melhorar o título e legenda
    fig <- fig %>% layout(
      title = list(
        text = paste0("Previsão ARIMA"),
        font = list(size = 16, color = "#2c3e50")
      ),
      xaxis = list(title = "Data", titlefont = list(color = "#2c3e50")),
      yaxis = list(title = "Valores", titlefont = list(color = "#2c3e50")),
      legend = list(orientation = 'v', x = 1, y = 1),
      plot_bgcolor = "#f8f9fa",
      paper_bgcolor = "#ffffff"
    )
    
    # CORREÇÃO: Nova abordagem para df_projections_final
    # Criar um dataframe base com dados reais e projeções futuras
    
    # Obter todos os dados realizados (incluindo treino e teste)
    df_realizado <- df_filtered %>% 
      filter(Data <= data_atual) %>%
      select(Data, Target = !!sym(target_var))
    
    # Obter datas futuras para projeção
    future_dates <- seq.Date(
      from = standardize_date(data_atual %m+% months(1)), 
      by = "month", 
      length.out = h_future
    )
    
    # Criar dataframe base
    all_dates <- unique(c(df_realizado$Data, future_dates))
    all_dates <- all_dates[order(all_dates)]
    
    df_base <- data.frame(Data = all_dates)
    df_base <- left_join(df_base, df_realizado, by = "Data")
    
    # Adicionar coluna Period
    df_base$Period <- NA
    df_base$Period[df_base$Data <= data_atual] <- "Realizado"
    df_base$Period[df_base$Data > data_atual] <- "Projeção"
    
    # Para cada modelo, adicionar tanto valores reais (para períodos realizados) quanto projeções (para períodos futuros)
    for (model_name in names(forecast_results)) {
      model_df <- forecast_results[[model_name]]
      
      if (!is.null(model_df) && nrow(model_df) > 0) {
        # Para as datas já realizadas, usamos o valor real (Target)
        pred_col_name <- paste0("Pred_", model_name)
        lo95_col_name <- paste0("Lo95_", model_name)
        hi95_col_name <- paste0("Hi95_", model_name)
        
        # Inicializar colunas vazias
        df_base[[pred_col_name]] <- NA
        df_base[[lo95_col_name]] <- NA
        df_base[[hi95_col_name]] <- NA
        
        # Preencher valores realizados
        realized_rows <- df_base$Data <= data_atual & !is.na(df_base$Target)
        df_base[[pred_col_name]][realized_rows] <- df_base$Target[realized_rows]
        df_base[[lo95_col_name]][realized_rows] <- df_base$Target[realized_rows]
        df_base[[hi95_col_name]][realized_rows] <- df_base$Target[realized_rows]
        
        # Preencher valores projetados
        future_proj <- model_df %>% 
          filter(Data > data_atual) %>%
          select(Data, ec_res_predict, Lo95, Hi95)
        
        for (i in 1:nrow(future_proj)) {
          date_idx <- which(df_base$Data == future_proj$Data[i])
          if (length(date_idx) > 0) {
            df_base[[pred_col_name]][date_idx] <- future_proj$ec_res_predict[i]
            df_base[[lo95_col_name]][date_idx] <- future_proj$Lo95[i]
            df_base[[hi95_col_name]][date_idx] <- future_proj$Hi95[i]
          }
        }
      }
    }
    
    execution_time <- difftime(Sys.time(), start_time, units = "secs")
    cat("Análise ARIMA concluída em", round(execution_time, 2), "segundos\n")
    
    return(list(
      models = arima_models, 
      plots = list(fig), 
      errors = rmse_results, 
      df_arima = df_base,
      deviations = deviation_results,
      execution_time = execution_time,
      end_predict = end_predict,
      periods = list(
        train_start = data_inicial,
        train_end = cutoff_date,
        test_start = if(n_teste > 0) cutoff_date + 1 else NA,
        test_end = data_atual,
        forecast_start = data_atual + 1,
        forecast_end = end_predict
      )
    ))
  }
  
  # Função RMSE personalizada (compartilhada)
  my.rmse <- function(actual, predicted) {
    sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  }
  
  #________________________________________________________________________________
  # Função para agregar dados e gerar visões mensal, trimestral, semestral e anual
  #_________________________________________________________________________________
  
  generate_df_visoes <- function(df, variable, title, date_col = "Data") {
    
    # Converter o nome da coluna de data para um símbolo
    date_sym <- sym(date_col)
    
    ###########   Função para gerar tabela mensal ###########  
    process_df_table <- function(df, variable, date_sym) {
      Tabela_1 <- df %>%
        select(!!date_sym, all_of(variable)) %>%
        mutate(
          Ano = year(!!date_sym),  
          Mes = factor(
            month(!!date_sym), 
            levels = 1:12, 
            labels = c("Jan", "Fev", "Mar", "Abr", "Mai", "Jun",  "Jul", "Ago", "Set", "Out", "Nov", "Dez")
          )
        ) %>%
        group_by(Ano, Mes) %>%
        summarise(Valor = sum(get(variable), na.rm = TRUE), .groups = "drop") %>%
        pivot_wider(names_from = Mes, values_from = Valor, values_fill = 0)
      return(Tabela_1)
    }
    
    ###########   Função para gerar tabela anual ###########  
    process_annual_table <- function(df, variable, date_sym) {
      Tabela_Anual <- df %>%
        select(!!date_sym, all_of(variable)) %>%
        mutate(Ano = year(!!date_sym)) %>%
        group_by(Ano) %>%
        summarise(Valor_Anual = sum(get(variable), na.rm = TRUE), .groups = "drop") %>% 
        mutate(Valor_Anual = round(Valor_Anual, 0))
      return(Tabela_Anual)
    }
    
    ###########   Função para gerar tabela trimestral ###########  
    process_quarterly_table <- function(df, variable, date_sym) {
      Tabela_Trimestral <- df %>%
        select(!!date_sym, all_of(variable)) %>%
        mutate(
          Ano = year(!!date_sym), 
          Trimestre = paste0("Q", quarter(!!date_sym))
        ) %>%
        group_by(Ano, Trimestre) %>%
        summarise(Valor_Trimestral = sum(get(variable), na.rm = TRUE), .groups = "drop") %>%
        pivot_wider(names_from = Trimestre, values_from = Valor_Trimestral, values_fill = 0)
      return(Tabela_Trimestral)
    }
    
    ###########  Função para gerar tabela semestral ###########  
    process_semester_table <- function(df, variable, date_sym) {
      Tabela_Semestral <- df %>%
        select(!!date_sym, all_of(variable)) %>%
        mutate(
          Ano = year(!!date_sym), 
          Semestre = ifelse(month(!!date_sym) <= 6, "S1", "S2")
        ) %>%
        group_by(Ano, Semestre) %>%
        summarise(Valor_Semestral = sum(get(variable), na.rm = TRUE), .groups = "drop") %>%
        pivot_wider(names_from = Semestre, values_from = Valor_Semestral, values_fill = 0)
      return(Tabela_Semestral)
    }
    
    ###########  Função para calcular as taxas de crescimento em relação ao ano anterior ########### 
    calculate_growth_rates <- function(tabela) {
      tabela_long <- tabela %>%
        pivot_longer(-Ano, names_to = "Periodo", values_to = "Valor") %>%
        arrange(Ano, Periodo) %>%
        group_by(Periodo) %>%
        mutate(Taxa_Crescimento = (Valor / lag(Valor) - 1) * 100) %>%
        mutate(Taxa_Crescimento = round(Taxa_Crescimento, 4)) %>%
        ungroup() %>%
        select(Ano, Periodo, Taxa_Crescimento) %>%
        pivot_wider(names_from = Periodo, values_from = Taxa_Crescimento)
      return(tabela_long)
    }
    
    ###########  Função para calcular as variações em valores absolutos em relação ao ano anterior ########### 
    calculate_value_differences <- function(tabela) {
      tabela_long <- tabela %>%
        pivot_longer(-Ano, names_to = "Periodo", values_to = "Valor") %>%
        arrange(Ano, Periodo) %>%
        group_by(Periodo) %>%
        mutate(Variacao_Valor = Valor - lag(Valor)) %>%
        mutate(Variacao_Valor = round(Variacao_Valor, 0)) %>%
        ungroup() %>%
        select(Ano, Periodo, Variacao_Valor) %>%
        pivot_wider(names_from = Periodo, values_from = Variacao_Valor)
      return(tabela_long)
    }
    
    ###########  Função para unir as tabelas de crescimento e variações com a tabela original ########### 
    merge_growth_and_differences <- function(tabela_original, tabela_growth, tabela_diff, prefix = "") {
      merged_table <- tabela_original %>%
        left_join(tabela_growth, by = "Ano", suffix = c("", "_Growth")) %>%
        left_join(tabela_diff, by = "Ano", suffix = c("", "_Diff"))
      
      # Renomear as colunas de crescimento e variação
      col_names <- names(merged_table)
      new_names <- sapply(col_names, function(x) {
        if (grepl("_Growth", x)) {
          paste0(prefix, "Taxa_Crescimento_", gsub("_Growth", "", x))
        } else if (grepl("_Diff", x)) {
          paste0(prefix, "Variacao_Valor_", gsub("_Diff", "", x))
        } else {
          x
        }
      })
      names(merged_table) <- new_names
      return(merged_table)
    }
    
    # Processar os dados e gerar as tabelas
    tabela_mensal <- process_df_table(df, variable, date_sym)
    tabela_anual <- process_annual_table(df, variable, date_sym)
    tabela_trimestral <- process_quarterly_table(df, variable, date_sym)
    tabela_semestral <- process_semester_table(df, variable, date_sym)
    
    # Calcular as taxas de crescimento e as variações em valores absolutos
    growth_mensal <- calculate_growth_rates(tabela_mensal)
    growth_anual <- calculate_growth_rates(tabela_anual)
    growth_trimestral <- calculate_growth_rates(tabela_trimestral)
    growth_semestral <- calculate_growth_rates(tabela_semestral)
    
    diff_mensal <- calculate_value_differences(tabela_mensal)
    diff_anual <- calculate_value_differences(tabela_anual)
    diff_trimestral <- calculate_value_differences(tabela_trimestral)
    diff_semestral <- calculate_value_differences(tabela_semestral)
    
    # Unir as tabelas de crescimento e variações com as tabelas originais
    tabela_mensal_unificada <- merge_growth_and_differences(tabela_mensal, growth_mensal, diff_mensal, prefix = "Mensal_")
    tabela_anual_unificada <- merge_growth_and_differences(tabela_anual, growth_anual, diff_anual, prefix = "Anual_")
    tabela_trimestral_unificada <- merge_growth_and_differences(tabela_trimestral, growth_trimestral, diff_trimestral, prefix = "Trimestral_")
    tabela_semestral_unificada <- merge_growth_and_differences(tabela_semestral, growth_semestral, diff_semestral, prefix = "Semestral_")
    
    ########### Definir a função para transformar o dataframe mensal  ###########
    
    transformar_mensal_long <- function(df) {
      # Mapeamento dos meses em português para números
      month_map <- c(
        "Jan" = 1, "Fev" = 2, "Mar" = 3, "Abr" = 4, "Mai" = 5, "Jun" = 6, 
        "Jul" = 7, "Ago" = 8, "Set" = 9, "Out" = 10, "Nov" = 11, "Dez" = 12
      )
      
      # Transformar 'Valor' para formato longo (Jan a Dez)
      valor_long <- df %>% 
        pivot_longer(cols = Jan:Dez, names_to = "Month", values_to = "Valor")
      
      # Transformar 'Taxa_Crescimento' para formato longo
      tx_long <- df %>% 
        pivot_longer(
          cols = starts_with("Mensal_Taxa_Crescimento_"), 
          names_to = "Month", 
          names_prefix = "Mensal_Taxa_Crescimento_", 
          values_to = "Taxa_Crescimento"
        )
      
      # Transformar 'Variacao_Valor' para formato longo
      va_long <- df %>% 
        pivot_longer(
          cols = starts_with("Mensal_Variacao_Valor_"),  
          names_to = "Month", 
          names_prefix = "Mensal_Variacao_Valor_", 
          values_to = "Variacao_Valor"
        )
      
      # Combinar os dataframes longos baseados em 'Ano' e 'Month'
      Tabela_Mensal_long <- valor_long %>%
        left_join(tx_long, by = c("Ano", "Month")) %>%
        left_join(va_long, by = c("Ano", "Month")) %>%
        mutate(
          Month_Num = month_map[Month],
          Data = as.Date(paste0(Ano, "-", sprintf("%02d", Month_Num), "-01")),
          .before = "Valor"
        ) %>%
        select(Data, Valor, Taxa_Crescimento, Variacao_Valor)
      
      return(Tabela_Mensal_long)
    }
    
    ###########  Definir a função para transformar o dataframe trimestral ########### 
    
    transformar_trimestre_long <- function(df) {
      # Transformar 'Valor' para formato longo (Q1, Q2, Q3, Q4)
      valor_long <- df %>% 
        pivot_longer(
          cols = starts_with("Q"), 
          names_to = "Quarter", 
          values_to = "Valor"
        )
      
      # Transformar 'Taxa_Crescimento' para formato longo (colunas que contêm "Taxa")
      tx_long <- df %>%
        pivot_longer(
          cols = starts_with("Trimestral_Taxa_Crescimento_Q"), 
          names_to = "Quarter", 
          names_pattern = "Trimestral_Taxa_Crescimento_Q(\\d)", 
          values_to = "Taxa_Crescimento"
        )
      
      # Transformar 'Variacao_Valor' para formato longo (colunas que contêm "variacao")
      va_long <- df %>%
        pivot_longer(
          cols = starts_with("Trimestral_Variacao_Valor_Q"), 
          names_to = "Quarter", 
          names_pattern = "Trimestral_Variacao_Valor_Q(\\d)", 
          values_to = "Variacao_Valor"
        )
      
      # Combinar os dataframes longos baseados em 'Ano' e 'Quarter'
      Tabela_Trimestral_long <- valor_long %>%
        left_join(tx_long, by = c("Ano", "Quarter")) %>%
        left_join(va_long, by = c("Ano", "Quarter")) %>%
        mutate(
          Quarter = as.integer(gsub("Q", "", Quarter)), 
          Data = as.yearqtr(paste0(Ano, " Q", Quarter), format = "%Y Q%q")
        ) %>%
        select(Data, Valor, Taxa_Crescimento, Variacao_Valor)
      
      return(Tabela_Trimestral_long)
    }
    
    ########### Definir a função para processar o dataframe semestral  ###########
    
    transformar_semestral_long <- function(df) {
      # Criar as tabelas de valores para os dois semestres
      S_1 <- df %>% select(Ano, Valor = S1) %>% mutate(Semestre = 1)
      S_2 <- df %>% select(Ano, Valor = S2) %>% mutate(Semestre = 2)
      SEM <- bind_rows(S_1, S_2)
      
      # Criar as tabelas de taxas de crescimento para os dois semestres
      TX_S_1 <- df %>% select(Ano, Taxa_Crescimento_Semestral = Semestral_Taxa_Crescimento_S1) %>% mutate(Semestre = 1)
      TX_S_2 <- df %>% select(Ano, Taxa_Crescimento_Semestral = Semestral_Taxa_Crescimento_S2) %>% mutate(Semestre = 2)
      TX_SEM <- bind_rows(TX_S_1, TX_S_2)
      
      # Criar as tabelas de variação de valor para os dois semestres
      VA_S_1 <- df %>% select(Ano, Variacao_Semestral = Semestral_Variacao_Valor_S1) %>% mutate(Semestre = 1)
      VA_S_2 <- df %>% select(Ano, Variacao_Semestral = Semestral_Variacao_Valor_S2) %>% mutate(Semestre = 2)
      VA_SEM <- bind_rows(VA_S_1, VA_S_2)
      
      # Juntar todas as tabelas usando 'Ano' e 'Semestre' como chaves
      joined <- list(SEM, TX_SEM, VA_SEM) %>% reduce(left_join, by = c("Ano", "Semestre"))
      
      # Gerar a coluna de Data que representa o início do semestre
      joined <- joined %>%
        mutate(
          Data = case_when(
            Semestre == 1 ~ as.Date(paste0(Ano, "-01-01")),
            Semestre == 2 ~ as.Date(paste0(Ano, "-07-01"))
          )
        ) %>% 
        arrange(Data)
      
      joined <- joined %>% 
        select(Data, Valor, Taxa_Crescimento_Semestral, Variacao_Semestral)
      
      return(joined)
    }
    
    ########### Definir a função para transformar o dataframe anual ###########
    
    transformar_anual_long <- function(df) {
      Tabela_Anual_long <- df %>%
        mutate(Data = as.Date(paste0(Ano, "-01-01"))) %>%
        select(Data, everything()) %>%
        rename(Taxa_Crescimento_Anual = Anual_Taxa_Crescimento_Valor_Anual,
               Variacao_Valor_Anual = Anual_Variacao_Valor_Valor_Anual)
      return(Tabela_Anual_long)
    }
    
    # Gerar dataframes longs
    Tabela_Mensal_long <- transformar_mensal_long(tabela_mensal_unificada)
    Tabela_Trimestral_long <- transformar_trimestre_long(tabela_trimestral_unificada)
    Tabela_semestral_long <- transformar_semestral_long(tabela_semestral_unificada)
    Tabela_Anual_long <- transformar_anual_long(tabela_anual_unificada)
    
    # Retornar as tabelas geradas
    return(list(
      Tabela_Mensal = tabela_mensal_unificada, 
      Tabela_Trimestral = tabela_trimestral_unificada,
      Tabela_Semestral = tabela_semestral_unificada,
      Tabela_Anual = tabela_anual_unificada,
      
      Tabela_Mensal_long = Tabela_Mensal_long,
      Tabela_Trimestral_long = Tabela_Trimestral_long,
      Tabela_semestral_long = Tabela_semestral_long,
      Tabela_Anual_long = Tabela_Anual_long
    ))
  }
  
  #__________________________________________
  # Função para arredondar e remover índices
  #__________________________________________
  process_table <- function(df) {
    df <- df %>%
      mutate(across(where(is.numeric), ~round(., 2))) %>%
      as.data.frame()
    return(df)
  }
  
  # Função para transpor as tabelas
  transpose_table <- function(df) {
    df_transposta <- as.data.frame(t(df))
    colnames(df_transposta) <- df_transposta[1, ]
    df_transposta <- df_transposta[-1, ]
    return(df_transposta)
  }
}

#======================================================================
### UI
#======================================================================
{
  
  ui_arima <- dashboardPage(
    skin = "black",
    dashboardHeader(
      title = span(icon("chart-line"), "Análise de Séries Temporais", style = "font-size: 18px; color: #ffffff;")
    ),
    dashboardSidebar(
      width = 260,
      sidebarMenu(
        menuItem("Upload e Configuração", tabName = "upload", icon = icon("upload")),
        menuItem("Modelos ARIMA", tabName = "arima", icon = icon("chart-area")),
        menuItem("Análise de Crescimento", tabName = "crescimento", icon = icon("percentage"))
      ),
      tags$div(
        style = "padding: 20px; margin-top: 20px; border-top: 1px solid #34495e; text-align: center;",
        tags$p("Análise Avançada de Séries Temporais", style = "font-size: 11px; color: #95a5a6; margin: 0;")
      )
    ),
    dashboardBody(
      useShinyjs(), # Inicializa shinyjs
      tags$head(
        tags$style(HTML("
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          background-color: #f8f9fa;
          color: #2c3e50;
        }
        .content-wrapper {
          background-color: #f8f9fa;
        }
        .box {
          border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
          border: 1px solid #e9ecef;
          margin-bottom: 20px;
        }
        .box-header {
          padding: 20px;
          background-color: #ffffff;
          border-bottom: 1px solid #e9ecef;
          border-radius: 8px 8px 0 0;
        }
        .box-title {
          font-size: 16px;
          font-weight: 600;
          color: #2c3e50;
        }
        .box-body {
          background-color: #ffffff;
          padding: 20px;
          border-radius: 0 0 8px 8px;
        }
        .selectize-input {
          border-radius: 6px;
          border: 1px solid #ced4da;
          padding: 8px 12px;
          font-size: 14px;
        }
        .selectize-input:focus {
          border-color: #5dade2;
          box-shadow: 0 0 0 0.2rem rgba(93, 173, 226, 0.25);
        }
        .btn {
          border-radius: 6px;
          margin-right: 5px;
          font-weight: 500;
          padding: 8px 16px;
          border: none;
          transition: all 0.3s ease;
        }
        .btn-primary {
          background-color: #5dade2;
          color: white;
        }
        .btn-primary:hover {
          background-color: #3498db;
          transform: translateY(-1px);
        }
        .btn-success {
          background-color: #58d68d;
          color: white;
        }
        .btn-success:hover {
          background-color: #2ecc71;
          transform: translateY(-1px);
        }
        .btn-info {
          background-color: #85c1e9;
          color: white;
        }
        .btn-info:hover {
          background-color: #5dade2;
          transform: translateY(-1px);
        }
        .btn-warning {
          background-color: #f7dc6f;
          color: #2c3e50;
        }
        .btn-warning:hover {
          background-color: #f4d03f;
          transform: translateY(-1px);
        }
        .arima-section {
          background-color: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 20px;
          border: 1px solid #e9ecef;
        }
        .help-text {
          color: #6c757d;
          font-size: 12px;
          margin-top: 8px;
          font-style: italic;
        }
        .tab-content {
          padding-top: 25px;
        }
        .param-group {
          margin-bottom: 20px;
          padding: 15px;
          background-color: #f8f9fa;
          border-radius: 8px;
          border: 1px solid #e9ecef;
        }
        .param-title {
          font-weight: 600;
          color: #2c3e50;
          margin-bottom: 15px;
          border-bottom: 2px solid #5dade2;
          padding-bottom: 8px;
          font-size: 14px;
        }
        .datatable-container {
          margin-top: 20px;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .file-input-container {
          border: 2px dashed #bdc3c7;
          border-radius: 8px;
          padding: 30px;
          text-align: center;
          margin-bottom: 25px;
          background-color: #f8f9fa;
          transition: all 0.3s ease;
        }
        .file-input-container:hover {
          border-color: #5dade2;
          background-color: #f0f8ff;
        }
        .shiny-input-container {
          margin-bottom: 20px;
        }
        .info-box {
          background-color: #e8f4f8;
          padding: 15px;
          border-radius: 6px;
          margin-bottom: 20px;
          border-left: 4px solid #5dade2;
        }
        .tab-selector {
          margin-bottom: 25px;
          border-bottom: 1px solid #e9ecef;
          padding-bottom: 15px;
        }
        .nav-tabs {
          border-bottom: 1px solid #e9ecef;
        }
        .nav-tabs .nav-item .nav-link {
          border: none;
          color: #6c757d;
          font-weight: 500;
          padding: 10px 20px;
          border-radius: 6px 6px 0 0;
        }
        .nav-tabs .nav-item .nav-link.active {
          color: #2c3e50;
          background-color: #ffffff;
          border-bottom: 2px solid #5dade2;
        }
        .form-control {
          border-radius: 6px;
          border: 1px solid #ced4da;
          padding: 8px 12px;
          font-size: 14px;
        }
        .form-control:focus {
          border-color: #5dade2;
          box-shadow: 0 0 0 0.2rem rgba(93, 173, 226, 0.25);
        }
        .pretty input:checked~.state label:after {
          background-color: #5dade2 !important;
        }
        .pretty .state label:before {
          border-color: #ced4da;
        }
        .main-header .logo {
          background-color: #2c3e50 !important;
        }
        .main-header .navbar {
          background-color: #2c3e50 !important;
        }
        .main-sidebar {
          background-color: #34495e !important;
        }
        .sidebar-menu > li > a {
          color: #ecf0f1 !important;
        }
        .sidebar-menu > li > a:hover {
          background-color: #5dade2 !important;
          color: #ffffff !important;
        }
        .sidebar-menu > li.active > a {
          background-color: #5dade2 !important;
          color: #ffffff !important;
        }
      "))
      ),
      tabItems(
        # Aba de Upload
        tabItem(tabName = "upload",
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("file-excel"), "Upload de Arquivo"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           radioButtons("data_source", "Fonte dos Dados:",
                                        choices = c("Upload do Usuário" = "upload", "Usar Dados de Exemplo" = "exemplo"),
                                        selected = "upload",
                                        inline = TRUE),
                           # Esconde o fileInput quando for exemplo
                           conditionalPanel(
                             condition = "input.data_source == 'upload'",
                             div(class = "file-input-container",
                                 fileInput("file", NULL, 
                                           accept = c(".xlsx", ".csv"), 
                                           buttonLabel = "Escolher arquivo XLSX ou CSV",
                                           placeholder = "Nenhum arquivo selecionado")
                             )
                           ),
                           conditionalPanel(
                             condition = "output.fileUploaded == true",
                             div(class = "param-group",
                                 div(class = "param-title", "Configuração de Dados"),
                                 fluidRow(
                                   column(12,
                                          uiOutput("sheet_selector")
                                   )
                                 ),
                                 fluidRow(
                                   column(6,
                                          uiOutput("date_column_selector")
                                   ),
                                   column(6,
                                          selectInput("date_format", "Formato da Data:",
                                                      choices = c("ymd" = "ymd", 
                                                                  "dmy" = "dmy", 
                                                                  "mdy" = "mdy",
                                                                  "ydm" = "ydm"),
                                                      selected = "ymd")
                                   )
                                 ),
                                 div(style = "text-align: center; margin-top: 20px;",
                                     actionButton("process_data", "Processar Dados", 
                                                  icon = icon("check"), 
                                                  class = "btn-success")
                                 ),
                                 div(class = "info-box",
                                     icon("info-circle"), " Selecione a aba do Excel, a coluna que contém as datas e o formato correto."
                                 )
                             )
                           )
                         )
                  )
                ),
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("table"), "Visualização de Dados"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           div(class = "datatable-container",
                               DTOutput("contents")
                           ),
                           br(),
                           div(style = "text-align: center;",
                               downloadButton("downloadData", "Baixar Amostra", class = "btn-info")
                           )
                         )
                  )
                )
        ),
        
        # Aba ARIMA
        tabItem(tabName = "arima",
                fluidRow(
                  column(6,
                         box(
                           title = span(icon("sliders-h"), "Configuração do Modelo ARIMA"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           
                           div(class = "param-group",
                               div(class = "param-title", "Seleção de Variáveis"),
                               uiOutput("target_selector"),
                               uiOutput("exog_selector")
                           ),
                           
                           div(class = "param-group",
                               div(class = "param-title", "Período Temporal"),
                               uiOutput("arima_date_range_ui"),
                               dateInput("end_predict_arima", "Data Final da Previsão:", 
                                         value = Sys.Date() + lubridate::years(2),
                                         format = "yyyy-mm-dd"),
                               tags$p(class = "help-text", icon("info-circle"), 
                                      " Esta data é automaticamente definida como 2 anos após a data final do período de análise.")
                           ),
                           
                           div(class = "param-group",
                               div(class = "param-title", "Parâmetros Gerais"),
                               numericInput("n_test", "Períodos para Teste:", value = 6, min = 0),
                               tags$p(class = "help-text", icon("info-circle"), 
                                      " Número de meses mais recentes que ficam de fora dos dados de treino para validação do modelo.")
                           ),
                           
                           div(class = "arima-section",
                               div(style = "font-weight: 600; margin-bottom: 15px; color: #2c3e50;", "Configuração ARIMA:"),
                               prettyRadioButtons("arima_mode", NULL, 
                                                  choices = c("Automático" = "auto", "Manual" = "manual"), 
                                                  selected = "auto",
                                                  inline = TRUE,
                                                  status = "primary")
                           ),
                           
                           conditionalPanel(
                             condition = "input.arima_mode == 'manual'",
                             div(class = "arima-section",
                                 div(style = "font-weight: 600; margin-bottom: 15px; color: #2c3e50;", 
                                     icon("wrench"), " Configuração Manual ARIMA"),
                                 
                                 div(style = "margin-bottom: 15px;",
                                     div(style = "font-weight: 500; margin-bottom: 10px;", "Parâmetros Não-Sazonais"),
                                     fluidRow(
                                       column(4, numericInput("arima_p", "p (AR):", value = 1, min = 0, max = 10)),
                                       column(4, numericInput("arima_d", "d (Diff):", value = 1, min = 0, max = 3)),
                                       column(4, numericInput("arima_q", "q (MA):", value = 1, min = 0, max = 10))
                                     )
                                 ),
                                 
                                 div(style = "margin-bottom: 15px;",
                                     div(style = "font-weight: 500; margin-bottom: 10px;", "Configuração Sazonal"),
                                     prettySwitch("seasonal_manual", "Incluir Sazonalidade", value = TRUE, status = "primary")
                                 ),
                                 
                                 conditionalPanel(
                                   condition = "input.seasonal_manual == true",
                                   div(style = "margin-top: 15px;",
                                       div(style = "font-weight: 500; margin-bottom: 10px;", "Parâmetros Sazonais"),
                                       fluidRow(
                                         column(4, numericInput("arima_P", "P (SAR):", value = 1, min = 0, max = 5)),
                                         column(4, numericInput("arima_D", "D (SDiff):", value = 1, min = 0, max = 2)),
                                         column(4, numericInput("arima_Q", "Q (SMA):", value = 1, min = 0, max = 5))
                                       ),
                                       numericInput("period_manual", "Período Sazonal:", value = 12, min = 1)
                                   )
                                 ),
                                 tags$p(class = "help-text", icon("info-circle"), 
                                        " No modo manual, apenas um modelo será ajustado com os parâmetros especificados.")
                             )
                           ),
                           
                           conditionalPanel(
                             condition = "input.arima_mode == 'auto'",
                             div(class = "arima-section",
                                 div(style = "font-weight: 600; margin-bottom: 15px; color: #2c3e50;", 
                                     icon("robot"), " Configuração Automática ARIMA"),
                                 
                                 numericInput("num_models", "Número de Modelos ARIMA:", value = 5, min = 1, max = 20),
                                 prettySwitch("seasonal", "Incluir Sazonalidade", value = TRUE, status = "primary"),
                                 conditionalPanel(
                                   condition = "input.seasonal == true",
                                   numericInput("period", "Período Sazonal:", value = 12, min = 1)
                                 ),
                                 tags$p(class = "help-text", icon("info-circle"), 
                                        " No modo automático, o sistema determinará a melhor configuração e criará variações.")
                             )
                           ),
                           
                           div(style = "margin-top: 25px; text-align: center;",
                               actionButton("run_arima", "Executar Análise ARIMA", 
                                            icon = icon("play"), 
                                            class = "btn-success",
                                            style = "padding: 10px 30px; font-size: 16px;")
                           )
                         )
                  ),
                  
                  column(6,
                         box(
                           title = span(icon("poll"), "Resultados dos Modelos"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           div(style = "overflow-x: auto;",
                               tableOutput("arima_errors")
                           ),
                           div(class = "info-box",
                               "Métricas de erro ordenadas pelo RMSE. Valores mais baixos indicam melhor desempenho do modelo."
                           )
                         )
                  )
                ),
                
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("chart-area"), "Visualização de Previsões ARIMA"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           div(style = "height: 600px;",
                               plotlyOutput("arima_plot", height = "100%")
                           )
                         )
                  )
                ),
                
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("file-export"), "Exportar Projeções ARIMA"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           div(style = "padding: 20px; background-color: #f8f9fa; border-radius: 8px;",
                               div(style = "margin-bottom: 20px;",
                                   uiOutput("export_model_selector")
                               ),
                               div(style = "text-align: center;",
                                   downloadButton("download_xlsx", "Baixar Projeções XLSX", class = "btn-info")
                               )
                           )
                         )
                  )
                )
        ),
        
        # Aba para Análise de Crescimento
        tabItem(tabName = "crescimento",
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("percentage"), "Análise de Taxas de Crescimento"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           
                           div(class = "info-box",
                               icon("info-circle"), " Esta seção permite analisar as taxas de crescimento e variações nas projeções dos modelos ARIMA. Selecione um modelo e o tipo de visualização desejada para prosseguir."
                           ),
                           
                           fluidRow(
                             column(4,
                                    uiOutput("growth_model_selector")
                             ),
                             column(4,
                                    selectInput("periodo_analise", "Período de Análise:",
                                                choices = c("Mensal" = "mensal", 
                                                            "Trimestral" = "trimestral", 
                                                            "Semestral" = "semestral",
                                                            "Anual" = "anual"),
                                                selected = "mensal")
                             ),
                             column(4,
                                    div(style = "margin-top: 25px;",
                                        actionButton("analisar_crescimento", "Analisar Crescimento", 
                                                     icon = icon("chart-line"), 
                                                     class = "btn-primary")
                                    )
                             )
                           )
                         )
                  )
                ),
                
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("table"), "Tabelas de Crescimento"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           
                           div(class = "tab-selector",
                               tabsetPanel(id = "tabelas_crescimento",
                                           tabPanel("Valores", DTOutput("tabela_valores")),
                                           tabPanel("Taxas de Crescimento", DTOutput("tabela_taxas")),
                                           tabPanel("Variações Absolutas", DTOutput("tabela_variacoes"))
                               )
                           )
                         )
                  )
                ),
                
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("chart-bar"), "Visualização de Crescimento"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           
                           fluidRow(
                             column(3,
                                    selectInput("tipo_grafico", "Tipo de Gráfico:",
                                                choices = c("Valores Absolutos" = "valores", 
                                                            "Taxas de Crescimento" = "taxas", 
                                                            "Variação Absoluta" = "variacoes"),
                                                selected = "valores")
                             ),
                             column(3,
                                    selectInput("tipo_plotly", "Estilo de Gráfico:",
                                                choices = c("Linhas" = "lines", 
                                                            "Barras" = "bars",
                                                            "Combinado" = "combined"),
                                                selected = "lines")
                             ),
                             column(6,
                                    uiOutput("filtro_periodo")
                             )
                           ),
                           
                           div(style = "height: 500px;",
                               plotlyOutput("grafico_crescimento", height = "100%")
                           )
                         )
                  )
                ),
                
                fluidRow(
                  column(12,
                         box(
                           title = span(icon("download"), "Exportar Análise de Crescimento"), 
                           status = "primary", 
                           solidHeader = TRUE, 
                           width = 12,
                           
                           fluidRow(
                             column(6,
                                    div(style = "padding: 25px; text-align: center;",
                                        downloadButton("download_crescimento_xlsx", "Baixar Análise em Excel", 
                                                       class = "btn-success")
                                    )
                             ),
                             column(6,
                                    div(style = "padding: 25px; text-align: center;",
                                        downloadButton("download_crescimento_pdf", "Baixar Relatório PDF", 
                                                       class = "btn-info")
                                    )
                             )
                           )
                         )
                  )
                )
        )
      )
    )
  )
}

#======================================================================
### SERVER
#======================================================================
{
  
  server_arima <- function(input, output, session) {
    
    # Variáveis reativas para upload e visualização
    df <- reactiveVal(NULL)
    raw_df <- reactiveVal(NULL)
    sheet_names <- reactiveVal(NULL)
    file_type <- reactiveVal(NULL)
    example_loaded <- reactiveVal(FALSE)
    
    # Observer para carregar dados de exemplo
    observeEvent(input$data_source, {
      if (input$data_source == "exemplo") {
        showNotification(paste("Tentando carregar:", EXAMPLE_DATA_PATH), type = "message")
        if (file.exists(EXAMPLE_DATA_PATH)) {
          available_sheets <- readxl::excel_sheets(EXAMPLE_DATA_PATH)
          sheet_names(available_sheets)
          first_sheet_data <- readxl::read_excel(EXAMPLE_DATA_PATH, sheet = available_sheets[1])
          
          # Tenta identificar a coluna de data automaticamente
          possiveis_nomes_data <- c("Data", "data", "DATA", "date", "Date")
          nome_coluna_data <- names(first_sheet_data)[tolower(names(first_sheet_data)) %in% tolower(possiveis_nomes_data)]
          
          if (length(nome_coluna_data) == 1) {
            names(first_sheet_data)[names(first_sheet_data) == nome_coluna_data] <- "Data"
          } else if (length(nome_coluna_data) > 1) {
            # Se houver mais de uma, pega a primeira
            names(first_sheet_data)[names(first_sheet_data) == nome_coluna_data[1]] <- "Data"
          } else {
            showNotification("Não foi encontrada uma coluna de data no arquivo de exemplo.", type = "error")
          }
          
          raw_df(first_sheet_data)
          file_type("xlsx")
          example_loaded(TRUE)
          # Processa automaticamente os dados de exemplo
          processed_data <- first_sheet_data
          # Supondo que a coluna de data se chame "Data" e formato seja "ymd"
          if ("Data" %in% names(processed_data)) {
            processed_data$Data <- lubridate::ymd(processed_data$Data)
          }
          df(processed_data)
          showNotification("Dados de exemplo carregados e processados!", type = "message")
        } else {
          showNotification("Arquivo de exemplo não encontrado.", type = "error")
        }
      } else {
        example_loaded(FALSE)
        # Limpa dados se voltar para upload
        raw_df(NULL)
        sheet_names(NULL)
        file_type(NULL)
        df(NULL)
      }
    })
    
    # Ajusta a visibilidade do fileInput
    observe({
      shinyjs::toggleState("file", condition = input$data_source == "upload")
    })
    
    output$fileUploaded <- reactive({
      if (input$data_source == "exemplo") {
        return(!is.null(raw_df()))
      } else {
        return(!is.null(input$file))
      }
    })
    outputOptions(output, "fileUploaded", suspendWhenHidden = FALSE)
    
    output$isExcelFile <- reactive({
      return(file_type() == "xlsx")
    })
    outputOptions(output, "isExcelFile", suspendWhenHidden = FALSE)
    
    output$isCsvFile <- reactive({
      return(file_type() == "csv")
    })
    outputOptions(output, "isCsvFile", suspendWhenHidden = FALSE)
    
    # Lógica de upload de arquivo (mantém igual)
    observeEvent(input$file, {
      req(input$file)
      if (input$data_source != "upload") return()
      ext <- tools::file_ext(input$file$name)
      file_type(ext)
      
      if (ext == "xlsx") {
        withProgress(message = 'Carregando arquivo Excel...', value = 0.3, {
          tryCatch({
            incProgress(0.2, detail = "Identificando abas...")
            available_sheets <- readxl::excel_sheets(input$file$datapath)
            sheet_names(available_sheets)
            
            first_sheet_data <- readxl::read_excel(input$file$datapath, sheet = available_sheets[1])
            raw_df(first_sheet_data)
            
            showNotification(paste("Arquivo Excel identificado com", length(available_sheets), "aba(s)."), type = "message")
          }, error = function(e) {
            showNotification(paste("Erro ao ler o arquivo Excel:", e$message), type = "error")
          })
        })
      } else if (ext == "csv") {
        withProgress(message = 'Carregando arquivo CSV...', value = 0.3, {
          tryCatch({
            incProgress(0.2, detail = "Lendo arquivo CSV...")
            sheet_names(c("CSV_Data"))
            
            # Detectar separador automaticamente
            sample_lines <- readLines(input$file$datapath, n = 5)
            comma_count <- sum(stringr::str_count(sample_lines, ","))
            semicolon_count <- sum(stringr::str_count(sample_lines, ";"))
            
            separator <- ifelse(semicolon_count > comma_count, ";", ",")
            
            updateSelectInput(session, "csv_separator", selected = separator)
            
            csv_data <- read.csv(input$file$datapath, sep = separator, stringsAsFactors = FALSE)
            raw_df(csv_data)
            
            showNotification(paste("Arquivo CSV carregado com", nrow(csv_data), "linhas. Separador detectado:", separator), type = "message")
          }, error = function(e) {
            showNotification(paste("Erro ao ler o arquivo CSV:", e$message), type = "error")
          })
        })
      } else {
        showNotification("Por favor, carregue um arquivo XLSX ou CSV.", type = "error")
      }
    })
    
    # ADICIONANDO AS FUNÇÕES FALTANTES DO SERVIDOR
    
    # Seletor de aba do Excel
    output$sheet_selector <- renderUI({
      req(sheet_names())
      if (file_type() == "xlsx") {
        selectInput("selected_sheet", "Selecione a Aba do Excel:",
                    choices = sheet_names(),
                    selected = sheet_names()[1])
      } else {
        NULL
      }
    })
    
    # Seletor de coluna de data
    output$date_column_selector <- renderUI({
      req(raw_df())
      column_names <- names(raw_df())
      
      # Tentar identificar automaticamente colunas que possam ser datas
      possible_date_cols <- column_names[grepl("data|date|periodo|mes|ano|year|month|time", 
                                               tolower(column_names))]
      
      if (length(possible_date_cols) > 0) {
        selected_col <- possible_date_cols[1]
      } else {
        selected_col <- column_names[1]
      }
      
      selectInput("date_column", "Coluna que contém as Datas:",
                  choices = column_names,
                  selected = selected_col)
    })
    
    # Atualizar dados quando aba selecionada mudar (mantém igual)
    observeEvent(input$selected_sheet, {
      req(input$data_source == "upload" || input$data_source == "exemplo")
      req(input$selected_sheet)
      if (input$data_source == "exemplo") {
        selected_data <- readxl::read_excel(EXAMPLE_DATA_PATH, sheet = input$selected_sheet)
        raw_df(selected_data)
        showNotification(paste("Aba", input$selected_sheet, "(exemplo) carregada com", nrow(selected_data), "linhas."), type = "message")
      } else {
        req(input$file, file_type() == "xlsx")
        selected_data <- readxl::read_excel(input$file$datapath, sheet = input$selected_sheet)
        raw_df(selected_data)
        showNotification(paste("Aba", input$selected_sheet, "carregada com", nrow(selected_data), "linhas."), type = "message")
      }
    })
    
    observeEvent(input$process_data, {
      if (input$data_source == "upload") {
        req(input$file, input$date_column, input$date_format)
        if (file_type() == "xlsx") {
          req(input$selected_sheet)
        }
      } else {
        req(input$date_column, input$date_format)
        req(raw_df())
      }
      
      withProgress(message = 'Processando dados...', value = 0.5, {
        tryCatch({
          processed_data <- raw_df()
          date_func <- get(input$date_format, asNamespace("lubridate"))
          processed_data[[input$date_column]] <- date_func(processed_data[[input$date_column]])
          
          if (sum(is.na(processed_data[[input$date_column]])) == nrow(processed_data)) {
            showNotification("Erro: Não foi possível converter a coluna para o formato de data especificado.", type = "error")
            return(NULL)
          }
          
          na_count <- sum(is.na(processed_data[[input$date_column]]))
          if (na_count > 0) {
            showNotification(paste("Atenção:", na_count, "valores de data não puderam ser convertidos e serão tratados como NA."), type = "warning")
          }
          
          if (input$date_column != "Data") {
            names(processed_data)[names(processed_data) == input$date_column] <- "Data"
            showNotification(paste("A coluna", input$date_column, "foi renomeada para 'Data'."), type = "message")
          }
          
          df(processed_data)
          showNotification("Dados processados com sucesso!", type = "message")
        }, error = function(e) {
          showNotification(paste("Erro ao processar dados:", e$message), type = "error")
        })
      })
    })
    
    output$contents <- renderDT({
      req(df())
      datatable(head(df(), 100), 
                options = list(
                  scrollX = TRUE,
                  pageLength = 10,
                  dom = 'Bfrtip',
                  autoWidth = TRUE
                ),
                class = 'cell-border stripe')
    })
    
    output$downloadData <- downloadHandler(
      filename = function() { paste("sample-data-", Sys.Date(), ".csv", sep = "") },
      content = function(file) {
        req(df())
        write.csv(head(df(), 100), file, row.names = FALSE)
      }
    )
    
    # Interface ARIMA
    output$target_selector <- renderUI({
      req(df())
      num_cols <- names(df())[sapply(df(), is.numeric)]
      selectizeInput("target_var", "Variável para Previsão:", 
                     choices = num_cols, 
                     selected = num_cols[1],
                     options = list(
                       placeholder = 'Selecione uma variável...'
                     ))
    })
    
    output$exog_selector <- renderUI({
      req(df(), input$target_var)
      num_cols <- names(df())[sapply(df(), is.numeric)]
      exog_choices <- setdiff(num_cols, input$target_var)
      
      tagList(
        selectizeInput("exog_vars", "Variáveis Explicativas (Exógenas):", 
                       choices = exog_choices, 
                       selected = NULL,
                       multiple = TRUE,
                       options = list(
                         placeholder = 'Selecione uma ou mais variáveis...',
                         plugins = list('remove_button'),
                         closeAfterSelect = FALSE
                       )),
        tags$div(
          id = "colinear_warning",
          style = "display: none; color: #e74c3c; margin-top: 8px; font-size: 12px;",
          icon("exclamation-triangle"), 
          "Atenção: Se as variáveis selecionadas forem colineares, apenas os modelos univariados serão estimados."
        ),
        tags$script("
        $(document).ready(function() {
          $('#exog_vars').on('change', function() {
            if ($(this).val() && $(this).val().length > 1) {
              $('#colinear_warning').show();
            } else {
              $('#colinear_warning').hide();
            }
          });
        });
      ")
      )
    })
    
    output$arima_date_range_ui <- renderUI({
      req(df())
      dates_available <- df()$Data[complete.cases(df())]
      if (length(dates_available) > 0) {
        min_date <- min(dates_available, na.rm = TRUE)
        max_date <- max(dates_available, na.rm = TRUE)
        
        # Calcular data final da previsão como 2 anos após a data máxima
        prediction_end_date <- max_date + lubridate::years(2)
        
        # Atualizar a data final da previsão
        updateDateInput(session, "end_predict_arima", value = prediction_end_date)
        
      } else {
        min_date <- Sys.Date() - lubridate::years(10)
        max_date <- Sys.Date()
      }
      tagList(
        dateRangeInput("arima_date_range", "Período de Análise:",
                       start = min_date,
                       end = max_date,
                       separator = " até "),
        tags$div(
          style = "display: none;",
          textInput("arima_data_inicial", NULL),
          textInput("arima_data_atual", NULL)
        )
      )
    })
    
    observe({
      req(input$arima_date_range)
      updateTextInput(session, "arima_data_inicial", value = as.character(input$arima_date_range[1]))
      updateTextInput(session, "arima_data_atual", value = as.character(input$arima_date_range[2]))
    })
    
    # Observer para atualizar automaticamente a data final da previsão
    # baseada em 2 anos após a data final do período de análise
    observe({
      req(input$arima_date_range)
      end_analysis_date <- input$arima_date_range[2]
      if (!is.null(end_analysis_date)) {
        # Calcular 2 anos após a data final do período de análise
        new_prediction_end <- end_analysis_date + lubridate::years(2)
        updateDateInput(session, "end_predict_arima", value = new_prediction_end)
      }
    })
    
    arima_results <- reactiveVal(NULL)
    
    observeEvent(input$run_arima, {
      req(df(), input$target_var, input$arima_data_inicial, input$arima_data_atual, input$n_test)
      withProgress(message = 'Realizando análise ARIMA...', value = 0, {
        data_inicial <- input$arima_data_inicial
        data_atual <- input$arima_data_atual
        n_teste <- input$n_test
        mode <- input$arima_mode
        
        if (mode == "auto") {
          seasonal <- as.logical(input$seasonal)
          period <- input$period
          num_models <- input$num_models
          arima_params <- NULL
        } else {
          seasonal <- as.logical(input$seasonal_manual)
          period <- input$period_manual
          num_models <- 1
          arima_params <- list(
            p = input$arima_p,
            d = input$arima_d,
            q = input$arima_q,
            P = ifelse(seasonal, input$arima_P, 0),
            D = ifelse(seasonal, input$arima_D, 0),
            Q = ifelse(seasonal, input$arima_Q, 0)
          )
        }
        
        exog_vars <- NULL
        has_collinearity <- FALSE
        
        if (!is.null(input$exog_vars) && length(input$exog_vars) > 0) {
          exog_vars <- as.character(input$exog_vars)
          cat("Variáveis exógenas selecionadas:", paste(exog_vars, collapse=", "), "\n")
          missing_vars <- setdiff(exog_vars, names(df()))
          if (length(missing_vars) > 0) {
            stop(paste("As seguintes variáveis exógenas não existem no conjunto de dados:", 
                       paste(missing_vars, collapse=", ")))
          }
          
          if (length(exog_vars) > 1) {
            filtered_data <- df() %>% 
              filter(Data >= as.Date(data_inicial) & Data <= as.Date(data_atual))
            
            has_collinearity <- check_collinearity(filtered_data, exog_vars)
            
            if (has_collinearity) {
              showNotification(
                "Detectada colinearidade entre as variáveis exógenas. Apenas modelos univariados serão estimados.",
                type = "warning",
                duration = 10
              )
            }
          }
        }
        
        tryCatch({
          if (!("Data" %in% names(df()))) stop("O arquivo deve conter uma coluna chamada 'Data'")
          if (!(input$target_var %in% names(df()))) stop(paste("A variável selecionada", input$target_var, "não existe no conjunto de dados"))
          
          filtered_data <- df() %>% filter(Data >= as.Date(data_inicial) & Data <= as.Date(data_atual))
          if (nrow(filtered_data) < 24) stop("Período selecionado contém poucos dados para análise (mínimo recomendado: 24 pontos)")
          
          incProgress(0.1, detail = "Preparando dados...")
          missing_values <- sum(is.na(filtered_data[[input$target_var]]))
          if (missing_values > 0) {
            showNotification(paste("Atenção: A variável selecionada contém", missing_values, "valores faltantes."), type = "warning")
          }
          
          incProgress(0.2, detail = "Executando modelos ARIMA...")
          
          results <- run_arima_models_extended(
            df = df(),
            target_var = input$target_var,
            exog_vars = exog_vars,
            data_inicial = data_inicial,
            data_atual = data_atual,
            end_predict = input$end_predict_arima,
            n_teste = n_teste,
            seasonal = seasonal,
            period = period,
            num_models = num_models,
            manual_mode = (mode == "manual"),
            arima_params = arima_params
          )
          
          incProgress(0.7, detail = "Processando resultados...")
          
          if (has_collinearity && length(results$models) > 0) {
            multi_models <- grep("multi", names(results$models), value = TRUE)
            if (length(multi_models) == 0 && length(exog_vars) > 0) {
              showNotification(
                "Não foi possível estimar modelos multivariados devido à colinearidade entre as variáveis exógenas.",
                type = "warning",
                duration = 10
              )
            }
          }
          
          arima_results(results)
          showNotification("Análise ARIMA concluída com sucesso!", type = "message")
        }, error = function(e) {
          showNotification(paste("Erro na análise ARIMA:", e$message), type = "error")
        })
      })
    })
    
    output$arima_errors <- renderTable({
      req(arima_results())
      arima_results()$errors %>% 
        select(model, MAPE, RMSE, MAE) %>%
        arrange(RMSE) %>%
        mutate(
          MAPE = round(MAPE, 2),
          RMSE = round(RMSE, 3),
          MAE = round(MAE, 3)
        )
    })
    
    output$arima_plot <- renderPlotly({
      req(arima_results())
      arima_results()$plots[[1]]
    })
    
    output$export_model_selector <- renderUI({
      req(arima_results())
      model_choices <- unique(arima_results()$errors$model)
      pickerInput("export_model", "Escolha o modelo para exportação:", 
                  choices = c("Todos", model_choices), 
                  selected = "Todos",
                  options = list(
                    style = "btn-info"
                  ))
    })
    
    output$download_xlsx <- downloadHandler(
      filename = function() {
        paste("ARIMA-projections-", Sys.Date(), ".xlsx", sep = "")
      },
      content = function(file) {
        req(arima_results())
        
        periods <- arima_results()$periods
        realized_end <- as.Date(periods$train_end)
        forecast_start <- as.Date(periods$forecast_start)
        end_predict <- as.Date(arima_results()$end_predict)
        
        all_dates <- seq.Date(from = as.Date(periods$train_start), to = end_predict, by = "month")
        df_all_dates <- data.frame(Data = all_dates)
        
        df_historical <- df() %>% 
          filter(Data >= as.Date(periods$train_start) & Data <= realized_end) %>%
          select(Data, ec_res = all_of(input$target_var))
        
        if(input$export_model != "Todos" && input$export_model %in% names(arima_results()$models)) {
          model_name <- input$export_model
          model_obj <- arima_results()$models[[model_name]]
          
          future_dates <- seq.Date(from = forecast_start, to = end_predict, by = "month")
          h_future <- length(future_dates)
          
          if(h_future > 0) {
            future_forecast <- forecast(model_obj, h = h_future)
            df_forecast <- data.frame(
              Data = future_dates,
              forecast_value = as.numeric(future_forecast$mean)
            )
          } else {
            df_forecast <- data.frame(Data = as.Date(character()), forecast_value = numeric(0))
          }
          
          df_export <- full_join(df_all_dates, df_historical, by = "Data") %>%
            full_join(df_forecast, by = "Data") %>%
            mutate(
              final_value = ifelse(is.na(ec_res), forecast_value, ec_res),
              Data_Type = ifelse(Data <= realized_end, "Realizado", "Projetado")
            )
          
        } else {
          df_export <- arima_results()$df_arima
        }
        
        df_export <- df_export %>% arrange(Data)
        write_xlsx(df_export, path = file)
        cat("Exportação concluída para:", file, "\n")
      }
    )
    
    ####### ABA ANÁLISE DE CRESCIMENTO ########
    
    # Armazenar os resultados das análises de crescimento
    growth_analysis_results <- reactiveVal(NULL)
    
    # Seletor de modelo para análise de crescimento
    output$growth_model_selector <- renderUI({
      req(arima_results())
      model_choices <- unique(arima_results()$errors$model)
      # Ordenar por RMSE para mostrar os melhores modelos primeiro
      best_models <- arima_results()$errors %>%
        arrange(RMSE) %>%
        pull(model)
      
      selectInput("growth_model", "Selecione o Modelo:",
                  choices = best_models,
                  selected = best_models[1])
    })
    
    # Filtro de período para os gráficos
    output$filtro_periodo <- renderUI({
      req(growth_analysis_results())
      
      if (input$periodo_analise == "mensal") {
        anos_disponiveis <- growth_analysis_results()$Tabela_Mensal$Ano
      } else if (input$periodo_analise == "trimestral") {
        anos_disponiveis <- growth_analysis_results()$Tabela_Trimestral$Ano
      } else if (input$periodo_analise == "semestral") {
        anos_disponiveis <- growth_analysis_results()$Tabela_Semestral$Ano
      } else {
        anos_disponiveis <- growth_analysis_results()$Tabela_Anual$Ano
      }
      
      anos_disponiveis <- unique(anos_disponiveis)
      
      sliderInput("range_anos", "Filtrar Anos:",
                  min = min(anos_disponiveis),
                  max = max(anos_disponiveis),
                  value = c(min(anos_disponiveis), max(anos_disponiveis)),
                  step = 1,
                  ticks = TRUE,
                  round = TRUE)
    })
    
    # Função para preparar dataframe para análise de crescimento
    prepare_growth_data <- function(arima_df, model_name, target_var) {
      # Filtrar apenas as colunas relevantes
      df_selected <- arima_df %>%
        select(Data, !!as.name(paste0("Pred_", model_name)))
      
      # Renomear a coluna de previsão para o nome da variável alvo
      names(df_selected)[2] <- target_var
      
      return(df_selected)
    }
    
    # Executar análise de crescimento quando o botão for clicado
    observeEvent(input$analisar_crescimento, {
      req(arima_results(), input$growth_model, input$target_var)
      
      withProgress(message = 'Analisando taxas de crescimento...', value = 0.3, {
        # Preparar dados para análise
        df_arima <- arima_results()$df_arima
        
        # Preparar o dataframe para a função generate_df_visoes
        df_growth <- prepare_growth_data(df_arima, input$growth_model, input$target_var)
        
        incProgress(0.4, detail = "Gerando análises...")
        
        # Gerar as visões de crescimento
        visoes_result <- tryCatch({
          generate_df_visoes(
            df = df_growth,
            variable = input$target_var,
            title = paste("Análise de Crescimento -", input$growth_model),
            date_col = "Data"
          )
        }, error = function(e) {
          showNotification(paste("Erro na análise de crescimento:", e$message), type = "error")
          NULL
        })
        
        if (!is.null(visoes_result)) {
          growth_analysis_results(visoes_result)
          showNotification("Análise de crescimento concluída com sucesso!", type = "message")
        }
      })
    })
    
    # Função para reordenar e renomear colunas
    reorder_and_rename_columns <- function(tabela, periodo) {
      if (periodo == "mensal") {
        # Ordem cronológica dos meses
        month_order <- c("Ano", "Jan", "Fev", "Mar", "Abr", "Mai", "Jun", 
                         "Jul", "Ago", "Set", "Out", "Nov", "Dez")
        
        # Renomear colunas de taxa de crescimento
        names(tabela) <- gsub("Mensal_Taxa_Crescimento_", "Mes_txc_", names(tabela))
        names(tabela) <- gsub("Mensal_Variacao_Valor_", "Mes_var_", names(tabela))
        
        # Reordenar colunas existentes
        existing_cols <- names(tabela)[names(tabela) %in% month_order]
        other_cols <- names(tabela)[!names(tabela) %in% month_order]
        
        # Reordenar mantendo a ordem cronológica
        ordered_cols <- c(existing_cols[order(match(existing_cols, month_order))], other_cols)
        tabela <- tabela[, ordered_cols]
        
      } else if (periodo == "trimestral") {
        # Ordem cronológica dos trimestres
        quarter_order <- c("Ano", "Q1", "Q2", "Q3", "Q4")
        
        # Renomear colunas de taxa de crescimento
        names(tabela) <- gsub("Trimestral_Taxa_Crescimento_", "Trim_txc_", names(tabela))
        names(tabela) <- gsub("Trimestral_Variacao_Valor_", "Trim_var_", names(tabela))
        
        # Reordenar colunas existentes
        existing_cols <- names(tabela)[names(tabela) %in% quarter_order]
        other_cols <- names(tabela)[!names(tabela) %in% quarter_order]
        
        # Reordenar mantendo a ordem cronológica
        ordered_cols <- c(existing_cols[order(match(existing_cols, quarter_order))], other_cols)
        tabela <- tabela[, ordered_cols]
        
      } else if (periodo == "semestral") {
        # Ordem cronológica dos semestres
        semester_order <- c("Ano", "S1", "S2")
        
        # Renomear colunas de taxa de crescimento
        names(tabela) <- gsub("Semestral_Taxa_Crescimento_", "Sem_txc_", names(tabela))
        names(tabela) <- gsub("Semestral_Variacao_Valor_", "Sem_var_", names(tabela))
        
        # Reordenar colunas existentes
        existing_cols <- names(tabela)[names(tabela) %in% semester_order]
        other_cols <- names(tabela)[!names(tabela) %in% semester_order]
        
        # Reordenar mantendo a ordem cronológica
        ordered_cols <- c(existing_cols[order(match(existing_cols, semester_order))], other_cols)
        tabela <- tabela[, ordered_cols]
        
      } else if (periodo == "anual") {
        # Renomear colunas de taxa de crescimento
        names(tabela) <- gsub("Anual_Taxa_Crescimento_", "Ano_txc_", names(tabela))
        names(tabela) <- gsub("Anual_Variacao_Valor_", "Ano_var_", names(tabela))
      }
      
      return(tabela)
    }
    
    # Renderizar tabelas de valores
    output$tabela_valores <- renderDT({
      req(growth_analysis_results(), input$periodo_analise)
      
      if (input$periodo_analise == "mensal") {
        tabela <- growth_analysis_results()$Tabela_Mensal
        titulo <- "Valores Mensais"
      } else if (input$periodo_analise == "trimestral") {
        tabela <- growth_analysis_results()$Tabela_Trimestral
        titulo <- "Valores Trimestrais"
      } else if (input$periodo_analise == "semestral") {
        tabela <- growth_analysis_results()$Tabela_Semestral
        titulo <- "Valores Semestrais"
      } else {
        tabela <- growth_analysis_results()$Tabela_Anual
        titulo <- "Valores Anuais"
      }
      
      # Reordenar e renomear colunas
      tabela <- reorder_and_rename_columns(tabela, input$periodo_analise)
      
      # Processar tabela para exibição
      tabela_processada <- process_table(tabela)
      
      # Formatar e retornar a tabela
      DT::datatable(
        tabela_processada,
        caption = titulo,
        rownames = FALSE,
        options = list(
          pageLength = -1,  # Mostrar todas as linhas
          scrollX = TRUE,
          dom = 'Bfrtip',
          buttons = c('copy', 'csv', 'excel', 'pdf'),
          columnDefs = list(
            list(className = 'dt-center', targets = '_all')
          )
        ),
        extensions = 'Buttons'
      ) %>%
        DT::formatRound(columns = 2:ncol(tabela_processada), digits = 2)
    })
    
    # Renderizar tabelas de taxas de crescimento
    output$tabela_taxas <- renderDT({
      req(growth_analysis_results(), input$periodo_analise)
      
      if (input$periodo_analise == "mensal") {
        # Extrair apenas as colunas de taxa de crescimento
        cols_taxa <- grep("Taxa_Crescimento", names(growth_analysis_results()$Tabela_Mensal), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Mensal %>%
          select(Ano, all_of(cols_taxa))
        titulo <- "Taxas de Crescimento Mensais (%)"
      } else if (input$periodo_analise == "trimestral") {
        cols_taxa <- grep("Taxa_Crescimento", names(growth_analysis_results()$Tabela_Trimestral), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Trimestral %>%
          select(Ano, all_of(cols_taxa))
        titulo <- "Taxas de Crescimento Trimestrais (%)"
      } else if (input$periodo_analise == "semestral") {
        cols_taxa <- grep("Taxa_Crescimento", names(growth_analysis_results()$Tabela_Semestral), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Semestral %>%
          select(Ano, all_of(cols_taxa))
        titulo <- "Taxas de Crescimento Semestrais (%)"
      } else {
        cols_taxa <- grep("Taxa_Crescimento", names(growth_analysis_results()$Tabela_Anual), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Anual %>%
          select(Ano, all_of(cols_taxa))
        titulo <- "Taxas de Crescimento Anuais (%)"
      }
      
      # Reordenar e renomear colunas
      tabela <- reorder_and_rename_columns(tabela, input$periodo_analise)
      
      # Processar tabela para exibição
      tabela_processada <- process_table(tabela)
      
      # Formatar e retornar a tabela
      DT::datatable(
        tabela_processada,
        caption = titulo,
        rownames = FALSE,
        options = list(
          pageLength = -1,  # Mostrar todas as linhas
          scrollX = TRUE,
          dom = 'Bfrtip',
          buttons = c('copy', 'csv', 'excel', 'pdf'),
          columnDefs = list(
            list(className = 'dt-center', targets = '_all')
          )
        ),
        extensions = 'Buttons'
      ) %>%
        DT::formatRound(columns = 2:ncol(tabela_processada), digits = 2) %>%
        DT::formatStyle(
          columns = 2:ncol(tabela_processada),
          color = styleInterval(c(-0.1, 0, 5, 10), c('#c0392b', '#e67e22', '#2c3e50', '#27ae60', '#16a085')),
          fontWeight = styleInterval(c(5), c('normal', 'bold'))
        )
    })
    
    # Renderizar tabelas de variações
    output$tabela_variacoes <- renderDT({
      req(growth_analysis_results(), input$periodo_analise)
      
      if (input$periodo_analise == "mensal") {
        # Extrair apenas as colunas de variações
        cols_var <- grep("Variacao_Valor", names(growth_analysis_results()$Tabela_Mensal), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Mensal %>%
          select(Ano, all_of(cols_var))
        titulo <- "Variações Absolutas Mensais"
      } else if (input$periodo_analise == "trimestral") {
        cols_var <- grep("Variacao_Valor", names(growth_analysis_results()$Tabela_Trimestral), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Trimestral %>%
          select(Ano, all_of(cols_var))
        titulo <- "Variações Absolutas Trimestrais"
      } else if (input$periodo_analise == "semestral") {
        cols_var <- grep("Variacao_Valor", names(growth_analysis_results()$Tabela_Semestral), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Semestral %>%
          select(Ano, all_of(cols_var))
        titulo <- "Variações Absolutas Semestrais"
      } else {
        cols_var <- grep("Variacao_Valor", names(growth_analysis_results()$Tabela_Anual), value = TRUE)
        tabela <- growth_analysis_results()$Tabela_Anual %>%
          select(Ano, all_of(cols_var))
        titulo <- "Variações Absolutas Anuais"
      }
      
      # Reordenar e renomear colunas
      tabela <- reorder_and_rename_columns(tabela, input$periodo_analise)
      
      # Processar tabela para exibição
      tabela_processada <- process_table(tabela)
      
      # Formatar e retornar a tabela
      DT::datatable(
        tabela_processada,
        caption = titulo,
        rownames = FALSE,
        options = list(
          pageLength = -1,  # Mostrar todas as linhas
          scrollX = TRUE,
          dom = 'Bfrtip',
          buttons = c('copy', 'csv', 'excel', 'pdf'),
          columnDefs = list(
            list(className = 'dt-center', targets = '_all')
          )
        ),
        extensions = 'Buttons'
      ) %>%
        DT::formatRound(columns = 2:ncol(tabela_processada), digits = 2) %>%
        DT::formatStyle(
          columns = 2:ncol(tabela_processada),
          color = styleInterval(c(-1000, 0, 1000), c('#c0392b', '#e67e22', '#2c3e50', '#27ae60')),
          fontWeight = styleInterval(c(1000), c('normal', 'bold'))
        )
    })
    
    # Renderizar gráfico de crescimento
    output$grafico_crescimento <- renderPlotly({
      req(growth_analysis_results(), input$periodo_analise, input$tipo_grafico, input$tipo_plotly)
      
      # Selecionar o tipo de dataframe de acordo com o período
      if (input$periodo_analise == "mensal") {
        df_viz <- growth_analysis_results()$Tabela_Mensal_long
      } else if (input$periodo_analise == "trimestral") {
        df_viz <- growth_analysis_results()$Tabela_Trimestral_long
      } else if (input$periodo_analise == "semestral") {
        df_viz <- growth_analysis_results()$Tabela_semestral_long
      } else {
        df_viz <- growth_analysis_results()$Tabela_Anual_long
      }
      
      # Filtrar por anos selecionados, se disponível
      if (!is.null(input$range_anos)) {
        df_viz <- df_viz %>%
          filter(year(Data) >= input$range_anos[1] & year(Data) <= input$range_anos[2])
      }
      
      # Determinar qual coluna usar com base no tipo de gráfico
      if (input$tipo_grafico == "valores") {
        y_col <- "Valor"
        titulo <- paste("Valores", ifelse(input$periodo_analise == "mensal", "Mensais",
                                          ifelse(input$periodo_analise == "trimestral", "Trimestrais",
                                                 ifelse(input$periodo_analise == "semestral", "Semestrais", "Anuais"))))
        y_label <- "Valor"
      } else if (input$tipo_grafico == "taxas") {
        if (input$periodo_analise == "mensal") {
          y_col <- "Taxa_Crescimento"
        } else if (input$periodo_analise == "trimestral") {
          y_col <- "Taxa_Crescimento"
        } else if (input$periodo_analise == "semestral") {
          y_col <- "Taxa_Crescimento_Semestral"
        } else {
          y_col <- "Taxa_Crescimento_Anual"
        }
        titulo <- paste("Taxas de Crescimento", ifelse(input$periodo_analise == "mensal", "Mensais",
                                                       ifelse(input$periodo_analise == "trimestral", "Trimestrais",
                                                              ifelse(input$periodo_analise == "semestral", "Semestrais", "Anuais"))), "(%)")
        y_label <- "Taxa de Crescimento (%)"
      } else {  # variações
        if (input$periodo_analise == "mensal") {
          y_col <- "Variacao_Valor"
        } else if (input$periodo_analise == "trimestral") {
          y_col <- "Variacao_Valor"
        } else if (input$periodo_analise == "semestral") {
          y_col <- "Variacao_Semestral"
        } else {
          y_col <- "Variacao_Valor_Anual"
        }
        titulo <- paste("Variações Absolutas", ifelse(input$periodo_analise == "mensal", "Mensais",
                                                      ifelse(input$periodo_analise == "trimestral", "Trimestrais",
                                                             ifelse(input$periodo_analise == "semestral", "Semestrais", "Anuais"))))
        y_label <- "Variação Absoluta"
      }
      
      # Gerar gráfico de acordo com o tipo selecionado
      if (input$tipo_plotly == "lines") {
        fig <- plot_ly(df_viz, x = ~Data, y = as.formula(paste0("~", y_col)), 
                       type = 'scatter', mode = 'lines+markers',
                       line = list(color = '#5dade2', width = 3),
                       marker = list(color = '#2c3e50', size = 8))
      } else if (input$tipo_plotly == "bars") {
        fig <- plot_ly(df_viz, x = ~Data, y = as.formula(paste0("~", y_col)), 
                       type = 'bar',
                       marker = list(color = ~ifelse(get(y_col) < 0, '#e74c3c', '#27ae60')))
      } else {  # combinado
        fig <- plot_ly()
        fig <- fig %>% add_bars(data = df_viz, x = ~Data, y = as.formula(paste0("~", y_col)),
                                name = "Valores",
                                marker = list(color = ~ifelse(get(y_col) < 0, '#e74c3c', '#27ae60')))
        fig <- fig %>% add_trace(data = df_viz, x = ~Data, y = as.formula(paste0("~", y_col)),
                                 type = 'scatter', mode = 'lines+markers',
                                 name = "Tendência",
                                 line = list(color = '#5dade2', width = 2),
                                 marker = list(color = '#2c3e50', size = 6))
      }
      
      # Formatar o layout
      fig <- fig %>% layout(
        title = list(text = titulo, font = list(size = 16, color = "#2c3e50")),
        xaxis = list(title = "Data", tickangle = 45, titlefont = list(color = "#2c3e50")),
        yaxis = list(title = y_label, titlefont = list(color = "#2c3e50")),
        margin = list(l = 60, r = 40, b = 80, t = 80),
        legend = list(orientation = 'h', x = 0.5, y = 1.1, xanchor = 'center'),
        hovermode = 'closest',
        plot_bgcolor = "#f8f9fa",
        paper_bgcolor = "#ffffff"
      )
      
      # Se for taxa de crescimento ou variação, adicionar uma linha de referência no zero
      if (input$tipo_grafico != "valores") {
        fig <- fig %>% layout(
          shapes = list(
            list(
              type = "line",
              x0 = min(df_viz$Data),
              x1 = max(df_viz$Data),
              y0 = 0,
              y1 = 0,
              line = list(color = "#95a5a6", width = 1, dash = "dot")
            )
          )
        )
      }
      
      return(fig)
    })
    
    # Handler para download da análise em Excel
    output$download_crescimento_xlsx <- downloadHandler(
      filename = function() {
        paste("analise-crescimento-", input$growth_model, "-", Sys.Date(), ".xlsx", sep = "")
      },
      content = function(file) {
        req(growth_analysis_results())
        
        # Aplicar reordenação e renomeação às tabelas antes de exportar
        tabela_mensal <- reorder_and_rename_columns(growth_analysis_results()$Tabela_Mensal, "mensal")
        tabela_trimestral <- reorder_and_rename_columns(growth_analysis_results()$Tabela_Trimestral, "trimestral")
        tabela_semestral <- reorder_and_rename_columns(growth_analysis_results()$Tabela_Semestral, "semestral")
        tabela_anual <- reorder_and_rename_columns(growth_analysis_results()$Tabela_Anual, "anual")
        
        # Criar lista de planilhas para o arquivo Excel
        sheets <- list(
          "Valores_Mensais" = process_table(tabela_mensal),
          "Valores_Trimestrais" = process_table(tabela_trimestral),
          "Valores_Semestrais" = process_table(tabela_semestral),
          "Valores_Anuais" = process_table(tabela_anual),
          "Dados_Mensais_Long" = process_table(growth_analysis_results()$Tabela_Mensal_long),
          "Dados_Trimestrais_Long" = process_table(growth_analysis_results()$Tabela_Trimestral_long),
          "Dados_Semestrais_Long" = process_table(growth_analysis_results()$Tabela_semestral_long),
          "Dados_Anuais_Long" = process_table(growth_analysis_results()$Tabela_Anual_long)
        )
        
        # Exportar para Excel
        write_xlsx(sheets, path = file)
      }
    )
    
    # Handler para download do relatório em PDF 
    output$download_crescimento_pdf <- downloadHandler(
      filename = function() {
        paste("relatorio-crescimento-", input$growth_model, "-", Sys.Date(), ".pdf", sep = "")
      },
      content = function(file) {
        showNotification("Funcionalidade de exportação para PDF ainda não implementada.", type = "warning")
      }
    )
  }
}

#======================================================================
# RUN
#======================================================================
shinyApp(ui_arima, server_arima)