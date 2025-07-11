# Fix for the projection function and visualization issues



library(lavaan)
library(dplyr)
library(readxl)
library(plotly)
library(lubridate)
library(tidyr)

treinar_e_projetar_modelo <- function(dados_completos, coluna_data = "data", 
                                      tipo_modelo = "lm", # Novo parâmetro para escolher o tipo de modelo
                                      alpha = 0.5,       # Parâmetro de elastic net (0 = ridge, 1 = lasso)
                                      lambda = NULL) {   # Parâmetro de regularização (NULL = escolha automática)
  
  # Carregar pacotes necessários
  if(tipo_modelo == "glmnet" & !requireNamespace("glmnet", quietly = TRUE)) {
    stop("Pacote 'glmnet' é necessário para usar o modelo tipo GLMNET. Instale com install.packages('glmnet')")
  }
  
  # Função para ajustar um modelo usando glmnet
  ajustar_glmnet <- function(x, y, alpha = 0.5, lambda = NULL) {
    require(glmnet)
    
    # Se lambda não for especificado, escolha o melhor por validação cruzada
    if(is.null(lambda)) {
      cv_model <- cv.glmnet(x, y, alpha = alpha)
      lambda <- cv_model$lambda.min
      message(paste("Lambda ótimo escolhido por validação cruzada:", lambda))
    }
    
    # Ajuste do modelo com lambda especificado/escolhido
    model <- glmnet(x, y, alpha = alpha, lambda = lambda)
    
    return(list(model = model, lambda = lambda))
  }
  
  # Função para projetar usando modelos GLMNET
  projetar_glmnet <- function(modelo, x_new, lambda = NULL) {
    require(glmnet)
    if(is.null(lambda) && !is.null(modelo$lambda)) {
      lambda <- modelo$lambda
    }
    predict(modelo, newx = x_new, s = lambda)
  }
  
  # Função para verificar NA em um data frame
  verificar_na <- function(df, nome = "dataframe") {
    na_count <- colSums(is.na(df))
    if(sum(na_count) > 0) {
      message(paste("Valores NA detectados em", nome, ":"))
      print(na_count[na_count > 0])
      
      # Verificar se há linhas completas
      linhas_completas <- sum(complete.cases(df))
      total_linhas <- nrow(df)
      message(paste("Linhas completas:", linhas_completas, "de", total_linhas, 
                    "(" , round(linhas_completas/total_linhas*100, 1), "%)"))
    } else {
      message(paste("Não foram detectados valores NA em", nome))
    }
  }
  
  # Função para projetar consumo de energia com qualquer tipo de modelo (LM ou GLMNET)
  projetar_consumo_energia <- function(modelos, dados_projecao, dados_historicos = NULL) {
    
    # Verificar NA nos dados de projeção
    verificar_na(dados_projecao, "dados de projeção")
    
    # Initialize trend if necessary
    if(!"trend" %in% names(dados_projecao)) {
      if(!is.null(dados_historicos)) {
        # If we have historical data, continue the trend
        n_hist <- nrow(dados_historicos)
        n_proj <- nrow(dados_projecao)
        
        # Create continuous time sequence
        dados_projecao$trend <- (n_hist + 1):(n_hist + n_proj)
        
        # Get scaling parameters from historical data
        if("trend_scaled" %in% names(dados_historicos)) {
          trend_mean <- attr(dados_historicos$trend_scaled, "scaled:center")
          trend_sd <- attr(dados_historicos$trend_scaled, "scaled:scale")
          if(is.null(trend_mean) || is.null(trend_sd)) {
            trend_mean <- mean(dados_historicos$trend)
            trend_sd <- sd(dados_historicos$trend)
          }
        } else {
          trend_mean <- mean(dados_historicos$trend)
          trend_sd <- sd(dados_historicos$trend)
        }
        
        # Apply the same transformation
        dados_projecao$trend_scaled <- (dados_projecao$trend - trend_mean) / trend_sd
      } else {
        # If we don't have historical data, start from 1
        dados_projecao$trend <- 1:nrow(dados_projecao)
        dados_projecao$trend_scaled <- scale(dados_projecao$trend)[,1]
      }
    }
    
    # Create dataframe to store results
    resultados <- dados_projecao
    
    # Verificar se temos valores completos para projeção
    colunas_necessarias <- c("log_IPCA", "log_Populacao", "log_Temperatura", 
                             "log_Chuva", "log_PrecoEnergia", "trend_scaled")
    
    linhas_completas <- complete.cases(dados_projecao[, colunas_necessarias])
    if(sum(linhas_completas) == 0) {
      message("ERRO: Não há linhas completas com variáveis explicativas nos dados de projeção.")
      return(resultados)
    } else if(sum(linhas_completas) < nrow(dados_projecao)) {
      message(paste("AVISO: Apenas", sum(linhas_completas), "de", nrow(dados_projecao), 
                    "linhas têm valores completos para projeção."))
      # Filtrar apenas linhas completas para projeção
      dados_projecao_completos <- dados_projecao[linhas_completas, ]
    } else {
      dados_projecao_completos <- dados_projecao
    }
    
    tryCatch({
      # Verificar o tipo de modelo
      if("glmnet" %in% class(modelos$PIB)) {
        # GLMNET workflow
        message("Usando modelos GLMNET para projeção")
        
        # Criar matrizes para predição
        X_PIB <- as.matrix(dados_projecao_completos[, c("log_IPCA", "trend_scaled")])
        resultados$log_PIB[linhas_completas] <- as.numeric(projetar_glmnet(modelos$PIB, X_PIB, modelos$PIB$lambda))
        
        X_MassaRenda <- as.matrix(dados_projecao_completos[, c("log_Populacao", "log_IPCA", 
                                                               "log_PIB", "trend_scaled")])
        resultados$log_MassaRenda[linhas_completas] <- as.numeric(projetar_glmnet(modelos$MassaRenda, X_MassaRenda, modelos$MassaRenda$lambda))
        
        X_PIM <- as.matrix(dados_projecao_completos[, c("log_PIB", "trend_scaled")])
        resultados$log_PIM[linhas_completas] <- as.numeric(projetar_glmnet(modelos$PIM, X_PIM, modelos$PIM$lambda))
        
        X_PMC <- as.matrix(dados_projecao_completos[, c("log_PIB", "log_MassaRenda", "trend_scaled")])
        resultados$log_PMC[linhas_completas] <- as.numeric(projetar_glmnet(modelos$PMC, X_PMC, modelos$PMC$lambda))
        
        X_Energia <- as.matrix(dados_projecao_completos[, c("log_PIB", "log_PIM", "log_PMC", 
                                                            "log_MassaRenda", "log_Temperatura", 
                                                            "log_Chuva", "log_PrecoEnergia", 
                                                            "trend_scaled")])
        resultados$log_ConsumoEnergia[linhas_completas] <- as.numeric(projetar_glmnet(modelos$Energia, X_Energia, modelos$Energia$lambda))
        
      } else {
        # LM workflow (como antes)
        message("Usando modelos de regressão linear para projeção")
        
        coefs <- list(
          PIB = coef(modelos$PIB),
          MassaRenda = coef(modelos$MassaRenda),
          PIM = coef(modelos$PIM),
          PMC = coef(modelos$PMC),
          Energia = coef(modelos$Energia)
        )
        
        # Projetar apenas linhas completas
        # Project PIB
        resultados$log_PIB[linhas_completas] <- coefs$PIB["(Intercept)"] + 
          coefs$PIB["log_IPCA"] * dados_projecao_completos$log_IPCA + 
          coefs$PIB["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Project MassaRenda
        resultados$log_MassaRenda[linhas_completas] <- coefs$MassaRenda["(Intercept)"] + 
          coefs$MassaRenda["log_Populacao"] * dados_projecao_completos$log_Populacao +
          coefs$MassaRenda["log_IPCA"] * dados_projecao_completos$log_IPCA +
          coefs$MassaRenda["log_PIB"] * dados_projecao_completos$log_PIB +
          coefs$MassaRenda["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Project PIM
        resultados$log_PIM[linhas_completas] <- coefs$PIM["(Intercept)"] + 
          coefs$PIM["log_PIB"] * dados_projecao_completos$log_PIB +
          coefs$PIM["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Project PMC
        resultados$log_PMC[linhas_completas] <- coefs$PMC["(Intercept)"] + 
          coefs$PMC["log_PIB"] * dados_projecao_completos$log_PIB +
          coefs$PMC["log_MassaRenda"] * dados_projecao_completos$log_MassaRenda +
          coefs$PMC["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Finally, project ConsumoEnergia
        resultados$log_ConsumoEnergia[linhas_completas] <- coefs$Energia["(Intercept)"] + 
          coefs$Energia["log_PIB"] * dados_projecao_completos$log_PIB +
          coefs$Energia["log_PIM"] * dados_projecao_completos$log_PIM +
          coefs$Energia["log_PMC"] * dados_projecao_completos$log_PMC +
          coefs$Energia["log_MassaRenda"] * dados_projecao_completos$log_MassaRenda +
          coefs$Energia["log_Temperatura"] * dados_projecao_completos$log_Temperatura +
          coefs$Energia["log_Chuva"] * dados_projecao_completos$log_Chuva +
          coefs$Energia["log_PrecoEnergia"] * dados_projecao_completos$log_PrecoEnergia +
          coefs$Energia["trend_scaled"] * dados_projecao_completos$trend_scaled
      }
      
      # Convert to original scale
      resultados$ConsumoEnergia_Projetado <- exp(resultados$log_ConsumoEnergia)
      
      message("Projeções calculadas com sucesso")
      
    }, error = function(e) {
      message("Erro ao calcular projeções: ", e$message)
    })
    
    # Verificar resultados
    verificar_na(resultados[, c("log_ConsumoEnergia", "ConsumoEnergia_Projetado")], "resultados das projeções")
    
    return(resultados)
  }
  
  # Verificar se a coluna de data existe
  if (!(coluna_data %in% names(dados_completos))) {
    stop(paste("A coluna", coluna_data, "não existe no conjunto de dados."))
  }
  
  # Criar uma cópia dos dados para não alterar o original
  dados <- dados_completos
  
  # Converter a coluna de data para o formato Date se não estiver
  if (!inherits(dados[[coluna_data]], "Date")) {
    dados[[coluna_data]] <- as.Date(dados[[coluna_data]])
    message("Coluna de data convertida para o formato Date.")
  }
  
  # Verificar se todas as colunas necessárias existem
  mapeamento_colunas <- list(
    "Date" = coluna_data,
    "PIB" = "PIB_4i",
    "MassaRenda" = "massa_total",
    "PIM" = "pim_geral",
    "PMC" = "pmc",
    "ConsumoEnergia" = "ei_total",
    "IPCA" = "indice_de_precos",
    "Populacao" = "pop_domicilio",
    "Temperatura" = "Temperatura_media",
    "Chuva" = "Pluviometria_total",
    "PrecoEnergia" = "Tarifa_res"  # Assumindo que a tarifa residencial é a principal
  )
  
  colunas_necessarias <- unlist(mapeamento_colunas)
  colunas_faltantes <- colunas_necessarias[!colunas_necessarias %in% names(dados)]
  
  if(length(colunas_faltantes) > 0) {
    stop(paste("As seguintes colunas estão faltando no conjunto de dados:", 
               paste(colunas_faltantes, collapse = ", ")))
  }
  
  # Criar um dataframe com as colunas renomeadas para o padrão esperado pelo modelo
  dados_padronizados <- dados %>%
    rename(
      Date = !!mapeamento_colunas$Date,
      PIB = !!mapeamento_colunas$PIB,
      MassaRenda = !!mapeamento_colunas$MassaRenda,
      PIM = !!mapeamento_colunas$PIM,
      PMC = !!mapeamento_colunas$PMC,
      ConsumoEnergia = !!mapeamento_colunas$ConsumoEnergia,
      IPCA = !!mapeamento_colunas$IPCA,
      Populacao = !!mapeamento_colunas$Populacao,
      Temperatura = !!mapeamento_colunas$Temperatura,
      Chuva = !!mapeamento_colunas$Chuva,
      PrecoEnergia = !!mapeamento_colunas$PrecoEnergia
    ) %>%
    # Selecionar apenas as colunas necessárias para o modelo
    select(Date, PIB, MassaRenda, PIM, PMC, ConsumoEnergia, IPCA, Populacao, Temperatura, Chuva, PrecoEnergia)
  
  # Exibir as primeiras linhas para diagnóstico
  message("Primeiras linhas dos dados padronizados:")
  print(head(dados_padronizados))
  
  # Verificar se há valores ausentes ou infinitos
  verificar_na(dados_padronizados, "dados padronizados")
  
  # Determinar a data de corte como a última data não faltante na coluna alvo
  dados_com_alvo <- dados_padronizados %>%
    filter(!is.na(ConsumoEnergia)) %>%
    arrange(Date)
  
  data_corte <- max(dados_com_alvo$Date)
  message(paste("Data de corte automática definida como:", format(data_corte, "%Y-%m-%d")))
  
  # Aplicar transformações logarítmicas e criar a variável de tendência
  dados_padronizados <- dados_padronizados %>%
    arrange(Date) %>%
    mutate(
      trend = row_number(),
      trend_scaled = scale(trend)[,1],
      log_PIB = log(PIB),
      log_MassaRenda = log(MassaRenda),
      log_PIM = log(PIM),
      log_PMC = log(PMC),
      log_ConsumoEnergia = ifelse(is.na(ConsumoEnergia), NA, log(ConsumoEnergia)),
      log_IPCA = log(IPCA),
      log_Populacao = log(Populacao),
      log_Temperatura = log(Temperatura),
      log_Chuva = log(Chuva + 1),
      log_PrecoEnergia = log(PrecoEnergia)
    )
  
  # Verificar NA após transformações
  verificar_na(dados_padronizados, "dados após transformações logarítmicas")
  
  # Split the data into training and testing
  dados_treinamento <- dados_padronizados %>%
    filter(Date <= data_corte) %>%
    filter(!is.na(ConsumoEnergia))  # Remover linhas com valores NA em ConsumoEnergia
  
  # Verificar NA nos dados de treinamento
  verificar_na(dados_treinamento, "dados de treinamento")
  
  # Get all dates after the cutoff where we have exogenous variables
  dados_projecao <- dados_padronizados %>%
    filter(Date > data_corte)
  
  # Se não houver dados para projeção após a data de corte, usar os dados com NA na variável alvo
  if (nrow(dados_projecao) == 0) {
    message("Não foram encontrados dados após a data de corte. Usando registros com valores ausentes na variável alvo.")
    dados_projecao <- dados_padronizados %>%
      filter(is.na(ConsumoEnergia)) %>%
      arrange(Date)
  }
  
  # Verificar NA nos dados de projeção
  verificar_na(dados_projecao, "dados de projeção")
  
  # Number of months to project
  meses_futuros <- nrow(dados_projecao)
  
  if(meses_futuros == 0) {
    stop("Não há dados para realizar projeções. Verifique se existem valores ausentes na variável alvo ou dados após a data de corte.")
  }
  
  # Substituir NA por valores imputados nas variáveis exógenas para projeção
  # Isso é importante para evitar problemas na projeção
  dados_projecao_imputado <- dados_projecao
  
  # Funções de imputação simples
  imputar_na <- function(x) {
    if(all(is.na(x))) {
      return(x)  # Se todos são NA, manter NA
    }
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    return(x)
  }
  
  # Imputar valores ausentes nas variáveis exógenas
  variaveis_exogenas <- c("log_IPCA", "log_Populacao", "log_Temperatura", 
                          "log_Chuva", "log_PrecoEnergia", "trend_scaled")
  
  for(var in variaveis_exogenas) {
    if(any(is.na(dados_projecao_imputado[[var]]))) {
      dados_projecao_imputado[[var]] <- imputar_na(dados_projecao_imputado[[var]])
      message(paste("Valores ausentes imputados na variável", var))
    }
  }
  
  # Verificar NA após imputação
  verificar_na(dados_projecao_imputado[, variaveis_exogenas], "dados de projeção após imputação")
  
  # Ajustar os modelos - LM ou GLMNET dependendo do parâmetro tipo_modelo
  modelos <- list()
  
  if(tipo_modelo == "lm") {
    # Modelo de regressão linear tradicional
    modelos$PIB <- lm(log_PIB ~ log_IPCA + trend_scaled, data = dados_treinamento)
    modelos$MassaRenda <- lm(log_MassaRenda ~ log_Populacao + log_IPCA + log_PIB + trend_scaled, 
                             data = dados_treinamento)
    modelos$PIM <- lm(log_PIM ~ log_PIB + trend_scaled, data = dados_treinamento)
    modelos$PMC <- lm(log_PMC ~ log_PIB + log_MassaRenda + trend_scaled, data = dados_treinamento)
    modelos$Energia <- lm(log_ConsumoEnergia ~ log_PIB + log_PIM + log_PMC + log_MassaRenda + 
                            log_Temperatura + log_Chuva + log_PrecoEnergia + trend_scaled, 
                          data = dados_treinamento)
    
    message("Modelos de regressão linear ajustados com sucesso.")
    
  } else if(tipo_modelo == "glmnet") {
    # Modelo GLMNET
    require(glmnet)
    
    # Criar matrizes para o glmnet
    X_PIB <- as.matrix(dados_treinamento[, c("log_IPCA", "trend_scaled")])
    y_PIB <- dados_treinamento$log_PIB
    
    X_MassaRenda <- as.matrix(dados_treinamento[, c("log_Populacao", "log_IPCA", 
                                                    "log_PIB", "trend_scaled")])
    y_MassaRenda <- dados_treinamento$log_MassaRenda
    
    X_PIM <- as.matrix(dados_treinamento[, c("log_PIB", "trend_scaled")])
    y_PIM <- dados_treinamento$log_PIM
    
    X_PMC <- as.matrix(dados_treinamento[, c("log_PIB", "log_MassaRenda", "trend_scaled")])
    y_PMC <- dados_treinamento$log_PMC
    
    X_Energia <- as.matrix(dados_treinamento[, c("log_PIB", "log_PIM", "log_PMC", 
                                                 "log_MassaRenda", "log_Temperatura", 
                                                 "log_Chuva", "log_PrecoEnergia", "trend_scaled")])
    y_Energia <- dados_treinamento$log_ConsumoEnergia
    
    # Ajustar modelos GLMNET
    ajuste_PIB <- tryCatch({
      ajustar_glmnet(X_PIB, y_PIB, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para PIB: ", e$message)
      NULL
    })
    
    ajuste_MassaRenda <- tryCatch({
      ajustar_glmnet(X_MassaRenda, y_MassaRenda, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para MassaRenda: ", e$message)
      NULL
    })
    
    ajuste_PIM <- tryCatch({
      ajustar_glmnet(X_PIM, y_PIM, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para PIM: ", e$message)
      NULL
    })
    
    ajuste_PMC <- tryCatch({
      ajustar_glmnet(X_PMC, y_PMC, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para PMC: ", e$message)
      NULL
    })
    
    ajuste_Energia <- tryCatch({
      ajustar_glmnet(X_Energia, y_Energia, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para Energia: ", e$message)
      NULL
    })
    
    # Guardar modelos
    if(!is.null(ajuste_PIB)) {
      modelos$PIB <- ajuste_PIB$model
      modelos$PIB$lambda <- ajuste_PIB$lambda
    } else {
      # Fallback para modelo linear
      message("Usando modelo linear como fallback para PIB")
      modelos$PIB <- lm(log_PIB ~ log_IPCA + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_MassaRenda)) {
      modelos$MassaRenda <- ajuste_MassaRenda$model
      modelos$MassaRenda$lambda <- ajuste_MassaRenda$lambda
    } else {
      message("Usando modelo linear como fallback para MassaRenda")
      modelos$MassaRenda <- lm(log_MassaRenda ~ log_Populacao + log_IPCA + log_PIB + trend_scaled, 
                               data = dados_treinamento)
    }
    
    if(!is.null(ajuste_PIM)) {
      modelos$PIM <- ajuste_PIM$model
      modelos$PIM$lambda <- ajuste_PIM$lambda
    } else {
      message("Usando modelo linear como fallback para PIM")
      modelos$PIM <- lm(log_PIM ~ log_PIB + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_PMC)) {
      modelos$PMC <- ajuste_PMC$model
      modelos$PMC$lambda <- ajuste_PMC$lambda
    } else {
      message("Usando modelo linear como fallback para PMC")
      modelos$PMC <- lm(log_PMC ~ log_PIB + log_MassaRenda + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_Energia)) {
      modelos$Energia <- ajuste_Energia$model
      modelos$Energia$lambda <- ajuste_Energia$lambda
    } else {
      message("Usando modelo linear como fallback para Energia")
      modelos$Energia <- lm(log_ConsumoEnergia ~ log_PIB + log_PIM + log_PMC + log_MassaRenda + 
                              log_Temperatura + log_Chuva + log_PrecoEnergia + trend_scaled, 
                            data = dados_treinamento)
    }
    
    message("Modelos GLMNET ajustados com sucesso.")
    
  } else {
    stop(paste("Tipo de modelo não reconhecido:", tipo_modelo, 
               "Tipos válidos são 'lm' ou 'glmnet'"))
  }
  
  # Generate base projections usando dados imputados
  projecoes_base <- projetar_consumo_energia(
    modelos = modelos,
    dados_projecao = dados_projecao_imputado,
    dados_historicos = dados_treinamento
  )
  
  # Verificar projeções para diagnóstico
  message("Resumo das projeções base:")
  print(summary(projecoes_base$ConsumoEnergia_Projetado))
  
  # Create optimistic scenario
  cenario_otimista <- dados_projecao_imputado
  cenario_otimista$IPCA <- cenario_otimista$IPCA * 0.98  # Lower inflation
  cenario_otimista$PrecoEnergia <- cenario_otimista$PrecoEnergia * 0.95  # Lower price
  cenario_otimista$log_IPCA <- log(cenario_otimista$IPCA)
  cenario_otimista$log_PrecoEnergia <- log(cenario_otimista$PrecoEnergia)
  
  # Create pessimistic scenario
  cenario_pessimista <- dados_projecao_imputado
  cenario_pessimista$IPCA <- cenario_pessimista$IPCA * 1.05  # Higher inflation
  cenario_pessimista$PrecoEnergia <- cenario_pessimista$PrecoEnergia * 1.10  # Higher price
  cenario_pessimista$log_IPCA <- log(cenario_pessimista$IPCA)
  cenario_pessimista$log_PrecoEnergia <- log(cenario_pessimista$PrecoEnergia)
  
  # Project scenarios
  proj_otimista <- projetar_consumo_energia(
    modelos = modelos,
    dados_projecao = cenario_otimista,
    dados_historicos = dados_treinamento
  )
  
  proj_pessimista <- projetar_consumo_energia(
    modelos = modelos,
    dados_projecao = cenario_pessimista,
    dados_historicos = dados_treinamento
  )
  
  # Prepare historical data for plotting
  dados_historicos_plot <- dados_treinamento %>%
    select(Date, ConsumoEnergia) %>%
    mutate(Cenario = "Histórico")
  
  # Prepare projected data for plotting
  dados_base <- projecoes_base %>%
    select(Date, ConsumoEnergia_Projetado) %>%
    rename(ConsumoEnergia = ConsumoEnergia_Projetado) %>%
    mutate(Cenario = "Base")
  
  dados_otimista <- proj_otimista %>%
    select(Date, ConsumoEnergia_Projetado) %>%
    rename(ConsumoEnergia = ConsumoEnergia_Projetado) %>%
    mutate(Cenario = "Otimista")
  
  dados_pessimista <- proj_pessimista %>%
    select(Date, ConsumoEnergia_Projetado) %>%
    rename(ConsumoEnergia = ConsumoEnergia_Projetado) %>%
    mutate(Cenario = "Pessimista")
  
  # Combine scenarios
  dados_cenarios <- bind_rows(dados_base, dados_otimista, dados_pessimista)
  todos_dados <- bind_rows(dados_historicos_plot, dados_cenarios)
  
  # Create summary statistics of projections
  resumo_cenarios <- dados_cenarios %>%
    group_by(Cenario) %>%
    summarise(
      Consumo_Medio = mean(ConsumoEnergia, na.rm = TRUE),
      Consumo_Total = sum(ConsumoEnergia, na.rm = TRUE),
      Consumo_Min = min(ConsumoEnergia, na.rm = TRUE),
      Consumo_Max = max(ConsumoEnergia, na.rm = TRUE)
    )
  
  # Calculate percentage variation relative to base scenario
  base_medio <- resumo_cenarios$Consumo_Medio[resumo_cenarios$Cenario == "Base"]
  resumo_cenarios <- resumo_cenarios %>%
    mutate(Variacao_Percentual = (Consumo_Medio / base_medio - 1) * 100)
  
  # Create Plotly visualization - removing NA values for plotting
  todos_dados_plot <- todos_dados %>%
    filter(!is.na(ConsumoEnergia))
  
  plot_cenarios <- plot_ly(type = 'scatter', mode = 'lines') %>%
    # Historical data
    add_trace(
      data = filter(todos_dados_plot, Cenario == "Histórico"),
      x = ~Date,
      y = ~ConsumoEnergia,
      name = 'Histórico',
      line = list(color = 'blue', width = 2)
    ) %>%
    # Base scenario
    add_trace(
      data = filter(todos_dados_plot, Cenario == "Base"),
      x = ~Date,
      y = ~ConsumoEnergia,
      name = 'Base',
      line = list(color = 'purple', width = 2)
    ) %>%
    # Optimistic scenario
    add_trace(
      data = filter(todos_dados_plot, Cenario == "Otimista"),
      x = ~Date,
      y = ~ConsumoEnergia,
      name = 'Otimista',
      line = list(color = 'green', width = 2)
    ) %>%
    # Pessimistic scenario
    add_trace(
      data = filter(todos_dados_plot, Cenario == "Pessimista"),
      x = ~Date,
      y = ~ConsumoEnergia,
      name = 'Pessimista',
      line = list(color = 'red', width = 2)
    ) %>%
    layout(
      title = "Projeções de Consumo de Energia por Cenário",
      xaxis = list(title = "Data"),
      yaxis = list(title = "Consumo de Energia (MWh)"),
      legend = list(x = 0.1, y = 0.9),
      hovermode = "closest"
    )
  
  # Calcular métricas de erro se tivermos valores reais de ConsumoEnergia
  # para o período de projeção
  metricas_erro <- NULL
  
  # Verificar se existem dados para validação (não NA nos dados de projeção)
  dados_validacao <- dados_projecao %>%
    filter(!is.na(ConsumoEnergia))
  
  if (nrow(dados_validacao) > 0) {
    message("Dados para validação encontrados. Calculando métricas de erro.")
    
    # Dados para cálculo das métricas
    dados_erro <- dados_validacao %>%
      select(Date, ConsumoEnergia) %>%
      inner_join(
        projecoes_base %>% select(Date, ConsumoEnergia_Projetado),
        by = "Date"
      ) %>%
      inner_join(
        proj_otimista %>% select(Date, ConsumoEnergia_Projetado_Otimista = ConsumoEnergia_Projetado),
        by = "Date"
      ) %>%
      inner_join(
        proj_pessimista %>% select(Date, ConsumoEnergia_Projetado_Pessimista = ConsumoEnergia_Projetado),
        by = "Date"
      )
    
    if (nrow(dados_erro) > 0) {
      # Calcular métricas de erro para cada cenário
      calcular_metricas <- function(real, previsto, nome_cenario) {
        # Erro absoluto
        erro_absoluto <- abs(real - previsto)
        
        # Mean Absolute Error (MAE)
        mae <- mean(erro_absoluto, na.rm = TRUE)
        
        # Mean Absolute Percentage Error (MAPE)
        mape <- mean(erro_absoluto / real, na.rm = TRUE) * 100
        
        # Root Mean Square Error (RMSE)
        rmse <- sqrt(mean((real - previsto)^2, na.rm = TRUE))
        
        # Coeficiente de determinação (R²)
        ss_total <- sum((real - mean(real, na.rm = TRUE))^2, na.rm = TRUE)
        ss_residual <- sum((real - previsto)^2, na.rm = TRUE)
        r_squared <- 1 - (ss_residual / ss_total)
        
        # Erro médio (ME) - para verificar viés
        me <- mean(real - previsto, na.rm = TRUE)
        
        # Criar dataframe com resultados
        data.frame(
          Cenario = nome_cenario,
          MAE = mae,
          MAPE = mape,
          RMSE = rmse,
          R_Squared = r_squared,
          ME = me
        )
      }
      
      # Métricas para cenário base
      metricas_base <- calcular_metricas(
        dados_erro$ConsumoEnergia, 
        dados_erro$ConsumoEnergia_Projetado,
        "Base"
      )
      
      # Métricas para cenário otimista
      metricas_otimista <- calcular_metricas(
        dados_erro$ConsumoEnergia, 
        dados_erro$ConsumoEnergia_Projetado_Otimista,
        "Otimista"
      )
      
      # Métricas para cenário pessimista
      metricas_pessimista <- calcular_metricas(
        dados_erro$ConsumoEnergia, 
        dados_erro$ConsumoEnergia_Projetado_Pessimista,
        "Pessimista"
      )
      
      # Combinando todas as métricas
      metricas_erro <- bind_rows(metricas_base, metricas_otimista, metricas_pessimista)
      
      # Criar visualização de comparação entre valores reais e projetados
      plot_comparacao <- plot_ly(type = 'scatter', mode = 'lines') %>%
        add_trace(
          data = dados_erro,
          x = ~Date,
          y = ~ConsumoEnergia,
          name = 'Valores Reais',
          line = list(color = 'black', width = 2)
        ) %>%
        add_trace(
          data = dados_erro,
          x = ~Date,
          y = ~ConsumoEnergia_Projetado,
          name = 'Projeção Base',
          line = list(color = 'purple', width = 2)
        ) %>%
        add_trace(
          data = dados_erro,
          x = ~Date,
          y = ~ConsumoEnergia_Projetado_Otimista,
          name = 'Projeção Otimista',
          line = list(color = 'green', width = 2)
        ) %>%
        add_trace(
          data = dados_erro,
          x = ~Date,
          y = ~ConsumoEnergia_Projetado_Pessimista,
          name = 'Projeção Pessimista',
          line = list(color = 'red', width = 2)
        ) %>%
        layout(
          title = "Comparação: Valores Reais vs. Projetados",
          xaxis = list(title = "Data"),
          yaxis = list(title = "Consumo de Energia (MWh)"),
          legend = list(x = 0.1, y = 0.9),
          hovermode = "closest"
        )
    } else {
      message("Não foi possível calcular métricas de erro, pois não há coincidência entre datas nos dados de validação e projeção.")
    }
  } else {
    message("Não há dados para validação (todos os valores de ConsumoEnergia após a data de corte são NA).")
  }
  
  # Return results as a list
  return(list(
    modelos = modelos,
    tipo_modelo = tipo_modelo,
    data_corte = data_corte,
    projecoes_base = projecoes_base,
    projecoes_otimista = proj_otimista,
    projecoes_pessimista = proj_pessimista,
    dados_plot = todos_dados,
    resumo = resumo_cenarios,
    grafico = plot_cenarios,
    metricas_erro = metricas_erro,
    grafico_comparacao = if(exists("plot_comparacao")) plot_comparacao else NULL,
    # Adicionalmente, devolver os dados originais e os mapeamentos de colunas
    dados_originais = dados_completos,
    dados_padronizados = dados_padronizados,
    mapeamento_colunas = mapeamento_colunas
  ))
}

dados_completos <- read_excel("GitHub/app_forecasts_R/www/data/dados.xlsx")


# Usando LASSO
resultados_lasso <- treinar_e_projetar_modelo(
  dados_completos = dados_completos,
  coluna_data = "data",
  tipo_modelo = "glmnet",
  alpha = 1  # LASSO
)

# Acessar os resultados
resultados_lasso$grafico  # Mostrar o gráfico


# Usando Ridge
resultados_ridge <- treinar_e_projetar_modelo(
  dados_completos = dados_completos,
  coluna_data = "data",
  tipo_modelo = "glmnet",
  alpha = 0  # Ridge
)

# Acessar os resultados
resultados_ridge$grafico  # Mostrar o gráfico


# Usando Elastic Net
resultados_elasticnet <- treinar_e_projetar_modelo(
  dados_completos = dados_completos,
  coluna_data = "data",
  tipo_modelo = "glmnet",
  alpha = 0.5  # Elastic Net
)

# Acessar os resultados
resultados_elasticnet$grafico  # Mostrar o gráfico







