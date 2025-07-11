treinar_e_projetar_modelo_sem <- function(dados_completos, coluna_data = "data", 
                                          tipo_modelo = "lm", # Parâmetro para escolher o tipo de modelo
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
    # Ajustado para incluir as novas variáveis necessárias
    colunas_necessarias <- c("log_indice_de_precos", "log_dias_uteis", "log_pmc", 
                             "log_Temperatura_media", "log_Pluviometria_total", 
                             "log_Amplitude_termica", "log_Dias_chuvas", 
                             "D_EQT_PA", "log_Tarifa_res", "log_Tarifa_ind", 
                             "log_Tarifa_com", "log_Tarifa_rur", "log_pop_domicilio", "trend_scaled")
    
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
      if("glmnet" %in% class(modelos$PIB_4i)) {
        # GLMNET workflow
        message("Usando modelos GLMNET para projeção")
        
        # PIB_4i (variável exógena ao modelo, mas endógena no sistema)
        X_PIB_4i <- as.matrix(dados_projecao_completos[, c("log_indice_de_precos", "trend_scaled")])
        resultados$log_PIB_4i[linhas_completas] <- as.numeric(projetar_glmnet(modelos$PIB_4i, X_PIB_4i, modelos$PIB_4i$lambda))
        
        # Variáveis dependentes do PIB_4i
        X_pim_geral <- as.matrix(dados_projecao_completos[, c("log_PIB_4i", "trend_scaled")])
        resultados$log_pim_geral[linhas_completas] <- as.numeric(projetar_glmnet(modelos$pim_geral, X_pim_geral, modelos$pim_geral$lambda))
        
        X_pim_energia <- as.matrix(dados_projecao_completos[, c("log_PIB_4i", "trend_scaled")])
        resultados$log_pim_energia[linhas_completas] <- as.numeric(projetar_glmnet(modelos$pim_energia, X_pim_energia, modelos$pim_energia$lambda))
        
        X_pim_extrativa <- as.matrix(dados_projecao_completos[, c("log_PIB_4i", "trend_scaled")])
        resultados$log_pim_extrativa[linhas_completas] <- as.numeric(projetar_glmnet(modelos$pim_extrativa, X_pim_extrativa, modelos$pim_extrativa$lambda))
        
        X_pib_servicos <- as.matrix(dados_projecao_completos[, c("log_PIB_4i", "trend_scaled")])
        resultados$log_pib_servicos[linhas_completas] <- as.numeric(projetar_glmnet(modelos$pib_servicos, X_pib_servicos, modelos$pib_servicos$lambda))
        
        X_pib_agropecuario <- as.matrix(dados_projecao_completos[, c("log_PIB_4i", "trend_scaled")])
        resultados$log_pib_agropecuario[linhas_completas] <- as.numeric(projetar_glmnet(modelos$pib_agropecuario, X_pib_agropecuario, modelos$pib_agropecuario$lambda))
        
        # producao_agropecuaria depende de pib_agropecuario
        X_producao_agropecuaria <- as.matrix(dados_projecao_completos[, c("log_pib_agropecuario", "trend_scaled")])
        resultados$log_producao_agropecuaria[linhas_completas] <- as.numeric(projetar_glmnet(modelos$producao_agropecuaria, X_producao_agropecuaria, modelos$producao_agropecuaria$lambda))
        
        # massa_total depende de pop_domicilio
        X_massa_total <- as.matrix(dados_projecao_completos[, c("log_pop_domicilio", "log_PIB_4i", "trend_scaled")])
        resultados$log_massa_total[linhas_completas] <- as.numeric(projetar_glmnet(modelos$massa_total, X_massa_total, modelos$massa_total$lambda))
        
        # renda_domiciliar depende de massa_total
        X_renda_domiciliar <- as.matrix(dados_projecao_completos[, c("log_massa_total", "trend_scaled")])
        resultados$log_renda_domiciliar[linhas_completas] <- as.numeric(projetar_glmnet(modelos$renda_domiciliar, X_renda_domiciliar, modelos$renda_domiciliar$lambda))
        
        # Modelo final para energia injetada (ei_total)
        X_Energia <- as.matrix(dados_projecao_completos[, c(
          "log_Temperatura_media", "log_Pluviometria_total", "log_Amplitude_termica", 
          "log_Dias_chuvas", "log_indice_de_precos", "log_pmc", "log_dias_uteis", 
          "log_pim_geral", "log_pim_energia", "log_pim_extrativa", 
          "log_producao_agropecuaria", "log_renda_domiciliar", "log_massa_total", 
          "log_pib_servicos", "log_Tarifa_res", "log_Tarifa_ind", "log_Tarifa_com", 
          "log_Tarifa_rur", "D_EQT_PA", "trend_scaled"
        )])
        
        resultados$log_ConsumoEnergia[linhas_completas] <- as.numeric(projetar_glmnet(modelos$Energia, X_Energia, modelos$Energia$lambda))
        
      } else {
        # LM workflow
        message("Usando modelos de regressão linear para projeção")
        
        coefs <- list(
          PIB_4i = coef(modelos$PIB_4i),
          pim_geral = coef(modelos$pim_geral),
          pim_energia = coef(modelos$pim_energia),
          pim_extrativa = coef(modelos$pim_extrativa),
          pib_servicos = coef(modelos$pib_servicos),
          pib_agropecuario = coef(modelos$pib_agropecuario),
          producao_agropecuaria = coef(modelos$producao_agropecuaria),
          massa_total = coef(modelos$massa_total),
          renda_domiciliar = coef(modelos$renda_domiciliar),
          Energia = coef(modelos$Energia)
        )
        
        # Projetar PIB_4i (exógena no modelo, mas endógena no sistema)
        resultados$log_PIB_4i[linhas_completas] <- coefs$PIB_4i["(Intercept)"] + 
          coefs$PIB_4i["log_indice_de_precos"] * dados_projecao_completos$log_indice_de_precos + 
          coefs$PIB_4i["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Projetar variáveis dependentes do PIB_4i
        resultados$log_pim_geral[linhas_completas] <- coefs$pim_geral["(Intercept)"] + 
          coefs$pim_geral["log_PIB_4i"] * dados_projecao_completos$log_PIB_4i + 
          coefs$pim_geral["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        resultados$log_pim_energia[linhas_completas] <- coefs$pim_energia["(Intercept)"] + 
          coefs$pim_energia["log_PIB_4i"] * dados_projecao_completos$log_PIB_4i + 
          coefs$pim_energia["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        resultados$log_pim_extrativa[linhas_completas] <- coefs$pim_extrativa["(Intercept)"] + 
          coefs$pim_extrativa["log_PIB_4i"] * dados_projecao_completos$log_PIB_4i + 
          coefs$pim_extrativa["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        resultados$log_pib_servicos[linhas_completas] <- coefs$pib_servicos["(Intercept)"] + 
          coefs$pib_servicos["log_PIB_4i"] * dados_projecao_completos$log_PIB_4i + 
          coefs$pib_servicos["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        resultados$log_pib_agropecuario[linhas_completas] <- coefs$pib_agropecuario["(Intercept)"] + 
          coefs$pib_agropecuario["log_PIB_4i"] * dados_projecao_completos$log_PIB_4i + 
          coefs$pib_agropecuario["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Projetar producao_agropecuaria baseado em pib_agropecuario
        resultados$log_producao_agropecuaria[linhas_completas] <- coefs$producao_agropecuaria["(Intercept)"] + 
          coefs$producao_agropecuaria["log_pib_agropecuario"] * dados_projecao_completos$log_pib_agropecuario + 
          coefs$producao_agropecuaria["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Projetar massa_total baseado em pop_domicilio
        resultados$log_massa_total[linhas_completas] <- coefs$massa_total["(Intercept)"] + 
          coefs$massa_total["log_pop_domicilio"] * dados_projecao_completos$log_pop_domicilio + 
          coefs$massa_total["log_PIB_4i"] * dados_projecao_completos$log_PIB_4i + 
          coefs$massa_total["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Projetar renda_domiciliar baseado em massa_total
        resultados$log_renda_domiciliar[linhas_completas] <- coefs$renda_domiciliar["(Intercept)"] + 
          coefs$renda_domiciliar["log_massa_total"] * dados_projecao_completos$log_massa_total + 
          coefs$renda_domiciliar["trend_scaled"] * dados_projecao_completos$trend_scaled
        
        # Finalmente, projetar ConsumoEnergia
        resultados$log_ConsumoEnergia[linhas_completas] <- coefs$Energia["(Intercept)"] + 
          coefs$Energia["log_Temperatura_media"] * dados_projecao_completos$log_Temperatura_media +
          coefs$Energia["log_Pluviometria_total"] * dados_projecao_completos$log_Pluviometria_total +
          coefs$Energia["log_Amplitude_termica"] * dados_projecao_completos$log_Amplitude_termica +
          coefs$Energia["log_Dias_chuvas"] * dados_projecao_completos$log_Dias_chuvas +
          coefs$Energia["log_indice_de_precos"] * dados_projecao_completos$log_indice_de_precos +
          coefs$Energia["log_pmc"] * dados_projecao_completos$log_pmc +
          coefs$Energia["log_dias_uteis"] * dados_projecao_completos$log_dias_uteis +
          coefs$Energia["log_pim_geral"] * dados_projecao_completos$log_pim_geral +
          coefs$Energia["log_pim_energia"] * dados_projecao_completos$log_pim_energia +
          coefs$Energia["log_pim_extrativa"] * dados_projecao_completos$log_pim_extrativa +
          coefs$Energia["log_producao_agropecuaria"] * dados_projecao_completos$log_producao_agropecuaria +
          coefs$Energia["log_renda_domiciliar"] * dados_projecao_completos$log_renda_domiciliar +
          coefs$Energia["log_massa_total"] * dados_projecao_completos$log_massa_total +
          coefs$Energia["log_pib_servicos"] * dados_projecao_completos$log_pib_servicos +
          coefs$Energia["log_Tarifa_res"] * dados_projecao_completos$log_Tarifa_res +
          coefs$Energia["log_Tarifa_ind"] * dados_projecao_completos$log_Tarifa_ind +
          coefs$Energia["log_Tarifa_com"] * dados_projecao_completos$log_Tarifa_com +
          coefs$Energia["log_Tarifa_rur"] * dados_projecao_completos$log_Tarifa_rur +
          coefs$Energia["D_EQT_PA"] * dados_projecao_completos$D_EQT_PA +
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
  # Atualizado para incluir todas as variáveis do novo modelo SEM
  mapeamento_colunas <- list(
    "Date" = coluna_data,
    "PIB_4i" = "PIB_4i",
    "pim_geral" = "pim_geral",
    "pim_energia" = "pim_energia",
    "pim_extrativa" = "pim_extrativa",
    "pib_servicos" = "pib_servicos",
    "pib_agropecuario" = "pib_agropecuario",
    "producao_agropecuaria" = "producao_agropecuaria",
    "pop_domicilio" = "pop_domicilio",
    "massa_total" = "massa_total",
    "renda_domiciliar" = "renda_domiciliar",
    "ConsumoEnergia" = "ei_total",
    "indice_de_precos" = "indice_de_precos",
    "dias_uteis" = "dias_uteis",
    "pmc" = "pmc",
    "Temperatura_media" = "Temperatura_media",
    "Pluviometria_total" = "Pluviometria_total",
    "Amplitude_termica" = "Amplitude_termica",
    "Dias_chuvas" = "Dias_chuvas",
    "D_EQT_PA" = "D_EQT_PA",
    "Tarifa_res" = "Tarifa_res",
    "Tarifa_ind" = "Tarifa_ind",
    "Tarifa_com" = "Tarifa_com", 
    "Tarifa_rur" = "Tarifa_rur"
  )
  
  colunas_necessarias <- unlist(mapeamento_colunas)
  colunas_faltantes <- colunas_necessarias[!colunas_necessarias %in% names(dados)]
  
  if(length(colunas_faltantes) > 0) {
    stop(paste("As seguintes colunas estão faltando no conjunto de dados:", 
               paste(colunas_faltantes, collapse = ", ")))
  }
  
  # Criar um dataframe com as colunas renomeadas para o padrão esperado pelo modelo
  dados_padronizados <- dados %>%
    rename(!!!setNames(mapeamento_colunas, names(mapeamento_colunas))) %>%
    # Selecionar apenas as colunas necessárias para o modelo
    select(all_of(names(mapeamento_colunas)))
  
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
      # Transformação log de todas as variáveis, exceto D_EQT_PA que é uma dummy
      across(
        .cols = c(-Date, -trend, -trend_scaled, -D_EQT_PA),
        .fns = ~ifelse(is.na(.) | . <= 0, NA, log(.)),
        .names = "log_{.col}"
      )
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
  variaveis_exogenas <- c(
    "log_indice_de_precos", "log_dias_uteis", "log_pmc", 
    "log_Temperatura_media", "log_Pluviometria_total", 
    "log_Amplitude_termica", "log_Dias_chuvas", 
    "D_EQT_PA", "log_Tarifa_res", "log_Tarifa_ind", 
    "log_Tarifa_com", "log_Tarifa_rur", "log_pop_domicilio", 
    "trend_scaled"
  )
  
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
    # Ajustado para implementar o novo modelo SEM
    
    # PIB_4i (exógeno no modelo, mas endógeno no sistema)
    modelos$PIB_4i <- lm(log_PIB_4i ~ log_indice_de_precos + trend_scaled, data = dados_treinamento)
    
    # Variáveis dependentes do PIB_4i
    modelos$pim_geral <- lm(log_pim_geral ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    modelos$pim_energia <- lm(log_pim_energia ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    modelos$pim_extrativa <- lm(log_pim_extrativa ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    modelos$pib_servicos <- lm(log_pib_servicos ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    modelos$pib_agropecuario <- lm(log_pib_agropecuario ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    
    # producao_agropecuaria depende de pib_agropecuario
    modelos$producao_agropecuaria <- lm(log_producao_agropecuaria ~ log_pib_agropecuario + trend_scaled, 
                                        data = dados_treinamento)
    
    # massa_total depende de pop_domicilio e PIB_4i
    modelos$massa_total <- lm(log_massa_total ~ log_pop_domicilio + log_PIB_4i + trend_scaled, 
                              data = dados_treinamento)
    
    # renda_domiciliar depende de massa_total
    modelos$renda_domiciliar <- lm(log_renda_domiciliar ~ log_massa_total + trend_scaled, 
                                   data = dados_treinamento)
    
    # Modelo final para energia injetada (ei_total)
    modelos$Energia <- lm(log_ConsumoEnergia ~ 
                            log_Temperatura_media + log_Pluviometria_total + 
                            log_Amplitude_termica + log_Dias_chuvas + 
                            log_indice_de_precos + log_pmc + log_dias_uteis + 
                            log_pim_geral + log_pim_energia + log_pim_extrativa + 
                            log_producao_agropecuaria + log_renda_domiciliar + 
                            log_massa_total + log_pib_servicos + 
                            log_Tarifa_res + log_Tarifa_ind + log_Tarifa_com + 
                            log_Tarifa_rur + D_EQT_PA + trend_scaled, 
                          data = dados_treinamento)
    
    message("Modelos de regressão linear ajustados com sucesso.")
    
  } else if(tipo_modelo == "glmnet") {
    # Modelo GLMNET
    require(glmnet)
    
    # Criar matrizes para o glmnet
    # PIB_4i
    X_PIB_4i <- as.matrix(dados_treinamento[, c("log_indice_de_precos", "trend_scaled")])
    y_PIB_4i <- dados_treinamento$log_PIB_4i
    
    # Variáveis dependentes do PIB_4i
    X_pim_geral <- as.matrix(dados_treinamento[, c("log_PIB_4i", "trend_scaled")])
    y_pim_geral <- dados_treinamento$log_pim_geral
    
    X_pim_energia <- as.matrix(dados_treinamento[, c("log_PIB_4i", "trend_scaled")])
    y_pim_energia <- dados_treinamento$log_pim_energia
    
    X_pim_extrativa <- as.matrix(dados_treinamento[, c("log_PIB_4i", "trend_scaled")])
    y_pim_extrativa <- dados_treinamento$log_pim_extrativa
    
    X_pib_servicos <- as.matrix(dados_treinamento[, c("log_PIB_4i", "trend_scaled")])
    y_pib_servicos <- dados_treinamento$log_pib_servicos
    
    X_pib_agropecuario <- as.matrix(dados_treinamento[, c("log_PIB_4i", "trend_scaled")])
    y_pib_agropecuario <- dados_treinamento$log_pib_agropecuario
    
    # producao_agropecuaria
    X_producao_agropecuaria <- as.matrix(dados_treinamento[, c("log_pib_agropecuario", "trend_scaled")])
    y_producao_agropecuaria <- dados_treinamento$log_producao_agropecuaria
    
    # massa_total
    X_massa_total <- as.matrix(dados_treinamento[, c("log_pop_domicilio", "log_PIB_4i", "trend_scaled")])
    y_massa_total <- dados_treinamento$log_massa_total
    
    # renda_domiciliar
    X_renda_domiciliar <- as.matrix(dados_treinamento[, c("log_massa_total", "trend_scaled")])
    y_renda_domiciliar <- dados_treinamento$log_renda_domiciliar
    
    # Energia (modelo final)
    X_Energia <- as.matrix(dados_treinamento[, c(
      "log_Temperatura_media", "log_Pluviometria_total", "log_Amplitude_termica", 
      "log_Dias_chuvas", "log_indice_de_precos", "log_pmc", "log_dias_uteis", 
      "log_pim_geral", "log_pim_energia", "log_pim_extrativa", 
      "log_producao_agropecuaria", "log_renda_domiciliar", "log_massa_total", 
      "log_pib_servicos", "log_Tarifa_res", "log_Tarifa_ind", "log_Tarifa_com", 
      "log_Tarifa_rur", "D_EQT_PA", "trend_scaled"
    )])
    y_Energia <- dados_treinamento$log_ConsumoEnergia
    
    # Ajustar modelos GLMNET
    ajuste_PIB_4i <- tryCatch({
      ajustar_glmnet(X_PIB_4i, y_PIB_4i, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para PIB_4i: ", e$message)
      NULL
    })
    
    ajuste_pim_geral <- tryCatch({
      ajustar_glmnet(X_pim_geral, y_pim_geral, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para pim_geral: ", e$message)
      NULL
    })
    
    ajuste_pim_energia <- tryCatch({
      ajustar_glmnet(X_pim_energia, y_pim_energia, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para pim_energia: ", e$message)
      NULL
    })
    
    ajuste_pim_extrativa <- tryCatch({
      ajustar_glmnet(X_pim_extrativa, y_pim_extrativa, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para pim_extrativa: ", e$message)
      NULL
    })
    
    ajuste_pib_servicos <- tryCatch({
      ajustar_glmnet(X_pib_servicos, y_pib_servicos, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para pib_servicos: ", e$message)
      NULL
    })
    
    ajuste_pib_agropecuario <- tryCatch({
      ajustar_glmnet(X_pib_agropecuario, y_pib_agropecuario, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para pib_agropecuario: ", e$message)
      NULL
    })
    
    ajuste_producao_agropecuaria <- tryCatch({
      ajustar_glmnet(X_producao_agropecuaria, y_producao_agropecuaria, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para producao_agropecuaria: ", e$message)
      NULL
    })
    
    ajuste_massa_total <- tryCatch({
      ajustar_glmnet(X_massa_total, y_massa_total, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para massa_total: ", e$message)
      NULL
    })
    
    ajuste_renda_domiciliar <- tryCatch({
      ajustar_glmnet(X_renda_domiciliar, y_renda_domiciliar, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para renda_domiciliar: ", e$message)
      NULL
    })
    
    ajuste_Energia <- tryCatch({
      ajustar_glmnet(X_Energia, y_Energia, alpha = alpha, lambda = lambda)
    }, error = function(e) {
      message("Erro ao ajustar modelo GLMNET para Energia: ", e$message)
      NULL
    })
    
    # Guardar modelos com fallback para modelos lineares quando necessário
    if(!is.null(ajuste_PIB_4i)) {
      modelos$PIB_4i <- ajuste_PIB_4i$model
      modelos$PIB_4i$lambda <- ajuste_PIB_4i$lambda
    } else {
      message("Usando modelo linear como fallback para PIB_4i")
      modelos$PIB_4i <- lm(log_PIB_4i ~ log_indice_de_precos + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_pim_geral)) {
      modelos$pim_geral <- ajuste_pim_geral$model
      modelos$pim_geral$lambda <- ajuste_pim_geral$lambda
    } else {
      message("Usando modelo linear como fallback para pim_geral")
      modelos$pim_geral <- lm(log_pim_geral ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_pim_energia)) {
      modelos$pim_energia <- ajuste_pim_energia$model
      modelos$pim_energia$lambda <- ajuste_pim_energia$lambda
    } else {
      message("Usando modelo linear como fallback para pim_energia")
      modelos$pim_energia <- lm(log_pim_energia ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_pim_extrativa)) {
      modelos$pim_extrativa <- ajuste_pim_extrativa$model
      modelos$pim_extrativa$lambda <- ajuste_pim_extrativa$lambda
    } else {
      message("Usando modelo linear como fallback para pim_extrativa")
      modelos$pim_extrativa <- lm(log_pim_extrativa ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_pib_servicos)) {
      modelos$pib_servicos <- ajuste_pib_servicos$model
      modelos$pib_servicos$lambda <- ajuste_pib_servicos$lambda
    } else {
      message("Usando modelo linear como fallback para pib_servicos")
      modelos$pib_servicos <- lm(log_pib_servicos ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_pib_agropecuario)) {
      modelos$pib_agropecuario <- ajuste_pib_agropecuario$model
      modelos$pib_agropecuario$lambda <- ajuste_pib_agropecuario$lambda
    } else {
      message("Usando modelo linear como fallback para pib_agropecuario")
      modelos$pib_agropecuario <- lm(log_pib_agropecuario ~ log_PIB_4i + trend_scaled, data = dados_treinamento)
    }
    
    if(!is.null(ajuste_producao_agropecuaria)) {
      modelos$producao_agropecuaria <- ajuste_producao_agropecuaria$model
      modelos$producao_agropecuaria$lambda <- ajuste_producao_agropecuaria$lambda
    } else {
      message("Usando modelo linear como fallback para producao_agropecuaria")
      modelos$producao_agropecuaria <- lm(log_producao_agropecuaria ~ log_pib_agropecuario + trend_scaled, 
                                          data = dados_treinamento)
    }
    
    if(!is.null(ajuste_massa_total)) {
      modelos$massa_total <- ajuste_massa_total$model
      modelos$massa_total$lambda <- ajuste_massa_total$lambda
    } else {
      message("Usando modelo linear como fallback para massa_total")
      modelos$massa_total <- lm(log_massa_total ~ log_pop_domicilio + log_PIB_4i + trend_scaled, 
                                data = dados_treinamento)
    }
    
    if(!is.null(ajuste_renda_domiciliar)) {
      modelos$renda_domiciliar <- ajuste_renda_domiciliar$model
      modelos$renda_domiciliar$lambda <- ajuste_renda_domiciliar$lambda
    } else {
      message("Usando modelo linear como fallback para renda_domiciliar")
      modelos$renda_domiciliar <- lm(log_renda_domiciliar ~ log_massa_total + trend_scaled, 
                                     data = dados_treinamento)
    }
    
    if(!is.null(ajuste_Energia)) {
      modelos$Energia <- ajuste_Energia$model
      modelos$Energia$lambda <- ajuste_Energia$lambda
    } else {
      message("Usando modelo linear como fallback para Energia")
      modelos$Energia <- lm(log_ConsumoEnergia ~ 
                              log_Temperatura_media + log_Pluviometria_total + 
                              log_Amplitude_termica + log_Dias_chuvas + 
                              log_indice_de_precos + log_pmc + log_dias_uteis + 
                              log_pim_geral + log_pim_energia + log_pim_extrativa + 
                              log_producao_agropecuaria + log_renda_domiciliar + 
                              log_massa_total + log_pib_servicos + 
                              log_Tarifa_res + log_Tarifa_ind + log_Tarifa_com + 
                              log_Tarifa_rur + D_EQT_PA + trend_scaled, 
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
  cenario_otimista$indice_de_precos <- cenario_otimista$indice_de_precos * 0.98  # Lower inflation
  cenario_otimista$Tarifa_res <- cenario_otimista$Tarifa_res * 0.95  # Lower residential tariff
  cenario_otimista$Tarifa_ind <- cenario_otimista$Tarifa_ind * 0.95  # Lower industrial tariff
  cenario_otimista$Tarifa_com <- cenario_otimista$Tarifa_com * 0.95  # Lower commercial tariff
  cenario_otimista$Tarifa_rur <- cenario_otimista$Tarifa_rur * 0.95  # Lower rural tariff
  
  # Atualizar os logs das variáveis modificadas
  cenario_otimista$log_indice_de_precos <- log(cenario_otimista$indice_de_precos)
  cenario_otimista$log_Tarifa_res <- log(cenario_otimista$Tarifa_res)
  cenario_otimista$log_Tarifa_ind <- log(cenario_otimista$Tarifa_ind)
  cenario_otimista$log_Tarifa_com <- log(cenario_otimista$Tarifa_com)
  cenario_otimista$log_Tarifa_rur <- log(cenario_otimista$Tarifa_rur)
  
  # Create pessimistic scenario
  cenario_pessimista <- dados_projecao_imputado
  cenario_pessimista$indice_de_precos <- cenario_pessimista$indice_de_precos * 1.05  # Higher inflation
  cenario_pessimista$Tarifa_res <- cenario_pessimista$Tarifa_res * 1.10  # Higher residential tariff
  cenario_pessimista$Tarifa_ind <- cenario_pessimista$Tarifa_ind * 1.10  # Higher industrial tariff
  cenario_pessimista$Tarifa_com <- cenario_pessimista$Tarifa_com * 1.10  # Higher commercial tariff
  cenario_pessimista$Tarifa_rur <- cenario_pessimista$Tarifa_rur * 1.10  # Higher rural tariff
  
  # Atualizar os logs das variáveis modificadas
  cenario_pessimista$log_indice_de_precos <- log(cenario_pessimista$indice_de_precos)
  cenario_pessimista$log_Tarifa_res <- log(cenario_pessimista$Tarifa_res)
  cenario_pessimista$log_Tarifa_ind <- log(cenario_pessimista$Tarifa_ind)
  cenario_pessimista$log_Tarifa_com <- log(cenario_pessimista$Tarifa_com)
  cenario_pessimista$log_Tarifa_rur <- log(cenario_pessimista$Tarifa_rur)
  
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

#Exemplo de uso:
dados_completos <- read_excel("D:/OneDrive/Área de Trabalho/SEM/ei_total_(PA).xlsx")

# Usando LASSO
resultados_lasso <- treinar_e_projetar_modelo_sem(
  dados_completos = dados_completos,
  coluna_data = "data",
  tipo_modelo = "glmnet",
  alpha = 0.5  # LASSO
)

# Acessar os resultados
resultados_lasso$grafico  # Mostrar o gráfico