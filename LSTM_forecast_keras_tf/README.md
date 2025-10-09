# Previsão de Consumo de Energia com LSTM

Exercício vertical de modelagem de séries temporais usando três níveis de abstração: AutoKeras → Keras → TensorFlow.

## Objetivo

Prever consumo mensal de energia (2025-2030) comparando três abordagens:
- **AutoKeras**: Busca automatizada de arquiteturas
- **Keras**: Controle manual da arquitetura
- **TensorFlow**: Máximo controle e customização

## Dados

- **Período**: 2004-2025 (histórico) + 2025-2030 (previsão)
- **Frequência**: Mensal
- **Target**: Consumo de energia (MWh)
- **Features**: Sazonalidade, tendência, dummies mês/ano

## Modelos Implementados

1. AutoKeras (AutoML)
2. Keras Sequential 
3. TensorFlow customizado

## Métricas

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
