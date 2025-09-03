# Time Series Chronos Forecast

## Descrição

Este README documenta o projeto Time_Series_Chronos_Forecast.ipynb, uma implementação completa e prática para previsão de séries temporais utilizando os modelos de última geração Amazon Chronos (famílias T5 e BOLT).

O projeto apresenta uma demonstração hands-on do package chronos-forecasting, explorando suas capacidades para forecasting univariado através de diferentes variantes de modelos pré-treinados.

Desenvolvo uma função robusta e reutilizável que simplifica significativamente a implementação dos modelos Chronos, automatizando todo o pipeline desde o carregamento dos dados até a geração de previsões com intervalos de confiança e métricas de avaliação detalhadas.

A solução inclui visualizações avançadas de performance com gráficos interativos que facilitam a interpretação dos resultados e comparação entre diferentes modelos. Para tornar a ferramenta acessível a usuários não-técnicos, também desenvolvo uma aplicação web em Streamlit que permite upload de dados, configuração de parâmetros via interface gráfica e execução de forecasts univariados de forma intuitiva e eficiente.

### Características principais:
- Preparação dos dados.
- Definição da função de previsão multimarcas (vários modelos de uma vez)
- Separação treino/teste
- Métricas (MAE, RMSE, MAPE, MedAE, R²)
- Intervalos preditivos (quantis como P10/P50/P90)
- Gráficos interativos (Plotly)
- Tabelas consolidadas com as previsões por modelo e ensemble

## Estrutura do Conteúdo

1. **Setup** — pacotes, seeds e utilitários
2. **Dados** — série alvo (ex.: consumo de energia) e período
3. **Preparação** — agregação, pivoteamento e padronização de colunas
4. **Função principal** — rotina que:
   - Carrega um ou mais modelos Chronos (online/offline)
   - Calcula quantis (pL / p50 / pU)
   - Avalia no período de teste
   - Plota as curvas e intervalos
   - Retorna `metrics_df`, `df_treino`, `df_teste`, `df_forecast`
5. **Resultados** — gráficos, métricas e (opcional) ensemble
6. **Exportação** — salvamento dos artefatos (tabelas/figuras)

### Resultado

## Saídas da Função de Forecast

### `metrics_df`
Uma linha por modelo com:
- MAE, RMSE, MAPE (%), MedAE, R²
- Quantis usados

### `forecasts`
Dicionário `{label → DataFrame}` com colunas:
- `pL` (limite inferior)
- `p50` (mediana) 
- `pU` (limite superior)

### `df_forecast`
DataFrame unificado = dados originais + colunas de previsão no padrão:
```
{target}__{label}__pL | {target}__{label}__p50 | {target}__{label}__pU
```

### Gráfico Plotly
- Série real (treino/teste)
- P50 de cada modelo
- Faixas dos intervalos na mesma cor do modelo (mais clara)

## Interpretação das Métricas

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| **MAE** | Erro médio absoluto | Quanto menor, melhor |
| **RMSE** | Erro quadrático médio | Quanto menor, melhor |
| **MAPE (%)** | Erro percentual médio | Útil para comparar séries em magnitudes distintas |
| **MedAE** | Erro absoluto mediano | Robusto a outliers |
| **R²** | Variação explicada | Quanto mais próximo de 1, melhor |
| **P50** | Mediana | Previsão "central" |
| **P10/P90** | Limites do intervalo | Incerteza da previsão |

## Dados e Modelos

### Dados
- **Formato:** Série alvo em frequência mensal (ex.: consumo de energia)
- **Ajuste:** Nome da coluna de data e coluna alvo conforme sua base

### Modelos Chronos

#### Offline
Pastas com `config.json` e pesos; passe o diretório-base na função.

> **Observação:** Modelos BOLT aceitam quantis no intervalo [0.1, 0.9]; a função ajusta automaticamente se necessário.

## Autor

**Dr. Silvio da Rosa Paula**
- GitHub: [https://github.com/silviopaula](https://github.com/silviopaula)
--
