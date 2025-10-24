# Endogeneidade e Elasticidade-Preço: OLS vs IV

Demonstração dos efeitos da endogeneidade na estimação de elasticidades preço-demanda e correção via variáveis instrumentais.

## Problema

Quando preço e quantidade são determinados simultaneamente no mercado, surge correlação entre o preço e o termo de erro, violando o pressuposto fundamental de MQO e gerando estimativas enviesadas.

## Estrutura dos Dados

**Variáveis:**
- Dependente: `log(vendas)`
- Endógena: `log(preço)`
- Exógenas: `log(renda)`, `sazonalidade`
- Instrumento: `tempo` (tendência temporal)

**Fonte da endogeneidade:**
Choques de demanda afetam simultaneamente vendas e preços, criando correlação espúria.

## Resultados

### Elasticidade-Preço Estimada

| Modelo | Coeficiente | Viés |
|--------|-------------|------|
| Verdadeiro | -1.80 | - |
| OLS | -0.73 | +1.07 (59.7%) |
| IV/2SLS | -1.69 | +0.11 (6.1%) |

### Diagnósticos do Modelo IV

- **Weak instruments:** F = 73.45 (p < 0.001) - instrumento forte
- **Wu-Hausman:** p < 0.001 - endogeneidade confirmada
- **Conclusão:** IV é necessário e adequado

## Implicações Práticas

**OLS subestima elasticidade:**
- Sugere baixa sensibilidade ao preço
- Leva a decisões de precificação incorretas
- Pode resultar em perda de receita

**IV corrige o viés:**
- Revela alta elasticidade (-1.69)
- Demanda é sensível a variações de preço
- Permite decisões estratégicas corretas

## Trade-off: Previsão vs Causalidade

### Métricas Preditivas

| Modelo | RMSE | MAE | R² |
|--------|------|-----|-----|
| OLS | 0.129 | 0.100 | 0.699 |
| IV | 0.458 | 0.261 | 0.616 |

**Por que OLS prevê melhor?**
- Captura correlação espúria (aprende a endogeneidade)
- Menor variância, maior viés
- Bom para forecasting, ruim para inferência causal

**Por que usar IV então?**
- Objetivo é estimar efeito causal
- Necessário para decisões baseadas em elasticidades
- R² e RMSE não são apropriados para avaliar IV

## Quando Usar Cada Método

**Use OLS se:**
- Objetivo é predição pura
- Exogeneidade é plausível
- Não precisa interpretar coeficientes

**Use IV se:**
- Objetivo é inferência causal
- Há endogeneidade suspeita/confirmada
- Decisões dependem de relações causais (pricing, políticas)

## Requisitos
```r
pacman::p_load(tidyverse, AER, plotly, lubridate, stargazer, Metrics)
```

## Reprodução
```r
set.seed(456)
source("MQO-vs-IV.R")
```

## Referências

- Angrist & Pischke (2009, 2015) - Mostly Harmless Econometrics
- Wooldridge (2010) - Econometric Analysis of Cross Section and Panel Data
- Stock & Watson (2015) - Introduction to Econometrics

## Autor

Silvio da Rosa Paula