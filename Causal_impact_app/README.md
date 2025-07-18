# Causal_impact_app

**Descrição para o repositório no GitHub:**

Análise Causal com CausalImpact
Um aplicativo Shiny para análise de impacto causal em séries temporais utilizando a metodologia do pacote CausalImpact do Google.
Sobre o projeto
Este aplicativo facilita a implementação da metodologia de inferência causal em séries temporais desenvolvida pelo Google, proporcionando uma interface gráfica intuitiva que elimina a necessidade de codificação para realizar análises sofisticadas.
O CausalImpact utiliza modelos bayesianos de séries temporais estruturais (BSTS) para estimar o impacto causal de uma intervenção, comparando a série observada após o evento com uma contrafactual projetada.
Funcionalidades
Entrada de dados

Carregamento de arquivos CSV e XLSX
Seleção flexível de colunas para análise
Definição personalizada do período de análise e data de intervenção

**Tratamento de séries temporais**

Opções de desazonalização via decomposição STL
Extração de componentes de tendência
Configuração da frequência sazonal (mensal, trimestral, semanal)

**Modelagem avançada**

Configuração manual de parâmetros BSTS
Controle do nível de significância (alpha)
Ajuste do número de iterações MCMC
Especificação de desvios-padrão para priors

**Visualização de resultados**

Gráficos interativos da série real vs. contrafactual
Visualização de coeficientes do modelo
Intervalo de confiança configurável
Tabelas de impacto agregadas por diferentes períodos

**Exportação e relatórios**

Download de relatórios completos em formato Excel
Exportação de relatórios textuais detalhados
Tabelas de impacto agrupadas por diferentes frequências temporais
Métricas-chave como efeito absoluto, relativo e probabilidade causal

**Como usar**

Carregue um arquivo CSV ou XLSX contendo seus dados
Selecione a coluna de data e as variáveis para análise
Defina o período de análise e a data do evento
Configure as opções avançadas, se necessário
Clique em "Executar Análise" para gerar resultados
Explore os resultados nas diferentes abas
Exporte relatórios para documentação e apresentações

**Requisitos**

R versão 3.6 ou superior
Pacotes: shiny, tidyverse, lubridate, plotly, zoo, CausalImpact, DT, htmltools, readxl, shinythemes, shinyjs, shinydashboard, shinyWidgets, data.table, openxlsx

**Limitações e considerações**

A inferência causal depende da qualidade das covariáveis selecionadas
Resultados robustos geralmente requerem períodos pré-intervenção suficientemente longos
A análise pressupõe que o modelo contrafactual seja adequado
Em séries muito curtas, a desazonalização pode ser menos confiável

**Referências**                                          

[Documentação oficial do CausalImpact](https://google.github.io/CausalImpact/CausalImpact.html)                                                                        
[Brodersen et al. (2015)](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.full)
