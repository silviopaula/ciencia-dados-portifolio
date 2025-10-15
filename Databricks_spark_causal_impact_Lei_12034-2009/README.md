# Análise do Impacto Causal da Lei nº 12.034/2009 em Eleições Brasileiras

## Nota:
Este projeto faz parte de um estudo complementar ao **artigo científico sobre o impacto causal da Lei nº 12.034/2009 (Minirreforma Eleitoral)**, que analisou os efeitos da legislação sobre a **participação feminina nas eleições brasileiras**.  

No artigo principal, aplicamos métodos de **avaliação de impacto causal**, em especial o modelo de **Diferenças em Diferenças (Difference-in-Differences)** na formulação de **Callaway & Sant’Anna (2021)**, que permite estimar efeitos de políticas públicas em períodos distintos e para diferentes grupos tratados.  

Quando o artigo for publicado disponibilizarei toda análise.

## Sobre este exercício

Este projeto analisa o impacto causal da Lei nº 12.034/2009 (Minirreforma Eleitoral) na participação feminina nas eleições brasileiras. O estudo foca especificamente nas candidaturas para vereadora, deputada estadual e distrital, utilizando dados do TSE (Tribunal Superior Eleitoral) e modelos econométricos de inferência causal.

## Contexto da Lei

A Lei 12.034/2009 estabeleceu uma importante mudança nas regras eleitorais brasileiras, especialmente em relação às cotas de gênero para candidaturas proporcionais. Os principais pontos são:

- **Exigência**: Mínimo de 30% e máximo de 70% de candidaturas para cada gênero
- **Aplicação**: Eleições proporcionais (deputado federal, estadual/distrital e vereador)
- **Mudança Fundamental**: Alteração do termo "reservar" para "preencher" vagas, eliminando brechas legais

## Escopo da Análise

- **Dados**: Utilização da base de dados do TSE via Google BigQuery (Base dos Dados)
- **Tecnologias**: 
  - PySpark para processamento de dados em larga escala
  - Plotly para visualizações
  - Pandas para manipulação de dados
  - Scipy para análises estatísticas
  - Causal impact

## Estrutura do Projeto

- `Databricks Python Spark - Prepare datapanel.ipynb`: Notebook principal com a preparação dos dados
- `Databricks Python Spark - Prepare datapanel.html`: Versão HTML do notebook para visualização
- `requirements.txt`: Lista de dependências Python necessárias

## Aspectos Legais Relevantes

- A verificação do cumprimento da cota é feita por partido ou federação
- O não cumprimento pode resultar no indeferimento do DRAP (Demonstrativo de Regularidade de Atos Partidários)
- Fraudes à cota podem resultar em:
  - Cassação do DRAP e diplomas dos eleitos
  - Anulação dos votos da chapa
  - Inelegibilidade dos responsáveis

## Observação Adicional

O TSE e o STF vincularam a distribuição do fundo eleitoral e do tempo de rádio/TV às mesmas proporções de gênero, exigindo mínimo de 30% desses recursos para candidaturas femininas quando o partido atinge o mínimo de 30% de candidatas.

## Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o notebook no ambiente Databricks ou em um ambiente local com Jupyter

## Fonte dos Dados

Os dados utilizados neste projeto são provenientes da Base dos Dados, disponíveis através do Google BigQuery:
[Link para os dados](https://console.cloud.google.com/bigquery?p=basedosdados&d=br_tse_eleicoes&t=candidatos)