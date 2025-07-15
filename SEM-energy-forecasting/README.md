Com certeza! Baseado no conteúdo do arquivo HTML, preparei um `README.md` completo e bem estruturado. Este README destaca os objetivos, a metodologia, os resultados e as considerações importantes do seu projeto de modelagem SEM.

---

# Análise e Projeção de Consumo de Energia com Modelagem de Equações Estruturais (SEM)

##  Visão Geral

Este projeto apresenta uma análise detalhada e a projeção do consumo de energia elétrica utilizando **Modelos de Equações Estruturais (SEM)**. A abordagem integra um sistema complexo de variáveis econômicas (PIB, renda, produção industrial), demográficas (população) e climáticas (temperatura, sazonalidade) para construir um modelo robusto que captura tanto as relações diretas quanto as indiretas que influenciam o consumo energético.

O resultado é um framework que não apenas explica as dinâmicas do consumo de energia, mas também oferece uma capacidade preditiva para projeções futuras, validada por métricas estatísticas.

### Objetivos do Projeto

-   **Modelar Relações Complexas:** Capturar a interdependência entre indicadores econômicos, demográficos e o consumo de energia.
-   **Desenvolver Capacidade Preditiva:** Criar um modelo robusto para realizar projeções de consumo energético.
-   **Incorporar Fatores Externos:** Integrar o impacto da sazonalidade, clima e eventos macroeconômicos (como crises e a pandemia) na análise.
-   **Avaliar a Performance:** Medir a acurácia do modelo através de métricas como o MAPE (Mean Absolute Percentage Error).

---

## Diagrama do Modelo Estrutural

O diagrama abaixo ilustra a estrutura causal do modelo SEM implementado. As variáveis são agrupadas em exógenas (fatores externos), temporais e endógenas (variáveis modeladas dentro do sistema).


*Diagrama da estrutura do Modelo de Equações Estruturais (SEM).*

---

## Metodologia

O fluxo de trabalho do projeto foi estruturado nas seguintes etapas:

### 1. Configuração do Ambiente
A análise é conduzida em R, utilizando o gerenciador de pacotes `pacman` para garantir que todas as bibliotecas necessárias estejam instaladas e carregadas.

### 2. Carga e Pré-processamento de Dados
Os dados, contendo séries temporais mensais, foram carregados de um arquivo Excel. Uma inspeção inicial foi feita para verificar a integridade e o formato das variáveis.

### 3. Engenharia de Variáveis
Para enriquecer o modelo e capturar dinâmicas complexas, foram criadas novas variáveis:
-   **Transformações Logarítmicas:** Para normalizar distribuições e estabilizar a variância.
-   **Componentes Sazonais:** Uso de funções seno e cosseno para modelar a ciclicidade anual.
-   **Variáveis Dummy:** Para isolar o efeito de eventos específicos como a pandemia de COVID-19 e crises econômicas, além de períodos climáticos extremos (ondas de calor/frio).
-   **Tendência Temporal:** Uma variável de tendência foi incluída para capturar o crescimento secular.

### 4. Modelagem com SEM
O coração do projeto é um sistema de equações simultâneas especificado e estimado com o pacote `lavaan`. A estrutura hierárquica do modelo é a seguinte:
1.  **Equações Macroeconômicas:** PIB, Massa de Renda, Produção Industrial (PIM) e Vendas no Comércio (PMC) são modelados como funções de variáveis exógenas e de outras variáveis endógenas.
2.  **Equação de Consumo de Energia:** A variável-alvo é modelada como uma função de todos os indicadores econômicos, fatores climáticos, preço da energia e controles temporais.

A estimação foi realizada utilizando o método **Full Information Maximum Likelihood (FIML)**, que é robusto para lidar com dados ausentes.

### 5. Projeção e Validação
-   Os dados foram divididos em conjuntos de **treino** (até julho de 2024) e **projeção** (após julho de 2024).
-   O modelo treinado foi utilizado para gerar projeções para o período futuro.
-   A performance do modelo foi avaliada calculando o **MAPE** nos últimos 6 meses do período de treino, resultando em um erro de apenas **1.78%**, o que indica uma alta acurácia preditiva.

### 6. Visualização
O resultado final, comparando os dados históricos com as projeções do modelo, foi visualizado em um gráfico interativo.


*Comparação entre o consumo de energia realizado e o previsto/projetado pelo modelo SEM.*

---

## Tecnologias Utilizadas

-   **Linguagem:** R
-   **Pacotes Principais:**
    -   `lavaan`: Para Modelagem de Equações Estruturais.
    -   `tidyverse`: (dplyr, ggplot2) para manipulação e visualização de dados.
    -   `lubridate`: Para manipulação de datas.
    -   `visNetwork`: Para a criação do diagrama do modelo.
    -   `readxl`: Para importação de dados.

---


## Limitações e Considerações

Embora poderoso, o modelo SEM possui limitações importantes para projeções de séries temporais:
-   **Linearidade:** O modelo assume relações lineares, o que pode simplificar excessivamente dinâmicas complexas.
-   **Estacionariedade dos Parâmetros:** A análise assume que as relações causais são estáveis ao longo do tempo, o que pode não ser verdade em cenários de mudanças estruturais (ex: transição energética).
-   **Dependência de Projeções Externas:** A acurácia das projeções depende da qualidade das projeções de variáveis exógenas como PIB, inflação e população.
-   **Propagação de Erro:** Em um sistema de equações, erros em uma equação podem se propagar e amplificar em outras.

Para mitigar esses pontos, abordagens futuras poderiam incluir modelos híbridos, machine learning ou análise de cenários.