# 🚀 Análise Causal com CausalImpact: Seu Guia para Entender o Impacto de Eventos em Séries Temporais

## ✨ Transforme Dados em Insights Acionáveis\!

Este aplicativo Shiny, construído com a poderosa metodologia do pacote `CausalImpact` do Google, permite que você analise o verdadeiro impacto de eventos e intervenções em suas séries temporais de forma intuitiva, sem escrever uma única linha de código. Diga adeus à complexidade e olá à clareza\!

## 💡 Sobre o Projeto

O `CausalImpact` utiliza modelos bayesianos de séries temporais estruturais (BSTS) para desvendar o efeito causal de uma intervenção. Ele faz isso comparando o que realmente aconteceu após um evento com o que *teria acontecido* se o evento nunca tivesse ocorrido (o cenário contrafactual). Nosso aplicativo encapsula essa sofisticação em uma interface gráfica amigável, tornando a inferência causal acessível a todos.

## 🎯 Funcionalidades que Vão Impulsionar Suas Análises

### 📁 Entrada de Dados Simplificada

  * **Compatibilidade Ampla:** Carregue seus dados sem esforço em formatos CSV ou XLSX.
  * **Seleção Flexível:** Escolha facilmente a coluna de data, a variável resposta (Y) e as variáveis explicativas (Xs) para sua análise.
  * **Controle Total de Período:** Defina com precisão o período de análise e a data exata da sua intervenção.

### 📈 Tratamento Inteligente de Séries Temporais

  * **Desazonalização Avançada:** Utilize a decomposição STL para remover a sazonalidade e obter uma visão mais clara da tendência.
  * **Foco na Tendência:** Opção para extrair apenas o componente de tendência da sua série.
  * **Frequência Personalizada:** Configure a frequência sazonal (mensal, trimestral, semanal) para se adequar perfeitamente aos seus dados.

### ⚙️ Modelagem Avançada ao Seu Alcance

  * **Parâmetros BSTS:** Ajuste manualmente os parâmetros do modelo BSTS para análises mais aprofundadas.
  * **Controle de Significância:** Defina o nível de significância ($\\alpha$) desejado.
  * **Iterações MCMC:** Configure o número de iterações MCMC para maior precisão.
  * **Priors Configuráveis:** Especifique desvios-padrão para os priors do modelo.

### 📊 Visualização de Resultados Imersiva

  * **Gráficos Interativos:** Compare dinamicamente a série real com a contrafactual projetada.
  * **Coeficientes do Modelo:** Entenda a contribuição de cada covariável com a visualização dos coeficientes.
  * **Intervalo de Confiança:** Monitore a incerteza da sua previsão contrafactual com intervalos de confiança configuráveis.
  * **Tabelas de Impacto:** Veja o impacto agregado por diferentes períodos.

### 📤 Exportação e Relatórios Completos

  * **Relatórios em Excel:** Baixe um relatório abrangente em formato XLSX, consolidando todas as suas análises.
  * **Relatórios Textuais Detalhados:** Obtenha resumos e relatórios detalhados em texto.
  * **Métricas Chave:** Acesse métricas cruciais como efeito absoluto, efeito relativo e a probabilidade causal.

## 🚀 Como Usar (Guia Rápido)

1.  **Carregue seus Dados:** Faça o upload de um arquivo CSV ou XLSX contendo seus dados.
2.  **Configure as Colunas:** Selecione a coluna de data e as variáveis para análise.
3.  **Defina os Períodos:** Defina o período de análise e a data do evento.
4.  **Ajustes Avançados (Opcional):** Configure as opções avançadas, se necessário.
5.  **Execute a Análise:** Clique em "Executar Análise" para gerar resultados.
6.  **Explore e Exporte:** Explore os resultados nas diferentes abas e exporte relatórios para documentação e apresentações.

## 🛠️ Requisitos

  * **R:** Versão 3.6 ou superior.
  * **Pacotes Essenciais:** `shiny`, `tidyverse`, `lubridate`, `plotly`, `zoo`, `CausalImpact`, `DT`, `htmltools`, `readxl`, `shinythemes`, `shinyjs`, `shinydashboard`, `shinyWidgets`, `data.table`, `openxlsx`.

## ⚠️ Limitações e Considerações Importantes

  * A inferência causal depende da qualidade das covariáveis selecionadas.
  * Resultados robustos geralmente requerem períodos pré-intervenção suficientemente longos.
  * A análise pressupõe que o modelo contrafactual seja adequado.
  * Em séries muito curtas, a desazonalização pode ser menos confiável.

## 📚 Referências

  * [**Documentação Oficial do CausalImpact**](https://google.github.io/CausalImpact/CausalImpact.html)
  * [**Artigo Científico: Brodersen et al. (2015)**](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.full)

-----
