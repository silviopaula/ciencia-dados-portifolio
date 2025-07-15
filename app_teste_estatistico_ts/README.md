# 📊 Dashboard Premium de Análise de Séries Temporais

Este projeto é um aplicativo interativo desenvolvido em R com Shiny, focado em análise estatística completa de séries temporais. Ele oferece uma interface moderna, intuitiva e responsiva, ideal para profissionais, pesquisadores e estudantes que desejam explorar, diagnosticar e prever séries temporais de forma prática e visual.

## 🚀 Funcionalidades Principais

- **Suporte a múltiplos formatos de dados:**
  - Excel (.xlsx)
  - CSV (vírgula ou ponto e vírgula)
  - Dados simulados para demonstração
- **Testes estatísticos automáticos:**
  - Estacionariedade: ADF, Phillips-Perron, KPSS
  - Autocorrelação: Box-Pierce, Ljung-Box
  - Heterocedasticidade: ARCH
  - Normalidade: Anderson-Darling
- **Visualizações interativas:**
  - Série temporal, histograma, boxplot, resíduos
  - Gráficos ACF e PACF
  - Decomposição STL (tendência, sazonalidade, resíduo)
- **Modelagem e previsão:**
  - Ajuste automático de modelos ARIMA
  - Previsão com intervalos de confiança
- **Exportação completa:**
  - Resultados e dados processados em Excel
- **Interface premium:**
  - Design moderno, responsivo e customizado
  - Guias explicativos e dicas de interpretação


## 🖥️ Como Usar

1. **Escolha a fonte de dados:**
   - Carregue um arquivo Excel, CSV ou utilize os dados simulados.
2. **Selecione as colunas de data e valor (para arquivos próprios).**
3. **Clique em "Executar Análise" para processar e visualizar os resultados.**
4. **Explore as abas:**
   - Testes estatísticos
   - Visualizações
   - Decomposição & Previsão
   - Dados processados
   - Guia completo de interpretação
4. **Exporte os resultados ou dados processados em Excel.**

## 📚 Estrutura do App

- `app_time_series_test.r`: Código principal do aplicativo Shiny
- `README.md`: Este arquivo de documentação
- `rsconnect/`: Arquivos de configuração para publicação (opcional)

## 📝 Exemplos de Aplicação

- Diagnóstico de séries temporais financeiras, econômicas, ambientais, etc.
- Ensino de conceitos de estacionariedade, autocorrelação e modelagem ARIMA
- Demonstração de técnicas de previsão e decomposição de séries


## 📄 Licença

Este projeto está licenciado sob a licença MIT. Sinta-se livre para usar, modificar e compartilhar!
---

