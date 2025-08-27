# Exemplo de Aplicação do MLflow – Previsão de Churn

Este repositório foi criado com o objetivo de demonstrar, como integrar uma aplicação simples de Machine Learning ao MLflow.  
A ideia central não é a complexidade dos modelos em si, mas sim mostrar o fluxo completo de rastreamento, registro e versionamento de experimentos de ML.

---

##  Objetivo do exercício

- **Construir um caso didático** de Machine Learning para previsão de **churn (rotatividade de clientes)**.  
- **Treinar modelos básicos** (Decision Tree, Random Forest e Ensemble) em uma base de clientes fictícia.  
- **Avaliar métricas de desempenho** como acurácia e F1-score.  
- **Integrar o processo ao MLflow**, utilizando recursos de:
  - Rastreamento de parâmetros, métricas e artefatos;
  - Criação e gerenciamento de experimentos;
  - Registro de modelos no **Model Registry**;
  - Uso de **aliases** para controlar a versão em produção.

---

## O que foi feito

1. **Configuração do MLflow local**  
   - Execução da interface UI na porta local (`localhost:5000`).  
   - Organização de um diretório dedicado para armazenar runs e artefatos.  

2. **Aplicação de Machine Learning**  
   - Carregamento da base `abt.csv`.  
   - Definição de variáveis explicativas e da variável alvo (`flag_churn`).  
   - Treinamento de modelos supervisionados clássicos.  
   - Cálculo e comparação das métricas.

3. **Integração ao MLflow**  
   - Registro automático de parâmetros, métricas e artefatos com `autolog`.  
   - Registro manual de métricas adicionais (acurácia treino/teste).  
   - Criação de experimentos nomeados para organização.  
   - Comparação estruturada de diferentes execuções (runs).

4. **Versionamento e Governança de Modelos**  
   - Registro de modelos no **Model Registry**.  
   - Consulta a diferentes versões registradas.  
   - Utilização de **aliases** (ex.: `@production`) para definir qual modelo deve ser consumido em produção sem alterar o código.

---

## Aprendizados principais

- O MLflow não é apenas uma ferramenta de visualização: ele fornece **estrutura e governança** para todo o ciclo de vida de modelos de ML.  
- Cada execução gera um **run** rastreável, permitindo **comparar hiperparâmetros, métricas e resultados**.  
- O **Model Registry** possibilita gerenciar versões de modelos de forma profissional, incluindo estágios (Staging, Production) e aliases.  
- Mesmo em um **exemplo simples de churn**, é possível aplicar práticas utilizadas em ambientes corporativos de ML.

---

## Conclusão

Este exercício serviu para **explorar, em pequena escala, o ciclo completo de um projeto de Machine Learning com MLflow**:

- Da preparação dos dados ao treino de modelos;  
- Do rastreamento de execuções ao registro de modelos;  
- Da avaliação de métricas à gestão de versões em produção.  

Ou seja, um **mini-pipeline de MLOps** aplicado a um caso prático e acessível, que pode ser expandido para projetos reais de maior porte.

---
