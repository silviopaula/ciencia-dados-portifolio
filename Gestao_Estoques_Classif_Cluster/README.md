# Gestão de Estoques com Machine Learning

Este projeto tem como objetivo **otimizar a gestão de inventário** de uma empresa de varejo por meio de análise exploratória de dados e modelos preditivos de machine learning.  
A motivação vem de um problema real: **apenas ~10% dos produtos em estoque vendem anualmente**, gerando custos com capital de giro e armazenagem.

---

## Objetivos do Projeto
- **Analisar** o comportamento dos produtos em estoque e históricos de vendas  
- **Construir modelos de Clusterização** para gerar features para classificação
- **Construir modelos de classificação** para prever a probabilidade de venda de cada item  
- **Gerar scores de decisão** para determinar quais produtos devem ser **mantidos** e quais devem ser **descartados**  
- **Apoiar a gestão** na tomada de decisões estratégicas, reduzindo custos e otimizando o espaço de armazenagem  

---

## Estrutura dos Dados
- **Dados Históricos**:  
  Contêm vendas dos últimos 6 meses e a variável `SoldFlag` (se vendeu ou não).  
  → Usados para treinar os modelos.

- **Dados Ativos**:  
  Inventário atual da empresa (sem vendas registradas).  
  → Recebem scores preditivos dos modelos.

**Principais variáveis:**
- `PriceReg`, `LowUserPrice`, `LowNetPrice` → preços  
- `ReleaseYear`, `ReleaseNumber` → ciclo de vida do produto  
- `MarketingType` → estratégia de marketing (Padrão ou Direto)  
- `ItemCount` → quantidade/unidade do item  
- `StrengthFactor` → métrica interna da empresa  

---

## Análise Exploratória
- **Dataset** com ~199k registros e 14 variáveis  
- **Distribuição**: 62% ativos vs 38% históricos  
- **Marketing**: Padrão e Direto quase equilibrados (com leve inversão entre ativos e históricos)  
- **Preços**: concentração em faixas baixas, cauda longa à direita  
- **Ano de lançamento**: centrado entre 2000 e 2015  
- **Outliers**: presentes em preço, fator de força e quantidade de itens → tratados com análise log/boxplots  

---

## Modelagem
Foi utilizada a biblioteca [PyCaret](https://pycaret.org/) para automatizar a comparação de modelos de classificação.  

**Modelos testados e resultados principais:**
- **LightGBM** → melhor *Accuracy*, *AUC* e *Precisão*  
- **Logistic Regression** → melhor *F1-Score*  
- **AdaBoost** → melhor *Kappa*  
- **QDA** → melhor *Recall*  

> A escolha do modelo depende do **custo de erro de negócio**:  
> - Quando o risco de manter produto encalhado é alto → priorizar **Precisão**  
> - Quando o risco de descartar produto que venderia é alto → priorizar **Recall**

---

## Aplicação ao Inventário
O melhor modelo (LightGBM) foi usado para **estimar a probabilidade de venda** dos 122k produtos ativos.

- **Produtos com Score ≥ 0.70** → recomendados para manter no estoque  
- **Produtos com 0.40 ≤ Score < 0.70** → manter sob observação (testar marketing/preço)  
- **Produtos com Score < 0.40** → candidatos a desinvestimento (liquidação/remoção)  

Arquivos finais exportados:
- `saida_scores_inventario_ativo.csv` → todos os SKUs com scores  
- `saida_top_manter.csv` → ranking dos melhores itens para manter  
- `saida_descartar.csv` → candidatos ao descarte  

---

## Principais Insights
- Apenas **17% dos produtos históricos venderam** nos últimos 6 meses → confirma excesso de estoque  
- Produtos com **Marketing Padrão** tiveram maior rotatividade no passado, enquanto o estoque atual é dominado por **Marketing Direto**  
- **Preço, ano de lançamento e estratégia de marketing** se mostraram variáveis decisivas na previsão de venda  
- O modelo preditivo permite diferenciar produtos promissores de itens encalhados, trazendo **precisão muito maior** do que a regra simples “se vendeu antes, vai vender de novo”  

---

## Tecnologias Utilizadas
- **Python 3.10**  
- **Bibliotecas**: Pandas, Numpy, Matplotlib, Seaborn, Plotly, Scikit-learn, Statsmodels, PyCaret  
- **Ambiente**: Jupyter Notebook / Anaconda  
- **Machine Learning**: classificação binária, avaliação de métricas (AUC, F1, Recall, Precisão, Kappa)  

---

## Impacto de Negócio Esperado
- **Redução de custos operacionais** com armazenagem de itens de baixa rotatividade  
- **Liberação de capital de giro** com a remoção de produtos 'encalhados'  
- **Aumento da eficiência** na estratégia de marketing e precificação  
- **Decisão baseada em dados**, com suporte estatístico e preditivo

---
