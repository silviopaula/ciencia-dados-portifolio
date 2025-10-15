# Comércio Exterior — Análise com PySpark e Método de Simulação

Este projeto apresenta uma análise exploratória e econométrica do **comércio exterior brasileiro**, com destaque para a relação com os **Estados Unidos** e o **impacto tarifário sobre o suco de laranja**.  
A análise foi desenvolvida em um Jupyter Notebook (`Comércio_exterior_com_Google_colab_e_spark.ipynb`) utilizando **PySpark** para processamento de dados em larga escala e **métodos de simulação** para estimar a incerteza dos resultados.

---

## Objetivo

- Explorar o comércio exterior brasileiro sob diferentes perspectivas: **geográfica, setorial e temporal**.  
- Quantificar a importância de parceiros comerciais (como China, EUA e Rússia) e produtos-chave nas exportações.  
- Avaliar o **impacto de políticas tarifárias** sobre produtos específicos, aplicando um modelo de **elasticidade-preço da demanda** com simulação Monte Carlo.

---

## Tecnologias Utilizadas

- **Python / PySpark** → Processamento distribuído de grandes bases (SECEX/ComexStat).  
- **Pandas / NumPy** → Manipulação e cálculo de métricas.  
- **Plotly / Seaborn / Matplotlib** → Visualizações interativas e exploratórias.  
- **Statsmodels** → Regressão log-log e estimação da elasticidade-preço.  
- **Monte Carlo Simulation** → Avaliação da incerteza das estimativas.  
- **Google Colab** → Execução em ambiente em nuvem, com suporte ao PySpark.

---

## Estrutura da Análise

1. **Carregamento e tratamento dos dados**  
   - Leitura de arquivos `.parquet` e integração com dicionários auxiliares (NCM e países).  
   - Limpeza, filtragem e criação de indicadores agregados.

2. **Exploração descritiva**  
   - Identificação dos principais destinos de exportação e produtos líderes por valor e quantidade.  
   - Análise regional dos municípios com maior volume comercializado.  
   - Estudo de sazonalidade e concentração geográfica das exportações.

3. **Estudo de Caso — Suco de Laranja**  
   - Estimação da **elasticidade-preço da demanda** para os EUA com modelo log-log.  
   - Aplicação de **simulação Monte Carlo** para mensurar incerteza e efeitos de variações de preço.  
   - Avaliação de cenários tarifários, incluindo a tarifa vigente de **10% ad valorem + US$ 415/tonelada**.

4. **Resultados principais**  
   - Elasticidade estimada ≈ **−1,5**, indicando **demanda elástica**.  
   - As tarifas atuais elevam o preço efetivo ao importador em cerca de **38%**, reduzindo as compras em magnitude semelhante.  
   - O comércio brasileiro mantém forte concentração em poucos parceiros e produtos, refletindo **vulnerabilidade externa e baixa diversificação**.

---

## Método de Simulação

O método de Monte Carlo foi utilizado para avaliar como **incertezas no parâmetro de elasticidade** afetam as projeções de consumo.

Etapas principais:
1. Estimação do modelo `ln(Q) = α + β ln(P)` para obter a elasticidade-preço.  
2. Amostragem de milhares de valores possíveis de β, segundo sua distribuição estatística (`Normal(β̂, SE(β)^2)`).  
3. Cálculo da nova quantidade demandada:
   \[
   Q_{novo} = Q_0 (1 + \Delta P)^{\beta}
   \]
4. Geração da distribuição simulada de variações percentuais na demanda.  
5. Visualização da incerteza (mediana, intervalos de 10%–90%).

---

## Principais Conclusões

- O comércio exterior brasileiro continua **altamente concentrado** em poucos parceiros (China e EUA) e produtos primários (soja, minério, petróleo).  
- As **exportações aos EUA** incluem produtos de maior valor agregado (aeronaves, químicos, sucos), mas são **fortemente sensíveis ao preço**.  
- O caso do **suco de laranja** mostra que, mesmo sem a sobretaxa de 50%, as tarifas atuais **reduzem significativamente a competitividade** brasileira.  
- O país precisa **diversificar mercados e produtos**, fortalecendo setores industriais e de maior valor agregado para reduzir vulnerabilidades externas.

---

## Execução Local

1. **Instale as dependências:**
   ```bash
   pip install pyspark statsmodels plotly seaborn matplotlib jupyterlab
