# ⚖️ Endogeneidade e Elasticidade-Preço: MQO (OLS) vs. Variáveis Instrumentais (IV / 2SLS) em Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-2C5BB4?style=for-the-badge)
![Linearmodels](https://img.shields.io/badge/Linearmodels-IV2SLS-1E5631?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

**Demonstração didática em Python de como a endogeneidade destrói decisões de precificação (*Pricing*), como o método 2SLS/IV corrige o viés e o paradoxo entre Previsão (RMSE) e Causalidade**

</div>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [O Problema da Endogeneidade no Pricing](#-o-problema-da-endogeneidade-no-pricing)
- [A Solução: Variáveis Instrumentais (IV / 2SLS)](#-a-solução-variáveis-instrumentais-iv--2sls)
- [Principais Resultados](#-principais-resultados)
- [O Paradoxo: Previsão (RMSE) vs. Decisão de Preço](#-o-paradoxo-previsão-rmse-vs-decisão-de-preço)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Autor](#-autor)
- [Referências](#-referências)

---

## 🎯 Sobre o Projeto

Uma das perguntas mais valiosas em negócios é: **"Se eu aumentar o preço em 10%, quanto as vendas vão cair e o que acontece com a receita?"**.

Este projeto em **Python** (`statsmodels` e `linearmodels`) demonstra de forma passo a passo e intuitiva:
1. Por que a regressão linear comum (**MQO / OLS**) falha gravemente ao estimar elasticidade-preço quando há choques de demanda não observados (**endogeneidade**).
2. Como o método de **Variáveis Instrumentais (IV / 2SLS)**, usando um choque de custo exógeno (`custo_insumo`) como instrumento, isola a variação limpa do preço e recupera a verdadeira elasticidade causal.
3. Por que o MQO vence em métricas de previsão histórica (**menor RMSE e MAE**), mas **leva a empresa ao prejuízo** em uma simulação contrafactual de aumento de preço (+10%).

---

## ⚠️ O Problema da Endogeneidade no Pricing

No mundo real, empresas não sorteiam preços aleatoriamente. Quando acontece um **choque de demanda não observado** (um feriado, clima favorável ou produto viralizado):
1. As **vendas sobem** devido à procura aquecida.
2. A empresa **aumenta o preço** aproveitando a demanda alta.

Ao jogar esses dados em uma regressão linear comum (**MQO / OLS**), o modelo observa preços altos acompanhados de vendas altas e mistura o efeito causal do preço com o choque de demanda ($\text{Corr}(\log\text{preço}, \text{choque\_demanda}) = +0,908$), violando $E[\varepsilon \mid X] = 0$ e **subestimando drasticamente a sensibilidade do consumidor**.

---

## 🛠️ A Solução: Variáveis Instrumentais (IV / 2SLS)

Para corrigir a endogeneidade, utilizamos o **custo da matéria-prima (`log_custo`)** como instrumento econômico clássico:
- **Regra 1 — Relevância**: O custo do insumo encarece o preço final ($\text{Corr} = +0,394$, **Estatística F do 1º Estágio = `95,55`**, muito acima do corte clássico $F > 10$).
- **Regra 2 — Restrição de Exclusão**: O custo da matéria-prima não afeta o desejo do consumidor diretamente — apenas por meio do preço final na prateleira ($\text{Corr}(\log\text{custo}, \text{choque\_demanda}) = +0,052$).

O método **2SLS (Mínimos Quadrados em Dois Estágios)** descarta a parcela contaminada do preço e estima a elasticidade usando apenas a variação exógena induzida pelo custo.

---

## 📈 Principais Resultados

### 1. Elasticidade-Preço Estimada e Diagnósticos

| Modelo | Elasticidade-Preço | Viés Absoluto | Erro Relativo | Classificação Econômica |
| :--- | :---: | :---: | :---: | :--- |
| **Verdadeiro (Simulação)** | **`-1,8000`** | `0,0000` | `0,0%` | Demanda Elástica ($|\varepsilon| > 1$) |
| **MQO / OLS (Enviesado)** | **`-0,7735`** | `+1,0265` | **`57,0%`** | Demanda Inelástica ($|\varepsilon| < 1$) — **ERRADO!** |
| **IV / 2SLS (Corrigido)** | **`-1,6824`** | `+0,1176` | **`6,5%`** | Demanda Elástica ($|\varepsilon| > 1$) — **CORRETO!** |

- **Diagnóstico de Instrumento Fraco (1º Estágio)**: $F = 95,55$ ($p < 0,001$) $\rightarrow$ **Instrumento Forte**.
- **Teste de Endogeneidade de Wu-Hausman**: $\text{Stat} = 735,11$ ($p < 0,001$) $\rightarrow$ **Endogeneidade confirmada; uso de IV indispensável**.

---

## 🔄 O Paradoxo: Previsão (RMSE) vs. Decisão de Preço

### 1. No Passado (Sem Intervenção): MQO vence no RMSE
| Modelo | RMSE (Vendas) | MAE (Vendas) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **MQO / OLS (Enviesado)** | **`0,2382`** | **`0,1737`** | **`17,6%`** |
| **IV / 2SLS (Corrigido)** | `0,6794` | `0,4166` | `37,9%` |

*Por quê?* Porque o MQO aproveita a correlação espúria (usa o preço como "termômetro" do choque de demanda oculto) para colar melhor nos dados históricos. O IV descarta essa correlação espúria de propósito, aumentando a variância das previsões históricas.

### 2. No Futuro (Decisão de Pricing: Subir o Preço em +10%): IV salva a empresa!
| Cenário (+10% no Preço) | Elasticidade Usada | Variação nas Vendas | Impacto na Receita Total | Recomendação do Modelo |
| :--- | :---: | :---: | :---: | :--- |
| **Previsão do MQO / OLS** | `-0,7735` | `-7,1%` | **`+2,2%` (Lucro ilusório)** | Subir preço! *(Erro grave)* |
| **Previsão do IV / 2SLS** | `-1,6824` | `-14,8%` | **`-6,3%` (Alerta real)** | **Não subir preço!** |
| **REALIDADE DO MERCADO** | `-1,8000` | `-15,8%` | **`-7,3%` (Prejuízo real)** | Subir preço destrói receita |

> **Lição de Ouro**: Use **MQO / ML** quando o objetivo for **previsão pura** sem mudar a política atual. Use **IV / Inferência Causal** quando a decisão exigir saber **o que acontece se você intervir no preço**.

---

## 📁 Estrutura do Projeto

```text
endogeneidade-mqo-vs-2sls/
├── MQO_vs_IV.ipynb       # Jupyter Notebook didático em Python (passo a passo + 3 gráficos)
├── MQO_vs_IV.html        # Versão exportada em HTML para leitura rápida
├── requirements.txt      # Dependências Python do projeto
└── README.md             # Documentação executiva do projeto
```

---

## 🚀 Como Executar

1. **Clone o repositório e acesse a pasta do projeto:**
   ```bash
   git clone https://github.com/silviopaula/ciencia-dados-portifolio.git
   cd ciencia-dados-portifolio/endogeneidade-mqo-vs-2sls
   ```

2. **Instale as dependências em Python:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Abra e execute o Jupyter Notebook:**
   ```bash
   jupyter notebook MQO_vs_IV.ipynb
   ```

---

## 👤 Autor

**Silvio da Rosa Paula**  
*Doutor em Economia Aplicada (UFPel) | Especialista em Inteligência Artificial e Ciência de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/silvio-paula/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/silviopaula)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:silvio.rosa.paula@gmail.com)

---

## 📚 Referências

1. Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
2. Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
3. Stock, J. H., & Watson, M. W. (2015). *Introduction to Econometrics*. Pearson.
