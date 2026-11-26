# ⏳ Análise de Sobrevivência Aplicada (*Survival Analysis*): Prevendo *Quando* o Cliente Cancela (*Time-to-Churn*)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Lifelines](https://img.shields.io/badge/Lifelines-0.30%2B-1E5631?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Survival%3ACox-EB5E28?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

**Guia completo e aplicado de Análise de Sobrevivência em Python: Censura à Direita, Kaplan-Meier, Teste Log-Rank, Nelson-Aalen, Regressão de Cox (*Hazard Ratios*) e XGBoost `survival:cox`**

</div>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Por que Análise de Sobrevivência e Não Classificação Binária?](#-por-que-análise-de-sobrevivência-e-não-classificação-binária)
- [Dataset e Estrutura dos Dados](#-dataset-e-estrutura-dos-dados)
- [Metodologia Passo a Passo](#-metodologia-passo-a-passo)
- [Principais Resultados](#-principais-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Autor](#-autor)
- [Referências](#-referências)

---

## 🎯 Sobre o Projeto

Em negócios de receita recorrente (SaaS, telecomunicações, streaming, clubes de assinatura), saber **se** um cliente vai cancelar é apenas metade da resposta. O que define o *Lifetime Value* (LTV) e a prioridade do time de retenção é saber **QUANDO** esse cancelamento tende a ocorrer e **quais fatores aceleram ou protegem o tempo de vida do cliente**.

Este projeto apresenta um fluxo completo, didático e 100% reprodutível de **Análise de Sobrevivência (*Survival Analysis*)** em Python, cobrindo desde estimadores não-paramétricos (**Kaplan-Meier** e **Nelson-Aalen**) e econometria semiparamétrica (**Regressão de Riscos Proporcionais de Cox**) até **Machine Learning de Sobrevivência com XGBoost (`objective="survival:cox"`)**.

---

## 💡 Por que Análise de Sobrevivência e Não Classificação Binária?

1. **Captura a Dimensão Temporal ($S(t) = P(T > t)$)**: Diferencia um cliente que sai no 2º mês de um cliente que permanece ativo por 30 meses.
2. **Trata Corretamente a Censura à Direita (*Right-Censoring*)**: Na nossa base de 3.000 assinantes, **60,6% ainda estão ativos** no fim da janela de observação.
   - A mediana ingênua do tempo observado bruto é de apenas **`13,1 meses`**.
   - Ao tratar corretamente a censura via **Kaplan-Meier**, descobrimos que o **tempo mediano real de sobrevivência é de `26,9 meses`** — ignorar a censura subestimaria a vida útil do cliente em mais da metade!

---

## 📊 Dataset e Estrutura dos Dados

O arquivo [`data/assinaturas_survival.csv`](data/assinaturas_survival.csv) contém **3.000 assinantes** acompanhados em uma janela de até **36 meses** (`1.181` cancelamentos observados = `39,4%`; `1.819` clientes censurados/ativos = `60,6%`):

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `cliente_id` | `string` | Identificador único do assinante |
| `tempo_meses` | `float` | Tempo de permanência observado em meses (`0,5` a `36,0`) |
| `churn` | `int` | Indicador de evento (`1` = cancelou, `0` = censurado à direita / ainda ativo) |
| `tipo_contrato` | `string` | Modalidade contratual (`Mensal`, `Anual`, `Bienal`) |
| `plano` | `string` | Nível do plano contratado (`Básico`, `Intermediário`, `Premium`) |
| `forma_pagamento` | `string` | Meio de pagamento (`Boleto`, `Cartão de Crédito`, `Débito Automático`) |
| `mensalidade` | `float` | Valor pago mensalmente (R$) |
| `dias_uso_mes` | `int` | Frequência de engajamento (dias ativos/mês no app) |
| `chamados_suporte` | `int` | Quantidade de chamados abertos no suporte |

---

## 🔬 Metodologia Passo a Passo

1. **Visualização da Censura (*Swimmer Plot*)**: Linha do tempo individual de 20 clientes diferenciando cancelamentos observados de observações censuradas à direita.
2. **Estimador de Kaplan-Meier ($S(t)$) + Teste Log-Rank**: Curva global de sobrevivência (com marcos de 6, 12, 18, 24 e 36 meses) e comparação estatística estratificada por **Tipo de Contrato** e **Chamados de Suporte**.
3. **Estimador de Nelson-Aalen ($H(t)$)**: Curvas de risco acumulado evidenciando a aceleração do churn por contrato.
4. **Regressão de Riscos Proporcionais de Cox (`lifelines.CoxPHFitter`)**: Estimativa dos ***Hazard Ratios* ($\exp(\beta)$)** com intervalos de confiança de 95% e *Forest Plot*.
5. **XGBoost Survival (`objective="survival:cox"`) + Curvas Individuais para CRM**: Avaliação fora da amostra via **Concordance Index (C-Index)** (80% treino / 20% teste) e projeção de curvas individuais de sobrevivência $S(t \mid X)$ para 3 perfis de risco (Alto, Moderado e Baixo).

---

## 📈 Principais Resultados

### 1. Sobrevivência Global (Kaplan-Meier — Mediana = `26,9 meses`)

| Horizonte ($t$) | Probabilidade de Permanecer Ativo $S(t)$ | Probabilidade Acumulada de Churn $1 - S(t)$ |
| :---: | :---: | :---: |
| **6 meses** | `83,8%` | `16,2%` |
| **12 meses** | `71,5%` | `28,5%` |
| **18 meses** | `62,1%` | `37,9%` |
| **24 meses** | `53,2%` | `46,8%` |
| **36 meses** | `40,3%` | `59,7%` |

### 2. Fatores de Risco e Proteção (Regressão de Cox — *Hazard Ratios*)

| Variável | Hazard Ratio (HR) | Impacto no Risco Instantâneo | P-Valor | Interpretação Prática |
| :--- | :---: | :---: | :---: | :--- |
| **`pagto_boleto`** *(vs. Cartão)* | **`1,534`** | **`+53,4%`** | `< 0,001` | Forte fator de risco (atrito de renovação mensal) |
| **`chamados_suporte`** *(por chamado)* | **`1,431`** | **`+43,1%`** | `< 0,001` | Cada chamado extra eleva drasticamente o risco de churn |
| **`mensalidade`** *(por R$ 1,00)* | `1,005` | `+0,5%` | `< 0,001` | Sensibilidade moderada a preço |
| **`dias_uso_mes`** *(por dia ativo)* | **`0,935`** | **`-6,5%`** | `< 0,001` | Engajamento contínuo protege fortemente contra churn |
| **`contrato_anual`** *(vs. Mensal)* | **`0,437`** | **`-56,3%`** | `< 0,001` | Reduz o risco instantâneo em mais da metade |
| **`contrato_bienal`** *(vs. Mensal)* | **`0,225`** | **`-77,5%`** | `< 0,001` | Maior fator de retenção e LTV da base |

### 3. Comparação de Modelos (Concordance Index — C-Index)

| Modelo | C-Index (Treino) | C-Index (Teste) | Destaque |
| :--- | :---: | :---: | :--- |
| **Regressão de Cox (`CoxPHFitter`)** | `0,732` | **`0,754`** | Alta interpretabilidade direta via *Hazard Ratios* e IC 95% |
| **XGBoost Survival (`survival:cox`)** | `0,759` | **`0,749`** | Captura não-linearidades e interações complexas sem suposições paramétricas |

---

## 📁 Estrutura do Projeto

```text
analise-sobrevivencia-churn-clientes/
├── data/
│   └── assinaturas_survival.csv     # Base de 3.000 assinantes com censura à direita
├── analise_sobrevivencia.ipynb      # Notebook completo e didático (executado com 5 gráficos)
├── analise_sobrevivencia.html       # Versão renderizada em HTML para leitura rápida
├── requirements.txt                 # Dependências Python do projeto
└── README.md                        # Documentação executiva do projeto
```

---

## 🚀 Como Executar

1. **Acesse a pasta e instale as dependências:**
   ```bash
   cd "analise-sobrevivencia-churn-clientes"
   pip install -r requirements.txt
   ```
2. **Abra e execute o notebook:**
   ```bash
   jupyter notebook analise_sobrevivencia.ipynb
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

1. Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis: A Self-Learning Text* (3rd ed.). Springer.
2. Davidson-Pilon, C. (2019). *lifelines: survival analysis in Python*. Journal of Open Source Software, 4(40), 1317.
3. Cox, D. R. (1972). *Regression Models and Life-Tables*. Journal of the Royal Statistical Society: Series B.
