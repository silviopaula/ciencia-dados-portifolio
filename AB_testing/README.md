# 🎮 Cookie Cats A/B Testing: Impacto de Design de Produto na Retenção de Jogadores

<div align="center">

![Banner Cookie Cats](img/cover.png)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-2C5BB4?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

**Avaliação estatística, econométrica e comportamental do posicionamento de barreiras de progressão (*gates*) em jogos mobile**

</div>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Problema de Negócio e Hipóteses](#-problema-de-negócio-e-hipóteses)
- [Dataset e Desenho Experimental](#-dataset-e-desenho-experimental)
- [Metodologia Analítica](#-metodologia-analítica)
- [Principais Resultados](#-principais-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Autor](#-autor)
- [Referências](#-referências)

---

## 🎯 Sobre o Projeto

Este repositório apresenta uma análise completa de um experimento controlado (**Teste A/B**) realizado no jogo mobile **Cookie Cats** (Tactile Entertainment), um jogo do gênero *match-3* (estilo *Candy Crush*).

Ao avançar pelas fases do jogo, o usuário encontra **"gates"** (barreiras temporárias de progressão) que exigem aguardar um período de tempo ou realizar uma compra no aplicativo (*in-app purchase*) para continuar jogando. Além de viabilizarem a monetização, essas pausas cumprem uma função psicológica central (**adaptação hedônica**): evitam a fadiga precoce do jogador e estimulam o hábito de retorno nos dias seguintes.

O estudo avalia se o deslocamento do primeiro *gate* do **nível 30** (Controle) para o **nível 40** (Tratamento) melhora ou prejudica o engajamento e a retenção dos jogadores.

---

## 💼 Problema de Negócio e Hipóteses

### Pergunta Central
> **Mover o primeiro *gate* do nível 30 (`gate_30`) para o nível 40 (`gate_40`) impacta significativamente o volume de partidas jogadas e as taxas de retenção no 1º dia (D1) e no 7º dia (D7) após a instalação?**

### Hipóteses Estatísticas
- **$H_0$ (Hipótese Nula)**: Não há diferença estatisticamente significativa nas métricas de engajamento e retenção (D1, D7 e Super-Retenção) entre `gate_30` e `gate_40`.
- **$H_1$ (Hipótese Alternativa)**: O deslocamento do *gate* para o nível 40 altera significativamente as métricas de retenção e/ou engajamento ($\alpha = 0,05$, teste bicaudal).

---

## 📊 Dataset e Desenho Experimental

O conjunto de dados compreende **90.189 jogadores** que instalaram o jogo durante o período experimental, alocados aleatoriamente entre dois grupos:
- **Grupo Controle (`gate_30`)**: 44.700 jogadores (44.699 após remoção de 1 outlier extremo de telemetria com 49.854 partidas).
- **Grupo Tratamento (`gate_40`)**: 45.489 jogadores.

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `userid` | `int64` | Identificador único do jogador |
| `version` | `string` | Variante experimental (`gate_30` = Controle, `gate_40` = Tratamento) |
| `sum_gamerounds` | `int64` | Total de rodadas jogadas nos primeiros 7 dias após a instalação |
| `retention_1` | `bool` | Retornou e jogou 1 dia após a instalação? (`True` / `False`) |
| `retention_7` | `bool` | Retornou e jogou 7 dias após a instalação? (`True` / `False`) |

---

## 🔬 Metodologia Analítica

1. **Auditoria de Qualidade e SRM (*Sample Ratio Mismatch*)**:
   - Verificação de unicidade de `userid`, ausência de valores faltantes e teste Qui-Quadrado de aderência ($\chi^2$) para auditar o balanceamento 50/50 da aleatorização.
2. **Análise Exploratória e Tratamento de Outliers**:
   - Estudo dos quantis extremos (`P01` a `P99`) da distribuição de cauda longa de `sum_gamerounds` e remoção fundamentada de anomalia de registro (`49.854` partidas em 7 dias).
3. **Análise de Funil e Alcance do Tratamento (*Compliance*)**:
   - Mapeamento da curva de abandono por nível e distinção entre o efeito médio na base total instalada (*Intent-to-Treat* - ITT) e o efeito no subgrupo que efetivamente alcançou o nível 30 (`sum_gamerounds >= 30`, correspondente a **37,26%** da base).
4. **Segmentação Comportamental (4 Perfis de Retenção)**:
   - Classificação dos usuários em `Não Retido (0-0)`, `Só D1 (1-0)`, `Só D7 (0-1)` e `Super-Retido (1-1)` para quantificar o multiplicador de engajamento e *Lifetime Value* (LTV).
5. **Inferência Estatística e Reamostragem**:
   - **Volume de Jogo (`sum_gamerounds`)**: Teste não-paramétrico de **Mann-Whitney U** e **Teste $t$ de Welch**.
   - **Proporções de Retenção (`D1`, `D7` e `D1+D7`)**: **Teste Z para Duas Proporções**, **Qui-Quadrado de Independência ($\chi^2$)**, intervalos de confiança de 95% (Wald) e **Reamostragem Bootstrap com 10.000 iterações**.
   - **Magnitude do Efeito**: Cálculo do **$h$ de Cohen** aliado à interpretação de impacto relativo em experimentos digitais de larga escala.

---

## 📈 Principais Resultados

### 1. Síntese dos Testes de Hipótese

| Métrica Avaliada | Controle (`gate_30`) | Tratamento (`gate_40`) | Diferença Absoluta | Impacto Relativo | P-Valor | Significativo ($\alpha = 5\%$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Média de Rodadas (1ª Semana)** | 51,34 | 51,30 | -0,04 rodadas | -0,08% | 0,0509 (MWU) | Não (Equivalente) |
| **Retenção D1 (Amostra Global)** | 44,82% | 44,23% | -0,59 pp | -1,32% | 0,0739 | Marginal (96,3% Boot) |
| **Retenção D7 (Amostra Global)** | **19,02%** | **18,20%** | **-0,82 pp** | **-4,31%** | **0,0016** | **Sim (Gate 30 Superior)** |
| **Super-Retidos (`D1 + D7`)** | **14,94%** | **14,30%** | **-0,63 pp** | **-4,23%** | **0,0072** | **Sim (Gate 30 Superior)** |
| **Retenção D7 (Expostos $\ge 30$ rodadas)** | **43,85%** | **42,08%** | **-1,77 pp** | **-4,04%** | **0,0011** | **Sim (Gate 30 Superior)** |

### 2. Segmentação Comportamental Revelada

```text
Perfis de Retenção e Multiplicador de Engajamento:
├── 51,5% — Não Retidos (0-0)      : ~12 rodadas médias (1,0x base — abandonam rapidamente)
├── 29,9% — Retidos Só D1 (1-0)    : ~50 rodadas médias (4,2x — engajamento inicial sem persistência)
├──  4,0% — Retidos Só D7 (0-1)    : ~74 rodadas médias (6,2x — perfil "redescobridor", oportunidade de CRM)
└── 14,6% — Super-Retidos (1-1)    : ~187 rodadas médias (15,6x — núcleo de altíssimo LTV do jogo)
```

### 3. Recomendação Executiva
> **MANTER O GATE NO NÍVEL 30 (`gate_30`).**
>
> - **Superioridade Estatística e Empírica**: A retenção de 7 dias (D7) cai de **19,02%** para **18,20%** no `gate_40` ($p = 0,0016$, com **99,92% de probabilidade Bootstrap** a favor do `gate_30`). Entre os jogadores que efetivamente chegam ao nível 30 (`37,26%` da base), a perda em D7 atinge **-1,77 pontos percentuais** ($p = 0,0011$).
> - **Preservação de LTV**: O `gate_30` preserva significativamente a formação de **Super-Retidos** ($p = 0,0072$), que jogam **15,6 vezes mais partidas** que os usuários não retidos.
> - **Prioridade de Produto no Onboarding**: Como **55,2% dos usuários abandonam o jogo nas primeiras 20 rodadas** (antes de qualquer *gate*), iniciativas de produto voltadas para retenção D1 devem focar na redução de atrito nos primeiros 15 níveis.

---

## 📁 Estrutura do Projeto

```text
AB_testing/
├── data/
│   └── cookie_cats.csv       # Dataset original do experimento (90.189 jogadores)
├── img/
│   └── cover.png             # Imagem de capa do projeto
├── AB_testing.ipynb          # Jupyter Notebook principal com toda a análise e gráficos
├── AB_Testing.html           # Versão exportada em HTML para visualização rápida
├── AB_Testing.pdf            # Relatório estático em PDF
├── requirements.txt          # Dependências Python do projeto
└── README.md                 # Documentação executiva do projeto
```

---

## 🚀 Como Executar

1. **Clone o repositório e acesse a pasta do projeto:**
   ```bash
   git clone https://github.com/silviopaula/ciencia-dados-portifolio.git
   cd ciencia-dados-portifolio/AB_testing
   ```

2. **Crie um ambiente virtual (opcional) e instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Abra e execute o Jupyter Notebook:**
   ```bash
   jupyter notebook AB_testing.ipynb
   ```

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Manipulação e Cálculo Numérico**: `pandas`, `numpy`
- **Inferência Estatística e Econometria**: `scipy.stats`, `statsmodels`
- **Visualização de Dados**: `matplotlib`, `seaborn`
- **Ambiente Interativo**: `jupyter`, `nbconvert`

---

## 👤 Autor

**Silvio da Rosa Paula**  
*Doutor em Economia Aplicada (UFPel) | Especialista em Inteligência Artificial e Ciência de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/silvio-paula/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/silviopaula)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:silvio.rosa.paula@gmail.com)

---

## 📚 Referências

1. Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
2. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
3. Tactile Entertainment / DataCamp — *Mobile Games A/B Testing with Cookie Cats Dataset*.
