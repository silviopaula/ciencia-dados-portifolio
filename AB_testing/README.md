# 🎮 Cookie Cats A/B Testing Analysis

<div align="center">
  
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-v1.3.0+-green.svg)
![Scipy](https://img.shields.io/badge/scipy-v1.7.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-red.svg)
![Status](https://img.shields.io/badge/status-complete-success.svg)

**Análise estatística rigorosa do impacto da posição de gates na retenção de jogadores**


</div>

---

## Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Problema de Negócio](#-problema-de-negócio)
- [Dataset](#-dataset)
- [Metodologia](#-metodologia)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação e Uso](#-instalação-e-uso)
- [Principais Resultados](#-principais-resultados)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Contribuindo](#-contribuindo)
- [Autor](#-autor)
- [Licença](#-licença)

---

## Sobre o Projeto

Este repositório contém uma análise completa de um teste A/B realizado no jogo mobile **Cookie Cats**, um popular jogo puzzle "match-three" estilo Candy Crush. O estudo avalia o impacto da mudança na posição do primeiro "gate" (barreira de progressão) do nível 30 para o nível 40 nas métricas de retenção dos jogadores.

### O que são Gates?
Gates são pontos de espera forçados no jogo onde os jogadores devem aguardar um período ou fazer uma compra para continuar jogando. São elementos críticos para monetização e engajamento em jogos mobile.

---

## Problema de Negócio

### Pergunta Principal
> **Mover o primeiro gate do nível 30 para o nível 40 afeta negativamente a retenção dos jogadores no 1º e 7º dia após a instalação?**

### Hipóteses
- **H₀**: Não há diferença significativa na retenção entre gate_30 e gate_40
- **H₁**: Existe diferença significativa na retenção entre as configurações

### Métricas Avaliadas
- **Retenção D1**: Jogadores que retornam 1 dia após instalação
- **Retenção D7**: Jogadores que retornam 7 dias após instalação  
- **Engajamento**: Número total de rodadas jogadas
- **Super-Retidos**: Jogadores que retornam em D1 E D7

---

## Dataset

### Características
- **Tamanho**: 90.189 jogadores
- **Período**: Dados de experimento A/B controlado
- **Grupos**: 
  - Control (gate_30): 44.700 usuários
  - Teste (gate_40): 45.489 usuários

### Variáveis

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `userid` | int | Identificador único do jogador |
| `version` | string | Grupo do experimento (gate_30 ou gate_40) |
| `sum_gamerounds` | int | Total de rodadas jogadas na primeira semana |
| `retention_1` | bool | Retornou no dia 1? (True/False) |
| `retention_7` | bool | Retornou no dia 7? (True/False) |

### Fonte
Dataset inspirado no [Kaggle](https://www.kaggle.com/code/ekrembayar/a-b-testing-step-by-step-hypothesis-testing)

---

## Metodologia

### 1. Análise Exploratória (EDA)
- Verificação de qualidade dos dados
- Identificação e tratamento de outliers
- Estatísticas descritivas por grupo
- Visualizações comparativas

### 2. Testes Estatísticos

### **Fluxograma de Decisão:**

```
Dados dos Grupos A e B
        ↓
Teste de Shapiro (Normalidade)
        ↓
   Normal?
    ↙     ↘
  SIM      NÃO
   ↓        ↓
Teste de   Mann
Levene     Whitney U
   ↓
Homogêneo?
 ↙     ↘
SIM    NÃO
 ↓      ↓
Teste T  Teste de
         Welch
```

### 3. Análise de Significância
- **Nível de significância**: α = 0.05
- **Testes aplicados**:
  - Mann-Whitney U (variáveis contínuas não-paramétricas)
  - Teste Z para proporções (retenção)
  - Chi-quadrado (validação)

### 4. Tamanho do Efeito
- **Cohen's h** para diferenças entre proporções
- Interpretação de magnitude além da significância estatística


---

##  Principais Resultados

### Descobertas Estatísticas

| Métrica | Gate 30 | Gate 40 | P-valor | Significativo? |
|---------|---------|---------|---------|----------------|
| **Rodadas Jogadas** | 51.34 | 51.30 | 0.051 |  Não |
| **Retenção D1** | 44.82% | 44.23% | 0.074 |  Não |
| **Retenção D7** | 19.02% | 18.20% | **0.002** |  **Sim** |
| **Super-Retidos** | 14.94% | 14.30% | - |  -170 usuários |

### Segmentação de Usuários Descoberta

```python
Padrões de Comportamento:
├── 51% - Não Retidos (abandonam rapidamente)
├── 30% - Retidos D1 apenas (engajamento inicial)
├── 4%  - Retidos D7 apenas (padrão "redescoberta")
└── 15% - Super-Retidos D1+D7 (jogam 15.6x mais!)
```

###  Recomendação Final

** MANTER GATE NO NÍVEL 30**

**Justificativa**:
- Retenção D7 significativamente superior (p = 0.002)
- 170 super-retidos adicionais por ciclo
- Sem impacto negativo no engajamento
- Efeito cumulativo prejudicial do Gate 40

---

##  Tecnologias Utilizadas

### Linguagens e Frameworks
- **Python 3.8+** - Linguagem principal
- **Jupyter Notebook** - Desenvolvimento interativo

### Bibliotecas de Dados
- **Pandas 1.3.0** - Manipulação de dados
- **NumPy 1.21.0** - Computação numérica

### Análise Estatística
- **SciPy 1.7.0** - Testes estatísticos
- **Statsmodels 0.12.0** - Modelos estatísticos

### Visualização
- **Matplotlib 3.4.0** - Gráficos base
- **Seaborn 0.11.0** - Visualizações estatísticas
- **Plotly 5.3.0** - Gráficos interativos

---

##  Referências

1. Kohavi, R., Tang, D., & Xu, Y. (2020). **Trustworthy Online Controlled Experiments**
2. Ellis, P. D. (2010). **The Essential Guide to Effect Sizes**
3. [Mobile Games A/B Testing Best Practices](https://docs.gameanalytics.com)

---
