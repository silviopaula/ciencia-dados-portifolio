# Avaliação Causal de Campanha de Marketing Digital

[![R Version](https://img.shields.io/badge/R-%E2%89%A5%204.0-blue)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-complete-success)](.)

## Descrição

Este projeto demonstra **como estimar o efeito causal** de uma campanha de marketing digital sobre vendas de lojas físicas quando **a alocação não foi aleatória** (viés de seleção). 

Simulamos um cenário realista de rede varejista onde lojas com maior penetração digital e renda receberam a campanha, criando viés de seleção que invalida testes A/B simples.

**Desafio:** Lojas tratadas já vendiam 38% mais ANTES da campanha. Como isolar o efeito causal?

---

## O Que Você Vai Aprender

-  Por que **testes A/B ingênuos falham** com seleção não-aleatória
-  Como **balanceamento de covariáveis** elimina 98% do viés
-  Quando usar **DiD, Synthetic Control, ou Machine Learning**
-  Como validar suposições e escolher o método correto
-  Comparação prática de **15+ métodos de inferência causal**

---

##  Estrutura dos Dados

### Painel Balanceado
```
250 lojas × 24 meses = 6.000 observações
├── 155 lojas tratadas (campanha digital no mês 13)
└── 95 lojas controle (sem campanha)
```

### Períodos
```
Meses 1-12:  Pré-intervenção (baseline)
Mês 13:      Início da campanha
Meses 13-24: Pós-intervenção (avaliação)
```

### Covariáveis

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `vendas` | Contínua | Vendas mensais (R$) |
| `tratamento` | Binária | 1 = recebeu campanha, 0 = controle |
| `renda_per_capita` | Contínua | Renda média da região |
| `penetracao_internet` | Contínua | % população com internet (0-1) |
| `n_concorrentes` | Inteira | Número de concorrentes próximos |
| `potencial_crescimento` | Contínua | Score de potencial (0-1) |

### Viés de Seleção (Proposital)
```r
# Lojas com MAIOR potencial receberam campanha
# Isso cria correlação entre tratamento e outcome potencial

Probabilidade(tratamento) = f(renda, internet, potencial)
                             ↓
                   Viés de seleção observável
```

---

##  Quick Start

### Instalação de Pacotes
```r
# Instalar todos os pacotes necessários
install.packages("pacman")

pacman::p_load(
  # Core
  tidyverse, fixest, plm,
  
  # DiD Avançado
  did, DIDmultiplegtDYN,
  
  # Synthetic Control
  augsynth, Synth,
  
  # Balanceamento
  MatchIt, WeightIt, cobalt,
  
  # Machine Learning
  grf,
  
  # Séries Temporais
  CausalImpact
)
```

### Executar Análise
```r
# 1. Gerar dados simulados
source("01_simulacao_dados.R")

# 2. Análise cross-section
source("02_ab_test.R")
source("03_entropy_balancing.R")

# 3. Métodos de painel
source("04_did_twfe.R")
source("05_sun_abraham.R")
source("06_callaway_santanna.R")

# 4. Synthetic Control
source("07_synthetic_control.R")

# 5. Machine Learning
source("08_causal_forest.R")

# 6. Comparação final
source("09_comparacao_completa.R")
```

---

##  Resultados Principais

### Efeito Verdadeiro vs Estimativas

**Ground truth (simulação):** R$ 6.012/mês por loja

### Ranking de Métodos

| 🏆 | Método | Efeito | Viés | Erro % | Performance |
|----|--------|--------|------|--------|-------------|
| 🥇 | **A/B Entropy** | R$ 5.982 | -R$ 30 | **-0.5%** | ⭐⭐⭐⭐⭐ |
| 🥈 | **Synthetic Control** | R$ 5.953 | -R$ 59 | **-1.0%** | ⭐⭐⭐⭐⭐ |
| 🥉 | **Causal Forest** | R$ 6.086 | +R$ 74 | **+1.2%** | ⭐⭐⭐⭐⭐ |
| 4º | TWFE + Entropy | R$ 5.889 | -R$ 123 | -2.1% | ⭐⭐⭐⭐⭐ |
| 5º | Synthetic DiD | R$ 6.280 | +R$ 268 | +4.4% | ⭐⭐⭐⭐ |
| 6º | S&A + Pesos | R$ 6.352 | +R$ 340 | +5.6% | ⭐⭐⭐⭐ |
| 7º | C&S Doubly Robust | R$ 6.394 | +R$ 382 | +6.4% | ⭐⭐⭐⭐ |
| 8º | de Chaisemartin | R$ 6.659 | +R$ 647 | +10.8% | ⭐⭐⭐ |
| 9º | DiD/TWFE Simples | R$ 6.737 | +R$ 725 | +12.1% | ⭐⭐ |
| 10º | CausalImpact Bivariado | R$ 7.120 | +R$ 1.108 | +18.4% | ⭐⭐ |
| ❌ | **A/B Simples** | R$ 8.310 | +R$ 2.298 | **+38.2%** | ⭐ |
| ❌ | CausalImpact Univariado | R$ 12.271 | +R$ 6.259 | +104% | ❌ |

---

##  Métodos Implementados

### Por Categoria

####  Cross-Sectional

| Método | Pacote | Viés | Quando Usar |
|--------|--------|------|-------------|
| A/B Simples | `t.test()` | +38% | ❌ Nunca com seleção não-aleatória |
| A/B + Entropy | `WeightIt` | **-0.5%** | ✅ Dados cross-section |
| A/B + Matching | `MatchIt` | -1.8% | ✅ Alternativa ao Entropy |

####  Diferença-em-Diferenças

| Método | Pacote | Viés | Quando Usar |
|--------|--------|------|-------------|
| DiD Canônico | Base R | +12% | ❌ Sem balanceamento |
| TWFE | `fixest` | +12% | ❌ Sem balanceamento |
| TWFE + Entropy | `fixest` + `WeightIt` | **-2.1%** | ✅ Painel balanceado |
| Sun & Abraham | `fixest::sunab` | +11% | ✅ Interaction-weighted |
| Callaway & Sant'Anna | `did` | +6.4% | ✅ Doubly robust |
| de Chaisemartin | `DIDmultiplegtDYN` | +11% | ✅ Event-study completo |

####  Synthetic Control

| Método | Pacote | Viés | Quando Usar |
|--------|--------|------|-------------|
| SC Tradicional | `Synth` | **-1.0%** | ✅ 1 tratado, muitos doadores |
| Augmented SC | `augsynth` | **-1.0%** | ✅ SC + Ridge correction |
| Synthetic DiD | `augsynth` | +4.4% | ✅ SC + DiD robusto |

#### Machine Learning

| Método | Pacote | Viés | Quando Usar |
|--------|--------|------|-------------|
| Causal Forest | `grf` | **+1.2%** | ✅ Heterogeneidade, não-linear |


#### Séries Temporais

| Método | Pacote | Viés | Quando Usar |
|--------|--------|------|-------------|
| CausalImpact Univariado | `CausalImpact` | +104% | ❌ Sem covariáveis |
| CausalImpact Bivariado | `CausalImpact` | +18% | ⚠️ Com boas covariáveis temporais |

---

##  Insights e Lições

### 1. Balanceamento é Crítico

**Lição:** Balancear covariáveis elimina viés observável, independente do método.

### 2. Métodos Convergem Quando Bem Especificados
```
Top 5 métodos: 5.889 - 6.086 (amplitude de R$ 197)
Consenso: ~R$ 6.000/mês
```

**Lição:** Métodos diferentes com suposições corretas chegam ao mesmo resultado.

### 3. Hierarquia de Robustez
```
Tier S (Erro 0-2%):   Balanceamento + SC + ML
Tier A (Erro 2-5%):   Synthetic DiD, S&A com pesos
Tier B (Erro 5-10%):  C&S robusto, C&S IPW
Tier C (Erro 10-15%): DiD/TWFE sem balanceamento
Tier D (Erro >15%):   CausalImpact sem boas covariáveis
Tier F (Erro >30%):   A/B ingênuo
```

### 4. Trade-offs

| Aspecto | A/B Entropy | Synthetic Control | Causal Forest |
|---------|-------------|-------------------|---------------|
| **Precisão** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Simplicidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Interpretabilidade** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Usa estrutura temporal** | ❌ | ✅ | ❌ |
| **Identifica heterogeneidade** | ❌ | ❌ | ✅ |
| **Requer painel** | ❌ | ✅ | ❌ |

---

##  Recomendações Práticas

### Fluxograma de Decisão
```
Tem dados em painel (longitudinais)?
│
├─ NÃO → Use A/B Entropy ou Causal Forest
│         (cross-section balanceado)
│
└─ SIM → Quantos tratados?
          │
          ├─ 1 (ou agregável) → Synthetic Control
          │
          └─ Múltiplos → TWFE + Entropy
                         ou C&S Doubly Robust
```

### Por Objetivo

**1. Decisão rápida de negócio:**
- ✅ **A/B Entropy** (simples, rápido, preciso)

**2. Análise robusta com painel:**
- ✅ **TWFE + Entropy** (usa estrutura temporal)

**3. Identificar quais lojas se beneficiam mais:**
- ✅ **Causal Forest** (efeitos heterogêneos)

**4. Validação cruzada:**
- ✅ Rode **3-4 métodos** e compare
- Se convergem → confiança alta
- Se divergem → investigar suposições

---

## Estrutura do Projeto
```
.
├── 01_simulacao_dados.R          # Gera dados sintéticos
├── 02_ab_test.R                  # A/B simples (baseline)
├── 03_entropy_balancing.R        # Balanceamento de covariáveis
├── 04_did_twfe.R                 # DiD e TWFE
├── 05_sun_abraham.R              # Sun & Abraham
├── 06_callaway_santanna.R        # Callaway & Sant'Anna
├── 07_dechaisemartin.R           # de Chaisemartin & D'Haultfoeuille
├── 08_synthetic_control.R        # SC Tradicional + Augmented + SDID
├── 09_causal_forest.R            # Machine Learning causal
├── 10_causal_impact.R            # Bayesian time series
├── 11_comparacao_final.R         # Comparação e visualizações
├── README.md                     # Este arquivo
└── resultados/
    ├── tabelas/                  # Tabelas de resultados
    ├── graficos/                 # Visualizações
    └── relatorio.html            # Relatório completo
```

---

##  Referências

### Artigos Seminais

1. **Abadie, A., Diamond, A., & Hainmueller, J. (2010).** "Synthetic Control Methods for Comparative Case Studies." *JASA*
2. **Callaway, B., & Sant'Anna, P. H. (2021).** "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*
3. **Sun, L., & Abraham, S. (2021).** "Estimating Dynamic Treatment Effects in Event Studies." *JoE*
4. **Arkhangelsky, D., et al. (2021).** "Synthetic Difference-in-Differences." *AER*
5. **Wager, S., & Athey, S. (2018).** "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *JASA*
6. **Hainmueller, J. (2012).** "Entropy Balancing for Causal Effects." *Political Analysis*

### Tutoriais e Documentação

- [Causal Inference: The Mixtape](https://mixtape.scunning.com/)
- [The Effect Book](https://theeffectbook.net/)
- [Causal Inference in R](https://www.r-causal.org/)

---

##  Limitações e Extensões

### Limitações

- Simulação com efeito homogêneo base (extensível para heterogeneidade complexa)
- Não inclui spillovers ou interferência entre unidades
- Seleção apenas em observáveis (sem instrumentos)

### Extensões Possíveis

- [ ] Regression Discontinuity Design (RDD)
- [ ] Instrumental Variables (IV)
- [ ] Bounds (Manski, etc.)
- [ ] Sensitivity analysis (Rosenbaum bounds)
- [ ] Staggered adoption (timing variado)


---

##  Licença

MIT License - Sinta-se livre para usar em projetos acadêmicos ou comerciais.

---

## c Autor

**Seu Nome**
- GitHub: [@seu-usuario](https://github.com/silviopaula)
- LinkedIn: [Seu Perfil](https://www.linkedin.com/in/silvio-paula/)

---

## Agradecimentos

Agradecimentos aos autores dos pacotes utilizados e à comunidade R de inferência causal.
