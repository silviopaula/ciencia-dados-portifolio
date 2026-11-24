# 📉 Poder Estatístico (*Power*) e *Minimum Detectable Effect* (MDE) em Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-2C5BB4?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

**Guia prático e didático em Python sobre como calcular e interpretar Power e MDE em avaliações de impacto causal (*Difference-in-Differences*)**

</div>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Por que Power e MDE Importam?](#-por-que-power-e-mde-importam)
- [A Regra Prática do MDE (2,8 x SE)](#-a-regra-prática-do-mde-28-x-se)
- [Roteiro Passo a Passo do Notebook](#-roteiro-passo-a-passo-do-notebook)
- [Principais Resultados](#-principais-resultados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Autor](#-autor)

---

## 🎯 Sobre o Projeto

Em avaliações de impacto causal (como **Difference-in-Differences**, **Testes A/B** ou modelos de *uplift*), é muito comum encontrar um resultado **"não significativo"** ($p\text{-valor} > 0,05$). Na prática, muita gente interpreta isso automaticamente como *"não houve efeito"*, mas essa conclusão pode ser um erro grave.

Um resultado não significativo pode significar simplesmente que **o estudo não tinha sensibilidade suficiente** para enxergar o impacto que realmente existia. Este notebook em **Python** ensina, passo a passo e de forma didática, como calcular e interpretar o **Poder Estatístico (*Power*)** e o **Efeito Mínimo Detectável (*Minimum Detectable Effect* — MDE)** em um painel de Diferenças-em-Diferenças (*Two-Way Fixed Effects* — TWFE).

---

## 💡 Por que Power e MDE Importam?

Pense em um **detector de metais na praia**:
- **Power (Poder Estatístico)**: É a chance de o detector apitar quando realmente existe um objeto enterrado ali.
- **MDE (*Minimum Detectable Effect*)**: É o **menor tamanho de objeto** que o detector consegue perceber com segurança (por exemplo, com 80% de chance).

Se o **MDE** do seu estudo for maior que o efeito esperado da campanha, o estudo fica **"míope"**: o impacto pode estar acontecendo e gerando retorno para a empresa, mas a amostra não tem resolução suficiente para enxergá-lo.

> **Mensagem Central**:  
> *Ausência de significância estatística não significa ausência de impacto. Muitas vezes, significa apenas ausência de poder estatístico.*

---

## 📐 A Regra Prática do MDE (2,8 x SE)

Tudo depende do **Erro-Padrão (*Standard Error* — $SE$)** do coeficiente estimado no seu modelo, que mede o nível de ruído da estimativa:

$$\text{MDE} = \left( z_{1-\alpha/2} + z_{1-\beta} \right) \times SE$$

Para o padrão internacional de **95% de confiança** ($\alpha = 0,05 \implies z = 1,96$) e **80% de poder estatístico** ($1 - \beta = 0,80 \implies z = 0,84$), a soma dos quantis é $1,96 + 0,84 = \mathbf{2,80}$:

$$\boxed{\text{MDE}_{80\%} \approx 2,80 \times SE}$$

---

## 🔬 Roteiro Passo a Passo do Notebook

O notebook [`Power_MDE.ipynb`](Power_MDE.ipynb) foi estruturado de forma totalmente didática, explicando **o que** está sendo feito e **por que** antes de cada bloco de código:

1. **Intuição e Conceitos Fundamentais**: Explicação clara de *Power*, *MDE* e *Standard Error* ($SE$) sem excesso de jargão.
2. **Passo 1 — Simulação de Dados em Painel**: Criação de um painel balanceado de $N = 200$ lojas acompanhadas por $T = 10$ meses ($2.000$ observações), onde metade recebe uma campanha a partir do mês 6.
3. **Passo 2 — Estimação do Modelo DiD (TWFE)**: Regressão com efeitos fixos de loja e tempo (`statsmodels`) e erros-padrão clusterizados por loja para extrair o Erro-Padrão realista ($SE = 0,0921$).
4. **Passo 3 — Funções em Python (`calcular_power` e `calcular_mde`)**: Implementação limpa e comentada usando `scipy.stats.norm`.
5. **Passo 4 — Cálculo do MDE do Estudo**: Descoberta de que o menor efeito visível com 80% de segurança neste painel é **`MDE = 0,2580`** (aprox. `0,26`).
6. **Passo 5 — Curva de Poder Estatístico (*Power Curve*)**: Tabela e gráfico mostrando como a chance de detectar o impacto cresce conforme o tamanho do efeito aumenta.
7. **Passo 6 — Demonstração Prática da "Armadilha do P-Valor"**: Comparação entre uma campanha com efeito forte (`+2,00`, acima do MDE) e uma campanha com efeito real modesto (`+0,15`, abaixo do MDE), mostrando na prática como um efeito verdadeiro positivo gera um p-valor não significativo ($p = 0,2871$) quando o estudo tem pouco poder.

---

## 📈 Principais Resultados

| Cenário Simulado no Painel ($N=200, T=10$) | Efeito Real nos Dados | MDE (80%) do Estudo | Power do Estudo | Estimativa do Modelo | P-Valor | Conclusão pelo P-Valor ($< 0,05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Campanha com Efeito Grande (Acima do MDE)** | `+2,00` | `0,2580` | `100,0%` | `+1,9466` | `0,0000` | **Significativo** (Detectou o efeito!) |
| **2. Campanha com Efeito Modesto (Abaixo do MDE)** | **`+0,15`** | **`0,2580`** | **`36,6%`** | **`+0,0966`** | **`0,2871`** | **NÃO Significativo** (Falso Negativo!) |

---

## 📁 Estrutura do Projeto

```text
Power_MDE/
├── Power_MDE.ipynb       # Jupyter Notebook didático em Python
├── Power_MDE.html        # Versão exportada em HTML para leitura rápida
├── requirements.txt      # Dependências Python do projeto
└── README.md             # Documentação do projeto
```

---

## 🚀 Como Executar

1. **Clone o repositório e acesse a pasta do projeto:**
   ```bash
   git clone https://github.com/silviopaula/ciencia-dados-portifolio.git
   cd ciencia-dados-portifolio/Power_MDE
   ```

2. **Instale as dependências em Python:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Abra e execute o Jupyter Notebook:**
   ```bash
   jupyter notebook Power_MDE.ipynb
   ```

---

## 👤 Autor

**Silvio da Rosa Paula**  
*Doutor em Economia Aplicada (UFPel) | Especialista em Inteligência Artificial e Ciência de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/silvio-paula/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/silviopaula)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:silvio.rosa.paula@gmail.com)
