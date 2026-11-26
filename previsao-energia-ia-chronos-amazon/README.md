# ⚡ Previsão de Séries Temporais *Zero-Shot* com Amazon Chronos (T5 & Bolt) vs. Modelos Clássicos

<div align="center">

<img src="https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/figures/chronos-logo.png" alt="Amazon Chronos" width="240">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Chronos](https://img.shields.io/badge/Amazon_Chronos-Bolt%20%7C%20T5-FF9900?style=for-the-badge&logo=amazon&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%2FGPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App_Interativo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

**Benchmark *Zero-Shot* dos *Foundation Models* Amazon Chronos (Bolt & T5) contra baselines estatísticos clássicos (Seasonal Naive e Holt-Winters ETS) sobre 21 anos de Consumo de Energia Elétrica no Brasil (EPE) + App Web em Streamlit**

</div>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Como o Amazon Chronos Funciona](#-como-o-amazon-chronos-funciona)
- [Dataset Oficial (EPE — 2004 a 2025)](#-dataset-oficial-epe--2004-a-2025)
- [Resultados do Benchmark (Backtest de 12 Meses)](#-resultados-do-benchmark-backtest-de-12-meses)
- [Projeção Futura de 36 Meses (2025–2028)](#-projeção-futura-de-36-meses-20252028)
- [Aplicação Interativa (Streamlit)](#-aplicação-interativa-streamlit)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Autor](#-autor)

---

## 🎯 Sobre o Projeto

Este projeto avalia na prática o poder dos **Modelos Fundacionais Pré-Treinados (*Foundation Models*)** para previsão de séries temporais — especificamente as famílias **Amazon Chronos-Bolt** e **Chronos-T5** — aplicados ao **Consumo Mensal de Energia Elétrica do Brasil (dados oficiais da EPE)**.

Além de comparar **6 variantes *Zero-Shot* do Chronos + Ensemble Bolt** contra dois baselines clássicos (**Seasonal Naive** e **Holt-Winters ETS**) em acurácia, cobertura probabilística (**P10–P90**) e latência em CPU, o projeto entrega uma **aplicação web interativa em Streamlit** ([`app/App_Chronos.py`](app/App_Chronos.py)) para que usuários técnicos e de negócio façam previsões em qualquer série temporal própria.

---

## 🧠 Como o Amazon Chronos Funciona

O Chronos trata a previsão de séries temporais como uma tarefa de linguagem natural (*"autocompletar de números"*):
1. **Escalonamento e Tokenização**: Normaliza a série histórica e quantiza os valores em *tokens* discretos.
2. **Pré-Treinamento em Milhões de Séries**: Aprende padrões universais de tendência, sazonalidade e choques sem precisar treinar do zero para cada nova série (**Zero-Shot**).
3. **Previsão Probabilística (P10 / P50 / P90)**: Entrega tanto a estimativa central (**P50**) quanto a faixa de incerteza (**P10–P90**).
4. **Por que `Chronos-Bolt` se destaca?** A 2ª geração (*Bolt*) utiliza *patching* direto de múltiplos quantis, alcançando acurácia superior à família T5 em **menos de 0,4 segundo em CPU**.

---

## 📊 Dataset Oficial (EPE — 2004 a 2025)

- **Arquivo**: [`dados/consumo_energia_pronto.xlsx`](dados/consumo_energia_pronto.xlsx) (Empresa de Pesquisa Energética — EPE).
- **Período**: **Janeiro/2004 a Julho/2025** (**259 meses**, 63 séries por região e classe de consumo).
- **Série Alvo Principal**: `ec_total_BR` (convertida para **GWh**), que cresceu **`+67,5%`** no período (de `26.978 GWh` em Jan/2004 para `45.177 GWh` em Jul/2025), exibindo forte sazonalidade de verão e o choque da pandemia em 2020.

---

## 📈 Resultados do Benchmark (Backtest de 12 Meses)

- **Treino / Contexto**: `247 meses` (Jan/2004 – Jul/2024) | **Backtest**: `12 meses` (Ago/2024 – Jul/2025), rodando em **CPU**.

| Posição | Modelo | Família | MAPE (%) | sMAPE (%) | MAE (GWh) | RMSE (GWh) | $R^2$ | Cobertura P10–P90 | Tempo CPU |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`chronos-bolt-small`** | **Chronos-Bolt** | **`1,02%`** | **`1,02%`** | **`481,1`** | `639,6` | `0,69` | `100,0%` | **`0,37 s`** |
| 🥈 | **`Holt-Winters (ETS)`** | Baseline Clássico | `1,10%` | `1,09%` | `511,6` | **`595,5`** | **`0,73`** | `100,0%` | `0,07 s` |
| 🥉 | **`Ensemble Chronos-Bolt`** | Ensemble Bolt | `1,18%` | `1,19%` | `559,9` | `724,5` | `0,60` | `100,0%` | `2,38 s` |
| 4º | `chronos-bolt-mini` | Chronos-Bolt | `1,21%` | `1,22%` | `575,5` | `767,1` | `0,55` | `100,0%` | `0,27 s` |
| 5º | `chronos-bolt-base` | Chronos-Bolt | `1,23%` | `1,23%` | `578,7` | `709,6` | `0,61` | `100,0%` | `1,46 s` |
| 6º | `chronos-bolt-tiny` | Chronos-Bolt | `1,29%` | `1,30%` | `607,7` | `785,8` | `0,53` | `100,0%` | `0,28 s` |
| 7º | `chronos-t5-tiny` | Chronos-T5 | `2,04%` | `2,02%` | `948,5` | `1.155,2` | `-0,02` | `91,7%` | `0,52 s` |
| 8º | `Seasonal Naive (t-12)` | Baseline Clássico | `2,16%` | `2,20%` | `1.016,3` | `1.292,3` | `-0,28` | — | `0,00 s` |
| 9º | `chronos-t5-small` | Chronos-T5 | `2,36%` | `2,33%` | `1.100,0` | `1.331,5` | `-0,36` | `100,0%` | `0,89 s` |

### Principais Insights
- **`chronos-bolt-small` venceu em MAPE (`1,02%`) e MAE (`481,1 GWh`) em modo 100% Zero-Shot**, superando o **Holt-Winters otimizado** sem nenhum ajuste manual de sazonalidade ou tendência, em apenas **`0,37 s`** em CPU.
- **Salto geracional (`Bolt` vs. `T5`)**: Todas as 4 variantes da família **Bolt** alcançaram MAPE entre **`1,02%` e `1,29%`**, enquanto a geração anterior (`T5`) ficou acima de `2,0%`.

---

## 🔮 Projeção Futura de 36 Meses (2025–2028)

Usando **100% dos 259 meses reais** com o campeão **`chronos-bolt-small`** (exportado em [`outputs/projecao_energia_brasil_36m.csv`](outputs/projecao_energia_brasil_36m.csv)):

| Ano | Cenário Conservador — P10 (GWh/mês) | Cenário Base — P50 (GWh/mês) | Cenário Alta Demanda — P90 (GWh/mês) |
| :---: | :---: | :---: | :---: |
| **2025** *(Ago–Dez)* | `45.589,1` | `47.249,8` | `48.969,3` |
| **2026** | `44.852,2` | `47.702,8` | `50.600,7` |
| **2027** | `44.204,4` | `48.151,7` | `52.212,7` |
| **2028** *(Jan–Jul)* | `43.646,0` | `48.563,4` | `53.844,1` |

---

## 🖥️ Aplicação Interativa (Streamlit)

O arquivo [`app/App_Chronos.py`](app/App_Chronos.py) oferece uma interface gráfica completa compatível com `chronos-forecasting 1.x` e `2.x`:
- Upload de arquivos CSV/Excel próprios ou uso da base demonstrativa da EPE.
- Seleção de múltiplos modelos Chronos (online via Hugging Face ou offline local) e opção de **Ensemble**.
- Gráficos interativos em Plotly e exportação das previsões.

---

## 📁 Estrutura do Projeto

```text
previsao-energia-ia-chronos-amazon/
├── app/
│   ├── App_Chronos.py                        # Aplicação Web interativa em Streamlit
│   └── requirements.txt                      # Dependências dedicadas para deploy do app
├── dados/
│   └── consumo_energia_pronto.xlsx           # Base oficial EPE tratada (259 meses x 63 séries)
├── outputs/
│   └── projecao_energia_brasil_36m.csv       # Projeção mensal de 36 meses (P10, P50 e P90)
├── Time_Series_Chronos_Forecast.ipynb        # Notebook completo e didático (executado)
├── Time_Series_Chronos_Forecast.html         # Versão renderizada em HTML para leitura rápida
├── requirements.txt                          # Dependências Python limpas
└── README.md                                 # Documentação executiva do projeto
```

*(Observação: a pasta local de pesos `Models_cronos_amazon/` fica ignorada no `.gitignore`; caso não exista localmente, o notebook baixa os pesos automaticamente do Hugging Face Hub).*

---

## 🚀 Como Executar

1. **Acesse a pasta e instale as dependências:**
   ```bash
   cd previsao-energia-ia-chronos-amazon
   pip install -r requirements.txt
   ```
2. **Para executar o notebook:**
   ```bash
   jupyter notebook Time_Series_Chronos_Forecast.ipynb
   ```
3. **Para subir a aplicação interativa em Streamlit:**
   ```bash
   streamlit run app/App_Chronos.py
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

1. Ansari, A. F., Stella, L., Turkmen, C., et al. (2024). *Chronos: Learning the Language of Time Series*. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).
2. Amazon Science — [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting).
3. Empresa de Pesquisa Energética (EPE) — *Dados Abertos de Consumo Mensal de Energia Elétrica*.
