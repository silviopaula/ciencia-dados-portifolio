# Análise de Sentimento das Atas do COPOM 🇧🇷📉

Este projeto aplica técnicas de **Processamento de Linguagem Natural (NLP)** e análise de dados macroeconômicos para extrair, quantificar e interpretar o **sentimento** presente nas atas das reuniões do **Comitê de Política Monetária (COPOM)** do Banco Central do Brasil, relacionando os resultados com indicadores como a **Taxa Selic** e o **IPCA**.

---

## 🎯 Objetivos

- Avaliar o tom das atas do COPOM ao longo dos últimos 12 anos
- Relacionar o sentimento textual com variáveis macroeconômicas
- Detectar padrões sazonais, estruturais e institucionais na comunicação do BCB
- Explorar o uso de NLP na análise de política monetária

---

## 🛠️ Metodologia

- **Coleta automatizada** das atas via API do Banco Central
- **Tokenização e análise de polaridade** com o dicionário financeiro *Loughran-McDonald*
- Integração com dados da **Taxa Selic (SGS 432)** e do **IPCA (IBGE)**
- Criação de gráficos interativos e análises com defasagens, séries temporais e correlações

---

## 📊 Principais Resultados

- O sentimento das atas apresenta **correlação negativa com a Selic**, com maior impacto defasado (~5-7 meses)
- A partir de 2016, o tom das atas torna-se consistentemente mais **pessimista**
- A **autocorrelação** entre reuniões é alta (ρ ≈ 0.81), indicando estabilidade na comunicação
- A relação com o IPCA é mais **fraca e positiva**, sugerindo que o sentimento capta expectativas futuras mais do que a inflação corrente

---

## 📂 Estrutura do Projeto

Analise_sentimento_COPOM/
│
├── Análise de sentimentos atas do COPOM.ipynb   # Notebook principal com todo o fluxo de análise
├── functions.py                                 # Funções auxiliares utilizadas no notebook
├── atas_progresso.pkl                           # Arquivo de dados processados (pickle)
├── selic_historico_completo.csv                 # Base de dados histórica da taxa Selic
│
├── imagens/                                     # (Opcional) Pasta para imagens utilizadas no projeto
│     └── capa.jpg
│
├── dados/                                       # (Opcional) Pasta para armazenar dados brutos
│     └── atas_copom_brutas.csv
│
├── resultados/                                  # (Opcional) Pasta para salvar gráficos e resultados finais
│     └── grafico_sentimento_selic.png
│
├── README.md                                    # (Opcional) Descrição do projeto, instruções de uso e objetivos
├── requirements.txt                             # (Opcional) Lista de dependências do projeto
└── .gitignore                                   # (Opcional) Arquivos e pastas a serem ignorados pelo git