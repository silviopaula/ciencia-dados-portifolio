# Classificação de Reviews: BoW vs GloVe

Projeto de classificação binária de avaliações do Yelp utilizando Apache Spark e técnicas de Processamento de Linguagem Natural. Comparação entre Bag of Words e GloVe embeddings.

## Objetivo

Classificar avaliações de estabelecimentos do Yelp em duas categorias:
- **Positiva**: 5 estrelas (label 1)
- **Negativa**: menos de 5 estrelas (label 0)

**Foco principal**: comparar performance de Bag of Words (BoW) vs GloVe na classificação de sentimentos.

## Conjunto de Dados

**Yelp Reviews Dataset**

Contém avaliações de estabelecimentos como restaurantes, hotéis e lojas, incluindo:
- Texto da avaliação
- Nota (1-5 estrelas)
- Informações do autor
- Categoria do negócio

## Técnicas de Vetorização

### Bag of Words (BoW)
Representa texto baseado na frequência de termos, sem capturar semântica.

### GloVe (Global Vectors for Word Representation)
Algoritmo de Stanford (2014) que cria representações vetoriais densas capturando relações semânticas e sintáticas entre palavras. Diferencial: combina estatísticas globais do corpus com contexto local.

**Vantagens do GloVe sobre BoW:**
- Captura similaridade semântica
- Dimensionalidade fixa (100 features vs milhares no BoW)
- Generaliza conceitos e sentimentos
- Melhor performance em NLP moderno

## Pipeline do Projeto

### 1. Análise Exploratória
- Carregamento dos dados JSON
- Análise da distribuição das avaliações
- Conversão para classificação binária

### 2. Pré-processamento (NLP)
- **Tokenização**: divisão do texto em palavras
- **Remoção de Stopwords**: eliminação de palavras comuns
- **Vetorização** (BoW vs GloVe):
  - Bag of Words (BoW) com HashingTF
  - GloVe (Global Vectors) com embeddings pré-treinados de 100 dimensões

### 3. Treinamento de Modelos

Cada técnica de vetorização (BoW e GloVe) foi testada com:
- Regressão Logística
- Gradient-Boosted Tree (GBT)
- Random Forest

### 4. Avaliação
Métricas utilizadas:
- AUC-ROC
- AUC-PR
- F1-Score
- Acurácia
- Precisão
- Recall

## Tecnologias

- **Apache Spark**: processamento distribuído
- **PySpark ML**: pipeline de machine learning
- **Python**: pandas, numpy, plotly
- **Google Colab**: ambiente de execução

## Estrutura Principal

```
NLP_classificação_de_reviews.ipynb
├── Importação de bibliotecas
├── Obtenção e análise exploratória dos dados
├── Pré-processamento de texto
├── Treinamento de modelos
└── Avaliação e comparação
```

## Resultados

Comparação entre duas abordagens de vetorização:
- **Bag of Words (BoW)**: representação baseada em frequência de termos
- **GloVe**: representação baseada em embeddings semânticos pré-treinados

**Combinações testadas:**
- LogisticRegression_BoW
- GBTClassifier_BoW
- RandomForestClassifier_BoW
- LogisticRegression_GloVe
- GBTClassifier_GloVe
- RandomForestClassifier_GloVe

Os modelos são avaliados através das métricas: AUC-ROC, AUC-PR, F1-Score, Acurácia, Precisão e Recall.

GloVe demonstra desempenho superior ao BoW em todas as combinações de modelos testadas.

## Execução

O notebook foi desenvolvido para execução no Google Colab com dados armazenados no Google Drive.

## Requisitos

```
pyspark
pandas
numpy
plotly
```
