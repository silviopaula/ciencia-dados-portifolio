# Previsão de Consumo de Energia com AutoKeras, Keras e TensorFlow

Este projeto tem como objetivo realizar previsões de consumo de energia elétrica com base em dados históricos, utilizando uma abordagem progressiva: começando com ferramentas de alto nível (AutoKeras) e avançando para Keras e por seguinte para uma linguagem de baixo nível TensorFlow.

## Etapas do Projeto

1. **Pré-processamento dos dados**
   - Criação de variáveis sazonais (mês, trimestre, dia da semana etc.)
   - Normalização com `MinMaxScaler`

2. **Modelagem com AutoKeras (Alto Nível)**
   - Uso de `AutoModel` com `RegressionHead`
   - Seleção automática de arquiteturas e hiperparâmetros
   - Salvamento dos melhores modelos e escaladores

3. **Reconstrução Manual com Keras (Intermediário)**
   - Em desenvolvimento

4. **Treinamento com TensorFlow Puro (Baixo Nível)**
   - Em desenvolvimento

5. **Avaliação e Métricas**
   - MAPE
   - Comparação entre modelos AutoKeras e manuais
   - Visualização de previsões x observações

## Tecnologias Utilizadas

- Python 3
- AutoKeras
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib / Seaborn

