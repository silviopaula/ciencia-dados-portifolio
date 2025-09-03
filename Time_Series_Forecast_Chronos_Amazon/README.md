# Time Series Chronos Forecast

## 📄 Descrição

Este README descreve o relatório `Time_Series_Chronos_Forecast.html` (exportado de um notebook Jupyter) e como reproduzi-lo/atualizá-lo.

Um relatório HTML estático com os resultados de previsão de séries temporais usando os modelos Amazon Chronos (linhas T5 e BOLT). 

### Características principais:
- ✅ Preparação dos dados
- ✅ Definição da função de previsão multimarcas (vários modelos de uma vez)
- ✅ Separação treino/teste
- ✅ Métricas (MAE, RMSE, MAPE, MedAE, R²)
- ✅ Intervalos preditivos (quantis como P10/P50/P90)
- ✅ Gráficos interativos (Plotly)
- ✅ Tabelas consolidadas com as previsões por modelo e ensemble

> **Nota:** Você não precisa de Python para visualizar: basta abrir no navegador.

## 🗂️ Estrutura do Conteúdo

1. **Setup** — pacotes, seeds e utilitários
2. **Dados** — série alvo (ex.: consumo de energia) e período
3. **Preparação** — agregação, pivoteamento e padronização de colunas
4. **Função principal** — rotina que:
   - Carrega um ou mais modelos Chronos (online/offline)
   - Calcula quantis (pL / p50 / pU)
   - Avalia no período de teste
   - Plota as curvas e intervalos
   - Retorna `metrics_df`, `df_treino`, `df_teste`, `df_forecast`
5. **Resultados** — gráficos, métricas e (opcional) ensemble
6. **Exportação** — salvamento dos artefatos (tabelas/figuras)

## ▶️ Como Visualizar

### Método 1: Duplo clique
Clique duas vezes no arquivo `Time_Series_Chronos_Forecast.html`

### Método 2: Navegador
Abra diretamente pelo navegador (Chrome/Edge/Firefox)

> Por ser estático, os gráficos interativos já funcionam no browser.

## 🔁 Como Reproduzir/Atualizar

### 1. Abrir o notebook
```bash
jupyter notebook Time_Series_Chronos_Forecast.ipynb
```

### 2. Configurar ambiente
Instale as dependências necessárias:

```bash
pip install pandas numpy scikit-learn plotly torch transformers accelerate sentencepiece safetensors huggingface_hub chronos-forecasting
```

### 3. Executar células
Execute todas as células (ajuste modelos/dados se necessário)

### 4. Exportar para HTML

**Via Jupyter:**
```
File → Save and Export As → HTML
```

**Via terminal:**
```bash
jupyter nbconvert --to html Time_Series_Chronos_Forecast.ipynb
```

### 5. Resultado
O novo `Time_Series_Chronos_Forecast.html` refletirá os resultados atualizados.

## 🧪 Saídas da Função de Forecast

### `metrics_df`
Uma linha por modelo com:
- MAE, RMSE, MAPE (%), MedAE, R²
- Quantis usados

### `forecasts`
Dicionário `{label → DataFrame}` com colunas:
- `pL` (limite inferior)
- `p50` (mediana) 
- `pU` (limite superior)

### `df_forecast`
DataFrame unificado = dados originais + colunas de previsão no padrão:
```
{target}__{label}__pL | {target}__{label}__p50 | {target}__{label}__pU
```

### Gráfico Plotly
- Série real (treino/teste)
- P50 de cada modelo
- Faixas dos intervalos na mesma cor do modelo (mais clara)

## 📊 Interpretação das Métricas

| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| **MAE** | Erro médio absoluto | Quanto menor, melhor |
| **RMSE** | Erro quadrático médio | Quanto menor, melhor |
| **MAPE (%)** | Erro percentual médio | Útil para comparar séries em magnitudes distintas |
| **MedAE** | Erro absoluto mediano | Robusto a outliers |
| **R²** | Variação explicada | Quanto mais próximo de 1, melhor |
| **P50** | Mediana | Previsão "central" |
| **P10/P90** | Limites do intervalo | Incerteza da previsão |

## 📁 Dados e Modelos

### Dados
- **Formato:** Série alvo em frequência mensal (ex.: consumo de energia)
- **Ajuste:** Nome da coluna de data e coluna alvo conforme sua base

### Modelos Chronos

#### Online (Hugging Face)
```python
amazon/chronos-t5-*
amazon/chronos-bolt-*
```

#### Offline
Pastas com `config.json` e pesos; passe o diretório-base na função.

> **⚠️ Observação:** Modelos BOLT aceitam quantis no intervalo [0.1, 0.9]; a função ajusta automaticamente se necessário.

## 🧑‍💻 Dicas de Uso

- 📏 Para horizontes > 64 pontos, a função faz previsão em blocos (auto-concatenação)
- 🔄 Você pode ativar ensemble (mean, median ou weighted) para combinar modelos
- 🎨 A paleta de cores é consistente entre P50 e o intervalo, facilitando leitura
- ☁️ Em ambientes na nuvem sem disco persistente, prefira modelos online

## 👨‍💻 Autor

**Dr. Silvio da Rosa Paula**
- GitHub: [https://github.com/silviopaula](https://github.com/silviopaula)

## 📜 Como Citar

```bibtex
Paula, S. R. (2025). Time Series Forecast with Amazon Chronos — Relatório HTML (Time_Series_Chronos_Forecast.html). Disponível no repositório do autor.
```

## 📝 Notas Adicionais

- Possível gerar README em inglês
- Possível versão para integração ao app Streamlit (menu "Sobre/Help")

---

**Versão:** 1.0  
**Última atualização:** 2025