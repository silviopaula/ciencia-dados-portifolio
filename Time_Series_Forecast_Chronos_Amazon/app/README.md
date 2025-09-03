# Forecast com Amazon Chronos (Chronos/Bolt)

Este diretório contém:

* **`app.py`** — Aplicativo **Streamlit** para carregar dados, escolher modelos **Amazon Chronos/Chronos‑Bolt** (online/offline), gerar previsões com **intervalos (quantis)**, opcional **ensemble**, visualizar gráfico interativo (Plotly) e **baixar resultados em `.xlsx`**.
* **`script` da função** (ex.: `forecast_core.py`) — Implementa a função `chronos_forecast_multi_models_single_df` usada pelo app. Caso você tenha mantido a função dentro de `app.py`, este arquivo é opcional.

---

## O que o app faz

1. **Carrega arquivo** CSV/Excel.
2. Permite informar **coluna de data** e **coluna alvo**.
3. Seleciona **modelos Amazon Chronos** via **caixinhas (checkbox)**:

   * **Online (Hugging Face)** → nomes com “/” (ex.: `amazon/chronos-bolt-small`).
   * **Offline (pastas locais)** → nomes **sem “/”** (ex.: `amazon_chronos-bolt-small`) e você informa a **pasta base**.
4. Configura **horizonte**, **período de teste**, **seed**, **quantis** (ou `interval_alpha`), **horizonte longo via blocos (>64)** e **ensemble** (média/mediana/ponderado).
5. Plota **P50** por modelo e **faixa do intervalo** com a **mesma cor** (transparente).
6. Gera **`df_forecast` consolidado** (dados originais + colunas de projeção pL/p50/pU por modelo) e **métricas** (MAE, RMSE, MAPE, MedAE, R²).
7. Permite **baixar `.xlsx`** com abas: `metrics`, `treino`, `teste`, `forecast`.

---

## Parâmetros importantes (UI)

* **Colunas**

  * `Data` (ou o nome que você escolher) — deve ser conversível para `datetime`. Frequência mensal (MS) é assumida por padrão.
  * `Alvo` — série a ser prevista.
* **Modelos (Amazon Chronos)**

  * **Online (Hugging Face)**: selecione pelos sufixos (ex.: `chronos-bolt-small`). O app usa o nome completo internamente (ex.: `amazon/chronos-bolt-small`).
  * **Offline (pastas)**: informe a **pasta base** (vazia por padrão, nada pré‑preenchido) e marque as pastas (ex.: `amazon_chronos-bolt-small`).
* **Teste**: número de pontos finais usados para avaliar o `p50` (sombreado no gráfico).
* **Horizonte**: quantos períodos futuros prever (se >64, o app divide em **blocos** automaticamente).
* **Quantis**: informe lista JSON (ex.: `[0.1,0.5,0.9]`) **ou** use `interval_alpha=0.95` (gera `[0.025,0.5,0.975]` — ajustado para Bolt em `[0.1,0.5,0.9]`).
* **Seed**: define seeds (`numpy`, `random`, `torch`) para reprodutibilidade prática.
* **Ensemble (opcional)**: `mean`, `median` ou `weighted` + pesos opcionais.

---

## Online vs. Offline

* **Online**

  * Use nomes com “/”, ex.: `amazon/chronos-t5-mini`, `amazon/chronos-bolt-small`.
  * Requer internet no ambiente. Modelos ficam em cache local do Hugging Face após o primeiro uso.
* **Offline**
  * Utilizar os modelos mais pesos requer baixar mais de 1gb, portanto, foi criado o modelo off line, o script encontra-se dentro da parte models chronos. baixando os modelos da um total de 4.7gb todos.
  * Use nomes **sem “/”** (pastas no disco), ex.: `amazon_chronos-bolt-small`.
  * Informe a **pasta base** no app. Dentro dela, cada subpasta deve conter `config.json` do modelo.
  * Opção **“Forçar OFFLINE”** impede acessos externos (útil em ambientes restritos).

---

## Saídas

* **Gráfico (Plotly)** com linhas P50 por modelo e **intervalos** na mesma cor (transparente). A legenda **não exibe valores dos quantis**.
* **`metrics_df`**: MAE, RMSE, MAPE, MedAE, R², quantis usados.
* **`df_forecast`**: dados originais + colunas de projeção `alvo__modelo__pL|p50|pU`. Inclui `alvo_real`.
* **Excel** (`.xlsx`): `metrics`, `treino`, `teste`, `forecast`.

---

## Dicas & Boas práticas

* **Datas regulares**: garanta frequência mensal (ou ajuste no código para sua frequência). Preencha lacunas com `NaN` ou interpole.
* **Escala**: séries com magnitudes muito diferentes podem se beneficiar de normalização (MinMax/Padronização) fora do escopo do app.
* **Bolt em CPU**: os modelos **Bolt** são rápidos e robustos em CPU; `t5-*` maiores preferem GPU.
* **Quantis em Bolt**: são limitados a `[0.1, 0.9]` — o app ajusta automaticamente.

---

## Solução de problemas

* **“Falha ao carregar modelo …”** (online):

  * Verifique a internet e se o nome inclui “/” (ex.: `amazon/chronos-bolt-small`).
  * Alguns proxies corporativos exigem configurar variáveis de ambiente do sistema.
* **“Pasta do modelo local inválida”** (offline):

  * Confirme `local_base_dir` e a existência de `config.json` dentro da subpasta do modelo.
* **CUDA não disponível**:

  * O app troca automaticamente para CPU e exibe um aviso. Isso não impede o uso (especialmente com **Bolt**).
* **Quantis rejeitados** em Bolt:

  * Use `[0.1, 0.5, 0.9]` ou deixe `interval_alpha` que o app ajusta.

---

## Licenças & Créditos

* **Amazon Chronos / Chronos‑Bolt**: modelos de previsão pré‑treinados pela Amazon. Consulte as licenças no repositório do Hugging Face.
* Este projeto é um **exemplo educacional** usando bibliotecas de terceiros; verifique as licenças ao distribuir.

