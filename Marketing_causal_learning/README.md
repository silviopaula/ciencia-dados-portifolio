# Marketing Causal Learning

Notebook em Python que reproduz os estudos de caso do paper *Causal Machine Learning for Promotions: Industry Evidence and Applications* (DoorDash, KDD 2025). O projeto mostra, com dados sinteticos, como usar meta-learners e Double Machine Learning para decidir quem recebe uma promocao e qual desconto oferecer.

## Visao geral
- Divide o problema em dois exercicios: targeting discreto (Case 3.1) e profundidade de desconto continuo (Case 3.2).
- Gera dados sinteticos realistas (15k a 20k clientes) com variaveis de comportamento, tratamento e outcome binario.
- Compara S-Learner, T-Learner e duas variantes de DML usando metricas de uplift (Qini, Uplift@k) e diagnosticos visuais.
- Propaga estimativas individuais de uplift/tau para politicas de custo, ROI e economia esperada.

## Arquivos principais
- `Jupyter py - reproduzir experimento do paper.ipynb`: notebook completo (codigo, graficos e interpretacoes).
- `Jupyter py - reproduzir experimento do paper.html`: export em HTML para leitura rapida sem Jupyter.
- `paper.pdf`: copia do artigo base usado como referencia.
- `README.md`: este resumo.

## Requisitos
- Python 3.10 ou superior
- jupyter
- numpy
- pandas
- seaborn
- scikit-learn
- matplotlib

Instale tudo em um ambiente virtual (exemplo no Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install jupyter numpy pandas seaborn scikit-learn matplotlib
```

## Como rodar
1. Clone ou copie esta pasta para o seu ambiente local.
2. Crie/ative o ambiente virtual e instale as dependencias (ver comando acima).
3. Abra o servidor com `jupyter lab` ou `jupyter notebook`.
4. Carregue `Jupyter py - reproduzir experimento do paper.ipynb`.
5. Execute todas as celulas em ordem; o notebook e dividido em blocos claramente sinalizados para cada experimento.

## Fluxo do notebook

### Exercico 1 - Targeting discreto (quem recebe)
O primeiro exercicio imita o Case 3.1 do paper, focando em decidir quem recebe uma promocao fixa.

- Dados: 15k usuarios com `ordem_historica`, `engajamento_vertical`, `redemption_hist`, `dias_desde_ultimo`, tratamento `T` e outcome `Y`.
- Pipeline: geracao sintetica, split treino/teste estratificado, treino de S-Learner, T-Learner, DML basico e DML heterogeneo (Gradient Boosting), avaliacao por Qini e Uplift@30 e graficos (distribuicao, curva Qini, ranking).
- Diagnosticos: tabela automatica compara Qini in/out, gap de overfitting e destaca o metodo com melhor poder de targeting.

| Metodo | Qini_out | Uplift@30_out | Observacoes |
| --- | --- | --- | --- |
| S-Learner | 16.43% | 14.18% | baseline simples; efeito quase uniforme e maior gap in/out |
| T-Learner | 17.94% | 16.83% | dois modelos separados entregam maior precisao para targeting |
| DML basico | 18.27% | 18.17% | efeito constante (nao rankeia tao bem) mas quase sem overfitting |
| DML heterogeneo | 22.76% | 23.84% | melhor ranking causal; base para as politicas de targeting e ROI |

Aplicando o ranking causal no simulador, o custo promocional cai ~57% mantendo o mesmo volume de conversoes.

### Exercico 2 - Tratamento continuo (qual desconto dar)
O segundo exercicio replica o Case 3.2, estimando a sensibilidade individual ao desconto e escolhendo a profundidade ideal por cliente.

- Dados: 20k usuarios com descontos entre US$0.50 e US$4.97, taxa de conversao global de 53.7% e heterogeneidade de tau variando 30x entre perfis.
- Modelagem: DML continuo com cross-fitting (5 folds), Gradient Boosting para outcome/tratamento, pseudo-outcome regressivo e normalizacao.
- Politica: para cada cliente o notebook testa todos os niveis de desconto, calcula uplift esperado, custo, receita incremental e ROI, escolhendo o valor que maximiza lucro.
- Diagnosticos: correlacao entre tau real e estimado, analise de shrinkage, erro por nivel de desconto e DataFrame com recomendacoes (`tau_pred`, `optimal_discount`, `optimal_roi`).

| Indicador | Valor | Comentario |
| --- | --- | --- |
| Base de usuarios | 20.000 clientes, desconto US$0.50-US$4.97 | dataset sintetico com heterogeneidade de 30x |
| Taxa de conversao global | 53.7% | maior que o caso discreto (36.6%) |
| Tau medio estimado | 5.54 pp por US$1 | erro de +0.22 pp versus o valor real (5.32 pp) |
| Correlacao tau_real vs tau_pred | 0.441 | ranking razoavel mesmo com shrinkage |
| ROI medio simulado | ~8.2x | cada dolar de desconto gera ~US$8.2 de lucro incremental |
| Economia so com desconto otimo | 28% | mesmo numero de conversoes com menor desembolso |

O notebook tambem descreve passo a passo (estimativa de tau, testes contrafactuais, escolha via ROI) e aponta limitacoes (compressao das predicoes e necessidade de mais dados para capturar toda a variancia).

## Economia simulada
| Cenario | Custo promocoes | Conversoes | Economia vs baseline |
| --- | --- | --- | --- |
| Baseline uniforme | US$100.000 | 50.000 | - |
| Targeting otimizado (Exercico 1) | US$43.000 | 50.000 | US$57.000 (57%) |
| Discount otimizado (Exercico 2) | US$72.000 | 50.000 | US$28.000 (28%) |
| Ambos otimizados | US$31.000 | 50.000 | US$69.000 (69%) |

A combinacao dos dois frameworks se aproxima dos ganhos reportados pela DoorDash (70% de reducao de custo com o mesmo numero de pedidos).

## Referencias
- Paper oficial: veja `paper.pdf` (direto da selecao KDD 2025).
- Notebook em HTML: `Jupyter py - reproduzir experimento do paper.html` (visualizacao rapida).

## Proximos passos sugeridos
- Publicar um `requirements.txt` ou arquivo Poetry para fixar versoes e facilitar reproducao.
- Extrair as funcoes de geracao de dados e dos meta-learners para modulos reutilizaveis ou pacotes.
- Experimentar meta-learners adicionais (X-Learner, R-Learner, Causal Forest) ou bibliotecas como `econml`/`causalml`.
- Substituir os dados sinteticos por logs reais e conectar o fluxo a testes A/B e monitoramento de producao.
