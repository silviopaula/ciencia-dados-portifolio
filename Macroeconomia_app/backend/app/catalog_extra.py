"""Extensão autoral do Catálogo Macroeconométrico — Estudos M31 a M60.

Todos os 30 estudos adicionais são 100% autorais, equilibrando os 6 pilares temáticos
(10 estudos por pilar = 60 estudos totais) e alimentados por séries oficiais do
Banco Central do Brasil (SGS e Olinda Focus) persistidas no DuckDB e Parquet.
"""

from __future__ import annotations

from typing import Any, Dict, List

STUDIES_CATALOG_EXTRA: List[Dict[str, Any]] = [
    # =========================================================================
    # PILAR 1: POLÍTICA FISCAL & DÍVIDA (+5 -> M31..M35, total 10)
    # =========================================================================
    {
        "code": "M31",
        "slug": "dbgg-vs-dlsp-reservas-internacionais",
        "title": "Dívida Bruta (DBGG) vs. Dívida Líquida (DLSP): O Papel das Reservas Internacionais",
        "subtitle": "Por que o hiato entre dívida bruta e dívida líquida no Brasil reage intensamente ao estoque de reservas e ao câmbio?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "Enquanto a Dívida Bruta (DBGG) mede o passivo total do governo geral sem abater ativos, a Dívida Líquida (DLSP) "
            "deduz haveres financeiros — em especial as reservas internacionais. Em episódios de depreciação cambial, o valor "
            "em reais das reservas cresce e reduz a DLSP no curto prazo, abrindo uma cunha expressiva em relação à DBGG."
        ),
        "methodology_explanation": (
            "Aplicamos o teste de cointegração de Engle-Granger entre DBGG (% PIB) e DLSP (% PIB) com dados oficiais do BCB "
            "(SGS 13762, 4513 e 3546), estimando a relação de equilíbrio de longo prazo e o parâmetro de correção de erros (α_ECT)."
        ),
        "interpretation_guide": (
            "Verifique a magnitude do coeficiente β de longo prazo e os períodos em que o resíduo ECT se amplia — normalmente "
            "associados a fortes choques cambiais ou operações compromissadas do Banco Central."
        ),
        "references": [
            "Banco Central do Brasil — Manual de Estatísticas Fiscais (Conceitos DBGG e DLSP).",
            "Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction. Econometrica, 55(2).",
        ],
        "dataset_keys": [],
        "bcb_series": [[13762, "DBGG (% PIB)"], [4513, "DLSP (% PIB)"], [3546, "Reservas Int. (US$ mi)"]],
    },
    {
        "code": "M32",
        "slug": "divida-mobiliaria-primario-selic",
        "title": "Resultado Primário, Juros e Acumulação da Dívida Mobiliária Federal",
        "subtitle": "Como oscilações no resultado primário e na taxa Selic se propagam sobre o estoque de títulos públicos federais?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "A Dívida Mobiliária Federal representa a maior parcela do financiamento do Tesouro Nacional em mercado. "
            "Como grande fração desses títulos é indexada à Selic (LFTs) ou à inflação (NTN-Bs), choques monetários e "
            "deteriorações primárias alteram rapidamente a necessidade de rolagem e o perfil de endividamento."
        ),
        "methodology_explanation": (
            "Estimamos um VAR em primeiras diferenças entre a Dívida Mobiliária Federal (% PIB — SGS 4503) e o Resultado "
            "Primário acumulado em 12 meses (% PIB — SGS 5793), acompanhados de Funções Impulso-Resposta (IRF) e testes de Granger."
        ),
        "interpretation_guide": (
            "Analise o sinal da IRF após um choque no resultado primário: quedas no superávit primário tendem a elevar a "
            "trajetória da dívida mobiliária nos trimestres subsequentes."
        ),
        "references": [
            "Tesouro Nacional — Relatório Mensal da Dívida Pública Federal (RMD).",
            "Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis. Springer.",
        ],
        "dataset_keys": [],
        "bcb_series": [[4503, "Dívida Mobiliária (% PIB)"], [5793, "Primário 12m (% PIB)"], [4189, "Selic (% a.a.)"]],
    },
    {
        "code": "M33",
        "slug": "nfsp-nominal-custo-divida-pib",
        "title": "NFSP Nominal, Custo da Dívida e Dinâmica do Produto Mensal",
        "subtitle": "Qual o peso relativo da taxa de juros e do tamanho nominal do PIB sobre a Necessidade de Financiamento do Setor Público?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos",
        "economic_intuition": (
            "O resultado nominal (NFSP) soma o resultado primário à conta de juros nominais incidentes sobre a dívida pública. "
            "Em economias com endividamento elevado e taxa básica alta, a despesa financeira domina a variação mensal do déficit nominal."
        ),
        "methodology_explanation": (
            "Estimamos uma regressão MQO com erros-padrão robustos à heterocedasticidade e autocorrelação (Newey-West HAC) "
            "relacionando NFSP Nominal (SGS 4649) a Selic (SGS 4189), PIB Mensal Corrente (SGS 4380) e DBGG (SGS 13762)."
        ),
        "interpretation_guide": (
            "Observe a significância estatística (p < 0,05) dos coeficientes de Selic e DBGG na explicação do déficit nominal "
            "e avalie o R² ajustado para o poder explicativo conjunto."
        ),
        "references": [
            "Giambiagi, F., & Além, A. C. (2016). Finanças Públicas: Teoria e Prática no Brasil. Elsevier.",
        ],
        "dataset_keys": [],
        "bcb_series": [[4649, "NFSP Nominal (R$ mi)"], [4189, "Selic (% a.a.)"], [4380, "PIB Mensal (R$ mi)"], [13762, "DBGG (% PIB)"]],
    },
    {
        "code": "M34",
        "slug": "ciclo-fiscal-primario-estrutural",
        "title": "Ciclo Fiscal Brasileiro: Componente Estrutural vs. Cíclico do Resultado Primário",
        "subtitle": "Quanto da melhora ou piora do resultado primário reflete esforço fiscal genuíno versus efeito do ciclo econômico?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP)",
        "economic_intuition": (
            "A arrecadação tributária cresce automaticamente em expansões econômicas e contrai em recessões. Separar a "
            "tendência estrutural do resultado primário de suas flutuações cíclicas evita confundir ganhos temporários de "
            "receita com equilíbrio fiscal permanente."
        ),
        "methodology_explanation": (
            "Aplicamos o Filtro Hodrick-Prescott (λ = 14.400) sobre a série mensal do Resultado Primário (% PIB — SGS 5793), "
            "contrastando a tendência estrutural com o componente cíclico e a correlação com o IBC-Br."
        ),
        "interpretation_guide": (
            "Valores cíclicos persistentemente negativos indicam deterioração além da tendência; compare o ponto mais recente "
            "da curva de tendência para avaliar o esforço fiscal de médio prazo."
        ),
        "references": [
            "Gobetti, S. W., & Orair, R. O. (2017). Resultado Estrutural e Impulso Fiscal no Brasil. Texto para Discussão IPEA.",
        ],
        "dataset_keys": [],
        "bcb_series": [[5793, "Primário 12m (% PIB)"], [24364, "IBC-Br"], [13762, "DBGG (% PIB)"]],
    },
    {
        "code": "M35",
        "slug": "transbordamento-fiscal-cambio-inflacao",
        "title": "Transbordamento Fiscal: Impacto de Choques na DBGG sobre Câmbio e Inflação",
        "subtitle": "Elevações inesperadas da dívida pública pressionam a taxa de câmbio e se transmitem à inflação acumulada?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "Pela Teoria Fiscal do Nível de Preços (FTPL) e pelo canal de prêmio de risco, trajetórias fiscais percebidas "
            "como insustentáveis ampliam a demanda por proteção cambial, depreciam o real e encarecem bens comercializáveis."
        ),
        "methodology_explanation": (
            "Estimamos um VAR bivariado em primeiras diferenças entre DBGG (% PIB — SGS 13762) e Câmbio Comercial (SGS 3698), "
            "integrando o IPCA 12m (SGS 13522) na matriz de correlação e na análise exploratória."
        ),
        "interpretation_guide": (
            "Examine os p-valores de Granger e a função impulso-resposta em 10 meses para identificar se choques fiscais "
            "precedem movimentos cambiais estatisticamente relevantes."
        ),
        "references": [
            "Cochrane, J. H. (2023). The Fiscal Theory of the Price Level. Princeton University Press.",
        ],
        "dataset_keys": [],
        "bcb_series": [[13762, "DBGG (% PIB)"], [3698, "Câmbio R$/US$"], [13522, "IPCA 12m (%)"]],
    },

    # =========================================================================
    # PILAR 2: POLÍTICA MONETÁRIA & JUROS (+4 -> M36..M39, total 10)
    # =========================================================================
    {
        "code": "M36",
        "slug": "agregados-monetarios-m2-base-pib",
        "title": "Meios de Pagamento (M1 & M2), Base Monetária e PIB no Longo Prazo",
        "subtitle": "A expansão da liquidez monetária acompanha a trajetória nominal da economia brasileira de forma cointegrada?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "Na Teoria Quantitativa da Moeda (MV = PY), a demanda por agregados monetários amplos como o M2 guarda relação "
            "de longo prazo com a renda nominal e com a liquidez criada pelo Banco Central via base monetária, ainda que "
            "inovações financeiras (como o Pix) alterem a velocidade de circulação no curto prazo."
        ),
        "methodology_explanation": (
            "Testamos cointegração de Engle-Granger entre M2 (SGS 27842) e Base Monetária (SGS 1788), analisando o desvio "
            "de equilíbrio (ECT) e a associação contemporânea com o PIB Mensal Corrente (SGS 4380) e M1 (SGS 27841)."
        ),
        "interpretation_guide": (
            "Um parâmetro α_ECT negativo e significante confirma que excessos temporários de liquidez retornam gradualmente "
            "à relação estrutural entre moeda restrita e moeda ampla."
        ),
        "references": [
            "Friedman, M., & Schwartz, A. J. (1963). A Monetary History of the United States. Princeton University Press.",
            "Banco Central do Brasil — Nota Técnica sobre Agregados Monetários (M1–M4).",
        ],
        "dataset_keys": [],
        "bcb_series": [[27842, "M2 (R$ mil)"], [1788, "Base Monetária (R$ mil)"], [4380, "PIB Mensal (R$ mi)"], [27841, "M1 (R$ mil)"]],
    },
    {
        "code": "M37",
        "slug": "spread-bancario-inadimplencia-selic",
        "title": "Spread Bancário, Taxa de Juros de Mercado e Inadimplência no Brasil",
        "subtitle": "Choques de inadimplência na carteira de crédito precedem a abertura do spread bancário brasileiro?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "A decomposição do spread bancário do Banco Central aponta a inadimplência como um dos principais fatores por "
            "trás das elevadas margens de intermediação financeira no Brasil, ao lado de custos administrativos, compulsórios "
            "e cunha tributária."
        ),
        "methodology_explanation": (
            "Estimamos um VAR em primeiras diferenças entre Spread Bancário Médio (SGS 20783) e Inadimplência Total (SGS 21082), "
            "avaliando precedência temporal de Granger e dinâmica impulso-resposta junto à Taxa Selic (SGS 4189)."
        ),
        "interpretation_guide": (
            "Verifique se ΔInadimplência Granger-causa ΔSpread (p < 0,05) e observe na IRF quantos meses os bancos levam "
            "para reprecificar o risco de crédito nas novas concessões."
        ),
        "references": [
            "Banco Central do Brasil — Relatório de Economia Bancária (REB).",
        ],
        "dataset_keys": [],
        "bcb_series": [[20783, "Spread Bancário (p.p.)"], [21082, "Inadimplência (%)"], [4189, "Selic (% a.a.)"]],
    },
    {
        "code": "M38",
        "slug": "canal-credito-pib-selic-inadimplencia",
        "title": "Canal do Crédito: Aprofundamento Financeiro (Crédito/PIB) e Política Monetária",
        "subtitle": "Como a relação Crédito/PIB reage ao custo do dinheiro, à Selic e aos ciclos de inadimplência?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos",
        "economic_intuition": (
            "O canal do crédito amplifica decisões de política monetária: juros básicos mais altos elevam o custo de captação "
            "e deterioram balanços de famílias e firmas (acelerador financeiro), restringindo a razão Crédito/PIB."
        ),
        "methodology_explanation": (
            "Ajustamos regressão MQO com erros robustos Newey-West HAC explicando Crédito Total/PIB (SGS 20622) a partir da "
            "Selic (SGS 4189), da Taxa Média de Juros do Crédito (SGS 20714) e da Inadimplência (SGS 21082)."
        ),
        "interpretation_guide": (
            "Avalie os sinais esperados (negativos para juros e inadimplência) e a estacionariedade dos resíduos via ADF "
            "para verificar a consistência da especificação."
        ),
        "references": [
            "Bernanke, B. S., & Gertler, M. (1995). Inside the Black Box: The Credit Channel of Monetary Policy Transmission. JEP, 9(4).",
        ],
        "dataset_keys": [],
        "bcb_series": [[20622, "Crédito / PIB (%)"], [4189, "Selic (% a.a.)"], [20714, "Juros Crédito (% a.a.)"], [21082, "Inadimplência (%)"]],
    },
    {
        "code": "M39",
        "slug": "assimetria-juros-pf-vs-pj-selic",
        "title": "Assimetria da Transmissão Monetária: Crédito Pessoa Física vs. Pessoa Jurídica",
        "subtitle": "O repasse da taxa Selic ocorre com a mesma intensidade nas linhas de crédito para famílias e empresas?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Matriz de Correlação",
        "economic_intuition": (
            "Empréstimos a empresas (PJ) possuem garantias mais robustas e competição com o mercado de capitais (debêntures), "
            "enquanto o crédito à pessoa física (PF) inclui modalidades sem garantia (rotativo e cheque especial), gerando "
            "forte dispersão de taxas e repasses assimétricos."
        ),
        "methodology_explanation": (
            "Estimamos regressão MQO-HAC relacionando a Taxa Média de Juros PF (SGS 20716) às taxas PJ (SGS 20715), à Selic "
            "(SGS 4189) e à Inadimplência Total (SGS 21082), complementada por matriz de correlação completa."
        ),
        "interpretation_guide": (
            "Compare a sensibilidade de PF em relação a PJ e Selic: coeficientes elevados junto à inadimplência sinalizam "
            "que o risco idiossincrático pesa tanto quanto a taxa básica na ponta final."
        ),
        "references": [
            "Mishkin, F. S. (1995). Symposium on the Monetary Transmission Mechanism. Journal of Economic Perspectives, 9(4).",
        ],
        "dataset_keys": [],
        "bcb_series": [[20716, "Juros PF (% a.a.)"], [20715, "Juros PJ (% a.a.)"], [4189, "Selic (% a.a.)"], [21082, "Inadimplência (%)"]],
    },

    # =========================================================================
    # PILAR 3: INFLAÇÃO & EXPECTATIVAS (+4 -> M40..M43, total 10)
    # =========================================================================
    {
        "code": "M40",
        "slug": "ancoragem-expectativas-focus-meta",
        "title": "Ancoragem das Expectativas: Boletim Focus 12m vs. Meta CMN e IPCA Realizado",
        "subtitle": "As expectativas de inflação do mercado respondem mais à meta oficial do CMN ou à inércia da inflação corrente?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) sobre Dados Focus Olinda BCB",
        "economic_intuition": (
            "Em regimes de metas de inflação críveis, as expectativas para 12 meses à frente permanecem ancoradas na meta "
            "mesmo diante de choques temporários no IPCA corrente. Quando há desancoragem, o peso da inflação passada sobe "
            "e o custo de desinflação em termos de produto aumenta."
        ),
        "methodology_explanation": (
            "Combinamos a mediana suavizada de Expectativas de Inflação 12m do Boletim Focus (API Olinda BCB) com IPCA 12m "
            "(SGS 13522), Meta CMN (SGS 13521) e Selic (SGS 4189) em uma regressão MQO com erros robustos Newey-West HAC."
        ),
        "interpretation_guide": (
            "Compare o coeficiente sobre a Meta CMN versus o coeficiente sobre o IPCA 12m: quanto menor a sensibilidade à "
            "inflação passada e maior a aderência à meta, mais forte é a credibilidade monetária."
        ),
        "references": [
            "Gürkaynak, R. S., Levin, A., & Swanson, E. (2010). Does Inflation Targeting Anchor Long-Run Inflation Expectations? JEEA, 8(6).",
            "Banco Central do Brasil — API Olinda de Expectativas de Mercado (Boletim Focus).",
        ],
        "dataset_keys": [],
        "bcb_series": [[-1, "Focus IPCA 12m (%)"], [13522, "IPCA 12m (%)"], [13521, "Meta CMN (%)"], [4189, "Selic (% a.a.)"]],
    },
    {
        "code": "M41",
        "slug": "precos-administrados-vs-livres-choques",
        "title": "Preços Administrados vs. Preços Livres: Propagação de Choques de Oferta",
        "subtitle": "Choques em tarifas e combustíveis (administrados) contaminam os preços livres de mercado nos meses seguintes?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "Preços monitorados (energia elétrica, combustíveis, planos de saúde e transportes) correspondem a cerca de um "
            "quarto do IPCA e são pouco sensíveis à taxa de juros no curto prazo, mas podem gerar efeitos secundários sobre "
            "preços livres via custos logísticos e indexação."
        ),
        "methodology_explanation": (
            "Estimamos um modelo VAR entre IPCA Preços Livres (SGS 11428) e IPCA Administrados (SGS 4449), mensurando o "
            "transbordamento de choques via IRF de 10 meses e testes de Causalidade de Granger."
        ),
        "interpretation_guide": (
            "Se ΔAdministrados Granger-causa ΔLivres com IRF positiva nos primeiros meses, há evidência de efeitos de segunda "
            "ordem que exigem vigilância da autoridade monetária."
        ),
        "references": [
            "Minella, A., Freitas, P. S., Goldfajn, I., & Muinhos, M. K. (2003). Inflation Targeting in Brazil. BCB Working Paper 77.",
        ],
        "dataset_keys": [],
        "bcb_series": [[11428, "IPCA Livres (% m/m)"], [4449, "IPCA Administrados (% m/m)"], [433, "IPCA Cheio (% m/m)"]],
    },
    {
        "code": "M42",
        "slug": "inflacao-servicos-alimentacao-nucleo",
        "title": "Inflação de Serviços, Alimentação e Núcleos Subjacentes do Banco Central",
        "subtitle": "Qual grupo de preços explica com maior estabilidade a variação mensal do IPCA cheio no Brasil?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Decomposição de Grupos",
        "economic_intuition": (
            "A inflação de alimentação no domicílio apresenta alta volatilidade climática e cambial, enquanto serviços e "
            "núcleos por exclusão (EX0) refletem pressões domésticas de demanda, salários e hiato do produto."
        ),
        "methodology_explanation": (
            "Ajustamos regressão MQO-HAC do IPCA Mensal (SGS 433) sobre Serviços (SGS 10844), Alimentação e Bebidas "
            "(SGS 1635) e Núcleo EX0 (SGS 11426), avaliando a contribuição relativa e a estabilidade dos resíduos."
        ),
        "interpretation_guide": (
            "Compare os coeficientes e estatísticas t de cada componente: o Núcleo EX0 captura a tendência persistente, "
            "enquanto Alimentação explica grandes desvios transitórios do índice cheio."
        ),
        "references": [
            "Banco Central do Brasil — Relatório Trimestral de Inflação (Boxe: Novas Medidas de Núcleo de Inflação).",
        ],
        "dataset_keys": [],
        "bcb_series": [[433, "IPCA (% m/m)"], [10844, "IPCA Serviços (% m/m)"], [1635, "IPCA Alimentação (% m/m)"], [11426, "Núcleo EX0 (% m/m)"]],
    },
    {
        "code": "M43",
        "slug": "transmissao-atacado-varejo-ipa-ipca",
        "title": "Transmissão de Atacado ao Varejo: IPA-DI, IPCA-15 e INPC",
        "subtitle": "Quanto tempo pressões de preços ao produtor (IPA-DI) levam para atingir o consumidor final no IPCA?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "O Índice de Preços ao Produtor Amplo (IPA-DI) reage quase imediatamente a choques de câmbio e commodities "
            "agrícolas/industriais, funcionando como indicador antecedente de pressões de custo sobre o IPCA e o INPC."
        ),
        "methodology_explanation": (
            "Estimamos um VAR entre IPCA mensal (SGS 433) e IPA-DI (SGS 225) acompanhado da análise descritiva e de "
            "correlação com a prévia oficial IPCA-15 (SGS 7478) e o INPC (SGS 188)."
        ),
        "interpretation_guide": (
            "Observe o teste de Granger na direção IPA-DI → IPCA e o pico da IRF para estimar a defasagem média (em meses) "
            "do repasse de custos do atacado para o varejo."
        ),
        "references": [
            "FGV IBRE — Metodologia dos Índices Gerais de Preços (IGP-DI / IPA-DI).",
        ],
        "dataset_keys": [],
        "bcb_series": [[433, "IPCA (% m/m)"], [225, "IPA-DI (% m/m)"], [7478, "IPCA-15 (% m/m)"], [188, "INPC (% m/m)"]],
    },

    # =========================================================================
    # PILAR 4: ATIVIDADE ECONÔMICA & PIB (+5 -> M44..M48, total 10)
    # =========================================================================
    {
        "code": "M44",
        "slug": "ibcbr-industria-pim-varejo-pmc",
        "title": "O IBC-Br como Termômetro Mensal da Indústria (PIM-PF) e do Varejo (PMC)",
        "subtitle": "Qual a dinâmica de precedência entre a produção física industrial, vendas no varejo e o índice IBC-Br?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "Como as Contas Nacionais Trimestrais do IBGE são divulgadas com defasagem, o IBC-Br do Banco Central sintetiza "
            "mensalmente a atividade econômica a partir da indústria (PIM-PF), do comércio varejista (PMC), de serviços e da agropecuária."
        ),
        "methodology_explanation": (
            "Estimamos um VAR em primeiras diferenças entre IBC-Br com ajuste sazonal (SGS 24364) e Produção Industrial Geral "
            "(SGS 21859), integrando as Vendas no Varejo Restrito (SGS 1455) nas correlações e diagnósticos ADF."
        ),
        "interpretation_guide": (
            "Avalie a forte co-movimentação contemporânea na matriz de correlação e a resposta dinâmica na IRF para medir o "
            "peso de choques industriais sobre o ritmo mensal da atividade."
        ),
        "references": [
            "Banco Central do Brasil — Nota Metodológica do Índice de Atividade Econômica (IBC-Br).",
        ],
        "dataset_keys": [],
        "bcb_series": [[24364, "IBC-Br"], [21859, "Indústria PIM-PF"], [1455, "Varejo PMC"]],
    },
    {
        "code": "M45",
        "slug": "ciclo-industrial-bens-capital-consumo",
        "title": "Ciclo Industrial Brasileiro: Bens de Capital, Intermediários e de Consumo",
        "subtitle": "A produção de bens de capital — proxy de investimento produtivo — exibe ciclos mais profundos que o resto da indústria?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP)",
        "economic_intuition": (
            "Na macroeconomia de ciclos reais, o investimento (refletido na produção de máquinas e equipamentos — bens de capital) "
            "é o componente mais pró-cíclico e volátil da demanda agregada, antecipando viradas na formação bruta de capital fixo."
        ),
        "methodology_explanation": (
            "Aplicamos o Filtro Hodrick-Prescott sobre a série oficial de Bens de Capital da PIM-PF/IBGE (SGS 21862), "
            "comparando seu comportamento cíclico com Bens Intermediários (SGS 21863), Bens de Consumo (SGS 21864) e Indústria Geral (SGS 21859)."
        ),
        "interpretation_guide": (
            "Identifique os picos e vales do componente cíclico (hiato) de bens de capital e compare a amplitude do desvio-padrão "
            "nas estatísticas descritivas frente aos bens de consumo."
        ),
        "references": [
            "Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. Business Cycles: An Empirical Investigation. JMCB, 29(1).",
        ],
        "dataset_keys": [],
        "bcb_series": [[21862, "Bens de Capital"], [21863, "Bens Intermediários"], [21864, "Bens de Consumo"], [21859, "Indústria Geral"]],
    },
    {
        "code": "M46",
        "slug": "nuci-energia-industrial-producao",
        "title": "Utilização da Capacidade Instalada (NUCI), Energia Industrial e Produção Física",
        "subtitle": "Indicadores de alta frequência como consumo elétrico industrial e NUCI explicam a produção física da indústria?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) para Nowcasting Industrial",
        "economic_intuition": (
            "Fábricas em operação demandam imediatamente energia elétrica e elevam o nível de utilização da capacidade instalada "
            "(NUCI). Combinar indicadores coincidentes físicos permite fazer nowcasting preciso da produção industrial antes da divulgação oficial."
        ),
        "methodology_explanation": (
            "Estimamos regressão MQO-HAC da Produção Industrial PIM-PF (SGS 21859) sobre NUCI (SGS 24352), Consumo Industrial "
            "de Energia Elétrica (SGS 1404) e Produção de Autoveículos (SGS 1378)."
        ),
        "interpretation_guide": (
            "Observe o R² ajustado e os coeficientes positivos de NUCI e energia industrial: quanto maior o ajuste com resíduos "
            "estacionários, mais confiável é o modelo para monitoramento em tempo real."
        ),
        "references": [
            "Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The Real-Time Informational Content of Macroeconomic Data. JME, 55(4).",
        ],
        "dataset_keys": [],
        "bcb_series": [[21859, "Indústria PIM-PF"], [24352, "NUCI Indústria (%)"], [1404, "Energia Industrial (GWh)"], [1378, "Autoveículos"]],
    },
    {
        "code": "M47",
        "slug": "varejo-ampliado-credito-confianca-icc",
        "title": "Varejo Restrito vs. Varejo Ampliado, Crédito e Confiança do Consumidor (ICC)",
        "subtitle": "Existe equilíbrio de longo prazo entre vendas no varejo restrito e ampliado diante dos ciclos de crédito e confiança?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "O varejo ampliado acrescenta veículos e materiais de construção ao varejo restrito — segmentos altamente sensíveis "
            "ao prazo de financiamento, taxa de juros e confiança das famílias (ICC). Divergências entre os dois índices sinalizam "
            "mudanças no apetite por bens duráveis."
        ),
        "methodology_explanation": (
            "Executamos teste de cointegração de Engle-Granger e ECM entre Varejo Ampliado (SGS 28473) e Varejo Restrito "
            "(SGS 1455), em conjunto com a Confiança do Consumidor (SGS 4393) e Crédito/PIB (SGS 20622)."
        ),
        "interpretation_guide": (
            "Desvios positivos do termo de correção de erros (ECT) revelam aceleração relativa de veículos e construção civil "
            "impulsionada por crédito ou confiança mais favorável."
        ),
        "references": [
            "IBGE — Pesquisa Mensal de Comércio (PMC): Notas Técnicas.",
        ],
        "dataset_keys": [],
        "bcb_series": [[28473, "Varejo Ampliado"], [1455, "Varejo Restrito"], [4393, "Confiança ICC"], [20622, "Crédito / PIB (%)"]],
    },
    {
        "code": "M48",
        "slug": "previsao-consumo-energia-eletrica",
        "title": "Previsão de Curto Prazo do Consumo Total de Energia Elétrica (Termômetro do PIB)",
        "subtitle": "Qual a trajetória projetada para a carga mensal de energia elétrica brasileira nos próximos 12 meses?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "sarima_forecast",
        "methodology_label": "Modelagem ARIMA(1,1,1), Correlogramas ACF/PACF & Projeção 12 Meses",
        "economic_intuition": (
            "A carga de energia elétrica acompanha estreitamente o crescimento estrutural da economia, a renda das famílias "
            "(uso residencial/comercial) e o ritmo fabril, apresentando forte tendência ascendente combinada com sazonalidade térmica."
        ),
        "methodology_explanation": (
            "Diagnosticamos estacionariedade (ADF) e estrutura de autocorrelação (ACF/PACF) do Consumo Total de Energia "
            "Elétrica (SGS 1406) e projetamos 12 passos à frente via ARIMA(1,1,1) com banda de confiança de 95%."
        ),
        "interpretation_guide": (
            "Observe nos correlogramas a persistência sazonal nos lags 12 e 24 e avalie a inclinação da projeção frente à "
            "média histórica de consumo industrial e do IBC-Br."
        ),
        "references": [
            "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control.",
        ],
        "dataset_keys": [],
        "bcb_series": [[1406, "Energia Total (GWh)"], [1404, "Energia Industrial (GWh)"], [24364, "IBC-Br"]],
    },

    # =========================================================================
    # PILAR 5: MERCADO DE TRABALHO (+6 -> M49..M54, total 10)
    # =========================================================================
    {
        "code": "M49",
        "slug": "lei-de-okun-brasil-desocupacao-ibcbr",
        "title": "Lei de Okun no Brasil: Elasticidade do Desemprego ao Ciclo de Atividade",
        "subtitle": "Quanto a taxa de desocupação PNAD cai quando o nível de atividade econômica (IBC-Br) acelera?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Lei de Okun Empírica (MQO Newey-West HAC) & Matriz de Correlação",
        "economic_intuition": (
            "A Lei de Okun descreve a relação empírica inversa entre variações no desemprego e crescimento do produto acima do "
            "potencial. No Brasil, a forte presença do setor de serviços e da informalidade amortecem parte dos choques de curto "
            "prazo, mas mantêm forte correlação negativa em horizontes cíclicos."
        ),
        "methodology_explanation": (
            "Estimamos regressão MQO com erros robustos Newey-West HAC explicando a Taxa de Desocupação PNAD (SGS 24369) "
            "pelo IBC-Br (SGS 24364), Produção Industrial (SGS 21859) e Vendas no Varejo (SGS 1455)."
        ),
        "interpretation_guide": (
            "Confirme o sinal negativo e significante do coeficiente de atividade (IBC-Br/Varejo): quanto maior sua magnitude "
            "absoluta, maior a sensibilidade do mercado de trabalho brasileiro ao ciclo econômico."
        ),
        "references": [
            "Okun, A. M. (1962). Potential GNP: Its Measurement and Significance. American Statistical Association.",
            "Ball, L., Leigh, D., & Loungani, P. (2017). Okun's Law: Fit at 50? Journal of Money, Credit and Banking, 49(7).",
        ],
        "dataset_keys": [],
        "bcb_series": [[24369, "Desocupação PNAD (%)"], [24364, "IBC-Br"], [21859, "Indústria PIM-PF"], [1455, "Varejo PMC"]],
    },
    {
        "code": "M50",
        "slug": "rendimento-real-pnad-inflacao-servicos",
        "title": "Rendimento Médio Real (PNAD) e Pressões na Inflação de Serviços",
        "subtitle": "Ganhos reais nos rendimentos do trabalho mantêm vínculo estrutural com a dinâmica dos preços de serviços?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "O setor de serviços é intensivo em mão de obra e pouco exposto à concorrência externa. Quando o mercado de "
            "trabalho se aquece e os salários reais avançam acima da produtividade, os custos salariais pressionam a inflação de serviços."
        ),
        "methodology_explanation": (
            "Analisamos a relação de equilíbrio entre Rendimento Médio Real Habitual (SGS 24381) e Efetivo (SGS 24382) da "
            "PNAD Contínua via Engle-Granger/ECM, cruzando sua trajetória com a Inflação de Serviços (SGS 10844)."
        ),
        "interpretation_guide": (
            "Verifique a convergência entre rendimento efetivo e habitual (ECM) e inspecione na matriz de correlação o vínculo "
            "entre aceleração salarial e IPCA de serviços."
        ),
        "references": [
            "Blanchard, O., & Katz, L. F. (1999). Wage Dynamics: Reconciling Theory and Evidence. AER, 89(2).",
        ],
        "dataset_keys": [],
        "bcb_series": [[24381, "Rendimento Habitual (R$)"], [24382, "Rendimento Efetivo (R$)"], [10844, "IPCA Serviços (% m/m)"]],
    },
    {
        "code": "M51",
        "slug": "demografia-forca-trabalho-ocupados-pnad",
        "title": "Demografia da Força de Trabalho: População Ocupada vs. Inatividade (PNAD)",
        "subtitle": "A queda recente do desemprego decorre principalmente da geração de vagas ou da saída de trabalhadores da força de trabalho?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP)",
        "economic_intuition": (
            "A taxa de desocupação pode cair por dois motivos bem distintos: expansão genuína da população ocupada ou "
            "desalento/envelhecimento demográfico que eleva a população fora da força de trabalho. Decompor a tendência "
            "da ocupação esclarece a saúde estrutural do emprego."
        ),
        "methodology_explanation": (
            "Aplicamos o Filtro Hodrick-Prescott sobre a População Ocupada da PNAD Contínua (SGS 24379, em mil pessoas), "
            "comparando tendência e hiato com a Força de Trabalho (SGS 24378) e População Fora da Força (SGS 24371)."
        ),
        "interpretation_guide": (
            "Um componente cíclico positivo da população ocupada acima da tendência estrutural aponta mercado de trabalho "
            "apertado e potencial pressão de custos salariais."
        ),
        "references": [
            "IBGE — Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD Contínua).",
        ],
        "dataset_keys": [],
        "bcb_series": [[24379, "População Ocupada (mil)"], [24378, "Força de Trabalho (mil)"], [24371, "Fora da Força (mil)"]],
    },
    {
        "code": "M52",
        "slug": "novo-caged-desocupados-absorcao-formal",
        "title": "Novo CAGED vs. População Desocupada: Velocidade de Absorção Formal",
        "subtitle": "Saldos mensais de contratações formais no Novo CAGED antecipam a redução do contingente de desocupados na PNAD?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "O Novo CAGED registra mensalmente admissões e desligamentos celetistas (CLT) com grande tempestividade, enquanto "
            "a PNAD Contínua mede o contingente total de desocupados em média móvel trimestral — criando defasagens naturais "
            "entre o registro administrativo e a pesquisa domiciliar."
        ),
        "methodology_explanation": (
            "Estimamos um VAR em primeiras diferenças entre o Saldo Mensal do Novo CAGED (SGS 28763) e a População Desocupada "
            "da PNAD (SGS 24380), acompanhados do IBC-Br (SGS 24364) desde a implantação da nova série CAGED."
        ),
        "interpretation_guide": (
            "Examine os p-valores de Granger para confirmar se o CAGED atua como indicador antecedente da queda no estoque de "
            "desocupados da PNAD."
        ),
        "references": [
            "Ministério do Trabalho e Emprego — Nota Técnica do Novo CAGED.",
        ],
        "dataset_keys": [],
        "bcb_series": [[28763, "Saldo Novo CAGED"], [24380, "Desocupados PNAD (mil)"], [24364, "IBC-Br"]],
    },
    {
        "code": "M53",
        "slug": "salario-minimo-cesta-basica-inpc",
        "title": "Salário Mínimo Nominal, Cesta Básica (DIEESE) e Poder de Compra (INPC)",
        "subtitle": "Como a evolução do salário mínimo e da inflação de alimentos molda o custo da cesta básica das famílias?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Análise de Poder de Compra",
        "economic_intuition": (
            "Para famílias de menor renda, a alimentação representa a maior fração do orçamento mensal. Comparar o valor da "
            "cesta básica com o salário mínimo nominal e o INPC permite avaliar ganhos ou perdas reais no padrão de consumo básico."
        ),
        "methodology_explanation": (
            "Estimamos regressão MQO-HAC relacionando o Custo da Cesta Básica em São Paulo (SGS 7493) ao Salário Mínimo "
            "Nominal (SGS 1619), ao INPC (SGS 188) e ao IPCA Alimentação e Bebidas (SGS 1635)."
        ),
        "interpretation_guide": (
            "Avalie a elasticidade da cesta básica frente ao piso salarial e observe nos gráficos de série histórica os períodos "
            "de choque de alimentos em que a cesta se descola do salário mínimo."
        ),
        "references": [
            "DIEESE — Metodologia da Pesquisa Nacional da Cesta Básica de Alimentos.",
        ],
        "dataset_keys": [],
        "bcb_series": [[7493, "Cesta Básica SP (R$)"], [1619, "Salário Mínimo (R$)"], [188, "INPC (% m/m)"], [1635, "IPCA Alimentos (% m/m)"]],
    },
    {
        "code": "M54",
        "slug": "projecao-arima-desocupacao-pnad",
        "title": "Projeção da Taxa de Desocupação PNAD Contínua via ARIMA",
        "subtitle": "Qual a trajetória inercial projetada para o desemprego brasileiro nos próximos 12 meses considerando sua forte inércia?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "sarima_forecast",
        "methodology_label": "Modelagem ARIMA(1,1,1), Correlogramas ACF/PACF & Projeção 12 Meses",
        "economic_intuition": (
            "Por ser divulgada em trimestres móveis, a taxa de desocupação PNAD possui alta persistência estatística e "
            "sazonalidade marcante (alta no 1º trimestre após demissões temporárias de fim de ano e queda no 4º trimestre)."
        ),
        "methodology_explanation": (
            "Aplicamos teste de raiz unitária ADF, inspeção dos correlogramas ACF/PACF e modelagem ARIMA(1,1,1) sobre a "
            "série oficial da Taxa de Desocupação PNAD (SGS 24369), gerando projeção de 12 meses com banda de confiança de 95%."
        ),
        "interpretation_guide": (
            "Use a banda de 95% como referência de incerteza estatística: desvios persistentes abaixo da projeção inercial "
            "indicam surpresa positiva de atividade ou redução da taxa de participação."
        ),
        "references": [
            "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.",
        ],
        "dataset_keys": [],
        "bcb_series": [[24369, "Desocupação PNAD (%)"], [24379, "Ocupados PNAD (mil)"], [24380, "Desocupados PNAD (mil)"]],
    },

    # =========================================================================
    # PILAR 6: CÂMBIO & FINANÇAS (+6 -> M55..M60, total 10)
    # =========================================================================
    {
        "code": "M55",
        "slug": "balanca-comercial-exportacoes-importacoes-cambio",
        "title": "Balança Comercial Brasileira: Exportações, Importações e Câmbio Real Efetivo",
        "subtitle": "Exportações e importações brasileiras cointegram em uma trajetória sustentável de saldo comercial?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "A condição de sustentabilidade intertemporal da balança comercial requer que exportações e importações não "
            "divirjam indefinidamente, sendo reconciliadas no longo prazo por ajustes na taxa de câmbio real efetiva e na renda doméstica."
        ),
        "methodology_explanation": (
            "Testamos cointegração de Engle-Granger e ECM entre Exportações FOB (SGS 22708) e Importações FOB (SGS 22709), "
            "incorporando o Índice da Taxa de Câmbio Real Efetiva (SGS 11752) e o Saldo Comercial (SGS 22707)."
        ),
        "interpretation_guide": (
            "Um coeficiente β > 1 e α_ECT negativo indicam que a expansão exportadora sustenta estruturalmente superávits "
            "comerciais robustos mesmo em ciclos de recuperação das importações."
        ),
        "references": [
            "Husted, S. (1992). The Emerging U.S. Current Account Deficit in the 1980s: A Cointegration Analysis. REStat, 74(1).",
        ],
        "dataset_keys": [],
        "bcb_series": [[22708, "Exportações FOB (US$ mi)"], [22709, "Importações FOB (US$ mi)"], [11752, "Câmbio Real Efetivo"], [22707, "Saldo Comercial (US$ mi)"]],
    },
    {
        "code": "M56",
        "slug": "commodities-icbr-agro-metal-exportacoes",
        "title": "Ciclo de Commodities (IC-Br Agro, Metal e Energia) e Exportações Brasileiras",
        "subtitle": "Choques nos preços internacionais de commodities agropecuárias e metálicas lideram as receitas de exportação?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "Sendo um dos maiores exportadores globais de soja, minério de ferro, proteína animal e petróleo, o Brasil tem "
            "seus termos de troca, sua arrecadação e sua entrada líquida de dólares altamente correlacionados aos subíndices do IC-Br."
        ),
        "methodology_explanation": (
            "Estimamos um VAR em primeiras diferenças entre Exportações FOB (SGS 22708) e IC-Br Agropecuária (SGS 27575), "
            "incorporando IC-Br Metal (SGS 27576) e IC-Br Energia (SGS 27577) nas estatísticas e matriz de correlação."
        ),
        "interpretation_guide": (
            "Observe a função impulso-resposta e os p-valores de Granger para verificar a velocidade com que ciclos de "
            "commodities se convertem em divisas na balança comercial."
        ),
        "references": [
            "Banco Central do Brasil — Metodologia do Índice de Commodities Brasil (IC-Br).",
        ],
        "dataset_keys": [],
        "bcb_series": [[22708, "Exportações FOB (US$ mi)"], [27575, "IC-Br Agro"], [27576, "IC-Br Metal"], [27577, "IC-Br Energia"]],
    },
    {
        "code": "M57",
        "slug": "transacoes-correntes-financiamento-idp",
        "title": "Contas Externas: Transações Correntes e Financiamento por Investimento Direto (IDP)",
        "subtitle": "O ingresso de Investimento Direto no País (IDP) e o saldo comercial garantem financiamento de qualidade ao balanço de pagamentos?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) sobre Balanço de Pagamentos",
        "economic_intuition": (
            "Déficits em transações correntes financiados por Investimento Direto no País (capital produtivo de longo prazo) "
            "conferem vulnerabilidade externa muito menor do que déficits financiados por capital especulativo de curto prazo."
        ),
        "methodology_explanation": (
            "Ajustamos regressão MQO-HAC explicando o Saldo de Transações Correntes (SGS 22701) a partir do Saldo Comercial "
            "(SGS 22707), Investimento Direto no País — IDP (SGS 22885) e Câmbio Comercial (SGS 3698)."
        ),
        "interpretation_guide": (
            "Compare a média mensal de IDP com o saldo de transações correntes nas estatísticas descritivas e avalie o papel "
            "dominante do saldo comercial na redução de vulnerabilidades externas."
        ),
        "references": [
            "Banco Central do Brasil — Notas Metodológicas de Estatísticas do Setor Externo (BPM6).",
        ],
        "dataset_keys": [],
        "bcb_series": [[22701, "Transações Correntes (US$ mi)"], [22885, "IDP Mensal (US$ mi)"], [22707, "Saldo Comercial (US$ mi)"], [3698, "Câmbio R$/US$"]],
    },
    {
        "code": "M58",
        "slug": "desalinhamento-cambio-real-efetivo-hp",
        "title": "Desalinhamento Cambial: Tendência de Equilíbrio da Taxa de Câmbio Real Efetiva",
        "subtitle": "O real encontra-se depreciado ou apreciado frente à sua tendência estrutural de competitividade externa?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP)",
        "economic_intuition": (
            "A Taxa de Câmbio Real Efetiva pondera a cotação do real frente a uma cesta de parceiros comerciais descontando "
            "os diferenciais de inflação (IPCA vs. IPC externo). Desvios acima da tendência indicam ganho temporário de "
            "competitividade cambial; desvios abaixo indicam apreciação real."
        ),
        "methodology_explanation": (
            "Aplicamos o Filtro Hodrick-Prescott (λ = 14.400) ao Índice de Câmbio Real Efetivo IPCA (SGS 11752), contrastando "
            "os desvios cíclicos com o Câmbio Nominal (SGS 3698) e o IC-Br (SGS 27574)."
        ),
        "interpretation_guide": (
            "Observe o último valor do ciclo HP: desvios fortemente positivos indicam câmbio real mais depreciado que o "
            "equilíbrio histórico recente, favorecendo exportadores mas elevando pressões de custos importados."
        ),
        "references": [
            "Edwards, S. (1989). Real Exchange Rates, Devaluation, and Adjustment. MIT Press.",
        ],
        "dataset_keys": [],
        "bcb_series": [[11752, "Câmbio Real Efetivo"], [3698, "Câmbio R$/US$"], [27574, "IC-Br Commodities"]],
    },
    {
        "code": "M59",
        "slug": "volatilidade-condicional-commodities-icbr",
        "title": "Volatilidade Condicional do Índice de Commodities Brasil (IC-Br) e Choques Globais",
        "subtitle": "Como a volatilidade das commodities globais se agrupa em regimes de estresse geopolítico e inflacionário?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "volatility",
        "methodology_label": "Econometria Financeira: Retornos Log, Volatilidade EWMA (λ=0,94) & Curtose",
        "economic_intuition": (
            "Preços de commodities agrícolas, metálicas e energéticas apresentam agrupamento de volatilidade (volatility clustering) "
            "em eventos climáticos, pandemias e choques geopolíticos, impactando diretamente o IPCA, o IGP-M e a receita fiscal."
        ),
        "methodology_explanation": (
            "Computamos retornos logarítmicos percentuais e a volatilidade condicional EWMA (RiskMetrics, λ = 0,94) do IC-Br "
            "Geral (SGS 27574), junto a métricas de curtose, assimetria e correlação com IC-Br Energia (SGS 27577) e Câmbio (SGS 3698)."
        ),
        "interpretation_guide": (
            "Picos na curva EWMA identificam janelas de choque global de custos; curtose elevada confirma maior probabilidade "
            "de saltos abruptos nos preços de matérias-primas."
        ),
        "references": [
            "J.P. Morgan / Reuters (1996). RiskMetrics Technical Document (4th ed.).",
        ],
        "dataset_keys": [],
        "bcb_series": [[27574, "IC-Br Geral"], [27577, "IC-Br Energia"], [3698, "Câmbio R$/US$"]],
    },
    {
        "code": "M60",
        "slug": "reservas-internacionais-liquidez-externa-arima",
        "title": "Colchão de Liquidez Externa: Reservas Internacionais e Projeção ARIMA",
        "subtitle": "Qual a dinâmica e a estabilidade projetada do estoque de reservas internacionais do Banco Central?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "sarima_forecast",
        "methodology_label": "Modelagem ARIMA(1,1,1), Correlogramas ACF/PACF & Projeção 12 Meses",
        "economic_intuition": (
            "As reservas internacionais (em torno de US$ 350–370 bilhões) funcionam como seguro soberano contra paradas "
            "súbitas de fluxos de capitais, ancoram a liquidez em moeda estrangeira e reduzem o risco de crises cambiais clássicas."
        ),
        "methodology_explanation": (
            "Aplicamos teste ADF, análise ACF/PACF e projeção ARIMA(1,1,1) de 12 meses sobre o estoque oficial de Reservas "
            "Internacionais no conceito liquidez (SGS 3546), em conjunto com Câmbio (SGS 3698) e IDP (SGS 22885)."
        ),
        "interpretation_guide": (
            "Acompanhe a estabilidade da série histórica e a largura do intervalo de 95% para avaliar a robustez do colchão "
            "externo frente a intervenções pontuais no mercado à vista."
        ),
        "references": [
            "Obstfeld, M., Shambaugh, J. C., & Taylor, A. M. (2010). Financial Stability, the Trilemma, and International Reserves. AEJ: Macro, 2(2).",
        ],
        "dataset_keys": [],
        "bcb_series": [[3546, "Reservas Int. (US$ mi)"], [3698, "Câmbio R$/US$"], [22885, "IDP (US$ mi)"]],
    },
]
