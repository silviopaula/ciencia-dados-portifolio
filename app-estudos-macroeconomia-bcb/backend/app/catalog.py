"""Catálogo 100% autoral dos 60 Estudos Macroeconométricos do Observatório Macro Brasil Studio.

Todos os textos, intuições econômicas, explicações metodológicas, guias de interpretação e
referências acadêmicas foram redigidos do zero com foco didático, rigor econométrico e
clareza executiva sobre a economia brasileira (10 estudos por pilar x 6 pilares = 60).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .catalog_extra import STUDIES_CATALOG_EXTRA

THEMES: List[Dict[str, str]] = [
    {"key": "fiscal", "label": "Política Fiscal & Dívida"},
    {"key": "politica_monetaria", "label": "Política Monetária & Juros"},
    {"key": "inflacao", "label": "Inflação & Expectativas"},
    {"key": "atividade_pib", "label": "Atividade Econômica & PIB"},
    {"key": "trabalho", "label": "Mercado de Trabalho"},
    {"key": "cambio_externo", "label": "Câmbio & Finanças"},
]

STUDIES_CATALOG: List[Dict[str, Any]] = [
    # =========================================================================
    # PILAR 1: POLÍTICA FISCAL & DÍVIDA PÚBLICA
    # =========================================================================
    {
        "code": "M01",
        "slug": "dominancia-fiscal-selic-cds",
        "title": "Dominância Fiscal no Brasil: Taxa Selic e Risco Soberano (CDS 5 Anos)",
        "subtitle": "Quando um aperto monetário pode elevar o prêmio de risco soberano em vez de ancorar a inflação?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": True,
        "model_type": "dedicated_m01",
        "methodology_label": "VAR, Teste ADF, Correlação Cruzada (CCF) & Causalidade de Granger",
        "economic_intuition": (
            "Em um regime de dominância monetária, elevar a taxa básica de juros (Selic) contrai a demanda agregada, "
            "aprecia o câmbio e reduz a inflação. Sob dominância fiscal, porém, se o endividamento público é elevado e "
            "percebido como vulnerável, juros mais altos encarecem o serviço da dívida, elevam o risco de insolvência "
            "(medido pelo CDS soberano de 5 anos), induzem fuga de capitais e depreciam a moeda — minando a eficácia do Banco Central."
        ),
        "methodology_explanation": (
            "Aplicamos o teste de raiz unitária Augmented Dickey-Fuller (ADF) em nível e em primeira diferença para evitar "
            "regressões espúrias. Em seguida, estimamos um Vetor Autorregressivo (VAR) sobre ΔSelic e ΔCDS 5Y, selecionamos "
            "a defasagem ótima pelos critérios AIC/BIC/HQIC/FPE e executamos testes de Causalidade de Granger na amostra "
            "completa e em subamostras de estresse fiscal."
        ),
        "interpretation_guide": (
            "Observe os p-valores de Granger na subamostra: p < 0,05 na direção ΔSelic → ΔCDS (sem causalidade reversa "
            "estatisticamente significante) é um sinal empírico compatível com a hipótese de dominância fiscal descrita por Blanchard."
        ),
        "references": [
            "Blanchard, O. (2004). Fiscal Dominance and Inflation Targeting: Lessons from Brazil. NBER Working Paper 10389.",
            "Sargent, T. J., & Wallace, N. (1981). Some Unpleasant Monetarist Arithmetic. Federal Reserve Bank of Minneapolis.",
        ],
        "dataset_keys": ["cds5y", "selic"],
        "bcb_series": [[432, "Selic Meta (% a.a.)"]],
    },
    {
        "code": "M02",
        "slug": "sustentabilidade-projecao-divida-bruta",
        "title": "Sustentabilidade e Projeção da Dívida Bruta do Governo Geral (DBGG/PIB)",
        "subtitle": "Como o diferencial entre juro real e crescimento (r − g) e o resultado primário moldam a trajetória fiscal?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": True,
        "model_type": "dedicated_m02",
        "methodology_label": "Painel Fiscal (r − g), Teste ADF & Projeção ARIMA(1,1,1) com Bandas 80%/95%",
        "economic_intuition": (
            "A equação de acumulação da dívida pública estabelece que a relação Dívida/PIB cresce sempre que o superávit "
            "primário efetivo fica abaixo do superávit requerido para compensar o custo financeiro real acima do crescimento "
            "do produto (r − g). Monitorar essa dinâmica é essencial para antecipar pressões sobre juros longos e risco-país."
        ),
        "methodology_explanation": (
            "Integramos o painel fiscal histórico (DBGG/PIB, PIB 12m, juro real, primário efetivo vs. requerido e câmbio) "
            "à série oficial atualizada da DBGG do Banco Central (SGS 13762). Verificamos estacionariedade via ADF e ajustamos "
            "um modelo ARIMA(1,1,1) com drift para projetar a trajetória inercial com intervalos de confiança de 80% e 95%."
        ),
        "interpretation_guide": (
            "Compare a curva do resultado primário efetivo com o primário requerido no painel histórico; no gráfico de projeção, "
            "avalie tanto o ponto central quanto a amplitude da banda de 95% para mensurar o risco de cauda fiscal."
        ),
        "references": [
            "Blanchard, O. (2019). Public Debt and Low Interest Rates. American Economic Review, 109(4), 1197–1229.",
            "Banco Central do Brasil — Nota Metodológica das Estatísticas Fiscais (SGS 13762).",
        ],
        "dataset_keys": ["painel_fiscal"],
        "bcb_series": [[13762, "DBGG (% PIB)"], [4513, "Dívida Líquida (% PIB)"]],
    },
    {
        "code": "M03",
        "slug": "divida-publica-incerteza-cointegracao",
        "title": "Dívida Pública e Incerteza Econômica: Equilíbrio de Longo Prazo",
        "subtitle": "Existe relação estável de cointegração entre a deterioração fiscal e o nível de incerteza da economia?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "Incerteza fiscal elevada leva empresas a adiar investimentos e consumidores a ampliar a poupança precaucional, "
            "reduzindo a arrecadação e retroalimentando o próprio endividamento público. Testar se ambas as séries compartilham "
            "uma tendência estocástica comum ajuda a separar ruídos transitórios de desequilíbrios estruturais."
        ),
        "methodology_explanation": (
            "Estimamos a equação de longo prazo entre endividamento e incerteza, aplicamos o teste de cointegração de "
            "Engle-Granger sobre os resíduos (termo de correção de erros — ECT) e quantificamos a velocidade de retorno ao "
            "equilíbrio (α_ECT) na especificação de curto prazo."
        ),
        "interpretation_guide": (
            "Um p-valor de cointegração inferior a 0,05 acompanhado de α_ECT negativo e significante confirma convergência "
            "para a relação de equilíbrio de longo prazo após desvios temporários."
        ),
        "references": [
            "Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction. Econometrica, 55(2), 251–276.",
            "Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring Economic Policy Uncertainty. QJE, 131(4).",
        ],
        "dataset_keys": ["divida_incerteza"],
        "bcb_series": [[13762, "DBGG (% PIB)"], [4189, "Selic (% a.a.)"], [3698, "Câmbio R$/US$"]],
    },
    {
        "code": "M04",
        "slug": "econometria-ajuste-fiscal-receitas-despesas",
        "title": "Econometria do Ajuste Fiscal: Receitas Líquidas vs. Despesas Primárias",
        "subtitle": "A dinâmica orçamentária brasileira obedece à hipótese de 'gastar e tributar' ou à disciplina de receita?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Dinâmica de Ajuste Fiscal (ECM)",
        "economic_intuition": (
            "Quando despesas obrigatórias crescem de forma inercial e acima da expansão do PIB potencial, o setor público "
            "fica exposto a déficits estruturais persistentes em períodos de desaceleração cíclica. Medir o descompasso "
            "intertemporal entre receita líquida e despesa primária esclarece a rigidez do orçamento."
        ),
        "methodology_explanation": (
            "Analisamos a relação de longo prazo entre receitas e despesas primárias usando cointegração de Engle-Granger "
            "e estimação do mecanismo de correção de erros, complementados por testes de estacionariedade ADF e matriz de correlação."
        ),
        "interpretation_guide": (
            "Verifique se o desvio de equilíbrio (ECT) retorna rapidamente a zero ou se exibe longa persistência — "
            "sinal de baixa capacidade automática de reequilíbrio orçamentário."
        ),
        "references": [
            "Bohn, H. (1998). The Behavior of U.S. Public Debt and Deficits. Quarterly Journal of Economics, 113(3), 949–963.",
        ],
        "dataset_keys": ["ajuste_fiscal"],
        "bcb_series": [[13762, "DBGG (% PIB)"], [5793, "Resultado Primário (% PIB)"], [4513, "DLSP (% PIB)"]],
    },
    {
        "code": "M05",
        "slug": "regra-fiscal-juro-equilibrio",
        "title": "Âncora Fiscal, Regras de Gasto e Taxa de Juros de Equilíbrio",
        "subtitle": "De que maneira reformas fiscais críveis comprimem a tendência estrutural da taxa de juros real?",
        "theme": "fiscal",
        "theme_label": "Política Fiscal & Dívida",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP)",
        "economic_intuition": (
            "A taxa de juros neutra de uma economia emergente incorpora um componente de prêmio fiscal: quanto menor o risco "
            "de trajetória explosiva da dívida, menor o juro real exigido para equilibrar poupança e investimento no longo prazo. "
            "Regras fiscais críveis atuam diretamente sobre a tendência de baixa do juro estrutural."
        ),
        "methodology_explanation": (
            "Aplicamos o Filtro Hodrick-Prescott (HP) para separar flutuações cíclicas de curto prazo da tendência estrutural "
            "de longo prazo nas variáveis fiscais e de juros reais, avaliando mudanças de regime ao longo do tempo."
        ),
        "interpretation_guide": (
            "Compare a inclinação da curva de tendência HP antes e depois de marcos fiscais relevantes; quedas persistentes "
            "da tendência sugerem compressão estrutural do prêmio de risco."
        ),
        "references": [
            "Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. Business Cycles: An Empirical Investigation. JMCB, 29(1).",
        ],
        "dataset_keys": ["regra_fiscal"],
        "bcb_series": [[13762, "DBGG (% PIB)"], [4189, "Selic (% a.a.)"], [13522, "IPCA 12m (%)"]],
    },

    # =========================================================================
    # PILAR 2: POLÍTICA MONETÁRIA & JUROS
    # =========================================================================
    {
        "code": "M06",
        "slug": "funcao-reacao-banco-central-taylor",
        "title": "Função de Reação do Banco Central: Regra de Taylor e Hiato do Produto",
        "subtitle": "O COPOM eleva a taxa de juros real mais do que proporcionalmente quando as expectativas superam a meta?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": True,
        "model_type": "dedicated_m06",
        "methodology_label": "Regra de Taylor Dinâmica (MQO Newey-West HAC) + Hiato via Filtro HP",
        "economic_intuition": (
            "No regime de metas para a inflação, o Banco Central calibra a Selic reagindo aos desvios da inflação esperada "
            "em relação à meta e ao hiato do produto, com suavização gradual (ρ). Para ancorar a inflação, é indispensável "
            "satisfazer o Princípio de Taylor: resposta de longo prazo β₁/(1 − ρ) estritamente maior que 1."
        ),
        "methodology_explanation": (
            "Extraímos o hiato do produto aplicando o Filtro HP configurável (λ) sobre o log do PIB/IBC-Br e estimamos "
            "i_t = c + ρ·i_{t-1} + β₁·(E[π] − π*) + β₂·h_t via MQO com erros-padrão robustos Newey-West HAC, suportando "
            "atualização ao vivo via BCB SGS (4189, 13521, 24364) + Olinda Focus."
        ),
        "interpretation_guide": (
            "Verifique o indicador 'Princípio de Taylor': valores de β₁/(1 − ρ) > 1 indicam postura estabilizadora da "
            "autoridade monetária; um ρ elevado (0,7–0,9) reflete movimentos graduais e previsíveis de juros."
        ),
        "references": [
            "Taylor, J. B. (1993). Discretion versus Policy Rules in Practice. Carnegie-Rochester Conference Series, 39.",
            "Woodford, M. (2003). Interest and Prices: Foundations of a Theory of Monetary Policy. Princeton Univ. Press.",
        ],
        "dataset_keys": ["curva_reacao"],
        "bcb_series": [[4189, "Selic (% a.a.)"], [13521, "Meta (%)"], [24364, "IBC-Br"]],
    },
    {
        "code": "M07",
        "slug": "taxa-juros-neutra-economia-brasileira",
        "title": "Estimação da Taxa de Juros Neutra na Economia Brasileira",
        "subtitle": "Qual é o patamar de juro real que mantém a economia no produto potencial sem acelerar a inflação?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Filtro Hodrick-Prescott (HP) & Comparação Juro Real vs. Neutro",
        "economic_intuition": (
            "A taxa de juros neutra (r*) não é diretamente observável, mas funciona como bússola da política monetária: "
            "juros reais acima de r* exercem efeito contracionista sobre atividade e inflação; abaixo de r*, estimulam a demanda. "
            "Estimar sua trajetória de tendência permite diagnosticar a postura monetária em cada ciclo."
        ),
        "methodology_explanation": (
            "Utilizamos a base histórica de juros reais e expectativas para extrair a componente de tendência de longo prazo "
            "via Filtro HP, confrontando a taxa efetiva observada com a proxy estrutural de equilíbrio."
        ),
        "interpretation_guide": (
            "Períodos em que a linha observada permanece acima da tendência HP correspondem a ciclos de política monetária "
            "restritiva (gap de juros positivo)."
        ),
        "references": [
            "Laubach, T., & Williams, J. C. (2003). Measuring the Natural Rate of Interest. Review of Economics and Statistics, 85(4).",
        ],
        "dataset_keys": ["juro_real_vs_neutro", "juro_neutro"],
        "bcb_series": [[4189, "Selic (% a.a.)"], [13522, "IPCA 12m (%)"], [24364, "IBC-Br"]],
    },
    {
        "code": "M08",
        "slug": "juro-real-ex-ante-ex-post",
        "title": "Juro Real Ex-Ante vs. Ex-Post e Postura da Política Monetária",
        "subtitle": "Por que decisões de investimento respondem ao juro real esperado (ex-ante) e não apenas à inflação passada?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo (HP) sobre Séries de Juro Real",
        "economic_intuition": (
            "O juro real ex-post desconta a inflação já ocorrida, enquanto o juro real ex-ante utiliza expectativas de mercado "
            "para os próximos 12 meses — refletindo o custo efetivo prospectivo do crédito. Em choques inflacionários agudos, "
            "as duas métricas divergem drasticamente e podem induzir diagnósticos opostos sobre o aperto monetário."
        ),
        "methodology_explanation": (
            "Decompomos a série histórica de juro real em tendência estrutural e desvio cíclico via Filtro Hodrick-Prescott, "
            "comparando a evolução conjunta das medidas ex-ante e ex-post."
        ),
        "interpretation_guide": (
            "Observe o comportamento divergente em episódios de choque de preços: o juro ex-post cai mecanicamente com a alta "
            "da inflação corrente, enquanto o juro ex-ante reflete a credibilidade futura do Banco Central."
        ),
        "references": [
            "Fisher, I. (1930). The Theory of Interest. Macmillan.",
        ],
        "dataset_keys": ["juro_real_diario"],
        "bcb_series": [[4189, "Selic (% a.a.)"], [13522, "IPCA 12m (%)"]],
    },
    {
        "code": "M09",
        "slug": "credito-direcionado-eficiencia-monetaria",
        "title": "Crédito Direcionado e Potência dos Canais de Transmissão Monetária",
        "subtitle": "Em que medida o crédito com taxas subsidiadas reduz a sensibilidade da economia à taxa Selic?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos",
        "economic_intuition": (
            "Quando parcela relevante do crédito corporativo é concedida com taxas administradas desconectadas da Selic, "
            "o canal de crédito da política monetária perde potência. Como consequência, o Banco Central precisa elevar "
            "ainda mais a taxa básica sobre o segmento livre para obter o mesmo efeito desinflacionário."
        ),
        "methodology_explanation": (
            "Estimamos um modelo linear com correção de heterocedasticidade e autocorrelação (Newey-West HAC) relacionando "
            "condições monetárias, parcela de crédito direcionado e variáveis macroeconômicas de controle."
        ),
        "interpretation_guide": (
            "Analise o sinal e a significância (p < 0,05) dos coeficientes na tabela HAC, além do R² ajustado e do comportamento "
            "dos resíduos ao longo do tempo."
        ),
        "references": [
            "Bonomo, M., Brito, R. D., & Martins, B. (2015). The After-Crisis Government-Driven Credit Expansion in Brazil. JIBF.",
        ],
        "dataset_keys": ["credito_direcionado"],
        "bcb_series": [[20542, "Saldo Crédito Total"], [4189, "Selic (% a.a.)"], [7415, "Desembolsos BNDES"]],
    },
    {
        "code": "M10",
        "slug": "intervencoes-cambiais-swaps-bcb",
        "title": "Intervenções Cambiais via Swaps do Banco Central e Volatilidade",
        "subtitle": "Qual é a resposta dinâmica do câmbio a atuações do Banco Central no mercado de derivativos cambiais?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "Em um regime de câmbio flutuante, leilões de swaps cambiais fornecem proteção (hedge) em moeda estrangeira sem "
            "consumir diretamente reservas internacionais, buscando suavizar episódios de estresse de liquidez sem fixar "
            "artificialmente a taxa de câmbio."
        ),
        "methodology_explanation": (
            "Modelamos a interação dinâmica entre variações cambiais e intervenções via VAR em primeiras diferenças, "
            "computando Funções Impulso-Resposta (IRF) em 10 passos e testes de Causalidade de Granger bidirecionais."
        ),
        "interpretation_guide": (
            "Verifique se o efeito no gráfico de IRF dissipa-se rapidamente (alívio de liquidez de curto prazo) ou gera "
            "deslocamento persistente, e cheque se a causalidade corre do câmbio para a atuação do BC (função de reação)."
        ),
        "references": [
            "Kohlscheen, E., & Andrade, S. C. (2014). Official FX Interventions through Derivatives. Journal of International Money and Finance.",
        ],
        "dataset_keys": ["swaps_cambiais"],
        "bcb_series": [[3698, "Câmbio R$/US$"], [11752, "Câmbio Real Efetivo"], [4189, "Selic (% a.a.)"]],
    },
    {
        "code": "M11",
        "slug": "transmissao-selic-juros-pf-spread",
        "title": "Transmissão Monetária: Selic, Juros à Pessoa Física e Spread Bancário",
        "subtitle": "Quais determinantes explicam o elevado spread bancário e o repasse da taxa básica ao tomador final?",
        "theme": "politica_monetaria",
        "theme_label": "Política Monetária & Juros",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Matriz de Correlação",
        "economic_intuition": (
            "A distância entre o custo de captação (próximo à Selic) e a taxa paga por famílias e empresas depende de "
            "inadimplência, cunha tributária, custos administrativos, compulsórios e poder de mercado bancário. "
            "Quantificar essa elasticidade mostra quão rápido cortes ou altas da Selic chegam à economia real."
        ),
        "methodology_explanation": (
            "Ajustamos um modelo MQO com erros robustos Newey-West HAC sobre a série histórica de spread bancário e seus "
            "fatores macroeconômicos, permitindo atualização direta pelas séries oficiais de juros e spread do BCB."
        ),
        "interpretation_guide": (
            "Coeficientes positivos e estatisticamente significativos indicam pressão direta sobre o custo final do crédito; "
            "compare a aderência entre a série observada e a projetada pelo modelo."
        ),
        "references": [
            "Banco Central do Brasil — Relatório de Economia Bancária (Decomposição do Spread / ICC).",
        ],
        "dataset_keys": ["spread_bancario"],
        "bcb_series": [[20714, "Taxa de Juros PF (% a.a.)"], [20786, "Spread Bancário PF (p.p.)"], [4189, "Selic (% a.a.)"]],
    },

    # =========================================================================
    # PILAR 3: INFLAÇÃO & EXPECTATIVAS
    # =========================================================================
    {
        "code": "M12",
        "slug": "inercia-inflacionaria-rolling-ar1",
        "title": "Persistência e Inércia Inflacionária no Brasil (Rolling AR(1))",
        "subtitle": "Como evoluiu o grau de memória da inflação brasileira ao longo de diferentes ciclos monetários?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": True,
        "model_type": "dedicated_m12",
        "methodology_label": "Regressão em Janela Móvel (Rolling AR(1)) + IPCA 12m vs. Meta CMN",
        "economic_intuition": (
            "A inércia inflacionária mede o quanto a inflação passada se propaga automaticamente para o período seguinte "
            "via contratos indexados, reajustes salariais e expectativas adaptativas. Quanto maior o coeficiente γ(t), "
            "maior é a taxa de sacrifício necessária para trazer o IPCA de volta à meta."
        ),
        "methodology_explanation": (
            "Calculamos o IPCA acumulado em 12 meses confrontado à meta do CMN e estimamos regressões autorregressivas "
            "AR(1) em janela móvel configurável (18 a 96 meses), traçando γ(t) e sua banda de confiança de 95% ao longo do tempo."
        ),
        "interpretation_guide": (
            "Acompanhe elevações de γ(t) acima de 0,60: elas sinalizam desancoragem ou propagação prolongada de choques, "
            "enquanto quedas sustentadas indicam maior credibilidade do regime de metas."
        ),
        "references": [
            "Fuhrer, J. C. (2010). Inflation Persistence. Handbook of Monetary Economics, Vol. 3, Elsevier.",
        ],
        "dataset_keys": ["ipca_inercia", "meta_inflacao"],
        "bcb_series": [[433, "IPCA (% m/m)"], [13522, "IPCA 12m (%)"]],
    },
    {
        "code": "M13",
        "slug": "sarima-previsao-ipca",
        "title": "Modelagem Sazonal e Previsão do IPCA (SARIMA Box-Jenkins)",
        "subtitle": "Como capturar a sazonalidade típica da inflação brasileira para projetar o acumulado em 12 meses?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": True,
        "model_type": "dedicated_m13",
        "methodology_label": "SARIMA(p,d,q)(P,D,Q)[12], Teste ADF & Correlogramas ACF/PACF",
        "economic_intuition": (
            "O IPCA mensal brasileiro exibe um padrão sazonal marcante — com pressões típicas em educação no início do ano "
            "e alimentos/serviços no quarto trimestre. Modelos SARIMA permitem isolar essa sazonalidade estocástica e gerar "
            "benchmarks de previsão de curto e médio prazo com bandas probabilísticas."
        ),
        "methodology_explanation": (
            "A série oficial do IPCA (SGS 433) passa por diagnóstico ADF e inspeção de ACF/PACF até 24 defasagens. O usuário "
            "pode ajustar livremente as ordens (p,d,q)(P,D,Q)[12] e o horizonte de projeção, comparando AIC, BIC e RMSE."
        ),
        "interpretation_guide": (
            "Prefira especificações com menor AIC/BIC e sem picos significativos restantes na ACF/PACF dos resíduos; compare "
            "o acumulado projetado em 12 meses com a banda da meta de inflação."
        ),
        "references": [
            "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control.",
        ],
        "dataset_keys": ["ipca_historico"],
        "bcb_series": [[433, "IPCA (% m/m)"]],
    },
    {
        "code": "M14",
        "slug": "curva-de-phillips-brasil",
        "title": "Curva de Phillips Brasileira: Inflação, Expectativas e Ciclo Econômico",
        "subtitle": "Qual a sensibilidade da inflação brasileira às expectativas de mercado, ao câmbio e ao hiato de atividade?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": True,
        "model_type": "dedicated_m14",
        "methodology_label": "Curva de Phillips Híbrida (MQO Newey-West HAC) & Diagnóstico de Resíduos",
        "economic_intuition": (
            "A Curva de Phillips híbrida conecta a inflação corrente a três pilares: inércia passada (componente backward-looking), "
            "expectativas ancoradas (forward-looking), ociosidade da economia (hiato/desemprego) e choques de oferta/câmbio. "
            "A inclinação em relação à atividade define o custo real de desinflacionar."
        ),
        "methodology_explanation": (
            "Estimamos múltiplas especificações da Curva de Phillips sobre o painel macroeconômico brasileiro utilizando "
            "MQO com matriz de covariância Newey-West HAC, avaliando peso das expectativas, inércia e sensibilidade ao ciclo."
        ),
        "interpretation_guide": (
            "Compare o coeficiente das expectativas com o da inflação defasada: pesos mais altos nas expectativas indicam "
            "economia menos inercial e mais responsiva à comunicação do Banco Central."
        ),
        "references": [
            "Friedman, M. (1968). The Role of Monetary Policy. American Economic Review, 58(1), 1–17.",
            "Galí, J., & Gertler, M. (1999). Inflation Dynamics: A Structural Econometric Analysis. JME, 44(2).",
        ],
        "dataset_keys": ["curva_phillips"],
        "bcb_series": [[433, "IPCA (% m/m)"], [24364, "IBC-Br"], [3698, "Câmbio R$/US$"]],
    },
    {
        "code": "M15",
        "slug": "expectativas-especialistas-vs-consumidores",
        "title": "Expectativas de Inflação: Especialistas (Focus) vs. Consumidores",
        "subtitle": "Consumidores formam expectativas inflacionárias com mais inércia e maior erro de previsão que analistas?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO HAC) & Análise de Acurácia de Expectativas",
        "economic_intuition": (
            "A literatura de informação imperfeita mostra que famílias reagem fortemente a preços frequentes (alimentos e "
            "combustíveis), enquanto economistas profissionais incorporam projeções macro e a meta oficial. Entender essa "
            "assimetria melhora a leitura dos riscos de desancoragem."
        ),
        "methodology_explanation": (
            "Comparamos as séries históricas de expectativas do Boletim Focus, consumidores e inflação realizada via "
            "estatísticas descritivas, matriz de correlação e estimação MQO com erros robustos HAC."
        ),
        "interpretation_guide": (
            "Observe a correlação cruzada e a magnitude dos desvios-padrão: expectativas mais dispersas entre famílias "
            "costumam persistir mesmo após o início da queda do IPCA cheio."
        ),
        "references": [
            "Carvalho, F. A., & Minella, A. (2012). Survey Forecasts in Brazil: A Prismatic Assessment of Epidemiology, Performance, and Determinants. JMCB.",
        ],
        "dataset_keys": ["expectativas_grupos"],
        "bcb_series": [[13522, "IPCA 12m (%)"], [13521, "Meta (%)"], [4189, "Selic (% a.a.)"]],
    },
    {
        "code": "M16",
        "slug": "repasse-cambial-commodities-ipca",
        "title": "Repasse Cambial e de Commodities (IC-Br) para a Inflação",
        "subtitle": "Como choques na taxa de câmbio e no índice de commodities se transmitem aos preços domésticos?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "VAR, Função Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "O Brasil é grande exportador de commodities agropecuárias e metálicas; variações no índice IC-Br e no câmbio "
            "R$/US$ afetam diretamente insumos industriais, alimentos e combustíveis. A intensidade do repasse (pass-through) "
            "varia conforme o estado do ciclo econômico e a ancoragem das expectativas."
        ),
        "methodology_explanation": (
            "Ajustamos um modelo VAR em primeiras diferenças entre câmbio, IC-Br e inflação, calculando a Função "
            "Impulso-Resposta em 10 meses e testes de Causalidade de Granger para medir a velocidade de propagação."
        ),
        "interpretation_guide": (
            "Identifique no gráfico de IRF em qual horizonte (normalmente 1 a 4 meses) o choque externo atinge pico de "
            "transmissão sobre os preços internos."
        ),
        "references": [
            "Belaisch, A. (2003). Exchange Rate Pass-Through in Brazil. IMF Working Paper WP/03/141.",
        ],
        "dataset_keys": ["icbr_cambio"],
        "bcb_series": [[27574, "Commodities IC-Br"], [3698, "Câmbio R$/US$"], [433, "IPCA (% m/m)"]],
    },
    {
        "code": "M17",
        "slug": "igpm-nucleos-inflacao-previsao",
        "title": "Dinâmica do IGP-M e Núcleos de Inflação (Tendência Subjacente)",
        "subtitle": "Como separar choques temporários de oferta da inflação subjacente medida pelos núcleos e pelo IGP-M?",
        "theme": "inflacao",
        "theme_label": "Inflação & Expectativas",
        "specialized": False,
        "model_type": "sarima_forecast",
        "methodology_label": "Modelagem ARIMA, Correlogramas ACF/PACF & Projeção 12 Meses",
        "economic_intuition": (
            "Enquanto o IGP-M reflete intensamente preços no atacado (IPA) e oscilações cambiais — indexando aluguéis e "
            "tarifas —, os núcleos do IPCA filtram itens voláteis para revelar a pressão inflacionária de fundo que norteia "
            "as decisões do Banco Central."
        ),
        "methodology_explanation": (
            "Avaliamos a estrutura de autocorrelação (ACF/PACF) e ajustamos um modelo ARIMA sobre a inflação geral/núcleos "
            "para extrair projeções em 12 horizontes com intervalo de confiança de 95%."
        ),
        "interpretation_guide": (
            "Observe o decaimento da ACF: persistência lenta em múltiplos lags reforça o peso da indexação no IGP-M frente "
            "aos núcleos de serviços."
        ),
        "references": [
            "Bryan, M. F., & Cecchetti, S. G. (1994). Measuring Core Inflation. Monetary Policy, NBER.",
        ],
        "dataset_keys": ["nucleos_ipca", "igpm_historico"],
        "bcb_series": [[189, "IGP-M (% m/m)"], [433, "IPCA (% m/m)"], [11426, "IPCA Núcleo (% m/m)"]],
    },

    # =========================================================================
    # PILAR 4: ATIVIDADE ECONÔMICA, CICLOS & PIB
    # =========================================================================
    {
        "code": "M18",
        "slug": "processo-gerador-pib-brasil",
        "title": "Processo Gerador do PIB Brasileiro: Raiz Unitária e Persistência de Choques",
        "subtitle": "Choques negativos sobre o PIB brasileiro têm efeito transitório ou deslocam permanentemente a trajetória?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "sarima_forecast",
        "methodology_label": "Teste ADF, Estrutura ACF/PACF & Modelagem ARIMA do Produto",
        "economic_intuition": (
            "Uma questão clássica em macroeconomia (Nelson & Plosser, 1982) é se o PIB oscila em torno de uma tendência "
            "determinística ou se segue um processo integrado I(1) com tendência estocástica. No segundo caso, crises severas "
            "deixam cicatrizes permanentes (histerese) no nível de renda do país."
        ),
        "methodology_explanation": (
            "Executamos testes ADF para raiz unitária, examinamos a estrutura de dependência temporal via ACF/PACF e "
            "ajustamos um modelo ARIMA(1,1,1) para projetar a trajetória de curto prazo do indicador de produto."
        ),
        "interpretation_guide": (
            "Se o teste ADF em nível não rejeitar a raiz unitária (p > 0,05) e a ACF cair muito lentamente, há forte "
            "evidência de tendência estocástica e alta persistência de choques."
        ),
        "references": [
            "Nelson, C. R., & Plosser, C. I. (1982). Trends and Random Walks in Macroeconomic Time Series. JME, 10(2).",
        ],
        "dataset_keys": ["pib_processo"],
        "bcb_series": [[24364, "IBC-Br Dessazonalizado"], [4380, "PIB Mensal (R$ mi)"]],
    },
    {
        "code": "M19",
        "slug": "hiato-produto-filtro-hp",
        "title": "Hiato do Produto e Produto Potencial via Filtro Hodrick-Prescott",
        "subtitle": "A economia opera com capacidade ociosa ou sobreaquecida em relação ao seu produto potencial?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Filtro Hodrick-Prescott (HP) — Separação de Tendência Potencial e Hiato",
        "economic_intuition": (
            "O hiato do produto mede a diferença percentual entre o PIB efetivo e o produto potencial (nível máximo sustentável "
            "sem gerar pressões inflacionárias). Hiatos positivos sinalizam demanda aquecida; hiatos negativos indicam ociosidade "
            "de capital e trabalho."
        ),
        "methodology_explanation": (
            "Aplicamos a decomposição de Hodrick-Prescott calibrada à frequência da série (λ = 1.600 trimestral ou 14.400 "
            "mensal) para estimar o produto potencial e a trajetória histórica do hiato cíclico."
        ),
        "interpretation_guide": (
            "Observe no gráfico de componente cíclico os cruzamentos do eixo zero: transições sustentadas para terreno "
            "positivo costumam anteceder pressões de demanda sobre preços de serviços."
        ),
        "references": [
            "Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. Business Cycles. Journal of Money, Credit and Banking.",
        ],
        "dataset_keys": ["hiato_produto"],
        "bcb_series": [[24364, "IBC-Br Dessazonalizado"], [21859, "Produção Industrial PIM-PF"]],
    },
    {
        "code": "M20",
        "slug": "incerteza-economica-atividade-nuci",
        "title": "Incerteza Econômica, Nível de Atividade (PIB) e Capacidade Instalada",
        "subtitle": "Qual é o impacto dinâmico de um choque de incerteza econômica sobre o crescimento do produto?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "A teoria das opções reais demonstra que, sob alta incerteza macroeconômica ou regulatória, esperar antes de "
            "executar investimentos irreversíveis torna-se valioso. O congelamento coordenado de decisões de capital e "
            "contratação derruba a utilização da capacidade instalada (NUCI) e o crescimento do PIB."
        ),
        "methodology_explanation": (
            "Estimamos um modelo VAR bivariado relacionando variações do indicador de incerteza econômica e da atividade "
            "real, extraindo a Função Impulso-Resposta (IRF) e testes de Causalidade de Granger em múltiplas defasagens."
        ),
        "interpretation_guide": (
            "Verifique se choques positivos na incerteza produzem resposta negativa estatisticamente relevante na atividade "
            "ao longo dos trimestres seguintes na curva de IRF."
        ),
        "references": [
            "Bloom, N. (2009). The Impact of Uncertainty Shocks. Econometrica, 77(3), 623–685.",
        ],
        "dataset_keys": ["incerteza_pib", "incerteza_nuci"],
        "bcb_series": [[24364, "IBC-Br Dessazonalizado"], [21859, "Produção Industrial PIM-PF"]],
    },
    {
        "code": "M21",
        "slug": "contas-nacionais-ibcbr-industria",
        "title": "Contas Nacionais, IBC-Br e Dinâmica Setorial da Atividade",
        "subtitle": "Como os grandes componentes de oferta e demanda explicam os pontos de inflexão do ciclo brasileiro?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo (Filtro HP) & Correlação Setorial",
        "economic_intuition": (
            "Enquanto o PIB das Contas Nacionais do IBGE é divulgado trimestralmente com defasagem, indicadores mensais "
            "de alta frequência — como o IBC-Br, a produção industrial (PIM-PF) e as vendas do varejo (PMC) — permitem "
            "acompanhar em tempo quase real a aceleração ou desaceleração econômica."
        ),
        "methodology_explanation": (
            "Aplicamos o Filtro HP sobre as óticas das Contas Nacionais e indicadores coincidentes de atividade, avaliando "
            "sincronia cíclica por meio de matrizes de correlação e testes de estacionariedade."
        ),
        "interpretation_guide": (
            "Analise o alinhamento entre consumo das famílias, formação bruta de capital fixo (FBCF) e setor industrial na "
            "matriz de correlação para distinguir ciclos puxados por consumo de ciclos de investimento."
        ),
        "references": [
            "IBGE — Sistema de Contas Nacionais Trimestrais (Notas Metodológicas).",
        ],
        "dataset_keys": ["contas_nacionais"],
        "bcb_series": [[24364, "IBC-Br Dessazonalizado"], [4380, "PIB Mensal (R$ mi)"], [21859, "PIM-PF"]],
    },
    {
        "code": "M22",
        "slug": "bndes-energia-taxa-investimento",
        "title": "Crédito de Fomento (BNDES), Energia e Taxa de Investimento",
        "subtitle": "A expansão dos desembolsos de financiamento de longo prazo eleva estruturalmente a taxa de investimento?",
        "theme": "atividade_pib",
        "theme_label": "Atividade Econômica & PIB",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão Econométrica (MQO Newey-West HAC) & Diagnóstico de Resíduos",
        "economic_intuition": (
            "Bancos públicos de desenvolvimento buscam mitigar falhas de mercado no crédito de longo prazo para infraestrutura "
            "e bens de capital. Contudo, se recursos subsidiados apenas substituem financiamento privado (efeito crowding-out), "
            "o ganho líquido sobre a taxa agregada de investimento (FBCF/PIB) pode ser limitado."
        ),
        "methodology_explanation": (
            "Estimamos a associação empírica entre desembolsos de fomento e formação bruta de capital via MQO com erros "
            "robustos Newey-West HAC, verificando significância estatística e estabilidade dos resíduos."
        ),
        "interpretation_guide": (
            "Avalie não apenas o p-valor do coeficiente, mas também a magnitude econômica e a persistência dos resíduos — "
            "lembrando que correlação contemporânea não garante causalidade sem controle de endogeneidade."
        ),
        "references": [
            "Musacchio, A., & Lazzarini, S. G. (2014). Reinventing State Capitalism. Harvard University Press.",
        ],
        "dataset_keys": ["bndes_investimento", "energia"],
        "bcb_series": [[7415, "Desembolsos BNDES (R$ mi)"], [1406, "Consumo de Energia (GWh)"], [24364, "IBC-Br"]],
    },

    # =========================================================================
    # PILAR 5: MERCADO DE TRABALHO
    # =========================================================================
    {
        "code": "M23",
        "slug": "retropolacao-desemprego-pnad-pme",
        "title": "Retropolação da Taxa de Desemprego: PNAD Contínua vs. Antiga PME",
        "subtitle": "Como construir uma série longa e consistente de desemprego unindo pesquisas com metodologias distintas?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "ols_regression",
        "methodology_label": "Regressão de Encadeamento (MQO HAC) & Diagnóstico de Pareamento",
        "economic_intuition": (
            "A descontinuidade da Pesquisa Mensal de Emprego (PME, restrita a 6 regiões metropolitanas) em 2016 e a adoção "
            "da PNAD Contínua (abrangência nacional a partir de 2012) criaram uma quebra metodológica que exige técnicas de "
            "retropolação para análises econométricas de longo prazo."
        ),
        "methodology_explanation": (
            "Utilizamos o período de sobreposição entre PME e PNAD Contínua para estimar via MQO-HAC a relação estrutural "
            "entre ambas as séries, avaliando o grau de aderência (R²) e o erro residual de encadeamento."
        ),
        "interpretation_guide": (
            "Um R² elevado e resíduos estáveis no período comum validam a ponte estatística para estender a série nacional "
            "para anos anteriores a 2012."
        ),
        "references": [
            "IBGE — Pesquisa Nacional por Amostra de Domicílios Contínua (PNAD Contínua): Notas Técnicas.",
        ],
        "dataset_keys": ["pnad_pme"],
        "bcb_series": [[24369, "Taxa de Desocupação PNAD (%)"], [24379, "Pessoas Ocupadas PNAD (mil)"]],
    },
    {
        "code": "M24",
        "slug": "caged-vs-pnad-cointegracao",
        "title": "Emprego Formal (CAGED) vs. PNAD Contínua: Dinâmica e Cointegração",
        "subtitle": "O saldo mensal de empregos formais do CAGED antecipa os movimentos da taxa de desocupação da PNAD?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração de Engle-Granger & Modelo de Correção de Erros (ECM)",
        "economic_intuition": (
            "O CAGED é um registro administrativo censitário do emprego celetista, enquanto a PNAD Contínua é uma pesquisa "
            "domiciliar que captura também informalidade e força de trabalho. Comparar ambas revela se a recuperação do "
            "mercado de trabalho ocorre pela via formal ou por ocupações por conta própria."
        ),
        "methodology_explanation": (
            "Aplicamos o teste de cointegração de Engle-Granger e o mecanismo de correção de erros (ECM) para avaliar o "
            "acoplamento de longo prazo e a velocidade de ajuste entre emprego formal e ocupação agregada."
        ),
        "interpretation_guide": (
            "Acompanhe o parâmetro de equilíbrio de longo prazo e os desvios do termo ECT: divergências prolongadas costumam "
            "sinalizar ciclos de formalização ou informalização acelerada."
        ),
        "references": [
            "IPEA — Carta de Conjuntura: Mercado de Trabalho e Compatibilização CAGED/PNAD.",
        ],
        "dataset_keys": ["caged_pnad"],
        "bcb_series": [[28763, "Saldo Novo CAGED"], [24369, "Taxa de Desocupação PNAD (%)"], [24364, "IBC-Br"]],
    },
    {
        "code": "M25",
        "slug": "projecao-desemprego-ecm-vecm",
        "title": "Projeção da Taxa de Desemprego com Equilíbrio de Longo Prazo (ECM)",
        "subtitle": "Como a Lei de Okun e os indicadores de atividade ancoram a trajetória futura da desocupação?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "coint_vecm",
        "methodology_label": "Cointegração (Engle-Granger) & Correção de Erros (ECM/VECM)",
        "economic_intuition": (
            "Pela Lei de Okun, o desemprego responde com defasagem às oscilações da atividade econômica e das condições "
            "financeiras. Modelos de correção de erros capturam simultaneamente a relação estrutural de longo prazo e a "
            "inércia de curto prazo típica das contratações e demissões."
        ),
        "methodology_explanation": (
            "Estimamos a equação cointegrante e a dinâmica de correção de erros entre a taxa de desemprego e seus "
            "determinantes macroeconômicos, reportando estatística de cointegração, β de longo prazo e velocidade α_ECT."
        ),
        "interpretation_guide": (
            "Quanto maior em módulo for α_ECT (com sinal negativo), mais rápido o mercado de trabalho corrige desvios em "
            "relação ao nível compatível com a atividade econômica."
        ),
        "references": [
            "Okun, A. M. (1962). Potential GNP: Its Measurement and Significance. ASA Proceedings.",
        ],
        "dataset_keys": ["desemprego_vecm"],
        "bcb_series": [[24369, "Taxa de Desocupação PNAD (%)"], [24364, "IBC-Br"], [4189, "Selic (% a.a.)"]],
    },
    {
        "code": "M26",
        "slug": "emprego-formal-privado-ciclo",
        "title": "Emprego Formal Privado (CLT) e Sensibilidade ao Ciclo Econômico",
        "subtitle": "Qual a amplitude cíclica do emprego com carteira assinada nas expansões e recessões brasileiras?",
        "theme": "trabalho",
        "theme_label": "Mercado de Trabalho",
        "specialized": False,
        "model_type": "hp_neutral",
        "methodology_label": "Decomposição Tendência-Ciclo via Filtro Hodrick-Prescott (HP)",
        "economic_intuition": (
            "O emprego formal privado reage de maneira assimétrica ao ciclo: contrai rapidamente em recessões agudas e "
            "demora trimestres para retomar a tendência prévia devido a custos de contratação, demissão e incerteza "
            "quanto à duração da retomada."
        ),
        "methodology_explanation": (
            "Extraímos a componente estrutural e o ciclo do emprego formal via Filtro HP, analisando os picos e vales "
            "cíclicos e a dispersão dos desvios em relação à tendência."
        ),
        "interpretation_guide": (
            "Use o gráfico do componente cíclico para localizar períodos de sobre-emprego (ciclo positivo) ou ociosidade "
            "no mercado formal (ciclo negativo)."
        ),
        "references": [
            "Pissarides, C. A. (2000). Equilibrium Unemployment Theory (2nd ed.). MIT Press.",
        ],
        "dataset_keys": ["emprego_formal"],
        "bcb_series": [[24379, "Pessoas Ocupadas PNAD (mil)"], [28763, "Saldo Novo CAGED"]],
    },

    # =========================================================================
    # PILAR 6: CÂMBIO, SETOR EXTERNO & FINANÇAS
    # =========================================================================
    {
        "code": "M27",
        "slug": "determinantes-taxa-cambio",
        "title": "Determinantes Macroeconômicos da Taxa de Câmbio (R$/US$)",
        "subtitle": "Como o diferencial de juros (Selic) e o risco soberano (CDS 5Y) dirigem a dinâmica do dólar?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "Pela paridade descoberta de juros ajustada pelo risco, a taxa de câmbio responde ao diferencial entre juros "
            "domésticos e externos, ao prêmio de risco soberano (CDS) e aos termos de troca. Embora de difícil previsão "
            "pontual no curtíssimo prazo (enigma de Meese-Rogoff), choques de risco exercem impacto imediato sobre o real."
        ),
        "methodology_explanation": (
            "Combinamos as séries diárias de câmbio, CDS 5Y e Selic em um modelo VAR em primeiras diferenças, analisando "
            "a transmissão de choques via IRF e testes de Causalidade de Granger."
        ),
        "interpretation_guide": (
            "Verifique a resposta contemporânea e defasada do câmbio a choques no CDS/juros; significância em lags curtos "
            "reflete a rápida precificação dos mercados cambiais líquidos."
        ),
        "references": [
            "Meese, R. A., & Rogoff, K. (1983). Empirical Exchange Rate Models of the Seventies. JIE, 14(1–2).",
            "Dornbusch, R. (1976). Expectations and Exchange Rate Dynamics. Journal of Political Economy, 84(6).",
        ],
        "dataset_keys": ["cambio_diario", "cds_cambio", "selic_cambio"],
        "bcb_series": [[3698, "Câmbio R$/US$"], [4189, "Selic (% a.a.)"], [27574, "Commodities IC-Br"]],
    },
    {
        "code": "M28",
        "slug": "volatilidade-condicional-cambio-ewma",
        "title": "Volatilidade Condicional do Câmbio e Risco Global (EWMA)",
        "subtitle": "Como capturar agrupamentos de volatilidade (volatility clustering) e caudas pesadas no mercado cambial?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "volatility",
        "methodology_label": "Econometria Financeira: Retornos Log, Volatilidade EWMA (λ=0,94) & Curtose",
        "economic_intuition": (
            "Séries financeiras e cambiais raramente apresentam variância constante: períodos turbulentos tendem a ser "
            "seguidos por dias igualmente turbulentos (clustering de volatilidade) e choques extremos ocorrem com frequência "
            "muito superior à prevista pela distribuição Normal (caudas pesadas / leptocurtose)."
        ),
        "methodology_explanation": (
            "Calculamos retornos logarítmicos percentuais e estimamos a volatilidade condicional variante no tempo pelo "
            "modelo EWMA (RiskMetrics, λ = 0,94), acompanhada de estatísticas de assimetria e excesso de curtose."
        ),
        "interpretation_guide": (
            "Excesso de curtose > 1 confirma caudas pesadas (risco extremo subestimado por modelos gaussianos simples); "
            "picos na curva EWMA marcam regimes de estresse cambial e aversão global ao risco."
        ),
        "references": [
            "J.P. Morgan / Reuters (1996). RiskMetrics Technical Document (4th ed.).",
            "Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity. Econometrica, 50(4).",
        ],
        "dataset_keys": ["incerteza_cambio"],
        "bcb_series": [[3698, "Câmbio R$/US$"], [11752, "Câmbio Real Efetivo"]],
    },
    {
        "code": "M29",
        "slug": "mercado-capitais-ifix-ibov-macro",
        "title": "Mercado de Capitais (IFIX & Ibovespa): Retornos, Risco e Juros",
        "subtitle": "Como ativos listados sensíveis à curva de juros reagem a ciclos monetários e volatilidade?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "volatility",
        "methodology_label": "Econometria Financeira: Retornos, Volatilidade Condicional EWMA & Correlações",
        "economic_intuition": (
            "Fundos imobiliários (IFIX) e ações (Ibovespa) precificam fluxos de caixa descontados pela taxa de juros real "
            "livre de risco mais prêmio de risco. Ciclos de aperto monetário comprimem múltiplos e elevam a volatilidade "
            "condicional, enquanto cortes críveis da Selic induzem reprecificação positiva."
        ),
        "methodology_explanation": (
            "Analisamos retornos, volatilidade condicional EWMA (λ = 0,94), curtose, assimetria e correlação cruzada "
            "com indicadores macroeconômicos de juros e atividade."
        ),
        "interpretation_guide": (
            "Observe a assimetria negativa típica de ativos de risco (quedas abruptas mais intensas que altas graduais) "
            "e o comportamento da volatilidade em viradas do ciclo da Selic."
        ),
        "references": [
            "Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). The Econometrics of Financial Markets. Princeton Univ. Press.",
        ],
        "dataset_keys": ["ifix_macro"],
        "bcb_series": [[7, "Ibovespa (pts)"], [4189, "Selic (% a.a.)"], [3698, "Câmbio R$/US$"]],
    },
    {
        "code": "M30",
        "slug": "poupanca-investimento-causalidade",
        "title": "Poupança Doméstica e Investimento: Causalidade e Restrição Externa",
        "subtitle": "No Brasil, elevar a poupança precede o aumento do investimento ou o crescimento do investimento gera poupança?",
        "theme": "cambio_externo",
        "theme_label": "Câmbio & Finanças",
        "specialized": False,
        "model_type": "var_granger",
        "methodology_label": "Vetor Autorregressivo (VAR), Impulso-Resposta (IRF) & Causalidade de Granger",
        "economic_intuition": (
            "O debate clássico entre visão neoclássica (poupança financia e precede o investimento) e visão keynesiana "
            "(investimento expande a renda e gera poupança ex-post), somado ao enigma de Feldstein-Horioka sobre "
            "mobilidade internacional de capitais, permanece central para políticas de crescimento de longo prazo."
        ),
        "methodology_explanation": (
            "Estimamos um modelo VAR em primeiras diferenças entre as taxas de poupança e investimento, aplicando testes "
            "de Causalidade de Granger em ambas as direções e Funções Impulso-Resposta em 10 períodos."
        ),
        "interpretation_guide": (
            "Compare os p-valores de Granger nas duas direções: precedência temporal significativa ajuda a avaliar qual "
            "variável lidera o ajuste macroeconômico de curto prazo."
        ),
        "references": [
            "Feldstein, M., & Horioka, C. (1980). Domestic Saving and International Capital Flows. Economic Journal, 90(358).",
        ],
        "dataset_keys": ["poupanca_invest"],
        "bcb_series": [[24364, "IBC-Br"], [4380, "PIB Mensal (R$ mi)"], [3698, "Câmbio R$/US$"]],
    },
]

STUDIES_CATALOG.extend(STUDIES_CATALOG_EXTRA)
STUDIES_BY_CODE: Dict[str, Dict[str, Any]] = {s["code"]: s for s in STUDIES_CATALOG}
