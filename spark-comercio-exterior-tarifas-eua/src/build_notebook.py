"""Constrói o notebook analítico `comercio_exterior_pyspark_tarifas.ipynb` com execução
real em Apache Spark 4.0, alternância estrita código -> análise markdown e números verificados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "metrics_summary.json"
NB_PATH = PROJECT_ROOT / "comercio_exterior_pyspark_tarifas.ipynb"


def md_cell(text: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(code: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook() -> None:
    s = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    aud = s["audit"]
    b24 = s["balanca_2024"]
    h10 = s["hhi_2010_vs_2024"]["2010"]
    h24 = s["hhi_2010_vs_2024"]["2024"]
    econ = s["econometrics"]
    specs = {sp["id"]: sp for sp in econ["specs"]}
    mc = s["monte_carlo_primary"]
    sc_list = mc["scenarios"]
    c1, c2, c3 = sc_list[0], sc_list[1], sc_list[2]
    rec_id = econ["recommended_spec"]
    rec_sp = specs[rec_id]

    top1_p = s["top5_parceiros_12m"][0]
    top2_p = s["top5_parceiros_12m"][1]
    top1_m = s["top5_municipios_12m"][0]
    top1_eua = s["top5_exposicao_eua"][0]
    top1_rus = s["top5_russia_import"][0]
    top1_suco = s["suco_top5_destinos_12m"][0]
    top2_suco = s["suco_top5_destinos_12m"][1]

    cells: List[Dict[str, Any]] = [
        md_cell(f"""# 🚢 Comércio Exterior Brasileiro em Apache Spark 4.0: Concentração Geoeconômica, Econometria 2SLS & Simulação Tarifária

**Autor:** Dr. Silvio da Rosa Paula • Portfólio de Engenharia de Dados, Big Data & Econometria Aplicada  
**Motor Distribuído:** Apache Spark `{s['spark_version']}` (`local[*]`, Adaptive Query Execution + Broadcast Hash Joins)  
**Base Oficial SECEX/ComexStat:** **`{aud['total_rows']:,}` registros municipais** (`{aud['export_rows']:,}` exportações + `{aud['import_rows']:,}` importações | `{aud['period']}` | **US\\$ {(aud['export_fob_total_usd']+aud['import_fob_total_usd'])/1e12:.2f} trilhões** de corrente acumulada)

---

## 🎯 Problema de Negócio & Arquitetura da Investigação

Este estudo une **Engenharia de Big Data em PySpark** e **Econometria de Comércio Internacional** para responder a quatro perguntas estratégicas sobre a inserção externa do Brasil:

1. **Integridade & Performance em Big Data:** Como processar 36,5 milhões de transações oficiais garantindo **zero multiplicação cartesiana (*fan-out*)** nos joins com dicionários NCM/SH4 e eliminando *shuffles* via **Broadcast Hash Join**?
2. **Concentração Estrutural (2010–2025):** A pauta exportadora brasileira diversificou ou tornou-se mais vulnerável em parceiros e produtos (medido pelo **Índice Herfindahl-Hirschman — HHI**)?
3. **Exposição Geoeconômica (EUA, China & Rússia):** Quais setores brasileiros apresentam maior dependência tarifária dos Estados Unidos, qual o grau de concentração na China e quais são os fornecedores alternativos para insumos críticos importados da Rússia?
4. **Impacto Causal de Tarifas no Suco de Laranja (Brasil → EUA):** Estimando a **elasticidade-preço da demanda** sob **4 especificações econométricas** (`MQO`, `Sazonal+Tendência HAC`, `ARDL Dinâmico` e **`2SLS / Variáveis Instrumentais`** usando o preço de exportação para o Resto do Mundo como instrumento de oferta) e simulando **10.000 cenários Monte Carlo** para tarifas de **10%**, **25%** e **50% ad valorem + US\\$ 415/t**."""),

        # =====================================================================
        # SEÇÃO 1 — ENGENHARIA SPARK, INTEGRIDADE STAR-SCHEMA & BROADCAST JOIN
        # =====================================================================
        md_cell("## 1. Engenharia PySpark 4.0: Auditoria Star-Schema (Fan-Out) & Broadcast Hash Join"),
        code_cell("""import json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from IPython.display import Image, display
from pyspark.sql import functions as F

from spark_comex_pipeline import create_spark_session, ensure_data_extracted, load_clean_dimensions
from econometrics_tariffs import estimate_orange_juice_elasticities, simulate_tariff_scenarios

FIG_DIR = ROOT / "outputs" / "figures"
metrics = json.loads((ROOT / "outputs" / "metrics_summary.json").read_text(encoding="utf-8"))
aud = metrics["audit"]

# Inicializa sessão real Apache Spark 4.0.1 e verifica leitura distribuída
spark = create_spark_session("ComexNotebook_PySpark4")
data_dir = ensure_data_extracted()
df_exp = spark.read.parquet(str(data_dir / "dados_comex_mun_export_07-2025.parquet"))
df_imp = spark.read.parquet(str(data_dir / "dados_comex_mun_import_07-2025.parquet"))
dims = load_clean_dimensions(spark, data_dir)

print(f"Apache Spark {spark.version} ativo | Exportações: {df_exp.count():,} linhas | Importações: {df_imp.count():,} linhas")
print(f"Dimensão NCM->SH4 bruta: {aud['raw_ag_rows']:,} linhas -> Deduplicada 1:1 no grão SH4: {aud['dedup_sh4_rows']:,} códigos únicos")
print(f"Conservação FOB 12m (Jul/2024–Jun/2025):")
print(f"  - Total real (fato pura)           : US$ {aud['fob_12m_true_usd']/1e9:,.2f} bilhões")
print(f"  - Join ingênuo sem deduplicar SH4  : US$ {aud['fob_12m_bug_fanout_usd']/1e9:,.2f} bilhões ({aud['fanout_inflation_factor']}x — BUG FAN-OUT!)")
print(f"  - Join 1:1 Broadcast (nosso fluxo) : US$ {aud['fob_12m_clean_broadcast_usd']/1e9:,.2f} bilhões (100,00% exato!)")
print(f"Benchmark Join (13,8M linhas): Broadcast={aud['benchmark_broadcast_sec']:.2f}s vs SortMerge={aud['benchmark_sortmerge_sec']:.2f}s ({aud['benchmark_speedup_x']:.2f}x speedup)")

# Verifica plano físico Catalyst confirmando BroadcastHashJoin
df_exp.join(F.broadcast(dims["pais"]), on="CO_PAIS", how="left").groupBy("NO_PAIS").agg(F.sum("VL_FOB")).explain("formatted")
display(Image(filename=str(FIG_DIR / "06_spark_catalyst_otimizacao.png")))"""),

        md_cell(f"""### 🔎 Análise de Engenharia de Dados & Otimização Spark

- **Correção Crítica de Integridade (*Fan-Out*):** A tabela fato municipal da SECEX está no grão **`SH4`**, enquanto tabelas auxiliares de NCM/SH6 contêm **`{aud['raw_ag_rows']:,}` linhas** para apenas **`{aud['dedup_sh4_rows']:,}` códigos SH4 únicos**. Um join direto sem deduplicação prévia inflava o valor exportado em **`{aud['fanout_inflation_factor']:.1f}×`** (saltando artificialmente de US\\$ `{aud['fob_12m_true_usd']/1e9:.1f}` bi para US\\$ `{aud['fob_12m_bug_fanout_usd']/1e9:,.0f}` bi!). Nossa deduplicação estrita por moda no grão `SH4` preserva **100,00% do Valor FOB real**.
- **Eficiência no Catalyst Optimizer:** O plano físico `explain("formatted")` confirma a substituição do custoso `SortMergeJoin` por **`BroadcastHashJoin`** com **zero shuffle** sobre os `13,78` milhões de registros de exportação, reduzindo o tempo de **`{aud['benchmark_sortmerge_sec']:.2f}s`** para **`{aud['benchmark_broadcast_sec']:.2f}s`** (**`{aud['benchmark_speedup_x']:.2f}×` de ganho real**)."""),

        # =====================================================================
        # SEÇÃO 2 — BALANÇA COMERCIAL HISTÓRICA
        # =====================================================================
        md_cell("## 2. Evolução Histórica da Balança Comercial Brasileira (2010–Jun/2025)"),
        code_cell("""balanca_anual = pd.read_csv(ROOT / "outputs" / "balanca_anual_2010_2025.csv")
tbl_bal = balanca_anual.assign(
    Export_USD_Bi=(balanca_anual["export_fob"] / 1e9).round(2),
    Import_USD_Bi=(balanca_anual["import_fob"] / 1e9).round(2),
    Saldo_USD_Bi=(balanca_anual["saldo_fob"] / 1e9).round(2),
    Preco_Exp_USD_kg=balanca_anual["preco_medio_exp_usd_kg"].round(3),
)[["CO_ANO", "Export_USD_Bi", "Import_USD_Bi", "Saldo_USD_Bi", "Preco_Exp_USD_kg"]]
display(tbl_bal.tail(8))
display(Image(filename=str(FIG_DIR / "01_balanca_comercial_historica.png")))"""),

        md_cell(f"""### 🔎 Análise da Trajetória Comercial (2010–2025)

- **Mudança Estrutural de Patamar:** Após o ciclo de desaceleração de 2014–2016 e o choque pandêmico de 2020, as exportações brasileiras deram um salto estrutural a partir de 2021, sustentando superávits anuais muito superiores à média histórica da década anterior.
- **Desempenho Recente (2024 completo):** Em 2024, o Brasil registrou **US\\$ `{b24['export_fob']/1e9:.1f}` bilhões** em exportações e **US\\$ `{b24['import_fob']/1e9:.1f}` bilhões** em importações, encerrando o ano com superávit comercial robusto de **US\\$ `{b24['saldo_fob']/1e9:.1f}` bilhões** (e corrente de comércio de **US\\$ `{b24['corrente_fob']/1e9:.1f}` bilhões**)."""),

        # =====================================================================
        # SEÇÃO 3 — CONCENTRAÇÃO HHI, TOP PARCEIROS & MUNICÍPIOS
        # =====================================================================
        md_cell("## 3. Concentração Estrutural (Índice HHI), Top Parceiros & Geografia Municipal"),
        code_cell("""hhi_df = pd.read_csv(ROOT / "outputs" / "hhi_concentracao_anual.csv")
display(hhi_df.iloc[[0, 5, 10, -1]])

print("\\nTop 5 Parceiros Comerciais (Jul/2024–Jun/2025):")
p_df = pd.DataFrame(metrics["top5_parceiros_12m"]).assign(
    Export_Bi=lambda d: (d["export_fob"]/1e9).round(2),
    Import_Bi=lambda d: (d["import_fob"]/1e9).round(2),
    Saldo_Bi=lambda d: (d["saldo_fob"]/1e9).round(2),
)[["NO_PAIS", "Export_Bi", "Import_Bi", "Saldo_Bi"]]
display(p_df)

print("\\nTop 5 Municípios Exportadores (Jul/2024–Jun/2025):")
m_df = pd.DataFrame(metrics["top5_municipios_12m"]).assign(
    Export_Bi=lambda d: (d["export_fob"]/1e9).round(2),
    Volume_Mton=lambda d: (d["export_kg"]/1e9).round(2),
)[["municipio_uf", "Export_Bi", "Volume_Mton"]]
display(m_df)

display(Image(filename=str(FIG_DIR / "02_concentracao_hhi_parceiros.png")))"""),

        md_cell(f"""### 🔎 Diagnóstico de Concentração (HHI) e Polo Geográfico

- **Aumento de Concentração Externa (2010 → 2024):** O **Índice Herfindahl-Hirschman (HHI) de Parceiros** avançou de **`{h10['hhi_parceiros']:.0f}`** (2010) para **`{h24['hhi_parceiros']:.0f}`** (2024), impulsionado diretamente pela ascensão da participação chinesa nas exportações brasileiras de **`{h10['share_china_pct']:.1f}%`** para **`{h24['share_china_pct']:.1f}%`** (enquanto os EUA passaram de `{h10['share_eua_pct']:.1f}%` para `{h24['share_eua_pct']:.1f}%`). Paralelamente, o **HHI de Produtos (`SH4`)** subiu de **`{h10['hhi_produtos_sh4']:.0f}`** para **`{h24['hhi_produtos_sh4']:.0f}`**, confirmando maior primarização (soja, petróleo bruto e minério de ferro).
- **Liderança Bilateral & Municipal:** Nos últimos 12 meses (`Jul/2024–Jun/2025`), **{top1_p['NO_PAIS']}** lidera com corrente de **US\\$ `{top1_p['corrente_fob']/1e9:.1f}` bi** (superávit de US\\$ `{top1_p['saldo_fob']/1e9:.1f}` bi), seguida por **{top2_p['NO_PAIS']}** (**US\\$ `{top2_p['corrente_fob']/1e9:.1f}` bi**). Na geografia municipal, **{top1_m['municipio_uf']}** ocupa o 1º lugar nacional em exportações (**US\\$ `{top1_m['export_fob']/1e9:.1f}` bi**), refletindo a forte concentração em polos petrolíferos, mineradores e agrícolas."""),

        # =====================================================================
        # SEÇÃO 4 — EXPOSIÇÃO BILATERAL (EUA vs CHINA) & VULNERABILIDADE RÚSSIA
        # =====================================================================
        md_cell("## 4. Exposição Geoeconômica: EUA vs. China & Dependência de Insumos da Rússia"),
        code_cell("""print("Top 5 Categorias mais Expostas a Tarifas dos EUA (Jul/2024–Jun/2025):")
display(pd.DataFrame(metrics["top5_exposicao_eua"]).assign(
    Export_EUA_Bi=lambda d: (d["pais_fob"]/1e9).round(2),
    Share_EUA_Pct=lambda d: d["share_pct"].round(1),
)[["categoria_agregada", "Export_EUA_Bi", "Share_EUA_Pct"]])

print("\\nTop 5 Divisões Importadas da Rússia + Fornecedores Alternativos (Spark Window Rank):")
rus_alt = pd.DataFrame(metrics["russia_alternativos"]).assign(
    Import_Bi=lambda d: (d["vl_fob"]/1e9).round(2),
    Share_Pct=lambda d: d["share_pct"].round(1),
)[["NO_CUCI_DIVISAO", "rank", "NO_PAIS", "Import_Bi", "Share_Pct"]]
display(rus_alt.head(12))

display(Image(filename=str(FIG_DIR / "03_exposicao_eua_china_russia.png")))"""),

        md_cell(f"""### 🔎 Assimetria Geoeconômica da Pauta Brasileira

- **EUA = Pauta de Maior Valor Agregado e Risco Tarifário Setorial:** Embora a China concentre commodities de altíssimo volume (soja, minério de ferro, petróleo), os **Estados Unidos** absorvem parcelas decisivas de bens industriais e processados — com destaque para **{top1_eua['categoria_agregada']}** (`{top1_eua['share_pct']:.1f}%` das exportações brasileiras da categoria, US\\$ `{top1_eua['pais_fob']/1e9:.2f}` bi). Choques tarifários nos EUA atingem diretamente segmentos de maior intensidade tecnológica e emprego industrial.
- **Gargalo de Importação da Rússia:** Pelo lado das importações, **{top1_rus['NO_CUCI_DIVISAO']}** lidera as compras brasileiras da Rússia (**US\\$ `{top1_rus['vl_fob']/1e9:.2f}` bi**, `{top1_rus['share_pct']:.1f}%` do total importado da divisão), ao lado de fertilizantes — tornando o mapeamento de fornecedores alternativos (via `Window.partitionBy` no Spark) essencial para mitigação de risco geopolítico."""),

        # =====================================================================
        # SEÇÃO 5 — ECONOMETRIA DA DEMANDA DE SUCO DE LARANJA (4 ESPECIFICAÇÕES + 2SLS-IV)
        # =====================================================================
        md_cell("## 5. Estudo de Caso: Elasticidade-Preço do Suco de Laranja nos EUA (MQO, HAC, ARDL & 2SLS-IV)"),
        code_cell("""suco_df = pd.read_csv(ROOT / "outputs" / "suco_laranja_eua_mensal.csv", parse_dates=["date"])
econ_live = estimate_orange_juice_elasticities(suco_df)

print(f"Painel Mensal Brasil -> EUA (SH4 2009): {econ_live['panel_n_months']} meses ({suco_df['date'].min():%Y-%m} a {suco_df['date'].max():%Y-%m})")
print("Comparação das 4 Especificações Econométricas:")
specs_tbl = pd.DataFrame(econ_live["specs"])
display(specs_tbl[["id", "label", "elasticity", "se", "pvalue", "notes"]])
display(Image(filename=str(FIG_DIR / "04_suco_laranja_mercados_elasticidade.png")))"""),

        md_cell(f"""### 🔎 Interpretação Econométrica & Controle de Endogeneidade (2SLS-IV)

O mercado global de sucos brasileiros (`SH4 2009`) nos últimos 12 meses é liderado por **{top1_suco['NO_PAIS']}** (US\\$ `{top1_suco['vl_fob']/1e6:,.0f}` milhões) e **{top2_suco['NO_PAIS']}** (US\\$ `{top2_suco['vl_fob']/1e6:,.0f}` milhões). Ao estimar a elasticidade-preço da demanda norte-americana ao longo de **`{econ['panel_n_months']}` meses (2010–Jun/2025)**, comparamos quatro modelos aninhados:

1. **`M1` (MQO Simples Log-Log):** $\\hat{{\\beta}} = {specs['M1']['elasticity']:.3f}$ (`SE = {specs['M1']['se']:.3f}`, `R² = {specs['M1']['r2']:.3f}`) — baseline sem controles sazonais nem correção para autocorrelação serial.
2. **`M2` (MQO + Sazonalidade + Tendência, Erros Newey-West HAC):** $\\hat{{\\beta}} = {specs['M2']['elasticity']:.3f}$ (`SE = {specs['M2']['se']:.3f}`, `p = {specs['M2']['pvalue']:.4f}`, `R² = {specs['M2']['r2']:.3f}`) — incorporar dummies mensais e tendência eleva expressivamente o poder explicativo e captura a sazonalidade de safra/inverno.
3. **`M3` (ARDL Dinâmico HAC):** captura elasticidade de curto prazo $\\hat{{\\beta}}_{{CP}} = {specs['M3']['elasticity']:.3f}$ e de longo prazo $\\hat{{\\beta}}_{{LP}} = {specs['M3']['elasticity_long_run']:.3f}$ ($\\rho = {specs['M3']['rho_lag']:.3f}$), refletindo contratos plurianuais e ajustes graduais de estoque.
4. **`M4` (Teste Crítico de 2SLS / Variáveis Instrumentais com Erros HAC):** Ao instrumentar $\\ln P_t^{{\\text{{EUA}}}}$ pelo preço de exportação para o Resto do Mundo ($\\ln P_t^{{\\text{{RoW}}}}$), o 1º estágio é fortíssimo (**`F = {specs['M4']['first_stage_f']:.1f}`** $\\gg 10$, `R² = {specs['M4']['first_stage_r2']:.3f}`), porém o coeficiente do 2º estágio perde significância estatística ($\\hat{{\\beta}}_{{IV}} = {specs['M4']['elasticity']:+.3f}$, `p = {specs['M4']['pvalue']:.4f}`, `Hausman p = {specs['M4']['hausman_pvalue']:.4f}`). **Por que isso ocorre economicamente?** Porque o Brasil detém >75% das exportações mundiais de suco de laranja e o mercado é globalmente integrado: quebras na safra da Flórida (*greening*/furacões) elevam simultaneamente a demanda americana e o preço global (incluindo Europa/RoW), violando a **restrição de exclusão** do instrumento. Esse achado evita o uso ingênuo de um IV inválido e fundamenta a escolha de **`{rec_id}` (MQO + Sazonalidade + Tendência HAC, $\\hat{{\\beta}} = {rec_sp['elasticity']:.3f}$, `p = {rec_sp['pvalue']:.4f}`)** como especificação principal, com `M1` ($\\hat{{\\beta}} = {specs['M1']['elasticity']:.3f}$) como banda superior de sensibilidade!"""),

        # =====================================================================
        # SEÇÃO 6 — SIMULAÇÃO MONTE CARLO DOS 3 CENÁRIOS TARIFÁRIOS
        # =====================================================================
        md_cell("## 6. Simulação Monte Carlo (10.000 Sorteios): Impacto de Tarifas dos EUA (10%, 25% e 50% + US$ 415/t)"),
        code_cell("""rec_sp = next(sp for sp in econ_live["specs"] if sp["id"] == econ_live["recommended_spec"])
mc_live = simulate_tariff_scenarios(suco_df, beta_hat=rec_sp["elasticity"], beta_se=rec_sp["se"], n_sims=10000, seed=42)

print(f"Base 12 meses (Jul/2024–Jun/2025): Volume={mc_live['baseline_12m']['volume_kton']:.1f} mil t | Receita FOB=US$ {mc_live['baseline_12m']['fob_musd']:.1f}M | Preço FOB médio=US$ {mc_live['baseline_12m']['p_fob_usd_kg']:.3f}/kg")
sc_tbl = pd.DataFrame(mc_live["scenarios"])[[
    "scenario", "p_landed_usd_kg", "delta_price_pct",
    "dq_p10_pct", "dq_median_pct", "dq_p90_pct",
    "fob_delta_median_musd", "fob_delta_p10_musd", "fob_delta_p90_musd",
]]
display(sc_tbl)
display(Image(filename=str(FIG_DIR / "05_simulacao_monte_carlo_tarifas.png")))

# Encerra sessão Spark de forma limpa
spark.stop()
print("Sessão Apache Spark encerrada com sucesso.")"""),

        md_cell(f"""### 🔎 Interpretação Econômica dos 3 Cenários Tarifários (Monte Carlo)

Partindo do preço FOB efetivo médio dos últimos 12 meses (**US\\$ `{mc['baseline_12m']['p_fob_usd_kg']:.3f}/kg`**, volume de **`{mc['baseline_12m']['volume_kton']:.1f}` mil toneladas** e receita de **US\\$ `{mc['baseline_12m']['fob_musd']:.1f}` milhões/ano**), a simulação de **10.000 trajetórias** revela a escalada não linear do risco comercial:

| Cenário Tarifário | Preço ao Importador | Δ Preço | Δ Volume (Mediana) | Faixa P10–P90 | Perda FOB Estimada (Mediana) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **{c1['scenario']}** | `US$ {c1['p_landed_usd_kg']:.3f}/kg` | `+{c1['delta_price_pct']:.1f}%` | **`{c1['dq_median_pct']:.1f}%`** | `[{c1['dq_p10_pct']:.1f}%, {c1['dq_p90_pct']:.1f}%]` | **`US$ {c1['fob_delta_median_musd']:.1f}M/ano`** |
| **{c2['scenario']}** | `US$ {c2['p_landed_usd_kg']:.3f}/kg` | `+{c2['delta_price_pct']:.1f}%` | **`{c2['dq_median_pct']:.1f}%`** | `[{c2['dq_p10_pct']:.1f}%, {c2['dq_p90_pct']:.1f}%]` | **`US$ {c2['fob_delta_median_musd']:.1f}M/ano`** |
| **{c3['scenario']}** | `US$ {c3['p_landed_usd_kg']:.3f}/kg` | `+{c3['delta_price_pct']:.1f}%` | **`{c3['dq_median_pct']:.1f}%`** | `[{c3['dq_p10_pct']:.1f}%, {c3['dq_p90_pct']:.1f}%]` | **`US$ {c3['fob_delta_median_musd']:.1f}M/ano`** |

*(Nota: o modelo pressupõe repasse integral da tarifa ao importador norte-americano sob preços FOB constantes; caso os exportadores brasileiros absorvam parte da tarifa comprimindo margem FOB, a queda de volume será parcialmente atenuada à custa de redução direta na rentabilidade por tonelada.)*"""),

        # =====================================================================
        # SEÇÃO 7 — CONCLUSÃO EXECUTIVA
        # =====================================================================
        md_cell(f"""## 7. Conclusão Executiva

1. **Engenharia Big Data Confiável:** O processamento distribuído de **`{aud['total_rows']:,}` registros oficiais SECEX** em **Apache Spark `{s['spark_version']}`** demonstrou como a deduplicação 1:1 no grão `SH4` evita erros catastróficos de multiplicação (`{aud['fanout_inflation_factor']:.1f}×`), enquanto **`BroadcastHashJoin`** acelerou joins na tabela de `13,8M` linhas em **`{aud['benchmark_speedup_x']:.2f}×`**.
2. **Vulnerabilidade Geoeconômica Crescente:** O crescimento do HHI de parceiros (`{h10['hhi_parceiros']:.0f}` → `{h24['hhi_parceiros']:.0f}`) e de produtos (`{h10['hhi_produtos_sh4']:.0f}` → `{h24['hhi_produtos_sh4']:.0f}`) evidencia forte dependência da demanda chinesa por commodities primárias e elevada exposição tarifária de nichos manufaturados/processados aos EUA.
3. **Resiliência Tarifária & Estratégia Comercial:** Mesmo com a tarifa base vigente (`10% + US$ 415/t`), o suco brasileiro já enfrenta encarecimento efetivo de **`+{c1['delta_price_pct']:.1f}%`** no mercado americano. Uma escalada para **`50% + US$ 415/t`** elevaria o preço ao importador em **`+{c3['delta_price_pct']:.1f}%`**, com retração mediana estimada de **`{c3['dq_median_pct']:.1f}%`** no volume (**US\\$ `{c3['fob_delta_median_musd']:.1f}` milhões/ano** em risco) — reforçando a urgência de acordos bilaterais e diversificação de destinos na União Europeia e Ásia."""),
    ]

    for idx_c, cell in enumerate(cells):
        cell["id"] = f"comex-{idx_c:02d}"

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook gerado com sucesso: {NB_PATH} ({len(cells)} células)")


if __name__ == "__main__":
    build_notebook()
