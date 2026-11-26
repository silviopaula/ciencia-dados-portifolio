"""Pipeline ETL PySpark modular (Databricks Unity Catalog — Arquitetura Medallion).

Constrói o painel econométrico balanceado `UF × cargo × ano` (1998–2022) e a tabela
agregada `painel_causal_impact` para avaliação do impacto causal da Lei nº 12.034/2009
(Minirreforma Eleitoral — cota efetiva de 30% de candidaturas femininas em eleições
proporcionais).

Camadas Unity Catalog (`<catalog>.<schema>`):
- Bronze (entrada): `candidatos`, `candidatos_eleitos`, `missing_gender_human_input`, `explicativas`
- Silver (saneada): `candidatos_genero_ok`
- Gold (analítica): `panel_data`, `painel_causal_impact`

Uso no Databricks Workflows / Asset Bundle:
    python src/databricks_etl_pipeline.py --catalog workspace --schema avaliacao_lei_tse
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

# Mapeamento oficial UF -> Macrorregião IBGE
UF_TO_REGION: Dict[str, str] = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

PROPORTIONAL_OFFICES = (
    "vereador",
    "deputado federal",
    "deputado estadual",
    "deputado distrital",
)
MAJORITARIAN_OFFICES = ("senador", "governador", "prefeito")


def load_bronze_tables(
    spark: SparkSession,
    catalog: str = "workspace",
    schema: str = "avaliacao_lei_tse",
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Lê as 4 tabelas Bronze do Unity Catalog no Databricks."""
    prefix = f"{catalog}.{schema}"
    df_candidatos = spark.table(f"{prefix}.candidatos")
    df_eleitos = spark.table(f"{prefix}.candidatos_eleitos")
    df_human_gender = spark.table(f"{prefix}.missing_gender_human_input")
    df_explicativas = spark.table(f"{prefix}.explicativas")

    if "eleitores total" in df_explicativas.columns:
        df_explicativas = df_explicativas.withColumnRenamed("eleitores total", "eleitores_total")

    return df_candidatos, df_eleitos, df_human_gender, df_explicativas


def merge_elected_candidates(df_candidatos: DataFrame, df_candidatos_eleitos: DataFrame) -> DataFrame:
    """Padroniza chaves e faz left-join entre candidaturas e resultados eleitorais do TSE."""
    df_eleitos = (
        df_candidatos_eleitos
        .filter((F.col("ano") >= 1992) & F.col("resultado").rlike("(?i)^eleit"))
        .select(
            F.col("ano").cast("int").alias("ano"),
            F.upper(F.trim(F.col("sigla_uf"))).alias("sigla_uf_eleito"),
            F.trim(F.col("cargo")).alias("cargo_eleito"),
            F.trim(F.col("titulo_eleitoral_candidato")).alias("titulo_eleitoral"),
            F.lit("eleito").alias("resultado"),
            F.col("votos"),
        )
        .dropDuplicates(["titulo_eleitoral", "ano", "sigla_uf_eleito", "cargo_eleito"])
    )

    df_cand_clean = (
        df_candidatos
        .withColumn("ano", F.col("ano").cast("int"))
        .withColumn("sigla_uf", F.upper(F.trim(F.regexp_replace(F.col("sigla_uf"), "\u00A0", " "))))
        .withColumn(
            "sigla_uf_nascimento",
            F.upper(F.trim(F.regexp_replace(F.col("sigla_uf_nascimento"), "\u00A0", " "))),
        )
        .withColumn("titulo_eleitoral", F.trim(F.col("titulo_eleitoral")))
    )

    return (
        df_cand_clean
        .join(
            df_eleitos.select("ano", "titulo_eleitoral", "resultado").dropDuplicates(["ano", "titulo_eleitoral"]),
            on=["ano", "titulo_eleitoral"],
            how="left",
        )
        .fillna({"resultado": "nao_eleito"})
    )


def impute_missing_gender_1996(df_candidatos: DataFrame, missing_gender_human_input: DataFrame) -> DataFrame:
    """Corrige ausência histórica de gênero (100% faltante em 1996) em dois estágios:
    1) Inferência probabilística pela moda do primeiro nome em pleitos rotulados;
    2) Complementação por validação humana nominal (`missing_gender_human_input`).

    Resultado auditado no workspace Databricks: 12.537 registros ausentes -> 0 ausentes.
    """
    df_labeled = (
        df_candidatos
        .withColumn("primeiro_nome", F.lower(F.split(F.trim(F.col("nome")), " ").getItem(0)))
        .withColumn("genero_l", F.lower(F.col("genero")))
        .filter(
            F.col("primeiro_nome").isNotNull() & (F.col("primeiro_nome") != "")
            & F.col("genero_l").isNotNull() & (F.col("genero_l") != "")
        )
    )

    w_name = Window.partitionBy("primeiro_nome")
    w_rank = Window.partitionBy("primeiro_nome").orderBy(F.col("p").desc(), F.col("n").desc())

    amostra_genero = (
        df_labeled
        .groupBy("primeiro_nome", "genero_l")
        .agg(F.count("*").alias("n"))
        .withColumn("p", F.col("n") / F.sum("n").over(w_name))
        .withColumn("rk", F.row_number().over(w_rank))
        .filter(F.col("rk") == 1)
        .select("primeiro_nome", F.col("genero_l").alias("genero_inferido"))
    )

    df_step1 = (
        df_candidatos
        .withColumn("primeiro_nome", F.lower(F.split(F.trim(F.col("nome")), " ").getItem(0)))
        .join(F.broadcast(amostra_genero), on="primeiro_nome", how="left")
        .withColumn(
            "genero",
            F.when(F.col("genero").isNull() | (F.trim(F.col("genero")) == ""), F.col("genero_inferido"))
             .otherwise(F.col("genero")),
        )
        .drop("genero_inferido", "primeiro_nome")
    )

    human_input = (
        missing_gender_human_input
        .select(
            F.trim(F.regexp_replace(F.col("nome"), "\u00A0", " ")).alias("nome_norm"),
            F.lower(F.col("genero_input_humano")).alias("genero_input_humano"),
        )
        .dropDuplicates(["nome_norm"])
    )

    return (
        df_step1
        .withColumn("nome_norm", F.trim(F.regexp_replace(F.col("nome"), "\u00A0", " ")))
        .join(F.broadcast(human_input), on="nome_norm", how="left")
        .withColumn(
            "genero",
            F.when(F.col("genero").isNull() | (F.trim(F.col("genero")) == ""), F.col("genero_input_humano"))
             .otherwise(F.col("genero")),
        )
        .drop("genero_input_humano", "nome_norm")
    )


def enrich_region_and_filter_valid(df_genero_fix: DataFrame) -> DataFrame:
    """Valida UFs, cria coluna de macrorregião, filtra candidaturas válidas e padroniza cargos (Camada Silver)."""
    df_uf = df_genero_fix.withColumn("sigla_uf", F.coalesce(F.col("sigla_uf"), F.col("sigla_uf_nascimento")))

    ufs_invalidas = df_uf.filter(~F.col("sigla_uf").isin(list(UF_TO_REGION.keys()))).select("sigla_uf").distinct()
    if ufs_invalidas.count() > 0:
        lista = [r[0] for r in ufs_invalidas.collect()]
        raise ValueError(f"UF(s) fora do domínio IBGE detectada(s): {lista}")

    mapping_expr = F.create_map([F.lit(x) for kv in UF_TO_REGION.items() for x in kv])
    df_reg = df_uf.withColumn("regioes", mapping_expr[F.col("sigla_uf")])

    # A coluna `situacao` possui preenchimento consistente no TSE a partir de 2002;
    # candidaturas anteriores a 2002 ou deferidas/sub judice são consideradas efetivas.
    cond_valido = (F.col("ano") < 2002) | F.lower(F.trim(F.col("situacao"))).isin(
        "deferido", "sub judice", "deferido com recurso"
    )

    return (
        df_reg
        .withColumn("status_candidatura", F.when(cond_valido, F.lit("valido")).otherwise(F.lit("invalido")))
        .filter(F.col("status_candidatura") == "valido")
        .withColumn(
            "cargo_std",
            F.when(F.col("cargo").rlike("(?i)deputado\\s*distrital"), F.lit("deputado distrital"))
             .when(F.col("cargo").rlike("(?i)deputado\\s*estadual"), F.lit("deputado estadual"))
             .when(F.col("cargo").rlike("(?i)deputado\\s*federal"), F.lit("deputado federal"))
             .when(F.col("cargo").rlike("(?i)vereador"), F.lit("vereador"))
             .when(F.col("cargo").rlike("(?i)(senador|suplente.*senador|senador.*suplente)"), F.lit("senador"))
             .when(F.col("cargo").rlike("(?i)(governador|vice[- ]?governador)"), F.lit("governador"))
             .when(F.col("cargo").rlike("(?i)(prefeit|vice[- ]?prefeit)"), F.lit("prefeito"))
             .otherwise(F.lit(None).cast("string")),
        )
        .withColumn("resultado_low", F.lower(F.col("resultado")))
        .withColumn("genero_low", F.lower(F.col("genero")))
        .withColumn(
            "eleito",
            F.when(
                F.col("resultado_low").rlike("eleit") & (~F.col("resultado_low").rlike("n(?:ão|ao)")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .filter(F.col("cargo_std").isNotNull())
    )


def _log_pos(colname: str, eps: float = 1e-6):
    """Logaritmo natural seguro com piso numérico (evita -inf / NaN em proporções zeradas)."""
    return F.log(F.when(F.col(colname) <= 0, F.lit(eps)).otherwise(F.col(colname)))


def build_balanced_panel_locf(
    spark: SparkSession,
    df_candidatos_validos: DataFrame,
    df_explicativas: DataFrame,
    ano_min: int = 1998,
    ano_max: int = 2022,
) -> DataFrame:
    """Agrega o painel `sigla_uf × cargo_std × ano`, balanceia com LOCF e integra covariáveis socioeconômicas."""
    cond_gen = F.col("genero_low").isin("feminino", "masculino")

    painel_uf = (
        df_candidatos_validos
        .groupBy("sigla_uf", "ano", "cargo_std")
        .agg(
            F.sum(F.when(cond_gen, 1).otherwise(0)).alias("n_total"),
            F.sum(F.when(F.col("genero_low") == "feminino", 1).otherwise(0)).alias("n_candidatas"),
            F.sum(F.when((F.col("eleito") == 1) & cond_gen, 1).otherwise(0)).alias("n_eleitos_total"),
            F.sum(F.when((F.col("eleito") == 1) & (F.col("genero_low") == "feminino"), 1).otherwise(0)).alias("n_eleitas"),
        )
        .withColumn("prop_candidatas", F.when(F.col("n_total") > 0, F.col("n_candidatas") / F.col("n_total")))
        .withColumn("prop_eleitas", F.when(F.col("n_eleitos_total") > 0, F.col("n_eleitas") / F.col("n_eleitos_total")))
        .filter((F.col("ano") >= ano_min) & (F.col("ano") <= ano_max))
    )

    anos_df = spark.createDataFrame([(a,) for a in range(ano_min, ano_max + 1)], ["ano"])
    chaves = ["sigla_uf", "cargo_std"]
    grid = painel_uf.select(*chaves).distinct().crossJoin(anos_df)
    painel_full = grid.join(painel_uf, on=chaves + ["ano"], how="left")

    # LOCF (forward) + NOCB (backward inicial) para anos intereleitorais
    w_fwd = Window.partitionBy("sigla_uf", "cargo_std").orderBy("ano").rowsBetween(Window.unboundedPreceding, 0)
    w_back = Window.partitionBy("sigla_uf", "cargo_std").orderBy(F.col("ano").desc()).rowsBetween(Window.unboundedPreceding, 0)

    for col_name in ("n_total", "n_candidatas", "prop_candidatas", "n_eleitos_total", "n_eleitas", "prop_eleitas"):
        painel_full = (
            painel_full
            .withColumn(col_name, F.last(F.col(col_name), ignorenulls=True).over(w_fwd))
            .withColumn(col_name, F.coalesce(F.col(col_name), F.last(F.col(col_name), ignorenulls=True).over(w_back)))
        )

    painel_bal = (
        painel_full
        .withColumn("tratado", F.when(F.col("cargo_std").isin(*PROPORTIONAL_OFFICES), F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "treat_post",
            F.when(F.col("cargo_std") == "vereador", F.when(F.col("ano") >= 2012, 1).otherwise(0))
             .when(
                 F.col("cargo_std").isin("deputado federal", "deputado estadual", "deputado distrital"),
                 F.when(F.col("ano") >= 2010, 1).otherwise(0),
             )
             .otherwise(0),
        )
        .withColumn(
            "ano_trat",
            F.when(F.col("cargo_std") == "vereador", F.lit(2012))
             .when(F.col("cargo_std").isin("deputado federal", "deputado estadual", "deputado distrital"), F.lit(2010))
             .otherwise(F.lit(0)),
        )
        .withColumn("id_estado", F.col("sigla_uf"))
        .withColumn("id_cargo", F.col("cargo_std"))
        .withColumn("id", F.concat_ws("_", F.col("sigla_uf"), F.col("cargo_std")))
    )

    panel_data = (
        painel_bal
        .join(df_explicativas, on=["sigla_uf", "ano"], how="left")
        .withColumn("prop_eleitoras_mulheres", F.col("eleitoras_mulheres") / F.col("eleitores_total"))
        .withColumn("ln_prop_eleitas", _log_pos("prop_eleitas"))
        .withColumn("ln_prop_candidatas", _log_pos("prop_candidatas"))
        .withColumn("ln_populacao_feminina", _log_pos("populacao_feminina"))
        .withColumn("ln_Proporcao_pop_feminina", _log_pos("Proporcao_pop_feminina"))
        .withColumn("ln_eleitoras_mulheres", _log_pos("eleitoras_mulheres"))
        .withColumn("ln_IDHM_renda", _log_pos("IDHM_renda"))
        .withColumn("ln_IDHM_educacao", _log_pos("IDHM_educacao"))
        .withColumn("ln_IDHM_longevidade", _log_pos("IDHM_longevidade"))
        .withColumn("ln_prop_eleitoras_mulheres", _log_pos("prop_eleitoras_mulheres"))
        .withColumn("ano", F.col("ano").cast("int"))
        .orderBy("sigla_uf", "cargo_std", "ano")
    )
    return panel_data


def validate_panel_integrity(panel_data: DataFrame) -> Dict[str, object]:
    """Audita unicidade de chave primária, limites [0, 1] das proporções e dimensões do painel Gold."""
    dup_count = (
        panel_data
        .groupBy("sigla_uf", "cargo_std", "ano")
        .agg(F.count(F.lit(1)).alias("n"))
        .filter(F.col("n") > 1)
        .count()
    )
    if dup_count != 0:
        raise AssertionError(f"Painel Gold contém {dup_count} chave(s) duplicada(s) em sigla_uf×cargo_std×ano.")

    ranges = panel_data.select(
        F.min("prop_candidatas").alias("min_prop_candidatas"),
        F.max("prop_candidatas").alias("max_prop_candidatas"),
        F.min("prop_eleitas").alias("min_prop_eleitas"),
        F.max("prop_eleitas").alias("max_prop_eleitas"),
    ).first().asDict()

    return {
        "linhas_painel": panel_data.count(),
        "n_ufs": panel_data.select("sigla_uf").distinct().count(),
        "n_cargos": panel_data.select("cargo_std").distinct().count(),
        "n_anos": panel_data.select("ano").distinct().count(),
        "faixas_proporcoes": ranges,
    }


def build_causal_impact_panel(panel_data: DataFrame) -> DataFrame:
    """Agrega o painel Gold anual (`1998–2022`) com série tratada, controles majoritários e covariáveis."""
    treated = (
        panel_data
        .filter(F.col("cargo_std").isin(*PROPORTIONAL_OFFICES))
        .groupBy("ano")
        .agg(F.mean("prop_candidatas").alias("prop_candidatas_tratado"))
    )

    controls = (
        panel_data
        .filter(F.col("cargo_std").isin(*MAJORITARIAN_OFFICES))
        .groupBy("ano")
        .pivot("cargo_std", list(MAJORITARIAN_OFFICES))
        .agg(F.mean("prop_candidatas"))
        .withColumnRenamed("senador", "prop_cand_senador")
        .withColumnRenamed("governador", "prop_cand_governador")
        .withColumnRenamed("prefeito", "prop_cand_prefeito")
    )

    explicativas = (
        panel_data
        .groupBy("ano")
        .agg(
            F.mean("populacao_total").alias("populacao_total"),
            F.mean("populacao_feminina").alias("populacao_feminina"),
            F.mean("Proporcao_pop_feminina").alias("prop_pop_fem"),
            F.mean("eleitoras_mulheres").alias("eleitoras_mulheres"),
            F.mean("eleitores_total").alias("eleitores_total"),
            F.mean("IDHM").alias("idhm"),
            F.mean("IDHM_renda").alias("idhm_renda"),
            F.mean("IDHM_educacao").alias("idhm_educacao"),
            F.mean("IDHM_longevidade").alias("idhm_longevidade"),
            F.mean("homicidios_total").alias("homicidios_total"),
            F.mean("feminicidios").alias("feminicidios"),
        )
    )

    return (
        treated
        .join(controls, on="ano", how="left")
        .join(explicativas, on="ano", how="left")
        .orderBy("ano")
    )


def run_databricks_pipeline(catalog: str, schema: str, ano_min: int, ano_max: int) -> None:
    """Ponto de entrada principal para execução no Databricks Workflows / Unity Catalog."""
    spark = SparkSession.builder.appName("TSE_Lei12034_Medallion_ETL").getOrCreate()
    prefix = f"{catalog}.{schema}"

    df_cand, df_eleitos, df_human, df_exp = load_bronze_tables(spark, catalog=catalog, schema=schema)
    df_merged = merge_elected_candidates(df_cand, df_eleitos)
    df_gen = impute_missing_gender_1996(df_merged, df_human)
    df_silver = enrich_region_and_filter_valid(df_gen)

    # Camada Silver
    (
        df_silver.write.mode("overwrite")
        .option("mergeSchema", "true")
        .saveAsTable(f"{prefix}.candidatos_genero_ok")
    )

    # Camada Gold — Painel Econométrico (DiD Callaway & Sant'Anna)
    panel_gold = build_balanced_panel_locf(spark, df_silver, df_exp, ano_min=ano_min, ano_max=ano_max)
    audit = validate_panel_integrity(panel_gold)
    print(f"[Gold panel_data] Auditoria concluída: {audit}")
    panel_gold.write.mode("overwrite").saveAsTable(f"{prefix}.panel_data")

    # Camada Gold — Painel Agregado para CausalImpact (BSTS)
    ci_gold = build_causal_impact_panel(panel_gold)
    ci_gold.write.mode("overwrite").saveAsTable(f"{prefix}.painel_causal_impact")
    print(f"[Gold painel_causal_impact] Tabela salva com sucesso em `{prefix}.painel_causal_impact`.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL Medallion Databricks — Lei nº 12.034/2009 (TSE)")
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="avaliacao_lei_tse")
    parser.add_argument("--ano-inicio", type=int, default=1998)
    parser.add_argument("--ano-fim", type=int, default=2022)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_databricks_pipeline(
        catalog=args.catalog,
        schema=args.schema,
        ano_min=args.ano_inicio,
        ano_max=args.ano_fim,
    )
