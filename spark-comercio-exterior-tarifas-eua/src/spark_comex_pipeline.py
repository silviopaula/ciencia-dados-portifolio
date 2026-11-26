"""Pipeline distribuído em Apache Spark (PySpark 4.0) para Comércio Exterior Brasileiro (SECEX/ComexStat).

Processa 36,5 milhões de registros municipais oficiais (13,8M exportações + 22,7M importações,
2010–Jun/2025) aplicando boas práticas de engenharia Spark:
- Compatibilidade automática com Java JDK 17/21 no Windows e Spark Connect (sem uso de `.sparkContext`).
- Deduplicação estrita 1:1 das dimensões (`dim_pais`, `dim_mun`, `dim_sh4_agreg`, `dim_sh4_cuci`)
  antes de qualquer join — eliminando multiplicação cartesiana (fan-out) de até ~9x.
- Broadcast Hash Joins explícitos + Adaptive Query Execution (AQE) + Benchmark real Broadcast vs SortMerge.
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_ZIP = PROJECT_ROOT / "data.zip"

CO_CHINA = 160
CO_EUA = 249
CO_RUSSIA = 676
SH4_SUCO_FRUTAS = 2009


def ensure_data_extracted() -> Path:
    """Garante que os arquivos Parquet oficiais estejam extraídos em `data/`."""
    exp_path = DATA_DIR / "dados_comex_mun_export_07-2025.parquet"
    if exp_path.exists():
        return DATA_DIR
    if DATA_ZIP.exists():
        with zipfile.ZipFile(DATA_ZIP, "r") as zf:
            zf.extractall(PROJECT_ROOT)
        return DATA_DIR
    raise FileNotFoundError(
        f"Arquivos Parquet não encontrados em {DATA_DIR} nem {DATA_ZIP}."
    )


def _configure_java_for_spark() -> None:
    """Seleciona automaticamente JDK 17/21 no Windows quando disponível (evita incompatibilidade do JDK 24 com Hadoop UGI)."""
    if os.name == "nt":
        for candidate in (
            r"C:\Program Files\Java\jdk-17",
            r"C:\Program Files\Java\jdk-21",
        ):
            if Path(candidate).exists():
                os.environ["JAVA_HOME"] = candidate
                os.environ["PATH"] = f"{candidate}\\bin;{os.environ.get('PATH', '')}"
                break
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def create_spark_session(app_name: str = "ComexBrasil_PySpark_Analytics") -> SparkSession:
    """Inicializa sessão Apache Spark com Adaptive Query Execution (AQE)."""
    _configure_java_for_spark()
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "24")
        .getOrCreate()
    )
    return spark


def load_clean_dimensions(spark: SparkSession, data_dir: Path) -> Dict[str, DataFrame]:
    """Carrega e deduplica 1:1 todas as tabelas-dimensão no grão exato dos joins."""
    dim_pais = (
        spark.read.parquet(str(data_dir / "pais.parquet"))
        .select(F.col("CO_PAIS").cast("int").alias("CO_PAIS"), F.col("NO_PAIS"))
        .dropDuplicates(["CO_PAIS"])
    )

    dim_mun = (
        spark.read.parquet(str(data_dir / "uf_mun.parquet"))
        .select(
            F.col("CO_MUN_GEO").cast("int").alias("CO_MUN"),
            F.col("NO_MUN_MIN").alias("NO_MUN"),
            F.col("SG_UF"),
        )
        .dropDuplicates(["CO_MUN"])
    )

    # Deduplica Minha_agregacao.parquet (13.720 linhas NCM -> 1.455 códigos SH4 únicos pela moda)
    raw_ag = (
        spark.read.parquet(str(data_dir / "Minha_agregacao.parquet"))
        .withColumn(
            "SH4",
            F.substring(F.regexp_replace(F.col("id_sh6").cast("string"), r"\.0$", ""), 1, 4).cast("int"),
        )
        .filter(F.col("SH4").isNotNull() & F.col("categoria_agregada").isNotNull())
    )
    w_ag = Window.partitionBy("SH4").orderBy(F.col("cnt").desc(), F.col("categoria_agregada").asc())
    dim_sh4_agreg = (
        raw_ag.groupBy("SH4", "categoria_agregada")
        .agg(F.count("*").alias("cnt"))
        .withColumn("rn", F.row_number().over(w_ag))
        .filter(F.col("rn") == 1)
        .select("SH4", "categoria_agregada")
    )

    # Deduplica classificação CUCI oficial no grão SH4
    raw_cuci = (
        spark.read.parquet(str(data_dir / "df_produtoscuci_master_select.parquet"))
        .select(F.col("CO_SH4").cast("int").alias("SH4"), F.col("NO_CUCI_DIVISAO"))
        .filter(F.col("SH4").isNotNull() & F.col("NO_CUCI_DIVISAO").isNotNull())
    )
    w_cuci = Window.partitionBy("SH4").orderBy(F.col("cnt").desc(), F.col("NO_CUCI_DIVISAO").asc())
    dim_sh4_cuci = (
        raw_cuci.groupBy("SH4", "NO_CUCI_DIVISAO")
        .agg(F.count("*").alias("cnt"))
        .withColumn("rn", F.row_number().over(w_cuci))
        .filter(F.col("rn") == 1)
        .select("SH4", "NO_CUCI_DIVISAO")
    )

    return {
        "pais": dim_pais,
        "mun": dim_mun,
        "sh4_agreg": dim_sh4_agreg,
        "sh4_cuci": dim_sh4_cuci,
        "raw_ag_count": raw_ag.count(),
        "clean_ag_count": dim_sh4_agreg.count(),
    }


def compute_hhi(shares_series: pd.Series) -> float:
    """Calcula Índice Herfindahl-Hirschman (escala 0–10.000): <1500 desconcentrado, 1500-2500 moderado, >2500 alto."""
    s = pd.to_numeric(shares_series, errors="coerce").dropna()
    tot = float(s.sum())
    if tot <= 0:
        return 0.0
    pct = (s / tot) * 100.0
    return float((pct ** 2).sum())


def run_spark_comex_analytics(spark: SparkSession) -> Dict[str, Any]:
    """Executa todas as agregações distribuídas sobre 36,5M registros SECEX e retorna DataFrames prontos."""
    t_total = time.perf_counter()
    data_dir = ensure_data_extracted()

    df_exp = spark.read.parquet(str(data_dir / "dados_comex_mun_export_07-2025.parquet"))
    df_imp = spark.read.parquet(str(data_dir / "dados_comex_mun_import_07-2025.parquet"))
    dims = load_clean_dimensions(spark, data_dir)
    dim_pais = dims["pais"]
    dim_mun = dims["mun"]
    dim_sh4_agreg = dims["sh4_agreg"]
    dim_sh4_cuci = dims["sh4_cuci"]

    # 1. Auditoria de Volume, Nulos e Fan-Out de Join (Star-Schema Integrity)
    exp_stats = df_exp.agg(
        F.count("*").alias("rows"),
        F.sum("VL_FOB").alias("fob_total"),
        F.sum("KG_LIQUIDO").alias("kg_total"),
        F.min("CO_ANO").alias("ano_min"),
        F.max("CO_ANO").alias("ano_max"),
        F.sum(F.when(F.col("VL_FOB").isNull() | (F.col("VL_FOB") < 0), 1).otherwise(0)).alias("fob_invalid"),
        F.sum(F.when(F.col("KG_LIQUIDO").isNull() | (F.col("KG_LIQUIDO") < 0), 1).otherwise(0)).alias("kg_invalid"),
    ).collect()[0].asDict()

    imp_stats = df_imp.agg(
        F.count("*").alias("rows"),
        F.sum("VL_FOB").alias("fob_total"),
        F.sum("KG_LIQUIDO").alias("kg_total"),
        F.min("CO_ANO").alias("ano_min"),
        F.max("CO_ANO").alias("ano_max"),
        F.sum(F.when(F.col("VL_FOB").isNull() | (F.col("VL_FOB") < 0), 1).otherwise(0)).alias("fob_invalid"),
        F.sum(F.when(F.col("KG_LIQUIDO").isNull() | (F.col("KG_LIQUIDO") < 0), 1).otherwise(0)).alias("kg_invalid"),
    ).collect()[0].asDict()

    # Janela últimos 12 meses fechados da base: Jul/2024 a Jun/2025
    cond_12m = ((F.col("CO_ANO") == 2024) & (F.col("CO_MES") >= 7)) | (
        (F.col("CO_ANO") == 2025) & (F.col("CO_MES") <= 6)
    )
    exp_12m = df_exp.filter(cond_12m)
    imp_12m = df_imp.filter(cond_12m)

    # Auditoria quantitativa do bug de multiplicação (fan-out) vs Join 1:1 Broadcast
    raw_ag = (
        spark.read.parquet(str(data_dir / "Minha_agregacao.parquet"))
        .withColumn(
            "SH4",
            F.substring(F.regexp_replace(F.col("id_sh6").cast("string"), r"\.0$", ""), 1, 4).cast("int"),
        )
        .select("SH4", "categoria_agregada")
    )
    fob_12m_true = float(exp_12m.agg(F.sum("VL_FOB")).collect()[0][0])
    fob_12m_bug = float(exp_12m.join(raw_ag, on="SH4", how="left").agg(F.sum("VL_FOB")).collect()[0][0])
    fob_12m_clean = float(exp_12m.join(F.broadcast(dim_sh4_agreg), on="SH4", how="left").agg(F.sum("VL_FOB")).collect()[0][0])

    # Benchmark real de engenharia Spark (13,8M registros): Broadcast Hash Join vs SortMergeJoin (sem broadcast)
    t0 = time.perf_counter()
    df_exp.join(F.broadcast(dim_pais), on="CO_PAIS", how="left").groupBy("NO_PAIS").agg(F.sum("VL_FOB")).collect()
    t_broadcast = time.perf_counter() - t0

    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
    t0 = time.perf_counter()
    df_exp.join(dim_pais, on="CO_PAIS", how="left").groupBy("NO_PAIS").agg(F.sum("VL_FOB")).collect()
    t_sortmerge = time.perf_counter() - t0
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")

    # 2. Balança Comercial Anual e Mensal (2010–Jun/2025)
    exp_ano = df_exp.groupBy("CO_ANO").agg(
        F.sum("VL_FOB").alias("export_fob"),
        F.sum("KG_LIQUIDO").alias("export_kg"),
    )
    imp_ano = df_imp.groupBy("CO_ANO").agg(
        F.sum("VL_FOB").alias("import_fob"),
        F.sum("KG_LIQUIDO").alias("import_kg"),
    )
    balanca_anual = (
        exp_ano.join(imp_ano, on="CO_ANO", how="outer")
        .orderBy("CO_ANO")
        .toPandas()
    )
    balanca_anual["saldo_fob"] = balanca_anual["export_fob"] - balanca_anual["import_fob"]
    balanca_anual["corrente_fob"] = balanca_anual["export_fob"] + balanca_anual["import_fob"]
    balanca_anual["preco_medio_exp_usd_kg"] = balanca_anual["export_fob"] / balanca_anual["export_kg"].clip(lower=1)
    balanca_anual["preco_medio_imp_usd_kg"] = balanca_anual["import_fob"] / balanca_anual["import_kg"].clip(lower=1)

    exp_mes = df_exp.groupBy("CO_ANO", "CO_MES").agg(F.sum("VL_FOB").alias("export_fob"))
    imp_mes = df_imp.groupBy("CO_ANO", "CO_MES").agg(F.sum("VL_FOB").alias("import_fob"))
    balanca_mensal = (
        exp_mes.join(imp_mes, on=["CO_ANO", "CO_MES"], how="outer")
        .orderBy("CO_ANO", "CO_MES")
        .toPandas()
    )
    balanca_mensal["date"] = pd.to_datetime(
        balanca_mensal["CO_ANO"].astype(int).astype(str)
        + "-"
        + balanca_mensal["CO_MES"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    balanca_mensal["saldo_fob"] = balanca_mensal["export_fob"] - balanca_mensal["import_fob"]

    # 3. Índice de Concentração Herfindahl-Hirschman (HHI) Anual (2010–2024)
    exp_ano_pais = df_exp.filter(F.col("CO_ANO") <= 2024).groupBy("CO_ANO", "CO_PAIS").agg(F.sum("VL_FOB").alias("fob")).toPandas()
    exp_ano_sh4 = df_exp.filter(F.col("CO_ANO") <= 2024).groupBy("CO_ANO", "SH4").agg(F.sum("VL_FOB").alias("fob")).toPandas()
    hhi_rows = []
    for yr in sorted(exp_ano_pais["CO_ANO"].unique()):
        p_sub = exp_ano_pais[exp_ano_pais["CO_ANO"] == yr]
        s_sub = exp_ano_sh4[exp_ano_sh4["CO_ANO"] == yr]
        tot_p = p_sub["fob"].sum()
        china_share = float(p_sub.loc[p_sub["CO_PAIS"] == CO_CHINA, "fob"].sum() / tot_p * 100)
        usa_share = float(p_sub.loc[p_sub["CO_PAIS"] == CO_EUA, "fob"].sum() / tot_p * 100)
        hhi_rows.append({
            "CO_ANO": int(yr),
            "hhi_parceiros": round(compute_hhi(p_sub["fob"]), 1),
            "hhi_produtos_sh4": round(compute_hhi(s_sub["fob"]), 1),
            "share_china_pct": round(china_share, 2),
            "share_eua_pct": round(usa_share, 2),
        })
    hhi_anual = pd.DataFrame(hhi_rows)

    # 4. Top 10 Parceiros Comerciais (Corrente = Exp + Imp) & Destinos de Exportação (Jul/2024–Jun/2025)
    e_pais_12m = exp_12m.groupBy("CO_PAIS").agg(F.sum("VL_FOB").alias("export_fob"))
    i_pais_12m = imp_12m.groupBy("CO_PAIS").agg(F.sum("VL_FOB").alias("import_fob"))
    parceiros_12m = (
        e_pais_12m.join(i_pais_12m, on="CO_PAIS", how="outer")
        .fillna(0)
        .withColumn("corrente_fob", F.col("export_fob") + F.col("import_fob"))
        .withColumn("saldo_fob", F.col("export_fob") - F.col("import_fob"))
        .join(F.broadcast(dim_pais), on="CO_PAIS", how="left")
        .orderBy(F.col("corrente_fob").desc())
        .limit(12)
        .toPandas()
    )

    # Top 10 Municípios Exportadores (Jul/2024–Jun/2025)
    top10_mun_exp = (
        exp_12m.groupBy("CO_MUN")
        .agg(F.sum("VL_FOB").alias("export_fob"), F.sum("KG_LIQUIDO").alias("export_kg"))
        .join(F.broadcast(dim_mun), on="CO_MUN", how="left")
        .withColumn("municipio_uf", F.concat_ws(" - ", F.col("NO_MUN"), F.col("SG_UF")))
        .orderBy(F.col("export_fob").desc())
        .limit(10)
        .toPandas()
    )

    # 5. Exposição Bilateral por Categoria Agregada (EUA vs. China — Jul/2024–Jun/2025, 1:1 Broadcast)
    exp_cat = (
        exp_12m.join(F.broadcast(dim_sh4_agreg), on="SH4", how="left")
        .fillna({"categoria_agregada": "Outros Produtos"})
    )
    cat_tot = exp_cat.groupBy("categoria_agregada").agg(F.sum("VL_FOB").alias("total_cat_fob"))
    cat_pais = (
        exp_cat.filter(F.col("CO_PAIS").isin([CO_EUA, CO_CHINA]))
        .groupBy("categoria_agregada", "CO_PAIS")
        .agg(F.sum("VL_FOB").alias("pais_fob"))
        .join(F.broadcast(cat_tot), on="categoria_agregada", how="left")
        .withColumn("share_pct", F.col("pais_fob") / F.col("total_cat_fob") * 100.0)
        .toPandas()
    )
    exposicao_eua = (
        cat_pais[(cat_pais["CO_PAIS"] == CO_EUA) & (cat_pais["pais_fob"] >= 150_000_000)]
        .sort_values("share_pct", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )
    exposicao_china = (
        cat_pais[(cat_pais["CO_PAIS"] == CO_CHINA) & (cat_pais["pais_fob"] >= 150_000_000)]
        .sort_values("share_pct", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    # 6. Vulnerabilidade de Importação: Top 5 Divisões CUCI importadas da Rússia + Top 3 Alternativas (Window)
    imp_cuci = (
        imp_12m.join(F.broadcast(dim_sh4_cuci), on="SH4", how="left")
        .filter(F.col("NO_CUCI_DIVISAO").isNotNull())
        .groupBy("NO_CUCI_DIVISAO", "CO_PAIS")
        .agg(F.sum("VL_FOB").alias("vl_fob"))
    )
    cuci_tot = imp_cuci.groupBy("NO_CUCI_DIVISAO").agg(F.sum("vl_fob").alias("total_div_fob"))
    imp_cuci_share = imp_cuci.join(F.broadcast(cuci_tot), on="NO_CUCI_DIVISAO", how="left").withColumn(
        "share_pct", F.col("vl_fob") / F.col("total_div_fob") * 100.0
    )

    top5_russia = (
        imp_cuci_share.filter(F.col("CO_PAIS") == CO_RUSSIA)
        .orderBy(F.col("vl_fob").desc())
        .limit(5)
        .toPandas()
    )
    top5_divs = top5_russia["NO_CUCI_DIVISAO"].tolist()
    w_alt = Window.partitionBy("NO_CUCI_DIVISAO").orderBy(F.col("vl_fob").desc())
    russia_alternativos = (
        imp_cuci_share.filter((F.col("NO_CUCI_DIVISAO").isin(top5_divs)) & (F.col("CO_PAIS") != CO_RUSSIA))
        .withColumn("rank", F.row_number().over(w_alt))
        .filter(F.col("rank") <= 3)
        .join(F.broadcast(dim_pais), on="CO_PAIS", how="left")
        .select("NO_CUCI_DIVISAO", "rank", "NO_PAIS", "vl_fob", "share_pct")
        .orderBy("NO_CUCI_DIVISAO", "rank")
        .toPandas()
    )

    # 7. Estudo de Caso Suco de Laranja (SH4 = 2009): Top Destinos 12m + Série Mensal EUA vs Resto do Mundo (IV)
    suco_12m = (
        exp_12m.filter(F.col("SH4") == SH4_SUCO_FRUTAS)
        .groupBy("CO_PAIS")
        .agg(F.sum("VL_FOB").alias("vl_fob"), F.sum("KG_LIQUIDO").alias("kg_liquido"))
        .join(F.broadcast(dim_pais), on="CO_PAIS", how="left")
        .orderBy(F.col("vl_fob").desc())
        .limit(10)
        .toPandas()
    )

    # Painel mensal 2010–Jun/2025: EUA vs Resto do Mundo (para Instrumento de Oferta 2SLS!)
    suco_eua_m = (
        df_exp.filter((F.col("SH4") == SH4_SUCO_FRUTAS) & (F.col("CO_PAIS") == CO_EUA))
        .groupBy("CO_ANO", "CO_MES")
        .agg(F.sum("VL_FOB").alias("vl_fob_eua"), F.sum("KG_LIQUIDO").alias("kg_eua"))
    )
    suco_row_m = (
        df_exp.filter((F.col("SH4") == SH4_SUCO_FRUTAS) & (F.col("CO_PAIS") != CO_EUA))
        .groupBy("CO_ANO", "CO_MES")
        .agg(F.sum("VL_FOB").alias("vl_fob_row"), F.sum("KG_LIQUIDO").alias("kg_row"))
    )
    suco_mensal = (
        suco_eua_m.join(suco_row_m, on=["CO_ANO", "CO_MES"], how="inner")
        .filter((F.col("kg_eua") > 0) & (F.col("vl_fob_eua") > 0) & (F.col("kg_row") > 0) & (F.col("vl_fob_row") > 0))
        .orderBy("CO_ANO", "CO_MES")
        .toPandas()
    )
    suco_mensal["date"] = pd.to_datetime(
        suco_mensal["CO_ANO"].astype(int).astype(str)
        + "-"
        + suco_mensal["CO_MES"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    suco_mensal["preco_fob_eua"] = suco_mensal["vl_fob_eua"] / suco_mensal["kg_eua"]
    suco_mensal["preco_fob_row"] = suco_mensal["vl_fob_row"] / suco_mensal["kg_row"]

    elapsed_total = round(time.perf_counter() - t_total, 2)
    return {
        "spark_version": spark.version,
        "elapsed_spark_sec": elapsed_total,
        "audit": {
            "export_rows": int(exp_stats["rows"]),
            "import_rows": int(imp_stats["rows"]),
            "total_rows": int(exp_stats["rows"]) + int(imp_stats["rows"]),
            "export_fob_total_usd": float(exp_stats["fob_total"]),
            "import_fob_total_usd": float(imp_stats["fob_total"]),
            "period": f"{int(exp_stats['ano_min'])}–Jun/{int(exp_stats['ano_max'])}",
            "raw_ag_rows": int(dims["raw_ag_count"]),
            "dedup_sh4_rows": int(dims["clean_ag_count"]),
            "fob_12m_true_usd": round(fob_12m_true, 2),
            "fob_12m_bug_fanout_usd": round(fob_12m_bug, 2),
            "fob_12m_clean_broadcast_usd": round(fob_12m_clean, 2),
            "fanout_inflation_factor": round(fob_12m_bug / fob_12m_true, 2),
            "benchmark_broadcast_sec": round(t_broadcast, 3),
            "benchmark_sortmerge_sec": round(t_sortmerge, 3),
            "benchmark_speedup_x": round(t_sortmerge / max(t_broadcast, 1e-4), 2),
        },
        "balanca_anual": balanca_anual,
        "balanca_mensal": balanca_mensal,
        "hhi_anual": hhi_anual,
        "parceiros_12m": parceiros_12m,
        "top10_mun_exp": top10_mun_exp,
        "exposicao_eua": exposicao_eua,
        "exposicao_china": exposicao_china,
        "top5_russia": top5_russia,
        "russia_alternativos": russia_alternativos,
        "suco_12m": suco_12m,
        "suco_mensal": suco_mensal,
    }
