"""
Data Import, Date Conversion, Dummy Generation, Variable Transformations,
Seasonality/Trend Decomposition, and Export functions.
Ported from R/functions_page_data_import.R and R/modulo_data_import.R.
"""
from __future__ import annotations
import io
import math
import os
import re
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def detectar_tipo_arquivo(nome_arquivo: str) -> str:
    ext = os.path.splitext(nome_arquivo)[1].lower().lstrip(".")
    if ext in ("csv", "txt"):
        return "csv"
    elif ext in ("xlsx", "xls", "xlsm"):
        return "xlsx"
    elif ext == "rds":
        return "rds"
    elif ext == "dta":
        return "dta"
    elif ext == "sav":
        return "sav"
    elif ext in ("sqlite", "db", "sqlite3", "db3", "sdb", "sl3"):
        return "sqlite"
    return "desconhecido"


def inspect_file_metadata(file_path: str, filename: str, force_type: str = "auto") -> Dict[str, Any]:
    detected = detectar_tipo_arquivo(filename) if force_type == "auto" else force_type
    sheets: List[str] = []
    tables: List[str] = []

    if detected == "xlsx":
        try:
            xl = pd.ExcelFile(file_path)
            sheets = list(xl.sheet_names)
        except Exception:
            sheets = []
    elif detected == "sqlite":
        try:
            conn = sqlite3.connect(file_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception:
            tables = []

    return {
        "filename": filename,
        "detected_type": detected,
        "sheets": sheets,
        "tables": tables,
    }


def importar_dados(
    file_path: str,
    filename: str,
    tipo_arquivo: str = "auto",
    planilha: Optional[str] = None,
    tabela: Optional[str] = None,
) -> pd.DataFrame:
    if tipo_arquivo == "auto":
        tipo_arquivo = detectar_tipo_arquivo(filename)

    if tipo_arquivo == "csv":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample_lines = [f.readline() for _ in range(5)]
        sample_text = "".join(sample_lines)
        sep_char = ";" if ";" in sample_text else ","
        df = pd.read_csv(
            file_path,
            sep=sep_char,
            na_values=["", "NA", "NULL", "NaN", "null"],
            keep_default_na=True,
        )
        # If decimal comma was used in semicolon CSV, try converting numeric-looking object columns
        for col in df.columns:
            if df[col].dtype == object:
                sample_non_na = df[col].dropna().astype(str).head(20)
                if len(sample_non_na) > 0 and all(
                    re.match(r"^-?\d+(,\d+)?$", s.strip()) for s in sample_non_na
                ):
                    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="ignore")

    elif tipo_arquivo == "xlsx":
        xl = pd.ExcelFile(file_path)
        if not planilha or planilha not in xl.sheet_names:
            if len(xl.sheet_names) == 0:
                raise ValueError("Nenhuma planilha encontrada no arquivo Excel.")
            planilha = xl.sheet_names[0]
        df = pd.read_excel(file_path, sheet_name=planilha, na_values=["", "NA", "NULL"])

    elif tipo_arquivo == "rds":
        import pyreadr
        res = pyreadr.read_r(file_path)
        if not res:
            raise ValueError("Nenhum objeto encontrado no arquivo RDS.")
        first_key = next(iter(res.keys()))
        df = res[first_key]

    elif tipo_arquivo == "dta":
        df = pd.read_stata(file_path)

    elif tipo_arquivo == "sav":
        import pyreadstat
        df, _ = pyreadstat.read_sav(file_path)

    elif tipo_arquivo == "sqlite":
        conn = sqlite3.connect(file_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tabelas = [r[0] for r in cur.fetchall()]
            if not tabela or tabela == "":
                if len(tabelas) == 0:
                    raise ValueError("Nenhuma tabela encontrada no banco SQLite.")
                tabela = tabelas[0]
            if tabela not in tabelas:
                raise ValueError(f"Tabela '{tabela}' não encontrada no SQLite.")
            df = pd.read_sql_query(f'SELECT * FROM "{tabela}"', conn)
        finally:
            conn.close()
    else:
        raise ValueError(f"Tipo de arquivo desconhecido ou não suportado: {tipo_arquivo}")

    df = pd.DataFrame(df).copy()

    # Standardize datetime/timestamp columns to date objects (YYYY-MM-DD) or datetime64
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col]).dt.date
        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].round(4)

    return df


def find_candidate_date_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    pattern = re.compile(r"(date|data|ano|year|dt_|dt|time|period|mes|month)", re.IGNORECASE)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(col)
        elif series.dropna().apply(lambda x: isinstance(x, (date, datetime))).all() and len(series.dropna()) > 0:
            candidates.append(col)
        elif pattern.search(str(col)):
            candidates.append(col)
    return candidates


def _try_parse_dates(series: pd.Series, fmt: str = "auto") -> Optional[pd.Series]:
    non_na = series.dropna()
    if len(non_na) == 0:
        return None

    # Already date or datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.date
    if non_na.apply(lambda x: isinstance(x, (date, datetime))).all():
        return pd.to_datetime(series, errors="coerce").dt.date

    # If numeric
    if pd.api.types.is_numeric_dtype(series):
        vals = non_na.astype(float)
        # Check if 4-digit year (1800..2150)
        if ((vals >= 1800) & (vals <= 2150) & ((vals - np.floor(vals)).abs() < 1e-3)).all():
            return pd.to_datetime(series.astype("Int64").astype(str) + "-01-01", errors="coerce").dt.date
        # Check if YYYYMM (6 digits)
        s_vals = non_na.astype(int).astype(str)
        if (s_vals.str.len() == 6).all():
            return pd.to_datetime(
                s_vals.str.slice(0, 4) + "-" + s_vals.str.slice(4, 6) + "-01",
                errors="coerce",
            ).dt.date
        # Check if YYYYMMDD (8 digits)
        if (s_vals.str.len() == 8).all():
            return pd.to_datetime(
                s_vals.str.slice(0, 4) + "-" + s_vals.str.slice(4, 6) + "-" + s_vals.str.slice(6, 8),
                errors="coerce",
            ).dt.date
        # Check Excel serial date (roughly 15000 to 60000 -> 1941 to 2064)
        if ((vals >= 10000) & (vals <= 80000)).all():
            return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce").dt.date
        # Check Unix epoch days or seconds
        if ((vals >= 1e8) & (vals <= 3e9)).all():
            return pd.to_datetime(series, unit="s", origin="unix", errors="coerce").dt.date

    # Strings
    str_series = series.astype(str).str.strip()
    if fmt != "auto":
        fmt_map = {
            "ymd": ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"],
            "dmy": ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"],
            "mdy": ["%m/%d/%Y", "%m-%d-%Y"],
        }
        for f in fmt_map.get(fmt, [fmt]):
            parsed = pd.to_datetime(str_series, format=f, errors="coerce")
            if parsed.notna().sum() >= len(non_na) * 0.8:
                return parsed.dt.date

    # Auto try explicit formats first
    formats_to_try = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m",
        "%m/%Y",
    ]
    best_parsed: Optional[pd.Series] = None
    best_count = 0
    for f in formats_to_try:
        try:
            parsed = pd.to_datetime(str_series, format=f, errors="coerce")
            valid_count = int(parsed.notna().sum())
            if valid_count > best_count:
                best_count = valid_count
                best_parsed = parsed
            if valid_count == len(non_na):
                return parsed.dt.date
        except Exception:
            continue

    # Fallback flexible parser
    try:
        parsed_flex = pd.to_datetime(str_series, dayfirst=True, errors="coerce")
        if int(parsed_flex.notna().sum()) > best_count:
            best_parsed = parsed_flex
            best_count = int(parsed_flex.notna().sum())
    except Exception:
        pass

    if best_parsed is not None and best_count > 0:
        return best_parsed.dt.date
    return None


def inspecionar_coluna_data(df: pd.DataFrame, coluna: str) -> Dict[str, Any]:
    if coluna not in df.columns:
        return {"status": "error", "mensagem": f"Coluna '{coluna}' não encontrada."}

    series = df[coluna]
    n_total = len(series)
    n_na = int(series.isna().sum())
    n_unico = int(series.nunique(dropna=True))
    amostras = [str(x) for x in series.dropna().head(8).tolist()]

    already_date = (
        pd.api.types.is_datetime64_any_dtype(series)
        or (len(series.dropna()) > 0 and series.dropna().apply(lambda x: isinstance(x, (date, datetime))).all())
    )

    parsed = _try_parse_dates(series)
    can_convert = parsed is not None and int(pd.Series(parsed).notna().sum()) > 0
    converted_samples = [str(x) for x in pd.Series(parsed).dropna().head(5).tolist()] if can_convert else []
    n_converted = int(pd.Series(parsed).notna().sum()) if can_convert else 0

    return {
        "status": "success",
        "coluna": coluna,
        "classe": str(series.dtype),
        "total_registros": n_total,
        "registros_na": n_na,
        "valores_unicos": n_unico,
        "amostras": amostras,
        "e_data": bool(already_date),
        "pode_converter": bool(can_convert),
        "n_convertido": n_converted,
        "amostras_convertidas": converted_samples,
    }


def converter_coluna_para_date_ok(df: pd.DataFrame, coluna: str, formato: str = "auto") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no conjunto de dados.")

    original_na = int(df[coluna].isna().sum())
    parsed = _try_parse_dates(df[coluna], fmt=formato)
    if parsed is None or pd.Series(parsed).notna().sum() == 0:
        samples = ", ".join(str(x) for x in df[coluna].dropna().head(5).tolist())
        raise ValueError(f"Falha ao converter '{coluna}' para data. Exemplos de valores: {samples}")

    df_out = df.copy()
    df_out[coluna] = parsed
    df_out["date_ok"] = parsed

    # Place date_ok as the first column, preserving all others
    other_cols = [c for c in df_out.columns if c != "date_ok"]
    df_out = df_out[["date_ok"] + other_cols]

    final_na = int(pd.Series(df_out["date_ok"]).isna().sum())
    new_nas = max(0, final_na - original_na)

    valid_dates = pd.Series(df_out["date_ok"]).dropna()
    min_date = str(valid_dates.min()) if len(valid_dates) > 0 else None
    max_date = str(valid_dates.max()) if len(valid_dates) > 0 else None

    return df_out, {
        "coluna_origem": coluna,
        "novos_nas": new_nas,
        "total_validas": int(len(valid_dates)),
        "data_min": min_date,
        "data_max": max_date,
    }


def _make_safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.]", "_", name.strip())
    if clean and clean[0].isdigit():
        clean = "x_" + clean
    return clean


def gerar_dummies(
    df: pd.DataFrame,
    col_data: str = "date_ok",
    anos: bool = True,
    meses: bool = True,
    tendencia_linear: bool = False,
    funcoes_trig: bool = False,
    estacoes: bool = False,
    bro_bro: bool = False,
    crises_selecionadas: Optional[List[str]] = None,
    periodo_personalizado: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or len(df) == 0:
        raise ValueError("Dados nulos ou vazios.")
    if col_data not in df.columns:
        raise ValueError(f"Coluna de data '{col_data}' não encontrada. Converta a data primeiro.")

    out = df.copy()
    dt_series = pd.to_datetime(out[col_data], errors="coerce")
    valid_mask = dt_series.notna()
    if not valid_mask.any():
        raise ValueError(f"Coluna '{col_data}' contém apenas valores ausentes (NA).")

    added_cols: List[str] = []

    if anos:
        out["ano"] = dt_series.dt.year
        added_cols.append("ano")
        anos_unicos = sorted(out.loc[valid_mask, "ano"].dropna().astype(int).unique().tolist())
        for ano in anos_unicos:
            col_nome = _make_safe_name(f"dummy_ano_{ano}")
            out[col_nome] = np.where(valid_mask & (out["ano"] == ano), 1, 0)
            added_cols.append(col_nome)

    if meses:
        out["mes"] = dt_series.dt.month
        if "mes" not in added_cols:
            added_cols.append("mes")
        for mes in range(1, 13):
            col_nome = _make_safe_name(f"dummy_mes_{mes}")
            out[col_nome] = np.where(valid_mask & (out["mes"] == mes), 1, 0)
            added_cols.append(col_nome)

    if tendencia_linear:
        out["trend"] = np.nan
        valid_indices = dt_series[valid_mask].sort_values().index
        out.loc[valid_indices, "trend"] = np.arange(1, len(valid_indices) + 1, dtype=int)
        added_cols.append("trend")

    if funcoes_trig:
        if "mes" not in out.columns:
            out["mes"] = dt_series.dt.month
            added_cols.append("mes")
        m_vals = out["mes"].astype(float)
        out["sin1"] = np.where(valid_mask, np.sin(2 * np.pi * m_vals / 12.0).round(6), np.nan)
        out["cos1"] = np.where(valid_mask, np.cos(2 * np.pi * m_vals / 12.0).round(6), np.nan)
        out["sin2"] = np.where(valid_mask, np.sin(4 * np.pi * m_vals / 12.0).round(6), np.nan)
        out["cos2"] = np.where(valid_mask, np.cos(4 * np.pi * m_vals / 12.0).round(6), np.nan)
        added_cols.extend(["sin1", "cos1", "sin2", "cos2"])

    if estacoes:
        if "mes" not in out.columns:
            out["mes"] = dt_series.dt.month
            added_cols.append("mes")
        m = dt_series.dt.month
        d = dt_series.dt.day

        # Hemisfério Sul (exatamente como no R):
        # Verão: Dec 21 - Mar 20
        out["verao"] = np.where(
            valid_mask & (((m == 12) & (d >= 21)) | (m == 1) | (m == 2) | ((m == 3) & (d <= 20))),
            1,
            0,
        )
        # Outono: Mar 21 - Jun 20
        out["outono"] = np.where(
            valid_mask & (((m == 3) & (d >= 21)) | (m == 4) | (m == 5) | ((m == 6) & (d <= 20))),
            1,
            0,
        )
        # Inverno: Jun 21 - Sep 20
        out["inverno"] = np.where(
            valid_mask & (((m == 6) & (d >= 21)) | (m == 7) | (m == 8) | ((m == 9) & (d <= 20))),
            1,
            0,
        )
        # Primavera: Sep 21 - Dec 20
        out["primavera"] = np.where(
            valid_mask & (((m == 9) & (d >= 21)) | (m == 10) | (m == 11) | ((m == 12) & (d <= 20))),
            1,
            0,
        )
        added_cols.extend(["verao", "outono", "inverno", "primavera"])

    if bro_bro:
        if "mes" not in out.columns:
            out["mes"] = dt_series.dt.month
            added_cols.append("mes")
        m = dt_series.dt.month
        out["bro_bro"] = np.where(valid_mask & (m >= 9) & (m <= 12), 1, 0)
        added_cols.append("bro_bro")

    if crises_selecionadas:
        periodos_crises = {
            "pandemia_covid": ("dummy_pandemia_covid", "2020-03-01", "2021-12-31"),
            "crise_covid_brasil": ("dummy_crise_covid_brasil", "2020-03-01", "2022-04-30"),
            "crise_2008": ("dummy_crise_2008", "2008-09-01", "2009-06-30"),
            "crise_dot_com": ("dummy_crise_dot_com", "2000-03-01", "2002-10-31"),
            "crise_dilma": ("dummy_crise_dilma", "2014-01-01", "2016-08-31"),
            "crise_divida_80": ("dummy_crise_divida_80", "1981-08-01", "1983-12-31"),
            "hiperinflacao_80_90": ("dummy_hiperinflacao_80_90", "1986-01-01", "1994-06-30"),
            "crise_cambio_1999": ("dummy_crise_cambio_1999", "1999-01-01", "1999-03-31"),
        }
        for crise_id in crises_selecionadas:
            if crise_id in periodos_crises:
                col_nome, dt_ini_s, dt_fim_s = periodos_crises[crise_id]
                dt_ini = pd.Timestamp(dt_ini_s)
                dt_fim = pd.Timestamp(dt_fim_s)
                out[col_nome] = np.where(
                    valid_mask & (dt_series >= dt_ini) & (dt_series <= dt_fim),
                    1,
                    0,
                )
                added_cols.append(col_nome)

    if periodo_personalizado and periodo_personalizado.get("nome"):
        nome_raw = periodo_personalizado["nome"].strip()
        if not re.match(r"^[A-Za-z0-9_.]+$", nome_raw):
            raise ValueError("Dummy personalizada: Nome inválido (use apenas letras, números, '_' ou '.').")
        col_nome_pers = _make_safe_name(f"dummy_{nome_raw}")
        if col_nome_pers in out.columns or nome_raw in out.columns:
            raise ValueError(f"Dummy personalizada: Nome '{col_nome_pers}' já existe nos dados.")
        dt_ini = pd.Timestamp(periodo_personalizado["inicio"])
        dt_fim = pd.Timestamp(periodo_personalizado["fim"])
        if dt_ini > dt_fim:
            raise ValueError("Dummy personalizada: A data inicial não pode ser maior que a data final.")
        out[col_nome_pers] = np.where(
            valid_mask & (dt_series >= dt_ini) & (dt_series <= dt_fim),
            1,
            0,
        )
        added_cols.append(col_nome_pers)

    return out, added_cols


def transformar_dados(
    df: pd.DataFrame,
    variaveis: List[str],
    transformacao: str = "nenhuma",
    parametro_boxcox: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if df is None or transformacao == "nenhuma" or not variaveis:
        return df, {}

    out = df.copy()
    mensagens: Dict[str, str] = {}

    for var in variaveis:
        if var not in out.columns:
            mensagens[var] = f"Coluna '{var}' não encontrada."
            continue
        if not pd.api.types.is_numeric_dtype(out[var]):
            mensagens[var] = f"Coluna '{var}' não é numérica."
            continue

        s = out[var].astype(float)
        non_na = s.dropna()
        if len(non_na) == 0:
            mensagens[var] = f"Coluna '{var}' contém apenas NAs."
            continue

        if transformacao == "log":
            col_new = _make_safe_name(f"ln_{var}")
            if (non_na <= 0).any():
                min_val = float(non_na[non_na <= 0].min())
                constante = abs(min_val) + 1e-9
                out[col_new] = np.log(s + constante).round(6)
                mensagens[var] = f"Valores <= 0 detectados. Aplicado ln(x + {constante:.4g}). Nova coluna: '{col_new}'"
            else:
                out[col_new] = np.log(s).round(6)
                mensagens[var] = f"Aplicado ln(x). Nova coluna: '{col_new}'"

        elif transformacao == "boxcox":
            col_new = _make_safe_name(f"{var}_boxcox")
            s_pos = s.copy()
            msg_prefix = ""
            if (s_pos.dropna() <= 0).any():
                min_val = float(s_pos.dropna()[s_pos.dropna() <= 0].min())
                const_add = abs(min_val) + 1e-6
                s_pos = s_pos + const_add
                msg_prefix = f"Valores <= 0 corrigidos (x + {const_add:.4g}). "

            valid_mask = s_pos.notna()
            if valid_mask.sum() < 3:
                mensagens[var] = f"{msg_prefix}Pontos insuficientes (< 3) para Box-Cox."
                continue

            if parametro_boxcox is None or (isinstance(parametro_boxcox, float) and math.isnan(parametro_boxcox)):
                transformed_vals, lmbda = stats.boxcox(s_pos[valid_mask].values)
                res_series = pd.Series(np.nan, index=s.index, dtype=float)
                res_series.loc[valid_mask] = np.round(transformed_vals, 6)
                out[col_new] = res_series
                mensagens[var] = f"{msg_prefix}Box-Cox automático (Lambda={lmbda:.4f}). Nova coluna: '{col_new}'"
            else:
                lmbda = float(parametro_boxcox)
                if abs(lmbda) < 1e-9:
                    out[col_new] = np.log(s_pos).round(6)
                    mensagens[var] = f"{msg_prefix}Box-Cox com Lambda~0 (ln). Nova coluna: '{col_new}'"
                else:
                    out[col_new] = (((s_pos ** lmbda) - 1.0) / lmbda).round(6)
                    mensagens[var] = f"{msg_prefix}Box-Cox (Lambda={lmbda:.4f}). Nova coluna: '{col_new}'"

        elif transformacao == "scale":
            col_new = _make_safe_name(f"{var}_scaled")
            mean_val = float(non_na.mean())
            std_val = float(non_na.std(ddof=1))
            if std_val == 0 or math.isnan(std_val):
                mensagens[var] = f"Desvio padrão zero na coluna '{var}', impossível padronizar."
                continue
            out[col_new] = ((s - mean_val) / std_val).round(6)
            mensagens[var] = f"Padronizado (Média=0, DP=1). Nova coluna: '{col_new}'"

        elif transformacao == "center":
            col_new = _make_safe_name(f"{var}_centered")
            mean_val = float(non_na.mean())
            out[col_new] = (s - mean_val).round(6)
            mensagens[var] = f"Centralizado (Média=0). Nova coluna: '{col_new}'"

    return out, mensagens


def extrair_sazonalidade(
    df: pd.DataFrame,
    colunas: List[str],
    data_inicio: str,
    data_fim: str,
    col_data: str = "date_ok",
) -> Tuple[pd.DataFrame, str, List[str]]:
    if col_data not in df.columns:
        raise ValueError(f"O conjunto de dados precisa ter a coluna '{col_data}'.")

    dt_ini = pd.Timestamp(data_inicio)
    dt_fim = pd.Timestamp(data_fim)
    periodo_meses = (dt_fim - dt_ini).days / 30.44
    if periodo_meses < 12:
        raise ValueError("O período selecionado deve ter pelo menos 12 meses para capturar a sazonalidade.")

    out = df.copy()
    dt_series = pd.to_datetime(out[col_data], errors="coerce")
    out["_dt_sort"] = dt_series
    out = out.sort_values("_dt_sort").reset_index(drop=True)
    dt_series = out["_dt_sort"]

    mask_periodo = (dt_series >= dt_ini) & (dt_series <= dt_fim)
    df_periodo = out.loc[mask_periodo].copy()
    if len(df_periodo) < 12:
        raise ValueError(
            f"Não há dados suficientes no período selecionado. Encontrados apenas {len(df_periodo)} registros (mínimo 12)."
        )

    added_cols: List[str] = []
    report_lines: List[str] = [
        "Extração de Sazonalidade e Tendência",
        f"Período analisado: {dt_ini.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}",
        f"Variáveis processadas: {', '.join(colunas)}",
        "",
    ]

    saz_lines: List[str] = []
    tend_lines: List[str] = []

    for col in colunas:
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue

        s_per = df_periodo[col].astype(float)
        if s_per.notna().sum() < 12:
            continue

        # Linear interpolation + boundary fill (matching zoo::na.approx + zoo::na.fill("extend"))
        valores = s_per.interpolate(method="linear").bfill().ffill().values

        # Classical seasonal decomposition with period=12 (or min half length if shorter)
        period = 12 if len(valores) >= 24 else max(2, len(valores) // 2)
        decomp = seasonal_decompose(valores, model="additive", period=period, extrapolate_trend="freq")

        seasonal = decomp.seasonal
        trend = decomp.trend

        # Map seasonal average by month (1..12) across full dataframe
        meses_per = df_periodo["_dt_sort"].dt.month.values
        saz_by_month = (
            pd.DataFrame({"mes": meses_per, "saz": seasonal})
            .groupby("mes")["saz"]
            .mean()
            .to_dict()
        )

        col_saz = f"{col}_sazonalidade"
        meses_all = out["_dt_sort"].dt.month
        out[col_saz] = meses_all.map(saz_by_month).round(6)
        added_cols.append(col_saz)

        # Fit linear trend model (tendencia ~ dataordinal) to interpolate/extrapolate across full df
        col_tend = f"{col}_tendencia"
        valid_t = ~np.isnan(trend)
        x_days_per = (df_periodo["_dt_sort"] - pd.Timestamp("1970-01-01")).dt.days.values[valid_t]
        y_trend = trend[valid_t]
        if len(x_days_per) > 1:
            slope, intercept = np.polyfit(x_days_per, y_trend, 1)
            x_days_all = (out["_dt_sort"] - pd.Timestamp("1970-01-01")).dt.days.values
            out[col_tend] = np.round(slope * x_days_all + intercept, 6)
            added_cols.append(col_tend)

        # Build detailed stats matching R app's info_sazonalidade
        s_orig = out[col].astype(float)
        total_var = float(s_orig.var(ddof=1)) if s_orig.notna().sum() > 5 else 0.0

        s_saz = out[col_saz].dropna()
        if len(s_saz) > 0:
            q1, med, q3 = np.percentile(s_saz, [25, 50, 75])
            saz_var = float(s_saz.var(ddof=1)) if len(s_saz) > 1 else 0.0
            saz_pct = round((saz_var / total_var) * 100, 2) if total_var > 0 else 0.0
            forca = "Fraca" if saz_pct < 15 else ("Moderada" if saz_pct < 35 else "Forte")
            saz_lines.extend([
                f"  * {col_saz}:",
                f"    - Mínimo: {s_saz.min():.4f} | 1º Quartil: {q1:.4f} | Mediana: {med:.4f}",
                f"    - Média: {s_saz.mean():.4f} | 3º Quartil: {q3:.4f} | Máximo: {s_saz.max():.4f}",
                f"    - Importância relativa: {saz_pct}% ({forca})",
                "",
            ])

        if col_tend in out.columns:
            s_t = out[col_tend].dropna()
            if len(s_t) > 0:
                q1, med, q3 = np.percentile(s_t, [25, 50, 75])
                t_var = float(s_t.var(ddof=1)) if len(s_t) > 1 else 0.0
                t_pct = round((t_var / total_var) * 100, 2) if total_var > 0 else 0.0
                forca_t = "Fraca" if t_pct < 15 else ("Moderada" if t_pct < 35 else "Forte")
                tend_lines.extend([
                    f"  * {col_tend}:",
                    f"    - Mínimo: {s_t.min():.4f} | 1º Quartil: {q1:.4f} | Mediana: {med:.4f}",
                    f"    - Média: {s_t.mean():.4f} | 3º Quartil: {q3:.4f} | Máximo: {s_t.max():.4f}",
                    f"    - Importância relativa: {t_pct}% ({forca_t})",
                    "",
                ])

    out = out.drop(columns=["_dt_sort"])

    if not added_cols:
        raise ValueError("Não foi possível extrair a sazonalidade e tendência para nenhuma das colunas selecionadas.")

    if saz_lines:
        report_lines.append("Colunas de sazonalidade adicionadas:")
        report_lines.extend(saz_lines)
    if tend_lines:
        report_lines.append("Colunas de tendência adicionadas:")
        report_lines.extend(tend_lines)

    return out, "\n".join(report_lines), added_cols


def serialize_dataframe_rows(df: pd.DataFrame, columns: Optional[List[str]] = None, limit: int = 5000) -> Dict[str, Any]:
    if df is None:
        return {"columns": [], "rows": [], "total_rows": 0, "numeric_columns": [], "date_columns": []}

    view_df = df
    if columns and "todas" not in columns:
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            view_df = df[valid_cols]

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and c != "date_ok"
    ]
    date_cols = find_candidate_date_columns(df)

    sliced = view_df.head(limit).copy()
    for c in sliced.columns:
        if pd.api.types.is_datetime64_any_dtype(sliced[c]):
            sliced[c] = pd.to_datetime(sliced[c]).dt.strftime("%Y-%m-%d")
        else:
            sliced[c] = sliced[c].apply(
                lambda v: v.strftime("%Y-%m-%d") if isinstance(v, (date, datetime)) else (None if pd.isna(v) else v)
            )

    rows = sliced.replace({np.nan: None}).to_dict(orient="records")
    return {
        "columns": list(view_df.columns),
        "all_columns": list(df.columns),
        "rows": rows,
        "total_rows": len(df),
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
    }
