"""Cenários econométricos demonstrativos e ingestão inteligente de CSV/Excel."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEMO_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "varejo_sazonal_i1",
        "title": "Vendas Varejo — Tendência, Sazonalidade & Ciclo",
        "subtitle": "Série mensal clássica (2005–2025, 252 obs.) com drift crescente, sazonalidade anual marcada e ciclo de 5 anos.",
        "regime_badge": "Não-Estacionária I(1) + Sazonal",
        "frequency": "Mensal (s=12)",
        "seasonal_period": 12,
        "expected_diagnosis": "Rejeita estacionariedade no nível (KPSS/ADF/PP); estabiliza após 1ª diferença ou diferença sazonal.",
        "recommended_transform": "diff1_seasonal",
    },
    {
        "id": "retornos_garch_arch",
        "title": "Retornos Financeiros — Clusters de Volatilidade (GARCH)",
        "subtitle": "Retornos logarítmicos semanais (260 obs.) estacionários na média, mas com forte volatilidade agrupada e caudas pesadas.",
        "regime_badge": "Estacionária + Efeito ARCH",
        "frequency": "Semanal (s=52)",
        "seasonal_period": 12,
        "expected_diagnosis": "Passa ADF/PP/KPSS (estacionária), mas falha em Engle ARCH-LM e Normalidade (Jarque-Bera / Anderson-Darling).",
        "recommended_transform": "level",
    },
    {
        "id": "quebra_estrutural_za",
        "title": "Atividade Econômica — Quebra Estrutural (Estilo 2020)",
        "subtitle": "Série mensal (2010–2025, 192 obs.) estacionária em torno de tendência, porém atingida por choque permanente de nível em Abr/2020.",
        "regime_badge": "Quebra Estrutural (Zivot-Andrews)",
        "frequency": "Mensal (s=12)",
        "seasonal_period": 12,
        "expected_diagnosis": "Engana o ADF clássico (viés de Perron a favor de raiz unitária), mas Zivot-Andrews detecta a data exata da quebra!",
        "recommended_transform": "diff1",
    },
    {
        "id": "consumo_variancia_crescente",
        "title": "Tráfego & Receita — Variância Multiplicativa Crescente",
        "subtitle": "Série mensal exponencial (2012–2025, 168 obs.) cuja amplitude sazonal e dispersão crescem proporcionalmente ao nível.",
        "regime_badge": "Heterocedástica (Exige Log/Box-Cox)",
        "frequency": "Mensal (s=12)",
        "seasonal_period": 12,
        "expected_diagnosis": "Apresenta forte heterocedasticidade temporal no nível; transformação Log Natural ou Box-Cox estabiliza a variância.",
        "recommended_transform": "log_diff",
    },
    {
        "id": "ar1_estacionario_controle",
        "title": "Processo AR(1) Gaussiano — Controle Estacionário Limpo",
        "subtitle": "Série estacionária bem-comportada (200 obs., φ = 0,55, ruído Gaussiano homocedástico) ideal para calibração didática.",
        "regime_badge": "Controle Estacionário Limpo",
        "frequency": "Mensal (s=12)",
        "seasonal_period": 12,
        "expected_diagnosis": "Passa ADF, PP e KPSS no nível; após AR(1), resíduos passam Ljung-Box, ARCH-LM e Normalidade.",
        "recommended_transform": "level",
    },
]


def get_demo_catalog() -> List[Dict[str, Any]]:
    out = []
    for item in DEMO_CATALOG:
        df, _ = load_demo_dataset(item["id"])
        out.append({**item, "n_obs": len(df)})
    return out


def load_demo_dataset(dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Gera deterministicamente um dos 5 cenários econométricos de demonstração."""
    meta = next((m for m in DEMO_CATALOG if m["id"] == dataset_id), None)
    if meta is None:
        raise ValueError(f"Dataset demonstrativo desconhecido: '{dataset_id}'.")

    if dataset_id == "varejo_sazonal_i1":
        dates = pd.date_range("2005-01-01", periods=252, freq="MS")
        rng = np.random.default_rng(2025)
        n = len(dates)
        t = np.arange(1, n + 1, dtype=float)
        # Tendência estocástica suave I(1) (ARIMA(1,1,0) com drift) + sazonalidade anual marcada
        deltas = np.zeros(n)
        shocks = rng.normal(0.35, 3.2, size=n)
        for i in range(1, n):
            deltas[i] = 0.65 * deltas[i - 1] + shocks[i]
        rw_trend = 220.0 + np.cumsum(deltas)
        seasonal = 20.0 * np.sin(2.0 * np.pi * t / 12.0) + 9.0 * np.cos(2.0 * np.pi * t / 12.0)
        noise = rng.normal(0.0, 2.5, size=n)
        values = np.maximum(rw_trend + seasonal + noise, 25.0)
        df = pd.DataFrame({"Date": dates, "Value": np.round(values, 2)})
        return df, meta

    if dataset_id == "retornos_garch_arch":
        dates = pd.date_range("2020-01-05", periods=260, freq="W-SUN")
        rng = np.random.default_rng(42)
        n = len(dates)
        # Dois regimes de volatilidade (calmaria vs estresse de mercado) + dinâmica GARCH(1,1) estável
        base_sigma = np.full(n, 0.70)
        base_sigma[40:75] = 2.80
        base_sigma[155:195] = 3.20
        z = rng.normal(0.0, 1.0, size=n)
        eps = base_sigma * z
        values = 0.08 + eps
        df = pd.DataFrame({"Date": dates, "Value": np.round(values, 4)})
        return df, meta

    if dataset_id == "quebra_estrutural_za":
        dates = pd.date_range("2010-01-01", periods=192, freq="MS")
        rng = np.random.default_rng(2020)
        n = len(dates)
        t = np.arange(n, dtype=float)
        break_idx = 123  # 2020-04-01
        ar_err = np.zeros(n)
        e = rng.normal(0.0, 2.2, size=n)
        for i in range(1, n):
            ar_err[i] = 0.45 * ar_err[i - 1] + e[i]
        level_shift = np.where(t >= break_idx, -26.0, 0.0)
        slope_shift = np.where(t >= break_idx, 0.18 * (t - break_idx), 0.0)
        values = 115.0 + 0.22 * t + level_shift + slope_shift + ar_err
        df = pd.DataFrame({"Date": dates, "Value": np.round(values, 2)})
        return df, meta

    if dataset_id == "consumo_variancia_crescente":
        dates = pd.date_range("2012-01-01", periods=168, freq="MS")
        rng = np.random.default_rng(77)
        t = np.arange(1, len(dates) + 1, dtype=float)
        log_level = 4.2 + 0.0125 * t
        log_seas = 0.16 * np.sin(2.0 * np.pi * t / 12.0) + 0.07 * np.cos(2.0 * np.pi * t / 12.0)
        log_noise = rng.normal(0.0, 0.045, size=len(dates))
        values = np.exp(log_level + log_seas + log_noise)
        df = pd.DataFrame({"Date": dates, "Value": np.round(values, 2)})
        return df, meta

    # ar1_estacionario_controle
    dates = pd.date_range("2009-01-01", periods=200, freq="MS")
    rng = np.random.default_rng(99)
    n = len(dates)
    y = np.zeros(n)
    eps = rng.normal(0.0, 2.5, size=n)
    for i in range(1, n):
        y[i] = 50.0 * (1.0 - 0.55) + 0.55 * y[i - 1] + eps[i]
    y[0] = 50.0 + eps[0]
    df = pd.DataFrame({"Date": dates, "Value": np.round(y, 3)})
    return df, meta


def detect_csv_separator(raw_bytes: bytes) -> Tuple[str, str]:
    """Detecta encoding (UTF-8 / Latin-1) e separador (`,`, `;`, `\\t`)."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw_bytes[:8192].decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes[:8192].decode("latin-1", errors="ignore")
        encoding = "latin-1"

    lines = [ln for ln in text.splitlines()[:10] if ln.strip()]
    sample = "\n".join(lines)
    counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = ","
    return sep, encoding


def parse_uploaded_file(
    raw_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lê CSV ou Excel preservando abas, separador e tipos numéricos brasileiros (vírgula decimal)."""
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(raw_bytes))
        sheets = list(xl.sheet_names)
        chosen = sheet_name if sheet_name in sheets else sheets[0]
        df = xl.parse(chosen)
        info = {
            "filename": filename,
            "file_type": "excel",
            "detected_separator": None,
            "sheets": sheets,
            "selected_sheet": chosen,
        }
    else:
        sep, enc = detect_csv_separator(raw_bytes)
        decimal = "," if sep == ";" else "."
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding=enc, decimal=decimal)
        info = {
            "filename": filename,
            "file_type": "csv",
            "detected_separator": sep,
            "sheets": [],
            "selected_sheet": None,
        }

    # Limpa nomes de colunas
    df.columns = [str(c).strip() for c in df.columns]
    return df, info


def infer_columns_and_period(df: pd.DataFrame) -> Tuple[str, str, List[str], int]:
    """Sugere coluna de data, coluna numérica de valor e período sazonal."""
    cols = list(df.columns)
    numeric_cols: List[str] = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            coerced = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if coerced.notna().mean() >= 0.75:
                numeric_cols.append(c)

    date_col = cols[0]
    for c in cols:
        lc = c.lower()
        if any(tok in lc for tok in ("date", "data", "mes", "mês", "ano", "time", "periodo", "period", "dt")):
            date_col = c
            break
    else:
        non_num = [c for c in cols if c not in numeric_cols]
        if non_num:
            date_col = non_num[0]

    val_candidates = [c for c in numeric_cols if c != date_col]
    value_col = val_candidates[0] if val_candidates else (cols[1] if len(cols) > 1 else cols[0])

    # Infere período sazonal a partir da mediana de dias entre datas
    period = 12
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
        if len(parsed) >= 6:
            med_days = float(parsed.diff().dt.days.dropna().median())
            if med_days <= 2.5:
                period = 7
            elif med_days <= 10:
                period = 12  # semanal / curto
            elif med_days <= 45:
                period = 12
            elif med_days <= 110:
                period = 4
            else:
                period = 1
    except Exception:
        period = 12

    return date_col, value_col, numeric_cols, period


def build_clean_series_df(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    date_format: str = "auto",
    aggregation: str = "none",
) -> pd.DataFrame:
    """Converte colunas de data e valor em DataFrame limpo ordenado (`Date`, `Value`)."""
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Colunas '{date_col}' ou '{value_col}' não encontradas no arquivo.")

    work = df[[date_col, value_col]].copy()
    if date_format == "dmy":
        work["Date"] = pd.to_datetime(work[date_col], dayfirst=True, errors="coerce")
    elif date_format == "mdy":
        work["Date"] = pd.to_datetime(work[date_col], dayfirst=False, errors="coerce")
    elif date_format == "ymd":
        work["Date"] = pd.to_datetime(work[date_col], format="%Y-%m-%d", errors="coerce")
    else:
        work["Date"] = pd.to_datetime(work[date_col], errors="coerce")
        if work["Date"].isna().mean() > 0.5:
            work["Date"] = pd.to_datetime(work[date_col], dayfirst=True, errors="coerce")

    if pd.api.types.is_numeric_dtype(work[value_col]):
        work["Value"] = pd.to_numeric(work[value_col], errors="coerce")
    else:
        cleaned_str = (
            work[value_col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.(?=\d{3}(?:,|$))", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        work["Value"] = pd.to_numeric(cleaned_str, errors="coerce")

    work = work[["Date", "Value"]].dropna().sort_values("Date").reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(
            f"A série limpa possui apenas {len(work)} observações válidas (mínimo exigido: 20). "
            "Verifique as colunas selecionadas e o formato de data."
        )

    if aggregation == "month_mean":
        work["Date"] = work["Date"].dt.to_period("M").dt.to_timestamp()
        work = work.groupby("Date", as_index=False)["Value"].mean()
    elif aggregation == "month_sum":
        work["Date"] = work["Date"].dt.to_period("M").dt.to_timestamp()
        work = work.groupby("Date", as_index=False)["Value"].sum()
    else:
        # Remove datas duplicadas tirando a média para evitar quebra em STL/ARIMA
        work = work.groupby("Date", as_index=False)["Value"].mean()

    work = work.sort_values("Date").reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(f"Após agregação restaram apenas {len(work)} períodos (mínimo: 20).")
    return work
