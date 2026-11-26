"""Camada de persistência analítica DuckDB + Parquet e integração com APIs oficiais do BCB."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DUCKDB_PATH = DATA_DIR / "observatorio_macro.duckdb"
PARQUET_PATH = DATA_DIR / "bcb_series.parquet"

CORE_CONJUNTURA_SERIES: List[Tuple[int, str, str]] = [
    (4189, "Selic Mensal (% a.a.)", "% a.a."),
    (13522, "IPCA Acumulado 12m (%)", "% 12m"),
    (13521, "Meta de Inflação (%)", "% a.a."),
    (13762, "Dívida Bruta DBGG (% PIB)", "% PIB"),
    (3698, "Taxa de Câmbio (R$/US$)", "R$/US$"),
    (24364, "Atividade Econômica IBC-Br", "índice"),
    (24369, "Taxa de Desocupação PNAD (%)", "%"),
    (433, "IPCA Mensal (% m/m)", "% m/m"),
    (27574, "Commodities IC-Br", "índice"),
    (28763, "Saldo Novo CAGED", "vagas"),
]


def _connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    con: Optional[duckdb.DuckDBPyConnection] = None
    for attempt in range(4):
        try:
            con = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
            break
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(0.35 * (attempt + 1))
    if con is None:
        raise last_exc or RuntimeError("Não foi possível abrir o banco DuckDB.")
    if not read_only:
        con.execute("""
            CREATE TABLE IF NOT EXISTS bcb_series (
                series_key VARCHAR,
                date TIMESTAMP,
                value DOUBLE,
                updated_at TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS study_datasets (
                study_code VARCHAR,
                dataset_key VARCHAR,
                table_name VARCHAR PRIMARY KEY,
                rows INTEGER,
                cols INTEGER,
                ingested_at TIMESTAMP
            )
        """)
    return con


def upsert_bcb_series(series_key: str, df: pd.DataFrame) -> None:
    upsert_many_bcb_series({series_key: df})


def upsert_many_bcb_series(series_map: Dict[str, pd.DataFrame]) -> int:
    frames = []
    keys = []
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    for series_key, df in series_map.items():
        if df is None or df.empty:
            continue
        inc = pd.DataFrame({
            "series_key": str(series_key),
            "date": pd.to_datetime(df.index).tz_localize(None),
            "value": pd.to_numeric(df["value"], errors="coerce"),
            "updated_at": now_utc,
        }).dropna(subset=["date", "value"])
        if not inc.empty:
            frames.append(inc)
            keys.append(str(series_key))
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)

    con = _connect(read_only=False)
    try:
        con.register("incoming_batch", combined)
        for k in keys:
            con.execute("DELETE FROM bcb_series WHERE series_key = ?", [k])
        con.execute("INSERT INTO bcb_series SELECT series_key, date, value, updated_at FROM incoming_batch")
        pq_str = str(PARQUET_PATH).replace("\\", "/")
        con.execute(f"COPY (SELECT * FROM bcb_series ORDER BY series_key, date) TO '{pq_str}' (FORMAT PARQUET)")
    finally:
        con.close()
    return len(keys)


def load_bcb_series(series_key: str) -> Optional[pd.DataFrame]:
    if DUCKDB_PATH.exists():
        con = _connect(read_only=True)
        try:
            df = con.execute(
                "SELECT date, value FROM bcb_series WHERE series_key = ? ORDER BY date",
                [str(series_key)],
            ).df()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")
        finally:
            con.close()
    if PARQUET_PATH.exists():
        con = duckdb.connect(database=":memory:")
        try:
            pq_str = str(PARQUET_PATH).replace("\\", "/")
            df = con.execute(
                f"SELECT date, value FROM read_parquet('{pq_str}') WHERE series_key = ? ORDER BY date",
                [str(series_key)],
            ).df()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")
        finally:
            con.close()
    return None


def fetch_bcb_sgs(series_id: int, start_ddmmyyyy: str = "01/01/2012", force_live: bool = False) -> Tuple[Optional[pd.DataFrame], bool]:
    """Busca série na API SGS do BCB com cache local DuckDB + Parquet."""
    cache_key = f"sgs_{series_id}"
    if not force_live:
        cached = load_bcb_series(cache_key)
        if cached is not None and not cached.empty:
            return cached, False

    try:
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
            f"?formato=json&dataInicial={start_ddmmyyyy}"
        )
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            rows = resp.json()
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["value"] = pd.to_numeric(df["valor"], errors="coerce")
            out = df[["date", "value"]].dropna().set_index("date").sort_index()
            if not out.empty:
                upsert_bcb_series(cache_key, out)
                return out, True
    except Exception:
        pass

    cached = load_bcb_series(cache_key)
    return cached, False


def fetch_bcb_olinda_expinf(start_date: str = "2015-01-01", force_live: bool = False) -> Tuple[Optional[pd.Series], bool]:
    """Busca Expectativas de Inflação IPCA 12m (Boletim Focus — Olinda BCB)."""
    cache_key = "olinda_expinf_12m"
    if not force_live:
        cached = load_bcb_series(cache_key)
        if cached is not None and not cached.empty:
            return cached["value"], False

    try:
        url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoInflacao12Meses"
        params = {
            "$filter": f"Indicador eq 'IPCA' and Suavizada eq 'S' and Data ge '{start_date}'",
            "$format": "json",
            "$select": "Data,Mediana",
            "$top": "10000",
        }
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            rows = resp.json().get("value", [])
        if rows:
            edf = pd.DataFrame(rows)
            edf["date"] = pd.to_datetime(edf["Data"])
            edf["Mediana"] = pd.to_numeric(edf["Mediana"], errors="coerce")
            monthly = edf.set_index("date")["Mediana"].resample("MS").mean().dropna()
            if not monthly.empty:
                upsert_bcb_series(cache_key, monthly.to_frame("value"))
                return monthly, True
    except Exception:
        pass

    cached = load_bcb_series(cache_key)
    return (cached["value"] if cached is not None and not cached.empty else None), False


def load_study_table(study_code: str, dataset_key: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not DUCKDB_PATH.exists():
        return None, None
    con = _connect(read_only=True)
    try:
        if dataset_key:
            row = con.execute(
                "SELECT table_name, dataset_key FROM study_datasets WHERE study_code = ? AND dataset_key = ?",
                [study_code, dataset_key],
            ).fetchone()
        else:
            row = con.execute(
                "SELECT table_name, dataset_key FROM study_datasets WHERE study_code = ? ORDER BY rows DESC LIMIT 1",
                [study_code],
            ).fetchone()
        if not row:
            return None, None
        tname, dkey = row
        df = con.execute(f'SELECT * FROM "{tname}"').df()
        return df, dkey
    except Exception:
        return None, None
    finally:
        con.close()


def get_store_stats() -> Dict[str, Any]:
    if not DUCKDB_PATH.exists():
        return {
            "exists": False,
            "path": "data/observatorio_macro.duckdb",
            "series_count": 0,
            "tables_count": 0,
            "total_rows": 0,
            "last_updated": None,
            "size_mb": 0.0,
        }
    con = _connect(read_only=True)
    try:
        s_row = con.execute("SELECT COUNT(DISTINCT series_key), COUNT(*), MAX(updated_at) FROM bcb_series").fetchone()
        t_row = con.execute("SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM study_datasets").fetchone()
        return {
            "exists": True,
            "path": "data/observatorio_macro.duckdb",
            "series_count": int(s_row[0] or 0),
            "tables_count": int(t_row[0] or 0),
            "total_rows": int(s_row[1] or 0) + int(t_row[1] or 0),
            "last_updated": s_row[2].strftime("%Y-%m-%d %H:%M:%S") if s_row[2] else None,
            "size_mb": round(DUCKDB_PATH.stat().st_size / (1024.0 * 1024.0), 2),
        }
    finally:
        con.close()
