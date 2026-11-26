"""Download reprodutível da base pública Yelp Reviews (50.000 avaliações 1–5★) em Parquet."""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "yelp_reviews_spark.parquet"
YELP_PARQUET_URL = "https://huggingface.co/api/datasets/Yelp/yelp_review_full/parquet/yelp_review_full/test/0.parquet"


def ensure_yelp_parquet(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PARQUET_PATH.exists() and not force:
        return PARQUET_PATH
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(YELP_PARQUET_URL)
        resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    df["stars"] = df["label"].astype(int) + 1
    df = df[["stars", "text"]].copy()
    df["review_id"] = [f"yelp_{i:06d}" for i in range(len(df))]
    df.to_parquet(PARQUET_PATH, index=False, compression="zstd")
    return PARQUET_PATH


if __name__ == "__main__":
    p = ensure_yelp_parquet(force=False)
    df = pd.read_parquet(p)
    print(f"Base Yelp pronta: {p} ({p.stat().st_size / 1e6:.2f} MB | {len(df):,} reviews)")
    print(df["stars"].value_counts().sort_index())
