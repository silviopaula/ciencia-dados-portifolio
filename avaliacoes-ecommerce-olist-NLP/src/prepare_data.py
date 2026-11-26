"""Pipeline de ingestão e consolidação da base oficial Olist Brazilian E-Commerce.

Baixa as 5 tabelas oficiais públicas da Olist (GitHub oficial `olist/work-at-olist-data`),
agrega itens por `order_id` para eliminar duplicação cartesiana (group leakage) e salva
o dataset consolidado em formato colunar Parquet (`data/olist_reviews_nlp.parquet`).
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import httpx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PARQUET = DATA_DIR / "olist_reviews_nlp.parquet"

OLIST_BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets"
TABLES = {
    "reviews": f"{OLIST_BASE_URL}/olist_order_reviews_dataset.csv",
    "orders": f"{OLIST_BASE_URL}/olist_orders_dataset.csv",
    "items": f"{OLIST_BASE_URL}/olist_order_items_dataset.csv",
    "products": f"{OLIST_BASE_URL}/olist_products_dataset.csv",
    "customers": f"{OLIST_BASE_URL}/olist_customers_dataset.csv",
}

UF_TO_REGION: Dict[str, str] = {
    "SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "ES": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "BA": "Nordeste", "PE": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "RN": "Nordeste", "AL": "Nordeste", "PI": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "PA": "Norte", "AM": "Norte", "RO": "Norte", "TO": "Norte", "AC": "Norte", "AP": "Norte", "RR": "Norte",
}


def _download_csv(name: str, url: str) -> tuple[str, pd.DataFrame]:
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))
    return name, df


def build_olist_nlp_dataset(force: bool = False) -> pd.DataFrame:
    """Constrói e persiste o dataset Olist consolidado em nível de avaliação única."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PARQUET.exists() and not force:
        return pd.read_parquet(OUTPUT_PARQUET)

    raw: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_download_csv, k, u) for k, u in TABLES.items()]
        for fut in futs:
            name, df = fut.result()
            raw[name] = df

    reviews = raw["reviews"].copy()
    orders = raw["orders"].copy()
    items = raw["items"].copy()
    products = raw["products"].copy()
    customers = raw["customers"].copy()

    # 1. Junta produtos nos itens e agrega estritamente no grão `order_id` (evita group leakage m:m)
    items_prod = items.merge(
        products[["product_id", "product_category_name", "product_weight_g", "product_photos_qty"]],
        on="product_id",
        how="left",
    )
    order_agg = (
        items_prod.groupby("order_id", as_index=False)
        .agg(
            qtd_itens=("order_item_id", "count"),
            qtd_vendedores=("seller_id", "nunique"),
            valor_itens=("price", "sum"),
            frete_total=("freight_value", "sum"),
            peso_medio_g=("product_weight_g", "mean"),
            fotos_produto=("product_photos_qty", "mean"),
            categoria_principal=("product_category_name", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "nao_informada"),
        )
    )
    order_agg["valor_total"] = order_agg["valor_itens"] + order_agg["frete_total"]
    order_agg["razao_frete"] = np.where(
        order_agg["valor_itens"] > 0,
        order_agg["frete_total"] / order_agg["valor_itens"],
        np.nan,
    )

    # 2. Enriquecimento logístico do pedido + estado do cliente
    for col in (
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ):
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["tempo_entrega_dias"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    orders["atraso_entrega_dias"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0
    orders["prazo_estimado_dias"] = (
        orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0
    orders["entregue_com_atraso"] = (orders["atraso_entrega_dias"] > 0).astype(int)
    orders["pedido_nao_entregue"] = orders["order_delivered_customer_date"].isna().astype(int)

    orders_full = (
        orders[
            [
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "tempo_entrega_dias",
                "atraso_entrega_dias",
                "prazo_estimado_dias",
                "entregue_com_atraso",
                "pedido_nao_entregue",
            ]
        ]
        .merge(customers[["customer_id", "customer_state", "customer_city"]], on="customer_id", how="left")
        .merge(order_agg, on="order_id", how="left")
    )
    orders_full["regiao_br"] = orders_full["customer_state"].map(UF_TO_REGION).fillna("Nao Informada")

    # 3. Consolida avaliações únicas por (review_id, order_id) preservando reviews com e sem título
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")
    reviews = (
        reviews.sort_values("review_creation_date")
        .drop_duplicates(subset=["order_id"], keep="last")
        .reset_index(drop=True)
    )

    df = reviews.merge(orders_full, on="order_id", how="inner")

    # 4. Campos textuais combinados (título + comentário) e indicadores de presença de texto
    title_clean = df["review_comment_title"].fillna("").astype(str).str.strip()
    msg_clean = df["review_comment_message"].fillna("").astype(str).str.strip()
    df["tem_titulo"] = (title_clean.str.len() > 0).astype(int)
    df["tem_comentario"] = (msg_clean.str.len() > 0).astype(int)
    df["texto_completo"] = np.where(
        (title_clean.str.len() > 0) & (msg_clean.str.len() > 0),
        title_clean + ". " + msg_clean,
        np.where(msg_clean.str.len() > 0, msg_clean, title_clean),
    )
    df["comprimento_chars"] = df["texto_completo"].str.len()
    df["qtd_palavras"] = df["texto_completo"].str.split().str.len()

    # 5. Targets verdadeiros derivados da nota real dada pelo cliente (review_score 1..5)
    df["sentimento_real"] = np.select(
        [df["review_score"] <= 2, df["review_score"] == 3, df["review_score"] >= 4],
        ["Negativo", "Neutro", "Positivo"],
        default="Neutro",
    )
    df["detrator"] = (df["review_score"] <= 2).astype(int)

    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df.to_parquet(OUTPUT_PARQUET, index=False, compression="zstd")
    return df


if __name__ == "__main__":
    data = build_olist_nlp_dataset(force=True)
    size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    text_df = data[data["texto_completo"].str.len() >= 3]
    print(f"Dataset salvo em: {OUTPUT_PARQUET} ({size_mb:.2f} MB)")
    print(f"Total de pedidos avaliados únicos: {len(data):,}")
    print(f"Total com texto escrito (>=3 chars): {len(text_df):,} ({len(text_df)/len(data)*100:.1f}%)")
    print(f"Período de compras: {data['order_purchase_timestamp'].min()} -> {data['order_purchase_timestamp'].max()}")
    print("\nDistribuição review_score (todos os pedidos):")
    print(data["review_score"].value_counts().sort_index())
    print("\nDistribuição sentimento_real (somente com texto escrito):")
    print(text_df["sentimento_real"].value_counts())
    print("\nTaxa de atraso por sentimento (com texto):")
    print(text_df.groupby("sentimento_real")["entregue_com_atraso"].mean().round(4))
    print("\nValores nulos (%):")
    print((data.isna().mean() * 100).round(2)[lambda s: s > 0])
