"""Inferência em lote consumindo diretamente o modelo `@champion` no MLflow Model Registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn_mlops import DEFAULT_MODEL_NAME, get_tracking_uri  # noqa: E402


def score_batch(
    input_csv: Path,
    output_csv: Path,
    alias: str = "champion",
    threshold: float = 0.45,
) -> pd.DataFrame:
    """Carrega `models:/churn-predictor@<alias>` do Registry e pontua um lote bruto (zero skew)."""
    mlflow.set_tracking_uri(get_tracking_uri())
    model_uri = f"models:/{DEFAULT_MODEL_NAME}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)

    raw_df = pd.read_csv(input_csv)
    feature_cols = [c for c in raw_df.columns if c not in ("dt_ref", "id_cliente", "flag_churn")]
    probs = model.predict_proba(raw_df[feature_cols])[:, 1]

    out = pd.DataFrame({
        "dt_ref": raw_df["dt_ref"] if "dt_ref" in raw_df.columns else "",
        "id_cliente": raw_df["id_cliente"] if "id_cliente" in raw_df.columns else range(len(raw_df)),
        "prob_churn": probs.round(4),
        "pred_churn": (probs >= threshold).astype(int),
        "faixa_risco": pd.cut(
            probs,
            bins=[-0.01, 0.25, 0.50, 0.75, 1.01],
            labels=["Baixo", "Moderado", "Alto", "Crítico"],
        ).astype(str),
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Scoring concluído via `{model_uri}` | {len(out):,} clientes -> {output_csv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência em lote via MLflow Model Registry")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "abt.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "scored_sample.csv")
    parser.add_argument("--alias", type=str, default="champion")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()
    score_batch(args.input, args.output, alias=args.alias, threshold=args.threshold)


if __name__ == "__main__":
    main()
