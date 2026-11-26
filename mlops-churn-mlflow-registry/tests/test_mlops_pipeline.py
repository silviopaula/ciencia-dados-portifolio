"""Testes unitários e de integração MLOps (OOT Split, Feature Engineering, PSI, ROI e MLflow Registry)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_mlops import (
    ChurnFeatureEngineer,
    chronological_oot_split,
    compute_psi,
    optimize_retention_roi,
)


class TestChurnMLOpsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(PROJECT_ROOT / "data" / "abt.csv")

    def test_01_chronological_oot_split_no_temporal_leakage(self) -> None:
        tr, va, te = chronological_oot_split(self.df)
        self.assertEqual(len(tr) + len(va) + len(te), len(self.df))
        self.assertLess(pd.to_datetime(tr["dt_ref"]).max(), pd.to_datetime(va["dt_ref"]).min())
        self.assertLess(pd.to_datetime(va["dt_ref"]).max(), pd.to_datetime(te["dt_ref"]).min())
        self.assertGreater(len(te), 400)

    def test_02_feature_engineer_eliminates_nans_and_adds_ratios(self) -> None:
        tr, _, te = chronological_oot_split(self.df)
        feat_cols = [c for c in self.df.columns if c not in ("dt_ref", "id_cliente", "flag_churn")]
        eng = ChurnFeatureEngineer()
        X_tr = eng.fit_transform(tr[feat_cols])
        X_te = eng.transform(te[feat_cols])
        self.assertEqual(int(X_tr.isna().sum().sum()), 0)
        self.assertEqual(int(X_te.isna().sum().sum()), 0)
        # Garantia de Zero Training-Serving Skew: batch grande vs 1 linha gera o MESMO resultado!
        X_te_one = eng.transform(te[feat_cols].iloc[:1])
        self.assertLess(float(np.max(np.abs(X_te.iloc[0].to_numpy() - X_te_one.iloc[0].to_numpy()))), 1e-12)
        for col in ("inativo_d7", "inativo_d14", "ratio_iter_d7_d28", "gap_ultima_vs_recorrencia"):
            self.assertIn(col, X_tr.columns)

    def test_03_psi_detects_drift_vs_stability(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.normal(0.0, 1.0, size=1000)
        same = rng.normal(0.0, 1.0, size=1000)
        shifted = rng.normal(1.2, 1.4, size=1000)
        self.assertLess(compute_psi(base, same), 0.10)
        self.assertGreater(compute_psi(base, shifted), 0.25)

    def test_04_roi_threshold_optimization(self) -> None:
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.55, 0.65, 0.75, 0.85, 0.95])
        roi = optimize_retention_roi(y_true, y_prob, ltv=450.0, contact_cost=45.0, save_rate=0.30)
        self.assertGreater(roi["best_net_profit"], roi["treat_all_profit"])
        self.assertGreater(roi["best_threshold"], 0.15)
        self.assertLess(roi["best_threshold"], 0.70)

    def test_05_mlflow_sqlite_registry_and_alias_roundtrip(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlflow_test_"))
        try:
            db_uri = f"sqlite:///{(tmp_dir / 'test_mlflow.db').as_posix()}"
            mlflow.set_tracking_uri(db_uri)
            mlflow.set_experiment("unit_test_churn")

            tr, _, te = chronological_oot_split(self.df)
            feat_cols = [c for c in self.df.columns if c not in ("dt_ref", "id_cliente", "flag_churn")]
            pipe = Pipeline([
                ("eng", ChurnFeatureEngineer()),
                ("scl", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ])
            pipe.fit(tr[feat_cols].iloc[:300], tr["flag_churn"].iloc[:300])
            sample_in = te[feat_cols].iloc[:5].copy()
            sig = infer_signature(sample_in, pipe.predict_proba(sample_in)[:, 1])

            with mlflow.start_run(run_name="test_lr") as run:
                mlflow.log_metric("oot_roc_auc", 0.81)
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    name="model",
                    signature=sig,
                    registered_model_name="unit-churn-model",
                )
                run_id = run.info.run_id

            client = MlflowClient()
            versions = client.search_model_versions("name='unit-churn-model'")
            self.assertEqual(len(versions), 1)
            client.set_registered_model_alias("unit-churn-model", "champion", versions[0].version)

            loaded = mlflow.sklearn.load_model("models:/unit-churn-model@champion")
            preds = loaded.predict_proba(sample_in)[:, 1]
            self.assertEqual(len(preds), 5)
            self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))
            self.assertTrue(bool(run_id))
        finally:
            mlflow.set_tracking_uri("")
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
