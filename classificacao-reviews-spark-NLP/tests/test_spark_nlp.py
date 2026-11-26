"""Testes unitários do pipeline distribuído Spark 4.0 MLlib — classificacao-reviews-spark-NLP."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from spark_nlp_mllib import (
    ENGLISH_NEGATIONS_KEEP,
    build_safe_english_stopwords,
    compute_binary_metrics,
    create_spark_session,
    add_stylistic_features_spark,
    stratified_split_spark,
)


class TestSparkNLPPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = create_spark_session("TestSparkNLP")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_01_negation_preserved_in_stopwords(self) -> None:
        sw = set(build_safe_english_stopwords())
        for neg in ("not", "no", "never", "nor", "neither", "don't", "didn't"):
            self.assertNotIn(neg, sw)
        self.assertIn("the", sw)
        self.assertTrue(len(ENGLISH_NEGATIONS_KEEP) >= 20)

    def _sdf_from_rows(self, rows, cols):
        """Carrega DataFrame de teste via Parquet temporário (100% JVM, sem Python worker)."""
        pdf = pd.DataFrame(rows, columns=cols)
        tmp = Path(tempfile.gettempdir()) / f"spark_nlp_test_{id(rows)}.parquet"
        pdf.to_parquet(tmp, index=False)
        return self.spark.read.parquet(str(tmp)), tmp

    def test_02_stylistic_features_and_target_mapping(self) -> None:
        rows = [
            ("r1", 5, "AMAZING food! Best service ever!"),
            ("r2", 4, "Good place, enjoyed it."),
            ("r3", 3, "Average meal, nothing special?"),
            ("r4", 1, "TERRIBLE! Never coming back!"),
        ]
        sdf, tmp = self._sdf_from_rows(rows, ["review_id", "stars", "text"])
        try:
            out = add_stylistic_features_spark(sdf).orderBy("review_id").toPandas()
        finally:
            tmp.unlink(missing_ok=True)
        self.assertEqual(list(out["label"]), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(out.loc[0, "excl_count"], 2.0)
        self.assertEqual(out.loc[2, "quest_count"], 1.0)
        self.assertGreater(out.loc[0, "upper_ratio"], out.loc[1, "upper_ratio"])

    def test_03_stratified_split_disjoint(self) -> None:
        rows = [(f"id_{i}", (i % 5) + 1, "sample review text here") for i in range(100)]
        sdf, tmp = self._sdf_from_rows(rows, ["review_id", "stars", "text"])
        try:
            tr, va, te = stratified_split_spark(sdf, seed=42)
            tr_ids = {r[0] for r in tr.select("review_id").collect()}
            va_ids = {r[0] for r in va.select("review_id").collect()}
            te_ids = {r[0] for r in te.select("review_id").collect()}
        finally:
            tmp.unlink(missing_ok=True)
        self.assertEqual(len(tr_ids) + len(va_ids) + len(te_ids), 100)
        self.assertTrue(tr_ids.isdisjoint(va_ids))
        self.assertTrue(tr_ids.isdisjoint(te_ids))
        self.assertTrue(va_ids.isdisjoint(te_ids))

    def test_04_binary_metrics_bootstrap(self) -> None:
        yt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        yp = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        ys = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.85, 0.95])
        m = compute_binary_metrics(yt, yp, ys, n_boot=100, seed=42)
        self.assertGreater(m["roc_auc"], 0.85)
        self.assertLessEqual(m["roc_ci_low"], m["roc_auc"])
        self.assertGreaterEqual(m["roc_ci_high"], m["roc_auc"])


if __name__ == "__main__":
    unittest.main()
