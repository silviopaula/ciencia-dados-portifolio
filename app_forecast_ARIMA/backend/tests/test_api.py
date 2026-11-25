"""Testes automatizados do Backend FastAPI + Motor SARIMAX."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestSarimaxBackendAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "online")

    def test_full_demo_workflow_diagnostics_forecast_and_export(self) -> None:
        # 1. Carrega dataset demo
        r_demo = self.client.post("/api/dataset/demo")
        self.assertEqual(r_demo.status_code, 200)
        meta = r_demo.json()
        self.assertEqual(meta["total_rows"], 360)
        self.assertEqual(meta["default_date_col"], "data")
        self.assertEqual(meta["default_target_col"], "consumo_energia_mwh")

        # 2. Diagnóstico Box-Jenkins (ADF, KPSS, ACF, PACF)
        r_diag = self.client.post(
            "/api/diagnostics",
            json={
                "dataset_id": meta["dataset_id"],
                "date_col": meta["default_date_col"],
                "target_col": meta["default_target_col"],
                "use_log": True,
                "diff_order": 1,
                "seasonal_diff": 0,
                "seasonal_period": 12,
                "nlags": 24,
            },
        )
        self.assertEqual(r_diag.status_code, 200)
        diag = r_diag.json()
        self.assertEqual(len(diag["correlations"]), 24)
        self.assertIn("p_value", diag["adf"])
        self.assertIn("p_value", diag["kpss"])

        # 3. Modelagem Auto-SARIMAX com variáveis exógenas reais
        r_fc = self.client.post(
            "/api/forecast",
            json={
                "dataset_id": meta["dataset_id"],
                "date_col": "data",
                "target_col": "consumo_energia_mwh",
                "exog_cols": ["Temperatura_media", "dias_uteis", "tarifa_energia"],
                "mode": "auto",
                "s": 12,
                "use_log": True,
                "test_periods": 12,
                "forecast_horizon": 24,
            },
        )
        self.assertEqual(r_fc.status_code, 200)
        fc = r_fc.json()
        self.assertIn("SARIMAX", fc["model_label"])
        self.assertLess(fc["test_metrics"]["mape"], 10.0)
        self.assertGreater(len(fc["coefficients"]), 3)
        self.assertGreater(len(fc["annual_growth"]), 3)

        # 4. Exportação para Excel (.xlsx)
        r_xlsx = self.client.get(f"/api/export/{meta['dataset_id']}")
        self.assertEqual(r_xlsx.status_code, 200)
        self.assertIn("spreadsheetml", r_xlsx.headers["content-type"])
        self.assertGreater(len(r_xlsx.content), 3000)

    def test_upload_csv_with_nans_and_br_decimals(self) -> None:
        csv_content = (
            "data;vendas;preco\n"
            "2023-01-01;1.250,50;\n"
            "2023-02-01;;12,40\n"
            "2023-03-01;1.310,75;12,60\n"
        ).encode("latin-1")
        r = self.client.post(
            "/api/dataset/upload",
            files={"file": ("vendas_br.csv", csv_content, "text/csv")},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["total_rows"], 3)
        self.assertIn("vendas", body["numeric_columns"])
        self.assertIsNone(body["preview"][0]["preco"])
        self.assertIsNone(body["preview"][1]["vendas"])
        self.assertAlmostEqual(body["preview"][0]["vendas"], 1250.5)


if __name__ == "__main__":
    unittest.main()
