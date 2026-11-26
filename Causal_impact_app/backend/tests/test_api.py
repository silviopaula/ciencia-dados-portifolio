"""Suíte de testes automatizados (unittest + TestClient) da API CausalImpact Studio."""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestCausalImpactAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_01_health_and_demos(self) -> None:
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        hd = h.json()
        self.assertEqual(hd["app_name"], "CausalImpact Studio API")
        self.assertTrue(hd["dataset_loaded"])
        self.assertEqual(hd["demos_available"], 3)

        demos = self.client.get("/api/demos")
        self.assertEqual(demos.status_code, 200)
        self.assertEqual(len(demos.json()), 3)

    def test_02_marketing_demo_positive_causal_effect_and_placebo(self) -> None:
        r_load = self.client.post("/api/dataset/demo/marketing_campanha")
        self.assertEqual(r_load.status_code, 200)
        meta = r_load.json()
        self.assertEqual(meta["rows"], 80)

        res = self.client.post(
            "/api/analyze",
            json={
                "date_col": "data",
                "y_col": "vendas_mil_brl",
                "x_cols": ["busca_organica_setor", "indice_varejo", "vendas_regiao_controle"],
                "intervention_date": "2025-02-02",
                "alpha": 0.05,
                "model_type": "local_level",
                "n_sims": 800,
                "run_placebo": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        m = data["metrics"]
        self.assertTrue(m["is_significant"])
        self.assertEqual(m["direction"], "positivo")
        self.assertGreater(m["avg_abs_effect"], 15.0)
        self.assertLess(m["avg_abs_effect"], 28.0)
        self.assertGreater(data["pre_fit"]["r2_pre"], 0.90)
        self.assertTrue(data["placebo"]["executed"])
        self.assertTrue(data["placebo"]["passed"])
        self.assertEqual(len(data["covariates"]), 3)
        self.assertIn("month", data["aggregated_tables"])

    def test_03_brodersen_classic_exact_recovery(self) -> None:
        self.assertEqual(self.client.post("/api/dataset/demo/brodersen_classico").status_code, 200)
        res = self.client.post(
            "/api/analyze",
            json={
                "date_col": "data",
                "y_col": "y",
                "x_cols": ["x1", "x2"],
                "intervention_date": "2022-10-08",
                "alpha": 0.05,
                "model_type": "local_level",
                "n_sims": 800,
                "run_placebo": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        m = res.json()["metrics"]
        self.assertTrue(m["is_significant"])
        # Efeito verdadeiro = +10.0 por semana
        self.assertAlmostEqual(m["avg_abs_effect"], 10.0, delta=2.0)

    def test_04_placebo_null_effect_correctly_not_significant(self) -> None:
        self.assertEqual(self.client.post("/api/dataset/demo/placebo_nulo").status_code, 200)
        res = self.client.post(
            "/api/analyze",
            json={
                "date_col": "data",
                "y_col": "receita_mensal",
                "x_cols": ["indice_macro_setor", "mercado_controle"],
                "intervention_date": "2024-10-01",
                "alpha": 0.05,
                "model_type": "local_level",
                "n_sims": 800,
                "run_placebo": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        m = res.json()["metrics"]
        self.assertFalse(m["is_significant"])
        self.assertEqual(m["direction"], "neutro")
        self.assertGreater(m["p_value"], 0.05)

    def test_05_univariate_and_stl_and_seasonal_model(self) -> None:
        self.assertEqual(self.client.post("/api/dataset/demo/placebo_nulo").status_code, 200)
        r_uni = self.client.post(
            "/api/analyze",
            json={
                "date_col": "data",
                "y_col": "receita_mensal",
                "x_cols": [],
                "intervention_date": "2024-10-01",
                "model_type": "seasonal_local_level",
                "seasonal_period": 12,
                "deseasonalize": "stl_deseasonalize",
                "stl_period": 12,
                "n_sims": 400,
                "run_placebo": False,
            },
        )
        self.assertEqual(r_uni.status_code, 200)
        self.assertTrue(r_uni.json()["univariate_mode"])
        self.assertGreaterEqual(len(r_uni.json()["warnings"]), 1)

    def test_06_exports_excel_csv_and_markdown(self) -> None:
        self.assertEqual(self.client.post("/api/dataset/demo/marketing_campanha").status_code, 200)
        self.assertEqual(
            self.client.post(
                "/api/analyze",
                json={
                    "date_col": "data",
                    "y_col": "vendas_mil_brl",
                    "x_cols": ["busca_organica_setor", "indice_varejo", "vendas_regiao_controle"],
                    "intervention_date": "2025-02-02",
                    "n_sims": 400,
                },
            ).status_code,
            200,
        )
        rx = self.client.get("/api/export/excel")
        self.assertEqual(rx.status_code, 200)
        self.assertGreater(len(rx.content), 2000)

        rc = self.client.get("/api/export/csv", params={"freq": "quarter"})
        self.assertEqual(rc.status_code, 200)
        self.assertIn("impacto_acumulado", rc.text)

        rm = self.client.get("/api/export/report")
        self.assertEqual(rm.status_code, 200)
        self.assertIn("Relatório de Inferência Causal", rm.text)

        rt = self.client.get("/api/export/template")
        self.assertEqual(rt.status_code, 200)
        self.assertGreater(len(rt.content), 2000)

    def test_07_csv_upload_brazilian_decimal(self) -> None:
        csv_text = "data;vendas;controle\n" + "\n".join(
            f"2023-01-{i+1:02d};{100 + i*1.5 + (15 if i>=20 else 0):.2f};{50 + i*1.2:.2f}".replace(".", ",")
            for i in range(30)
        )
        r_up = self.client.post(
            "/api/dataset/upload",
            files={"file": ("teste_br.csv", csv_text.encode("utf-8"), "text/csv")},
        )
        self.assertEqual(r_up.status_code, 200)
        meta = r_up.json()
        self.assertEqual(meta["rows"], 30)
        self.assertIn("vendas", meta["numeric_columns"])


if __name__ == "__main__":
    unittest.main()
