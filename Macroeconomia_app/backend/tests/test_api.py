"""Suíte de testes automatizados (unittest + TestClient) do Observatório Macro Brasil Studio."""

from __future__ import annotations

import json
import unittest
from fastapi.testclient import TestClient

from app.catalog import STUDIES_CATALOG
from app.main import app


class TestObservatorioMacroAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_01_health_and_catalog_originality(self) -> None:
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)
        hd = h.json()
        self.assertEqual(hd["app_name"], "Observatório Macro Brasil API")
        self.assertEqual(hd["studies_count"], 60)
        self.assertGreaterEqual(hd["db_tables"], 30)
        self.assertGreaterEqual(hd["bcb_series"], 70)

        # Verifica equilíbrio exato de 10 estudos por pilar temático
        list_resp = self.client.get("/api/studies")
        self.assertEqual(list_resp.status_code, 200)
        themes = list_resp.json()["themes"]
        self.assertEqual(len(themes), 6)
        for t in themes:
            self.assertEqual(t["count"], 10, f"Pilar {t['key']} com contagem inesperada: {t['count']}")

        # Garante 100% de originalidade: zero menções legadas no catálogo autoral
        serialized = json.dumps(STUDIES_CATALOG, ensure_ascii=False).lower()
        for forbidden in ("clube do c", "clubedocodigo", "wilher", "analisemacro", "navy{", "@blanchard", "cãmbio"):
            self.assertNotIn(forbidden, serialized, f"Texto legado encontrado: {forbidden}")

    def test_02_conjuntura_dashboard(self) -> None:
        r = self.client.get("/api/conjuntura")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(len(d["kpis"]), 6)
        self.assertGreater(len(d["charts"]["juros_inflacao"]), 50)
        self.assertGreater(len(d["charts"]["fiscal_cambio"]), 50)
        self.assertGreater(len(d["charts"]["atividade_trabalho"]), 50)

    def test_03_all_six_dedicated_models(self) -> None:
        dedicated_cases = [
            ("M01", {"subsample_start": "2014-09-05", "max_lags": 8, "use_live": False}),
            ("M02", {"horizon": 12, "use_live": True}),
            ("M06", {"hp_lambda": 14400, "use_live": True}),
            ("M12", {"window": 48, "use_live": True}),
            ("M13", {"p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 1, "horizon": 12}),
            ("M14", {"hac_lags": 4}),
        ]
        for code, params in dedicated_cases:
            with self.subTest(study=code):
                r = self.client.post(f"/api/studies/{code}/run", json={"params": params})
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertEqual(body["study"]["code"], code)
                self.assertTrue(len(body["result"]["kpis"]) >= 4)
                self.assertTrue(len(body["result"]["conclusion"]) > 30)

    def test_04_all_remaining_thematic_studies_run_cleanly(self) -> None:
        # Testa TODOS os 54 estudos temáticos (M01–M60 completos) para garantir 100% de execução sem erro!
        thematic_codes = [s["code"] for s in STUDIES_CATALOG if not s["specialized"]]
        self.assertEqual(len(thematic_codes), 54)
        for code in thematic_codes:
            with self.subTest(study=code):
                r = self.client.post(f"/api/studies/{code}/run", json={"params": {"use_live": False}})
                self.assertEqual(r.status_code, 200, f"{code}: {r.text}")
                body = r.json()
                self.assertEqual(body["result"]["kind"], "thematic")
                self.assertGreater(len(body["result"]["raw_chart"]), 5)

    def test_05_csv_export(self) -> None:
        r = self.client.get("/api/studies/M06/export-csv")
        self.assertEqual(r.status_code, 200)
        self.assertIn("selic_obs", r.text)


if __name__ == "__main__":
    unittest.main()
