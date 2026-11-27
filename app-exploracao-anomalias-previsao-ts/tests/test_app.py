"""Automated end-to-end and unit tests for TimeSeries Studio Pro."""
import io
import unittest

import numpy as np
import openpyxl
import pandas as pd
from fastapi.testclient import TestClient

from backend.analytics import correct_outliers, identify_outliers
from backend.forecasting import check_collinearity
from backend.main import STATE, app


def _read_excel_sheets(xlsx_bytes: bytes) -> list[str]:
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True)
    names = list(wb.sheetnames)
    wb.close()
    return names


class TestAnalyticsEdgeCases(unittest.TestCase):
    def test_mahalanobis_multivariate_no_false_univariate_fallback(self):
        rng = np.random.default_rng(7)
        n = 80
        # Smooth bounded data with no multivariate outliers
        u = np.linspace(-1.0, 1.0, n)
        df = pd.DataFrame({
            "x1": u + rng.uniform(-0.05, 0.05, n),
            "x2": 2.0 * u + rng.uniform(-0.05, 0.05, n),
            "x3": -0.5 * u + rng.uniform(-0.05, 0.05, n),
        })
        # Make x1 have a value > 1.96 SD univariately (which univariate chi2(1, 0.95)=3.84 flags)
        # while multivariate chi2(3, 0.95)=7.81 does not flag it
        df.loc[n - 1, "x1"] = 1.35
        df.loc[n - 1, "x2"] = 2.70
        df.loc[n - 1, "x3"] = -0.675

        # Verify multivariate runs cleanly and does not trigger univariate fallback
        mask = identify_outliers(df, "x1", method="mahalanobis")
        self.assertEqual(len(mask), n)
        self.assertFalse(bool(mask[n - 1]))

    def test_arima_outlier_correction_guards_index_zero_burn_in(self):
        # High-scale series where index 0 is an outlier; ARIMA(1,1,1) fittedvalues[0] would be 0.0 without guard
        arr = np.linspace(500_000.0, 560_000.0, 48)
        arr[0] = 950_000.0
        arr[20] = 100_000.0
        mask = np.zeros(len(arr), dtype=bool)
        mask[0] = True
        mask[20] = True

        corrected = correct_outliers(arr, mask, method="arima")
        self.assertGreater(corrected[0], 400_000.0)
        self.assertGreater(corrected[20], 400_000.0)
        self.assertLess(corrected[20], 650_000.0)

    def test_collinearity_detector(self):
        x = np.linspace(1.0, 100.0, 40)
        df = pd.DataFrame({
            "a": x,
            "b": x * 2.5 + 3.0,
            "c": np.sin(x),
        })
        has_coll, pairs = check_collinearity(df, ["a", "b", "c"], threshold=0.99)
        self.assertTrue(has_coll)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(pairs[0][:2]), {"a", "b"})


class TestTimeSeriesStudioAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        # Always reset to the canonical sample dataset before each test
        resp = self.client.post("/api/dataset/load-sample")
        self.assertEqual(resp.status_code, 200)

    def test_01_health_spa_and_sample_dataset(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        h_json = health.json()
        self.assertEqual(h_json["status"], "ok")
        self.assertTrue(h_json["dataset_loaded"])
        self.assertEqual(h_json["rows"], 360)

        spa = self.client.get("/")
        self.assertEqual(spa.status_code, 200)
        self.assertIn("TimeSeries Studio", spa.text)

        info = self.client.get("/api/dataset/info")
        self.assertEqual(info.status_code, 200)
        meta = info.json()
        self.assertTrue(meta["loaded"])
        self.assertIn("ei_total", meta["numeric_columns"])
        self.assertEqual(meta["total_rows"], 360)

        exp = self.client.get("/api/dataset/export-sample")
        self.assertEqual(exp.status_code, 200)
        self.assertIn("Dados", _read_excel_sheets(exp.content))

    def test_02_brazilian_csv_upload_and_configure(self):
        csv_text = (
            "Data_Ref;Produção;Tarifa\n"
            "01/01/2023;1.250,50;0,75\n"
            "01/02/2023;1.310,25;0,78\n"
            "01/03/2023;1.420,00;0,80\n"
            "01/04/2023;1.380,80;0,81\n"
        )
        raw_bytes = csv_text.encode("latin-1")
        up = self.client.post(
            "/api/dataset/upload",
            files={"file": ("producao_br.csv", raw_bytes, "text/csv")},
        )
        self.assertEqual(up.status_code, 200)
        meta = up.json()
        self.assertEqual(meta["filename"], "producao_br.csv")
        self.assertIn("Produção", meta["numeric_columns"])
        self.assertIn("Tarifa", meta["numeric_columns"])

        cfg = self.client.post(
            "/api/dataset/configure",
            json={
                "date_column": "Data_Ref",
                "date_format": "dmy",
                "csv_separator": ";",
            },
        )
        self.assertEqual(cfg.status_code, 200)
        cfg_meta = cfg.json()
        self.assertEqual(cfg_meta["min_date"], "2023-01-01")
        self.assertEqual(cfg_meta["max_date"], "2023-04-01")
        prod_summary = next(c for c in cfg_meta["column_summaries"] if c["column"] == "Produção")
        self.assertAlmostEqual(prod_summary["min"], 1250.5, places=2)
        self.assertAlmostEqual(prod_summary["max"], 1420.0, places=2)
        self.assertIsNone(STATE.last_arima_result)
        self.assertIsNone(STATE.last_autots_result)

    def test_03_outliers_all_6_detectors_7_correctors_undo_commit_export(self):
        detectors = [
            ("zscore", 2.0),
            ("iqr", 1.5),
            ("mahalanobis", None),
            ("isolation_forest", 0.55),
            ("lof", 1.3),
            ("svm", 0.08),
        ]
        for method, thresh in detectors:
            res = self.client.post(
                "/api/outliers/identify",
                json={"column": "ei_total", "method": method, "threshold": thresh},
            )
            self.assertEqual(res.status_code, 200, f"Detector failed: {method}")
            body = res.json()
            self.assertEqual(body["ident_method"], method)
            valid_n = body["kpis"]["valid_count"]
            expected_pct = round(100.0 * body["kpis"]["outliers_count"] / valid_n, 1)
            self.assertEqual(body["kpis"]["percent_outliers"], expected_pct)

        # Ensure at least one outlier is flagged for testing all 7 correction methods
        res_id = self.client.post(
            "/api/outliers/identify",
            json={"column": "ei_total", "method": "zscore", "threshold": 1.8},
        )
        self.assertEqual(res_id.status_code, 200)
        self.assertGreater(res_id.json()["kpis"]["outliers_count"], 0)

        correctors = [
            "mean",
            "median",
            "moving_average",
            "linear_interp",
            "winsorize",
            "arima",
            "seasonal_decomp",
        ]
        for corr_method in correctors:
            res_c = self.client.post(
                "/api/outliers/correct",
                json={"column": "ei_total", "method": corr_method, "window_size": 5},
            )
            self.assertEqual(res_c.status_code, 200, f"Corrector failed: {corr_method}")
            body_c = res_c.json()
            self.assertTrue(body_c["has_correction"])
            self.assertEqual(body_c["correction_method"], corr_method)

        # Undo once and check history depth decreased
        depth_before = len(STATE.outlier_history)
        res_u = self.client.post("/api/outliers/undo", params={"column": "ei_total"})
        self.assertEqual(res_u.status_code, 200)
        self.assertEqual(res_u.json()["history_depth"], depth_before - 1)

        # Commit treated series to workspace
        res_com = self.client.post("/api/outliers/commit", params={"column": "ei_total"})
        self.assertEqual(res_com.status_code, 200)
        self.assertEqual(res_com.json()["new_column"], "ei_total_tratado")
        self.assertIn("ei_total_tratado", res_com.json()["dataset"]["numeric_columns"])

        # Export Excel
        exp = self.client.get("/api/outliers/export", params={"column": "ei_total"})
        self.assertEqual(exp.status_code, 200)
        self.assertEqual(_read_excel_sheets(exp.content), ["Dados_Tratados", "Metadados"])

    def test_04_seasonality_xmr_stl_and_export(self):
        res = self.client.post(
            "/api/seasonality/analyze",
            json={
                "column": "ei_total",
                "start_year": 2015,
                "end_year": 2023,
                "stl_threshold_sigma": 1.8,
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["limits_table"]), 12)
        self.assertGreater(body["xmr"]["ucl_x"], body["xmr"]["lcl_x"])
        self.assertIsNotNone(body["stl"]["seasonality_strength"])
        self.assertGreaterEqual(body["stl"]["seasonality_strength"], 0.0)
        self.assertLessEqual(body["stl"]["seasonality_strength"], 1.0)
        self.assertEqual(body["stl"]["threshold_sigma"], 1.8)
        self.assertIsNotNone(body["seasonality_tests"]["anova_p"])
        self.assertIsNotNone(body["seasonality_tests"]["kruskal_p"])

        exp = self.client.get(
            "/api/seasonality/export",
            params={
                "column": "ei_total",
                "start_year": 2015,
                "end_year": 2023,
                "stl_threshold_sigma": 1.8,
            },
        )
        self.assertEqual(exp.status_code, 200)
        self.assertEqual(
            _read_excel_sheets(exp.content),
            ["Participacao_Limites", "Controle_XmR", "Anomalias_STL", "Decomposicao_STL"],
        )

    def test_05_editor_modify_undo_commit_reset_export(self):
        init_res = self.client.post("/api/editor/init", json={"column": "ei_total"})
        self.assertEqual(init_res.status_code, 200)
        init_body = init_res.json()
        self.assertGreater(init_body["kpis"]["num_periodos"], 100)
        orig_v0 = init_body["series"][0]["Valor_Original"]

        # Apply set, absolute, and percentual edits
        r1 = self.client.post(
            "/api/editor/modify",
            json={"column": "ei_total", "indices": [0], "change_type": "set", "change_value": 555_000.0},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertAlmostEqual(r1.json()["series"][0]["Serie_Ajustada"], 555_000.0, places=2)

        r2 = self.client.post(
            "/api/editor/modify",
            json={"column": "ei_total", "indices": [1, 2], "change_type": "absolute", "change_value": 10_000.0},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["kpis"]["modified_points_count"], 3)

        r3 = self.client.post(
            "/api/editor/modify",
            json={"column": "ei_total", "indices": [3], "change_type": "percentual", "change_value": 10.0},
        )
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json()["kpis"]["modified_points_count"], 4)
        self.assertGreaterEqual(len(r3.json()["modification_log"]), 4)

        # Undo last batch (index 3)
        ru = self.client.post("/api/editor/undo", params={"column": "ei_total"})
        self.assertEqual(ru.status_code, 200)
        self.assertEqual(ru.json()["kpis"]["modified_points_count"], 3)

        # Commit edited series & export
        rc = self.client.post("/api/editor/commit", params={"column": "ei_total"})
        self.assertEqual(rc.status_code, 200)
        self.assertEqual(rc.json()["new_column"], "ei_total_editado")

        exp = self.client.get("/api/editor/export", params={"column": "ei_total"})
        self.assertEqual(exp.status_code, 200)
        self.assertIn("Series_Comparadas", _read_excel_sheets(exp.content))
        self.assertIn("Log_Modificacoes", _read_excel_sheets(exp.content))

        # Reset back to original
        rr = self.client.post("/api/editor/reset", params={"column": "ei_total"})
        self.assertEqual(rr.status_code, 200)
        self.assertEqual(rr.json()["kpis"]["modified_points_count"], 0)
        self.assertAlmostEqual(rr.json()["series"][0]["Serie_Ajustada"], orig_v0, places=2)

    def test_06_arima_uni_multi_manual_autots_and_growth(self):
        # 1. ARIMA Auto (Univariate + Multivariate)
        arima_res = self.client.post(
            "/api/forecast/arima",
            json={
                "target_var": "ei_total",
                "exog_vars": ["PIB_4i", "Tarifa_res"],
                "data_inicial": "2016-01-01",
                "data_atual": "2023-12-01",
                "end_predict": "2024-12-01",
                "n_teste": 6,
                "seasonal": True,
                "period": 12,
                "num_models": 2,
                "manual_mode": False,
            },
        )
        self.assertEqual(arima_res.status_code, 200)
        ar_body = arima_res.json()
        self.assertFalse(ar_body["has_collinearity"])
        types = {e["type"] for e in ar_body["errors"]}
        self.assertIn("Univariado", types)
        self.assertIn("Multivariado", types)
        self.assertGreater(len(ar_body["recommendation"]["ranking"]), 0)

        exp_ar = self.client.get("/api/forecast/export-arima", params={"model": "Todos"})
        self.assertEqual(exp_ar.status_code, 200)
        self.assertEqual(
            _read_excel_sheets(exp_ar.content),
            ["Projecoes_ARIMA", "Metricas_Erro", "Recomendacao"],
        )

        # 2. ARIMA Manual mode
        ar_man = self.client.post(
            "/api/forecast/arima",
            json={
                "target_var": "ei_total",
                "data_inicial": "2016-01-01",
                "data_atual": "2023-12-01",
                "end_predict": "2024-06-01",
                "n_teste": 3,
                "seasonal": True,
                "period": 12,
                "manual_mode": True,
                "arima_params": {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 0},
            },
        )
        self.assertEqual(ar_man.status_code, 200)
        self.assertEqual(len(ar_man.json()["errors"]), 1)

        # 3. AutoTS All 7 Models
        autots_res = self.client.post(
            "/api/forecast/autots",
            json={
                "variable": "ei_total",
                "start_date": "2016-01-01",
                "end_date": "2023-12-01",
                "end_predict": "2024-12-01",
                "selected_models": ["prophet", "ets", "sarima", "tbats", "bats", "stlm", "bagged"],
            },
        )
        self.assertEqual(autots_res.status_code, 200)
        at_body = autots_res.json()
        self.assertEqual(
            set(at_body["models_available"]),
            {"prophet", "ets", "sarima", "tbats", "bats", "stlm", "bagged"},
        )
        self.assertIn(at_body["best_model"], at_body["models_available"])
        proj_rows = [r for r in at_body["timeline"] if r["Period"] == "Projeção"]
        self.assertEqual(len(proj_rows), 12)
        for m in at_body["models_available"]:
            for r in proj_rows:
                self.assertIsNotNone(r[m])
                self.assertLessEqual(r[f"{m}_lower"], r[f"{m}_upper"])

        exp_at = self.client.get("/api/forecast/export-autots", params={"model": "bagged"})
        self.assertEqual(exp_at.status_code, 200)
        self.assertEqual(
            _read_excel_sheets(exp_at.content),
            ["Projecoes_AutoTS", "Metricas_Erro", "Ranking_Desvios"],
        )

        # 4. Multi-resolution Growth Analysis across all 3 sources
        for src, target in [
            ("dataset", "ei_total"),
            ("arima", ar_man.json()["errors"][0]["model"]),
            ("autots", "bagged"),
        ]:
            g_res = self.client.post(
                "/api/growth/analyze",
                json={"source": src, "model_or_col": target},
            )
            self.assertEqual(g_res.status_code, 200, f"Growth failed for {src}/{target}")
            views = g_res.json()["views"]
            for period_key in ("mensal", "trimestral", "semestral", "anual"):
                self.assertGreater(len(views[period_key]["wide_valores"]), 0)
                self.assertGreater(len(views[period_key]["long"]), 0)

        exp_g = self.client.get(
            "/api/growth/export",
            params={"source": "autots", "model_or_col": "bagged"},
        )
        self.assertEqual(exp_g.status_code, 200)
        self.assertEqual(len(_read_excel_sheets(exp_g.content)), 9)


if __name__ == "__main__":
    unittest.main()
