"""Automated unit and integration test suite for App Forecasts (AutoML / ARIMA / AutoTS)."""
import io
import unittest

import joblib
import numpy as np
import openpyxl
import pandas as pd
from fastapi.testclient import TestClient

from backend.forecast_arima import run_arima_models_extended
from backend.main import STATE, app


def _read_excel_sheets(xlsx_bytes: bytes) -> list[str]:
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True)
    names = list(wb.sheetnames)
    wb.close()
    return names


class TestAppForecastsEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        res = self.client.post("/api/reset")
        self.assertEqual(res.status_code, 200)

    def _load_default_sample(self):
        res = self.client.post(
            "/api/import/sample",
            data={"filename": "dados_virgula.csv", "tipo_arquivo": "csv"},
        )
        self.assertEqual(res.status_code, 200)
        return res.json()

    def test_01_health_spa_samples_and_multiformat_import(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        spa = self.client.get("/")
        self.assertEqual(spa.status_code, 200)
        self.assertIn("App Forecasts", spa.text)

        samples = self.client.get("/api/samples").json()["samples"]
        names = {s["filename"] for s in samples}
        for expected in ("dados_virgula.csv", "dados_ponto_virgula.csv", "dados.xlsx", "dados.dta", "dados.rds", "DB.sqlite"):
            self.assertIn(expected, names)
            insp = self.client.get("/api/import/inspect-sample", params={"filename": expected})
            self.assertEqual(insp.status_code, 200)
            imp = self.client.post("/api/import/sample", data={"filename": expected, "tipo_arquivo": "auto"})
            self.assertEqual(imp.status_code, 200, f"Import failed for {expected}: {imp.text}")
            st = imp.json()["state"]
            self.assertEqual(st["total_rows"], 360)
            self.assertTrue(st["date_processing_complete"], f"Auto date_ok failed for {expected}")
            self.assertIn("date_ok", st["columns"])
            self.assertEqual(st["column_date_bounds"]["ei_total"]["max"], "2024-07-01")
            self.assertEqual(st["date_max"], "2034-12-01")

    def test_02_brazilian_csv_upload_and_8_step_prep(self):
        csv_text = (
            "Data_Ref;Produção;Tarifa\n"
            "01/01/2023;1.250,50;0,75\n"
            "01/02/2023;1.310,25;0,78\n"
            "01/03/2023;1.420,00;0,80\n"
            "01/04/2023;1.380,80;0,81\n"
        )
        up = self.client.post(
            "/api/import/upload",
            files={"file": ("br_latin1.csv", csv_text.encode("latin-1"), "text/csv")},
            data={"tipo_arquivo": "auto"},
        )
        self.assertEqual(up.status_code, 200)
        st = up.json()["state"]
        self.assertIn("Produção", st["numeric_columns"])
        self.assertIn("Tarifa", st["numeric_columns"])

        conv = self.client.post("/api/date/convert", json={"coluna": "Data_Ref", "formato": "dmy"})
        self.assertEqual(conv.status_code, 200)
        self.assertEqual(conv.json()["state"]["date_min"], "2023-01-01")

        # Now load full sample and test Steps 3, 4, 5, 6, 7
        self._load_default_sample()

        dum = self.client.post(
            "/api/dummies/generate",
            json={
                "anos": True,
                "meses": True,
                "tendencia_linear": True,
                "funcoes_trig": True,
                "estacoes": True,
                "bro_bro": True,
                "crises_selecionadas": ["pandemia_covid", "crise_2008"],
                "ativar_dummy_personalizada": True,
                "dummy_personalizada_nome": "evento_teste",
                "dummy_personalizada_inicio": "2022-01-01",
                "dummy_personalizada_fim": "2022-12-31",
            },
        )
        self.assertEqual(dum.status_code, 200)
        added = set(dum.json()["added_columns"])
        for col in ("trend", "sin1", "verao", "bro_bro", "dummy_pandemia_covid", "dummy_evento_teste"):
            self.assertIn(col, added)

        for transf, param in [("log", None), ("boxcox", None), ("boxcox", 0.5), ("scale", None), ("center", None)]:
            tr = self.client.post(
                "/api/transform/apply",
                json={"variaveis": ["ei_total"], "transformacao": transf, "parametro_boxcox": param},
            )
            self.assertEqual(tr.status_code, 200, f"Transform {transf} failed")

        seas = self.client.post(
            "/api/seasonality/extract",
            json={"variaveis": ["ei_total"], "data_inicio": "2010-01-01", "data_fim": "2023-12-01"},
        )
        self.assertEqual(seas.status_code, 200)
        self.assertIn("ei_total_sazonalidade", seas.json()["added_columns"])
        self.assertIn("ei_total_tendencia", seas.json()["added_columns"])

        tbl = self.client.get("/api/data/table", params={"limit": 50})
        self.assertEqual(tbl.status_code, 200)
        self.assertEqual(len(tbl.json()["rows"]), 50)

        for fmt in ("xlsx", "csv", "pkl"):
            exp = self.client.get("/api/data/export", params={"fmt": fmt})
            self.assertEqual(exp.status_code, 200)
            self.assertGreater(len(exp.content), 100)

    def test_03_caret_ml_recipes_tuned_gp_backtest_waterfall_elasticities(self):
        self._load_default_sample()

        catalog = self.client.get("/api/caret/models").json()
        self.assertGreaterEqual(len(catalog["models"]), 160)

        # Test custom models txt upload + reset
        custom_txt = "lm\nridge_scale\nrf\n"
        up_m = self.client.post(
            "/api/caret/custom-models",
            files={"file": ("modelos.txt", custom_txt.encode("utf-8"), "text/plain")},
        )
        self.assertEqual(up_m.status_code, 200)
        self.assertEqual(len(up_m.json()["models"]), 3)
        rst_m = self.client.post("/api/caret/reset-models")
        self.assertEqual(rst_m.status_code, 200)
        self.assertGreaterEqual(len(rst_m.json()["models"]), 160)

        # Train diverse models including:
        # - gaussprPoly_2 (tests valid GP kernel tuneGrid fix)
        # - ridge_scale + lasso_boxcox WITH overlapping recipe transforms ["scale", "BoxCox"] (tests case-insensitive dedup)
        # - rf + xgboost + lm
        train_res = self.client.post(
            "/api/caret/train",
            json={
                "date_col": "date_ok",
                "var_Y": "ei_total",
                "explicativas": ["PIB_4i", "Tarifa_res", "Temperatura_media", "dias_uteis", "massa_total"],
                "modelos": ["lm", "ridge_scale", "lasso_boxcox", "rf", "xgboost", "gaussprPoly_2"],
                "data_inicio": "2012-01-01",
                "data_teste": "2023-12-01",
                "n_test": 6,
                "metodo_impute": "mediana",
                "transformacoes_numericas": ["scale", "BoxCox"],
                "engenharia_features": ["date_features", "lag", "window"],
                "opcoes_features": {"lag_periods": [1, 2], "window_size": 3},
                "validation_method": "timeslice",
                "periodo_analise": 12,
            },
        )
        self.assertEqual(train_res.status_code, 200, train_res.text)
        body = train_res.json()
        self.assertEqual(
            set(body["modelos_treinados"]),
            {"lm", "ridge_scale", "lasso_boxcox", "rf", "xgboost", "gaussprPoly_2"},
        )
        self.assertEqual(len(body["recomendacao_backtest"]), 6)
        for row in body["recomendacao_backtest"]:
            self.assertIsNotNone(row["RMSE"])
            self.assertIsNotNone(row["score"])
            self.assertNotEqual(row["status"], "SEM DADOS COMUNS")

        # Waterfall importance sums to ~100% and elasticities populated
        imp_lm = body["importancia_variaveis"]["lm"]["dados"]
        self.assertAlmostEqual(sum(d["Importancia_Media"] for d in imp_lm), 100.0, delta=0.5)
        self.assertGreater(len(body["elasticidades"]["lm"]), 0)

        # Recalculate backtest window
        bt = self.client.post("/api/caret/backtest", json={"periodo_analise": 6})
        self.assertEqual(bt.status_code, 200)
        self.assertEqual(bt.json()["recomendacao_backtest"][0]["meses"], 6)

        # Exports (Excel 5 sheets, Importance CSV, Joblib models)
        exp_xlsx = self.client.get("/api/caret/export-projections")
        self.assertEqual(exp_xlsx.status_code, 200)
        self.assertEqual(
            _read_excel_sheets(exp_xlsx.content),
            ["Projecoes", "Backtest_Aderencia", "Metricas_Internas", "Configuracoes", "Dados_Treino_Usados"],
        )

        exp_imp = self.client.get("/api/caret/export-importance", params={"modelo": "lm"})
        self.assertEqual(exp_imp.status_code, 200)
        self.assertIn("Variavel", exp_imp.text)

        exp_jl = self.client.get("/api/caret/export-models")
        self.assertEqual(exp_jl.status_code, 200)
        loaded_models = joblib.load(io.BytesIO(exp_jl.content))
        self.assertIn("gaussprPoly_2", loaded_models)

    def test_04_arima_future_exog_clamping_collinearity_and_export(self):
        self._load_default_sample()

        # Deliberately pass data_atual="2034-12-01" to verify clamping to last realized target date (2024-07-01)
        ar = self.client.post(
            "/api/arima/run",
            json={
                "date_col": "date_ok",
                "target_var": "ei_total",
                "exog_vars": ["PIB_4i", "Tarifa_res"],
                "data_inicial": "2016-01-01",
                "data_atual": "2034-12-01",
                "end_predict": "2026-07-01",
                "n_teste": 6,
                "seasonal": True,
                "period": 12,
                "num_models": 2,
                "manual_mode": False,
            },
        )
        self.assertEqual(ar.status_code, 200, ar.text)
        body = ar.json()
        self.assertFalse(body["collinear_warning"])
        self.assertEqual(body["periods"]["test_end"], "2024-07-01")
        self.assertGreater(len(body["deviations"]), 0)
        types = {e["type"] for e in body["errors"]}
        self.assertIn("Univariado", types)
        self.assertIn("Multivariado (ARIMAX)", types)
        self.assertNotEqual(body["dashboard"][0]["status"], "SEM PERÍODO TESTE")

        # Verify future exogenous rows (where Y is NaN) are genuinely used instead of constant training mean
        dates = pd.date_range("2015-01-01", periods=72, freq="MS")
        x_future_up = np.concatenate([np.linspace(10.0, 20.0, 60), np.linspace(25.0, 80.0, 12)])
        x_future_flat = np.concatenate([np.linspace(10.0, 20.0, 60), np.full(12, 15.0)])
        y_vals = np.concatenate([100.0 + 10.0 * x_future_up[:60] + np.sin(np.arange(60) * 0.5), np.full(12, np.nan)])
        df_up = pd.DataFrame({"Data": dates, "y": y_vals, "x": x_future_up})
        df_flat = pd.DataFrame({"Data": dates, "y": y_vals, "x": x_future_flat})
        r_up = run_arima_models_extended(df_up, "Data", "y", ["x"], "2015-01-01", "2019-12-01", "2020-12-01", n_teste=0, seasonal=False, manual_mode=True, arima_params={"p": 1, "d": 0, "q": 0})
        r_fl = run_arima_models_extended(df_flat, "Data", "y", ["x"], "2015-01-01", "2019-12-01", "2020-12-01", n_teste=0, seasonal=False, manual_mode=True, arima_params={"p": 1, "d": 0, "q": 0})
        fc_up = next(f for f in r_up["forecasts"] if f["model"] == "arima_multi_1")["y"][-1]
        fc_fl = next(f for f in r_fl["forecasts"] if f["model"] == "arima_multi_1")["y"][-1]
        self.assertGreater(fc_up - fc_fl, 200.0)

        # Verify collinearity skips unstable multivariate models
        df_col = df_up.copy()
        df_col["x_dup"] = df_col["x"] * 2.0 + 1.0
        r_col = run_arima_models_extended(df_col, "Data", "y", ["x", "x_dup"], "2015-01-01", "2019-12-01", "2020-12-01", n_teste=3, seasonal=False, manual_mode=True, arima_params={"p": 1, "d": 0, "q": 0})
        self.assertTrue(r_col["collinear_warning"])
        self.assertEqual({e["type"] for e in r_col["errors"]}, {"Univariado"})

        # Dashboard update & Excel export
        dash_up = self.client.post("/api/arima/dashboard", json={"peso_recente": 3.0, "peso_desvio_max": 0.5, "periodo_maximo": 6})
        self.assertEqual(dash_up.status_code, 200)
        exp_ar = self.client.get("/api/arima/export", params={"model": "Todos"})
        self.assertEqual(exp_ar.status_code, 200)
        self.assertEqual(
            _read_excel_sheets(exp_ar.content),
            ["Projecoes_ARIMA", "Metricas_Erro", "Recomendacao", "Desvios_Teste"],
        )

    def test_05_autots_all_7_models_partial_year_growth_and_reset(self):
        self._load_default_sample()

        # Forecast ends in June 2026 (partial final year!) to verify Jul..Dez are None (not 0.0 / -100%)
        at = self.client.post(
            "/api/autots/run",
            json={
                "date_col": "date_ok",
                "variable": "ei_total",
                "start_date": "2015-01-01",
                "end_date": "2024-07-01",
                "end_predict": "2026-06-01",
                "selected_models": ["prophet", "ets", "sarima", "tbats", "bats", "stlm", "bagged"],
            },
        )
        self.assertEqual(at.status_code, 200, at.text)
        body = at.json()
        self.assertEqual(
            set(body["models"]),
            {"prophet", "ets", "sarima", "tbats", "bats", "stlm", "bagged"},
        )
        # Exactly 23 forecast months from 2024-08 to 2026-06
        self.assertEqual(len(body["chart_models"]["bagged"]["x_fut"]), 23)
        self.assertEqual(body["chart_models"]["bagged"]["x_fut"][-1], "2026-06-01")

        # Partial final year 2026 must have None (not 0.0 / -100%) for Jul..Dez
        row_2026_val = next(r for r in body["growth"]["mensal"]["valores"] if r["Ano"] == 2026)
        row_2026_tx = next(r for r in body["growth"]["mensal"]["taxas"] if r["Ano"] == 2026)
        self.assertIsNotNone(row_2026_val["Jun"])
        self.assertIsNone(row_2026_val["Jul"])
        self.assertIsNone(row_2026_val["Dez"])
        self.assertIsNone(row_2026_tx["Jul"])

        # Update desvios & regenerate growth for bagged
        d_res = self.client.post("/api/autots/desvios", json={"periodo_maximo": 12, "peso_recente": 2.5, "peso_desvio_max": 0.4})
        self.assertEqual(d_res.status_code, 200)

        g_res = self.client.post("/api/autots/growth", json={"model_name": "bagged", "variable": "ei_total"})
        self.assertEqual(g_res.status_code, 200)
        for gran in ("mensal", "trimestral", "semestral", "anual"):
            self.assertGreater(len(g_res.json()[gran]["valores"]), 0)

        # Exports
        exp_all = self.client.get("/api/autots/export")
        self.assertEqual(exp_all.status_code, 200)
        self.assertEqual(_read_excel_sheets(exp_all.content), ["Projecoes_AutoTS", "Erros_Treino", "Analise_Desvios"])

        exp_one = self.client.get("/api/autots/export", params={"model": "bagged"})
        self.assertEqual(exp_one.status_code, 200)
        self.assertEqual(_read_excel_sheets(exp_one.content), ["AutoTS_bagged"])

        exp_gr = self.client.get("/api/autots/export-growth")
        self.assertEqual(exp_gr.status_code, 200)
        self.assertEqual(len(_read_excel_sheets(exp_gr.content)), 16)

        # Reset must clear ALL cached model results
        rst = self.client.post("/api/reset")
        self.assertEqual(rst.status_code, 200)
        self.assertIsNone(STATE.caret_results)
        self.assertIsNone(STATE.arima_results)
        self.assertIsNone(STATE.autots_results)
        self.assertIsNone(STATE.autots_growth_results)


if __name__ == "__main__":
    unittest.main()
