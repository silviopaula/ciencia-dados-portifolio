"""Testes unitários e de integração do motor econométrico e da API FastAPI."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.datasets import get_demo_catalog, load_demo_dataset
from app.diagnostics import phillips_perron_test, run_full_diagnostics
from app.main import app


class TestTimeSeriesDiagnosticsStudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_01_health_and_demo_catalog(self) -> None:
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

        demos = self.client.get("/api/demos")
        self.assertEqual(demos.status_code, 200)
        items = demos.json()
        self.assertEqual(len(items), 5)
        self.assertTrue(all(d["n_obs"] >= 160 for d in items))

    def test_02_all_5_regimes_econometric_behavior(self) -> None:
        # 1) Controle AR(1) Estacionário -> deve ser estacionário no nível
        df_ar, meta_ar = load_demo_dataset("ar1_estacionario_controle")
        res_ar = run_full_diagnostics(df_ar, meta_ar["title"], dataset_id=meta_ar["id"], transform="level", regression="c")
        self.assertEqual(res_ar["stationarity_consensus"]["quadrant"], "estacionaria")
        self.assertEqual(len(res_ar["tests_summary"]), 10)

        # 2) GARCH Retornos -> estacionária na média, mas detecta efeito ARCH e não-normalidade
        df_g, meta_g = load_demo_dataset("retornos_garch_arch")
        res_g = run_full_diagnostics(df_g, meta_g["title"], dataset_id=meta_g["id"], transform="level", regression="c")
        t_map = {t["id"]: t for t in res_g["tests_summary"]}
        self.assertTrue(t_map["adf"]["passed"])
        self.assertFalse(t_map["arch_lm"]["passed"], "Série GARCH deve disparar alerta ARCH-LM!")
        self.assertFalse(t_map["anderson_darling"]["passed"], "Série GARCH t(5) deve rejeitar normalidade!")

        # 3) Quebra estrutural -> Zivot-Andrews identifica a quebra próxima a 2020-04
        df_za, meta_za = load_demo_dataset("quebra_estrutural_za")
        res_za = run_full_diagnostics(df_za, meta_za["title"], dataset_id=meta_za["id"], transform="level", regression="ct")
        self.assertIsNotNone(res_za["zivot_andrews"]["break_date"])
        self.assertTrue(res_za["zivot_andrews"]["break_date"].startswith("2020"))

        # 4) Variância multiplicativa -> Breusch-Pagan temporal dispara no nível e melhora após log_diff
        df_v, meta_v = load_demo_dataset("consumo_variancia_crescente")
        res_v_lvl = run_full_diagnostics(df_v, meta_v["title"], dataset_id=meta_v["id"], transform="level", regression="ct")
        t_lvl = {t["id"]: t for t in res_v_lvl["tests_summary"]}
        self.assertFalse(t_lvl["bp_temporal"]["passed"])
        res_v_ld = run_full_diagnostics(df_v, meta_v["title"], dataset_id=meta_v["id"], transform="log_diff", regression="c")
        t_ld = {t["id"]: t for t in res_v_ld["tests_summary"]}
        self.assertTrue(t_ld["adf"]["passed"])

        # 5) Varejo I(1) + Sazonal -> genuinamente não-estacionária no nível + STL forte
        df_s, meta_s = load_demo_dataset("varejo_sazonal_i1")
        res_s = run_full_diagnostics(df_s, meta_s["title"], dataset_id=meta_s["id"], transform="level", regression="ct")
        self.assertEqual(res_s["stationarity_consensus"]["quadrant"], "nao_estacionaria")
        self.assertGreater(res_s["stl"]["trend_strength"], 0.75)
        self.assertGreater(res_s["stl"]["seasonal_strength"], 0.60)
        self.assertEqual(len(res_s["transform_comparison"]), 7)

    def test_03_phillips_perron_distinguishes_rw_vs_wn(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        wn = rng.normal(0.0, 1.0, size=250)
        rw = np.cumsum(wn)
        pp_wn = phillips_perron_test(wn, regression="c")
        pp_rw = phillips_perron_test(rw, regression="c")
        self.assertLess(pp_wn["p_value"], 0.01)
        self.assertGreater(pp_rw["p_value"], 0.10)

    def test_04_csv_upload_br_semicolon_and_analyze(self) -> None:
        lines = ["Data;Vendas_BRL"]
        for i in range(48):
            yr = 2021 + i // 12
            mo = (i % 12) + 1
            val = 100.0 + 1.5 * i + (12.0 if mo == 12 else 0.0)
            lines.append(f"{yr:04d}-{mo:02d}-01;{val:.2f}".replace(".", ","))
        csv_bytes = "\n".join(lines).encode("utf-8")

        r_ins = self.client.post(
            "/api/upload/inspect",
            files={"file": ("vendas_br.csv", csv_bytes, "text/csv")},
        )
        self.assertEqual(r_ins.status_code, 200)
        ins = r_ins.json()
        self.assertEqual(ins["detected_separator"], ";")
        self.assertEqual(ins["suggested_date_col"], "Data")
        self.assertEqual(ins["suggested_value_col"], "Vendas_BRL")

        r_an = self.client.post(
            "/api/upload/analyze",
            files={"file": ("vendas_br.csv", csv_bytes, "text/csv")},
            data={
                "date_col": "Data",
                "value_col": "Vendas_BRL",
                "transform": "diff1",
                "seasonal_period": "12",
                "forecast_horizon": "6",
            },
        )
        self.assertEqual(r_an.status_code, 200)
        payload = r_an.json()
        self.assertEqual(len(payload["arima"]["forecast"]), 6)
        self.assertEqual(len(payload["tests_summary"]), 10)

    def test_05_excel_multi_sheet_export(self) -> None:
        r = self.client.post(
            "/api/export/excel",
            json={"dataset_id": "varejo_sazonal_i1", "transform": "level", "forecast_horizon": 12},
        )
        self.assertEqual(r.status_code, 200)
        wb = load_workbook(io.BytesIO(r.content))
        self.assertEqual(
            wb.sheetnames,
            ["Resumo_Executivo", "Testes_Estatisticos", "Matriz_Transformacoes", "Serie_e_STL", "Previsao_SARIMA"],
        )
        self.assertGreaterEqual(wb["Testes_Estatisticos"].max_row, 11)
        self.assertEqual(wb["Previsao_SARIMA"].max_row, 13)


if __name__ == "__main__":
    unittest.main()
