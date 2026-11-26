"""Testes unitários de integridade econômica, estatística e tarifária — Comercio_exterior_spark_análise."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from econometrics_tariffs import (
    estimate_orange_juice_elasticities,
    landed_price_usd_kg,
    simulate_tariff_scenarios,
)
from spark_comex_pipeline import compute_hhi


class TestComexPipeline(unittest.TestCase):
    def test_01_landed_price_math(self) -> None:
        # FOB 1.50 + 10% ad valorem (0.15) + US$415/t (0.415/kg) = 2.065 US$/kg
        p = landed_price_usd_kg(1.50, ad_valorem=0.10, specific_usd_per_ton=415.0)
        self.assertAlmostEqual(p, 2.065, places=6)

    def test_02_hhi_bounds(self) -> None:
        # Monopólio = 10.000; 4 parceiros iguais = 2.500
        self.assertAlmostEqual(compute_hhi(pd.Series([100.0])), 10000.0, places=4)
        self.assertAlmostEqual(compute_hhi(pd.Series([25.0, 25.0, 25.0, 25.0])), 2500.0, places=4)

    def test_03_elasticity_and_2sls_recovery(self) -> None:
        rng = np.random.default_rng(123)
        n = 120
        dates = pd.date_range("2015-01-01", periods=n, freq="MS")
        z_supply = rng.normal(0.0, 0.25, size=n)
        u_demand = rng.normal(0.0, 0.15, size=n)
        # Preço correlacionado positivamente com choque de oferta (IV) e levemente com demanda
        ln_p_row = 0.8 + z_supply + rng.normal(0, 0.05, size=n)
        ln_p_eua = 0.8 + 0.9 * z_supply + 0.3 * u_demand + rng.normal(0, 0.05, size=n)
        # Elasticidade estrutural verdadeira = -1.20
        ln_q = 15.0 - 1.20 * ln_p_eua + u_demand
        df = pd.DataFrame({
            "CO_ANO": dates.year,
            "CO_MES": dates.month,
            "preco_fob_eua": np.exp(ln_p_eua),
            "preco_fob_row": np.exp(ln_p_row),
            "kg_eua": np.exp(ln_q),
            "vl_fob_eua": np.exp(ln_q + ln_p_eua),
        })
        res = estimate_orange_juice_elasticities(df)
        self.assertEqual(len(res["specs"]), 4)
        m4 = next(s for s in res["specs"] if s["id"] == "M4")
        self.assertGreater(m4["first_stage_f"], 50.0)
        self.assertAlmostEqual(m4["elasticity"], -1.20, delta=0.25)

    def test_04_monte_carlo_tariff_monotonicity(self) -> None:
        df = pd.DataFrame({
            "CO_ANO": [2024] * 6 + [2025] * 6,
            "CO_MES": [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6],
            "kg_eua": [20_000_000.0] * 12,
            "vl_fob_eua": [60_000_000.0] * 12,
            "preco_fob_eua": [3.0] * 12,
            "preco_fob_row": [2.9] * 12,
        })
        mc = simulate_tariff_scenarios(df, beta_hat=-1.1, beta_se=0.15, n_sims=5000, seed=42)
        sc = mc["scenarios"]
        # Quanto maior a tarifa, maior a queda mediana no volume (mais negativa)
        self.assertGreater(sc[0]["dq_median_pct"], sc[1]["dq_median_pct"])
        self.assertGreater(sc[1]["dq_median_pct"], sc[2]["dq_median_pct"])
        # P10 <= Mediana <= P90 em todos os cenários
        for row in sc:
            self.assertLessEqual(row["dq_p10_pct"], row["dq_median_pct"])
            self.assertLessEqual(row["dq_median_pct"], row["dq_p90_pct"])


if __name__ == "__main__":
    unittest.main()
