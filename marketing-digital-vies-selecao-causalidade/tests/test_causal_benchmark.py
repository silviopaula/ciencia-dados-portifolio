"""Testes automatizados para o benchmark de inferência causal sob viés de seleção."""
from pathlib import Path
import unittest

import numpy as np

from src.causal_benchmark import (
    COVARIATES,
    compute_entropy_balancing_weights,
    load_datasets,
    run_full_benchmark,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestCausalMarketingBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = run_full_benchmark(DATA_DIR)

    def test_01_dataset_integrity_and_ground_truth(self):
        df_full, df_pos, df_store = load_datasets(DATA_DIR)
        self.assertEqual(len(df_full), 6000)
        self.assertEqual(len(df_pos), 3000)
        self.assertEqual(len(df_store), 250)
        self.assertEqual(self.bench["n_treated"], 155)
        self.assertEqual(self.bench["n_control"], 95)
        self.assertAlmostEqual(self.bench["true_att"], 6012.28, places=2)
        # Viés de seleção pré-campanha positivo e relevante
        self.assertGreater(self.bench["pre_selection_gap"], 1500.0)

    def test_02_entropy_balancing_exact_moment_matching(self):
        _, _, df_store = load_datasets(DATA_DIR)
        X = df_store[COVARIATES].to_numpy(dtype=float)
        W = df_store["tratamento"].to_numpy(dtype=int)
        w = compute_entropy_balancing_weights(X, W, include_second_moments=False)
        self.assertTrue(np.all(w > 0))
        self.assertAlmostEqual(float(np.sum(w[W == 0])), 1.0, places=8)
        self.assertAlmostEqual(float(np.sum(w[W == 1])), 1.0, places=8)

        smd = self.bench["smd_df"]
        # SMD bruto deve mostrar forte desbalanceamento (> 0.40 em potencial/internet, 4x acima do limiar 0.10)
        self.assertGreater(float(smd["SMD_Bruto"].abs().max()), 0.40)
        # Entropy Balancing deve zerar SMD para < 1e-5
        self.assertLess(float(smd["SMD_Entropy"].abs().max()), 1e-5)

    def test_03_naive_ab_and_canonical_did_biases(self):
        rk = self.bench["ranking_df"].set_index("Metodo")
        self.assertEqual(len(rk), 24)
        # A/B ingênuo superestima em ~38.2%
        self.assertGreater(rk.loc["A/B Simples (Ingênuo)", "Erro_Pct"], 35.0)
        # DiD canônico superestima em ~12.1% por violação de tendências paralelas
        self.assertAlmostEqual(rk.loc["DiD Canônico (2×2)", "Erro_Pct"], 12.06, delta=0.5)

    def test_04_tier_s_estimators_recover_ground_truth(self):
        rk = self.bench["ranking_df"].set_index("Metodo")
        top_methods = [
            "ETWFE (TWFE + Cov×Período, Wooldridge)",
            "A/B + Entropy Balancing",
            "AIPW / Doubly Robust (Cross-Section)",
            "PSM (3-NN + Ajuste de Viés Abadie-Imbens)",
            "Callaway & Sant'Anna / DR-DiD",
            "TWFE + Entropy Balancing",
            "Synthetic Control (Abadie et al.)",
            "Augmented Synthetic Control (Ridge)",
            "DML-IRM (ATTE — Ortogonal)",
            "DML-PLR (Store-Level — Sem Leakage)",
            "DML-PLR (Painel + GroupKFold por Loja)",
        ]
        for m in top_methods:
            err = abs(float(rk.loc[m, "Erro_Pct"]))
            self.assertLess(err, 2.5, f"{m} teve erro excessivo: {err:.2f}%")

    def test_05_event_study_parallel_trends_diagnostic(self):
        es = self.bench["event_study_df"]
        # TWFE simples rejeita pré-tendência linear nula a 5% (p ≈ 0.0235) com inclinação positiva
        self.assertGreater(float(es["slope_pre_simple"].iloc[0]), 60.0)
        self.assertLess(float(es["p_trend_simple"].iloc[0]), 0.05)
        # ETWFE (Cov×Período) NÃO rejeita tendências paralelas (tanto 1-df trend quanto 11-df Wald)
        self.assertGreater(float(es["p_trend_covper"].iloc[0]), 0.15)
        self.assertGreater(float(es["p_pre_covper"].iloc[0]), 0.15)

    def test_06_panel_cross_fitting_leakage_audit_and_cate(self):
        audit = self.bench["leakage_audit"]
        # K-Fold por linha sofre severo colapso/viés (> 25% erro)
        self.assertGreater(abs(audit["dml_plr_naive_err_pct"]), 25.0)
        # GroupKFold por loja e Store-Level corrigem o vazamento (< 2.5% erro)
        self.assertLess(abs(audit["dml_plr_group_err_pct"]), 2.5)
        self.assertLess(abs(audit["dml_plr_store_err_pct"]), 2.5)
        # CATE estimado correlaciona fortemente com o efeito verdadeiro individual
        self.assertGreater(audit["cate_corr_t_learner"], 0.85)


if __name__ == "__main__":
    unittest.main()
