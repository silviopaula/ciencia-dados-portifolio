"""Testes automatizados do projeto Inferência Causal & Otimização de Cupom na Black Friday."""
import unittest

from src.causal_models import (
    fit_bsts_black_friday,
    fit_did_and_dose_response,
    fit_naive_model,
    fit_synthetic_control_bf,
)
from src.dgp import EconomicParams, generate_synthetic_panel, true_incremental_lift_20_to_30
from src.optimizer import evaluate_coupon_grid


class TestCausalLiftMargem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.econ = EconomicParams()
        cls.df = generate_synthetic_panel(seed=2025)
        cls.naive = fit_naive_model(cls.df)
        cls.bsts = fit_bsts_black_friday(cls.df, n_draws=2000, seed=42)
        cls.sc = fit_synthetic_control_bf(cls.df)
        cls.did = fit_did_and_dose_response(cls.df, seed=42)
        cls.opt = evaluate_coupon_grid(cls.df, cls.did["fitted_curves"])

    def test_01_exact_breakeven_and_dgp_calibration(self):
        # Breakeven exato de +28,00% ao escalar de 20% para 30%
        self.assertAlmostEqual(self.econ.breakeven_lift_20_to_30 * 100.0, 28.00, places=6)
        self.assertEqual(len(self.df), 731 * 16)
        # Lift verdadeiro do salto 20% -> 30% estritamente dentro de 4% a 13% (global e por categoria)
        for cat in [None, "Moveis_Iluminacao", "Textil_Objetos"]:
            lift_pct = true_incremental_lift_20_to_30(cat) * 100.0
            self.assertGreaterEqual(lift_pct, 4.0)
            self.assertLessEqual(lift_pct, 13.0)

    def test_02_naive_model_illusion_above_breakeven(self):
        # Modelo ingênuo superestima na faixa de ~41% (bem acima do breakeven de +28%) com p < 0.001
        self.assertAlmostEqual(self.naive["naive_lift_pct"], 41.0, delta=3.5)
        self.assertGreater(self.naive["naive_lift_pct"], 28.0)
        self.assertLess(self.naive["p_value"], 1e-4)
        # E o cupom de 30% gera prejuízo real de margem vs. 20%
        self.assertLess(self.naive["prejuizo_real_margem_rs"], -10000.0)

    def test_03_bsts_and_synthetic_control_recover_true_lift(self):
        true_l = self.bsts["true_lift_pct"]
        self.assertAlmostEqual(self.bsts["lift_mean_pct"], true_l, delta=3.0)
        self.assertLess(self.bsts["ci_95_pct"][1], 28.0)
        self.assertAlmostEqual(self.sc["lift_sc_pct"], self.sc["true_lift_pct"], delta=2.5)
        self.assertLess(self.sc["pre_rmse_pct"], 5.0)

    def test_04_did_heterogeneous_and_parallel_trends(self):
        # Pré-tendências paralelas não rejeitadas no Event Study
        self.assertGreater(self.did["p_pre_wald"], 0.10)
        self.assertAlmostEqual(self.did["twfe_lift_pct"], self.did["true_lift_pct"], delta=2.0)

        cate = self.did["cate_df"].set_index("categoria")
        # Têxtil & Objetos deve ser substancialmente mais elástico que Móveis & Iluminação
        self.assertGreater(
            float(cate.loc["Textil_Objetos", "lift_est_pct"]),
            float(cate.loc["Moveis_Iluminacao", "lift_est_pct"]) + 4.0,
        )

    def test_05_coupon_optimization_and_segmentation_gains(self):
        # Cupom único ótimo global deve estar na faixa intermediária saudável (10% a 20%)
        self.assertGreaterEqual(self.opt["d_star_single_pct"], 10.0)
        self.assertLessEqual(self.opt["d_star_single_pct"], 20.0)

        # Política segmentada recomenda cupom maior para Têxtil do que para Móveis
        d_mov = self.opt["opt_by_cat"]["Moveis_Iluminacao"]["d_star_pct"]
        d_tex = self.opt["opt_by_cat"]["Textil_Objetos"]["d_star_pct"]
        self.assertLess(d_mov, d_tex)

        # Hierarquia financeira real: Política Segmentada > Cupom Único Ótimo > Status Quo 20% > Cupom 30%
        pol = self.opt["policy_comparison_df"].set_index("Politica")["Lucro_Liquido_Real_RS"].to_dict()
        vals = list(pol.values())
        p_30, _, p_20, p_single, p_seg = vals[0], vals[1], vals[2], vals[3], vals[4]
        self.assertGreater(p_seg, p_single)
        self.assertGreater(p_single, p_20)
        self.assertGreater(p_20, p_30)


if __name__ == "__main__":
    unittest.main()
