"""Testes automatizados para Causal ML & Uplift Modeling (DoorDash KDD 2025)."""
import unittest
import numpy as np

from src.uplift_engine import (
    optimize_continuous_discount_lagrangian,
    run_all_experiments,
    simulate_continuous_discount_data,
    simulate_discrete_promo_data,
)


class TestCausalMLUpliftDoorDash(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_all_experiments(seed=42)
        cls.c1 = cls.res["case1"]
        cls.c2 = cls.res["case2"]

    def test_01_discrete_dgp_confounding_and_heterogeneity(self):
        df = simulate_discrete_promo_data(n=5000, seed=42)
        self.assertEqual(len(df), 5000)
        self.assertEqual(int(df.isna().sum().sum()), 0)
        # Confundimento positivo: diferença ingênua superestima substancialmente o ATE verdadeiro
        self.assertGreater(self.c1["naive_ate"] - self.c1["true_ate"], 0.05)
        # Heterogeneidade real incluindo Sleeping Dogs (true_cate < 0) e Persuadables (> 0.18)
        self.assertLess(float(df["true_cate"].min()), 0.0)
        self.assertGreater(float(df["true_cate"].max()), 0.18)

    def test_02_constant_dml_has_zero_auuc_and_unbiased_ate(self):
        mdf = self.c1["metrics_df"].set_index("Modelo")
        # DML-PLR constante deve recuperar ATE próximo do verdadeiro (< 1.5 pp erro)
        ate_err = abs(float(mdf.loc["DML-PLR (Constante)", "ATE_Estimado_pp"]) - self.c1["true_ate"] * 100.0)
        self.assertLess(ate_err, 1.5)
        # Mas por ser constante, seu Oracle AUUC acima do aleatório deve ser ~0
        self.assertLess(abs(float(mdf.loc["DML-PLR (Constante)", "Oracle_AUUC"])), 0.002)
        # Enquanto o Qini ingênuo enganosamente dá um número alto (> 15%) por confundimento!
        self.assertGreater(float(mdf.loc["DML-PLR (Constante)", "Naive_Qini_Viesado"]), 14.0)

    def test_03_heterogeneous_learners_beat_random_and_dml_constant(self):
        mdf = self.c1["metrics_df"].set_index("Modelo")
        for m in ["R-Learner (Discrete)", "DR-Learner (DML-Het)", "X-Learner"]:
            self.assertGreater(float(mdf.loc[m, "Oracle_AUUC"]), 0.012, f"{m} com Oracle AUUC baixo")
            self.assertGreater(float(mdf.loc[m, "Corr_Pearson"]), 0.65, f"{m} com correlação CATE baixa")
        # Top 30% do melhor modelo deve entregar uplift muito superior ao ATE médio
        best = self.c1["best_model"]
        self.assertGreater(float(mdf.loc[best, "True_Uplift_Top30_pp"]), self.c1["true_ate"] * 100.0 + 4.0)

    def test_04_case1_targeting_policy_improves_profit_and_cuts_cost(self):
        sp = self.c1["summary_policy"]
        # Targeting seletivo deve reduzir custo em >= 40% e aumentar lucro líquido sobre uniforme 100%
        self.assertGreater(sp["cost_reduction_pct"], 40.0)
        self.assertGreater(sp["best_net_profit"], sp["uniform_net_profit"] * 1.5)

    def test_05_case2_weighted_rlearner_crushes_naive_clipped_dml(self):
        comp = self.c2["comp_df"].set_index("Modelo")
        r_naive = float(comp.loc["DML Contínuo Ingênuo (Truncado s/ Pesos)", "Corr_Pearson"])
        r_rl = float(comp.loc["R-Learner Contínuo (Ponderado Z_res²)", "Corr_Pearson"])
        pehe_naive = float(comp.loc["DML Contínuo Ingênuo (Truncado s/ Pesos)", "PEHE_Tau"])
        pehe_rl = float(comp.loc["R-Learner Contínuo (Ponderado Z_res²)", "PEHE_Tau"])

        # R-Learner ponderado por Z_res² deve ter correlação > 0.85 e PEHE bem menor que o ingênuo
        self.assertGreater(r_rl, 0.85)
        self.assertGreater(r_rl, r_naive + 0.15)
        self.assertLess(pehe_rl, pehe_naive)

    def test_06_case2_lagrangian_policy_personalizes_and_saves_cost(self):
        # Checar que a otimização Lagrangiana é não-degenerada (atribui descontos maiores a usuários mais sensíveis)
        grid = np.arange(0.0, 5.25, 0.5)
        tau_sample = np.array([0.01, 0.06, 0.15])
        p0_sample = np.array([0.50, 0.30, 0.15])
        t_opt = optimize_continuous_discount_lagrangian(tau_sample, p0_sample, grid, lam=1.0, margin_per_order=15.0)
        self.assertLessEqual(t_opt[0], t_opt[1])
        self.assertLessEqual(t_opt[1], t_opt[2])
        self.assertLess(t_opt[0], t_opt[2])

        # Fronteira de Pareto: R-Learner economiza >= 40% de custo mantendo o mesmo volume de conversões do uniforme $2.50
        sp = self.c2["summary_policy"]
        self.assertGreater(sp["rlearner_iso_savings_pct"], 40.0)
        self.assertGreater(sp["rlearner_max_profit"], sp["uniform_net_profit"])


if __name__ == "__main__":
    unittest.main()
