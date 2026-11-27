"""
Otimização Prescritiva de Cupom Promocional — Margem Líquida, Incerteza e Segmentação.

Combina as curvas causais estimadas L̂_k(d) com a estrutura financeira da loja para:
  1. Traçar a curva em U invertido do Lucro Líquido Π(d) vs. desconto d ∈ [0%, 35%]
  2. Encontrar o cupom único ótimo global d* e a fronteira de breakeven
  3. Avaliar otimização sob incerteza e aversão ao risco (Cenários P10 / Base / P90 e CVaR)
  4. Calcular a política segmentada ótima por categoria (d*_Móveis vs. d*_Têxtil) e quantificar
     o ganho financeiro sobre o cupom de 30% (erro ingênuo), sem cupom (0%) e cupom único!
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.dgp import CATEGORY_SPECS, EconomicParams, true_lift_curve


def evaluate_coupon_grid(
    df: pd.DataFrame,
    fitted_curves: Dict[str, Dict[str, Any]],
    d_grid: np.ndarray | None = None,
    risk_aversion_lambda: float = 1.5,
) -> Dict[str, Any]:
    """
    Avalia o lucro líquido da campanha Black Friday 2024 (toda a rede, 16 unidades × 10 dias)
    para cada nível de cupom d ∈ [0%, 35%], tanto com os parâmetros causais estimados quanto
    com a verdade sintética (Oráculo) e sob incerteza paramétrica (draws P10/P50/P90 + CVaR).
    """
    if d_grid is None:
        d_grid = np.linspace(0.0, 0.35, 71)  # passos de 0,5 p.p.

    econ = EconomicParams()
    bf24 = df[df["is_bf24"] == 1]

    # Volume orgânico Y(0) verdadeiro na janela da Black Friday 2024 por categoria (toda a rede)
    q0_by_cat = bf24.groupby("categoria")["pedidos_organicos_y0"].sum().to_dict()
    q0_total = float(sum(q0_by_cat.values()))

    # 1. Curvas por categoria (Estimada vs. Verdadeira)
    cat_tables: Dict[str, pd.DataFrame] = {}
    opt_by_cat: Dict[str, Dict[str, float]] = {}

    # Matriz de lucro total nos draws de incerteza: (n_draws, len(d_grid))
    n_draws = min(len(v["ab_draws"]) for v in fitted_curves.values())
    profit_draws_total = np.zeros((n_draws, len(d_grid)), dtype=float)

    for cat_key, spec in CATEGORY_SPECS.items():
        q0_k = float(q0_by_cat[cat_key])
        fc = fitted_curves[cat_key]
        a_hat, b_hat = fc["alpha_hat"], fc["beta_hat"]

        # Margem unitária específica da categoria (cuja média ponderada bate exatamente na loja!)
        cm_k = spec["ticket"] * (spec["m0"] - d_grid) - spec["cv"]
        lift_hat = a_hat * (1.0 - np.exp(-b_hat * d_grid))
        lift_true = np.asarray(true_lift_curve(d_grid, cat_key), dtype=float)

        prof_est_k = q0_k * (1.0 + lift_hat) * cm_k
        prof_true_k = q0_k * (1.0 + lift_true) * cm_k

        # Draws de incerteza para esta categoria
        ab_d = fc["ab_draws"][:n_draws]  # (n_draws, 2)
        lifts_d = ab_d[:, 0:1] * (1.0 - np.exp(-ab_d[:, 1:2] * d_grid[None, :]))
        prof_draws_k = q0_k * (1.0 + lifts_d) * cm_k[None, :]
        profit_draws_total += prof_draws_k

        best_idx = int(np.argmax(prof_est_k))
        d_star_k = float(d_grid[best_idx])
        # Avalia o lucro VERDADEIRO gerado ao aplicar a recomendação estimada d_star_k!
        true_prof_at_dstar = float(prof_true_k[best_idx])

        opt_by_cat[cat_key] = {
            "label": spec["label"],
            "q0_bf24": round(q0_k, 1),
            "d_star_pct": round(d_star_k * 100.0, 1),
            "lift_est_at_dstar_pct": round(float(lift_hat[best_idx]) * 100.0, 2),
            "profit_est_rs": round(float(prof_est_k[best_idx]), 2),
            "profit_true_realized_rs": round(true_prof_at_dstar, 2),
        }

        cat_tables[cat_key] = pd.DataFrame({
            "cupom_pct": np.round(d_grid * 100.0, 2),
            "lift_est_pct": np.round(lift_hat * 100.0, 2),
            "lift_true_pct": np.round(lift_true * 100.0, 2),
            "lucro_est_rs": np.round(prof_est_k, 2),
            "lucro_true_rs": np.round(prof_true_k, 2),
            "lucro_p10_rs": np.round(np.percentile(prof_draws_k, 10, axis=0), 2),
            "lucro_p90_rs": np.round(np.percentile(prof_draws_k, 90, axis=0), 2),
        })

    # 2. Curva Global da Loja (Cupom Único para todas as categorias)
    cm_global = econ.unit_margin(d_grid)
    prof_est_global = sum(t["lucro_est_rs"].to_numpy(float) for t in cat_tables.values())
    prof_true_global = sum(t["lucro_true_rs"].to_numpy(float) for t in cat_tables.values())

    # Lift global estimado implícito e lift mínimo de breakeven contra d = 0% (sem cupom)
    lift_est_global = (
        sum(float(q0_by_cat[k]) * (cat_tables[k]["lift_est_pct"].to_numpy(float) / 100.0) for k in CATEGORY_SPECS)
        / q0_total
    )
    lift_true_global = np.asarray(true_lift_curve(d_grid, None), dtype=float)
    cm0 = float(econ.unit_margin(0.0))
    breakeven_vs_zero_lift = np.where(cm_global > 1.0, (cm0 / np.maximum(cm_global, 1.0)) - 1.0, np.nan)

    # Estatísticas sob incerteza (P10 pessimista, Média, P90 otimista, DP e Equivalente de Certeza)
    p10_global = np.percentile(profit_draws_total, 10, axis=0)
    p90_global = np.percentile(profit_draws_total, 90, axis=0)
    sd_global = np.std(profit_draws_total, axis=0, ddof=1)
    # Utilidade ajustada ao risco: E[Π(d)] - λ_risk * SD(Π(d))
    risk_adj_utility = prof_est_global - risk_aversion_lambda * sd_global

    idx_opt_single = int(np.argmax(prof_est_global))
    idx_opt_risk = int(np.argmax(risk_adj_utility))
    idx_0 = int(np.argmin(np.abs(d_grid - 0.00)))
    idx_20 = int(np.argmin(np.abs(d_grid - 0.20)))
    idx_30 = int(np.argmin(np.abs(d_grid - 0.30)))

    global_curve_df = pd.DataFrame({
        "cupom_pct": np.round(d_grid * 100.0, 2),
        "margem_unit_rs": np.round(cm_global, 2),
        "lift_est_pct": np.round(lift_est_global * 100.0, 2),
        "lift_true_pct": np.round(lift_true_global * 100.0, 2),
        "lift_breakeven_vs_0_pct": np.round(breakeven_vs_zero_lift * 100.0, 2),
        "lucro_est_rs": np.round(prof_est_global, 2),
        "lucro_true_rs": np.round(prof_true_global, 2),
        "lucro_p10_rs": np.round(p10_global, 2),
        "lucro_p90_rs": np.round(p90_global, 2),
        "lucro_sd_rs": np.round(sd_global, 2),
        "utilidade_aversao_risco_rs": np.round(risk_adj_utility, 2),
    })

    # 3. Tabela Executiva de Comparação de Políticas (Lucro Realizado Verdadeiro na BF 2024!)
    prof_segmented_true = sum(v["profit_true_realized_rs"] for v in opt_by_cat.values())
    prof_single_opt_true = float(prof_true_global[idx_opt_single])
    prof_status_quo_20_true = float(prof_true_global[idx_20])
    prof_naive_30_true = float(prof_true_global[idx_30])
    prof_zero_true = float(prof_true_global[idx_0])

    d_star_lbl = f"{d_grid[idx_opt_single]*100:.1f}".replace(".", ",")
    mov_lbl = f"{opt_by_cat['Moveis_Iluminacao']['d_star_pct']:.1f}".replace(".", ",")
    tex_lbl = f"{opt_by_cat['Textil_Objetos']['d_star_pct']:.1f}".replace(".", ",")

    policy_comparison_df = pd.DataFrame([
        {
            "Politica": "1. Cupom 30% (Decisão Ingênua +40%)",
            "Regra_Desconto": "30,0% uniforme",
            "Lucro_Liquido_Real_RS": round(prof_naive_30_true, 2),
            "Ganho_vs_30pct_RS": 0.0,
            "Ganho_vs_20pct_RS": round(prof_naive_30_true - prof_status_quo_20_true, 2),
        },
        {
            "Politica": "2. Sem Cupom (0% — Margem Cheia)",
            "Regra_Desconto": "0,0% uniforme",
            "Lucro_Liquido_Real_RS": round(prof_zero_true, 2),
            "Ganho_vs_30pct_RS": round(prof_zero_true - prof_naive_30_true, 2),
            "Ganho_vs_20pct_RS": round(prof_zero_true - prof_status_quo_20_true, 2),
        },
        {
            "Politica": "3. Cupom 20% (Status Quo Histórico)",
            "Regra_Desconto": "20,0% uniforme",
            "Lucro_Liquido_Real_RS": round(prof_status_quo_20_true, 2),
            "Ganho_vs_30pct_RS": round(prof_status_quo_20_true - prof_naive_30_true, 2),
            "Ganho_vs_20pct_RS": 0.0,
        },
        {
            "Politica": f"4. Cupom Único Ótimo Causal (d* = {d_star_lbl}%)",
            "Regra_Desconto": f"{d_star_lbl}% uniforme",
            "Lucro_Liquido_Real_RS": round(prof_single_opt_true, 2),
            "Ganho_vs_30pct_RS": round(prof_single_opt_true - prof_naive_30_true, 2),
            "Ganho_vs_20pct_RS": round(prof_single_opt_true - prof_status_quo_20_true, 2),
        },
        {
            "Politica": "5. Política Segmentada Causal (CATE)",
            "Regra_Desconto": f"Móveis {mov_lbl}% | Têxtil {tex_lbl}%",
            "Lucro_Liquido_Real_RS": round(prof_segmented_true, 2),
            "Ganho_vs_30pct_RS": round(prof_segmented_true - prof_naive_30_true, 2),
            "Ganho_vs_20pct_RS": round(prof_segmented_true - prof_status_quo_20_true, 2),
        },
    ])

    return {
        "d_star_single_pct": round(float(d_grid[idx_opt_single]) * 100.0, 1),
        "d_star_risk_averse_pct": round(float(d_grid[idx_opt_risk]) * 100.0, 1),
        "opt_by_cat": opt_by_cat,
        "cat_tables": cat_tables,
        "global_curve_df": global_curve_df,
        "policy_comparison_df": policy_comparison_df,
        "money_saved_vs_30_segmented_rs": round(prof_segmented_true - prof_naive_30_true, 2),
        "gain_segmented_vs_single_rs": round(prof_segmented_true - prof_single_opt_true, 2),
    }
