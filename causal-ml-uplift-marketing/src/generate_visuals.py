"""
Geração de 6 gráficos em qualidade de publicação para o estudo de
Causal Machine Learning & Uplift Modeling em Promoções (DoorDash KDD 2025).

Cada gráfico está modularizado em sua própria função `plot_*` (retornando `Figure`),
permitindo renderização inline passo a passo no notebook e geração em lote via CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.uplift_engine import run_all_experiments

PLOT_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#fafbfc",
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.alpha": 0.75,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.2,
    "legend.frameon": True,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#e2e8f0",
    "text.parse_math": False,
}
plt.rcParams.update(PLOT_STYLE)

C_BLUE = "#2563eb"
C_GREEN = "#059669"
C_PURPLE = "#7c3aed"
C_ORANGE = "#d97706"
C_RED = "#dc2626"
C_SLATE = "#64748b"
C_DARK = "#0f172a"


def _fmt_usd(x: float, _pos: Any = None) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}US$ {abs(x):,.0f}".replace(",", ".")


def _fmt_usd_str(x: float) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}US$ {abs(x):,.0f}".replace(",", ".")


def _fmt_num_pt(x: float, dec: int = 2) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def plot_case1_confounding_and_heterogeneity(
    df_test: pd.DataFrame,
    true_ate: float,
    naive_ate: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 1: Confundimento no Propensity Score + Heterogeneidade Verdadeira do CATE (Caso 3.1)."""
    plt.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2))

    w = df_test["T"].to_numpy() == 1
    bins = np.linspace(0.15, 0.99, 20)
    ax1.hist(df_test.loc[w, "propensity_true"], bins=bins, density=True, alpha=0.58, color=C_BLUE, edgecolor="white", label=f"Tratados (n = {w.sum():,})".replace(",", "."))
    ax1.hist(df_test.loc[~w, "propensity_true"], bins=bins, density=True, alpha=0.58, color=C_SLATE, edgecolor="white", label=f"Controles (n = {(~w).sum():,})".replace(",", "."))
    ax1.axvline(float(df_test.loc[w, "propensity_true"].mean()), color=C_BLUE, ls="--", lw=1.8)
    ax1.axvline(float(df_test.loc[~w, "propensity_true"].mean()), color=C_SLATE, ls="--", lw=1.8)
    ax1.set_title(
        f"Viés de Seleção Observável (Dif. Ingênua = +{_fmt_num_pt(naive_ate*100, 1)} pp vs. ATE = +{_fmt_num_pt(true_ate*100, 1)} pp)",
        fontsize=10.8,
    )
    ax1.set_xlabel("Propensity Score Verdadeiro P(T = 1 | X)")
    ax1.set_ylabel("Densidade")
    ax1.legend(loc="upper left", fontsize=9)

    cate_pp = df_test["true_cate"].to_numpy() * 100.0
    ax2.hist(cate_pp, bins=32, color=C_GREEN, alpha=0.72, edgecolor="white")
    ax2.axvline(0.0, color=C_RED, ls="--", lw=1.8, label="Uplift Nulo (Sleeping Dogs < 0)")
    ax2.axvline(true_ate * 100.0, color=C_DARK, ls="-.", lw=1.8, label=f"ATE Verdadeiro (+{_fmt_num_pt(true_ate*100, 2)} pp)")
    ax2.axvspan(float(cate_pp.min()) - 0.5, 0.0, color="#fee2e2", alpha=0.45)
    pct_neg = float(np.mean(cate_pp < 0.0)) * 100.0
    ax2.text(
        float(cate_pp.min()) + 0.3, ax2.get_ylim()[1] * 0.78,
        f"Sleeping Dogs:\n{_fmt_num_pt(pct_neg, 1)}% dos usuários",
        color=C_RED, fontsize=8.8, fontweight="bold",
    )
    ax2.set_title("Distribuição do CATE Verdadeiro τ(X) — Justificativa do Targeting", fontsize=11.0)
    ax2.set_xlabel("Uplift Individual Verdadeiro (pontos percentuais)")
    ax2.set_ylabel("Frequência de Usuários (Teste)")
    ax2.legend(loc="upper right", fontsize=8.8)

    fig.suptitle("Caso 3.1 (DoorDash Discreto): Confundimento Histórico e Heterogeneidade de Resposta", fontsize=13.4, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_case1_uplift_curves_and_audit(
    case1: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 2: Curvas de Ganho Cumulativo (Uplift / AUUC) + Auditoria Qini Ingênuo vs. DR-AUUC."""
    plt.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [1.3, 1.0]})

    palette = {
        "Oráculo (Teto)": (C_DARK, ":", 2.2),
        "R-Learner (Discrete)": (C_GREEN, "-", 2.5),
        "X-Learner": (C_BLUE, "-", 2.2),
        "DR-Learner (DML-Het)": (C_PURPLE, "--", 2.1),
        "T-Learner": (C_ORANGE, "-.", 1.9),
        "S-Learner": ("#0891b2", ":", 1.9),
        "DML-PLR (Constante)": (C_SLATE, "--", 1.8),
    }
    curves = case1["curves_oracle"]
    for name, (col, ls, lw) in palette.items():
        if name not in curves:
            continue
        c = curves[name]
        lbl = f"{name} (AUUC = {_fmt_num_pt(c['auuc']*100, 2)})"
        ax1.plot(c["frac"] * 100.0, c["cum_gain"] * 100.0, color=col, ls=ls, lw=lw, label=lbl)

    ref = curves["Oráculo (Teto)"]
    ax1.plot(ref["frac"] * 100.0, ref["random_line"] * 100.0, color=C_RED, ls=":", lw=1.4, label="Targeting Aleatório (AUUC = 0,00)")
    ax1.axvline(30.0, color="#94a3b8", ls="--", lw=1.2)
    ax1.text(31.5, float(ref["cum_gain"].max()) * 100.0 * 0.12, "Top 30%", color="#475569", fontsize=8.8, fontweight="bold")
    ax1.set_title("Curvas de Uplift Cumulativo (Out-of-Sample vs. Oráculo)")
    ax1.set_xlabel("% da População Abordada (ordenada por τ̂ decrescente)")
    ax1.set_ylabel("Conversões Incrementais Cumulativas (% da base)")
    ax1.legend(loc="lower right", fontsize=8.0)

    # Painel direito: Auditoria Qini Ingênuo vs. Oracle AUUC & DR-AUUC
    mdf = case1["metrics_df"].copy()
    y = np.arange(len(mdf))
    h = 0.36
    ax2.barh(y + h / 2, mdf["Oracle_AUUC"] * 100.0, height=h, color=C_GREEN, label="Oracle AUUC × 100 (Verdadeiro)")
    ax2.barh(y - h / 2, mdf["DR_AUUC"] * 100.0, height=h, color=C_BLUE, alpha=0.85, label="DR-AUUC × 100 (Offline Debiased)")
    ax2.axvline(0.0, color=C_DARK, lw=1.2)
    for i, row in mdf.iterrows():
        v = row["Oracle_AUUC"] * 100.0
        ax2.text(max(v, 0.0) + 0.05, i + h / 2, f"{_fmt_num_pt(v, 2)}", va="center", fontsize=8.2, fontweight="bold", color=C_GREEN)
    ax2.set_yticks(y)
    ax2.set_yticklabels(mdf["Modelo"], fontsize=8.8)
    ax2.invert_yaxis()
    ax2.set_xlim(min(-0.15, float(mdf["DR_AUUC"].min()) * 100.0 - 0.1), float(mdf["Oracle_AUUC"].max()) * 100.0 + 0.48)
    ax2.set_xlabel("AUUC Acima do Aleatório (× 100; Modelo Constante = 0)")
    ax2.set_title("Avaliação Debiased: DR-AUUC Alinha com Oráculo")
    ax2.legend(loc="lower right", fontsize=8.2)

    fig.suptitle("Caso 3.1: Ranking Causal de 6 Estimadores e Validação Offline Sem Viés de Confundimento", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_case1_policy_economics(
    case1: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 3: Economia do Targeting (Custo vs. Pedidos Incrementais e Lucro Líquido — DoorDash Tabela 2)."""
    plt.rcParams.update(PLOT_STYLE)
    pb = case1["policy_best"]
    po = case1["policy_oracle"]
    pr = case1["policy_random"]
    sp = case1["summary_policy"]
    best_name = case1["best_model"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2))

    # Painel A: Fronteira Custo vs. Pedidos Incrementais
    ax1.plot(po["total_cost"], po["incremental_orders"], "o:", color=C_DARK, lw=2.0, ms=4, label="Oráculo (Teto Teórico)")
    ax1.plot(pb["total_cost"], pb["incremental_orders"], "s-", color=C_GREEN, lw=2.5, ms=5, label=f"Targeting Causal ({best_name})")
    ax1.plot(pr["total_cost"], pr["incremental_orders"], "^--", color=C_RED, lw=1.8, ms=4, label="Targeting Aleatório / Uniforme")

    # Destacar ponto ótimo de lucro líquido vs. Uniforme 100%
    opt_row = pb.loc[pb["net_profit"].idxmax()]
    unif_row = pb.loc[pb["pct_targeted"] == 1.0].iloc[0]
    ax1.scatter([opt_row["total_cost"]], [opt_row["incremental_orders"]], color=C_BLUE, s=130, zorder=6, edgecolor="white", lw=1.5)
    ax1.annotate(
        f"Ótimo (Top {int(opt_row['pct_targeted']*100)}%)\nCost Index = {_fmt_num_pt(opt_row['cost_index'], 1)}\n(−{_fmt_num_pt(sp['cost_reduction_pct'], 1)}% custo)",
        xy=(opt_row["total_cost"], opt_row["incremental_orders"]),
        xytext=(opt_row["total_cost"] + unif_row["total_cost"] * 0.14, opt_row["incremental_orders"] - 95),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.5),
        fontsize=8.6, fontweight="bold", color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec=C_BLUE, alpha=0.95),
    )
    ax1.scatter([unif_row["total_cost"]], [unif_row["incremental_orders"]], color=C_RED, s=110, zorder=6, edgecolor="white", lw=1.5)
    ax1.annotate(
        "Uniforme (100%)\nCost Index = 100",
        xy=(unif_row["total_cost"], unif_row["incremental_orders"]),
        xytext=(unif_row["total_cost"] * 0.68, unif_row["incremental_orders"] - 140),
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4),
        fontsize=8.4, fontweight="bold", color=C_RED,
        bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec=C_RED, alpha=0.92),
    )
    ax1.xaxis.set_major_locator(mtick.MaxNLocator(5))
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_usd))
    ax1.set_xlim(0, unif_row["total_cost"] * 1.08)
    ax1.set_xlabel("Custo Promocional Esperado no Teste (US$)")
    ax1.set_ylabel("Pedidos Incrementais Verdadeiros")
    ax1.set_title("Fronteira Custo × Pedidos Incrementais (DoorDash Tabela 2)")
    ax1.legend(loc="upper left", fontsize=8.5)

    # Painel B: Lucro Incremental Líquido por % da Base Abordada
    ax2.plot(po["pct_targeted"] * 100, po["net_profit"], ":", color=C_DARK, lw=2.0, label="Oráculo (Teto)")
    ax2.plot(pb["pct_targeted"] * 100, pb["net_profit"], "o-", color=C_GREEN, lw=2.5, ms=4.8, label=f"Targeting Causal ({best_name})")
    ax2.plot(pr["pct_targeted"] * 100, pr["net_profit"], "--", color=C_RED, lw=1.8, label="Targeting Aleatório")
    ax2.axhline(0.0, color="#334155", lw=1.1)
    ax2.axvline(opt_row["pct_targeted"] * 100, color=C_BLUE, ls="--", lw=1.5, label=f"Corte Ótimo = Top {int(opt_row['pct_targeted']*100)}%")
    ax2.axhline(unif_row["net_profit"], color=C_SLATE, ls=":", lw=1.3, label=f"Lucro Uniforme ({_fmt_usd_str(unif_row['net_profit'])})")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_usd))
    ax2.set_xlabel("% da Base de Clientes Abordada")
    ax2.set_ylabel("Lucro Incremental Líquido (US$)")
    ax2.set_title(
        f"Lucro Líquido: +{_fmt_usd_str(sp['best_net_profit'])} vs. Uniforme {_fmt_usd_str(sp['uniform_net_profit'])} (+{_fmt_usd_str(sp['profit_delta_usd'])})",
        fontsize=10.3,
    )
    ax2.legend(loc="lower left", fontsize=8.1)

    fig.suptitle("Caso 3.1: Impacto Financeiro do Targeting Causal vs. Campanha Uniforme (Out-of-Sample)", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_case2_tau_estimation_comparison(
    case2: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 4: Caso 3.2 — DML Contínuo Ingênuo (Shrinkage/Truncamento) vs. R-Learner Ponderado (Z_res²)."""
    plt.rcParams.update(PLOT_STYLE)
    df_te = case2["df_test"]
    comp = case2["comp_df"]
    true_t = df_te["true_tau"].to_numpy()
    p_naive = df_te["tau_pred_naive"].to_numpy()
    p_rl = df_te["tau_pred_rlearner"].to_numpy()

    r_naive = float(comp.iloc[0]["Corr_Pearson"])
    pehe_naive = float(comp.iloc[0]["PEHE_Tau"])
    r_rl = float(comp.iloc[1]["Corr_Pearson"])
    pehe_rl = float(comp.iloc[1]["PEHE_Tau"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.3), sharey=True)

    # Amostra visual limpa de 1.500 pontos para legibilidade sem borrão
    rng = np.random.default_rng(42)
    idx = rng.choice(len(true_t), size=min(1500, len(true_t)), replace=False)

    for ax, pred, title, col, r_val, pehe_val in [
        (
            ax1, p_naive,
            f"DML Ingênuo (Truncado s/ Pesos) — r = {_fmt_num_pt(r_naive, 3)} | PEHE = {_fmt_num_pt(pehe_naive, 4)}",
            C_RED, r_naive, pehe_naive,
        ),
        (
            ax2, p_rl,
            f"R-Learner Contínuo (Ponderado Z_res²) — r = {_fmt_num_pt(r_rl, 3)} | PEHE = {_fmt_num_pt(pehe_rl, 4)}",
            C_GREEN, r_rl, pehe_rl,
        ),
    ]:
        ax.scatter(true_t[idx], pred[idx], alpha=0.28, s=14, color=col, edgecolor="none", label="Usuários (Teste)")
        ax.plot([0.0, 0.19], [0.0, 0.19], "k--", lw=1.8, label="Calibração Perfeita (τ̂ = τ)")
        # Decis médios para mostrar calibração limpa
        bins = pd.qcut(true_t, 10, duplicates="drop")
        df_b = pd.DataFrame({"t": true_t, "p": pred, "b": bins}).groupby("b", observed=False).mean()
        ax.plot(df_b["t"], df_b["p"], "o-", color=C_DARK, lw=2.2, ms=6, label="Média por Decil de τ")
        ax.set_xlim(0.0, 0.19)
        ax.set_ylim(0.0, 0.19)
        ax.set_xlabel("Sensibilidade Verdadeira τ(X) por unidade de ln(1 + t)")
        ax.set_title(title, fontsize=10.6)
        ax.legend(loc="upper left", fontsize=8.5)

    ax1.set_ylabel("Sensibilidade Estimada τ̂(X)")
    fig.suptitle("Caso 3.2 (Tratamento Contínuo): Por Que Ponderação por Z_res² Elimina o Shrinkage", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_case2_pareto_frontier(
    case2: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 5: Caso 3.2 — Fronteira de Pareto (Otimização Lagrangiana DoorDash Eq. 2.2.3 vs. Uniforme)."""
    plt.rcParams.update(PLOT_STYLE)
    fr = case2["frontier_r"]
    fn = case2["frontier_naive"]
    fo = case2["frontier_oracle"]
    uc = case2["uniform_curve"]
    sp = case2["summary_policy"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3))

    # Painel A: Fronteira de Pareto Custo vs. Pedidos Incrementais
    ax1.plot(fo["total_cost"], fo["incremental_orders"], ":", color=C_DARK, lw=2.2, label="Oráculo (Teto Teórico)")
    ax1.plot(fr["total_cost"], fr["incremental_orders"], "o-", color=C_GREEN, lw=2.5, ms=4.5, label="R-Learner Contínuo + Lagrangiano")
    ax1.plot(fn["total_cost"], fn["incremental_orders"], "s--", color=C_ORANGE, lw=1.9, ms=4.0, label="DML Ingênuo + Lagrangiano")
    ax1.plot(uc["total_cost"], uc["incremental_orders"], "^--", color=C_RED, lw=2.1, ms=5.0, label="Desconto Uniforme ($0,50 a $5,00)")

    # Seta horizontal destacando economia de custo mantendo o mesmo volume de conversões do uniforme $2.50!
    ax1.scatter([sp["uniform_cost"]], [sp["uniform_inc_orders"]], color=C_RED, s=110, zorder=6, edgecolor="white", lw=1.5)
    ax1.scatter([sp["rlearner_iso_cost"]], [sp["rlearner_iso_orders"]], color=C_BLUE, s=120, zorder=6, edgecolor="white", lw=1.5)
    ax1.plot(
        [sp["rlearner_iso_cost"], sp["uniform_cost"]],
        [sp["uniform_inc_orders"], sp["uniform_inc_orders"]],
        color=C_BLUE, ls="--", lw=1.5, alpha=0.8,
    )
    ax1.annotate(
        f"Mesmo Lift com\n−{_fmt_num_pt(sp['rlearner_iso_savings_pct'], 1)}% de Custo!\n(Cost Index = {_fmt_num_pt(sp['rlearner_iso_cost_index'], 1)})",
        xy=(sp["rlearner_iso_cost"], sp["rlearner_iso_orders"]),
        xytext=(sp["rlearner_iso_cost"] + float(uc["total_cost"].max()) * 0.14, sp["rlearner_iso_orders"] - float(fo["incremental_orders"].max()) * 0.16),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.6),
        fontsize=8.6, fontweight="bold", color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec=C_BLUE, alpha=0.95),
    )
    ax1.xaxis.set_major_locator(mtick.MaxNLocator(5))
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_usd))
    ax1.set_xlabel("Custo Promocional Esperado no Teste (US$)")
    ax1.set_ylabel("Pedidos Incrementais Verdadeiros")
    ax1.set_title("Fronteira de Pareto: Custo vs. Conversões Incrementais")
    ax1.legend(loc="lower right", fontsize=8.2)

    # Painel B: Lucro Incremental Líquido vs. Orçamento Gasto
    ax2.plot(fo["total_cost"], fo["net_profit"], ":", color=C_DARK, lw=2.2, label="Oráculo (Teto)")
    ax2.plot(fr["total_cost"], fr["net_profit"], "o-", color=C_GREEN, lw=2.5, ms=4.5, label="R-Learner Contínuo + Lagrangiano")
    ax2.plot(fn["total_cost"], fn["net_profit"], "s--", color=C_ORANGE, lw=1.9, ms=4.0, label="DML Ingênuo + Lagrangiano")
    ax2.plot(uc["total_cost"], uc["net_profit"], "^--", color=C_RED, lw=2.1, ms=5.0, label="Desconto Uniforme")
    ax2.axhline(0.0, color="#334155", lw=1.1)
    ax2.xaxis.set_major_locator(mtick.MaxNLocator(5))
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_usd))
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_usd))
    ax2.set_xlabel("Custo Promocional Esperado no Teste (US$)")
    ax2.set_ylabel("Lucro Incremental Líquido (US$)")
    ax2.set_title(
        f"Lucro Líquido: R-Learner +{_fmt_usd_str(sp['rlearner_max_profit'])} vs. Uniforme {_fmt_usd_str(sp['uniform_net_profit'])}",
        fontsize=10.6,
    )
    ax2.legend(loc="lower right", fontsize=8.3)

    fig.suptitle("Caso 3.2 (DoorDash Tabela 3): Otimização Contínua de Descontos via Multiplicador Lagrangiano λ", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_case2_discount_personalization(
    case2: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 6: Distribuição do Desconto Ótimo Personalizado (λ=1) e Segmentação Novos vs. Existentes."""
    plt.rcParams.update(PLOT_STYLE)
    df_te = case2["df_test"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    grid = np.arange(0.0, 5.25, 0.5)
    new_mask = df_te["is_new_user"].to_numpy() == 1
    pct_new = [float(np.mean(df_te.loc[new_mask, "optimal_discount_lam1"] == d)) * 100.0 for d in grid]
    pct_old = [float(np.mean(df_te.loc[~new_mask, "optimal_discount_lam1"] == d)) * 100.0 for d in grid]

    x = np.arange(len(grid))
    w = 0.38
    ax1.bar(x - w / 2, pct_old, width=w, color=C_SLATE, label=f"Usuários Existentes (n = {(~new_mask).sum():,})".replace(",", "."))
    ax1.bar(x + w / 2, pct_new, width=w, color=C_BLUE, label=f"Usuários Novos (n = {new_mask.sum():,})".replace(",", "."))
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"${d:.1f}".replace(".", ",") for d in grid], fontsize=8.8)
    ax1.set_ylim(0.0, max(max(pct_new), max(pct_old)) * 1.28)
    ax1.set_xlabel("Desconto Ótimo Recomendado t*(x) em λ = 1,0")
    ax1.set_ylabel("% de Usuários no Segmento")
    ax1.set_title("Personalização Real do Desconto: Novos vs. Existentes")
    ax1.legend(loc="upper right", fontsize=8.8)

    # Painel B: Curva Dose-Resposta Média Estimada por Segmento (retornos decrescentes!)
    t_line = np.linspace(0.0, 5.0, 60)
    z_line = np.log1p(t_line)
    tau_new = float(df_te.loc[new_mask, "tau_pred_rlearner"].mean())
    tau_old = float(df_te.loc[~new_mask, "tau_pred_rlearner"].mean())
    tau_new_true = float(df_te.loc[new_mask, "true_tau"].mean())
    tau_old_true = float(df_te.loc[~new_mask, "true_tau"].mean())

    ax2.plot(t_line, tau_new_true * z_line * 100, ":", color=C_DARK, lw=2.0, label="Novos — Verdadeiro (DGP)")
    ax2.plot(t_line, tau_new * z_line * 100, "-", color=C_BLUE, lw=2.5, label=f"Novos — R-Learner (τ̂ = {_fmt_num_pt(tau_new, 3)})")
    ax2.plot(t_line, tau_old_true * z_line * 100, ":", color=C_RED, lw=2.0, label="Existentes — Verdadeiro (DGP)")
    ax2.plot(t_line, tau_old * z_line * 100, "-", color=C_SLATE, lw=2.5, label=f"Existentes — R-Learner (τ̂ = {_fmt_num_pt(tau_old, 3)})")
    ax2.set_xlabel("Profundidade do Desconto t (US$)")
    ax2.set_ylabel("Lift Incremental de Conversão (pp)")
    ax2.set_title("Curva Dose-Resposta Côncava: Retornos Decrescentes")
    ax2.legend(loc="upper left", fontsize=8.4)

    fig.suptitle("Caso 3.2: Alocação Ótima de Descontos e Curvas Dose-Resposta por Ciclo de Vida", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def generate_all_figures(results: Dict[str, Any], img_dir: Path) -> Dict[str, Path]:
    img_dir.mkdir(parents=True, exist_ok=True)
    c1, c2 = results["case1"], results["case2"]
    paths = {
        "fig1": img_dir / "01_caso1_confundimento_heterogeneidade.png",
        "fig2": img_dir / "02_caso1_curvas_uplift_auditoria.png",
        "fig3": img_dir / "03_caso1_politica_targeting_economia.png",
        "fig4": img_dir / "04_caso2_tau_ingenuo_vs_rlearner.png",
        "fig5": img_dir / "05_caso2_fronteira_pareto_descontos.png",
        "fig6": img_dir / "06_caso2_personalizacao_dose_resposta.png",
    }
    for fig in [
        plot_case1_confounding_and_heterogeneity(c1["df_test"], c1["true_ate"], c1["naive_ate"], paths["fig1"]),
        plot_case1_uplift_curves_and_audit(c1, paths["fig2"]),
        plot_case1_policy_economics(c1, paths["fig3"]),
        plot_case2_tau_estimation_comparison(c2, paths["fig4"]),
        plot_case2_pareto_frontier(c2, paths["fig5"]),
        plot_case2_discount_personalization(c2, paths["fig6"]),
    ]:
        plt.close(fig)
    return paths


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    proj_root = Path(__file__).resolve().parent.parent
    res = run_all_experiments(seed=42)
    paths = generate_all_figures(res, proj_root / "img")
    for k, p in paths.items():
        print(f"[OK] {k}: {p.name} ({p.stat().st_size:,} bytes)")
