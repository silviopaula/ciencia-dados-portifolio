"""
Visualizações em qualidade de publicação para o projeto de Inferência Causal
e Otimização Prescritiva de Cupom na Black Friday.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

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
    "axes.titlesize": 12.0,
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


def _fmt_brl_mi(x: float, _pos: Any = None) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}R$ {abs(x)/1e6:,.2f} mi".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_brl_k(x: float, _pos: Any = None) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}R$ {abs(x)/1000:,.0f}k".replace(",", ".")


def _fmt_pt(x: float, dec: int = 1) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def plot_01_naive_illusion_and_margin(
    naive_res: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 1 (Módulo 1): Ilusão do +41% vs. Breakeven (+28%) vs. Lift Real e Erosão de Margem."""
    plt.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.3), gridspec_kw={"width_ratios": [1.1, 1.15]})

    # Painel A: Lift Estimado Ingênuo vs. Breakeven vs. Lift Causal Real
    labels = ["Modelo Ingênuo\n(Confunde BF)", "Lift Real (DGP)\nSalto 20% → 30%"]
    vals = [naive_res["naive_lift_pct"], naive_res["true_lift_pct"]]
    errs = [1.96 * naive_res["naive_se_pct"], 0.0]
    cols = [C_RED, C_GREEN]

    bars = ax1.bar(np.arange(2), vals, yerr=errs, capsize=5, width=0.48, color=cols, edgecolor="white")
    be = naive_res["breakeven_pct"]
    ax1.axhline(be, color=C_DARK, ls="--", lw=2.2, label=f"Limiar de Breakeven (+{_fmt_pt(be, 0)}% pedidos)")
    ax1.axhspan(0, be, color="#fee2e2", alpha=0.32, label="Zona de Destruição de Margem (< +28%)")

    for b, v, e in zip(bars, vals, errs):
        ax1.text(
            b.get_x() + b.get_width() / 2.0, v + e + 1.5,
            f"+{_fmt_pt(v, 1)}%", ha="center", va="bottom", fontweight="bold", fontsize=10.5,
        )
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels, fontweight="bold")
    ax1.set_ylim(0, (vals[0] + errs[0]) * 1.22)
    ax1.set_ylabel("Ganho Incremental de Pedidos (20% → 30%)")
    ax1.set_title("A Ilusão do +40%: Confundimento Sazonal vs. Realidade")
    ax1.legend(loc="upper right", fontsize=8.6)

    # Painel B: Lucro Projetado pelo Modelo Ingênuo vs. Lucro Real (Regiões Tratadas na BF 2024)
    p_labels = ["Status Quo\n(Cupom 20%)", "Projeção Ingênua\npara Cupom 30%", "Resultado REAL\ndo Cupom 30%"]
    p_vals = [
        naive_res["lucro_20_rs"],
        naive_res["lucro_projetado_ingenuo_rs"],
        naive_res["lucro_real_30_rs"],
    ]
    p_cols = [C_SLATE, C_ORANGE, C_RED]
    b2 = ax2.bar(np.arange(3), p_vals, width=0.52, color=p_cols, edgecolor="white")
    ax2.axhline(naive_res["lucro_20_rs"], color=C_SLATE, ls=":", lw=1.6)

    for b, v in zip(b2, p_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0, v + max(p_vals) * 0.02,
            f"R$ {_fmt_pt(v/1000, 1)}k", ha="center", va="bottom", fontweight="bold", fontsize=9.6,
        )
    loss = naive_res["prejuizo_real_margem_rs"]
    ax2.annotate(
        f"Erosão Real de Margem:\n−R$ {_fmt_pt(abs(loss), 0)}",
        xy=(2, naive_res["lucro_real_30_rs"]),
        xytext=(1.35, naive_res["lucro_real_30_rs"] * 0.62),
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.8),
        fontsize=9.2, fontweight="bold", color=C_RED,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec=C_RED),
    )
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(p_labels, fontweight="bold", fontsize=9.2)
    ax2.set_ylim(0, max(p_vals) * 1.18)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl_k))
    ax2.set_ylabel("Lucro Líquido na Black Friday (Regiões Tratadas)")
    ax2.set_title("Impacto Financeiro: Ganho Ilusório vs. Prejuízo Real")

    fig.suptitle("Módulo 1: Por Que Confundir o Pico da Black Friday com o Cupom Destrói Margem", fontsize=13.4, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_02_bsts_counterfactual(
    bsts_res: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 2 (Módulo 2A): BSTS Contrafactual na Black Friday + Distribuição Posterior do Lift."""
    plt.rcParams.update(PLOT_STYLE)
    ts = bsts_res["series_df"]
    ts_zoom = ts[ts["data"] >= "2024-10-20"].copy()
    draws = bsts_res["lift_draws_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={"width_ratios": [1.35, 1.0]})

    bf_start = pd.Timestamp("2024-11-22")
    bf_only = ts_zoom[ts_zoom["is_bf24"] == 1]
    ax1.axvspan(bf_start, ts_zoom["data"].max(), color="#dbeafe", alpha=0.40, label="Janela Black Friday 2024")
    ax1.axvline(bf_start, color=C_BLUE, ls="--", lw=1.6)
    ax1.fill_between(ts_zoom["data"], ts_zoom["bsts_lo"], ts_zoom["bsts_hi"], color=C_BLUE, alpha=0.18, label="IC Credibilidade 95% (BSTS)")
    ax1.plot(ts_zoom["data"], ts_zoom["observado"], "o-", color=C_DARK, lw=2.1, ms=3.8, label="Observado (Pré + Cupom 30% na BF)")
    ax1.plot(ts_zoom["data"], ts_zoom["bsts_media"], "--", color=C_BLUE, lw=2.2, label="Contrafactual BSTS (Cupom 20%)")
    ax1.plot(bf_only["data"], bf_only["contrafactual_verdadeiro_20"], "s:", color=C_GREEN, lw=2.2, ms=4.5, label="Verdade Contrafactual Y(20%) na BF")
    ax1.set_title("BSTS: Pedidos Diários Observados vs. Contrafactual (20%)")
    ax1.set_xlabel("Data (Out–Dez/2024)")
    ax1.set_ylabel("Pedidos Diários (Regiões Tratadas)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(loc="upper left", fontsize=8.3)

    ax2.hist(draws, bins=36, density=True, color=C_BLUE, alpha=0.68, edgecolor="white", label="Posterior BSTS (3.000 draws)")
    ax2.axvline(bsts_res["lift_mean_pct"], color=C_BLUE, lw=2.2, label=f"Média Posterior (+{_fmt_pt(bsts_res['lift_mean_pct'], 1)}%)")
    ax2.axvline(bsts_res["true_lift_pct"], color=C_GREEN, ls="-.", lw=2.0, label=f"Lift Verdadeiro (+{_fmt_pt(bsts_res['true_lift_pct'], 1)}%)")
    ax2.axvline(28.0, color=C_RED, ls="--", lw=2.2, label="Breakeven (+28,0%) — P(Lift ≥ 28%) = 0,0%")
    ax2.set_xlim(min(-2.0, float(draws.min()) - 2.0), 32.0)
    ax2.set_xlabel("Lift Causal do Salto 20% → 30% (%)")
    ax2.set_ylabel("Densidade Posterior")
    ax2.set_title("Distribuição Posterior do Lift Causal vs. Breakeven")
    ax2.legend(loc="upper right", fontsize=8.3)

    fig.suptitle("Módulo 2A (BSTS): Isolando o Pico Orgânico da Black Friday via Séries Estruturais", fontsize=13.3, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_03_synthetic_control_placebo(
    sc_res: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 3 (Módulo 2B): Controle Sintético + Testes Placebo In-Space."""
    plt.rcParams.update(PLOT_STYLE)
    s_df = sc_res["sc_series_df"]
    s_zoom = s_df[s_df["data"] >= "2024-10-20"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={"width_ratios": [1.25, 1.05]})

    bf_start = pd.Timestamp("2024-11-22")
    bf_only_sc = s_zoom[s_zoom["is_bf24"] == 1]
    ax1.axvspan(bf_start, s_zoom["data"].max(), color="#ecfdf5", alpha=0.55, label="Janela Black Friday 2024")
    ax1.axvline(bf_start, color=C_GREEN, ls="--", lw=1.6)
    ax1.plot(s_zoom["data"], s_zoom["observado"], "o-", color=C_DARK, lw=2.2, ms=3.8, label="Tratadas Observado (Cupom 30%)")
    ax1.plot(s_zoom["data"], s_zoom["sintetico_20"], "s--", color=C_GREEN, lw=2.1, ms=3.5, label=f"Controle Sintético 20% (Lift = +{_fmt_pt(sc_res['lift_sc_pct'], 1)}%)")
    ax1.plot(bf_only_sc["data"], bf_only_sc["verdadeiro_cf_20"], "d:", color=C_BLUE, lw=2.1, ms=4.5, label=f"Verdade Y(20%) na BF (Lift = +{_fmt_pt(sc_res['true_lift_pct'], 1)}%)")
    ax1.set_title(f"Controle Sintético (RMSE Pré = {_fmt_pt(sc_res['pre_rmse_pct'], 2)}%)")
    ax1.set_xlabel("Data (Out–Dez/2024)")
    ax1.set_ylabel("Pedidos Diários (Regiões Tratadas)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(loc="upper left", fontsize=8.3)

    # Painel direito: Curvas de Gap % no regime simétrico pré-BF (Out–Dez/2024)
    oct_mask = s_df["data"] >= "2024-10-01"
    s_oct = s_df.loc[oct_mask]
    ax2.axvspan(bf_start, s_oct["data"].max(), color="#fef2f2", alpha=0.45)
    ax2.axvline(bf_start, color=C_DARK, ls="--", lw=1.5)
    ax2.axhline(0.0, color="#334155", lw=1.1)
    ax2.axhline(28.0, color=C_RED, ls="--", lw=1.8, label="Breakeven (+28,0%)")

    for i, pl in enumerate(sc_res["placebo_curves"]):
        lbl = "Placebos In-Space (8 Doadores)" if i == 0 else None
        ax2.plot(s_oct["data"], pl["gap_pct"][oct_mask.to_numpy()], color="#94a3b8", lw=1.2, alpha=0.65, label=lbl)
    ax2.plot(s_oct["data"], s_oct["gap_tratada_pct"], color=C_GREEN, lw=2.6, label=f"Gap Tratadas (+{_fmt_pt(sc_res['lift_sc_pct'], 1)}%)")
    ax2.set_ylim(-14.0, 32.0)
    ax2.set_title("Teste de Permutação Placebo In-Space (Gap %)")
    ax2.set_xlabel("Data (Out–Dez/2024)")
    ax2.set_ylabel("Diferença Percentual vs. Sintético (%)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend(loc="upper left", fontsize=8.3)

    fig.suptitle("Módulo 2B (Controle Sintético): Combinação Convexa de Regiões Doadoras em 20%", fontsize=13.3, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_04_did_heterogeneous(
    did_res: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 4 (Módulo 2C): Event Study Dinâmico + CATE por Categoria & Curva Dose-Resposta."""
    plt.rcParams.update(PLOT_STYLE)
    es = did_res["event_study_df"]
    dr = did_res["dose_response_df"]
    fc = did_res["fitted_curves"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), gridspec_kw={"width_ratios": [1.15, 1.15]})

    # Painel A: Event Study Dinâmico
    ax1.axvspan(-0.5, 9.5, color="#ecfdf5", alpha=0.50, label="Janela Black Friday (20% → 30%)")
    ax1.axhline(0.0, color="#334155", lw=1.2)
    ax1.axhline(did_res["true_lift_pct"], color=C_GREEN, ls="-.", lw=1.8, label=f"Lift Verdadeiro Médio (+{_fmt_pt(did_res['true_lift_pct'], 1)}%)")
    ax1.axhline(28.0, color=C_RED, ls="--", lw=1.8, label="Breakeven (+28,0%)")
    ax1.axvline(-0.5, color=C_BLUE, ls="--", lw=1.5)
    yerr_lo = es["lift_pct"] - es["ci_lo"]
    yerr_hi = es["ci_hi"] - es["lift_pct"]
    ax1.errorbar(
        es["rel_day"], es["lift_pct"], yerr=[yerr_lo, yerr_hi],
        fmt="o-", color=C_BLUE, ecolor=C_BLUE, elinewidth=1.4, capsize=3, ms=4.8, lw=1.9,
        label="Coeficiente Event Study ± IC 95%",
    )
    ax1.set_ylim(-10.0, 32.0)
    ax1.set_xlabel("Dias Relativos ao Início da Black Friday 2024 (Ref. = −1)")
    ax1.set_ylabel("Efeito Causal Estimado nos Pedidos (%)")
    ax1.set_title(f"Event Study TWFE (Wald Pré-Tendências p = {_fmt_pt(did_res['p_pre_wald'], 3)})")
    ax1.legend(loc="upper left", fontsize=8.2)

    # Painel B: Curva Dose-Resposta Multi-Faixa por Categoria (Heterogeneidade CATE!)
    d_smooth = np.linspace(0.0, 0.35, 71)
    cat_styles = {
        "Moveis_Iluminacao": (C_PURPLE, "s", "Móveis & Iluminação"),
        "Textil_Objetos": (C_BLUE, "o", "Têxtil & Objetos"),
    }
    for cat_key, (col, mk, lbl) in cat_styles.items():
        sub = dr[dr["categoria"] == cat_key]
        ax2.errorbar(
            sub["cupom_pct"], sub["lift_est_pct"], yerr=1.96 * sub["se_pct"],
            fmt=mk, color=col, capsize=3.5, ms=6.5, label=f"{lbl} — Pontos FE (±IC 95%)",
        )
        a_h, b_h = fc[cat_key]["alpha_hat"], fc[cat_key]["beta_hat"]
        a_t, b_t = fc[cat_key]["alpha_true"], fc[cat_key]["beta_true"]
        ax2.plot(d_smooth * 100, a_h * (1.0 - np.exp(-b_h * d_smooth)) * 100, "-", color=col, lw=2.3, label=f"{lbl} — Curva Ajustada")
        ax2.plot(d_smooth * 100, a_t * (1.0 - np.exp(-b_t * d_smooth)) * 100, ":", color=col, lw=1.6, alpha=0.8)

    ax2.set_xlabel("Percentual de Cupom de Desconto (%)")
    ax2.set_ylabel("Lift Causal sobre Demanda Orgânica Y(0) (%)")
    ax2.set_title("Heterogeneidade Causal: Curvas Dose-Resposta L_k(d)")
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.suptitle("Módulo 2C (DiD Heterogêneo): Pré-Tendências Paralelas e Elasticidades Distintas por Categoria", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_05_inverted_u_profit_curve(
    opt_res: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 5 (Módulo 3A): Curva em U Invertido do Lucro Líquido vs. % de Cupom + Zona de Destruição."""
    plt.rcParams.update(PLOT_STYLE)
    gc = opt_res["global_curve_df"]
    d_star = opt_res["d_star_single_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [1.3, 1.0]})

    # Painel A: Curva em U Invertido do Lucro Líquido (Estimada c/ Banda P10–P90 vs. Verdadeira)
    prof_0 = float(gc.loc[gc["cupom_pct"] == 0.0, "lucro_est_rs"].iloc[0])
    prof_20 = float(gc.loc[gc["cupom_pct"] == 20.0, "lucro_est_rs"].iloc[0])
    prof_30 = float(gc.loc[gc["cupom_pct"] == 30.0, "lucro_est_rs"].iloc[0])
    prof_star = float(gc.loc[gc["cupom_pct"] == d_star, "lucro_est_rs"].iloc[0])

    ax1.axvspan(25.0, 35.0, color="#fee2e2", alpha=0.42, label="Zona de Destruição Severa de Margem (≥ 25%)")
    ax1.fill_between(gc["cupom_pct"], gc["lucro_p10_rs"], gc["lucro_p90_rs"], color=C_GREEN, alpha=0.18, label="Banda de Incerteza Causal (P10–P90)")
    ax1.plot(gc["cupom_pct"], gc["lucro_true_rs"], ":", color=C_DARK, lw=2.0, label="Lucro Verdadeiro (Oráculo DGP)")
    ax1.plot(gc["cupom_pct"], gc["lucro_est_rs"], "-", color=C_GREEN, lw=2.8, label="Lucro Causal Estimado Π(d)")
    ax1.plot(gc["cupom_pct"], gc["utilidade_aversao_risco_rs"], "--", color=C_BLUE, lw=1.8, label="Utilidade Ajustada ao Risco (E[Π] − 1,5·DP)")

    ax1.axhline(prof_0, color=C_SLATE, ls=":", lw=1.4, label="Patamar Sem Cupom (0%)")
    ax1.scatter([d_star], [prof_star], color=C_GREEN, s=135, zorder=6, edgecolor="white", lw=1.6)
    ax1.annotate(
        f"Cupom Único Ótimo\nd* = {_fmt_pt(d_star, 1)}%\n(R$ {_fmt_pt(prof_star/1000, 1)}k)",
        xy=(d_star, prof_star),
        xytext=(d_star - 9.5, prof_star - (prof_star - prof_30) * 0.28),
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.6),
        fontsize=8.8, fontweight="bold", color=C_GREEN,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec=C_GREEN),
    )
    ax1.scatter([30.0], [prof_30], color=C_RED, s=120, zorder=6, edgecolor="white", lw=1.5)
    ax1.annotate(
        f"Cupom 30%\n(R$ {_fmt_pt(prof_30/1000, 1)}k)",
        xy=(30.0, prof_30),
        xytext=(25.2, prof_30 + (prof_star - prof_30) * 0.28),
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
        fontsize=8.6, fontweight="bold", color=C_RED,
        bbox=dict(boxstyle="round,pad=0.28", fc="#fef2f2", ec=C_RED),
    )
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl_k))
    ax1.set_xlabel("Percentual de Cupom de Desconto d (%)")
    ax1.set_ylabel("Lucro Líquido Total na Black Friday (R$)")
    ax1.set_title("Curva em U Invertido do Lucro Líquido Π(d)")
    ax1.legend(loc="lower left", fontsize=8.0)

    # Painel B: Cruzamento entre Lift Real L(d) e Lift Necessário de Breakeven (contra 0%)
    sub_g = gc[gc["cupom_pct"] <= 34.0]
    ax2.plot(sub_g["cupom_pct"], sub_g["lift_est_pct"], "-", color=C_GREEN, lw=2.6, label="Lift Causal Estimado L̂(d)")
    ax2.plot(sub_g["cupom_pct"], sub_g["lift_breakeven_vs_0_pct"], "--", color=C_RED, lw=2.3, label="Lift Mínimo p/ Empatar c/ 0% (Breakeven)")
    ax2.fill_between(
        sub_g["cupom_pct"], sub_g["lift_est_pct"], sub_g["lift_breakeven_vs_0_pct"],
        where=sub_g["lift_est_pct"] >= sub_g["lift_breakeven_vs_0_pct"],
        color="#d1fae5", alpha=0.55, label="Região onde Cupom Agrega Valor vs. 0%",
    )
    ax2.fill_between(
        sub_g["cupom_pct"], sub_g["lift_est_pct"], sub_g["lift_breakeven_vs_0_pct"],
        where=sub_g["lift_est_pct"] < sub_g["lift_breakeven_vs_0_pct"],
        color="#fee2e2", alpha=0.55, label="Região abaixo do Breakeven vs. 0%",
    )
    ax2.axvline(d_star, color=C_GREEN, ls=":", lw=1.8, label=f"Ótimo Global d* = {_fmt_pt(d_star, 1)}%")
    ax2.set_ylim(0, 125)
    ax2.set_xlabel("Percentual de Cupom de Desconto d (%)")
    ax2.set_ylabel("Lift de Pedidos sobre Y(0) (%)")
    ax2.set_title("Anatomia Econômica: Lift Real vs. Curva de Breakeven")
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.suptitle("Módulo 3A: Otimização do Cupom Ideal e Identificação da Zona de Destruição de Margem", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_06_segmented_policy_comparison(
    opt_res: Dict[str, Any],
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 6 (Módulo 3B): Cupons Ótimos por Categoria + Ranking Financeiro das 5 Políticas."""
    plt.rcParams.update(PLOT_STYLE)
    ct = opt_res["cat_tables"]
    obc = opt_res["opt_by_cat"]
    pol = opt_res["policy_comparison_df"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), gridspec_kw={"width_ratios": [1.1, 1.2]})

    # Painel A: Curvas de Lucro por Categoria mostrando d*_Móveis != d*_Têxtil
    for cat_key, col in [("Moveis_Iluminacao", C_PURPLE), ("Textil_Objetos", C_BLUE)]:
        tdf = ct[cat_key]
        d_k = obc[cat_key]["d_star_pct"]
        lbl = obc[cat_key]["label"]
        p_k = float(tdf.loc[tdf["cupom_pct"] == d_k, "lucro_est_rs"].iloc[0])
        ax1.fill_between(tdf["cupom_pct"], tdf["lucro_p10_rs"], tdf["lucro_p90_rs"], color=col, alpha=0.14)
        ax1.plot(tdf["cupom_pct"], tdf["lucro_est_rs"], "-", color=col, lw=2.5, label=f"{lbl} (d* = {_fmt_pt(d_k, 1)}%)")
        ax1.plot(tdf["cupom_pct"], tdf["lucro_true_rs"], ":", color=col, lw=1.6)
        ax1.scatter([d_k], [p_k], color=col, s=110, zorder=6, edgecolor="white", lw=1.5)
        ax1.axvline(d_k, color=col, ls="--", lw=1.3, alpha=0.75)

    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl_k))
    ax1.set_xlabel("Percentual de Cupom de Desconto d (%)")
    ax1.set_ylabel("Lucro Líquido da Categoria na BF (R$)")
    ax1.set_title("Política Personalizada: Ótimo Distinto por Categoria")
    ax1.legend(loc="lower left", fontsize=8.6)

    # Painel B: Comparativo Financeiro entre as 5 Políticas
    y = np.arange(len(pol))
    vals = pol["Lucro_Liquido_Real_RS"].to_numpy(float)
    bar_cols = [C_RED, C_SLATE, C_ORANGE, C_BLUE, C_GREEN]
    bars = ax2.barh(y, vals, height=0.55, color=bar_cols, edgecolor="white")

    for b, v, g30 in zip(bars, vals, pol["Ganho_vs_30pct_RS"]):
        extra = f" (+R$ {_fmt_pt(g30/1000, 0)}k)" if g30 > 0 else " (Erro Base)"
        ax2.text(
            v + max(vals) * 0.015, b.get_y() + b.get_height() / 2.0,
            f"R$ {_fmt_pt(v/1e6, 2)} mi{extra}",
            va="center", fontsize=8.5, fontweight="bold",
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels(pol["Politica"], fontsize=8.9, fontweight="bold")
    ax2.set_xlim(0, max(vals) * 1.56)
    ax2.xaxis.set_major_locator(mtick.MaxNLocator(4))
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl_mi))
    ax2.set_xlabel("Lucro Líquido Realizado na Black Friday (Toda a Rede)")
    ax2.set_title("Ganho Financeiro: Do Erro de 30% à Política Segmentada")

    fig.suptitle("Módulo 3B: Valor Econômico da Inferência Causal e da Segmentação de Cupons", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
