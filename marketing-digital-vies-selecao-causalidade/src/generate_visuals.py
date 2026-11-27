"""
Geração de 6 gráficos em qualidade de publicação para o estudo de Inferência Causal
em Marketing Digital sob Viés de Seleção e Tendências Não-Paralelas.

Cada figura está modularizada em sua própria função `plot_*` (retornando `Figure`),
permitindo renderização inline passo a passo no notebook e geração em lote via CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.causal_benchmark import run_full_benchmark

# Estilo visual consistente (limpo, legível, sem poluição)
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
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "legend.frameon": True,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#e2e8f0",
    "text.parse_math": False,
}
plt.rcParams.update(PLOT_STYLE)

C_TREAT = "#2563eb"
C_CTRL = "#64748b"
C_GOOD = "#059669"
C_WARN = "#d97706"
C_BAD = "#dc2626"
C_PURPLE = "#7c3aed"


def _fmt_brl(x: float, _pos: Any = None) -> str:
    return f"R$ {x:,.0f}".replace(",", ".")


def _fmt_brl_str(val: float, dec: int = 0) -> str:
    s = f"{val:,.{dec}f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")


def plot_trajetorias_vies(
    df_full: pd.DataFrame,
    true_att: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 1: Evolução das vendas médias (Tratadas vs. Controle) e anatomia do viés pré-campanha."""
    plt.rcParams.update(PLOT_STYLE)
    traj = df_full.groupby(["periodo", "tratamento"], as_index=False)["vendas"].mean()
    tr_t = traj[traj["tratamento"] == 1].sort_values("periodo")
    tr_c = traj[traj["tratamento"] == 0].sort_values("periodo")
    gap = tr_t["vendas"].to_numpy() - tr_c["vendas"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax1.axvspan(12.5, 24.5, color="#dbeafe", alpha=0.35, label="Período Pós-Campanha (t ≥ 13)")
    ax1.axvline(12.5, color="#1e40af", linestyle="--", linewidth=1.6)
    ax1.plot(tr_t["periodo"], tr_t["vendas"], "o-", color=C_TREAT, lw=2.5, ms=5.5, label="Lojas Tratadas (n = 155)")
    ax1.plot(tr_c["periodo"], tr_c["vendas"], "s--", color=C_CTRL, lw=2.2, ms=5, label="Lojas Controle (n = 95)")
    ax1.set_title("Evolução das Vendas Médias Mensais (Tratadas vs. Controle)")
    ax1.set_xlabel("Mês (Período)")
    ax1.set_ylabel("Vendas Médias por Loja")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax1.set_xticks(range(1, 25, 2))
    ax1.legend(loc="upper left", fontsize=9)

    ax2.axvspan(12.5, 24.5, color="#fef3c7", alpha=0.35)
    ax2.axvline(12.5, color="#1e40af", linestyle="--", linewidth=1.6)
    ax2.plot(tr_t["periodo"], gap, "o-", color=C_BAD, lw=2.4, ms=5.5, label="Diferença Bruta (Tratadas − Controle)")
    t_pre = np.arange(1, 13)
    slope_pre, int_pre = np.polyfit(t_pre, gap[:12], 1)
    t_all = np.arange(1, 25)
    ax2.plot(
        t_all, int_pre + slope_pre * t_all, ":", color="#7c2d12", lw=2.0,
        label=f"Pré-tendência diferencial (+{_fmt_brl_str(slope_pre, 0)}/mês)",
    )
    ax2.axhline(true_att, color=C_GOOD, linestyle="-.", lw=1.8, label=f"ATT Verdadeiro ({_fmt_brl_str(true_att, 0)})")
    ax2.set_title("Anatomia do Viés: Nível Pré + Tendência Diferencial")
    ax2.set_xlabel("Mês (Período)")
    ax2.set_ylabel("Diferença Mensal (R$)")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax2.set_xticks(range(1, 25, 2))
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Viés de Seleção Observável e Violação de Tendências Paralelas no Varejo Físico",
        fontsize=13.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_love_plot(
    smd_df: pd.DataFrame,
    df_store: pd.DataFrame,
    ps: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 2: Love Plot (SMD antes vs. depois) + Distribuição do Escore de Propensão (Common Support)."""
    plt.rcParams.update(PLOT_STYLE)
    W = df_store["tratamento"].to_numpy() == 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    y_pos = np.arange(len(smd_df))
    labels_pt = {
        "renda_per_capita": "Renda Per Capita",
        "penetracao_internet": "Penetração de Internet",
        "n_concorrentes": "Nº de Concorrentes",
        "potencial_crescimento": "Potencial de Crescimento",
    }
    ylabels = [labels_pt.get(v, v) for v in smd_df["Variavel"]]

    ax1.axvspan(-0.1, 0.1, color="#d1fae5", alpha=0.55, label="Zona de Balanceamento (|SMD| < 0,10)")
    ax1.axvline(0.0, color="#0f172a", lw=1.2)
    ax1.scatter(smd_df["SMD_Bruto"], y_pos + 0.16, color=C_BAD, s=95, marker="X", zorder=4, label="Bruto (Sem Balanceamento)")
    ax1.scatter(smd_df["SMD_PSM"], y_pos + 0.05, color=C_WARN, s=75, marker="^", zorder=4, label="PSM (3-NN)")
    ax1.scatter(smd_df["SMD_IPW"], y_pos - 0.06, color=C_TREAT, s=75, marker="s", zorder=4, label="IPW (Hájek)")
    ax1.scatter(smd_df["SMD_Entropy"], y_pos - 0.17, color=C_GOOD, s=95, marker="o", zorder=5, label="Entropy Balancing (SMD = 0,0000)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ylabels, fontweight="bold")
    ax1.set_xlabel("Standardized Mean Difference (SMD)")
    ax1.set_title("Love Plot: Balanceamento de Covariáveis por Método")
    ax1.legend(loc="lower right", fontsize=8.5)

    bins = np.linspace(0.15, 0.95, 17)
    ax2.hist(ps[W], bins=bins, alpha=0.55, color=C_TREAT, edgecolor="white", density=True, label=f"Tratadas (n = {W.sum()})")
    ax2.hist(ps[~W], bins=bins, alpha=0.55, color=C_CTRL, edgecolor="white", density=True, label=f"Controles (n = {(~W).sum()})")
    ax2.axvline(float(np.mean(ps[W])), color=C_TREAT, ls="--", lw=1.8)
    ax2.axvline(float(np.mean(ps[~W])), color=C_CTRL, ls="--", lw=1.8)
    ax2.set_title("Distribuição do Escore de Propensão P(W=1 | X)")
    ax2.set_xlabel("Probabilidade Estimada de Receber Campanha")
    ax2.set_ylabel("Densidade")
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("Diagnóstico de Balanceamento e Sobreposição (Common Support)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_event_study(
    es: pd.DataFrame,
    true_att: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 3: Event Study Dinâmico comparando TWFE Simples vs. ETWFE (Wooldridge)."""
    plt.rcParams.update(PLOT_STYLE)
    p_s_tr = float(es["p_trend_simple"].iloc[0])
    p_c_tr = float(es["p_trend_covper"].iloc[0])
    sl_s = float(es["slope_pre_simple"].iloc[0])
    sl_c = float(es["slope_pre_covper"].iloc[0])
    p_s_lbl = "< 0,001" if p_s_tr < 0.001 else f"= {p_s_tr:.3f}".replace(".", ",")
    p_c_lbl = f"= {p_c_tr:.3f}".replace(".", ",")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)
    for ax, col_c, col_se, title, color in [
        (
            ax1, "simple_coef", "simple_se",
            f"TWFE Simples — Pré-Tendência Violada (+{_fmt_brl_str(sl_s, 0)}/mês; p {p_s_lbl})",
            C_BAD,
        ),
        (
            ax2, "covper_coef", "covper_se",
            f"ETWFE (Cov×Período) — Tendências Paralelas ({_fmt_brl_str(sl_c, 0)}/mês; p {p_c_lbl})",
            C_GOOD,
        ),
    ]:
        ax.axvspan(-0.5, 11.5, color="#ecfdf5" if color == C_GOOD else "#fef2f2", alpha=0.45)
        ax.axhline(0.0, color="#334155", lw=1.2)
        ax.axhline(true_att, color="#0f172a", ls="-.", lw=1.6, label=f"ATT Verdadeiro ({_fmt_brl_str(true_att, 0)})")
        ax.axvline(-0.5, color="#1e40af", ls="--", lw=1.5, label="Início da Campanha (t = 13)")
        ax.errorbar(
            es["rel_t"], es[col_c], yerr=1.96 * es[col_se],
            fmt="o-", color=color, ecolor=color, elinewidth=1.4, capsize=3, ms=5, lw=2.0,
            label="Coeficiente Event Study ± IC 95%",
        )
        ax.set_title(title, fontsize=10.8)
        ax.set_xlabel("Tempo Relativo ao Início da Campanha (Meses; Ref. = −1)")
        ax.set_xticks(range(-12, 12, 2))
        ax.legend(loc="upper left", fontsize=8.5)

    ax1.set_ylabel("Efeito Estimado nas Vendas (R$/mês)")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    fig.suptitle(
        "Estudo de Evento (Event Study): Diagnóstico Formal de Tendências Paralelas",
        fontsize=13.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_synthetic_control(
    sc_df: pd.DataFrame,
    true_att: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 4: Controle Sintético (Abadie), Augmented SC (Ridge) e Synthetic DiD vs. Ground Truth."""
    plt.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.3), gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax1.axvspan(12.5, 24.5, color="#dbeafe", alpha=0.35)
    ax1.axvline(12.5, color="#1e40af", ls="--", lw=1.6)
    ax1.plot(sc_df["periodo"], sc_df["tratadas"], "o-", color="#0f172a", lw=2.6, ms=5, label="Tratadas (Observado)")
    ax1.plot(sc_df["periodo"], sc_df["sc_tradicional"], "s--", color=C_GOOD, lw=2.1, ms=4.5, label="SC Tradicional (Abadie)")
    ax1.plot(sc_df["periodo"], sc_df["ascm_ridge"], "^:", color=C_PURPLE, lw=2.0, ms=4.5, label="Augmented SC (Ridge)")
    ax1.plot(sc_df["periodo"], sc_df["sdid_alinhado"], "d-.", color=C_WARN, lw=1.9, ms=4.0, label="Synthetic DiD (Alinhado)")
    ax1.plot(sc_df["periodo"], sc_df["controles_bruto"], ":", color=C_CTRL, lw=1.6, alpha=0.85, label="Controles Bruto (Não Ponderado)")
    ax1.set_title("Construção do Contrafactual Sintético (Pré e Pós)")
    ax1.set_xlabel("Mês (Período)")
    ax1.set_ylabel("Vendas Médias por Loja")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax1.set_xticks(range(1, 25, 2))
    ax1.legend(loc="upper left", fontsize=8.3)

    gap_sc = sc_df["tratadas"] - sc_df["sc_tradicional"]
    gap_ascm = sc_df["tratadas"] - sc_df["ascm_ridge"]
    gap_sdid = sc_df["tratadas"] - sc_df["sdid_alinhado"]
    ax2.axvspan(12.5, 24.5, color="#ecfdf5", alpha=0.45)
    ax2.axvline(12.5, color="#1e40af", ls="--", lw=1.6)
    ax2.axhline(0.0, color="#334155", lw=1.1)
    ax2.axhline(true_att, color=C_BAD, ls="-.", lw=1.8, label=f"ATT Verdadeiro ({_fmt_brl_str(true_att, 0)})")
    ax2.plot(sc_df["periodo"], gap_sc, "o-", color=C_GOOD, lw=2.1, ms=4.8, label="Gap SC Tradicional")
    ax2.plot(sc_df["periodo"], gap_ascm, "^:", color=C_PURPLE, lw=2.0, ms=4.5, label="Gap Augmented SC (Ridge)")
    ax2.plot(sc_df["periodo"], gap_sdid, "d--", color=C_WARN, lw=1.9, ms=4.2, label="Gap Synthetic DiD")
    ax2.set_title("Efeito Causal Dinâmico (Tratadas − Contrafactual)")
    ax2.set_xlabel("Mês (Período)")
    ax2.set_ylabel("Efeito Estimado (R$)")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax2.set_xticks(range(1, 25, 2))
    ax2.legend(loc="upper left", fontsize=8.3)

    fig.suptitle(
        "Controle Sintético (Abadie), Augmented SC (Ridge) e Synthetic DiD vs. Ground Truth",
        fontsize=13.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_ranking_forest(
    ranking_df: pd.DataFrame,
    true_att: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 5: Forest Plot completo comparando todas as 24 especificações contra o Ground Truth."""
    plt.rcParams.update(PLOT_STYLE)
    rk = ranking_df.copy().iloc[::-1].reset_index(drop=True)
    n_specs = len(rk)
    fig, ax = plt.subplots(figsize=(13.6, 9.6))

    tier_colors = {
        "Tier S (≤ 2%)": C_GOOD,
        "Tier A (2–5%)": C_TREAT,
        "Tier B (5–10%)": C_WARN,
        "Tier C (10–20%)": "#ea580c",
        "Tier F (> 20%)": C_BAD,
    }
    ax.axvspan(true_att * 0.98, true_att * 1.02, color="#d1fae5", alpha=0.55, label="Banda de Precisão ±2% (Tier S)")
    ax.axvline(true_att, color="#0f172a", ls="--", lw=2.0, label=f"Ground Truth ({_fmt_brl_str(true_att, 2)})")

    x_right_cap = 19500.0
    y_idx = np.arange(n_specs)
    for i, row in rk.iterrows():
        col = tier_colors.get(row["Tier"], C_CTRL)
        att_v = float(row["ATT"])
        se_v = float(row["SE"]) if np.isfinite(row["SE"]) else 0.0
        lo = att_v - 1.96 * se_v
        hi = att_v + 1.96 * se_v
        clipped_hi = hi > x_right_cap

        draw_lo = max(0.0, lo)
        draw_hi = min(hi, x_right_cap)
        ax.plot([draw_lo, draw_hi], [i, i], color=col, lw=2.0, zorder=3)
        ax.plot([draw_lo, draw_lo], [i - 0.16, i + 0.16], color=col, lw=1.6, zorder=3)
        if clipped_hi:
            ax.plot(x_right_cap, i, marker=">", color=col, ms=6.5, zorder=4)
        else:
            ax.plot([draw_hi, draw_hi], [i - 0.16, i + 0.16], color=col, lw=1.6, zorder=3)
        ax.plot(att_v, i, marker="o", color=col, ms=7.2, zorder=5)

        err_str = f"{row['Erro_Pct']:+.1f}%".replace(".", ",")
        lbl = f"{_fmt_brl_str(att_v, 0)} ({err_str})"
        if clipped_hi:
            lbl += f"  [IC até {_fmt_brl_str(hi, 0)}]"
        ax.text(draw_hi + 260.0, i, lbl, va="center", fontsize=8.3, fontweight="bold", color=col)

    ax.set_yticks(y_idx)
    ax.set_yticklabels([f"#{int(r['Rank'])}  {r['Metodo']}" for _, r in rk.iterrows()], fontsize=9.1)
    ax.set_xlabel("Efeito Causal Estimado nas Vendas Mensais (ATT ± IC 95%)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax.set_xlim(0.0, 28200.0)
    ax.set_title(
        f"Benchmark Completo: {n_specs} Especificações de Inferência Causal vs. Ground Truth ({_fmt_brl_str(true_att, 2)})"
    )
    ax.legend(loc="upper right", fontsize=9.2)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_cate_and_leakage(
    cate_df: pd.DataFrame,
    audit: Dict[str, Any],
    true_att: float,
    save_path: Path | None = None,
) -> plt.Figure:
    """Figura 6: Heterogeneidade do CATE por quartil de renda + Auditoria de Panel Leakage no DoubleML."""
    plt.rcParams.update(PLOT_STYLE)
    cate = cate_df.copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    cate["quartil_renda"] = pd.qcut(cate["renda_per_capita"], 4, labels=["Q1 (Menor)", "Q2", "Q3", "Q4 (Maior)"])
    q_agg = cate.groupby("quartil_renda", observed=False).agg(
        verdadeiro=("efeito_verdadeiro", "mean"),
        t_learner=("tau_t_learner", "mean"),
        x_learner=("tau_x_learner", "mean"),
        dr_learner=("tau_dr_learner", "mean"),
    )
    qx = np.arange(len(q_agg))
    bw = 0.21
    ax1.bar(qx - 1.5 * bw, q_agg["verdadeiro"], width=bw, color="#0f172a", label="Efeito Verdadeiro (DGP)")
    ax1.bar(qx - 0.5 * bw, q_agg["t_learner"], width=bw, color=C_GOOD, label="T-Learner (Cross-Fitted)")
    ax1.bar(qx + 0.5 * bw, q_agg["x_learner"], width=bw, color=C_TREAT, label="X-Learner (Cross-Fitted)")
    ax1.bar(qx + 1.5 * bw, q_agg["dr_learner"], width=bw, color=C_PURPLE, label="DR-Learner (Cross-Fitted)")
    ax1.set_xticks(qx)
    ax1.set_xticklabels(q_agg.index, fontweight="bold")
    ax1.set_ylim(4500, 7400)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax1.set_xlabel("Quartil de Renda Per Capita (Lojas Tratadas)")
    ax1.set_ylabel("CATE Médio (R$/mês)")
    r_str = f"{audit['cate_corr_t_learner']:.2f}".replace(".", ",")
    ax1.set_title(
        f"Heterogeneidade (CATE): Estimado vs. Verdadeiro (r = {r_str} | PEHE = {_fmt_brl_str(audit['cate_pehe_t_learner'], 0)})"
    )
    ax1.legend(loc="upper left", fontsize=8.3)

    labels_lk = [
        "DML-PLR\n(K-Fold Linha)\n[LEAKAGE!]",
        "DML-PLR\n(GroupKFold\npor Loja)",
        "DML-PLR\n(Store-Level\nN = 250)",
        "DML-IRM\n(ATTE\nOrtogonal)",
    ]
    vals_lk = [
        audit["dml_plr_naive_row_kfold"],
        audit["dml_plr_group_kfold"],
        audit["dml_plr_store_level"],
        audit["dml_irm_store_level"],
    ]
    errs_lk = [
        audit["dml_plr_naive_err_pct"],
        audit["dml_plr_group_err_pct"],
        audit["dml_plr_store_err_pct"],
        audit["dml_irm_store_err_pct"],
    ]
    bar_cols = [C_BAD, C_GOOD, C_TREAT, C_PURPLE]
    bars = ax2.bar(np.arange(len(labels_lk)), vals_lk, color=bar_cols, width=0.58, edgecolor="white")
    ax2.axhline(true_att, color="#0f172a", ls="--", lw=1.8, label=f"Ground Truth ({_fmt_brl_str(true_att, 0)})")
    for b, v, ep in zip(bars, vals_lk, errs_lk):
        ep_str = f"{ep:+.1f}%".replace(".", ",")
        ax2.text(
            b.get_x() + b.get_width() / 2.0, v + 220,
            f"{_fmt_brl_str(v, 0)}\n({ep_str})",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )
    ax2.set_xticks(np.arange(len(labels_lk)))
    ax2.set_xticklabels(labels_lk, fontsize=8.6)
    ax2.set_ylim(0, max(vals_lk + [true_att]) * 1.24)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_brl))
    ax2.set_ylabel("ATT Estimado (R$/mês)")
    ax2.set_title("Auditoria DML: Efeito do Panel Cross-Fitting Leakage")
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Causal Machine Learning: Heterogeneidade de Efeitos e Rigor no Cross-Fitting em Painel",
        fontsize=13.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def generate_all_figures(bench: Dict[str, Any], img_dir: Path) -> Dict[str, Path]:
    """Gera e salva em disco todas as 6 figuras do benchmark."""
    img_dir.mkdir(parents=True, exist_ok=True)
    true_att = bench["true_att"]
    paths = {
        "fig1": img_dir / "01_trajetorias_vies_selecao.png",
        "fig2": img_dir / "02_love_plot_entropy_balancing.png",
        "fig3": img_dir / "03_event_study_parallel_trends.png",
        "fig4": img_dir / "04_synthetic_control_sdid.png",
        "fig5": img_dir / "05_ranking_completo_metodos.png",
        "fig6": img_dir / "06_cate_heterogeneidade_dml_leakage.png",
    }

    for fig in [
        plot_trajetorias_vies(bench["df_full"], true_att, save_path=paths["fig1"]),
        plot_love_plot(bench["smd_df"], bench["df_store"], bench["propensity_scores"], save_path=paths["fig2"]),
        plot_event_study(bench["event_study_df"], true_att, save_path=paths["fig3"]),
        plot_synthetic_control(bench["sc_series_df"], true_att, save_path=paths["fig4"]),
        plot_ranking_forest(bench["ranking_df"], true_att, save_path=paths["fig5"]),
        plot_cate_and_leakage(bench["cate_df"], bench["leakage_audit"], true_att, save_path=paths["fig6"]),
    ]:
        plt.close(fig)

    return paths


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    proj_root = Path(__file__).resolve().parent.parent
    bench = run_full_benchmark(proj_root / "data")
    paths = generate_all_figures(bench, proj_root / "img")
    for k, p in paths.items():
        print(f"[OK] {k}: {p.name} ({p.stat().st_size:,} bytes)")
