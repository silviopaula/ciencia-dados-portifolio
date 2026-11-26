"""Etapa 2 (Referência Econométrica) — Estimação CausalImpact (BSTS) sobre o Painel Gold.

Consome o painel anual gerado pela Camada Gold do Databricks (`data_causalimpact.xlsx` ou
`workspace.avaliacao_lei_tse.painel_causal_impact`) para estimar o efeito contrafactual da
Lei nº 12.034/2009 (Minirreforma Eleitoral) sobre a proporção de candidaturas femininas em
eleições proporcionais.

Desenho de Identificação:
- Série Tratada (Y): `prop_candidatas_tratado` (vereadora, dep. federal, estadual e distrital — sujeitas à cota de 30%).
- Controles Sintéticos (X): cargos majoritários (`prop_cand_governador`, `prop_cand_prefeito`, `prop_cand_senador`)
  NÃO abrangidos pela regra da cota + controles demográficos/IDHM (`prop_pop_fem`, `idhm`).
- Período Pré-Intervenção : 1998 a 2009 (antes da vigência da obrigatoriedade de "preencher" 30%).
- Período Pós-Intervenção : 2010 a 2022 (vigência efetiva da Lei nº 12.034/2009).

Observação:
    Este script documenta a etapa econométrica executada em ambiente externo (Google Colab /
    Python com `pycausalimpact` ou `statsmodels` BSTS), complementar ao pipeline PySpark no
    Databricks e ao artigo científico principal baseado em Diferenças em Diferenças
    (Callaway & Sant'Anna, 2021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "data_causalimpact.xlsx"

TARGET_COL = "prop_candidatas_tratado"
DEFAULT_COVARIATES: List[str] = [
    "prop_cand_governador",
    "prop_cand_prefeito",
    "prop_cand_senador",
    "prop_pop_fem",
    "idhm",
]


def load_gold_causal_panel(
    excel_path: Path = DEFAULT_EXCEL_PATH,
    covariates: List[str] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """Carrega e ordena o painel Gold anual (`1998–2022`) com a variável tratada na primeira coluna."""
    df = pd.read_excel(excel_path)
    df["ano"] = df["ano"].astype(int)
    df = df.sort_values("ano").set_index("ano")

    cols = [TARGET_COL] + [c for c in covariates if c in df.columns]
    return df[cols].astype(float)


def descriptive_pre_post_summary(df: pd.DataFrame, split_year: int = 2009) -> Dict[str, float]:
    """Calcula estatísticas descritivas simples Pré-Lei (≤2009) vs. Pós-Lei (>2009)."""
    pre = df.loc[df.index <= split_year, TARGET_COL]
    post = df.loc[df.index > split_year, TARGET_COL]
    diff_pp = (post.mean() - pre.mean()) * 100.0
    rel_pct = ((post.mean() / pre.mean()) - 1.0) * 100.0
    return {
        "pre_mean": round(float(pre.mean()), 4),
        "post_mean": round(float(post.mean()), 4),
        "post_max": round(float(post.max()), 4),
        "diff_pp": round(float(diff_pp), 2),
        "rel_gain_pct": round(float(rel_pct), 1),
    }


def run_pycausalimpact(
    df: pd.DataFrame,
    pre_period: Tuple[int, int] = (1998, 2009),
    post_period: Tuple[int, int] = (2010, 2022),
):
    """Executa inferência causal bayesiana (Brodersen et al., 2015) via `pycausalimpact`."""
    from causalimpact import CausalImpact  # Import tardio (dependência opcional)

    ci = CausalImpact(df, list(pre_period), list(post_period))
    print(ci.summary())
    print("\n--- Relatório Narrativo ---")
    print(ci.summary(output="report"))
    return ci


def run_placebo_in_time_check(
    df: pd.DataFrame,
    placebo_pre: Tuple[int, int] = (1998, 2004),
    placebo_post: Tuple[int, int] = (2005, 2009),
):
    """Teste Placebo no Tempo (pré-lei): verifica ausência de efeito espúrio antes da Lei 12.034/2009."""
    pre_only = df.loc[df.index <= 2009].copy()
    return run_pycausalimpact(pre_only, pre_period=placebo_pre, post_period=placebo_post)


if __name__ == "__main__":
    panel = load_gold_causal_panel()
    print("Resumo descritivo Pré vs. Pós-Lei nº 12.034/2009:", descriptive_pre_post_summary(panel))
    # Descomente abaixo em ambiente com `pycausalimpact` instalado:
    # ci_model = run_pycausalimpact(panel)
    # ci_model.plot()
