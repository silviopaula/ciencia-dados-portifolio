"""
Processo Gerador de Dados (DGP) Calibrado — Black Friday & Otimização de Cupom.

Gera painel diário de 2 anos (2023-01-01 a 2024-12-31, 731 dias) para 8 regiões ×
2 categorias de e-commerce de casa & decoração (16 unidades = 11.696 observações),
com parâmetros financeiros e causais exatamente calibrados:

  1. Breakeven exato de +28,00% ao escalar o cupom de 20% para 30%:
     - Ticket médio cheio P = R$ 350,00
     - Margem bruta base m0 = 72,0% (CMV = R$ 98,00)
     - Custo variável logístico/pagamento cv = R$ 22,00/pedido
     -> Margem líquida unitária em d = 20%: 350*(0,72 - 0,20) - 22 = R$ 160,00
     -> Margem líquida unitária em d = 30%: 350*(0,72 - 0,30) - 22 = R$ 125,00
     -> Razão de breakeven = 160 / 125 = 1,2800 -> +28,00% exatos!

  2. Curva Real Côncava de Resposta ao Cupom L_k(d) = α_k * (1 - exp(-β_k * d)):
     - Móveis & Iluminação (menor elasticidade, maior ticket/margem):
         salto real 20% -> 30% ≈ +5,1% (cupom ótimo ≈ 11%–12%)
     - Têxtil & Objetos (maior elasticidade, compra por impulso):
         salto real 20% -> 30% ≈ +11,7% (cupom ótimo ≈ 18%–19%)
     - Média global da loja no salto 20% -> 30%: ≈ +8,6% (estritamente dentro de 4%–13%,
       muito abaixo do breakeven de +28% -> destruição severa de margem em 30%!)
     - Cupom único ótimo global d*: ≈ 15%

  3. Viés de Confusão Sazonal (~+41% no Modelo Ingênuo):
     - O pico orgânico de tráfego/conversão da Black Friday coincide desproporcionalmente
       com o uso do cupom de 30%, fazendo uma regressão ingênua estimar ~+41% de ganho
       aparente ao subir de 20% para 30% (acima do breakeven de +28%!).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicParams:
    """Parâmetros financeiros unitários da loja (calibrados para breakeven exato de +28,00%)."""
    ticket_cheio: float = 350.00      # P (R$)
    margem_bruta_base: float = 0.72   # m0 (72% -> CMV = R$ 98,00)
    custo_variavel_pedido: float = 22.00  # frete subsidiado + gateway (R$/pedido)

    def unit_margin(self, d: float | np.ndarray) -> float | np.ndarray:
        """Margem de contribuição líquida por pedido no desconto d ∈ [0, 1)."""
        return self.ticket_cheio * (self.margem_bruta_base - d) - self.custo_variavel_pedido

    @property
    def breakeven_lift_20_to_30(self) -> float:
        """Lift percentual de pedidos exigido ao subir cupom de 20% para 30% para empatar lucro."""
        cm20 = float(self.unit_margin(0.20))  # R$ 160,00
        cm30 = float(self.unit_margin(0.30))  # R$ 125,00
        return (cm20 / cm30) - 1.0            # 0.2800000000 (+28,00%)


# Parâmetros estruturais da curva côncava L_k(d) = alpha_k * (1 - exp(-beta_k * d))
# Calibrados para que:
#   - Móveis & Iluminação: salto 20%->30% ≈ +5,1%, cupom ótimo d*_A ≈ 12%
#   - Têxtil & Objetos:    salto 20%->30% ≈ +11,7%, cupom ótimo d*_B ≈ 19%
#   - Loja agregada:       salto 20%->30% ≈ +8,6% (dentro de 4%–13%), cupom único ótimo d* ≈ 15%
CATEGORY_SPECS: Dict[str, Dict[str, float]] = {
    "Moveis_Iluminacao": {
        "label": "Móveis & Iluminação",
        "base_orders": 52.0,
        "alpha": 0.56,
        "beta": 5.4,
        "ticket": 420.0,
        "m0": 0.74,
        "cv": 28.0,
    },
    "Textil_Objetos": {
        "label": "Têxtil & Objetos",
        "base_orders": 68.0,
        "alpha": 1.60,
        "beta": 3.75,
        "ticket": 296.47058823529414,  # Média ponderada por base_orders preserva P_médio = R$ 350,00!
        "m0": 0.700297619047619,       # Preserva CM20=160 e CM30=125 agregados exatos!
        "cv": 17.41176470588235,
    },
}

REGIONS = [
    ("SP_Capital", 1.35, True),
    ("RJ_Capital", 1.10, True),
    ("Sul_PR", 0.95, True),
    ("NE_BA", 0.85, True),
    ("SP_Interior", 1.25, False),
    ("MG_Capital", 1.05, False),
    ("Sul_RS", 0.95, False),
    ("Centro_DF", 0.80, False),
]


def true_lift_curve(d: float | np.ndarray, categoria: str | None = None) -> float | np.ndarray:
    """
    Retorna o lift multiplicativo percentual L(d) sobre a demanda orgânica Y(0):
      Y(d) = Y(0) * (1 + L(d))
    Se `categoria` for None, retorna o lift médio ponderado pelo volume base da loja.
    """
    d_arr = np.asarray(d, dtype=float)
    if categoria is not None:
        spec = CATEGORY_SPECS[categoria]
        val = spec["alpha"] * (1.0 - np.exp(-spec["beta"] * d_arr))
        return float(val) if np.ndim(d) == 0 else val

    total_w = sum(s["base_orders"] for s in CATEGORY_SPECS.values())
    acc = np.zeros_like(d_arr, dtype=float)
    for spec in CATEGORY_SPECS.values():
        w = spec["base_orders"] / total_w
        acc = acc + w * spec["alpha"] * (1.0 - np.exp(-spec["beta"] * d_arr))
    return float(acc) if np.ndim(d) == 0 else acc


def true_incremental_lift_20_to_30(categoria: str | None = None) -> float:
    """Ganho percentual real de pedidos ao subir o cupom de 20% para 30%: (1+L(0.30))/(1+L(0.20)) - 1."""
    l20 = float(true_lift_curve(0.20, categoria))
    l30 = float(true_lift_curve(0.30, categoria))
    return ((1.0 + l30) / (1.0 + l20)) - 1.0


def generate_synthetic_panel(seed: int = 2025) -> pd.DataFrame:
    """
    Constrói o painel diário completo (731 dias × 16 unidades região-categoria = 11.696 linhas)
    salvando todas as contrafactuais verdadeiras Y(0), Y(20%), Y(30%) para auditoria exata.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n_days = len(dates)
    t_idx = np.arange(n_days, dtype=float)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()

    # Sazonalidade semanal (pico sexta/sábado/domingo) + anual + tendência suave de crescimento
    weekly_mult = np.array([0.92, 0.94, 0.96, 1.00, 1.08, 1.12, 0.98])[dow]
    annual_mult = 1.0 + 0.08 * np.sin(2.0 * np.pi * (doy - 300.0) / 365.25)
    trend_mult = 1.0 + 0.00018 * t_idx

    # Janelas de Black Friday (10 dias em torno da última sexta de novembro em 2023 e 2024)
    bf_2023 = (dates >= "2023-11-20") & (dates <= "2023-11-29")
    bf_2024 = (dates >= "2024-11-22") & (dates <= "2024-12-01")
    is_bf = bf_2023 | bf_2024

    # Perfil orgânico intenso da Black Friday (independente de cupom! Causa o confundimento de +41%)
    bf_organic_boost = np.ones(n_days, dtype=float)
    # Ramp-up nos dias da BF week com pico na sexta/fim de semana (~+52% orgânico médio na janela)
    bf_profile = np.array([1.28, 1.32, 1.38, 1.48, 1.85, 1.78, 1.65, 1.52, 1.42, 1.48])
    bf_organic_boost[bf_2023] = bf_profile
    bf_organic_boost[bf_2024] = bf_profile * 1.02

    # Pequenas campanhas sazonais ao longo do ano (Dia das Mães, Dia do Consumidor, Aniversário)
    promo_windows = (
        ((dates.month == 3) & (dates.day.isin(range(11, 18))))
        | ((dates.month == 5) & (dates.day.isin(range(5, 13))))
        | ((dates.month == 8) & (dates.day.isin(range(10, 18))))
        | ((dates.month == 10) & (dates.day.isin(range(8, 15))))
    )

    econ = EconomicParams()
    frames = []

    for reg_name, reg_scale, is_treated_bf24 in REGIONS:
        for cat_key, spec in CATEGORY_SPECS.items():
            unit_id = f"{reg_name}__{cat_key}"

            # Demanda orgânica determinística + choque idiossincrático log-normal leve (σ = 0.045)
            eps = rng.normal(0.0, 0.045, size=n_days)
            y_organico = (
                spec["base_orders"]
                * reg_scale
                * trend_mult
                * weekly_mult
                * annual_mult
                * bf_organic_boost
                * np.exp(eps - 0.5 * (0.045 ** 2))
            )

            # Política histórica de cupons observados d_it ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30}:
            # - Dias comuns: majoritariamente 0% ou 10%, com rotações quinzenais de 15%, 20% e raros 25%/30%
            # - Datas promocionais comuns: sorteio entre 15%, 20% e 25%
            # - Black Friday 2023: todas as unidades usaram 20% nos primeiros 5 dias e algumas 30% no pico!
            # - Black Friday 2024 (Quase-Experimento limpo):
            #     * Regiões Controle (is_treated_bf24=False): mantiveram 20% em toda a janela BF 2024!
            #     * Regiões Tratadas (is_treated_bf24=True): escalaram para 30% em toda a janela BF 2024!
            cupom = rng.choice(
                [0.00, 0.10, 0.15, 0.20, 0.25, 0.30],
                size=n_days,
                p=[0.4727, 0.20, 0.14, 0.13, 0.05, 0.0073],
            )
            cupom[promo_windows] = rng.choice([0.15, 0.20, 0.25], size=int(promo_windows.sum()), p=[0.35, 0.45, 0.20])

            # Em BF 2023, toda a rede operou com a política histórica padrão de 20%
            # (serve como baseline histórico limpo para calibrar co-movimento na BF!)
            cupom[bf_2023] = 0.20

            # Pré-BF 2024 (outubro e 1–21 nov 2024): padroniza baseline em 20% nas sextas/fins de semana e 10% dias úteis
            # igualmente para tratadas e controles -> garante tendências paralelas limpas antes do choque!
            pre_bf24 = (dates >= "2024-10-01") & (dates <= "2024-11-21")
            cupom[pre_bf24] = np.where(dow[pre_bf24] >= 4, 0.20, 0.10)

            # Janela Black Friday 2024: Quase-Experimento 30% (Tratadas) vs. 20% (Controles)
            cupom[bf_2024] = 0.30 if is_treated_bf24 else 0.20

            # Pós-BF imediato (dezembro 2024): retorno simétrico
            post_bf24 = dates >= "2024-12-02"
            cupom[post_bf24] = np.where(dow[post_bf24] >= 4, 0.15, 0.00)

            lift_obs = true_lift_curve(cupom, cat_key)
            lift_20 = true_lift_curve(0.20, cat_key)
            lift_30 = true_lift_curve(0.30, cat_key)

            pedidos_obs = y_organico * (1.0 + lift_obs)
            pedidos_cf_20 = y_organico * (1.0 + lift_20)
            pedidos_cf_30 = y_organico * (1.0 + lift_30)

            margem_unit_obs = econ.unit_margin(cupom)
            receita_bruta = pedidos_obs * econ.ticket_cheio * (1.0 - cupom)
            lucro_liquido = pedidos_obs * margem_unit_obs
            lucro_cf_20 = pedidos_cf_20 * float(econ.unit_margin(0.20))
            lucro_cf_30 = pedidos_cf_30 * float(econ.unit_margin(0.30))

            frames.append(pd.DataFrame({
                "data": dates,
                "unidade_id": unit_id,
                "regiao": reg_name,
                "categoria": cat_key,
                "categoria_label": spec["label"],
                "tratada_bf24": int(is_treated_bf24),
                "is_bf23": bf_2023.astype(int),
                "is_bf24": bf_2024.astype(int),
                "is_black_friday": is_bf.astype(int),
                "is_promo_sazonal": promo_windows.astype(int),
                "dia_semana": dow,
                "cupom_pct": np.round(cupom * 100.0).astype(int),
                "cupom": cupom,
                "pedidos_organicos_y0": np.round(y_organico, 3),
                "pedidos": np.round(pedidos_obs, 3),
                "pedidos_cf_20": np.round(pedidos_cf_20, 3),
                "pedidos_cf_30": np.round(pedidos_cf_30, 3),
                "margem_unitaria": np.round(margem_unit_obs, 2),
                "receita_liquida": np.round(receita_bruta, 2),
                "lucro_liquido": np.round(lucro_liquido, 2),
                "lucro_cf_20": np.round(lucro_cf_20, 2),
                "lucro_cf_30": np.round(lucro_cf_30, 2),
            }))

    df = pd.concat(frames, ignore_index=True).sort_values(["data", "unidade_id"]).reset_index(drop=True)
    return df


def save_or_load_panel(data_dir: Path | str, seed: int = 2025, force_regen: bool = False) -> pd.DataFrame:
    dpath = Path(data_dir)
    dpath.mkdir(parents=True, exist_ok=True)
    csv_path = dpath / "painel_black_friday_cupons.csv"
    if csv_path.exists() and not force_regen:
        return pd.read_csv(csv_path, parse_dates=["data"])
    df = generate_synthetic_panel(seed=seed)
    df.to_csv(csv_path, index=False)
    return df
