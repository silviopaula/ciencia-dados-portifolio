"""Exportação multi-abas em Excel (.xlsx) formatado para auditoria executiva."""

from __future__ import annotations

import io
from typing import Any, Dict

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def _style_worksheet(ws, header_hex: str = "1E3A8A") -> None:
    header_fill = PatternFill(start_color=header_hex, end_color=header_hex, fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val = "" if cell.value is None else str(cell.value)
            if len(val) > max_len:
                max_len = len(val)
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 46)


def export_analysis_to_excel(analysis: Dict[str, Any]) -> bytes:
    """Gera workbook Excel (.xlsx) com 5 abas formatadas a partir do payload de análise."""
    buf = io.BytesIO()

    meta = analysis["meta"]
    cons = analysis["stationarity_consensus"]
    desc = analysis["descriptive_stats"]
    stl_s = analysis["stl"]
    model_info = analysis["arima"]

    summary_rows = [
        {"Métrica": "Dataset / Série", "Valor": meta["dataset_title"]},
        {"Métrica": "Transformação Aplicada", "Valor": meta["transform_label"]},
        {"Métrica": "Observações (N original / N analisado)", "Valor": f"{meta['n_original']} / {meta['n_analyzed']}"},
        {"Métrica": "Período Analisado", "Valor": f"{meta['date_start']} a {meta['date_end']}"},
        {"Métrica": "Frequência Sazonal (s)", "Valor": meta["seasonal_period"]},
        {"Métrica": "Nível de Significância (α)", "Valor": meta["alpha"]},
        {"Métrica": "Veredito Cruzado (ADF × PP × KPSS × ZA)", "Valor": cons["regime_title"]},
        {"Métrica": "Recomendação Econométrica", "Valor": cons["recommendation"]},
        {"Métrica": "Média / Desvio-Padrão", "Valor": f"{desc['mean']:.4f} / {desc['std']:.4f}"},
        {"Métrica": "Assimetria / Curtose (Excesso)", "Valor": f"{desc['skewness']:.4f} / {desc['kurtosis']:.4f}"},
        {"Métrica": "Força da Tendência STL (F_T)", "Valor": f"{stl_s['trend_strength'] * 100:.1f}%"},
        {"Métrica": "Força da Sazonalidade STL (F_S)", "Valor": f"{stl_s['seasonal_strength'] * 100:.1f}%"},
        {"Métrica": "Modelo SARIMA Ajustado", "Valor": model_info["spec_label"]},
        {"Métrica": "AIC / BIC", "Valor": f"{model_info['aic']:.2f} / {model_info['bic']:.2f}"},
        {"Métrica": "Backtest RMSE (SARIMA vs Naive)", "Valor": f"{model_info['backtest']['rmse']} vs {model_info['backtest']['naive_rmse']}"},
        {"Métrica": "Outliers Detectados (IQR / MAD)", "Valor": f"{analysis['outliers']['iqr_count']} / {analysis['outliers']['mad_count']}"},
    ]
    df_summary = pd.DataFrame(summary_rows)

    df_tests = pd.DataFrame([
        {
            "Categoria": t["category"],
            "Teste": t["name"],
            "Alvo": t["target_series"],
            "Estatística": t["statistic"],
            "p-valor": t["p_value"],
            "Valor Crítico (5%)": t.get("critical_5pct"),
            "Hipótese Nula (H0)": t["h0"],
            "Status": t["status_label"],
            "Interpretação": t["interpretation"],
        }
        for t in analysis["tests_summary"]
    ])

    df_transforms = pd.DataFrame([
        {
            "Transformação": r["label"],
            "N": r["n_obs"],
            "ADF p-valor": r["adf_p"],
            "PP p-valor": r["pp_p"],
            "KPSS p-valor": r["kpss_p"],
            "ARCH-LM p-valor": r["arch_p"],
            "Consenso": r["consensus"],
            "Recomendada": "Sim" if r["recommended"] else "Não",
        }
        for r in analysis["transform_comparison"]
    ])

    df_series = pd.DataFrame(analysis["series_table"])
    df_forecast = pd.DataFrame(analysis["arima"]["forecast"])

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Resumo_Executivo", index=False)
        df_tests.to_excel(writer, sheet_name="Testes_Estatisticos", index=False)
        df_transforms.to_excel(writer, sheet_name="Matriz_Transformacoes", index=False)
        df_series.to_excel(writer, sheet_name="Serie_e_STL", index=False)
        df_forecast.to_excel(writer, sheet_name="Previsao_SARIMA", index=False)

        for name, color in [
            ("Resumo_Executivo", "1E3A8A"),
            ("Testes_Estatisticos", "0F766E"),
            ("Matriz_Transformacoes", "4338CA"),
            ("Serie_e_STL", "1D4ED8"),
            ("Previsao_SARIMA", "7C3AED"),
        ]:
            _style_worksheet(writer.sheets[name], header_hex=color)

    return buf.getvalue()
