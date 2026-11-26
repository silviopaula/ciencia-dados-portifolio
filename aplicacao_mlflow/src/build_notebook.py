"""Gera o notebook executável `mlops_churn_mlflow.ipynb` conectado ao MLflow 3.x local."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "metrics_summary.json"
NB_PATH = PROJECT_ROOT / "mlops_churn_mlflow.ipynb"


def md_cell(text: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in text.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(code: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook() -> None:
    s = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ds = s["dataset"]
    lk = s["leakage_audit"]
    gov = s["registry_governance"]
    bm = s["models_benchmark"]
    roi = s["roi_optimization"]
    opt_oot = roi["oot_at_opt_threshold"]
    def_oot = roi["oot_at_default_050"]
    opt_m = roi["oot_metrics_at_opt"]
    dr = s["drift_monitoring"]
    top_psi0 = dr["top_12_psi"][0]
    top_psi1 = dr["top_12_psi"][1]
    top_imp = s["top_15_permutation_importance"]

    champ_row = next(r for r in bm if r["model"] == gov["champion_model"])
    base_row = next(r for r in bm if r["model"] == gov["baseline_model"])
    chal_row = next(r for r in bm if r["model"] == gov["challenger_model"])
    sl_map = {r["Fatia"]: r for r in s["slice_error_analysis"]}

    cells: List[Dict[str, Any]] = [
        md_cell(f"""# 🚀 MLOps End-to-End com MLflow {s['mlflow_version']} — Previsão de Churn Out-of-Time

**Autor:** Dr. Silvio da Rosa Paula • Portfólio de Machine Learning & MLOps  
**Plataforma de Governança:** **MLflow `{s['mlflow_version']}`** (SQLite Tracking Store + Model Registry + Aliases `@baseline` / `@challenger` / `@champion`)  
**Dataset:** **`{ds['total_rows']:,}` registros comportamentais (`{ds['unique_clients']:,}` clientes únicos, `92` features em 4 janelas)** de **Abr/2024 a Nov/2024** (`data/abt.csv`)

---

## 🎯 Contexto de Negócio & Arquitetura MLOps de Produção

Treinar um modelo em notebook é apenas o início do ciclo de vida de Machine Learning. Em produção, a perda silenciosa de desempenho decorre de quatro falhas clássicas — todas auditadas e resolvidas neste pipeline:

1. **Vazamento Temporal e de Entidade (*Leakage*):** substituímos o `train_test_split` aleatório do projeto legado por um **Split Cronológico Out-of-Time (OOT)** estrito (**Treino `Abr–Jul/2024`**, **Validação `Ago–Set/2024`** e **Teste Cego OOT `Out–Nov/2024`**), reduzindo sobreposição indevida de safras futuras (`{lk['random_client_overlap']}` → `{lk['oot_client_overlap']}` clientes entre janelas).
2. **Zero *Training-Serving Skew*:** encapsulamos o transformador customizado `ChurnFeatureEngineer` (indicadores de inatividade `d7/d14`, razões comportamentais e medianas congeladas no treino) dentro do `Pipeline` salvo no **MLflow Model Registry** (`serve_roundtrip_max_diff = {gov['serve_roundtrip_max_diff']:.2e}`).
3. **Governança Automatizada no MLflow Model Registry (`{s['registered_model_name']}`):** rastreamos **9 pipelines** com *Signature* (`infer_signature`), *Input Example*, IC 95% Bootstrap e Brier Score, promovendo `@champion` via **Quality Gate automatizado** (`{'APROVADO' if gov['promotion_approved'] else 'REVISAR'}`).
4. **Monitoramento de Drift (PSI) & Valor Financeiro (ROI):** medimos o **Population Stability Index (PSI)** entre Treino e Safras OOT (`{dr['n_features_crit_025']}` features com `PSI > 0,25`, lideradas pelo envelhecimento natural de `tenure`) e otimizamos o *threshold* de campanha de retenção maximizando o **Lucro Líquido** (**`R\\$ {opt_oot['net_profit']:,.2f}`** no OOT, **`+R\\$ {opt_oot['gain_vs_treat_all']:,.2f}`** sobre abordar toda a base)."""),

        # =====================================================================
        # SEÇÃO 1 — AUDITORIA DA ABT, COORTES & MISSINGNESS
        # =====================================================================
        md_cell("## 1. Auditoria da ABT, Coortes Mensais, Missingness Estrutural & Vazamento"),
        code_cell("""import json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from IPython.display import Image, display

from churn_mlops import DEFAULT_EXPERIMENT, DEFAULT_MODEL_NAME, get_tracking_uri

FIG_DIR = ROOT / "outputs" / "figures"
metrics = json.loads((ROOT / "outputs" / "metrics_summary.json").read_text(encoding="utf-8"))

# Conecta ao MLflow Tracking Store SQLite local real!
mlflow.set_tracking_uri(get_tracking_uri())
client = MlflowClient()
exp = client.get_experiment_by_name(DEFAULT_EXPERIMENT)
print(f"MLflow {mlflow.__version__} ativo | Experimento: '{exp.name}' (ID={exp.experiment_id})")
print(f"Tracking URI: {metrics['tracking_uri']} | Total runs rastreadas: {len(client.search_runs([exp.experiment_id]))}")

cohorts_df = pd.DataFrame(metrics["dataset"]["cohorts"])
display(cohorts_df.rename(columns={
    "ym": "Safra (YYYY-MM)", "n": "Registros", "unique_clients": "Clientes Únicos",
    "churn_rate": "Taxa Churn", "pct_inativo_d7": "% Inativo d7", "pct_inativo_d14": "% Inativo d14",
}))
display(Image(filename=str(FIG_DIR / "01_eda_coortes_missingness_leakage.png")))"""),

        md_cell(f"""### 🔎 Achados da Auditoria Temporal e de Dados Ausentes

- **Drift Real nas Safras Futuras:** Entre o Treino (`Abr–Jul/2024`, churn médio `{ds['train_churn_rate']*100:.1f}%`) e o Teste OOT (`Out–Nov/2024`, churn médio **`{ds['oot_churn_rate']*100:.1f}%`**), observa-se uma elevação expressiva de **`+{(ds['oot_churn_rate']-ds['train_churn_rate'])*100:.1f} p.p.`** na taxa de churn acompanhada pelo salto da inatividade recente (`inativo_d7` de `~38%` para `~48%–51%`).
- **Missingness Estrutural Não-Aleatório:** `40,2%` das linhas possuem `NaN` nas métricas de 7 dias (`d7`) e `25,0%` nas de 14 dias (`d14`) — não por falha de coleta, mas porque o cliente **não realizou nenhuma interação naquela janela**. O transformador `ChurnFeatureEngineer` converte essa ausência em indicadores explícitos (`inativo_d7`, `inativo_d14`, `sem_recorrencia_d7`), expandindo de `92` para **`{ds['n_engineered_features']}` features** sem nenhum vazamento.
- **Split Aleatório vs. Split OOT:** No split aleatório do projeto legado, **`{lk['random_client_overlap']}` clientes** apareciam simultaneamente em treino e teste com safras futuras misturadas, distorcendo o diagnóstico real de produção."""),

        # =====================================================================
        # SEÇÃO 2 — TRACKING DE 9 PIPELINES NO MLFLOW 3.x
        # =====================================================================
        md_cell("## 2. Rastreamento de Experimentos no MLflow 3.x (Benchmark OOT de 9 Pipelines)"),
        code_cell("""bm_df = pd.DataFrame(metrics["models_benchmark"])
display(
    bm_df[[
        "model", "family", "val_roc_auc", "oot_roc_auc", "oot_roc_ci_low", "oot_roc_ci_high",
        "oot_pr_auc", "oot_f1", "oot_brier", "train_sec"
    ]].rename(columns={
        "model": "Pipeline Rastreado (MLflow)", "family": "Família",
        "val_roc_auc": "Val AUC", "oot_roc_auc": "OOT ROC-AUC",
        "oot_roc_ci_low": "IC95 Inf", "oot_roc_ci_high": "IC95 Sup",
        "oot_pr_auc": "OOT PR-AUC", "oot_f1": "OOT F1 (0,5)",
        "oot_brier": "OOT Brier", "train_sec": "Treino (s)",
    }).style.highlight_max(subset=["Val AUC", "OOT ROC-AUC", "OOT PR-AUC"], color="#d1fae5")
           .highlight_min(subset=["OOT Brier"], color="#dbeafe")
)
display(Image(filename=str(FIG_DIR / "02_mlflow_benchmark_oot.png")))"""),

        md_cell(f"""### 🔎 Comparação Científica (Validação Estrita vs. Teste Cego OOT `N = {ds['oot_test_n']}`)

- **Seleção Livre de Data Leakage:** O candidato a produção é escolhido **exclusivamente na Validação (`Ago–Set/2024`)**, onde **`{gov['champion_model']}`** liderou com **`Val ROC-AUC = {champ_row['val_roc_auc']:.4f}`** e menor **`Val Brier = {champ_row['val_brier']:.4f}`** (`+{(champ_row['val_roc_auc']-base_row['val_roc_auc'])*100:.2f} p.p.` sobre a árvore legada `{base_row['val_roc_auc']:.4f}`).
- **Generalização no Teste Cego OOT (`Out–Nov/2024`):** O `@champion` atingiu **`OOT ROC-AUC = {champ_row['oot_roc_auc']:.4f}`** (`IC 95% Bootstrap: [{champ_row['oot_roc_ci_low']:.4f}, {champ_row['oot_roc_ci_high']:.4f}]`), **maior `OOT PR-AUC = {champ_row['oot_pr_auc']:.4f}`** entre todos os 9 modelos e **`OOT Brier = {champ_row['oot_brier']:.4f}`**, empatado dentro do erro amostral com `@challenger` (`{gov['challenger_model']}`, `OOT AUC = {chal_row['oot_roc_auc']:.4f}`) e `GradientBoosting` (`0,8346`).
- **Por que o Platt Scaling (`9. Calibrated`) não substituiu o Ensemble?** Na Validação, o `Soft-Voting Ensemble` já apresentou Brier menor (`0,1784` vs. `0,1789`), e no Teste OOT a confirmação se manteve (`0,1707` vs. `0,1751`) porque a mudança na taxa base de churn (`42,6% → 51,6%`) penaliza ajustes paramétricos rígidos na calibração."""),

        # =====================================================================
        # SEÇÃO 3 — MODEL REGISTRY, ALIASES & QUALITY GATE
        # =====================================================================
        md_cell("## 3. Governança no MLflow Model Registry, Aliases & Teste Real de Inferência"),
        code_cell("""# Consulta ao vivo no MLflow Model Registry local (`churn-predictor`)
reg_versions = client.search_model_versions(f"name='{DEFAULT_MODEL_NAME}'")
print(f"Modelo Registrado: '{DEFAULT_MODEL_NAME}' | Versões no Registry: {len(reg_versions)}")
display(pd.DataFrame(metrics["registry_governance"]["versions"]))

print("\\nQuality Gate Automático de Promoção:")
for chk, ok in metrics["registry_governance"]["quality_gate_checks"].items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {chk}")
print(f"Status Final de Promoção: {'APROVADO' if metrics['registry_governance']['promotion_approved'] else 'REVISAR'}")

# Carrega o modelo diretamente pelo alias `@champion` (sem acoplar versão no código!)
prod_model = mlflow.sklearn.load_model(f"models:/{DEFAULT_MODEL_NAME}@champion")
raw_abt = pd.read_csv(ROOT / "data" / "abt.csv")
feat_cols = [c for c in raw_abt.columns if c not in ("dt_ref", "id_cliente", "flag_churn")]
sample_raw = raw_abt.tail(5)[feat_cols]
sample_probs = prod_model.predict_proba(sample_raw)[:, 1]
print("\\nSmoke-test de Inferência via `models:/churn-predictor@champion` (dados brutos com NaNs):")
for idx, p in enumerate(sample_probs):
    print(f"  Cliente amostra #{idx+1} -> Prob. Churn = {p*100:5.1f}%")"""),

        md_cell(f"""### 🔎 Governança & Zero *Training-Serving Skew*

- **Aliases Desacoplados em Produção:** Em vez de fixar `models:/churn-predictor/3` no código de produção, o serviço de inferência (`src/serve_predict.py`) consome sempre **`models:/churn-predictor@champion`**. Trocar ou fazer *rollback* para `@challenger` (`v2`) ou `@baseline` (`v1`) exige apenas mover o alias no Registry, sem redeploy de código.
- **Zero Skew Comprovado:** Como o `ChurnFeatureEngineer` congela as medianas aprendidas no treino dentro do artefato `cloudpickle` do MLflow, pontuar um lote de 10 clientes ou a base inteira produz exatamente as mesmas probabilidades (**diferença máxima = `{gov['serve_roundtrip_max_diff']:.2e}`**)."""),

        # =====================================================================
        # SEÇÃO 4 — CALIBRAÇÃO & OTIMIZAÇÃO DE ROI DE RETENÇÃO
        # =====================================================================
        md_cell("## 4. Diagnóstico Probabilístico, Calibração & Otimização Financeira (ROI)"),
        code_cell("""roi = metrics["roi_optimization"]
print(f"Threshold Ótimo de Lucro (Validação): τ = {roi['val_opt_threshold']:.2f} (breakeven teórico = {roi['theoretical_breakeven_prob']:.2f})")
roi_comp = pd.DataFrame([
    {"Estratégia no Teste OOT (N=543)": "1. Contatar Todos (Sem Modelo)", "Contatados": 543, "Churns Alcançados": 280, "Custo (R$)": 543*45, "Receita Salva (R$)": 280*0.30*450, "Lucro Líquido (R$)": metrics["roi_optimization"]["oot_at_opt_threshold"]["treat_all_profit"]},
    {"Estratégia no Teste OOT (N=543)": "2. @champion (Threshold Padrão τ=0,50)", "Contatados": metrics["roi_optimization"]["oot_at_default_050"]["contacted"], "Churns Alcançados": metrics["roi_optimization"]["oot_at_default_050"]["tp_saved_targets"], "Custo (R$)": metrics["roi_optimization"]["oot_at_default_050"]["campaign_cost"], "Receita Salva (R$)": metrics["roi_optimization"]["oot_at_default_050"]["expected_revenue_saved"], "Lucro Líquido (R$)": metrics["roi_optimization"]["oot_at_default_050"]["net_profit"]},
    {"Estratégia no Teste OOT (N=543)": f"3. @champion (Threshold ROI Val τ={metrics['roi_optimization']['val_opt_threshold']:.2f})", "Contatados": metrics["roi_optimization"]["oot_at_opt_threshold"]["contacted"], "Churns Alcançados": metrics["roi_optimization"]["oot_at_opt_threshold"]["tp_saved_targets"], "Custo (R$)": metrics["roi_optimization"]["oot_at_opt_threshold"]["campaign_cost"], "Receita Salva (R$)": metrics["roi_optimization"]["oot_at_opt_threshold"]["expected_revenue_saved"], "Lucro Líquido (R$)": metrics["roi_optimization"]["oot_at_opt_threshold"]["net_profit"]},
])
display(roi_comp)
display(Image(filename=str(FIG_DIR / "03_diagnostico_calibracao_roi.png")))"""),

        md_cell(f"""### 🔎 Impacto Financeiro da Otimização de Threshold

- Com **LTV = R\\$ 450**, **custo de contato = R\\$ 45** e **taxa de retenção efetiva = 30%**, o ponto de equilíbrio teórico é `p* = 45 / (0,30 × 450) = 0,333` — muito abaixo do corte ingênuo `0,50`!
- Aplicando no Teste OOT o threshold ótimo ajustado na Validação (**`τ = {roi['val_opt_threshold']:.2f}`**), o `@champion` captura **`{opt_m['recall']*100:.1f}%` de todos os churns** (`F1 = {opt_m['f1']:.4f}`, Acurácia `= {opt_m['accuracy']*100:.1f}%`), gerando **Lucro Líquido de R\\$ `{opt_oot['net_profit']:,.2f}`** (**`+R\\$ {opt_oot['gain_vs_treat_all']:,.2f}`** acima de abordar todos os clientes e **`+R\\$ {opt_oot['net_profit'] - def_oot['net_profit']:,.2f}`** acima do corte padrão `τ = 0,50`!)."""),

        # =====================================================================
        # SEÇÃO 5 — EXPLICABILIDADE & DRIFT PSI EM PRODUÇÃO
        # =====================================================================
        md_cell("## 5. Explicabilidade (Permutation Importance) & Monitoramento de Data Drift (PSI)"),
        code_cell("""display(Image(filename=str(FIG_DIR / "04_permutation_importance_explainability.png")))
display(pd.DataFrame(metrics["drift_monitoring"]["top_12_psi"]).head(8))
display(Image(filename=str(FIG_DIR / "05_drift_psi_monitoring.png")))"""),

        md_cell(f"""### 🔎 Drivers de Churn & Alerta Real de Data Drift em Produção

- **Principais Drivers Preditivos (*Permutation Importance*):** **`{top_imp[0]['feature']}`** (`ΔAUC = +{top_imp[0]['imp_mean']:.4f}`), **`{top_imp[1]['feature']}`** (`+{top_imp[1]['imp_mean']:.4f}`), **`{top_imp[2]['feature']}`** (`+{top_imp[2]['imp_mean']:.4f}`) e **`{top_imp[3]['feature']}`** (`+{top_imp[3]['imp_mean']:.4f}`). Clientes sem interação há mais de 21 dias atingem **`78,1%` de churn**, contra apenas **`23,2%`** para quem interagiu há 3–7 dias!
- **Diagnóstico Real de Data Drift (PSI):** O monitoramento detectou **`{dr['n_features_crit_025']}` features com `PSI > 0,25`**, lideradas por **`{top_psi0['feature']}` (`PSI = {top_psi0['psi']:.3f}`)** e **`{top_psi1['feature']}` (`PSI = {top_psi1['psi']:.3f}`)**. Esse deslocamento reflete o amadurecimento natural da base de clientes ao longo de 2024 e sinaliza a necessidade de **retreino periódico em janela deslizante** (ex.: mensal) e monitoramento contínuo de calibração."""),

        # =====================================================================
        # SEÇÃO 6 — SLICE-BASED ERROR ANALYSIS
        # =====================================================================
        md_cell("## 6. Slice-Based Error Analysis & Conclusão Executiva"),
        code_cell("""display(pd.DataFrame(metrics["slice_error_analysis"]))
display(Image(filename=str(FIG_DIR / "06_slice_error_analysis_registry.png")))"""),

        md_cell(f"""### 🔎 Onde o Modelo Brilha e Onde Evoluir (Próxima Iteração MLOps)

1. **Estabilidade Temporal Comprovada:** Nas duas safras futuras cegas (`2024-10`: `ROC-AUC = {sl_map['2024-10']['ROC-AUC']:.3f}`, Acurácia `{sl_map['2024-10']['Acurácia']*100:.1f}%`; `2024-11`: `ROC-AUC = {sl_map['2024-11']['ROC-AUC']:.3f}`, Acurácia `{sl_map['2024-11']['Acurácia']*100:.1f}%`), o desempenho permanece sólido mesmo sob drift de taxa base.
2. **Análise por Segmento de Cliente:** O modelo é excepcional em **Veteranos (`>120d`: `ROC-AUC = {sl_map['Veterano (>120d)']['ROC-AUC']:.3f}`, `F1 = {sl_map['Veterano (>120d)']['F1 (τ ROI)']:.3f}`)** e em clientes **Ativos nos últimos 7 dias (`ROC-AUC = {sl_map['Ativo últimos 7d']['ROC-AUC']:.3f}`)** — justamente o público mais acionável para retenção preventiva! Entre clientes já inativos há mais de 7 dias (`ROC-AUC = {sl_map['Inativo últimos 7d']['ROC-AUC']:.3f}`), o churn já é dominante, exigindo ações de reativação específicas.
3. **Recomendação de Produção:** Promover **`models:/churn-predictor@champion`** (`{gov['champion_model']}`), operar a campanha de retenção com **`τ = {roi['val_opt_threshold']:.2f}`** e agendar gatilho automático de retreino + recalibração mensal sempre que `PSI > 0,25` em features de engajamento recente ou a taxa de churn variar `> 5 p.p.`."""),
    ]

    for i, c in enumerate(cells):
        c["id"] = f"mlflow-churn-{i:02d}"

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook gerado com sucesso: {NB_PATH} ({len(cells)} células)")


if __name__ == "__main__":
    build_notebook()
