"""Contratos tipados Pydantic v2 da API CausalImpact Studio."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DemoDatasetMeta(BaseModel):
    id: str
    name: str
    filename: str
    description: str
    frequency_label: str
    default_date_col: str
    default_y_col: str
    default_x_cols: List[str]
    default_intervention_date: str
    expected_insight: str


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    source: Literal["demo", "upload"]
    rows: int
    columns: List[str]
    numeric_columns: List[str]
    date_column: str
    suggested_y: str
    suggested_xs: List[str]
    min_date: str
    max_date: str
    suggested_intervention_date: str
    inferred_frequency: str
    preview: List[Dict[str, Any]]


class AnalyzeRequest(BaseModel):
    date_col: str
    y_col: str
    x_cols: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    intervention_date: str
    alpha: float = Field(default=0.05, gt=0.001, lt=0.5)
    model_type: Literal["local_level", "local_linear_trend", "seasonal_local_level"] = "local_level"
    seasonal_period: int = Field(default=12, ge=2, le=52)
    deseasonalize: Literal["none", "stl_deseasonalize", "stl_trend_only"] = "none"
    stl_period: int = Field(default=12, ge=2, le=52)
    n_sims: int = Field(default=1000, ge=200, le=5000)
    run_placebo: bool = True


class TimeSeriesPoint(BaseModel):
    date: str
    period: Literal["pre", "post"]
    actual: float
    predicted: float
    lower: float
    upper: float
    point_effect: float
    point_effect_lower: float
    point_effect_upper: float
    cum_effect: float
    cum_effect_lower: float
    cum_effect_upper: float
    level_component: float
    covariates_component: float
    seasonal_component: float


class ImpactSummaryMetrics(BaseModel):
    pre_start: str
    pre_end: str
    post_start: str
    post_end: str
    n_pre: int
    n_post: int
    alpha: float
    confidence_level_pct: float
    # Average (per period post-intervention)
    avg_actual: float
    avg_predicted: float
    avg_predicted_sd: float
    avg_predicted_lower: float
    avg_predicted_upper: float
    avg_abs_effect: float
    avg_abs_effect_sd: float
    avg_abs_effect_lower: float
    avg_abs_effect_upper: float
    avg_rel_effect_pct: float
    avg_rel_effect_lower_pct: float
    avg_rel_effect_upper_pct: float
    # Cumulative (sum over post-intervention)
    cum_actual: float
    cum_predicted: float
    cum_predicted_sd: float
    cum_predicted_lower: float
    cum_predicted_upper: float
    cum_abs_effect: float
    cum_abs_effect_sd: float
    cum_abs_effect_lower: float
    cum_abs_effect_upper: float
    cum_rel_effect_pct: float
    cum_rel_effect_lower_pct: float
    cum_rel_effect_upper_pct: float
    # Bayesian / Posterior significance
    p_value: float
    causal_probability_pct: float
    is_significant: bool
    direction: Literal["positivo", "negativo", "neutro"]


class CovariateDiagnostic(BaseModel):
    name: str
    beta_raw: float
    beta_std: float
    std_err: float
    z_stat: float
    p_value: float
    importance_pct: float
    pre_correlation: float
    post_z_shift: float
    spillover_warning: bool
    status_label: str


class PreFitDiagnostics(BaseModel):
    r2_pre: float
    rmse_pre: float
    mae_pre: float
    mape_pre_pct: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    residuals_white_noise: bool
    aic: float
    bic: float
    fit_quality_label: Literal["excelente", "bom", "atencao"]


class PlaceboPoint(BaseModel):
    date: str
    period: Literal["placebo_pre", "placebo_post"]
    actual: float
    predicted: float
    lower: float
    upper: float


class PlaceboResult(BaseModel):
    executed: bool
    reason_skipped: Optional[str] = None
    placebo_start: Optional[str] = None
    placebo_end: Optional[str] = None
    n_placebo_pre: int = 0
    n_placebo_post: int = 0
    avg_abs_effect: float = 0.0
    avg_rel_effect_pct: float = 0.0
    p_value: float = 1.0
    passed: bool = True
    verdict: str = ""
    series: List[PlaceboPoint] = Field(default_factory=list)


class AggregatedImpactRow(BaseModel):
    period_label: str
    n_obs: int
    actual_sum: float
    predicted_sum: float
    abs_effect_sum: float
    rel_effect_pct: float
    cum_effect: float
    lower_effect_sum: float
    upper_effect_sum: float


class DescriptiveStatRow(BaseModel):
    variable: str
    period: Literal["Pré-Intervenção", "Pós-Intervenção", "Total"]
    n: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float


class RawSeriesRow(BaseModel):
    date: str
    values: Dict[str, float]


class AnalyzeResponse(BaseModel):
    dataset_name: str
    y_col: str
    x_cols: List[str]
    model_type: str
    model_label: str
    deseasonalize: str
    univariate_mode: bool
    warnings: List[str]
    metrics: ImpactSummaryMetrics
    pre_fit: PreFitDiagnostics
    covariates: List[CovariateDiagnostic]
    placebo: PlaceboResult
    series: List[TimeSeriesPoint]
    aggregated_tables: Dict[str, List[AggregatedImpactRow]]
    descriptive_stats: List[DescriptiveStatRow]
    raw_series: List[RawSeriesRow]
    executive_summary: str
    detailed_report_md: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    dataset_loaded: bool
    current_dataset_name: Optional[str]
    demos_available: int
