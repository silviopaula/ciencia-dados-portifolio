"""Contratos Pydantic v2 para a API de Séries Temporais SARIMA / SARIMAX."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    total_rows: int
    columns: List[str]
    date_candidates: List[str]
    numeric_columns: List[str]
    default_date_col: str
    default_target_col: str
    observed_rows: int
    future_exog_rows: int
    date_min: str
    date_max: str
    preview: List[Dict[str, Any]]


class DiagnosticsRequest(BaseModel):
    dataset_id: str
    date_col: str
    target_col: str
    use_log: bool = False
    diff_order: int = Field(default=0, ge=0, le=2)
    seasonal_diff: int = Field(default=0, ge=0, le=1)
    seasonal_period: int = Field(default=12, ge=1, le=52)
    nlags: int = Field(default=24, ge=6, le=36)


class StationarityTestResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    interpretation: str


class CorrelationPoint(BaseModel):
    lag: int
    acf: float
    pacf: float
    conf_bound: float


class SeriesPoint(BaseModel):
    date: str
    value: Optional[float]


class DiagnosticsResponse(BaseModel):
    n_obs: int
    mean: float
    std: float
    min_val: float
    max_val: float
    recommended_d: int
    recommended_D: int
    adf: StationarityTestResult
    kpss: StationarityTestResult
    correlations: List[CorrelationPoint]
    transformed_series: List[SeriesPoint]
    didactic_summary: str


class ForecastRequest(BaseModel):
    dataset_id: str
    date_col: str
    target_col: str
    exog_cols: List[str] = Field(default_factory=list)
    mode: Literal["auto", "manual"] = "auto"
    p: int = Field(default=1, ge=0, le=4)
    d: int = Field(default=1, ge=0, le=2)
    q: int = Field(default=1, ge=0, le=4)
    P: int = Field(default=1, ge=0, le=2)
    D: int = Field(default=1, ge=0, le=1)
    Q: int = Field(default=1, ge=0, le=2)
    s: int = Field(default=12, ge=1, le=52)
    use_log: bool = True
    test_periods: int = Field(default=12, ge=3, le=36)
    forecast_horizon: int = Field(default=24, ge=1, le=120)
    confidence_level: float = Field(default=0.95, ge=0.80, le=0.99)


class MetricSet(BaseModel):
    mape: float
    rmse: float
    mae: float
    r2: float
    aic: Optional[float] = None
    bic: Optional[float] = None


class CoefficientItem(BaseModel):
    name: str
    estimate: float
    std_err: float
    z_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool


class ResidualDiagnostic(BaseModel):
    ljung_box_lag: int
    ljung_box_stat: float
    ljung_box_pvalue: float
    is_white_noise: bool
    residual_mean: float
    residual_std: float
    interpretation: str


class ForecastSeriesPoint(BaseModel):
    date: str
    actual: Optional[float] = None
    split: Literal["train", "test", "future"]
    fitted_or_backtest: Optional[float] = None
    forecast: Optional[float] = None
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    baseline_naive: Optional[float] = None


class ResidualPoint(BaseModel):
    date: str
    residual: float


class AnnualGrowthItem(BaseModel):
    year: int
    source: Literal["Histórico", "Projeção", "Misto"]
    months_count: int
    mean_value: float
    total_value: float
    yoy_growth_pct: Optional[float] = None


class ModelCandidateSummary(BaseModel):
    order: str
    aic: float
    bic: float
    selected: bool


class ForecastResponse(BaseModel):
    model_label: str
    order: List[int]
    seasonal_order: List[int]
    use_log: bool
    exog_cols: List[str]
    test_metrics: MetricSet
    baseline_metrics: MetricSet
    mape_improvement_pct: float
    coefficients: List[CoefficientItem]
    residuals_diagnostic: ResidualDiagnostic
    residuals_series: List[ResidualPoint]
    series: List[ForecastSeriesPoint]
    annual_growth: List[AnnualGrowthItem]
    candidates_tested: List[ModelCandidateSummary]
    executive_takeaway: str
