"""Schemas Pydantic v2 para o TimeSeries Diagnostics Studio."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

TransformType = Literal["level", "log", "boxcox", "diff1", "diff_seasonal", "diff1_seasonal", "log_diff"]
AggregationType = Literal["none", "month_mean", "month_sum"]
RegressionType = Literal["c", "ct"]


class AnalyzeDemoRequest(BaseModel):
    dataset_id: str = Field(default="varejo_sazonal_i1", description="ID do cenário demonstrativo")
    transform: TransformType = Field(default="level", description="Transformação aplicada à série")
    alpha: float = Field(default=0.05, ge=0.01, le=0.10, description="Nível de significância")
    regression: RegressionType = Field(default="ct", description="Especificação determinística (c ou ct)")
    max_lags: int = Field(default=12, ge=4, le=36, description="Lags máximos para testes de autocorrelação/ARCH")
    seasonal_period: Optional[int] = Field(default=None, ge=1, le=52, description="Período sazonal (auto se omitido)")
    auto_arima: bool = Field(default=True, description="Seleção automática SARIMA por AICc")
    order_p: int = Field(default=1, ge=0, le=5)
    order_d: int = Field(default=1, ge=0, le=2)
    order_q: int = Field(default=1, ge=0, le=5)
    seasonal_P: int = Field(default=0, ge=0, le=2)
    seasonal_D: int = Field(default=0, ge=0, le=1)
    seasonal_Q: int = Field(default=0, ge=0, le=2)
    forecast_horizon: int = Field(default=12, ge=3, le=36, description="Horizonte de previsão")


class DemoDatasetMeta(BaseModel):
    id: str
    title: str
    subtitle: str
    regime_badge: str
    frequency: str
    seasonal_period: int
    n_obs: int
    expected_diagnosis: str
    recommended_transform: TransformType


class UploadInspectResponse(BaseModel):
    filename: str
    file_type: Literal["csv", "excel"]
    detected_separator: Optional[str] = None
    sheets: List[str] = Field(default_factory=list)
    selected_sheet: Optional[str] = None
    columns: List[str]
    numeric_columns: List[str]
    suggested_date_col: str
    suggested_value_col: str
    suggested_period: int
    preview_rows: List[Dict[str, Any]]
    total_rows: int
