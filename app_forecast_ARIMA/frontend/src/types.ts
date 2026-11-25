export interface DatasetSummary {
  dataset_id: string;
  filename: string;
  total_rows: number;
  columns: string[];
  date_candidates: string[];
  numeric_columns: string[];
  default_date_col: string;
  default_target_col: string;
  observed_rows: number;
  future_exog_rows: number;
  date_min: string;
  date_max: string;
  preview: Record<string, unknown>[];
}

export interface StationarityTestResult {
  test_name: string;
  statistic: number;
  p_value: number;
  is_stationary: boolean;
  interpretation: string;
}

export interface CorrelationPoint {
  lag: number;
  acf: number;
  pacf: number;
  conf_bound: number;
}

export interface SeriesPoint {
  date: string;
  value: number | null;
}

export interface DiagnosticsResponse {
  n_obs: number;
  mean: number;
  std: number;
  min_val: number;
  max_val: number;
  recommended_d: number;
  recommended_D: number;
  adf: StationarityTestResult;
  kpss: StationarityTestResult;
  correlations: CorrelationPoint[];
  transformed_series: SeriesPoint[];
  didactic_summary: string;
}

export interface MetricSet {
  mape: number;
  rmse: number;
  mae: number;
  r2: number;
  aic: number | null;
  bic: number | null;
}

export interface CoefficientItem {
  name: string;
  estimate: number;
  std_err: number;
  z_stat: number;
  p_value: number;
  ci_lower: number;
  ci_upper: number;
  significant: boolean;
}

export interface ResidualDiagnostic {
  ljung_box_lag: number;
  ljung_box_stat: number;
  ljung_box_pvalue: number;
  is_white_noise: boolean;
  residual_mean: number;
  residual_std: number;
  interpretation: string;
}

export interface ForecastSeriesPoint {
  date: string;
  actual: number | null;
  split: "train" | "test" | "future";
  fitted_or_backtest: number | null;
  forecast: number | null;
  lower_ci: number | null;
  upper_ci: number | null;
  baseline_naive: number | null;
}

export interface ResidualPoint {
  date: string;
  residual: number;
}

export interface AnnualGrowthItem {
  year: number;
  source: "Histórico" | "Projeção" | "Misto";
  months_count: number;
  mean_value: number;
  total_value: number;
  yoy_growth_pct: number | null;
}

export interface ModelCandidateSummary {
  order: string;
  aic: number;
  bic: number;
  selected: boolean;
}

export interface ForecastResponse {
  model_label: string;
  order: number[];
  seasonal_order: number[];
  use_log: boolean;
  exog_cols: string[];
  test_metrics: MetricSet;
  baseline_metrics: MetricSet;
  mape_improvement_pct: number;
  coefficients: CoefficientItem[];
  residuals_diagnostic: ResidualDiagnostic;
  residuals_series: ResidualPoint[];
  series: ForecastSeriesPoint[];
  annual_growth: AnnualGrowthItem[];
  candidates_tested: ModelCandidateSummary[];
  executive_takeaway: string;
}
