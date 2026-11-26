export type ModelType = "local_level" | "local_linear_trend" | "seasonal_local_level";
export type DeseasonalizeMode = "none" | "stl_deseasonalize" | "stl_trend_only";

export interface DemoDatasetMeta {
  id: string;
  name: string;
  filename: string;
  description: string;
  frequency_label: string;
  default_date_col: string;
  default_y_col: string;
  default_x_cols: string[];
  default_intervention_date: string;
  expected_insight: string;
}

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  source: "demo" | "upload";
  rows: number;
  columns: string[];
  numeric_columns: string[];
  date_column: string;
  suggested_y: string;
  suggested_xs: string[];
  min_date: string;
  max_date: string;
  suggested_intervention_date: string;
  inferred_frequency: string;
  preview: Record<string, string | number | null>[];
}

export interface AnalyzeRequestPayload {
  date_col: string;
  y_col: string;
  x_cols: string[];
  start_date: string | null;
  end_date: string | null;
  intervention_date: string;
  alpha: number;
  model_type: ModelType;
  seasonal_period: number;
  deseasonalize: DeseasonalizeMode;
  stl_period: number;
  n_sims: number;
  run_placebo: boolean;
}

export interface TimeSeriesPoint {
  date: string;
  period: "pre" | "post";
  actual: number;
  predicted: number;
  lower: number;
  upper: number;
  point_effect: number;
  point_effect_lower: number;
  point_effect_upper: number;
  cum_effect: number;
  cum_effect_lower: number;
  cum_effect_upper: number;
  level_component: number;
  covariates_component: number;
  seasonal_component: number;
}

export interface ImpactSummaryMetrics {
  pre_start: string;
  pre_end: string;
  post_start: string;
  post_end: string;
  n_pre: number;
  n_post: number;
  alpha: number;
  confidence_level_pct: number;
  avg_actual: number;
  avg_predicted: number;
  avg_predicted_sd: number;
  avg_predicted_lower: number;
  avg_predicted_upper: number;
  avg_abs_effect: number;
  avg_abs_effect_sd: number;
  avg_abs_effect_lower: number;
  avg_abs_effect_upper: number;
  avg_rel_effect_pct: number;
  avg_rel_effect_lower_pct: number;
  avg_rel_effect_upper_pct: number;
  cum_actual: number;
  cum_predicted: number;
  cum_predicted_sd: number;
  cum_predicted_lower: number;
  cum_predicted_upper: number;
  cum_abs_effect: number;
  cum_abs_effect_sd: number;
  cum_abs_effect_lower: number;
  cum_abs_effect_upper: number;
  cum_rel_effect_pct: number;
  cum_rel_effect_lower_pct: number;
  cum_rel_effect_upper_pct: number;
  p_value: number;
  causal_probability_pct: number;
  is_significant: boolean;
  direction: "positivo" | "negativo" | "neutro";
}

export interface CovariateDiagnostic {
  name: string;
  beta_raw: number;
  beta_std: number;
  std_err: number;
  z_stat: number;
  p_value: number;
  importance_pct: number;
  pre_correlation: number;
  post_z_shift: number;
  spillover_warning: boolean;
  status_label: string;
}

export interface PreFitDiagnostics {
  r2_pre: number;
  rmse_pre: number;
  mae_pre: number;
  mape_pre_pct: number;
  ljung_box_stat: number;
  ljung_box_pvalue: number;
  residuals_white_noise: boolean;
  aic: number;
  bic: number;
  fit_quality_label: "excelente" | "bom" | "atencao";
}

export interface PlaceboPoint {
  date: string;
  period: "placebo_pre" | "placebo_post";
  actual: number;
  predicted: number;
  lower: number;
  upper: number;
}

export interface PlaceboResult {
  executed: boolean;
  reason_skipped: string | null;
  placebo_start: string | null;
  placebo_end: string | null;
  n_placebo_pre: number;
  n_placebo_post: number;
  avg_abs_effect: number;
  avg_rel_effect_pct: number;
  p_value: number;
  passed: boolean;
  verdict: string;
  series: PlaceboPoint[];
}

export interface AggregatedImpactRow {
  period_label: string;
  n_obs: number;
  actual_sum: number;
  predicted_sum: number;
  abs_effect_sum: number;
  rel_effect_pct: number;
  cum_effect: number;
  lower_effect_sum: number;
  upper_effect_sum: number;
}

export interface DescriptiveStatRow {
  variable: string;
  period: "Pré-Intervenção" | "Pós-Intervenção" | "Total";
  n: number;
  mean: number;
  median: number;
  std: number;
  min_val: number;
  max_val: number;
}

export interface RawSeriesRow {
  date: string;
  values: Record<string, number>;
}

export interface AnalyzeResponse {
  dataset_name: string;
  y_col: string;
  x_cols: string[];
  model_type: string;
  model_label: string;
  deseasonalize: string;
  univariate_mode: boolean;
  warnings: string[];
  metrics: ImpactSummaryMetrics;
  pre_fit: PreFitDiagnostics;
  covariates: CovariateDiagnostic[];
  placebo: PlaceboResult;
  series: TimeSeriesPoint[];
  aggregated_tables: Record<string, AggregatedImpactRow[]>;
  descriptive_stats: DescriptiveStatRow[];
  raw_series: RawSeriesRow[];
  executive_summary: string;
  detailed_report_md: string;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  dataset_loaded: boolean;
  current_dataset_name: string | null;
  demos_available: number;
}
