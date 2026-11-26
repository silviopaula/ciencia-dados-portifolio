export type TransformType =
  | "level"
  | "log"
  | "boxcox"
  | "diff1"
  | "diff_seasonal"
  | "diff1_seasonal"
  | "log_diff";

export type AggregationType = "none" | "month_mean" | "month_sum";
export type RegressionType = "c" | "ct";

export interface DemoDatasetMeta {
  id: string;
  title: string;
  subtitle: string;
  regime_badge: string;
  frequency: string;
  seasonal_period: number;
  n_obs: number;
  expected_diagnosis: string;
  recommended_transform: TransformType;
}

export interface UploadInspectResponse {
  filename: string;
  file_type: "csv" | "excel";
  detected_separator: string | null;
  sheets: string[];
  selected_sheet: string | null;
  columns: string[];
  numeric_columns: string[];
  suggested_date_col: string;
  suggested_value_col: string;
  suggested_period: number;
  preview_rows: Record<string, string>[];
  total_rows: number;
}

export interface AnalysisConfig {
  sourceMode: "demo" | "upload";
  datasetId: string;
  transform: TransformType;
  alpha: number;
  regression: RegressionType;
  maxLags: number;
  seasonalPeriod: number;
  autoArima: boolean;
  orderP: number;
  orderD: number;
  orderQ: number;
  seasonalP: number;
  seasonalD: number;
  seasonalQ: number;
  forecastHorizon: number;
  // Upload specific
  sheetName: string;
  dateCol: string;
  valueCol: string;
  dateFormat: string;
  aggregation: AggregationType;
}

export interface StatTestItem {
  id: string;
  category: string;
  name: string;
  target_series: string;
  statistic: number | null;
  p_value: number | null;
  p_formatted: string;
  critical_5pct: number | null;
  h0: string;
  passed: boolean;
  status_label: string;
  interpretation: string;
  extra_note: string | null;
}

export interface TransformRow {
  key: TransformType;
  label: string;
  available: boolean;
  reason: string | null;
  n_obs: number;
  adf_stat: number | null;
  adf_p: number | null;
  pp_p: number | null;
  kpss_p: number | null;
  arch_p: number | null;
  std: number | null;
  consensus: string;
  status: "ok" | "warn" | "fail" | "na";
  recommended: boolean;
}

export interface SeriesRow {
  date: string;
  raw_value: number | null;
  value: number | null;
  rolling_mean: number | null;
  rolling_std: number | null;
  stl_trend: number | null;
  stl_seasonal: number | null;
  stl_remainder: number | null;
  fitted: number | null;
  residual: number | null;
  is_outlier: boolean;
  outlier_value: number | null;
}

export interface ForecastPoint {
  date: string;
  mean: number | null;
  lower_80: number | null;
  upper_80: number | null;
  lower_95: number | null;
  upper_95: number | null;
}

export interface AnalysisPayload {
  meta: {
    dataset_id: string | null;
    dataset_title: string;
    transform: TransformType;
    transform_label: string;
    boxcox_lambda: number | null;
    n_original: number;
    n_analyzed: number;
    date_start: string;
    date_end: string;
    seasonal_period: number;
    alpha: number;
    regression: RegressionType;
    max_lags: number;
    rolling_window: number;
  };
  stationarity_consensus: {
    quadrant: "estacionaria" | "quebra_estrutural" | "nao_estacionaria" | "divergente";
    badge_color: "emerald" | "amber" | "rose" | "indigo";
    regime_title: string;
    summary: string;
    recommendation: string;
    suggested_d: number;
  };
  adf_specifications: Record<
    "c" | "ct" | "n",
    {
      statistic: number | null;
      p_value: number | null;
      lags: number;
      nobs: number;
      crit_1: number | null;
      crit_5: number | null;
      crit_10: number | null;
    }
  >;
  kpss_specifications: Record<
    "c" | "ct",
    {
      statistic: number | null;
      p_value: number | null;
      lags: number;
      crit_1: number | null;
      crit_5: number | null;
      crit_10: number | null;
      note: string | null;
    }
  >;
  zivot_andrews: {
    statistic: number | null;
    p_value: number | null;
    lags: number;
    break_index: number | null;
    break_date: string | null;
    crit_1: number | null;
    crit_5: number | null;
    crit_10: number | null;
  };
  tests_summary: StatTestItem[];
  ljung_box_curve: { lag: number; lb_p: number | null; alpha: number }[];
  durbin_watson: number | null;
  acf_pacf: {
    lag: number;
    acf: number | null;
    pacf: number | null;
    upper: number | null;
    lower: number | null;
    acf_sig: boolean;
    pacf_sig: boolean;
  }[];
  box_jenkins_hint: string;
  transform_comparison: TransformRow[];
  descriptive_stats: {
    n: number;
    mean: number | null;
    std: number | null;
    min: number | null;
    q1: number | null;
    median: number | null;
    q3: number | null;
    max: number | null;
    iqr: number | null;
    cv_pct: number | null;
    skewness: number | null;
    kurtosis: number | null;
    resid_skewness: number | null;
    resid_kurtosis: number | null;
  };
  outliers: {
    iqr_low: number | null;
    iqr_high: number | null;
    iqr_count: number;
    mad_count: number;
    total_flagged: number;
    top_outliers: {
      date: string;
      value: number | null;
      z_mad: number | null;
      methods: string;
    }[];
  };
  distributions: {
    values_histogram: { bin_label: string; bin_mid: number | null; count: number; normal_expected: number | null }[];
    residuals_histogram: { bin_label: string; bin_mid: number | null; count: number; normal_expected: number | null }[];
    residuals_qq: { theoretical: number | null; sample: number | null; ref_line: number | null }[];
  };
  stl: {
    trend_strength: number;
    seasonal_strength: number;
    has_seasonality: boolean;
  };
  arima: {
    spec_label: string;
    order: number[];
    seasonal_order: number[];
    aic: number | null;
    aicc: number | null;
    bic: number | null;
    forecast: ForecastPoint[];
    backtest: {
      horizon: number;
      rmse: number | null;
      mae: number | null;
      mape: number | null;
      naive_rmse: number | null;
      naive_mae: number | null;
      naive_mape: number | null;
      skill_vs_naive_pct: number | null;
      series: { date: string; actual: number | null; sarima: number | null; naive: number | null }[];
    };
  };
  series_table: SeriesRow[];
}
