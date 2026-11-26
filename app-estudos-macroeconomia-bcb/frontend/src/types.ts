export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  studies_count: number;
  db_tables: number;
  bcb_series: number;
}

export interface DbStats {
  exists: boolean;
  path: string;
  series_count: number;
  tables_count: number;
  total_rows: number;
  last_updated: string | null;
  size_mb: number;
}

export interface ConjunturaKpi {
  key: string;
  label: string;
  unit: string;
  value: number | null;
  delta_12m: number | null;
  last_date: string | null;
}

export interface SyncDetail {
  sgs_id: number;
  label: string;
  live: boolean;
  rows: number;
  last_date: string;
}

export interface ConjunturaResponse {
  synced_live_count: number;
  elapsed_ms: number;
  kpis: ConjunturaKpi[];
  charts: {
    juros_inflacao: Array<{
      date: string;
      selic: number | null;
      ipca12: number | null;
      meta: number | null;
    }>;
    fiscal_cambio: Array<{
      date: string;
      dbgg: number | null;
      cambio: number | null;
    }>;
    atividade_trabalho: Array<{
      date: string;
      ibcbr: number | null;
      desoc: number | null;
    }>;
  };
  db_stats: DbStats;
  sync_details?: SyncDetail[];
}

export interface ThemeMeta {
  key: string;
  label: string;
  count: number;
}

export interface StudyMeta {
  code: string;
  slug: string;
  title: string;
  subtitle: string;
  theme: string;
  theme_label: string;
  specialized: boolean;
  model_type: string;
  methodology_label: string;
  economic_intuition: string;
  methodology_explanation: string;
  interpretation_guide: string;
  references: string[];
  dataset_keys: string[];
  bcb_series: [number, string][];
}

export interface StudiesListResponse {
  total: number;
  specialized_count: number;
  themes: ThemeMeta[];
  db_stats: DbStats;
  items: StudyMeta[];
}

export interface StudyKpi {
  label: string;
  value: string;
  sub: string;
}

export interface AdfRow {
  series: string;
  statistic: number;
  pvalue: number;
  lags_used: number;
  nobs: number;
  stationary_5pct: boolean;
}

export interface CoefRow {
  term: string;
  coef: number;
  std_err: number;
  t_stat: number;
  pvalue: number;
  sig: boolean;
}

export interface SummaryStatRow {
  column: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  median: number;
  max: number;
}

export interface CorrMatrix {
  cols: string[];
  rows: number[][];
}

export interface DedicatedM01Result {
  kind: "dedicated_m01";
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  level_series: Array<{ date: string; cds5y: number; selic: number }>;
  diff_series: Array<{ date: string; d_cds5y: number; d_selic: number }>;
  adf: AdfRow[];
  lag_criteria: { AIC: number; BIC: number; HQIC: number; FPE: number };
  ccf: Array<{ lag: number; corr: number }>;
  granger_full: Array<{
    lag: number;
    f_selic_to_cds: number;
    p_selic_to_cds: number;
    sig_selic_to_cds: boolean;
    f_cds_to_selic: number;
    p_cds_to_selic: number;
    sig_cds_to_selic: boolean;
  }>;
  granger_sub: Array<{
    lag: number;
    f_selic_to_cds: number;
    p_selic_to_cds: number;
    sig_selic_to_cds: boolean;
    f_cds_to_selic: number;
    p_cds_to_selic: number;
    sig_cds_to_selic: boolean;
  }>;
  conclusion: string;
}

export interface DedicatedM02Result {
  kind: "dedicated_m02";
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  forecast_series: Array<{
    date: string;
    observed: number | null;
    forecast: number | null;
    ci95: [number, number] | null;
    ci80: [number, number] | null;
  }>;
  fiscal_panel: Array<{
    date: string;
    s_primario: number;
    s_req: number;
    juro_real: number;
    pib: number;
  }>;
  adf: AdfRow[];
  conclusion: string;
}

export interface DedicatedM06Result {
  kind: "dedicated_m06";
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  series: Array<{
    date: string;
    selic_obs: number;
    selic_fit: number;
    hiato: number;
    desvio_inf: number;
  }>;
  coefs: CoefRow[];
  conclusion: string;
}

export interface DedicatedM12Result {
  kind: "dedicated_m12";
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  ipca_series: Array<{ date: string; ipca12: number; meta: number }>;
  rolling_series: Array<{ date: string; gamma: number; band: [number, number] }>;
  conclusion: string;
}

export interface DedicatedM13Result {
  kind: "dedicated_m13";
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  forecast_series: Array<{
    date: string;
    observed: number | null;
    forecast: number | null;
    ci95: [number, number] | null;
  }>;
  correlogram: Array<{ lag: number; acf: number; pacf: number }>;
  adf: AdfRow[];
  conclusion: string;
}

export interface DedicatedM14Result {
  kind: "dedicated_m14";
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  dep_var: string;
  series: Array<{ date: string; actual: number; fitted: number; resid: number }>;
  coefs: CoefRow[];
  adf: AdfRow[];
  conclusion: string;
}

export type ThematicModelResult =
  | {
      model_type: "var_granger";
      short_label: string;
      var_names: [string, string];
      used_lag: number;
      aic: number;
      irf_points: Array<{ h: number; shock_2to1: number; shock_1to2: number }>;
      granger_rows: Array<{
        lag: number;
        f_2to1: number;
        p_2to1: number;
        sig_2to1: boolean;
        f_1to2: number;
        p_1to2: number;
        sig_1to2: boolean;
      }>;
      conclusion: string;
    }
  | {
      model_type: "coint_vecm";
      short_label: string;
      var_names: [string, string];
      coint_stat: number;
      coint_pval: number;
      is_cointegrated: boolean;
      beta_lr: number;
      alpha_ect: number;
      p_ect: number;
      points: Array<{ date: string; actual: number; equilibrium: number; ect: number }>;
      conclusion: string;
    }
  | {
      model_type: "hp_neutral";
      short_label: string;
      target: string;
      hp_lambda: number;
      latest_trend: number;
      latest_cycle: number;
      points: Array<{ date: string; observed: number; trend: number; cycle: number }>;
      conclusion: string;
    }
  | {
      model_type: "volatility";
      short_label: string;
      target: string;
      ret_label: string;
      latest_vol: number;
      max_vol: number;
      kurtosis: number;
      skewness: number;
      points: Array<{ date: string; ret: number; vol: number }>;
      conclusion: string;
    }
  | {
      model_type: "sarima_forecast";
      short_label: string;
      target: string;
      aic: number;
      bic: number;
      forecast_points: Array<{
        date: string;
        observed: number | null;
        forecast: number | null;
        ci95: [number, number] | null;
      }>;
      correlogram: Array<{ lag: number; acf: number; pacf: number }>;
      conclusion: string;
    }
  | {
      model_type: "ols_regression";
      short_label: string;
      dep_var: string;
      r2: number;
      adj_r2: number;
      f_stat: number;
      nobs: number;
      coefs: CoefRow[];
      points: Array<{ date: string; actual: number; fitted: number; resid: number }>;
      conclusion: string;
    };

export interface ThematicResult {
  kind: "thematic";
  model_type: string;
  elapsed_ms: number;
  source_note: string;
  kpis: StudyKpi[];
  series_keys: string[];
  raw_chart: Array<Record<string, string | number>>;
  model_result: ThematicModelResult;
  adf: AdfRow[];
  summary_stats: SummaryStatRow[];
  corr_matrix: CorrMatrix;
  conclusion: string;
}

export type StudyRunResult =
  | DedicatedM01Result
  | DedicatedM02Result
  | DedicatedM06Result
  | DedicatedM12Result
  | DedicatedM13Result
  | DedicatedM14Result
  | ThematicResult;

export interface ExecuteStudyResponse {
  study: StudyMeta;
  result: StudyRunResult;
}
