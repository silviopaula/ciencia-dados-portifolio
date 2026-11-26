/* ==========================================================================
   TimeSeries Studio Pro — Pure JavaScript SPA Controller
   Communicates with the Python FastAPI backend (/api/*) and renders
   interactive Plotly charts & responsive tables.
   ========================================================================== */

const APP = {
  dataset: null,
  outlierResult: null,
  outlierViewMode: "time",
  seasonalityResult: null,
  editorResult: null,
  editorSelectedIndices: [],
  editorActiveTab: "comp",
  arimaResult: null,
  autotsResult: null,
  growthResult: null,
  growthActiveTab: "wide_valores",
};

const VIEW_HEADERS = {
  "view-data": {
    title: "Central de Dados & Configuração",
    subtitle: "Importe arquivos Excel (.xlsx) ou CSV, selecione abas e configure a coluna temporal",
  },
  "view-outliers": {
    title: "Tratamento Avançado de Outliers",
    subtitle: "Identifique anomalias com 6 algoritmos estatísticos/ML e corrija com 7 métodos e histórico de Undo",
  },
  "view-seasonality": {
    title: "Testes de Sazonalidade, Participação Mensal & Controle XmR",
    subtitle: "Limites históricos de participação mensal, controle estatístico XmR e decomposição STL Residuals",
  },
  "view-editor": {
    title: "Editor Avançado de Séries Temporais",
    subtitle: "Clique ou selecione pontos com laço no gráfico para simular cenários absolutos ou percentuais (suporta Ctrl+Z)",
  },
  "view-arima": {
    title: "Previsão ARIMA / SARIMAX (Univariado & Multivariado)",
    subtitle: "Modelagem automática ou manual com variáveis exógenas, teste de colinearidade e ranking de desvios",
  },
  "view-autots": {
    title: "Previsão AutoTS Multi-Modelos Univariados",
    subtitle: "Benchmark simultâneo entre Prophet, ETS, SARIMA, TBATS, BATS, STLM e Ensemble Bagged",
  },
  "view-growth": {
    title: "Análise Multirresolução de Taxas de Crescimento",
    subtitle: "Matrizes e gráficos de valores, crescimento (%) e variações absolutas (Mensal, Trimestral, Semestral e Anual)",
  },
};

const MODEL_COLORS = {
  prophet: "#ef4444",
  ets: "#10b981",
  sarima: "#3b82f6",
  tbats: "#a855f7",
  bats: "#f59e0b",
  stlm: "#ec4899",
  bagged: "#06b6d4",
};

// ============================================================================
// UTILITIES: TOAST, LOADING, FETCH & FORMATTING
// ============================================================================

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 250);
  }, 4000);
}

function setLoading(active) {
  const bar = document.getElementById("loadingBar");
  if (active) bar.classList.add("active");
  else bar.classList.remove("active");
}

async function apiFetch(url, options = {}) {
  setLoading(true);
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let errMsg = `Erro HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errMsg = errJson.detail;
      } catch (_) {}
      throw new Error(errMsg);
    }
    return await res.json();
  } finally {
    setLoading(false);
  }
}

function fmtNum(val, decimals = 2) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
  return Number(val).toLocaleString("pt-BR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtInt(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
  return Math.round(Number(val)).toLocaleString("pt-BR");
}

function getPlotlyThemeLayout(extra = {}) {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  const textColor = isLight ? "#0f172a" : "#e2e8f0";
  const gridColor = isLight ? "rgba(148,163,184,0.22)" : "rgba(148,163,184,0.12)";
  const paperBg = "rgba(0,0,0,0)";
  const plotBg = "rgba(0,0,0,0)";

  return Object.assign(
    {
      paper_bgcolor: paperBg,
      plot_bgcolor: plotBg,
      font: { family: "Inter, sans-serif", color: textColor, size: 12 },
      margin: { t: 44, r: 24, b: 48, l: 60 },
      xaxis: { gridcolor: gridColor, zerolinecolor: gridColor },
      yaxis: { gridcolor: gridColor, zerolinecolor: gridColor },
      legend: { orientation: "h", y: -0.18, x: 0 },
      hovermode: "x unified",
    },
    extra
  );
}

function getStatusBadgeHTML(statusText) {
  if (!statusText) return `<span class="badge badge-neutral">—</span>`;
  const s = statusText.toLowerCase();
  if (s.includes("ótimo") || s.includes("otimo")) return `<span class="badge badge-otimo">${statusText}</span>`;
  if (s.includes("regular") || s.includes("recomendado") || s.includes("aceit")) return `<span class="badge badge-regular">${statusText}</span>`;
  if (s.includes("ruim") || s.includes("avaliar")) return `<span class="badge badge-ruim">${statusText}</span>`;
  if (s.includes("reprojetar")) return `<span class="badge badge-reprojetar">${statusText}</span>`;
  return `<span class="badge badge-neutral">${statusText}</span>`;
}

// ============================================================================
// NAVIGATION & THEME SWITCHING
// ============================================================================

function switchView(viewId) {
  document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));

  const target = document.getElementById(viewId);
  if (target) target.classList.add("active");

  const btn = document.querySelector(`.nav-item[data-view="${viewId}"]`);
  if (btn) btn.classList.add("active");

  const hdr = VIEW_HEADERS[viewId];
  if (hdr) {
    document.getElementById("pageTitle").textContent = hdr.title;
    document.getElementById("pageSubtitle").textContent = hdr.subtitle;
  }

  document.getElementById("sidebar").classList.remove("open");

  // Trigger module auto-initialization when visiting a tab for the first time
  if (viewId === "view-outliers" && !APP.outlierResult && APP.dataset?.numeric_columns?.length) {
    runOutlierIdentification();
  } else if (viewId === "view-seasonality" && !APP.seasonalityResult && APP.dataset?.numeric_columns?.length) {
    runSeasonalityAnalysis();
  } else if (viewId === "view-editor" && !APP.editorResult && APP.dataset?.numeric_columns?.length) {
    initEditorSeries();
  } else if (viewId === "view-growth" && !APP.growthResult && APP.dataset?.numeric_columns?.length) {
    populateGrowthTargets();
    runGrowthAnalysis();
  }

  // Resize Plotly charts inside newly visible tab
  setTimeout(() => {
    document.querySelectorAll(`#${viewId} .js-plotly-plot`).forEach((gd) => {
      Plotly.Plots.resize(gd);
    });
  }, 60);
}

// ============================================================================
// MODULE 0: DATA HUB & DATASET SYNC
// ============================================================================

async function loadDatasetInfo() {
  try {
    const info = await apiFetch("/api/dataset/info");
    if (info.loaded) {
      applyDatasetMetadata(info);
    }
  } catch (err) {
    showToast(`Erro ao carregar dados: ${err.message}`, "error");
  }
}

function applyDatasetMetadata(info) {
  APP.dataset = info;

  // Sidebar & Header
  document.getElementById("sidebarDatasetName").textContent = info.filename || "Dataset";
  document.getElementById("sidebarDatasetMeta").textContent = `${info.total_rows} linhas • ${info.numeric_columns.length} variáveis`;
  document.getElementById("headerDateRange").textContent = `${info.min_date || "--"} a ${info.max_date || "--"}`;

  // KPIs
  document.getElementById("kpiFileName").textContent = info.filename || "--";
  document.getElementById("kpiSheetName").textContent = `Aba: ${info.selected_sheet || "--"}`;
  document.getElementById("kpiTotalRows").textContent = fmtInt(info.total_rows);
  document.getElementById("kpiTotalVars").textContent = fmtInt(info.numeric_columns.length);

  if (info.years && info.years.length > 0) {
    document.getElementById("kpiYearsSpan").textContent = `${info.years[0]} – ${info.years[info.years.length - 1]}`;
  } else {
    document.getElementById("kpiYearsSpan").textContent = "--";
  }
  document.getElementById("kpiDateBounds").textContent = `${info.min_date || ""} até ${info.max_date || ""}`;

  // Config selects
  const sheetSel = document.getElementById("cfgSheetSelect");
  sheetSel.innerHTML = (info.sheets || [])
    .map((s) => `<option value="${s}" ${s === info.selected_sheet ? "selected" : ""}>${s}</option>`)
    .join("");

  const dateColSel = document.getElementById("cfgDateColSelect");
  dateColSel.innerHTML = (info.raw_columns || [])
    .map((c) => `<option value="${c}" ${c === info.selected_date_column ? "selected" : ""}>${c}</option>`)
    .join("");

  // Populate numeric variable selects across all modules
  const numCols = info.numeric_columns || [];
  const colOptionsHTML = numCols.map((c) => `<option value="${c}">${c}</option>`).join("");

  ["outColSelect", "seasColSelect", "edColSelect", "arimaTargetSelect", "autotsVarSelect"].forEach((selId) => {
    const el = document.getElementById(selId);
    const prevVal = el.value;
    el.innerHTML = colOptionsHTML;
    if (prevVal && numCols.includes(prevVal)) {
      el.value = prevVal;
    }
  });

  // Seasonality year bounds
  if (info.years && info.years.length > 0) {
    document.getElementById("seasStartYear").value = info.years[0];
    document.getElementById("seasEndYear").value = info.years[info.years.length - 1];
  }

  // ARIMA & AutoTS default dates based on first numeric column's valid range
  syncForecastDatesForVariable(numCols[0]);
  populateArimaExogCheckboxes();
  populateGrowthTargets();

  // Render Column Summaries Table
  const sumBody = document.querySelector("#tableColSummaries tbody");
  sumBody.innerHTML = (info.column_summaries || [])
    .map(
      (cs) => `
      <tr>
        <td><strong>${cs.column}</strong></td>
        <td>${fmtInt(cs.non_null)}</td>
        <td>${cs.missing > 0 ? `<span class="text-neg">${fmtInt(cs.missing)}</span>` : "0"}</td>
        <td>${fmtNum(cs.mean, 2)}</td>
        <td>${fmtNum(cs.min, 2)}</td>
        <td>${fmtNum(cs.max, 2)}</td>
        <td>${cs.start_date || "—"}</td>
        <td>${cs.end_date || "—"}</td>
      </tr>`
    )
    .join("");

  // Render Data Preview Table
  const prevHead = document.querySelector("#tableDataPreview thead");
  const prevBody = document.querySelector("#tableDataPreview tbody");
  const pCols = info.preview_columns || [];
  prevHead.innerHTML = `<tr>${pCols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  prevBody.innerHTML = (info.preview_rows || [])
    .slice(0, 60)
    .map(
      (row) =>
        `<tr>${pCols
          .map((c) => {
            const v = row[c];
            if (v === null || v === undefined) return `<td style="color:var(--text-muted)">NA</td>`;
            if (typeof v === "number") return `<td>${fmtNum(v, 2)}</td>`;
            return `<td>${v}</td>`;
          })
          .join("")}</tr>`
    )
    .join("");
}

function syncForecastDatesForVariable(varName) {
  if (!APP.dataset) return;
  const summary = (APP.dataset.column_summaries || []).find((s) => s.column === varName);
  const startDt = summary?.start_date || APP.dataset.min_date || "2010-01-01";
  const endDt = summary?.end_date || APP.dataset.max_date || "2024-12-01";

  // Default forecast horizon: +24 months after endDt
  const endObj = new Date(endDt + "T00:00:00");
  endObj.setFullYear(endObj.getFullYear() + 2);
  const predDt = endObj.toISOString().slice(0, 10);

  document.getElementById("arimaStartDate").value = startDt;
  document.getElementById("arimaEndDate").value = endDt;
  document.getElementById("arimaEndPredict").value = predDt;

  document.getElementById("autotsStartDate").value = startDt;
  document.getElementById("autotsEndDate").value = endDt;
  document.getElementById("autotsEndPredict").value = predDt;
}

function populateArimaExogCheckboxes() {
  if (!APP.dataset) return;
  const target = document.getElementById("arimaTargetSelect").value;
  const container = document.getElementById("arimaExogContainer");
  const exogCols = (APP.dataset.numeric_columns || []).filter((c) => c !== target);
  container.innerHTML = exogCols
    .map(
      (c) => `
      <label class="pill-checkbox">
        <input type="checkbox" name="arimaExog" value="${c}" />
        <span>${c}</span>
      </label>`
    )
    .join("");
}

// ============================================================================
// MODULE 1: OUTLIER IDENTIFICATION & CORRECTION
// ============================================================================

function updateOutlierMethodUI() {
  const identMethod = document.getElementById("outIdentMethod").value;
  const threshGroup = document.getElementById("outThresholdGroup");
  const slider = document.getElementById("outThresholdSlider");
  const valSpan = document.getElementById("outThresholdVal");

  if (identMethod === "zscore") {
    threshGroup.style.display = "block";
    slider.min = "1.0";
    slider.max = "5.0";
    slider.step = "0.1";
    slider.value = "3.0";
    valSpan.textContent = "3.0";
  } else if (identMethod === "iqr") {
    threshGroup.style.display = "block";
    slider.min = "1.0";
    slider.max = "3.5";
    slider.step = "0.1";
    slider.value = "1.5";
    valSpan.textContent = "1.5";
  } else if (identMethod === "isolation_forest") {
    threshGroup.style.display = "block";
    slider.min = "0.3";
    slider.max = "0.9";
    slider.step = "0.05";
    slider.value = "0.6";
    valSpan.textContent = "0.6";
  } else if (identMethod === "lof") {
    threshGroup.style.display = "block";
    slider.min = "1.0";
    slider.max = "3.0";
    slider.step = "0.1";
    slider.value = "1.5";
    valSpan.textContent = "1.5";
  } else if (identMethod === "svm") {
    threshGroup.style.display = "block";
    slider.min = "0.01";
    slider.max = "0.25";
    slider.step = "0.01";
    slider.value = "0.05";
    valSpan.textContent = "0.05";
  } else {
    threshGroup.style.display = "none";
  }
}

async function runOutlierIdentification() {
  const column = document.getElementById("outColSelect").value;
  const method = document.getElementById("outIdentMethod").value;
  const threshold = parseFloat(document.getElementById("outThresholdSlider").value);
  if (!column) return;

  try {
    const res = await apiFetch("/api/outliers/identify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column, method, threshold }),
    });
    APP.outlierResult = res;
    renderOutlierModule(res);
    showToast(`Identificação concluída: ${res.kpis.outliers_count} outliers detectados.`, "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function runOutlierCorrection() {
  const column = document.getElementById("outColSelect").value;
  const method = document.getElementById("outCorrectMethod").value;
  const window_size = parseInt(document.getElementById("outWindowInput").value || "5", 10);
  if (!column) return;

  try {
    const res = await apiFetch("/api/outliers/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column, method, window_size }),
    });
    APP.outlierResult = res;
    renderOutlierModule(res);
    showToast(`Correção aplicada com sucesso (${method}).`, "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function undoOutlierCorrection() {
  const column = document.getElementById("outColSelect").value;
  if (!column) return;
  try {
    const res = await apiFetch(`/api/outliers/undo?column=${encodeURIComponent(column)}`, {
      method: "POST",
    });
    APP.outlierResult = res;
    renderOutlierModule(res);
    showToast("Última correção desfeita.", "warning");
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderOutlierModule(res) {
  if (!res) return;
  document.getElementById("outKpiCount").textContent = fmtInt(res.kpis.outliers_count);
  document.getElementById("outKpiNaInfo").textContent = `${res.kpis.na_count} valores NA ignorados`;
  document.getElementById("outKpiPercent").textContent = `${res.kpis.percent_outliers}%`;
  document.getElementById("outKpiImpact").textContent = `${res.kpis.impact_mean_pct}%`;
  document.getElementById("outKpiMethodInfo").textContent = res.has_correction
    ? `Método: ${res.correction_method} (${res.history_depth} etapa(s))`
    : "Pendente correção";

  // Stats table
  const tbody = document.querySelector("#tableOutlierStats tbody");
  tbody.innerHTML = (res.stats_table || [])
    .map(
      (r) => `
      <tr>
        <td><strong>${r.estatistica}</strong></td>
        <td>${fmtNum(r.original, 2)}</td>
        <td>${r.tratado !== null ? fmtNum(r.tratado, 2) : "—"}</td>
        <td>${r.variacao}</td>
      </tr>`
    )
    .join("");

  // Histogram
  const validPts = res.series.filter((p) => !p.is_na);
  const histTraces = [];
  if (res.has_correction) {
    histTraces.push({
      x: validPts.map((p) => p.original),
      type: "histogram",
      name: "Original",
      opacity: 0.55,
      marker: { color: "#6366f1" },
    });
    histTraces.push({
      x: validPts.map((p) => p.treated),
      type: "histogram",
      name: "Tratado",
      opacity: 0.65,
      marker: { color: "#10b981" },
    });
  } else {
    histTraces.push({
      x: validPts.filter((p) => !p.is_outlier).map((p) => p.original),
      type: "histogram",
      name: "Normal",
      opacity: 0.7,
      marker: { color: "#3b82f6" },
    });
    histTraces.push({
      x: validPts.filter((p) => p.is_outlier).map((p) => p.original),
      type: "histogram",
      name: "Outliers",
      opacity: 0.85,
      marker: { color: "#f43f5e" },
    });
  }
  Plotly.newPlot(
    "plotOutlierHist",
    histTraces,
    getPlotlyThemeLayout({
      barmode: "overlay",
      title: { text: `Distribuição de ${res.column}`, font: { size: 13 } },
      xaxis: { title: "Valor" },
      yaxis: { title: "Frequência" },
    }),
    { responsive: true, displayModeBar: false }
  );

  // Main Plot (Time series or Index subplot)
  const xKey = APP.outlierViewMode === "time" ? "date" : "index";
  const xLabel = APP.outlierViewMode === "time" ? "Data" : "Índice da Observação";
  const outPts = validPts.filter((p) => p.is_outlier);

  const mainTraces = [
    {
      x: validPts.map((p) => p[xKey]),
      y: validPts.map((p) => p.original),
      type: "scatter",
      mode: "lines+markers",
      name: "Série Original",
      line: { color: "#6366f1", width: 2 },
      marker: { size: 5 },
    },
  ];

  if (outPts.length > 0) {
    mainTraces.push({
      x: outPts.map((p) => p[xKey]),
      y: outPts.map((p) => p.original),
      type: "scatter",
      mode: "markers",
      name: "Outliers Detectados",
      marker: { color: "#f43f5e", size: 11, symbol: "diamond" },
    });
  }

  if (res.has_correction) {
    mainTraces.push({
      x: validPts.map((p) => p[xKey]),
      y: validPts.map((p) => p.treated),
      type: "scatter",
      mode: "lines+markers",
      name: `Série Corrigida (${res.correction_method})`,
      line: { color: "#10b981", width: 2.5 },
      marker: { size: 5 },
    });
  }

  Plotly.newPlot(
    "plotOutlierMain",
    mainTraces,
    getPlotlyThemeLayout({
      title: { text: `Série Temporal & Outliers — ${res.column}`, font: { size: 14 } },
      xaxis: { title: xLabel },
      yaxis: { title: res.column },
    }),
    { responsive: true }
  );
}

// ============================================================================
// MODULE 2: SEASONALITY TESTS, PARTICIPATION, XmR & STL ANOMALIES
// ============================================================================

async function runSeasonalityAnalysis() {
  const column = document.getElementById("seasColSelect").value;
  const start_year = parseInt(document.getElementById("seasStartYear").value, 10) || null;
  const end_year = parseInt(document.getElementById("seasEndYear").value, 10) || null;
  const stl_threshold_sigma = parseFloat(document.getElementById("seasStlSigma").value || "2.0");
  if (!column) return;

  try {
    const res = await apiFetch("/api/seasonality/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column, start_year, end_year, stl_threshold_sigma }),
    });
    APP.seasonalityResult = res;
    renderSeasonalityModule(res, column);
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderSeasonalityModule(res, column) {
  // KPIs
  const strength = res.stl.seasonality_strength;
  document.getElementById("seasKpiStrength").textContent =
    strength !== null ? `${(strength * 100).toFixed(1)}%` : "N/A";
  const st = res.seasonality_tests;
  document.getElementById("seasKpiTestResult").textContent = st.anova_p !== null
    ? `ANOVA p=${st.anova_p.toFixed(4)} (${st.has_significant_seasonality ? "Sazonal" : "Não Sazonal"})`
    : "Teste estatístico mensal";

  document.getElementById("seasKpiMeanX").textContent = fmtNum(res.xmr.media_x, 1);
  document.getElementById("seasKpiLimitsX").textContent = `LCL: ${fmtNum(res.xmr.lcl_x, 0)} | UCL: ${fmtNum(res.xmr.ucl_x, 0)}`;
  document.getElementById("seasKpiMeanMR").textContent = fmtNum(res.xmr.media_mr, 1);
  document.getElementById("seasKpiViolations").textContent = `${res.xmr.x_violations_count} pontos fora de X | ${res.xmr.mr_violations_count} fora de MR`;
  document.getElementById("seasKpiAnomalies").textContent = fmtInt(res.stl.anomalies_count);

  // 1. Participation Plot with Shaded Control Band
  const lims = res.limits_table || [];
  const xMonths = lims.map((r) => r.Mes);
  const partTraces = [
    {
      x: xMonths,
      y: lims.map((r) => r.Limite_Superior),
      type: "scatter",
      mode: "lines",
      line: { color: "rgba(99, 102, 241, 0)" },
      showlegend: false,
      hoverinfo: "skip",
    },
    {
      x: xMonths,
      y: lims.map((r) => r.Limite_Inferior),
      type: "scatter",
      mode: "lines",
      fill: "tonexty",
      fillcolor: "rgba(99, 102, 241, 0.20)",
      line: { color: "rgba(99, 102, 241, 0)" },
      name: `Banda Limite (±1 DP: ${res.start_year}–${res.end_year})`,
    },
    {
      x: xMonths,
      y: lims.map((r) => r.Media_Participacao),
      type: "scatter",
      mode: "lines",
      line: { color: "#6366f1", width: 2.5, dash: "dash" },
      name: "Média Participação",
    },
  ];

  (res.yearly_traces || []).forEach((yt) => {
    partTraces.push({
      x: yt.month_names,
      y: yt.participations,
      type: "scatter",
      mode: "lines+markers",
      name: String(yt.year),
      line: { width: 1.5 },
      marker: { size: 5 },
    });
  });

  Plotly.newPlot(
    "plotSeasParticipation",
    partTraces,
    getPlotlyThemeLayout({
      title: { text: `Participação Mensal e Limites (${res.start_year}–${res.end_year}) — ${column}`, font: { size: 14 } },
      xaxis: { title: "Mês" },
      yaxis: { title: "Participação no Ano", tickformat: ".1%" },
    }),
    { responsive: true }
  );

  // 2. Participation & Limits Table
  const limHead = document.querySelector("#tableSeasLimits thead");
  const limBody = document.querySelector("#tableSeasLimits tbody");
  const yrs = res.years_available || [];
  limHead.innerHTML = `
    <tr>
      <th>Mês</th>
      <th>Média Part.</th>
      <th>Desvio Padrão</th>
      <th>Lim. Inferior</th>
      <th>Lim. Superior</th>
      ${yrs.map((y) => `<th>${y}</th>`).join("")}
    </tr>`;
  limBody.innerHTML = lims
    .map(
      (r) => `
      <tr>
        <td><strong>${r.Mes}</strong></td>
        <td>${(r.Media_Participacao * 100).toFixed(2)}%</td>
        <td>${(r.Desvio_Padrao * 100).toFixed(2)}%</td>
        <td>${(r.Limite_Inferior * 100).toFixed(2)}%</td>
        <td>${(r.Limite_Superior * 100).toFixed(2)}%</td>
        ${yrs.map((y) => `<td>${r[String(y)] !== null && r[String(y)] !== undefined ? (r[String(y)] * 100).toFixed(2) + "%" : "—"}</td>`).join("")}
      </tr>`
    )
    .join("");

  // 3. XmR Subplot Chart (Top: X, Bottom: MR)
  const pts = res.xmr.points || [];
  const xDates = pts.map((p) => p.date);
  const xmrTraces = [
    // X Individual
    {
      x: xDates,
      y: pts.map((p) => p.x),
      type: "scatter",
      mode: "lines+markers",
      name: "X (Valor Individual)",
      line: { color: "#3b82f6", width: 2 },
      marker: {
        color: pts.map((p) => (p.x_violation ? "#f43f5e" : "#3b82f6")),
        size: pts.map((p) => (p.x_violation ? 9 : 5)),
      },
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: xDates,
      y: xDates.map(() => res.xmr.media_x),
      type: "scatter",
      mode: "lines",
      name: "Média X̄",
      line: { color: "#06b6d4", dash: "dash", width: 1.5 },
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: xDates,
      y: xDates.map(() => res.xmr.ucl_x),
      type: "scatter",
      mode: "lines",
      name: "UCL X",
      line: { color: "#f43f5e", dash: "dot", width: 1.5 },
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: xDates,
      y: xDates.map(() => res.xmr.lcl_x),
      type: "scatter",
      mode: "lines",
      name: "LCL X",
      line: { color: "#10b981", dash: "dot", width: 1.5 },
      xaxis: "x",
      yaxis: "y",
    },
    // MR Moving Range
    {
      x: xDates,
      y: pts.map((p) => p.mr),
      type: "scatter",
      mode: "lines+markers",
      name: "MR (Amplitude Móvel)",
      line: { color: "#a855f7", width: 2 },
      marker: {
        color: pts.map((p) => (p.mr_violation ? "#f43f5e" : "#a855f7")),
        size: pts.map((p) => (p.mr_violation ? 9 : 5)),
      },
      xaxis: "x2",
      yaxis: "y2",
    },
    {
      x: xDates,
      y: xDates.map(() => res.xmr.media_mr),
      type: "scatter",
      mode: "lines",
      name: "Média MR̄",
      line: { color: "#06b6d4", dash: "dash", width: 1.5 },
      xaxis: "x2",
      yaxis: "y2",
    },
    {
      x: xDates,
      y: xDates.map(() => res.xmr.ucl_mr),
      type: "scatter",
      mode: "lines",
      name: "UCL MR",
      line: { color: "#f43f5e", dash: "dot", width: 1.5 },
      xaxis: "x2",
      yaxis: "y2",
    },
  ];

  Plotly.newPlot(
    "plotSeasXmR",
    xmrTraces,
    getPlotlyThemeLayout({
      grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
      yaxis: { title: "X (Valores Individuais)" },
      yaxis2: { title: "MR (Amplitude Móvel)" },
      xaxis2: { title: "Data" },
    }),
    { responsive: true }
  );

  // 4. STL Residuals Anomaly Plot & Table
  const stlSeries = res.stl.series || [];
  const stlAnoms = res.stl.anomalies || [];
  const stlTraces = [
    {
      x: stlSeries.map((s) => s.date),
      y: stlSeries.map((s) => s.value),
      type: "scatter",
      mode: "lines",
      name: "Série Temporal",
      line: { color: "#3b82f6", width: 2 },
    },
    {
      x: stlSeries.map((s) => s.date),
      y: stlSeries.map((s) => s.trend),
      type: "scatter",
      mode: "lines",
      name: "Tendência STL",
      line: { color: "#10b981", width: 2, dash: "dot" },
    },
  ];
  if (stlAnoms.length > 0) {
    stlTraces.push({
      x: stlAnoms.map((a) => a.date),
      y: stlAnoms.map((a) => a.value),
      type: "scatter",
      mode: "markers",
      name: "Anomalias STL Residuals",
      marker: { color: "#f43f5e", size: 11, symbol: "x" },
    });
  }

  Plotly.newPlot(
    "plotSeasSTL",
    stlTraces,
    getPlotlyThemeLayout({
      title: { text: `Anomalias STL Residuals (±${res.stl.threshold_sigma}σ)`, font: { size: 13 } },
      xaxis: { title: "Data" },
      yaxis: { title: column },
    }),
    { responsive: true }
  );

  const anomBody = document.querySelector("#tableSeasAnomalies tbody");
  if (stlAnoms.length === 0) {
    anomBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">Nenhuma anomalia STL detectada com o limiar atual.</td></tr>`;
  } else {
    anomBody.innerHTML = stlAnoms
      .map(
        (a) => `
        <tr>
          <td><strong>${a.date}</strong></td>
          <td>${fmtNum(a.value, 2)}</td>
          <td>${fmtNum(a.trend, 2)}</td>
          <td>${fmtNum(a.seasonal, 2)}</td>
          <td class="text-neg">${fmtNum(a.remainder, 2)}</td>
        </tr>`
      )
      .join("");
  }
}

// ============================================================================
// MODULE 3: ADVANCED INTERACTIVE SERIES EDITOR (page 6)
// ============================================================================

async function initEditorSeries() {
  const column = document.getElementById("edColSelect").value;
  if (!column) return;
  try {
    const res = await apiFetch("/api/editor/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column }),
    });
    APP.editorResult = res;
    APP.editorSelectedIndices = [];
    updateEditorSelectionBadge();
    renderEditorModule(res);
  } catch (err) {
    showToast(err.message, "error");
  }
}

function updateEditorSelectionBadge() {
  const badge = document.getElementById("edSelectedBadge");
  const idxs = APP.editorSelectedIndices;
  if (!idxs || idxs.length === 0 || !APP.editorResult) {
    badge.innerHTML = "Nenhum ponto selecionado. Clique em um ponto ou use a seleção Laço/Caixa no gráfico.";
    return;
  }
  const series = APP.editorResult.series;
  if (idxs.length === 1) {
    const pt = series[idxs[0]];
    if (pt) {
      badge.innerHTML = `<strong>1 ponto selecionado:</strong> Data <code>${pt.Data}</code> • Valor Atual: <strong>${fmtNum(pt.Serie_Ajustada, 2)}</strong> (Original: ${fmtNum(pt.Valor_Original, 2)})`;
      const changeType = document.getElementById("edChangeType").value;
      if (changeType === "set") {
        document.getElementById("edChangeValue").value = pt.Serie_Ajustada;
      }
    }
  } else {
    badge.innerHTML = `<strong>${idxs.length} pontos selecionados</strong> (de <code>${series[idxs[0]]?.Data}</code> até <code>${series[idxs[idxs.length - 1]]?.Data}</code>)`;
  }
}

async function applyEditorModification() {
  const column = document.getElementById("edColSelect").value;
  const change_type = document.getElementById("edChangeType").value;
  const change_value = parseFloat(document.getElementById("edChangeValue").value);

  if (!APP.editorSelectedIndices.length) {
    showToast("Selecione pelo menos um ponto no gráfico primeiro.", "warning");
    return;
  }
  if (Number.isNaN(change_value)) {
    showToast("Informe um valor numérico para a alteração.", "warning");
    return;
  }

  try {
    const res = await apiFetch("/api/editor/modify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        column,
        indices: APP.editorSelectedIndices,
        change_type,
        change_value,
      }),
    });
    APP.editorResult = res;
    renderEditorModule(res);
    updateEditorSelectionBadge();
    showToast(`Alteração aplicada em ${APP.editorSelectedIndices.length} ponto(s).`, "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

async function undoEditorModification() {
  const column = document.getElementById("edColSelect").value;
  if (!column) return;
  try {
    const res = await apiFetch(`/api/editor/undo?column=${encodeURIComponent(column)}`, {
      method: "POST",
    });
    APP.editorResult = res;
    renderEditorModule(res);
    updateEditorSelectionBadge();
    showToast("Última alteração desfeita (Ctrl+Z).", "info");
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderEditorModule(res) {
  if (!res) return;
  document.getElementById("edKpiPeriods").textContent = fmtInt(res.kpis.num_periodos);
  document.getElementById("edKpiModCount").textContent = `${res.kpis.modified_points_count} ponto(s) editado(s)`;
  document.getElementById("edKpiMax").textContent = fmtNum(res.kpis.valor_max, 0);
  document.getElementById("edKpiMin").textContent = fmtNum(res.kpis.valor_min, 0);
  document.getElementById("edKpiMean").textContent = fmtNum(res.kpis.media_valor, 0);

  const series = res.series || [];
  const xDates = series.map((s) => s.Data);

  // Main interactive plot
  const selectedSet = new Set(APP.editorSelectedIndices);
  const markerColors = series.map((s, idx) =>
    selectedSet.has(idx) ? "#f59e0b" : s.is_modified ? "#10b981" : "#3b82f6"
  );
  const markerSizes = series.map((s, idx) =>
    selectedSet.has(idx) ? 11 : s.is_modified ? 9 : 6
  );

  const traces = [
    {
      x: xDates,
      y: series.map((s) => s.Valor_Original),
      type: "scatter",
      mode: "lines",
      name: "Série Original",
      line: { color: "rgba(148, 163, 184, 0.55)", width: 1.5, dash: "dot" },
      hoverinfo: "x+y+name",
    },
    {
      x: xDates,
      y: series.map((s) => s.Serie_Ajustada),
      type: "scatter",
      mode: "lines+markers",
      name: "Série Editada (Clique/Laço)",
      line: { color: "#6366f1", width: 2.3 },
      marker: { color: markerColors, size: markerSizes },
    },
  ];

  Plotly.newPlot(
    "plotEditorMain",
    traces,
    getPlotlyThemeLayout({
      dragmode: "lasso",
      hovermode: "closest",
      title: { text: `Editor Interativo — ${res.column}`, font: { size: 14 } },
      xaxis: { title: "Data" },
      yaxis: { title: res.column },
    }),
    { responsive: true }
  );

  const plotDiv = document.getElementById("plotEditorMain");
  plotDiv.on("plotly_click", (ev) => {
    if (ev && ev.points && ev.points.length > 0) {
      const pt = ev.points[0];
      APP.editorSelectedIndices = [pt.pointIndex];
      updateEditorSelectionBadge();
      renderEditorModule(APP.editorResult);
    }
  });

  plotDiv.on("plotly_selected", (ev) => {
    if (ev && ev.points && ev.points.length > 0) {
      const idxs = Array.from(new Set(ev.points.map((p) => p.pointIndex))).sort((a, b) => a - b);
      APP.editorSelectedIndices = idxs;
      updateEditorSelectionBadge();
      renderEditorModule(APP.editorResult);
    }
  });

  // Growth Rates Plot (MoM and YoY)
  Plotly.newPlot(
    "plotEditorGrowthRates",
    [
      {
        x: xDates,
        y: series.map((s) => s.Taxa_Crescimento_Ajustado),
        type: "scatter",
        mode: "lines",
        name: "Crescimento Mês/Mês (%)",
        line: { color: "#06b6d4", width: 1.8 },
      },
      {
        x: xDates,
        y: series.map((s) => s.Taxa_Crescimento_Mes_Ano_Ajustado),
        type: "scatter",
        mode: "lines",
        name: "Crescimento Mês/Ano YoY (%)",
        line: { color: "#10b981", width: 2 },
      },
    ],
    getPlotlyThemeLayout({
      xaxis: { title: "Data" },
      yaxis: { title: "Taxa (%)" },
    }),
    { responsive: true, displayModeBar: false }
  );

  // Addition Rates Plot (MoM and YoY Absolute Diff)
  Plotly.newPlot(
    "plotEditorAdditionRates",
    [
      {
        x: xDates,
        y: series.map((s) => s.Taxa_Acrescimo_Ajustado),
        type: "bar",
        name: "Acréscimo Mês/Mês",
        marker: { color: "#6366f1", opacity: 0.6 },
      },
      {
        x: xDates,
        y: series.map((s) => s.Taxa_Acrescimo_Mes_Ano_Ajustado),
        type: "scatter",
        mode: "lines",
        name: "Acréscimo Mês/Ano (YoY)",
        line: { color: "#f59e0b", width: 2 },
      },
    ],
    getPlotlyThemeLayout({
      xaxis: { title: "Data" },
      yaxis: { title: "Variação Absoluta" },
    }),
    { responsive: true, displayModeBar: false }
  );

  // Multi-resolution small plots
  const vis = res.visoes;
  if (vis) {
    renderMultiPeriodMiniPlot("plotEdMensal", vis.mensal.long, "Valores Mensais");
    renderMultiPeriodMiniPlot("plotEdTrimestral", vis.trimestral.long, "Valores Trimestrais");
    renderMultiPeriodMiniPlot("plotEdSemestral", vis.semestral.long, "Valores Semestrais");
  }

  renderEditorDynamicTable();
}

function renderMultiPeriodMiniPlot(divId, longData, title) {
  const byPeriod = {};
  (longData || []).forEach((r) => {
    if (!byPeriod[r.Periodo]) byPeriod[r.Periodo] = { x: [], y: [] };
    byPeriod[r.Periodo].x.push(r.Ano);
    byPeriod[r.Periodo].y.push(r.Valor);
  });

  const traces = Object.keys(byPeriod).map((p) => ({
    x: byPeriod[p].x,
    y: byPeriod[p].y,
    type: "scatter",
    mode: "lines+markers",
    name: p,
    line: { width: 1.8 },
    marker: { size: 5 },
  }));

  Plotly.newPlot(
    divId,
    traces,
    getPlotlyThemeLayout({
      margin: { t: 20, r: 16, b: 40, l: 50 },
      xaxis: { title: "Ano" },
      yaxis: { title: "Valor" },
    }),
    { responsive: true, displayModeBar: false }
  );
}

function renderEditorDynamicTable() {
  if (!APP.editorResult) return;
  const tab = APP.editorActiveTab;
  const thead = document.querySelector("#tableEditorDynamic thead");
  const tbody = document.querySelector("#tableEditorDynamic tbody");

  if (tab === "comp") {
    thead.innerHTML = `
      <tr>
        <th>Data</th>
        <th>Valor Original</th>
        <th>Série Ajustada</th>
        <th>Tx. Cresc. Mês Orig (%)</th>
        <th>Tx. Cresc. Mês Ajust (%)</th>
        <th>Tx. Cresc. YoY Orig (%)</th>
        <th>Tx. Cresc. YoY Ajust (%)</th>
      </tr>`;
    tbody.innerHTML = (APP.editorResult.series || [])
      .map(
        (r) => `
        <tr class="${r.is_modified ? "highlight-row" : ""}">
          <td><strong>${r.Data}</strong></td>
          <td>${fmtNum(r.Valor_Original, 2)}</td>
          <td><strong>${fmtNum(r.Serie_Ajustada, 2)}</strong></td>
          <td>${fmtNum(r.Taxa_Crescimento_Original, 2)}%</td>
          <td>${fmtNum(r.Taxa_Crescimento_Ajustado, 2)}%</td>
          <td>${fmtNum(r.Taxa_Crescimento_Mes_Ano_Original, 2)}%</td>
          <td>${fmtNum(r.Taxa_Crescimento_Mes_Ano_Ajustado, 2)}%</td>
        </tr>`
      )
      .join("");
  } else if (tab === "log") {
    const logs = APP.editorResult.modification_log || [];
    thead.innerHTML = `
      <tr>
        <th>Índice</th>
        <th>Data</th>
        <th>Valor Original</th>
        <th>Valor Anterior</th>
        <th>Série Ajustada</th>
        <th>Tipo de Alteração</th>
      </tr>`;
    if (logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">Nenhuma modificação realizada ainda.</td></tr>`;
    } else {
      tbody.innerHTML = logs
        .map(
          (l) => `
          <tr>
            <td>#${l.Indice}</td>
            <td><strong>${l.Data}</strong></td>
            <td>${fmtNum(l.Valor_Original, 2)}</td>
            <td>${fmtNum(l.Valor_Anterior, 2)}</td>
            <td class="text-pos">${fmtNum(l.Serie_Ajustada, 2)}</td>
            <td>${l.Tipo}</td>
          </tr>`
        )
        .join("");
    }
  } else {
    const viewData = APP.editorResult.visoes?.[tab]?.long || [];
    thead.innerHTML = `
      <tr>
        <th>Data</th>
        <th>Ano</th>
        <th>Período</th>
        <th>Valor Agregado</th>
        <th>Taxa Crescimento YoY (%)</th>
        <th>Variação Absoluta YoY</th>
      </tr>`;
    tbody.innerHTML = viewData
      .map(
        (r) => `
        <tr>
          <td>${r.Data}</td>
          <td><strong>${r.Ano}</strong></td>
          <td>${r.Periodo}</td>
          <td>${fmtNum(r.Valor, 0)}</td>
          <td class="${(r.Taxa_Crescimento || 0) >= 0 ? "text-pos" : "text-neg"}">${r.Taxa_Crescimento !== null ? fmtNum(r.Taxa_Crescimento, 2) + "%" : "—"}</td>
          <td class="${(r.Variacao_Valor || 0) >= 0 ? "text-pos" : "text-neg"}">${r.Variacao_Valor !== null ? fmtNum(r.Variacao_Valor, 0) : "—"}</td>
        </tr>`
      )
      .join("");
  }
}

// ============================================================================
// MODULE 4: ARIMA UNIVARIATE & MULTIVARIATE FORECASTING
// ============================================================================

async function runArimaForecast() {
  const target_var = document.getElementById("arimaTargetSelect").value;
  const exog_vars = Array.from(document.querySelectorAll('input[name="arimaExog"]:checked')).map((el) => el.value);
  const data_inicial = document.getElementById("arimaStartDate").value || null;
  const data_atual = document.getElementById("arimaEndDate").value || null;
  const end_predict = document.getElementById("arimaEndPredict").value || null;
  const n_teste = parseInt(document.getElementById("arimaNTest").value || "6", 10);
  const mode = document.getElementById("arimaModeSelect").value;
  const seasonal = document.getElementById("arimaSeasonalCheck").checked;
  const period = parseInt(document.getElementById("arimaPeriod").value || "12", 10);
  const num_models = parseInt(document.getElementById("arimaNumModels").value || "5", 10);

  const arima_params =
    mode === "manual"
      ? {
          p: parseInt(document.getElementById("arimaP").value || "1", 10),
          d: parseInt(document.getElementById("arimaD").value || "1", 10),
          q: parseInt(document.getElementById("arimaQ").value || "1", 10),
          P: parseInt(document.getElementById("arimaSP").value || "1", 10),
          D: parseInt(document.getElementById("arimaSD").value || "1", 10),
          Q: parseInt(document.getElementById("arimaSQ").value || "1", 10),
        }
      : null;

  try {
    const res = await apiFetch("/api/forecast/arima", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_var,
        exog_vars,
        data_inicial,
        data_atual,
        end_predict,
        n_teste,
        seasonal,
        period,
        num_models,
        manual_mode: mode === "manual",
        arima_params,
      }),
    });
    APP.arimaResult = res;
    renderArimaModule(res);
    populateGrowthTargets();
    showToast(`Análise ARIMA concluída em ${res.execution_time_sec}s (${res.errors.length} modelos ajustados).`, "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderArimaModule(res) {
  if (!res) return;

  // Collinearity warning banner
  const banner = document.getElementById("arimaCollinearityBanner");
  if (res.has_collinearity) {
    banner.style.display = "block";
    banner.innerHTML = `⚠️ <strong>Colinearidade detectada entre variáveis exógenas</strong> (${res.collinear_pairs.map((p) => `${p[0]} × ${p[1]}: r=${p[2]}`).join(", ")}). Apenas modelos univariados foram estimados.`;
  } else {
    banner.style.display = "none";
  }

  // Export select
  const expSel = document.getElementById("arimaExportModelSelect");
  const modelList = (res.errors || []).map((e) => e.model);
  expSel.innerHTML = `<option value="Todos">Todos os Modelos</option>` + modelList.map((m) => `<option value="${m}">${m}</option>`).join("");

  // Errors Table
  const errBody = document.querySelector("#tableArimaErrors tbody");
  errBody.innerHTML = (res.errors || [])
    .map(
      (e, idx) => `
      <tr class="${idx === 0 ? "highlight-row" : ""}">
        <td><strong>${e.model}</strong></td>
        <td><span class="badge ${e.type === "Multivariado" ? "badge-regular" : "badge-neutral"}">${e.type}</span></td>
        <td style="font-size:11.5px;color:var(--text-secondary)">${e.spec}</td>
        <td><strong>${fmtNum(e.RMSE, 2)}</strong></td>
        <td>${fmtNum(e.MAE, 2)}</td>
        <td>${fmtNum(e.MAPE, 2)}%</td>
      </tr>`
    )
    .join("");

  // Recommendation Table
  const recBody = document.querySelector("#tableArimaRec tbody");
  recBody.innerHTML = (res.recommendation?.ranking || [])
    .map(
      (r) => `
      <tr class="${r.is_best ? "highlight-row" : ""}">
        <td><strong>${r.modelo}</strong> ${r.is_best ? "⭐" : ""}</td>
        <td>${r.meses}</td>
        <td><strong>${fmtNum(r.score, 2)}</strong></td>
        <td>${fmtNum(r.desv_med, 2)}%</td>
        <td>${fmtNum(r.desv_max, 2)}%</td>
        <td>${getStatusBadgeHTML(r.status)}</td>
      </tr>`
    )
    .join("");

  // Plot Realized + ARIMA Models
  const comb = res.combined_series || [];
  const realizedPts = comb.filter((r) => r.Target !== null && r.Target !== undefined);

  const traces = [
    {
      x: realizedPts.map((r) => r.Data),
      y: realizedPts.map((r) => r.Target),
      type: "scatter",
      mode: "lines",
      name: "Realizado",
      line: { color: "#f8fafc", width: 3 },
    },
  ];

  const palette = ["#6366f1", "#10b981", "#06b6d4", "#f59e0b", "#ec4899", "#a855f7", "#3b82f6", "#14b8a6"];
  modelList.forEach((mName, idx) => {
    const mRows = res.forecasts_by_model[mName] || [];
    const col = palette[idx % palette.length];
    traces.push({
      x: mRows.map((r) => r.Data),
      y: mRows.map((r) => r.predict),
      type: "scatter",
      mode: "lines",
      name: mName,
      line: {
        color: col,
        width: idx === 0 ? 2.8 : 1.8,
        dash: mName.includes("multi") ? "solid" : "dot",
      },
    });

    // Confidence ribbon for the #1 best model
    if (idx === 0 && mRows.length > 0) {
      traces.push({
        x: mRows.map((r) => r.Data),
        y: mRows.map((r) => r.Hi95),
        type: "scatter",
        mode: "lines",
        line: { width: 0 },
        showlegend: false,
        hoverinfo: "skip",
      });
      traces.push({
        x: mRows.map((r) => r.Data),
        y: mRows.map((r) => r.Lo95),
        type: "scatter",
        mode: "lines",
        fill: "tonexty",
        fillcolor: "rgba(99, 102, 241, 0.14)",
        line: { width: 0 },
        name: `IC 95% (${mName})`,
      });
    }
  });

  const shapes = [];
  if (res.periods.n_test > 0 && res.periods.train_end) {
    shapes.push({
      type: "line",
      x0: res.periods.train_end,
      x1: res.periods.train_end,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "#f43f5e", width: 1.8, dash: "dash" },
    });
  }
  if (res.periods.test_end) {
    shapes.push({
      type: "line",
      x0: res.periods.test_end,
      x1: res.periods.test_end,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "#06b6d4", width: 1.8, dash: "dash" },
    });
  }

  Plotly.newPlot(
    "plotArimaForecast",
    traces,
    getPlotlyThemeLayout({
      title: { text: `Previsões ARIMA — ${res.target_var}`, font: { size: 14 } },
      xaxis: { title: "Data" },
      yaxis: { title: res.target_var },
      shapes,
    }),
    { responsive: true }
  );
}

// ============================================================================
// MODULE 5: AUTOTS UNIVARIATE MULTI-MODEL BENCHMARK
// ============================================================================

async function runAutoTSForecast() {
  const variable = document.getElementById("autotsVarSelect").value;
  const start_date = document.getElementById("autotsStartDate").value || null;
  const end_date = document.getElementById("autotsEndDate").value || null;
  const end_predict = document.getElementById("autotsEndPredict").value || null;
  const selected_models = Array.from(document.querySelectorAll("#autotsModelsGroup input:checked")).map((el) => el.value);
  const periodo_maximo = parseInt(document.getElementById("autotsMaxMonths").value || "12", 10);
  const peso_recente = parseFloat(document.getElementById("autotsWeightRecent").value || "2.0");
  const peso_desvio_max = parseFloat(document.getElementById("autotsWeightMax").value || "0.4");

  if (!selected_models.length) {
    showToast("Selecione pelo menos um modelo AutoTS.", "warning");
    return;
  }

  try {
    const res = await apiFetch("/api/forecast/autots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variable,
        start_date,
        end_date,
        end_predict,
        selected_models,
        periodo_maximo,
        peso_recente,
        peso_desvio_max,
      }),
    });
    APP.autotsResult = res;
    renderAutoTSModule(res);
    populateGrowthTargets();
    showToast(`Benchmark AutoTS concluído em ${res.execution_time_sec}s! Melhor modelo: ${res.best_model.toUpperCase()}`, "success");
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderAutoTSModule(res) {
  if (!res) return;

  // Populate focus model select
  const focusSel = document.getElementById("autotsFocusModel");
  const prevFocus = focusSel.value;
  focusSel.innerHTML =
    `<option value="all">Mostrar Todos os Modelos (${res.models_available.length})</option>` +
    res.models_available.map((m) => `<option value="${m}">${m.toUpperCase()}${m === res.best_model ? " ⭐ (Melhor)" : ""}</option>`).join("");
  if (prevFocus && (prevFocus === "all" || res.models_available.includes(prevFocus))) {
    focusSel.value = prevFocus;
  }

  // Errors Table
  const errBody = document.querySelector("#tableAutoTSErrors tbody");
  errBody.innerHTML = (res.errors || [])
    .map(
      (e, idx) => `
      <tr class="${idx === 0 ? "highlight-row" : ""}">
        <td><strong>${e.model.toUpperCase()}</strong></td>
        <td><strong>${fmtNum(e.RMSE, 2)}</strong></td>
        <td>${fmtNum(e.MAE, 2)}</td>
        <td>${fmtNum(e.MAPE, 2)}%</td>
      </tr>`
    )
    .join("");

  // Recommendation Table
  const recBody = document.querySelector("#tableAutoTSRec tbody");
  recBody.innerHTML = (res.recommendation?.ranking || [])
    .map(
      (r) => `
      <tr class="${r.is_best ? "highlight-row" : ""}">
        <td><strong>${r.modelo.toUpperCase()}</strong> ${r.is_best ? "⭐" : ""}</td>
        <td>${r.meses}</td>
        <td><strong>${fmtNum(r.score, 2)}</strong></td>
        <td>${fmtNum(r.desv_med, 2)}%</td>
        <td>${fmtNum(r.desv_max, 2)}%</td>
        <td>${getStatusBadgeHTML(r.status)}</td>
      </tr>`
    )
    .join("");

  renderAutoTSPlot();
}

function renderAutoTSPlot() {
  const res = APP.autotsResult;
  if (!res) return;
  const focus = document.getElementById("autotsFocusModel").value;
  const showCI = document.getElementById("autotsShowIntervals").checked;

  const tl = res.timeline || [];
  const histRows = tl.filter((r) => r.Period === "Realizado");
  const projRows = tl.filter((r) => r.Period === "Projeção");

  const traces = [
    {
      x: histRows.map((r) => r.Data),
      y: histRows.map((r) => r.realizado),
      type: "scatter",
      mode: "lines",
      name: "Realizado",
      line: { color: "#f8fafc", width: 2.8 },
    },
  ];

  const modelsToPlot = focus === "all" ? res.models_available : [focus];
  modelsToPlot.forEach((mName) => {
    const col = MODEL_COLORS[mName] || "#6366f1";
    // Future mean projection
    traces.push({
      x: projRows.map((r) => r.Data),
      y: projRows.map((r) => r[mName]),
      type: "scatter",
      mode: "lines",
      name: `${mName.toUpperCase()} (Média)`,
      line: { color: col, width: 2.4 },
    });

    if (showCI && (focus !== "all" || mName === res.best_model)) {
      traces.push({
        x: projRows.map((r) => r.Data),
        y: projRows.map((r) => r[`${mName}_lower`]),
        type: "scatter",
        mode: "lines",
        name: `${mName.toUpperCase()} (Lower 95%)`,
        line: { color: col, width: 1.2, dash: "dash" },
      });
      traces.push({
        x: projRows.map((r) => r.Data),
        y: projRows.map((r) => r[`${mName}_upper`]),
        type: "scatter",
        mode: "lines",
        name: `${mName.toUpperCase()} (Upper 95%)`,
        line: { color: col, width: 1.2, dash: "dash" },
      });
    }
  });

  Plotly.newPlot(
    "plotAutoTSForecast",
    traces,
    getPlotlyThemeLayout({
      title: { text: `Projeções AutoTS — ${res.variable}`, font: { size: 14 } },
      xaxis: { title: "Data" },
      yaxis: { title: res.variable },
    }),
    { responsive: true }
  );
}

// ============================================================================
// MODULE 6: MULTI-RESOLUTION GROWTH & VARIATION ANALYTICS
// ============================================================================

function populateGrowthTargets() {
  const source = document.getElementById("growthSourceSelect").value;
  const targetSel = document.getElementById("growthTargetSelect");

  if (source === "arima" && APP.arimaResult) {
    const models = (APP.arimaResult.errors || []).map((e) => e.model);
    targetSel.innerHTML = models.map((m) => `<option value="${m}">${m}</option>`).join("");
  } else if (source === "autots" && APP.autotsResult) {
    const models = APP.autotsResult.models_available || [];
    targetSel.innerHTML = models.map((m) => `<option value="${m}">${m.toUpperCase()}</option>`).join("");
  } else if (APP.dataset) {
    const cols = APP.dataset.numeric_columns || [];
    targetSel.innerHTML = cols.map((c) => `<option value="${c}">${c}</option>`).join("");
  }
}

async function runGrowthAnalysis() {
  const source = document.getElementById("growthSourceSelect").value;
  const model_or_col = document.getElementById("growthTargetSelect").value;
  if (!model_or_col) return;

  try {
    const res = await apiFetch("/api/growth/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, model_or_col }),
    });
    APP.growthResult = res;

    // Set year filter bounds
    const annualRows = res.views?.anual?.wide_valores || [];
    if (annualRows.length > 0) {
      const yrs = annualRows.map((r) => r.Ano);
      document.getElementById("growthYearStart").value = Math.min(...yrs);
      document.getElementById("growthYearEnd").value = Math.max(...yrs);
    }

    renderGrowthModule();
  } catch (err) {
    showToast(err.message, "error");
  }
}

function renderGrowthModule() {
  if (!APP.growthResult) return;
  const period = document.getElementById("growthPeriodSelect").value;
  const metric = document.getElementById("growthMetricSelect").value;
  const style = document.getElementById("growthChartStyle").value;
  const yStart = parseInt(document.getElementById("growthYearStart").value || "1900", 10);
  const yEnd = parseInt(document.getElementById("growthYearEnd").value || "2100", 10);

  const pView = APP.growthResult.views?.[period];
  if (!pView) return;

  const longRows = (pView.long || []).filter((r) => r.Ano >= yStart && r.Ano <= yEnd);
  const yField =
    metric === "valores" ? "Valor" : metric === "taxas" ? "Taxa_Crescimento" : "Variacao_Valor";
  const yLabel =
    metric === "valores"
      ? "Valor Absoluto"
      : metric === "taxas"
      ? "Taxa de Crescimento YoY (%)"
      : "Variação Absoluta YoY";

  const xVals = longRows.map((r) => (period === "anual" ? String(r.Ano) : `${r.Ano}-${r.Periodo}`));
  const yVals = longRows.map((r) => r[yField]);
  const barColors = yVals.map((v) => ((v || 0) >= 0 ? "#10b981" : "#f43f5e"));

  const traces = [];
  if (style === "lines") {
    traces.push({
      x: xVals,
      y: yVals,
      type: "scatter",
      mode: "lines+markers",
      name: yLabel,
      line: { color: "#6366f1", width: 2.5 },
      marker: { size: 7, color: "#06b6d4" },
    });
  } else if (style === "bars") {
    traces.push({
      x: xVals,
      y: yVals,
      type: "bar",
      name: yLabel,
      marker: { color: barColors },
    });
  } else {
    traces.push({
      x: xVals,
      y: yVals,
      type: "bar",
      name: yLabel,
      marker: { color: barColors, opacity: 0.72 },
    });
    traces.push({
      x: xVals,
      y: yVals,
      type: "scatter",
      mode: "lines+markers",
      name: "Tendência",
      line: { color: "#6366f1", width: 2.2 },
      marker: { size: 6 },
    });
  }

  Plotly.newPlot(
    "plotGrowthMain",
    traces,
    getPlotlyThemeLayout({
      title: {
        text: `${yLabel} (${period.toUpperCase()}) — ${APP.growthResult.model_or_col}`,
        font: { size: 14 },
      },
      xaxis: { title: "Período", tickangle: -35 },
      yaxis: { title: yLabel },
    }),
    { responsive: true }
  );

  // Render Wide Matrix Table
  const rows = (pView[APP.growthActiveTab] || []).filter((r) => r.Ano >= yStart && r.Ano <= yEnd);
  const thead = document.querySelector("#tableGrowthDynamic thead");
  const tbody = document.querySelector("#tableGrowthDynamic tbody");

  if (!rows.length) {
    thead.innerHTML = "";
    tbody.innerHTML = `<tr><td style="text-align:center;color:var(--text-muted)">Sem dados para o período selecionado.</td></tr>`;
    return;
  }

  const cols = Object.keys(rows[0]);
  thead.innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const isRate = APP.growthActiveTab === "wide_taxas";
  const isVar = APP.growthActiveTab === "wide_variacoes";

  tbody.innerHTML = rows
    .map(
      (r) =>
        `<tr>${cols
          .map((c) => {
            if (c === "Ano") return `<td><strong>${r[c]}</strong></td>`;
            const val = r[c];
            if (val === null || val === undefined) return `<td style="color:var(--text-muted)">—</td>`;
            if (isRate) {
              const cls = val >= 0 ? "text-pos" : "text-neg";
              return `<td class="${cls}">${fmtNum(val, 2)}%</td>`;
            }
            if (isVar) {
              const cls = val >= 0 ? "text-pos" : "text-neg";
              return `<td class="${cls}">${fmtNum(val, 0)}</td>`;
            }
            return `<td>${fmtNum(val, 0)}</td>`;
          })
          .join("")}</tr>`
    )
    .join("");
}

// ============================================================================
// EVENT LISTENERS & INITIALIZATION
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Navigation
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  // Mobile sidebar toggle
  document.getElementById("mobileMenuBtn").addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
  });

  // Theme Toggle
  document.getElementById("themeToggleBtn").addEventListener("click", () => {
    const html = document.documentElement;
    const next = html.getAttribute("data-theme") === "light" ? "dark" : "light";
    html.setAttribute("data-theme", next);
    document.getElementById("themeIcon").textContent = next === "light" ? "🌙 Tema Escuro" : "☀️ Tema Claro";
    // Re-render active plots with new theme colors
    if (APP.outlierResult) renderOutlierModule(APP.outlierResult);
    if (APP.seasonalityResult) renderSeasonalityModule(APP.seasonalityResult, document.getElementById("seasColSelect").value);
    if (APP.editorResult) renderEditorModule(APP.editorResult);
    if (APP.arimaResult) renderArimaModule(APP.arimaResult);
    if (APP.autotsResult) renderAutoTSPlot();
    if (APP.growthResult) renderGrowthModule();
  });

  // File Dropzone & Upload
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      await handleFileUpload(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length > 0) {
      await handleFileUpload(fileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const info = await apiFetch("/api/dataset/upload", {
        method: "POST",
        body: formData,
      });
      applyDatasetMetadata(info);
      showToast(`Arquivo '${file.name}' carregado com sucesso!`, "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  }

  document.getElementById("btnLoadSample").addEventListener("click", async () => {
    try {
      const info = await apiFetch("/api/dataset/load-sample", { method: "POST" });
      applyDatasetMetadata(info);
      showToast("Base de demonstração ei_total_(PA).xlsx recarregada!", "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  document.getElementById("btnProcessDataset").addEventListener("click", async () => {
    const sheet_name = document.getElementById("cfgSheetSelect").value;
    const date_column = document.getElementById("cfgDateColSelect").value;
    const date_format = document.getElementById("cfgDateFormatSelect").value;
    const csv_separator = document.getElementById("cfgCsvSepSelect").value || null;
    try {
      const info = await apiFetch("/api/dataset/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sheet_name, date_column, date_format, csv_separator }),
      });
      applyDatasetMetadata(info);
      showToast("Configuração de dados processada com sucesso!", "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Module 1: Outliers events
  document.getElementById("outIdentMethod").addEventListener("change", updateOutlierMethodUI);
  document.getElementById("outThresholdSlider").addEventListener("input", (e) => {
    document.getElementById("outThresholdVal").textContent = e.target.value;
  });
  document.getElementById("outCorrectMethod").addEventListener("change", (e) => {
    document.getElementById("outWindowGroup").style.display = e.target.value === "moving_average" ? "block" : "none";
  });
  document.getElementById("btnOutWinDec").addEventListener("click", () => {
    const inp = document.getElementById("outWindowInput");
    inp.value = Math.max(2, parseInt(inp.value || "5", 10) - 1);
  });
  document.getElementById("btnOutWinInc").addEventListener("click", () => {
    const inp = document.getElementById("outWindowInput");
    inp.value = Math.min(50, parseInt(inp.value || "5", 10) + 1);
  });
  document.getElementById("outColSelect").addEventListener("change", runOutlierIdentification);
  document.getElementById("btnOutIdentify").addEventListener("click", runOutlierIdentification);
  document.getElementById("btnOutCorrect").addEventListener("click", runOutlierCorrection);
  document.getElementById("btnOutUndo").addEventListener("click", undoOutlierCorrection);
  document.getElementById("btnOutCommit").addEventListener("click", async () => {
    const col = document.getElementById("outColSelect").value;
    try {
      const res = await apiFetch(`/api/outliers/commit?column=${encodeURIComponent(col)}`, { method: "POST" });
      applyDatasetMetadata(res.dataset);
      showToast(`Nova variável '${res.new_column}' adicionada ao workspace!`, "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  });
  document.getElementById("btnOutExport").addEventListener("click", () => {
    const col = document.getElementById("outColSelect").value;
    window.location.href = `/api/outliers/export?column=${encodeURIComponent(col)}`;
  });
  document.getElementById("btnOutViewTime").addEventListener("click", (e) => {
    APP.outlierViewMode = "time";
    e.target.classList.add("active");
    document.getElementById("btnOutViewIndex").classList.remove("active");
    renderOutlierModule(APP.outlierResult);
  });
  document.getElementById("btnOutViewIndex").addEventListener("click", (e) => {
    APP.outlierViewMode = "index";
    e.target.classList.add("active");
    document.getElementById("btnOutViewTime").classList.remove("active");
    renderOutlierModule(APP.outlierResult);
  });

  // Module 2: Seasonality & XmR events
  document.getElementById("btnRunSeasonality").addEventListener("click", runSeasonalityAnalysis);
  document.getElementById("seasColSelect").addEventListener("change", runSeasonalityAnalysis);
  document.getElementById("btnExportSeasonality").addEventListener("click", () => {
    const col = document.getElementById("seasColSelect").value;
    const sy = document.getElementById("seasStartYear").value;
    const ey = document.getElementById("seasEndYear").value;
    window.location.href = `/api/seasonality/export?column=${encodeURIComponent(col)}&start_year=${sy}&end_year=${ey}`;
  });

  // Module 3: Advanced Editor events
  document.getElementById("edColSelect").addEventListener("change", initEditorSeries);
  document.getElementById("btnEdApply").addEventListener("click", applyEditorModification);
  document.getElementById("btnEdUndo").addEventListener("click", undoEditorModification);
  document.getElementById("btnEdClearSel").addEventListener("click", () => {
    APP.editorSelectedIndices = [];
    updateEditorSelectionBadge();
    renderEditorModule(APP.editorResult);
  });
  document.getElementById("btnEdReset").addEventListener("click", async () => {
    const col = document.getElementById("edColSelect").value;
    const res = await apiFetch(`/api/editor/reset?column=${encodeURIComponent(col)}`, { method: "POST" });
    APP.editorResult = res;
    APP.editorSelectedIndices = [];
    updateEditorSelectionBadge();
    renderEditorModule(res);
    showToast("Série restaurada para os valores originais.", "info");
  });
  document.getElementById("btnEdCommit").addEventListener("click", async () => {
    const col = document.getElementById("edColSelect").value;
    try {
      const res = await apiFetch(`/api/editor/commit?column=${encodeURIComponent(col)}`, { method: "POST" });
      applyDatasetMetadata(res.dataset);
      showToast(`Série salva no workspace como '${res.new_column}'!`, "success");
    } catch (err) {
      showToast(err.message, "error");
    }
  });
  document.getElementById("btnEdExport").addEventListener("click", () => {
    const col = document.getElementById("edColSelect").value;
    window.location.href = `/api/editor/export?column=${encodeURIComponent(col)}`;
  });
  document.querySelectorAll("#edTablesTabBar .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#edTablesTabBar .tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      APP.editorActiveTab = btn.dataset.edtab;
      renderEditorDynamicTable();
    });
  });

  // Global Ctrl+Z handler when on Advanced Editor or Outliers tab
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      if (document.getElementById("view-editor").classList.contains("active")) {
        e.preventDefault();
        undoEditorModification();
      } else if (document.getElementById("view-outliers").classList.contains("active")) {
        e.preventDefault();
        undoOutlierCorrection();
      }
    }
  });

  // Module 4: ARIMA events
  document.getElementById("arimaTargetSelect").addEventListener("change", (e) => {
    syncForecastDatesForVariable(e.target.value);
    populateArimaExogCheckboxes();
  });
  document.getElementById("arimaModeSelect").addEventListener("change", (e) => {
    const isManual = e.target.value === "manual";
    document.getElementById("arimaAutoBox").style.display = isManual ? "none" : "block";
    document.getElementById("arimaManualBox").style.display = isManual ? "block" : "none";
  });
  document.getElementById("btnRunArima").addEventListener("click", runArimaForecast);
  document.getElementById("btnExportArima").addEventListener("click", () => {
    const mod = document.getElementById("arimaExportModelSelect").value;
    window.location.href = `/api/forecast/export-arima?model=${encodeURIComponent(mod)}`;
  });
  document.getElementById("btnArimaToGrowth").addEventListener("click", () => {
    if (!APP.arimaResult) return;
    document.getElementById("growthSourceSelect").value = "arima";
    populateGrowthTargets();
    switchView("view-growth");
    runGrowthAnalysis();
  });

  // Module 5: AutoTS events
  document.getElementById("autotsVarSelect").addEventListener("change", (e) => {
    syncForecastDatesForVariable(e.target.value);
  });
  document.getElementById("btnRunAutoTS").addEventListener("click", runAutoTSForecast);
  document.getElementById("autotsFocusModel").addEventListener("change", renderAutoTSPlot);
  document.getElementById("autotsShowIntervals").addEventListener("change", renderAutoTSPlot);
  document.getElementById("btnExportAutoTSAll").addEventListener("click", () => {
    window.location.href = `/api/forecast/export-autots?model=Todos`;
  });
  document.getElementById("btnExportAutoTSSel").addEventListener("click", () => {
    const focus = document.getElementById("autotsFocusModel").value;
    const mod = focus === "all" ? (APP.autotsResult?.best_model || "Todos") : focus;
    window.location.href = `/api/forecast/export-autots?model=${encodeURIComponent(mod)}`;
  });
  document.getElementById("btnAutoTSToGrowth").addEventListener("click", () => {
    if (!APP.autotsResult) return;
    document.getElementById("growthSourceSelect").value = "autots";
    populateGrowthTargets();
    switchView("view-growth");
    runGrowthAnalysis();
  });

  // Module 6: Growth Analytics events
  document.getElementById("growthSourceSelect").addEventListener("change", () => {
    populateGrowthTargets();
    runGrowthAnalysis();
  });
  document.getElementById("growthTargetSelect").addEventListener("change", runGrowthAnalysis);
  document.getElementById("growthPeriodSelect").addEventListener("change", renderGrowthModule);
  document.getElementById("growthMetricSelect").addEventListener("change", renderGrowthModule);
  document.getElementById("growthChartStyle").addEventListener("change", renderGrowthModule);
  document.getElementById("growthYearStart").addEventListener("change", renderGrowthModule);
  document.getElementById("growthYearEnd").addEventListener("change", renderGrowthModule);
  document.getElementById("btnRunGrowth").addEventListener("click", runGrowthAnalysis);
  document.getElementById("btnExportGrowth").addEventListener("click", () => {
    const src = document.getElementById("growthSourceSelect").value;
    const target = document.getElementById("growthTargetSelect").value;
    window.location.href = `/api/growth/export?source=${encodeURIComponent(src)}&model_or_col=${encodeURIComponent(target)}`;
  });
  document.querySelectorAll("#growthTablesTabBar .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#growthTablesTabBar .tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      APP.growthActiveTab = btn.dataset.gtab;
      renderGrowthModule();
    });
  });

  // Initial Load
  loadDatasetInfo();
});
