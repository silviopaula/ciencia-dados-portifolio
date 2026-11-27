/**
 * App Forecasts — Modern JavaScript ES6 Frontend Application
 * Zero Python in Frontend — Pure REST API + Plotly.js Client
 */

const APP = {
  state: null,
  tableData: { columns: [], rows: [], total_rows: 0 },
  tablePage: 1,
  caretCatalog: [],
  caretResults: null,
  arimaResults: null,
  autotsResults: null,
  growthResults: null,
  activeGrowthTableTab: 'valores',
};

const PALETTE = [
  '#2563eb', '#dc2626', '#16a34a', '#d97706', '#9333ea',
  '#0891b2', '#db2777', '#4f46e5', '#65a30d', '#ea580c',
];

// ============================================================================
// Initialization & Theme
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.body.classList.add('dark-theme');
    updateThemeIcon();
  }
  await Promise.all([
    loadSampleFilesList(),
    loadCaretModelCatalog(),
    refreshGlobalState(),
  ]);
});

function toggleTheme() {
  document.body.classList.toggle('dark-theme');
  updateThemeIcon();
  // Re-render active Plotly charts for dark/light theme colors
  const exploreEl = document.getElementById('explore-plotly-chart');
  if (exploreEl && exploreEl.data && exploreEl.data.length > 0) {
    generateExploratoryPlot();
  }
  if (APP.caretResults) {
    renderCaretMainPlot();
    renderCaretImportanceAndElasticity();
  }
  if (APP.arimaResults) renderArimaMainPlot();
  if (APP.autotsResults) renderAutoTSForecastPlot();
  if (APP.growthResults) renderAutoTSGrowthChart();
}

function updateThemeIcon() {
  const btn = document.getElementById('theme-toggle-btn');
  const isDark = document.body.classList.contains('dark-theme');
  btn.innerHTML = isDark ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
}

function getPlotlyThemeLayout() {
  const isDark = document.body.classList.contains('dark-theme');
  return {
    paper_bgcolor: isDark ? '#111827' : '#ffffff',
    plot_bgcolor: isDark ? '#111827' : '#ffffff',
    font: { color: isDark ? '#e2e8f0' : '#1e293b', family: 'Inter, sans-serif' },
    xaxis: { gridcolor: isDark ? '#1e293b' : '#f1f5f9', zerolinecolor: isDark ? '#334155' : '#cbd5e1' },
    yaxis: { gridcolor: isDark ? '#1e293b' : '#f1f5f9', zerolinecolor: isDark ? '#334155' : '#cbd5e1' },
  };
}

function showToast(message, type = 'info', duration = 4500) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const iconMap = {
    success: 'fa-circle-check',
    warning: 'fa-triangle-exclamation',
    error: 'fa-circle-xmark',
    info: 'fa-circle-info',
  };
  el.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;">
      <i class="fa-solid ${iconMap[type] || iconMap.info}"></i>
      <span>${message}</span>
    </div>
    <button style="background:none;border:none;color:inherit;cursor:pointer;font-size:1rem;" onclick="this.parentElement.remove()">&times;</button>
  `;
  container.appendChild(el);
  if (duration > 0) {
    setTimeout(() => {
      if (el.parentElement) el.remove();
    }, duration);
  }
}

// ============================================================================
// Navigation & Sub-tabs
// ============================================================================

const TAB_TITLES = {
  introducao: 'Introdução',
  importacao: 'Importação e Preparação de Dados',
  forecast_caret: 'Projeções Machine Learning (PyCaret / Caret Recipes)',
  forecast_arima: 'Projeções ARIMA / SARIMAX',
  forecast_autots: 'Projeções AutoTS & Análise de Crescimento',
};

function switchMainTab(tabId) {
  document.querySelectorAll('.nav-btn').forEach((b) => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
  });
  document.querySelectorAll('.tab-pane').forEach((p) => {
    p.classList.toggle('active', p.id === `tab-${tabId}`);
  });
  document.getElementById('topbar-title').textContent = TAB_TITLES[tabId] || 'App Forecasts';
  window.dispatchEvent(new Event('resize'));
}

function navigateToForecastTab(tabId) {
  if (!APP.state || !APP.state.has_data) {
    showToast('Importe um conjunto de dados primeiro (ou clique em "Carregar Exemplo Rápido").', 'warning');
    switchMainTab('importacao');
    return;
  }
  switchMainTab(tabId);
}

function switchSubTab(groupPrefix, subId, btnEl) {
  const parent = btnEl.parentElement;
  parent.querySelectorAll('.subtab-btn').forEach((b) => b.classList.remove('active'));
  btnEl.classList.add('active');

  document.querySelectorAll(`[id^="${groupPrefix}-"]`).forEach((el) => {
    el.style.display = el.id === `${groupPrefix}-${subId}` ? 'block' : 'none';
  });
  window.dispatchEvent(new Event('resize'));
}

function toggleAllCheckboxes(containerId, checked) {
  const box = document.getElementById(containerId);
  if (!box) return;
  box.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    if (cb.offsetParent !== null || containerId !== 'caret-models-list') {
      cb.checked = checked;
    }
  });
}

function getCheckedValues(containerId) {
  const box = document.getElementById(containerId);
  if (!box) return [];
  return Array.from(box.querySelectorAll('input[type="checkbox"]:checked')).map((cb) => cb.value);
}

function setImportStepIndicator(activeStep) {
  document.querySelectorAll('#import-steps-bar .step-item').forEach((el) => {
    const stepNum = parseInt(el.getAttribute('data-step'), 10);
    el.classList.remove('active', 'done');
    if (stepNum < activeStep) el.classList.add('done');
    else if (stepNum === activeStep) el.classList.add('active');
  });
}

// ============================================================================
// Global State & Sample Loader
// ============================================================================

async function loadSampleFilesList() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    const sel = document.getElementById('sample-file-select');
    sel.innerHTML = '<option value="">Selecione um arquivo de exemplo...</option>';
    (data.samples || []).forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s.filename;
      opt.textContent = `${s.filename} (${s.type.toUpperCase()} — ${s.size_kb} KB)`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('Erro ao listar exemplos:', e);
  }
}

async function quickLoadDefaultSample(goToCaret = false) {
  try {
    showToast('Carregando dados de exemplo (dados_virgula.csv)...', 'info', 2500);
    const fd = new FormData();
    fd.append('filename', 'dados_virgula.csv');
    fd.append('tipo_arquivo', 'csv');
    const res = await fetch('/api/import/sample', { method: 'POST', body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || 'Falha ao carregar exemplo');
    const payload = await res.json();
    clearCachedModelResults();
    applyStateAndPreview(payload.state, payload.preview);
    showToast('Dados de exemplo carregados e coluna "date_ok" validada!', 'success');
    if (goToCaret) {
      switchMainTab('forecast_caret');
    } else {
      switchMainTab('importacao');
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function refreshGlobalState() {
  try {
    const res = await fetch('/api/state');
    const st = await res.json();
    let preview = null;
    if (st.has_data) {
      const tRes = await fetch('/api/data/table?limit=1000');
      preview = await tRes.json();
    }
    applyStateAndPreview(st, preview);
  } catch (e) {
    console.error('Erro ao atualizar estado:', e);
  }
}

function applyStateAndPreview(st, preview) {
  APP.state = st;
  if (preview) {
    APP.tableData = preview;
    APP.tablePage = 1;
  }

  // Update Topbar badge
  const badge = document.getElementById('global-dataset-badge');
  if (st.has_data) {
    badge.className = 'dataset-pill loaded';
    badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>${st.filename || 'Dados Ativos'}: <strong>${st.total_rows}</strong> linhas &times; <strong>${st.total_cols}</strong> colunas</span>`;
  } else {
    badge.className = 'dataset-pill';
    badge.innerHTML = `<i class="fa-solid fa-database"></i> <span>Nenhum dado carregado</span>`;
  }

  // Update Operation Log & Seasonality Report
  const logBox = document.getElementById('operation-log-box');
  logBox.textContent = (st.logs && st.logs.length > 0) ? st.logs.join('\n') : 'Nenhuma operação nesta sessão.';
  if (st.seasonality_report) {
    document.getElementById('seasonality-report-box').textContent = st.seasonality_report;
  }

  // Update Step indicator & Enable/Disable buttons
  const dateSel = document.getElementById('import-date-col-select');
  dateSel.disabled = !st.has_data;
  document.getElementById('btn-convert-date-ok').disabled = !st.has_data;
  document.getElementById('btn-apply-dummies').disabled = !st.date_processing_complete;
  document.getElementById('btn-apply-transform').disabled = !st.date_processing_complete;
  document.getElementById('btn-extract-seasonality').disabled = !st.date_processing_complete;
  document.getElementById('btn-export-data').disabled = !st.has_data;
  document.getElementById('btn-generate-explore-plot').disabled = !st.date_processing_complete;

  if (!st.has_data) {
    setImportStepIndicator(1);
    dateSel.innerHTML = '<option value="">Importe dados primeiro...</option>';
    renderImportDataTable();
    return;
  }

  if (!st.date_processing_complete) {
    setImportStepIndicator(2);
  } else {
    setImportStepIndicator(3);
  }

  // Populate Date column selector in Import tab
  dateSel.innerHTML = '';
  const orderedCols = [...new Set([...(st.date_columns || []), ...(st.columns || [])])];
  orderedCols.forEach((c) => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    if (c === st.official_date_col || (c === 'date_ok' && !st.official_date_col)) opt.selected = true;
    dateSel.appendChild(opt);
  });

  // Update date guidance box
  const gBox = document.getElementById('date-col-guidance-box');
  if (st.date_processing_complete) {
    gBox.className = 'guidance-box success';
    gBox.innerHTML = `
      <i class="fa-solid fa-circle-check" style="font-size:1.25rem;color:var(--success);margin-top:2px;"></i>
      <div>
        <strong>Coluna 'date_ok' ativa e validada!</strong><br/>
        Período: <code>${st.date_min || '-'}</code> até <code>${st.date_max || '-'}</code>
        (${st.is_monthly ? 'Frequência Mensal detectada' : 'Série Temporal pronta'}). Próximas etapas habilitadas.
      </div>
    `;
  } else {
    gBox.className = 'guidance-box warning';
    gBox.innerHTML = `
      <i class="fa-solid fa-triangle-exclamation" style="font-size:1.25rem;color:var(--warning);margin-top:2px;"></i>
      <div>Selecione a coluna de data e clique em <strong>Converter para 'date_ok'</strong> para habilitar dummies e análises temporais.</div>
    `;
  }

  // Populate numeric checkbox lists in Import tab (Transform, Seasonality, Exploratory Plot)
  populateCheckboxList('transform-vars-list', st.numeric_columns, []);
  populateCheckboxList('seasonality-vars-list', st.numeric_columns, st.numeric_columns.slice(0, 1));
  populateCheckboxList('explore-plot-vars-list', st.numeric_columns, st.numeric_columns.slice(0, 2));

  if (st.date_min && st.date_max) {
    document.getElementById('seasonality-start-date').value = st.date_min;
    document.getElementById('seasonality-end-date').value = st.date_max;
  }

  // Render table
  renderImportDataTable();

  // Sync selectors across Caret, ARIMA, and AutoTS tabs
  syncForecastingModulesSelectors(st);
}

function populateCheckboxList(containerId, items, defaultChecked = []) {
  const box = document.getElementById(containerId);
  if (!box) return;
  if (!items || items.length === 0) {
    box.innerHTML = '<div style="color:var(--text-muted);font-size:0.82rem;">Nenhuma coluna numérica disponível.</div>';
    return;
  }
  const checkedSet = new Set(defaultChecked);
  box.innerHTML = items
    .map(
      (col) => `
      <label class="check-row">
        <input type="checkbox" value="${col}" ${checkedSet.has(col) ? 'checked' : ''} />
        <span>${col}</span>
      </label>
    `
    )
    .join('');
}

// ============================================================================
// Tab 2: Importação & Preparação Actions
// ============================================================================

async function onFileInputChanged() {
  const fileInput = document.getElementById('file-upload-input');
  if (!fileInput.files || fileInput.files.length === 0) return;
  document.getElementById('sample-file-select').value = '';
  const file = fileInput.files[0];
  const forceType = document.getElementById('force-file-type').value;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('tipo_arquivo', forceType);

  try {
    const res = await fetch('/api/import/inspect', { method: 'POST', body: fd });
    const meta = await res.json();
    updateFileMetadataUI(meta);
  } catch (e) {
    console.error(e);
  }
}

async function onSampleFileSelected() {
  const sampleName = document.getElementById('sample-file-select').value;
  if (!sampleName) return;
  document.getElementById('file-upload-input').value = '';
  const forceType = document.getElementById('force-file-type').value;

  try {
    const res = await fetch(`/api/import/inspect-sample?filename=${encodeURIComponent(sampleName)}&tipo_arquivo=${encodeURIComponent(forceType)}`);
    const meta = await res.json();
    updateFileMetadataUI(meta);
  } catch (e) {
    console.error(e);
  }
}

function onForceTypeChanged() {
  const fileInput = document.getElementById('file-upload-input');
  const sampleName = document.getElementById('sample-file-select').value;
  if (fileInput.files && fileInput.files.length > 0) onFileInputChanged();
  else if (sampleName) onSampleFileSelected();
}

function updateFileMetadataUI(meta) {
  document.getElementById('detected-file-type-badge').textContent = `Detectado: ${(meta.detected_type || 'auto').toUpperCase()}`;

  const sheetWrap = document.getElementById('excel-sheet-container');
  const sheetSel = document.getElementById('excel-sheet-select');
  if (meta.detected_type === 'xlsx' && meta.sheets && meta.sheets.length > 0) {
    sheetWrap.style.display = 'block';
    sheetSel.innerHTML = meta.sheets.map((s) => `<option value="${s}">${s}</option>`).join('');
  } else {
    sheetWrap.style.display = 'none';
  }

  const tableWrap = document.getElementById('sqlite-table-container');
  const tableSel = document.getElementById('sqlite-table-select');
  if (meta.detected_type === 'sqlite' && meta.tables && meta.tables.length > 0) {
    tableWrap.style.display = 'block';
    tableSel.innerHTML = meta.tables.map((t) => `<option value="${t}">${t}</option>`).join('');
  } else {
    tableWrap.style.display = 'none';
  }
}

async function executeDataImport() {
  const fileInput = document.getElementById('file-upload-input');
  const sampleName = document.getElementById('sample-file-select').value;
  const forceType = document.getElementById('force-file-type').value;
  const sheet = document.getElementById('excel-sheet-select').value || '';
  const table = document.getElementById('sqlite-table-select').value || '';

  const fd = new FormData();
  fd.append('tipo_arquivo', forceType);
  if (sheet) fd.append('planilha', sheet);
  if (table) fd.append('tabela', table);

  let endpoint = '';
  if (fileInput.files && fileInput.files.length > 0) {
    fd.append('file', fileInput.files[0]);
    endpoint = '/api/import/upload';
  } else if (sampleName) {
    fd.append('filename', sampleName);
    endpoint = '/api/import/sample';
  } else {
    showToast('Selecione um arquivo local ou um arquivo de exemplo da pasta data/.', 'warning');
    return;
  }

  try {
    showToast('Importando dados...', 'info', 2000);
    const res = await fetch(endpoint, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro na importação');
    clearCachedModelResults();
    applyStateAndPreview(data.state, data.preview);
    showToast('Dados importados com sucesso!', 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function inspectSelectedDateCol() {
  const col = document.getElementById('import-date-col-select').value;
  if (!col) return;
  try {
    const res = await fetch(`/api/date/inspect?coluna=${encodeURIComponent(col)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao inspecionar coluna');
    const insp = data.inspection;
    const gBox = document.getElementById('date-col-guidance-box');

    if (insp.e_data) {
      applyStateAndPreview(data.state, null);
      showToast(`Coluna '${col}' já é uma data válida. 'date_ok' atualizada.`, 'success');
    } else if (insp.pode_converter) {
      gBox.className = 'guidance-box warning';
      gBox.innerHTML = `
        <i class="fa-solid fa-wand-magic-sparkles" style="font-size:1.2rem;color:var(--warning);margin-top:2px;"></i>
        <div>
          Coluna <strong>'${col}'</strong> (${insp.classe}) pode ser convertida (${insp.n_convertido}/${insp.total_registros} válidos).
          Exemplos: <code>${(insp.amostras_convertidas || []).slice(0, 3).join(', ')}</code>.
          Clique em <strong>Converter para 'date_ok'</strong>.
        </div>
      `;
    } else {
      gBox.className = 'guidance-box danger';
      gBox.innerHTML = `
        <i class="fa-solid fa-circle-xmark" style="font-size:1.2rem;color:var(--danger);margin-top:2px;"></i>
        <div>Coluna <strong>'${col}'</strong> não parece conter datas reconhecíveis. Exemplos: <code>${(insp.amostras || []).slice(0, 4).join(', ')}</code></div>
      `;
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function convertDateToDateOk() {
  const col = document.getElementById('import-date-col-select').value;
  const fmt = document.getElementById('import-date-format-select').value;
  if (!col) {
    showToast('Selecione uma coluna de data.', 'warning');
    return;
  }
  try {
    const res = await fetch('/api/date/convert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ coluna: col, formato: fmt }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Falha ao converter data');
    applyStateAndPreview(data.state, data.preview);
    showToast(data.message, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function applySelectedDummies() {
  const payload = {
    anos: document.getElementById('dummy-anos').checked,
    meses: document.getElementById('dummy-meses').checked,
    tendencia_linear: document.getElementById('dummy-trend').checked,
    funcoes_trig: document.getElementById('dummy-trig').checked,
    estacoes: document.getElementById('dummy-estacoes').checked,
    bro_bro: document.getElementById('dummy-bro-bro').checked,
    crises_selecionadas: getCheckedValues('crises-checkbox-list'),
    ativar_dummy_personalizada: document.getElementById('dummy-custom-enable').checked,
    dummy_personalizada_nome: document.getElementById('dummy-custom-name').value,
    dummy_personalizada_inicio: document.getElementById('dummy-custom-start').value,
    dummy_personalizada_fim: document.getElementById('dummy-custom-end').value,
  };

  try {
    const res = await fetch('/api/dummies/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao gerar dummies');
    applyStateAndPreview(data.state, data.preview);
    setImportStepIndicator(4);
    showToast(data.message, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function applyNumericTransform() {
  const vars = getCheckedValues('transform-vars-list');
  const tipo = document.getElementById('transform-type-select').value;
  const lambdaRaw = document.getElementById('boxcox-lambda-input').value;

  if (tipo === 'nenhuma') {
    showToast('Selecione um tipo de transformação (Log, Box-Cox, Scale ou Center).', 'warning');
    return;
  }
  if (vars.length === 0) {
    showToast('Selecione ao menos uma variável numérica para transformar.', 'warning');
    return;
  }

  const payload = {
    variaveis: vars,
    transformacao: tipo,
    parametro_boxcox: lambdaRaw !== '' ? parseFloat(lambdaRaw) : null,
  };

  try {
    const res = await fetch('/api/transform/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro na transformação');
    applyStateAndPreview(data.state, data.preview);
    setImportStepIndicator(5);
    showToast(`Transformação '${tipo.toUpperCase()}' aplicada com sucesso!`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function extractSeasonalityAndTrend() {
  const vars = getCheckedValues('seasonality-vars-list');
  const dtIni = document.getElementById('seasonality-start-date').value;
  const dtFim = document.getElementById('seasonality-end-date').value;

  if (vars.length === 0) {
    showToast('Selecione ao menos uma variável numérica para extrair sazonalidade e tendência.', 'warning');
    return;
  }

  try {
    const res = await fetch('/api/seasonality/extract', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variaveis: vars, data_inicio: dtIni, data_fim: dtFim }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao extrair sazonalidade');
    applyStateAndPreview(data.state, data.preview);
    setImportStepIndicator(6);
    showToast(`Colunas adicionadas: ${data.added_columns.join(', ')}`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

function exportProcessedData() {
  const fmt = document.getElementById('export-data-format').value;
  setImportStepIndicator(7);
  window.open(`/api/data/export?fmt=${encodeURIComponent(fmt)}`, '_blank');
}

function clearCachedModelResults(purgePlots = false) {
  APP.caretResults = null;
  APP.arimaResults = null;
  APP.autotsResults = null;
  APP.growthResults = null;
  if (purgePlots) {
    [
      'explore-plotly-chart',
      'caret-main-plot',
      'caret-waterfall-plot',
      'caret-elasticity-plot',
      'arima-main-plot',
      'autots-forecast-plot',
      'autots-growth-plot',
    ].forEach((id) => {
      const el = document.getElementById(id);
      if (el) Plotly.purge(el);
    });
  }
}

async function resetImportPage() {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    const st = await res.json();
    APP.tableData = { columns: [], rows: [], total_rows: 0 };
    clearCachedModelResults(true);
    applyStateAndPreview(st, APP.tableData);
    showToast('Página e dados da sessão resetados.', 'info');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

// ============================================================================
// Step 7 & 8: Interactive Data Table & Exploratory Plotly Charts
// ============================================================================

function renderImportDataTable() {
  const wrap = document.getElementById('import-table-container');
  const pagWrap = document.getElementById('import-table-pagination');
  const { columns, rows } = APP.tableData;

  if (!columns || columns.length === 0 || !rows || rows.length === 0) {
    wrap.innerHTML = '<div style="padding: 28px; text-align: center; color: var(--text-muted);">Nenhum dado carregado.</div>';
    pagWrap.innerHTML = '';
    return;
  }

  const query = (document.getElementById('table-search-input').value || '').toLowerCase().trim();
  const pageSize = parseInt(document.getElementById('table-page-size').value || '10', 10);

  const filtered = query
    ? rows.filter((r) => columns.some((c) => String(r[c] ?? '').toLowerCase().includes(query)))
    : rows;

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  if (APP.tablePage > totalPages) APP.tablePage = totalPages;
  const startIdx = (APP.tablePage - 1) * pageSize;
  const pageRows = filtered.slice(startIdx, startIdx + pageSize);

  let html = '<table class="data-table"><thead><tr>';
  columns.forEach((c) => {
    html += `<th>${c}</th>`;
  });
  html += '</tr></thead><tbody>';

  pageRows.forEach((r) => {
    html += '<tr>';
    columns.forEach((c) => {
      const val = r[c];
      const formatted = typeof val === 'number' ? Number(val).toLocaleString('pt-BR', { maximumFractionDigits: 4 }) : (val ?? '<span style="color:var(--text-muted)">NA</span>');
      html += `<td>${formatted}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;

  pagWrap.innerHTML = `
    <div>Exibindo <strong>${filtered.length > 0 ? startIdx + 1 : 0}–${Math.min(startIdx + pageSize, filtered.length)}</strong> de <strong>${filtered.length}</strong> registros (Total: ${APP.tableData.total_rows})</div>
    <div style="display:flex;gap:6px;">
      <button class="btn btn-outline" style="padding:4px 10px;" ${APP.tablePage <= 1 ? 'disabled' : ''} onclick="APP.tablePage--;renderImportDataTable()">Anterior</button>
      <span style="padding:4px 10px;font-weight:600;">Página ${APP.tablePage} / ${totalPages}</span>
      <button class="btn btn-outline" style="padding:4px 10px;" ${APP.tablePage >= totalPages ? 'disabled' : ''} onclick="APP.tablePage++;renderImportDataTable()">Próxima</button>
    </div>
  `;
}

function generateExploratoryPlot() {
  const vars = getCheckedValues('explore-plot-vars-list');
  const plotType = document.getElementById('explore-plot-type').value;
  if (vars.length === 0) {
    showToast('Selecione ao menos uma variável numérica para o gráfico.', 'warning');
    return;
  }

  const rows = APP.tableData.rows || [];
  const dateCol = APP.state.columns.includes('date_ok') ? 'date_ok' : (APP.state.official_date_col || APP.state.columns[0]);
  const xDates = rows.map((r) => r[dateCol]);

  const traces = vars.map((v, i) => {
    const color = PALETTE[i % PALETTE.length];
    const yVals = rows.map((r) => r[v]);

    if (plotType === 'box') {
      const months = xDates.map((d) => (d ? String(d).slice(5, 7) : '01'));
      return {
        x: months,
        y: yVals,
        name: v,
        type: 'box',
        marker: { color },
        boxpoints: 'outliers',
      };
    } else if (plotType === 'bar') {
      return { x: xDates, y: yVals, name: v, type: 'bar', marker: { color } };
    } else if (plotType === 'scatter') {
      return { x: xDates, y: yVals, name: v, type: 'scatter', mode: 'markers', marker: { color, size: 7 } };
    }
    return { x: xDates, y: yVals, name: v, type: 'scatter', mode: 'lines', line: { color, width: 2.2 } };
  });

  const theme = getPlotlyThemeLayout();
  const layout = {
    ...theme,
    title: { text: 'Análise Exploratória de Séries Temporais', font: { size: 15 } },
    xaxis: { ...theme.xaxis, title: plotType === 'box' ? 'Mês do Ano (01–12)' : 'Data' },
    yaxis: { ...theme.yaxis, title: 'Valor' },
    boxmode: plotType === 'box' ? 'group' : undefined,
    hovermode: 'x unified',
    legend: { orientation: 'h', y: -0.18, x: 0.5, xanchor: 'center' },
    margin: { l: 60, r: 30, t: 50, b: 70 },
  };

  setImportStepIndicator(8);
  Plotly.newPlot('explore-plotly-chart', traces, layout, { responsive: true, displaylogo: false });
}

// ============================================================================
// Sync Selectors for Caret, ARIMA, and AutoTS Tabs
// ============================================================================

function syncForecastingModulesSelectors(st) {
  if (!st || !st.has_data) return;

  const dateChoices = [...new Set([...(st.date_columns || []), ...(st.columns || [])])];
  const preferredDate = st.columns.includes('date_ok') ? 'date_ok' : (st.official_date_col || dateChoices[0]);

  ['caret-date-col', 'arima-date-col', 'autots-date-col'].forEach((id) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = dateChoices
      .map((c) => `<option value="${c}" ${c === preferredDate ? 'selected' : ''}>${c}</option>`)
      .join('');
  });

  const numCols = st.numeric_columns || [];
  ['caret-target-y', 'arima-target-y', 'autots-target-y'].forEach((id) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = numCols
      .map((c, idx) => `<option value="${c}" ${(prev && c === prev) || (!prev && idx === 0) ? 'selected' : ''}>${c}</option>`)
      .join('');
  });

  populateCaretExplanatoryVars();
  populateArimaExogVars();

  // Default Dates using valid realized bounds of each selected target variable
  const caretTarget = document.getElementById('caret-target-y')?.value;
  const arimaTarget = document.getElementById('arima-target-y')?.value;
  const autotsTarget = document.getElementById('autots-target-y')?.value;
  syncDatesForTarget('caret', caretTarget);
  syncDatesForTarget('arima', arimaTarget);
  syncDatesForTarget('autots', autotsTarget);
}

function syncDatesForTarget(prefix, varName) {
  const st = APP.state;
  if (!st || !st.has_data) return;
  const bounds = (st.column_date_bounds || {})[varName] || {};
  const minDtStr = bounds.min || st.date_min;
  const maxDtStr = bounds.max || st.date_max;
  if (!minDtStr || !maxDtStr) return;

  // UTC-noon prevents local timezone day-shift when adding 24 months
  const maxDt = new Date(maxDtStr + 'T12:00:00Z');
  let futureStr = maxDtStr;
  if (!isNaN(maxDt.getTime())) {
    maxDt.setUTCMonth(maxDt.getUTCMonth() + 24);
    futureStr = maxDt.toISOString().slice(0, 10);
  }

  if (prefix === 'caret') {
    document.getElementById('caret-train-start').value = minDtStr;
    document.getElementById('caret-train-end').value = maxDtStr;
  } else if (prefix === 'arima') {
    document.getElementById('arima-start-date').value = minDtStr;
    document.getElementById('arima-end-date').value = maxDtStr;
    document.getElementById('arima-forecast-end').value = futureStr;
  } else if (prefix === 'autots') {
    document.getElementById('autots-start-date').value = minDtStr;
    document.getElementById('autots-end-date').value = maxDtStr;
    document.getElementById('autots-forecast-end').value = futureStr;
  }
}

function populateCaretExplanatoryVars() {
  if (!APP.state || !APP.state.numeric_columns) return;
  const targetY = document.getElementById('caret-target-y').value;
  const dateCol = document.getElementById('caret-date-col').value;
  const candidates = APP.state.numeric_columns.filter((c) => c !== targetY && c !== dateCol);
  // Pre-select up to 6 explanatory columns excluding target_teste duplicates
  const defaults = candidates.filter((c) => !c.includes('_teste')).slice(0, 6);
  populateCheckboxList('caret-explicativas-list', candidates, defaults);
  syncDatesForTarget('caret', targetY);
}

function populateArimaExogVars() {
  if (!APP.state || !APP.state.numeric_columns) return;
  const targetY = document.getElementById('arima-target-y').value;
  const dateCol = document.getElementById('arima-date-col').value;
  const candidates = APP.state.numeric_columns.filter((c) => c !== targetY && c !== dateCol);
  populateCheckboxList('arima-exog-list', candidates, []);
  const box = document.getElementById('arima-exog-list');
  box.onchange = () => {
    const selected = getCheckedValues('arima-exog-list');
    document.getElementById('arima-collinear-note').style.display = selected.length > 0 ? 'block' : 'none';
  };
  syncDatesForTarget('arima', targetY);
}

// ============================================================================
// Tab 3: Projeções ML (PyCaret / Caret Recipes)
// ============================================================================

async function loadCaretModelCatalog() {
  try {
    const res = await fetch('/api/caret/models');
    const data = await res.json();
    APP.caretCatalog = data.models || [];
    renderCaretModelCheckboxes(['lm', 'ridge_scale', 'rf', 'gbm', 'xgboost']);
  } catch (e) {
    console.error('Erro ao carregar catálogo de modelos:', e);
  }
}

function renderCaretModelCheckboxes(defaultIds = []) {
  const box = document.getElementById('caret-models-list');
  const defSet = new Set(defaultIds);
  document.getElementById('caret-models-count-label').textContent = `${APP.caretCatalog.length} modelos disponíveis`;

  box.innerHTML = APP.caretCatalog
    .map(
      (m) => `
      <label class="check-row" data-model-search="${(m.id + ' ' + m.label + ' ' + m.group).toLowerCase()}">
        <input type="checkbox" value="${m.id}" ${defSet.has(m.id) ? 'checked' : ''} />
        <span><strong>${m.id}</strong> <small style="color:var(--text-muted)">— ${m.group}</small></span>
      </label>
    `
    )
    .join('');
}

function filterCaretModelList() {
  const q = (document.getElementById('caret-model-search').value || '').toLowerCase().trim();
  const rows = document.querySelectorAll('#caret-models-list .check-row');
  rows.forEach((row) => {
    const text = row.getAttribute('data-model-search') || '';
    row.style.display = !q || text.includes(q) ? 'flex' : 'none';
  });
}

function selectPresetCaretModels(preset) {
  const fastSet = new Set(['lm', 'rlm', 'ridge_scale', 'lasso_scale', 'glmnet_scale', 'rf', 'et', 'gbm', 'xgboost']);
  document.querySelectorAll('#caret-models-list input[type="checkbox"]').forEach((cb) => {
    if (preset === 'fast') cb.checked = fastSet.has(cb.value);
  });
}

function toggleCaretAutoTune() {
  const on = document.getElementById('caret-auto-tune').checked;
  document.getElementById('caret-tunelength-wrap').style.display = on ? 'block' : 'none';
}

async function uploadCustomCaretModelsTxt() {
  const inp = document.getElementById('caret-custom-txt-input');
  if (!inp.files || inp.files.length === 0) return;
  const fd = new FormData();
  fd.append('file', inp.files[0]);
  try {
    const res = await fetch('/api/caret/custom-models', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao carregar arquivo de modelos');
    APP.caretCatalog = data.models || [];
    renderCaretModelCheckboxes(data.default_selected || []);
    showToast(`Modelos customizados carregados (${APP.caretCatalog.length})!`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function resetCustomCaretModels() {
  try {
    const res = await fetch('/api/caret/reset-models', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao restaurar catálogo');
    APP.caretCatalog = data.models || [];
    renderCaretModelCheckboxes(['lm', 'ridge_scale', 'rf', 'gbm', 'xgboost']);
    showToast('Catálogo padrão de modelos restaurado.', 'info');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function runCaretForecast() {
  const modelos = getCheckedValues('caret-models-list');
  if (modelos.length === 0) {
    showToast('Selecione ao menos um modelo para treinar.', 'warning');
    return;
  }

  const lagStr = document.getElementById('caret-lag-periods').value || '1,2,3';
  const lagPeriods = lagStr.split(',').map((x) => parseInt(x.trim(), 10)).filter((x) => !isNaN(x));

  const payload = {
    date_col: document.getElementById('caret-date-col').value,
    var_Y: document.getElementById('caret-target-y').value,
    explicativas: getCheckedValues('caret-explicativas-list'),
    modelos,
    data_inicio: document.getElementById('caret-train-start').value,
    data_teste: document.getElementById('caret-train-end').value,
    n_test: parseInt(document.getElementById('caret-n-test').value || '12', 10),
    metodo_impute: document.getElementById('caret-impute-method').value || null,
    transformacoes_numericas: getCheckedValues('caret-recipe-transforms'),
    opcoes_transform: { corr_threshold: 0.9, log_offset: 1.0, df_knots: 3 },
    engenharia_features: getCheckedValues('caret-feat-eng-list'),
    opcoes_features: {
      degree: parseInt(document.getElementById('caret-poly-degree').value || '2', 10),
      lag_periods: lagPeriods,
      window_size: parseInt(document.getElementById('caret-window-size').value || '3', 10),
    },
    reducao_dimensao: document.getElementById('caret-dim-reduction').value || null,
    opcoes_reducao: {
      num_comp: parseInt(document.getElementById('caret-num-comp').value || '5', 10),
    },
    auto_tune: document.getElementById('caret-auto-tune').checked,
    tune_length: parseInt(document.getElementById('caret-tunelength').value || '3', 10),
    validation_method: document.getElementById('caret-validation-method').value,
    cv_folds: parseInt(document.getElementById('caret-cv-folds').value || '5', 10),
    cv_repeats: parseInt(document.getElementById('caret-cv-repeats').value || '1', 10),
    periodo_analise: parseInt(document.getElementById('caret-backtest-months').value || '12', 10),
  };

  const btn = document.getElementById('btn-run-caret');
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Treinando Modelos...';

  try {
    const res = await fetch('/api/caret/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro durante treinamento ML');

    APP.caretResults = data;
    renderCaretMainPlot();
    renderCaretBacktestTable(data.recomendacao_backtest);

    // Populate model selector for Variable Importance & Elasticity
    const impSel = document.getElementById('caret-imp-model-select');
    const trainedModels = data.modelos_treinados || [];
    impSel.innerHTML = trainedModels.map((m) => `<option value="${m}">${m}</option>`).join('');
    renderCaretImportanceAndElasticity();

    showToast(`Treinamento concluído! ${trainedModels.length} modelo(s) ajustado(s).`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Treinar Modelos ML';
  }
}

function renderCaretMainPlot() {
  if (!APP.caretResults || !APP.caretResults.chart_series) return;
  const traces = APP.caretResults.chart_series.map((s, idx) => {
    if (s.is_real) {
      return {
        x: s.x,
        y: s.y,
        name: s.name,
        type: 'scatter',
        mode: 'lines',
        line: { color: document.body.classList.contains('dark-theme') ? '#f8fafc' : '#0f172a', width: 3 },
      };
    }
    return {
      x: s.x,
      y: s.y,
      name: s.name,
      type: 'scatter',
      mode: 'lines',
      line: { color: PALETTE[(idx - 1) % PALETTE.length], width: 1.8 },
    };
  });

  const theme = getPlotlyThemeLayout();
  const layout = {
    ...theme,
    title: { text: `Projeções ML vs Realizado (${APP.caretResults.var_Y})`, font: { size: 15 } },
    xaxis: { ...theme.xaxis, title: 'Data' },
    yaxis: { ...theme.yaxis, title: APP.caretResults.var_Y },
    hovermode: 'x unified',
    legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
    margin: { l: 65, r: 30, t: 50, b: 85 },
  };
  Plotly.newPlot('caret-main-plot', traces, layout, { responsive: true, displaylogo: false });
}

function getStatusBadgeHtml(status) {
  if (!status) return '<span class="badge-status badge-neutro">-</span>';
  const s = status.toUpperCase();
  let cls = 'badge-neutro';
  if (s.includes('ÓTIMO') || s.includes('OTIMO')) cls = 'badge-otimo';
  else if (s.includes('RECOMENDADO')) cls = 'badge-recomendado';
  else if (s.includes('ACEITÁVEL') || s.includes('REGULAR')) cls = 'badge-aceitavel';
  else if (s.includes('AVALIAR')) cls = 'badge-avaliar';
  else if (s.includes('REPROJETAR') || s.includes('RUIM')) cls = 'badge-reprojetar';
  return `<span class="badge-status ${cls}">${status}</span>`;
}

function renderCaretBacktestTable(rows) {
  const wrap = document.getElementById('caret-backtest-table-wrap');
  if (!rows || rows.length === 0) {
    wrap.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);">Sem dados para backtest.</div>';
    return;
  }

  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th title="Nome do Modelo">Modelo</th>
          <th title="Meses comuns analisados">Meses</th>
          <th title="Raiz do Erro Quadrático Médio">RMSE</th>
          <th title="Erro Percentual Absoluto Médio (%)">MAPE (%)</th>
          <th title="Erro Percentual Médio / Viés (%)">MPE (%)</th>
          <th title="Mediana do Erro Absoluto">MedAE</th>
          <th title="Raiz do Erro Logarítmico Quadrático Médio">RMSLE</th>
          <th title="Coeficiente de Determinação R²">R²</th>
          <th title="Variância Explicada">Expl. Var</th>
          <th title="Desvio Percentual Absoluto Máximo (%)">Desv.Máx (%)</th>
          <th title="Score Composto (MAPE + 0.4 * Desv.Máx)">Score</th>
          <th title="Recomendação Automática">Status</th>
        </tr>
      </thead>
      <tbody>
  `;

  rows.forEach((r, idx) => {
    html += `
      <tr>
        <td><strong>${idx === 0 ? '🏆 ' : ''}${r.modelo}</strong></td>
        <td>${r.meses ?? '-'}</td>
        <td>${r.RMSE ?? '-'}</td>
        <td><strong>${r.MAPE !== null && r.MAPE !== undefined ? r.MAPE + '%' : '-'}</strong></td>
        <td>${r.MPE !== null && r.MPE !== undefined ? r.MPE + '%' : '-'}</td>
        <td>${r.MedAE ?? '-'}</td>
        <td>${r.RMSLE ?? '-'}</td>
        <td>${r['R²'] ?? '-'}</td>
        <td>${r.Explained_Var ?? '-'}</td>
        <td>${r['desv.máx_perc'] !== null && r['desv.máx_perc'] !== undefined ? r['desv.máx_perc'] + '%' : '-'}</td>
        <td><strong>${r.score ?? '-'}</strong></td>
        <td>${getStatusBadgeHtml(r.status)}</td>
      </tr>
    `;
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

async function updateCaretBacktest() {
  const m = parseInt(document.getElementById('caret-backtest-months').value || '12', 10);
  try {
    const res = await fetch('/api/caret/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ periodo_analise: m }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao atualizar backtest');
    renderCaretBacktestTable(data.recomendacao_backtest);
    showToast(`Backtest atualizado para os últimos ${m} meses.`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

function renderCaretImportanceAndElasticity() {
  if (!APP.caretResults) return;
  const modelName = document.getElementById('caret-imp-model-select').value;
  if (!modelName) return;

  const theme = getPlotlyThemeLayout();

  // 1. Waterfall / Cascade Importance Chart (criar_grafico_cascata)
  const impObj = (APP.caretResults.importancia_variaveis || {})[modelName];
  const impWrap = document.getElementById('caret-imp-table-wrap');
  if (impObj && impObj.dados && impObj.dados.length > 0) {
    const dados = impObj.dados;
    const xVars = dados.map((d) => d.Variavel);
    const bases = dados.map((d) => d.Start);
    const heights = dados.map((d) => d.Importancia_Media);
    const colors = dados.map((_, i) => PALETTE[i % PALETTE.length]);

    const traceWaterfall = {
      x: xVars,
      y: heights,
      base: bases,
      type: 'bar',
      marker: { color: colors },
      text: heights.map((h) => `${h.toFixed(1)}%`),
      textposition: 'outside',
      hovertemplate: '<b>%{x}</b><br>Importância: %{y:.2f}%<br>Acumulado: %{base:.2f}% → %{customdata:.2f}%<extra></extra>',
      customdata: dados.map((d) => d.End),
    };

    Plotly.newPlot(
      'caret-waterfall-plot',
      [traceWaterfall],
      {
        ...theme,
        title: { text: `Importância Cumulativa em Cascata: ${modelName}`, font: { size: 14 } },
        xaxis: { ...theme.xaxis, tickangle: -35 },
        yaxis: { ...theme.yaxis, title: 'Importância Cumulativa (%)', range: [0, 112] },
        margin: { l: 55, r: 20, t: 45, b: 95 },
        showlegend: false,
      },
      { responsive: true, displaylogo: false }
    );

    impWrap.innerHTML = `
      <table class="data-table">
        <thead><tr><th>#</th><th>Variável</th><th>Importância (%)</th><th>Acumulado (%)</th></tr></thead>
        <tbody>
          ${dados
            .map(
              (d) => `
            <tr>
              <td>${d.Index}</td>
              <td><strong>${d.Variavel}</strong></td>
              <td>
                <div style="display:flex;align-items:center;gap:8px;">
                  <div style="width:70px;background:var(--border-color);height:8px;border-radius:4px;overflow:hidden;">
                    <div style="width:${Math.min(100, d.Importancia_Media)}%;background:var(--primary);height:100%;"></div>
                  </div>
                  <span>${d.Importancia_Media.toFixed(2)}%</span>
                </div>
              </td>
              <td>${d.End.toFixed(2)}%</td>
            </tr>
          `
            )
            .join('')}
        </tbody>
      </table>
    `;
  } else {
    Plotly.purge('caret-waterfall-plot');
    impWrap.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Importância não disponível.</div>';
  }

  // 2. Elasticities Chart & Table (calcular_elasticidades)
  const elastList = (APP.caretResults.elasticidades || {})[modelName];
  const elastWrap = document.getElementById('caret-elast-table-wrap');
  if (elastList && elastList.length > 0) {
    const sorted = [...elastList].reverse();
    const traceElast = {
      y: sorted.map((e) => e.Variavel),
      x: sorted.map((e) => e.Elasticidade_Com_Sinal),
      type: 'bar',
      orientation: 'h',
      marker: {
        color: sorted.map((e) => (e.Sinal === 'Positivo' ? '#10b981' : '#ef4444')),
      },
      text: sorted.map((e) => e.Elasticidade_Com_Sinal.toFixed(3)),
      textposition: 'auto',
    };

    Plotly.newPlot(
      'caret-elasticity-plot',
      [traceElast],
      {
        ...theme,
        title: { text: `Elasticidades das Variáveis: ${modelName}`, font: { size: 14 } },
        xaxis: { ...theme.xaxis, title: 'Elasticidade (% ΔY / 1% ΔX)' },
        yaxis: { ...theme.yaxis, automargin: true },
        margin: { l: 120, r: 25, t: 45, b: 55 },
        showlegend: false,
      },
      { responsive: true, displaylogo: false }
    );

    elastWrap.innerHTML = `
      <table class="data-table">
        <thead><tr><th>Variável</th><th>Elasticidade</th><th>Sinal</th></tr></thead>
        <tbody>
          ${elastList
            .map(
              (e) => `
            <tr>
              <td><strong>${e.Variavel}</strong></td>
              <td>${e.Elasticidade_Com_Sinal.toFixed(4)}</td>
              <td><span class="badge-status ${e.Sinal === 'Positivo' ? 'badge-otimo' : 'badge-reprojetar'}">${e.Sinal}</span></td>
            </tr>
          `
            )
            .join('')}
        </tbody>
      </table>
    `;
  } else {
    Plotly.purge('caret-elasticity-plot');
    elastWrap.innerHTML = '<div style="padding:16px;color:var(--text-muted);">Elasticidades não disponíveis.</div>';
  }
}

function exportCaretProjections() {
  if (!APP.caretResults) {
    showToast('Treine ao menos um modelo ML antes de exportar.', 'warning');
    return;
  }
  window.open('/api/caret/export-projections', '_blank');
}

function exportCaretModels() {
  if (!APP.caretResults) {
    showToast('Treine ao menos um modelo ML antes de baixar o arquivo .joblib.', 'warning');
    return;
  }
  window.open('/api/caret/export-models', '_blank');
}

function exportCaretImportanceCsv() {
  if (!APP.caretResults) {
    showToast('Treine ao menos um modelo ML antes de exportar a importância.', 'warning');
    return;
  }
  const m = document.getElementById('caret-imp-model-select').value;
  if (!m) return;
  window.open(`/api/caret/export-importance?modelo=${encodeURIComponent(m)}`, '_blank');
}

// ============================================================================
// Tab 4: Projeções ARIMA / SARIMAX
// ============================================================================

function toggleArimaModeUI() {
  const mode = document.getElementById('arima-mode').value;
  document.getElementById('arima-auto-controls').style.display = mode === 'auto' ? 'block' : 'none';
  document.getElementById('arima-manual-controls').style.display = mode === 'manual' ? 'block' : 'none';
}

async function runArimaForecast() {
  const mode = document.getElementById('arima-mode').value;
  const isManual = mode === 'manual';
  const seasonal = isManual
    ? document.getElementById('arima-seasonal-manual').checked
    : document.getElementById('arima-seasonal-auto').checked;
  const period = isManual
    ? parseInt(document.getElementById('arima-period-manual').value || '12', 10)
    : parseInt(document.getElementById('arima-period-auto').value || '12', 10);

  const payload = {
    date_col: document.getElementById('arima-date-col').value,
    target_var: document.getElementById('arima-target-y').value,
    exog_vars: getCheckedValues('arima-exog-list'),
    data_inicial: document.getElementById('arima-start-date').value,
    data_atual: document.getElementById('arima-end-date').value,
    end_predict: document.getElementById('arima-forecast-end').value,
    n_teste: parseInt(document.getElementById('arima-n-test').value || '6', 10),
    seasonal,
    period,
    num_models: parseInt(document.getElementById('arima-num-models').value || '5', 10),
    manual_mode: isManual,
    arima_params: {
      p: parseInt(document.getElementById('arima-p').value || '1', 10),
      d: parseInt(document.getElementById('arima-d').value || '1', 10),
      q: parseInt(document.getElementById('arima-q').value || '1', 10),
      P: parseInt(document.getElementById('arima-P').value || '1', 10),
      D: parseInt(document.getElementById('arima-D').value || '1', 10),
      Q: parseInt(document.getElementById('arima-Q').value || '1', 10),
    },
    peso_recente: parseFloat(document.getElementById('arima-peso-recente').value || '2.0'),
    peso_desvio_max: parseFloat(document.getElementById('arima-peso-max').value || '0.4'),
  };

  const btn = document.getElementById('btn-run-arima');
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Ajustando Modelos ARIMA...';

  try {
    const res = await fetch('/api/arima/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro na análise ARIMA');

    APP.arimaResults = data;
    renderArimaMainPlot();
    renderArimaTables();

    // Populate export selector
    const expSel = document.getElementById('arima-export-model-select');
    expSel.innerHTML =
      '<option value="Todos">Exportar: Todos os Modelos</option>' +
      (data.errors || []).map((e) => `<option value="${e.model}">${e.model} ${e.order}</option>`).join('');

    if (data.collinear_warning) {
      showToast('Colinearidade severa (|r| > 0.99) detectada entre exógenas: apenas modelos univariados foram estimados.', 'warning', 6000);
    }
    showToast(`Análise ARIMA concluída em ${data.execution_time_sec}s!`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Executar Modelos ARIMA';
  }
}

function renderArimaMainPlot() {
  if (!APP.arimaResults) return;
  const res = APP.arimaResults;
  const isDark = document.body.classList.contains('dark-theme');

  const traces = [
    {
      x: res.real_series.x,
      y: res.real_series.y,
      name: `Real (${res.target_var})`,
      type: 'scatter',
      mode: 'lines',
      line: { color: isDark ? '#f8fafc' : '#0f172a', width: 2.8 },
    },
  ];

  // Add 95% confidence band for the top ranked model
  if (res.forecasts && res.forecasts.length > 0) {
    const best = res.forecasts[0];
    traces.push({
      x: [...best.x, ...best.x.slice().reverse()],
      y: [...best.hi95, ...best.lo95.slice().reverse()],
      fill: 'toself',
      fillcolor: 'rgba(37, 99, 235, 0.14)',
      line: { color: 'transparent' },
      name: `IC 95% (${best.model})`,
      showlegend: true,
      hoverinfo: 'skip',
    });
  }

  (res.forecasts || []).forEach((f, idx) => {
    traces.push({
      x: f.x,
      y: f.y,
      name: `${f.model} ${f.order}`,
      type: 'scatter',
      mode: 'lines',
      line: {
        color: PALETTE[idx % PALETTE.length],
        width: f.type.includes('Multi') ? 2.6 : 1.9,
        dash: f.type.includes('Multi') ? 'solid' : 'dot',
      },
    });
  });

  const shapes = [];
  if (res.periods && res.periods.train_end) {
    shapes.push({
      type: 'line',
      x0: res.periods.train_end,
      x1: res.periods.train_end,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#ef4444', width: 2, dash: 'dash' },
    });
  }
  if (res.periods && res.periods.test_end) {
    shapes.push({
      type: 'line',
      x0: res.periods.test_end,
      x1: res.periods.test_end,
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#2563eb', width: 2, dash: 'dash' },
    });
  }

  const theme = getPlotlyThemeLayout();
  Plotly.newPlot(
    'arima-main-plot',
    traces,
    {
      ...theme,
      title: {
        text: `Previsão ARIMA/SARIMAX — Treino: ${res.periods.train_start} a ${res.periods.train_end} | Teste até: ${res.periods.test_end}`,
        font: { size: 14 },
      },
      xaxis: { ...theme.xaxis, title: 'Data' },
      yaxis: { ...theme.yaxis, title: res.target_var },
      shapes,
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.22, x: 0.5, xanchor: 'center' },
      margin: { l: 65, r: 30, t: 50, b: 90 },
    },
    { responsive: true, displaylogo: false }
  );
}

function renderArimaTables() {
  if (!APP.arimaResults) return;
  const dash = APP.arimaResults.dashboard || [];
  const devs = APP.arimaResults.deviations || [];

  const dashWrap = document.getElementById('arima-dashboard-table-wrap');
  dashWrap.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Modelo</th>
          <th>Ordem SARIMA</th>
          <th>Tipo</th>
          <th>Meses Teste</th>
          <th>RMSE</th>
          <th>MAE</th>
          <th>MAPE (%)</th>
          <th>Desv.Méd.Pond (%)</th>
          <th>Desv.Máx (%)</th>
          <th>Score</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        ${dash
          .map(
            (r, i) => `
          <tr>
            <td><strong>${i === 0 ? '🏆 ' : ''}${r.modelo}</strong></td>
            <td><code>${r.ordem || '-'}</code></td>
            <td>${r.tipo || '-'}</td>
            <td>${r.meses ?? '-'}</td>
            <td>${r.RMSE ?? '-'}</td>
            <td>${r.MAE ?? '-'}</td>
            <td><strong>${r.MAPE !== null && r.MAPE !== undefined ? r.MAPE + '%' : '-'}</strong></td>
            <td>${r['desv.méd_pond'] !== null && r['desv.méd_pond'] !== undefined ? r['desv.méd_pond'] + '%' : '-'}</td>
            <td>${r['desv.máx_perc'] !== null && r['desv.máx_perc'] !== undefined ? r['desv.máx_perc'] + '%' : '-'}</td>
            <td><strong>${r.score ?? '-'}</strong></td>
            <td>${getStatusBadgeHtml(r.status)}</td>
          </tr>
        `
          )
          .join('')}
      </tbody>
    </table>
  `;

  const devWrap = document.getElementById('arima-deviations-table-wrap');
  if (devs.length === 0) {
    devWrap.innerHTML = '<div style="padding:14px;color:var(--text-muted);">Sem período de teste separado (n_teste = 0).</div>';
  } else {
    devWrap.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Modelo</th>
            <th>Desv. Mín</th>
            <th>Desv. Máx</th>
            <th>Desv. Médio</th>
            <th>Desv. Acumulado</th>
            <th>Desv. Mín (%)</th>
            <th>Desv. Máx (%)</th>
            <th>Desv. Médio (%)</th>
          </tr>
        </thead>
        <tbody>
          ${devs
            .map(
              (d) => `
            <tr>
              <td><strong>${d.model}</strong></td>
              <td>${d.min_deviation}</td>
              <td>${d.max_deviation}</td>
              <td>${d.mean_deviation}</td>
              <td>${d.cum_deviation}</td>
              <td>${d.min_pct_deviation}%</td>
              <td>${d.max_pct_deviation}%</td>
              <td>${d.mean_pct_deviation}%</td>
            </tr>
          `
            )
            .join('')}
        </tbody>
      </table>
    `;
  }
}

async function updateArimaDashboard() {
  if (!APP.arimaResults) return;
  const payload = {
    peso_recente: parseFloat(document.getElementById('arima-peso-recente').value || '2.0'),
    peso_desvio_max: parseFloat(document.getElementById('arima-peso-max').value || '0.4'),
    periodo_maximo: parseInt(document.getElementById('arima-n-test').value || '12', 10),
  };
  try {
    const res = await fetch('/api/arima/dashboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao atualizar dashboard ARIMA');
    APP.arimaResults.dashboard = data.dashboard;
    renderArimaTables();
    showToast('Dashboard ARIMA atualizado!', 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

function exportArimaExcel() {
  if (!APP.arimaResults) {
    showToast('Execute os modelos ARIMA antes de exportar.', 'warning');
    return;
  }
  const m = document.getElementById('arima-export-model-select').value || 'Todos';
  window.open(`/api/arima/export?model=${encodeURIComponent(m)}`, '_blank');
}

// ============================================================================
// Tab 5: Projeções AutoTS & Análise de Crescimento
// ============================================================================

async function runAutoTSAnalysis() {
  const models = getCheckedValues('autots-models-checks');
  if (models.length === 0) {
    showToast('Selecione ao menos um algoritmo AutoTS.', 'warning');
    return;
  }

  const payload = {
    date_col: document.getElementById('autots-date-col').value,
    variable: document.getElementById('autots-target-y').value,
    start_date: document.getElementById('autots-start-date').value,
    end_date: document.getElementById('autots-end-date').value,
    end_predict: document.getElementById('autots-forecast-end').value,
    selected_models: models,
    periodo_maximo: parseInt(document.getElementById('autots-desv-meses').value || '12', 10),
    peso_recente: parseFloat(document.getElementById('autots-peso-recente').value || '2.0'),
    peso_desvio_max: parseFloat(document.getElementById('autots-peso-max').value || '0.4'),
  };

  const btn = document.getElementById('btn-run-autots');
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Executando AutoTS...';

  try {
    const res = await fetch('/api/autots/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro na análise AutoTS');

    APP.autotsResults = data;
    APP.growthResults = data.growth;

    const selMod = document.getElementById('autots-selected-model');
    const growthMod = document.getElementById('growth-model-select');
    const optsHtml = (data.models || [])
      .map((m) => `<option value="${m}" ${m === data.best_model ? 'selected' : ''}>${m.toUpperCase()}</option>`)
      .join('');
    selMod.innerHTML = optsHtml;
    growthMod.innerHTML = optsHtml;

    renderAutoTSForecastPlot();
    renderAutoTSTables();
    renderAutoTSGrowthView();

    showToast(`AutoTS concluído! Melhor modelo: ${(data.best_model || '').toUpperCase()}`, 'success');
  } catch (e) {
    showToast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-play"></i> Executar Análise AutoTS';
  }
}

function renderAutoTSForecastPlot() {
  if (!APP.autotsResults) return;
  const res = APP.autotsResults;
  const selectedModel = document.getElementById('autots-selected-model').value || res.best_model;
  const onlySelected = document.getElementById('autots-show-only-selected').checked;
  const isDark = document.body.classList.contains('dark-theme');

  const traces = [
    {
      x: res.real_series.x,
      y: res.real_series.y,
      name: `Realizado (${res.variable})`,
      type: 'scatter',
      mode: 'lines',
      line: { color: isDark ? '#f8fafc' : '#0f172a', width: 2.8 },
    },
  ];

  const modelsToPlot = onlySelected ? [selectedModel] : res.models || [];
  modelsToPlot.forEach((m, idx) => {
    const mData = (res.chart_models || {})[m];
    if (!mData) return;
    const color = PALETTE[idx % PALETTE.length];
    const isSel = m === selectedModel;

    if (isSel) {
      traces.push({
        x: [...mData.x_fut, ...mData.x_fut.slice().reverse()],
        y: [...mData.y_upper, ...mData.y_lower.slice().reverse()],
        fill: 'toself',
        fillcolor: 'rgba(37, 99, 235, 0.15)',
        line: { color: 'transparent' },
        name: `IC 95% (${m.toUpperCase()})`,
        hoverinfo: 'skip',
      });
    }

    traces.push({
      x: [...mData.x_hist, ...mData.x_fut],
      y: [...mData.y_fitted, ...mData.y_mean],
      name: m.toUpperCase(),
      type: 'scatter',
      mode: 'lines',
      line: { color, width: isSel ? 2.6 : 1.5 },
    });
  });

  const theme = getPlotlyThemeLayout();
  Plotly.newPlot(
    'autots-forecast-plot',
    traces,
    {
      ...theme,
      title: { text: `Projeções AutoTS para ${res.variable}`, font: { size: 15 } },
      xaxis: { ...theme.xaxis, title: 'Data' },
      yaxis: { ...theme.yaxis, title: res.variable },
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' },
      margin: { l: 65, r: 30, t: 50, b: 85 },
    },
    { responsive: true, displaylogo: false }
  );
}

function renderAutoTSTables() {
  if (!APP.autotsResults) return;
  const desvios = APP.autotsResults.dashboard_desvios || [];
  const errors = APP.autotsResults.errors || [];

  document.getElementById('autots-desvios-table-wrap').innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Modelo</th>
          <th>Meses</th>
          <th>Score</th>
          <th>Desv.Méd (%)</th>
          <th>Desv.Máx (%)</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        ${desvios
          .map(
            (d, i) => `
          <tr>
            <td><strong>${i === 0 ? '🏆 ' : ''}${d.modelo.toUpperCase()}</strong></td>
            <td>${d.meses}</td>
            <td><strong>${d.score}</strong></td>
            <td>${d['desv.méd']}%</td>
            <td>${d['desv.máx']}%</td>
            <td>${getStatusBadgeHtml(d.status)}</td>
          </tr>
        `
          )
          .join('')}
      </tbody>
    </table>
  `;

  document.getElementById('autots-errors-table-wrap').innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Algoritmo</th>
          <th>RMSE</th>
          <th>MAE</th>
          <th>MAPE (%)</th>
        </tr>
      </thead>
      <tbody>
        ${errors
          .map(
            (e) => `
          <tr>
            <td><strong>${e.algo.toUpperCase()}</strong></td>
            <td>${e.RMSE ?? '-'}</td>
            <td>${e.MAE ?? '-'}</td>
            <td>${e.MAPE !== null && e.MAPE !== undefined ? e.MAPE + '%' : '-'}</td>
          </tr>
        `
          )
          .join('')}
      </tbody>
    </table>
  `;
}

async function updateAutoTSDesvios() {
  if (!APP.autotsResults) return;
  const payload = {
    periodo_maximo: parseInt(document.getElementById('autots-desv-meses').value || '12', 10),
    peso_recente: parseFloat(document.getElementById('autots-peso-recente').value || '2.0'),
    peso_desvio_max: parseFloat(document.getElementById('autots-peso-max').value || '0.4'),
  };
  try {
    const res = await fetch('/api/autots/desvios', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao atualizar desvios');
    APP.autotsResults.dashboard_desvios = data.dashboard_desvios;
    renderAutoTSTables();
    showToast('Tabela de desvios AutoTS atualizada!', 'success');
  } catch (e) {
    showToast(e.message, 'error');
  }
}

function exportAutoTSExcel(onlySelected) {
  if (!APP.autotsResults) {
    showToast('Execute a análise AutoTS antes de exportar.', 'warning');
    return;
  }
  if (onlySelected) {
    const m = document.getElementById('autots-selected-model').value || APP.autotsResults.best_model;
    window.open(`/api/autots/export?model=${encodeURIComponent(m)}`, '_blank');
  } else {
    window.open('/api/autots/export', '_blank');
  }
}

function exportAutoTSGrowthExcel() {
  if (!APP.growthResults) {
    showToast('Execute a análise AutoTS antes de exportar as visões de crescimento.', 'warning');
    return;
  }
  window.open('/api/autots/export-growth', '_blank');
}

async function fetchAutoTSGrowthViews() {
  if (!APP.autotsResults) return;
  const modelName = document.getElementById('growth-model-select').value;
  try {
    const res = await fetch('/api/autots/growth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName, variable: APP.autotsResults.variable }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Erro ao gerar análise de crescimento');
    APP.growthResults = data;
    renderAutoTSGrowthView();
  } catch (e) {
    showToast(e.message, 'error');
  }
}

function switchGrowthTableTab(tabKey, btnEl) {
  APP.activeGrowthTableTab = tabKey;
  btnEl.parentElement.querySelectorAll('.subtab-btn').forEach((b) => b.classList.remove('active'));
  btnEl.classList.add('active');
  renderAutoTSGrowthTable();
}

function renderAutoTSGrowthView() {
  if (!APP.growthResults) return;
  renderAutoTSGrowthChart();
  renderAutoTSGrowthTable();
}

function renderAutoTSGrowthChart() {
  if (!APP.growthResults) return;
  const gran = document.getElementById('growth-period-select').value || 'mensal';
  const metricType = document.getElementById('growth-metric-type').value || 'valores';
  const chartStyle = document.getElementById('growth-chart-style').value || 'combined';

  const viewObj = APP.growthResults[gran];
  if (!viewObj || !viewObj.long) return;

  const colMap = {
    valores: { key: 'Valor', label: 'Valor Absoluto' },
    taxas: { key: 'Taxa_Crescimento', label: 'Taxa de Crescimento YoY (%)' },
    variacoes: { key: 'Variacao_Valor', label: 'Variação Absoluta YoY' },
  };
  const { key, label } = colMap[metricType] || colMap.valores;

  const rows = viewObj.long.filter((r) => r[key] !== null && r[key] !== undefined);
  const xDates = rows.map((r) => r.Data);
  const yVals = rows.map((r) => r[key]);
  const barColors = yVals.map((v) => (v < 0 ? '#ef4444' : '#10b981'));

  const traces = [];
  if (chartStyle === 'bars' || chartStyle === 'combined') {
    traces.push({
      x: xDates,
      y: yVals,
      type: 'bar',
      name: label,
      marker: { color: metricType === 'valores' ? '#3b82f6' : barColors },
    });
  }
  if (chartStyle === 'lines' || chartStyle === 'combined') {
    traces.push({
      x: xDates,
      y: yVals,
      type: 'scatter',
      mode: 'lines+markers',
      name: chartStyle === 'combined' ? 'Tendência' : label,
      line: { color: '#2563eb', width: 2.3 },
      marker: { size: 5 },
    });
  }

  const theme = getPlotlyThemeLayout();
  Plotly.newPlot(
    'autots-growth-plot',
    traces,
    {
      ...theme,
      title: { text: `${label} — Visão ${gran.toUpperCase()} (${APP.growthResults.model.toUpperCase()})`, font: { size: 14 } },
      xaxis: { ...theme.xaxis, title: 'Data' },
      yaxis: { ...theme.yaxis, title: label },
      hovermode: 'x unified',
      margin: { l: 65, r: 25, t: 45, b: 60 },
      showlegend: chartStyle === 'combined',
    },
    { responsive: true, displaylogo: false }
  );
}

function renderAutoTSGrowthTable() {
  if (!APP.growthResults) return;
  const gran = document.getElementById('growth-period-select').value || 'mensal';
  const tabKey = APP.activeGrowthTableTab || 'valores';
  const viewObj = APP.growthResults[gran];
  const wrap = document.getElementById('autots-growth-table-wrap');

  if (!viewObj) {
    wrap.innerHTML = '';
    return;
  }

  const cols = viewObj.columns || [];
  const rows = viewObj[tabKey] || [];
  const isPct = tabKey === 'taxas';
  const isDiff = tabKey === 'variacoes';

  let html = '<table class="data-table"><thead><tr>';
  cols.forEach((c) => {
    html += `<th>${c}</th>`;
  });
  html += '</tr></thead><tbody>';

  rows.forEach((r) => {
    html += '<tr>';
    cols.forEach((c) => {
      const val = r[c];
      if (c === 'Ano') {
        html += `<td><strong>${val}</strong></td>`;
      } else if (val === null || val === undefined) {
        html += '<td style="color:var(--text-muted)">-</td>';
      } else {
        const num = Number(val);
        const formatted = num.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + (isPct ? '%' : '');
        let colorStyle = '';
        if (isPct || isDiff) {
          colorStyle = num > 0 ? 'color:#10b981;font-weight:600;' : (num < 0 ? 'color:#ef4444;font-weight:600;' : '');
        }
        html += `<td style="${colorStyle}">${formatted}</td>`;
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}
