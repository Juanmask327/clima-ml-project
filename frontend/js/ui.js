/**
 * ============================================================
 * PREDICCIÓN CLIMÁTICA ML · MICRO-ZONA
 * Módulo: Interfaz de Usuario (UI Controller)
 * ============================================================
 * Maneja todos los eventos del DOM, actualización de elementos
 * y construcción de gráficos. Se apoya en model.js para la
 * lógica de predicción.
 * ============================================================
 */

'use strict';

// ─── Estado global ──────────────────────────────────────────
let forecastChartInst = null;
let modelInfoLoaded   = false;   // /model/info ya cargado
let apiAvailable      = null;    // null=desconocido, true/false

// ============================================================
//  INICIALIZACIÓN
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
  // Primero se comprueba la API: si no, predict() cae al modo
  // local simulado y la UI lo indica en el badge.
  await probeApi();
  await predict();       // Predicción inicial al cargar la página
});

/**
 * Comprueba si la API está levantada y actualiza el badge.
 */
async function probeApi() {
  try {
    const res = await apiFetch('/health');
    const d = await res.json();
    apiAvailable = !!d.model_loaded;
    setApiStatus(apiAvailable ? 'api-ok' : 'api-degraded');
  } catch (err) {
    apiAvailable = false;
    setApiStatus('api-off');
    console.warn('[API] No se pudo contactar la API:', err.message);
  }
}

/** Actualiza el badge de estado de la API en el header. */
function setApiStatus(state) {
  const el = document.getElementById('api-status');
  if (!el) return;
  el.className = 'api-status ' + state;
  el.textContent = {
    'api-ok':        '● Modelo ML en vivo (API conectada)',
    'api-degraded':  '● API sin modelo cargado',
    'api-off':       '○ Modo local simulado (API desconectada)'
  }[state] || '';
}

// ============================================================
//  CONTROL DE TABS
// ============================================================

/**
 * Cambia la pestaña visible.
 * @param {string} tab  - ID de la pestaña ('predict' | 'forecast' | 'model')
 * @param {HTMLElement} btn - Botón que disparó el evento
 */
function switchTab(tab, btn) {
  ['predict', 'forecast', 'model'].forEach(t => {
    const el = document.getElementById('tab-' + t);
    if (el) el.style.display = t === tab ? '' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  // Las cargas son asíncronas: se capturan los errores para que
  // un fallo no quede como "unhandled promise rejection".
  if (tab === 'forecast') {
    buildForecast().catch(err => console.error('[Forecast]', err));
  }
  if (tab === 'model') {
    loadModelInfo().catch(err => console.error('[ModelInfo]', err));
  }
}

// ============================================================
//  LECTURA DE INPUTS
// ============================================================

/**
 * Lee los valores actuales de todos los sliders.
 * @returns {Object} Variables meteorológicas
 */
function getInputs() {
  const sliders = document.querySelectorAll('input[type=range]');
  const vals = Array.from(sliders).map(s => +s.value);
  // Se leen por id (data-var) para no depender del orden en el HTML.
  const byId = {};
  sliders.forEach(s => { if (s.dataset && s.dataset.var) byId[s.dataset.var] = +s.value; });
  const pick = (name, idx, def) => {
    const v = byId[name] != null ? byId[name] : (vals[idx] != null ? vals[idx] : def);
    return Number.isFinite(v) ? v : def;
  };
  return {
    temp:  pick('temp',  0, 18),
    hum:   pick('hum',   1, 65),
    pres:  pick('pres',  2, 1013),
    wind:  pick('wind',  3, 12),
    cloud: pick('cloud', 4, 40),
    uv:    pick('uv',    5, 5)
  };
}

/**
 * Actualiza el label de un slider en tiempo real.
 * @param {string} id   - ID del elemento span
 * @param {string} val  - Valor actual del slider
 * @param {string} unit - Unidad de medida
 */
function upd(id, val, unit) {
  document.getElementById(id).textContent = val + (unit || '');
}

// ============================================================
//  PREDICCIÓN PRINCIPAL
// ============================================================

/**
 * Ejecuta la predicción y actualiza el panel de resultados.
 * Usa la API real (fetchPrediction) cuando está disponible y
 * cae a gbModel() (simulación local) si el backend no responde.
 */
async function predict() {
  const inp = getInputs();
  let r, source;

  try {
    r = await fetchPrediction(inp);   // ← modelo ML real
    source = 'api';
    if (!apiAvailable) { apiAvailable = true; setApiStatus('api-ok'); }
  } catch (err) {
    console.warn('[API] /predict falló, se usa simulación local:', err.message);
    apiAvailable = false;
    setApiStatus('api-off');
    r = { ...gbModel(inp), source: 'local' };   // ← respaldo
    source = 'local';
  }

  // Mostrar panel de resultados
  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';

  // ── Actualizar elementos del DOM ────────────────────────
  setEl('res-icon',     r.icon);
  setEl('res-temp',     `${r.predTemp}°C predicho`);
  setEl('res-desc',     r.desc);
  setEl('res-conf',     `${r.conf}%`);
  setEl('r-lluvia',     `${r.rainProb}%`);
  setEl('r-viento',     `${r.windPred} km/h`);
  setEl('r-sensacion',  `${r.sensacion}°C`);
  setEl('res-rmse',     `±${r.rmse}°C`);
  setEl('res-source',   source === 'api'
    ? 'Predicción del modelo ML (XGBoost + RF + Ridge)'
    : 'Simulación local — inicia la API para usar el modelo real');

  // ── Barra de confianza animada ───────────────────────────
  const fill = document.getElementById('conf-fill');
  fill.style.width = '0%';
  setTimeout(() => { fill.style.width = `${r.conf}%`; }, 50);
}

// ============================================================
//  PRONÓSTICO 7 DÍAS
// ============================================================

/**
 * Construye la fila de tarjetas de pronóstico y el gráfico.
 * Usa /forecast (API, días reales) o generateForecast() como
 * respaldo local si el backend no está disponible.
 */
async function buildForecast() {
  const inp = getInputs();
  let forecast;

  try {
    forecast = await fetchForecast(inp);   // ← API real
    if (!apiAvailable) { apiAvailable = true; setApiStatus('api-ok'); }
  } catch (err) {
    console.warn('[API] /forecast falló, se usa pronóstico local:', err.message);
    apiAvailable = false;
    setApiStatus('api-off');
    forecast = generateForecast(inp);
  }

  const container = document.getElementById('forecast-row');
  container.innerHTML = '';

  const tempsHi = [], tempsLo = [];

  forecast.forEach(({ day, icon, hi, lo, rainProb }) => {
    tempsHi.push(hi);
    tempsLo.push(lo);
    const d = document.createElement('div');
    d.className = 'forecast-day';
    const rainBadge = rainProb != null
      ? `<div class="fd-rain">💧${Math.round(rainProb)}%</div>` : '';
    d.innerHTML = `
      <div class="fd-day">${day}</div>
      <div class="fd-icon">${icon}</div>
      <div class="fd-hi">${hi}°</div>
      <div class="fd-lo">${lo}°</div>
      ${rainBadge}
    `;
    container.appendChild(d);
  });

  renderForecastChart(forecast.map(f => f.day), tempsHi, tempsLo);
}

// ============================================================
//  GRÁFICO DE TENDENCIA (Chart.js)
// ============================================================

/**
 * Renderiza el gráfico de líneas de temperatura con Chart.js.
 * @param {string[]} labels  - Etiquetas de días
 * @param {number[]} tempsHi - Temperaturas máximas
 * @param {number[]} tempsLo - Temperaturas mínimas
 */
function renderForecastChart(labels, tempsHi, tempsLo) {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  if (forecastChartInst) forecastChartInst.destroy();

  forecastChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Temp máx (°C)',
          data: tempsHi,
          borderColor: '#185FA5',
          backgroundColor: 'rgba(24,95,165,0.08)',
          tension: 0.4,
          fill: true,
          pointRadius: 4,
          pointBackgroundColor: '#185FA5'
        },
        {
          label: 'Temp mín (°C)',
          data: tempsLo,
          borderColor: '#0F6E56',
          backgroundColor: 'rgba(15,110,86,0.08)',
          tension: 0.4,
          fill: true,
          borderDash: [4, 3],
          pointRadius: 4,
          pointBackgroundColor: '#0F6E56'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          ticks: { color: '#888', font: { size: 11 } },
          grid: { color: 'rgba(128,128,128,0.15)' }
        },
        x: {
          ticks: { color: '#888', font: { size: 11 } },
          grid: { display: false }
        }
      }
    }
  });
}

// ============================================================
//  MÉTRICAS DEL MODELO (desde /model/info)
// ============================================================

// Respaldo local si la API no responde (valores reales del último entrenamiento)
const FALLBACK_MODEL_INFO = {
  r2_score: 0.9309, rmse: 1.1138, mae: 0.8417, n_samples: 8760,
  feature_importances: {
    temp_actual: 81.3, heat_index: 15.3, indice_uv: 1.1,
    nubosidad: 0.5, humedad: 0.4, viento: 0.2
  },
  hyperparameters: {
    n_estimators: 571, max_depth: 3, learning_rate: 0.0512,
    subsample: 0.6289, colsample_bytree: 0.6851,
    reg_alpha: 0.4619, reg_lambda: 1.0292
  }
};

const FEATURE_LABELS = {
  temp_actual: 'Temperatura actual', humedad: 'Humedad relativa',
  presion: 'Presión atmosférica', viento: 'Velocidad del viento',
  nubosidad: 'Nubosidad', indice_uv: 'Índice UV',
  pressure_delta: 'Δ Presión', hum_effect: 'Efecto humedad',
  cloud_effect: 'Efecto nubosidad', wind_chill: 'Wind chill',
  heat_index: 'Índice de calor'
};

const HP_ROWS = [
  ['n_estimators', 'n_estimators'], ['max_depth', 'max_depth'],
  ['learning_rate', 'learning_rate'], ['subsample', 'subsample'],
  ['colsample_bytree', 'colsample_bytree'],
  ['reg_alpha', 'reg_alpha (L1)'], ['reg_lambda', 'reg_lambda (L2)']
];

/**
 * Carga métricas reales, importancia de features e hiperparámetros
 * desde /model/info. Se ejecuta una sola vez al abrir la pestaña.
 */
async function loadModelInfo() {
  if (modelInfoLoaded) return;

  let info;
  let fromApi = false;
  try {
    info = await fetchModelInfo();
    fromApi = true;
    apiAvailable = true;
    setApiStatus('api-ok');
  } catch (err) {
    console.warn('[API] /model/info falló, se usan valores de referencia:', err.message);
    apiAvailable = false;
    setApiStatus('api-off');
    info = FALLBACK_MODEL_INFO;
  }

  setEl('metrics-source', fromApi
    ? 'Métricas reales de validación cruzada (TimeSeriesSplit 5 folds)'
    : 'Valores de referencia (API desconectada)');

  // ── Métricas ─────────────────────────────────────────────
  setEl('m-r2',       info.r2_score.toFixed(3));
  setEl('m-rmse',     `${info.rmse.toFixed(2)}°C`);
  setEl('m-mae',      `${info.mae.toFixed(2)}°C`);
  setEl('m-samples',  info.n_samples.toLocaleString('es'));

  // ── Importancia de features (top 6 reales, cualquier feature) ──
  const imp = info.feature_importances || {};
  const colors = ['#378ADD', '#1D9E75', '#D85A30', '#BA7517', '#7F77DD', '#D4537E'];
  const texts  = ['#185FA5', '#0F6E56', '#993C1D', '#854F0B', '#534AB7', '#993556'];

  const top = Object.entries(imp)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);

  const grid = document.getElementById('importance-grid');
  if (grid && top.length) {
    grid.innerHTML = top.map(([f, v], i) => `
        <div class="feature-item">
          <div class="f-label">${FEATURE_LABELS[f] || f}</div>
          <div class="f-bar-wrap">
            <div class="f-bar" style="width:${Math.min(100, v)}%;background:${colors[i % colors.length]};"></div>
          </div>
          <div class="f-pct" style="color:${texts[i % texts.length]};">${v}%</div>
        </div>`).join('');
  }

  // ── Hiperparámetros reales (Optuna) ──────────────────────
  const hp = info.hyperparameters || {};
  if (Object.keys(hp).length > 0) {
    HP_ROWS.forEach(([key, label]) => {
      const el = document.getElementById('hp-' + key);
      if (el && hp[key] != null) {
        el.previousElementSibling.textContent = label;
        // Formato limpio: los floats largos de Optuna a 4 decimales
        el.textContent = (typeof hp[key] === 'number' && !Number.isInteger(hp[key]))
          ? hp[key].toFixed(4)
          : hp[key];
      }
    });
    setEl('hp-source', 'Valores óptimos encontrados por Optuna (50 trials)');
  }

  modelInfoLoaded = true;
}

// ============================================================
//  HELPERS DOM
// ============================================================

/** Actualiza el textContent de un elemento por su ID */
function setEl(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}