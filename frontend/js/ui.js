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

// ============================================================
//  INICIALIZACIÓN
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  predict();  // Predicción inicial al cargar la página
});

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
    document.getElementById('tab-' + t).style.display = t === tab ? '' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (tab === 'forecast') buildForecast();
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
  return {
    temp:  +sliders[0].value,
    hum:   +sliders[1].value,
    pres:  +sliders[2].value,
    wind:  +sliders[3].value,
    cloud: +sliders[4].value,
    uv:    +sliders[5].value
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
 * Usa gbModel() del módulo model.js (modo prototipo).
 * En producción: usar fetchPrediction() y hacer async.
 */
function predict() {
  const inp = getInputs();
  const r   = gbModel(inp);    // ← modelo.js

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
 * Llama a generateForecast() del módulo model.js.
 */
function buildForecast() {
  const inp      = getInputs();
  const forecast = generateForecast(inp);   // ← model.js
  const container = document.getElementById('forecast-row');
  container.innerHTML = '';

  const tempsHi = [], tempsLo = [];

  forecast.forEach(({ day, icon, hi, lo }) => {
    tempsHi.push(hi);
    tempsLo.push(lo);
    const d = document.createElement('div');
    d.className = 'forecast-day';
    d.innerHTML = `
      <div class="fd-day">${day}</div>
      <div class="fd-icon">${icon}</div>
      <div class="fd-hi">${hi}°</div>
      <div class="fd-lo">${lo}°</div>
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
//  HELPERS DOM
// ============================================================

/** Actualiza el textContent de un elemento por su ID */
function setEl(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}