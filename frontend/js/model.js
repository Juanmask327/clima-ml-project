/**
 * ============================================================
 * PREDICCIÓN CLIMÁTICA ML · MICRO-ZONA
 * Módulo: Modelo GBR (simulación frontend)
 * ============================================================
 * Este módulo implementa la lógica del modelo Gradient Boosting
 * Regressor en JavaScript para prototipo frontend.
 *
 * EN PRODUCCIÓN:
 *   - Reemplazar gbModel() por una llamada fetch() a la API
 *     Python (FastAPI), que ejecuta el modelo XGBoost real.
 *   - Ver: backend/api/main.py  →  POST /predict
 * ============================================================
 */

'use strict';

// ─── Constantes del modelo ──────────────────────────────────
const MODEL_CONFIG = {
  PRESSURE_REF:    1013,    // hPa referencia estándar
  HUMIDITY_REF:    50,      // % referencia neutra
  WIND_CHILL_THR:  20,      // km/h umbral para wind chill
  WIND_CHILL_RATE: 0.15,    // factor de enfriamiento por viento
  CONF_MAX:        97,      // % confianza máxima
  CONF_MIN:        72,      // % confianza mínima
  RMSE_BASE:       1.8,     // RMSE base estimado
};

// ─── API endpoint ───────────────────────────────────────────
// Usa el MISMO origen que sirvió la página (URL relativa). Así
// funciona igual en local y en cualquier despliegue en la nube
// (Render/Railway/Fly.io), sin recompilar ni cambiar nada.
const API_BASE_URL = window.location.origin;

// Si la UI se sirve como archivo local (file://), el origen es
// "null" y no hay API: se usará el modo local simulado.
const API_ENABLED = /^https?:$/.test(window.location.protocol);

// ============================================================
//  MODELO GBR SIMULADO (prototipo)
// ============================================================

/**
 * Ejecuta la predicción climática con el modelo GBR simulado.
 * @param {Object} inp - Variables meteorológicas de entrada
 * @param {number} inp.temp    - Temperatura actual (°C)
 * @param {number} inp.hum     - Humedad relativa (%)
 * @param {number} inp.pres    - Presión atmosférica (hPa)
 * @param {number} inp.wind    - Velocidad del viento (km/h)
 * @param {number} inp.cloud   - Nubosidad (%)
 * @param {number} inp.uv      - Índice UV
 * @returns {Object} Resultado de la predicción
 */
function gbModel(inp) {
  const { temp, hum, pres, wind, cloud, uv } = inp;

  // ── Feature engineering (mismo preprocesamiento que Python) ─
  const pressureDelta = (pres - MODEL_CONFIG.PRESSURE_REF) / 40;
  const humEffect     = (hum  - MODEL_CONFIG.HUMIDITY_REF) / 100;
  const cloudEffect   = cloud / 100;
  const windChill     = wind > MODEL_CONFIG.WIND_CHILL_THR
    ? -(wind - MODEL_CONFIG.WIND_CHILL_THR) * MODEL_CONFIG.WIND_CHILL_RATE
    : 0;

  // ── Predicción temperatura (regresión GBR) ───────────────
  const predTemp = round1(
    temp + pressureDelta * 2 - humEffect * 1.5 - cloudEffect * 2 + windChill
  );

  // ── Probabilidad de lluvia (clasificador auxiliar) ───────
  const rainProb = clamp(
    Math.round(hum * 0.6 + cloud * 0.4 - (pres - 970) * 0.3), 0, 100
  );

  // ── Variables derivadas ──────────────────────────────────
  const sensacion = round1(predTemp - windChill * 0.5 - humEffect * 2);
  const windPred  = Math.round(wind * (1 + (MODEL_CONFIG.PRESSURE_REF - pres) / 300));

  // ── Métricas de confianza ────────────────────────────────
  const conf = clamp(
    Math.round(94 - Math.abs(pressureDelta) * 5 - cloudEffect * 8),
    MODEL_CONFIG.CONF_MIN,
    MODEL_CONFIG.CONF_MAX
  );
  const rmse = round2(MODEL_CONFIG.RMSE_BASE - conf / 100);

  // ── Clasificación de condición climática ─────────────────
  const { icon, desc } = classifyWeather(predTemp, rainProb, cloud, wind);

  return { predTemp, rainProb, sensacion, windPred, conf, rmse, icon, desc };
}

/**
 * Clasifica la condición climática resultante.
 * @param {number} temp     - Temperatura predicha (°C)
 * @param {number} rainProb - Probabilidad de lluvia (%)
 * @param {number} cloud    - Nubosidad (%)
 * @param {number} wind     - Velocidad del viento (km/h)
 * @returns {{ icon: string, desc: string }}
 */
function classifyWeather(temp, rainProb, cloud, wind) {
  if (rainProb > 70)  return { icon: '🌧️', desc: 'Lluvia probable' };
  if (rainProb > 45)  return { icon: '⛅', desc: 'Parcialmente nublado con posible lluvia' };
  if (cloud > 60)     return { icon: '☁️', desc: 'Nublado' };
  if (wind > 40)      return { icon: '💨', desc: 'Vientoso' };
  if (temp > 32)      return { icon: '🌡️', desc: 'Caluroso y seco' };
  if (temp < 5)       return { icon: '🌨️', desc: 'Frío · posible helada' };
  return              { icon: '☀️', desc: 'Despejado y soleado' };
}

// ============================================================
//  LLAMADA REAL A LA API (FastAPI + XGBoost)
// ============================================================

/**
 * Wrapper de fetch con timeout y errores claros.
 * @param {string} path - Ruta del endpoint (ej. '/predict')
 * @param {Object} opts - Opciones de fetch
 * @returns {Promise<Response>} Respuesta HTTP
 */
async function apiFetch(path, opts = {}) {
  if (!API_ENABLED) {
    throw new Error('API no disponible (página abierta como file://)');
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...opts,
      signal: controller.signal,
      cache: 'no-store'
    });
    if (!response.ok) {
      let detail = '';
      try { detail = (await response.json()).detail || ''; } catch (_) {}
      throw new Error(`API ${response.status}${detail ? ': ' + detail : ''}`);
    }
    return response;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Llama al endpoint real de la API Python y normaliza la
 * respuesta al mismo formato que gbModel() para que la UI
 * no distinga entre modo API y modo local.
 *
 * @param {Object} inp - Variables de entrada
 * @returns {Promise<Object>} Predicción del servidor
 */
async function fetchPrediction(inp) {
  const response = await apiFetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp)
  });
  const d = await response.json();
  return {
    predTemp:  d.pred_temp,
    rainProb:  d.rain_prob,
    sensacion: d.sensacion,
    windPred:  d.wind_pred,
    conf:      d.confidence,
    rmse:      d.rmse,
    icon:      d.icon,
    desc:      d.condition,
    source:    'api'
  };
}

/**
 * Llama al endpoint /forecast de la API (días reales, reproducible).
 * @param {Object} inp - Variables de entrada
 * @returns {Promise<Array<Object>>} Array de 7 días {day, icon, hi, lo, rainProb}
 */
async function fetchForecast(inp) {
  const response = await apiFetch('/forecast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp)
  });
  const d = await response.json();
  return d.forecast.map(f => ({
    day: f.day, icon: f.icon, hi: f.hi, lo: f.lo, rainProb: f.rain_prob
  }));
}

/**
 * Llama a /model/info: métricas reales de CV, importancia de
 * features e hiperparámetros óptimos (Optuna).
 * @returns {Promise<Object>} Información del modelo
 */
async function fetchModelInfo() {
  const response = await apiFetch('/model/info');
  return response.json();
}

// ============================================================
//  GENERACIÓN DE PRONÓSTICO 7 DÍAS
// ============================================================

/**
 * Genera pronóstico de 7 días basado en los inputs actuales
 * usando interpolación temporal con patrones sinusoidales.
 * (Modo local de respaldo cuando la API no está disponible.)
 *
 * @param {Object} inp - Variables de entrada actuales
 * @returns {Array<Object>} Array de 7 días con hi/lo/icon
 */
function generateForecast(inp) {
  const days  = ['Hoy', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];
  const icons = ['☀️',  '⛅',  '🌧️', '🌤️', '☀️',  '⛅',  '🌦️'];

  return days.map((day, i) => {
    const variation = Math.sin(i * 0.8) * 3 + Math.cos(i * 1.2) * 2;
    const hi = round1(inp.temp + variation);
    const lo = round1(hi - 6 - Math.random() * 3);
    // Probabilidad de lluvia estimada (misma heurística que la API)
    const rainProb = clamp(
      Math.round(inp.hum * 0.6 + inp.cloud * 0.4 - (inp.pres - 970) * 0.3 - i * 2),
      0, 100
    );
    return { day, icon: icons[i], hi, lo, rainProb };
  });
}

// ============================================================
//  HELPERS
// ============================================================

/** Redondea a 1 decimal */
const round1 = v => Math.round(v * 10) / 10;

/** Redondea a 2 decimales */
const round2 = v => Math.round(v * 100) / 100;

/** Clamp entre min y max */
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

// Exportar para uso en Node.js / testing
if (typeof module !== 'undefined') {
  module.exports = {
    gbModel, classifyWeather, fetchPrediction, fetchForecast,
    fetchModelInfo, generateForecast
  };
}