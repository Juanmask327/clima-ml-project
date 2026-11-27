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
  HUM_MEAN:        68.8,   // media de humedad de las estaciones reales
  CONF_MAX:        97,      // % confianza máxima
  CONF_MIN:        72,      // % confianza mínima
  RMSE_BASE:       1.0,     // RMSE base estimado (modo local)
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
//  MODELO SIMULADO (prototipo, respaldo sin API)
// ============================================================

/**
 * Estimación local de la temperatura de la siguiente hora a partir
 * de las variables reales del sensor BME280 y la hora del día.
 * @param {Object} inp - { temp, hum, pres, hora }
 * @returns {Object} Resultado de la predicción
 */
function gbModel(inp) {
  const { temp, hum, pres, hora } = inp;
  const h = hora != null ? hora : new Date().getHours();

  // Ciclo diario: enfría de madrugada, calienta tras mediodía
  const ciclo = Math.sin((h - 9) / 24 * 2 * Math.PI) * 1.2;

  // La presión alta tiende a estabilizar (menos cambio)
  const estabilidad = (pres - 752) / 50 * 0.3;

  const predTemp = round1(temp + ciclo * 0.5 + estabilidad);

  // Probabilidad de lluvia: la humedad es el principal proxy
  const rainProb = clamp(Math.round((hum - 40) * 1.6), 0, 100);

  // Sensación térmica: humedad alta aumenta la percepción de calor
  const humEffect = (hum - MODEL_CONFIG.HUM_MEAN) / 100;
  const sensacion = round1(predTemp - humEffect * 2);

  // Confianza heurística
  const conf = clamp(
    Math.round(94 - Math.abs(hum - 50) / 8), MODEL_CONFIG.CONF_MIN, MODEL_CONFIG.CONF_MAX
  );
  const rmse = round2(MODEL_CONFIG.RMSE_BASE);

  const { icon, desc } = classifyWeather(predTemp, rainProb);

  return { predTemp, rainProb, sensacion, conf, rmse, icon, desc, horaUsada: h };
}

/**
 * Clasifica la condición climática resultante.
 * @param {number} temp     - Temperatura predicha (°C)
 * @param {number} rainProb - Probabilidad de lluvia (%)
 * @returns {{ icon: string, desc: string }}
 */
function classifyWeather(temp, rainProb) {
  if (rainProb > 70)  return { icon: '🌧️', desc: 'Lluvia probable' };
  if (rainProb > 45)  return { icon: '⛅', desc: 'Parcialmente nublado con posible lluvia' };
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
    conf:      d.confidence,
    rmse:      d.rmse,
    icon:      d.icon,
    desc:      d.condition,
    horaUsada: d.hora_usada,
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
    // Probabilidad de lluvia estimada desde la humedad (heurística)
    const rainProb = clamp(
      Math.round((inp.hum - 40) * 1.6 - i * 2), 0, 100
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