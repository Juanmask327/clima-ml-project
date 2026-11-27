# 📋 Registro de trabajo · Predicción Climática ML

> **Última actualización:** 27/11/2026
> **Estado general:** ✅ Modelo entrenado con DATOS REALES de estaciones · 30/30 tests · desplegado

---

## 1. Resumen del estado

| Elemento | Estado | Detalle |
|----------|--------|---------|
| App local | ✅ | http://localhost:8000/ |
| App pública | ✅ | Túnel de Cloudflare (URL nueva en cada sesión; ver readme) |
| Modelo | ✅ | **REAL**: 381 horas de 3 estaciones BME280, XGBoost |
| API REST | ✅ | `/predict` `/forecast` `/model/info` `/health` (esquema temp/hum/pres + hora) |
| Frontend | ✅ | 3 sliders reales + hora automática del navegador |
| Tests | ✅ | 30/30 (`pytest backend/tests/`) |
| Docker | ✅ | Imagen con modelo real embebido + healthcheck |
| CI/CD | ✅ | Tests + imagen en GHCR (GitHub Actions) |
| Git | ✅ | Commits al día, árbol limpio |

**Repositorio:** https://github.com/Juanmask327/clima-ml-project

---

## 2. Métricas actuales del modelo (DATOS REALES)

```
Validación cruzada temporal (TimeSeriesSplit, 5 folds, sin shuffle)

R²   : 0.8262 ± 0.1054
RMSE : 2.1466 ± 0.9874 °C
MAE  : 1.3949 ± 0.6729 °C
```

**Selección de arquitectura (misma CV):**

| Arquitectura | RMSE | R² |
|--------------|------|-----|
| **XGBoost (elegido)** | **2.1466** | **0.8262** |
| Stacking (XGB+RF+Ridge) | 2.6215 | 0.7154 |
| Persistencia (t+1h = t) | 2.1537 | — |

**Hiperparámetros óptimos** (Optuna, 50 trials):

| Parámetro | Valor |
|-----------|-------|
| n_estimators | 459 |
| max_depth | 4 |
| learning_rate | 0.01368 |
| subsample | 0.7967 |
| colsample_bytree | 0.6913 |
| reg_alpha (L1) | 0.4653 |
| reg_lambda (L2) | 1.4729 |

**Importancia de features** (%): `heat_index` 40.1 · `temp_actual` 33.6 ·
`humedad` 11.0 · `hum_effect` 4.9 · `presion` 4.3 · `hour_cos` 4.3 ·
`hour_sin` 1.9

> ✅ El modelo supera al baseline de persistencia (2.1466 < 2.1537),
> requisito mínimo de un predictor de temperatura. El margen es
> estrecho por la cobertura limitada (17 días de muestreo en 19
> meses). Capturar más días debería ampliarlo.

---

## 3. Bugs encontrados y corregidos

### 🔴 Críticos (impedían usar la app)

**1. `API_BASE_URL` hardcodeado** — `frontend/js/model.js`
```js
const API_BASE_URL = 'http://localhost:8000';  // ❌
```
En la nube el navegador pedía `/predict` al puerto 8000 **del PC del
usuario**, no al servidor. Por eso no funcionaban predicción, pronóstico
ni métricas.

```js
const API_BASE_URL = window.location.origin;   // ✅ mismo origen
```

**2. HTML malformado — pestañas anidadas** — `frontend/index.html`
Un `<div>` sin cerrar (`#res-source`) dejaba abierto el bloque de
confianza, que **se tragaba** `#tab-forecast` y `#tab-model` dentro de
`#tab-predict`.

*Symptom:* el canvas del gráfico se medía con la pestaña oculta (0×0) →
Chart.js no dibujaba nada. Y al ocultar `tab-predict` se ocultaban
también las otras dos pestañas.

*Fix:* cerrar `<div id="res-source">` + el bloque de confianza, y esperar
a que el canvas tenga tamaño real (`whenCanvasReady()`).

### 🟡 Funcionales

| # | Bug | Archivo | Fix |
|---|-----|---------|-----|
| 3 | `probeApi()` corría **después** de `predict()` → la UI caía al modo simulado | `ui.js` | `await probeApi()` antes de `predict()` |
| 4 | `apiAvailable` no se reintentaba tras un fallo → un error transitorio dejaba la app en modo local para siempre | `ui.js` | cada petición reintenta la API |
| 5 | `switchTab()` no capturaba rechazos async → errores silenciosos | `ui.js` | `.catch()` en las cargas |
| 6 | `loadModelInfo()` marcaba "cargado" aunque la API fallara → nunca reintentaba | `ui.js` | solo marca cargado si la API respondió |
| 7 | `generateForecast()` no devolvía `rainProb` → faltaba el badge de lluvia | `model.js` | añadido |
| 8 | `getInputs()` leía sliders por posición → frágil ante cambios de HTML | `ui.js` / `index.html` | leen por `data-var` |
| 9 | `apiFetch()` sin timeout → requests colgados | `model.js` | `AbortController` 15 s |

### 🟠 Backend / infraestructura

| # | Bug | Fix |
|---|-----|-----|
| 10 | `TerminatedWorkerError` en entrenamiento: `StackingRegressor(n_jobs=-1)` + `RandomForest(n_jobs=-1)` anidados → ~24 workers, el SO los mata por memoria | Paralelismo moderado: stacking `n_jobs=4`, RF `n_jobs=1`, XGB `n_jobs=2` |
| 11 | `KFold` con shuffle → **fuga temporal** en la validación | `TimeSeriesSplit` (sin shuffle) |
| 12 | `rmse` era una fórmula inventada, no la métrica real | `rmse` = RMSE real de CV |
| 13 | Importancia de features mal normalizada (sumaba >100%) | normalizar cada base learner y promediar |
| 14 | `estimators_` con tuplas → incompatible con sklearn 1.9 | formato corregido |
| 15 | `✓` en stdout → `UnicodeEncodeError` en Windows (cp1252) | `sys.stdout.reconfigure(encoding='utf-8')` |
| 16 | GHCR: "repository name must be lowercase" | `IMAGE_NAME` en minúsculas en el workflow |
| 17 | `/` servía JSON en vez del frontend (ruta duplicada) | info de API movida a `/api/info` |

### 🔵 Bugs en mi propia implementación (detectados por los tests)

| # | Bug | Fix |
|---|-----|-----|
| 18 | Fahrenheit: solo aplicaba el factor, faltaba el offset (−17.78) | conversión con `(factor, offset)` |
| 19 | `shift(-1)` aplicado **antes** de limpiar → desalineaba X e y | limpiar primero, luego shift |
| 20 | `pd.to_numeric` fallaba con dtype `float64` | `.astype('float64')` explícito |

---

## 4. Componentes del modelo (actual, datos reales)

```
XGBoost (Gradient Boosting) — elegido automáticamente por CV
├── n_estimators=459, max_depth=4, learning_rate=0.0137
├── Optimizado por Optuna (50 trials)
└── Selección por TimeSeriesSplit: XGB (2.147) < persistencia (2.154) < stacking (2.622)
```

**Features (7):** `temp_actual` `humedad` `presion` `hour_sin`
`hour_cos` `hum_effect` `heat_index`

**Flujo de entrenamiento:** carga CSV real → feature engineering
(ciclo horario) → Optuna (50 trials, TimeSeriesSplit) → selección de
arquitectura (XGB vs stacking, misma CV) → evaluación → serialización
`.joblib` con `pipeline`, `features`, `metrics`, `n_samples`,
`feature_importances`, `xgb_params`, `mode`, `arch`, `persist_rmse`,
`hum_mean`.

**Esquema sintético (legacy):** sigue soportado (11 features con
viento/nubosidad/UV); el pipeline lo detecta y usa `temp_pred`.

---

## 5. Datos reales de estaciones (INTEGRADO y funcionando)

Los dumps crudos de las estaciones Raspberry Pi (sensor **BME280**:
temperatura, humedad, presión — NO mide viento/nubosidad/UV) se
procesan con dos scripts:

| Script | Función |
|--------|---------|
| `data/parse_stations.py` | Unifica los 4 dumps (3 SQL phpMyAdmin + 1 CSV de una línea) → `weather_data_real.csv` (432,829 registros) |
| `data/prepare_real_dataset.py` | Agrega por hora + target `temp_next` (solo si la hora siguiente es contigua) → `weather_data.csv` (381 filas) |

### Cobertura real de muestreo

| Estación | Registros | Horas | Periodo |
|----------|-----------|-------|---------|
| rasberry_1 | 191,106 | 76 | 2023-05-31 → 2024-12-26 |
| rasberry_3 | 113,831 | 124 | 2023-05-31 → 2024-12-30 |
| rasberry_4 | 127,892 | 181 | 2024-12-17 → 2024-12-31 |

> Las estaciones registran cada 5 s pero **no operaron continuo**:
> los datos se concentran en 17 días de muestreo intenso entre mayo
> 2023 y diciembre 2024. Presión ~750 hPa → estación en altura.

### Columnas del dataset de entrenamiento

`estacion, time, temp_actual, humedad, presion, temp_next`

### Para re-entrenar (todo el pipeline)

```bash
python backend/data/parse_stations.py        # crudos → weather_data_real.csv
python backend/data/prepare_real_dataset.py  # → weather_data.csv horario
python backend/model/train_model.py          # entrena (detecta modo real)
```

---

## 6. Despliegue

### Archivos de despliegue incluidos

| Archivo | Plataforma |
|---------|-----------|
| `render.yaml` | Render (Blueprint) — botón de un clic en el readme |
| `fly.toml` | Fly.io |
| `Procfile` | Railway / Heroku |
| `Dockerfile` | Docker / K8s / VPS |
| `runtime.txt` | Versión de Python (3.12.8) |
| `start_deploy.ps1` | Script local: build + contenedor + túnel |
| `.github/workflows/ci-cd.yml` | Tests + publicación a GHCR |

### URL pública actual

```
https://counters-capitol-ago-institution.trycloudflare.com
```

> ⚠️ **Temporal.** Túnel de Cloudflare sin cuenta: funciona mientras el
> equipo esté encendido, sin garantía de uptime. Para URL fija hay que
> iniciar sesión en Render / Railway / Fly.io.

---

## 7. Estructura del proyecto

```
clima_ml_project/
├── backend/
│   ├── model/
│   │   ├── train_model.py       # entrenamiento: esquema real y sintético,
│   │   │                        #   Optuna + selección de arquitectura por CV
│   │   └── clima_model.joblib   # modelo REAL entrenado (versionado)
│   ├── api/
│   │   └── main.py              # FastAPI: /predict /forecast /model/info /health
│   ├── data/
│   │   ├── estacion_1.sql       # dumps crudos de las estaciones (evidencia)
│   │   ├── estacion_2.sql
│   │   ├── estacion4.sql
│   │   ├── estacion3-30-12-2024_actua.csv
│   │   ├── parse_stations.py    # crudos → weather_data_real.csv
│   │   ├── prepare_real_dataset.py  # → dataset horario de entrenamiento
│   │   ├── weather_data.csv     # 381 horas reales (dataset activo)
│   │   └── weather_data_synth.csv   # respaldo del sintético original
│   └── tests/
│       ├── conftest.py
│       ├── test_api.py          # 16 tests de la API (esquema real + legacy)
│       └── test_data_loading.py # 14 tests del cargador legacy
├── frontend/
│   ├── index.html              # 3 sliders BME280 + hora automática
│   ├── css/styles.css
│   └── js/
│       ├── model.js             # fetch a la API + respaldo local
│       └── ui.js                # controlador UI, tabs, Chart.js
├── Requirements.txt             # solo producción
├── Requirements-dev.txt         # + análisis, notebooks, testing
└── render.yaml / fly.toml / Procfile / Dockerfile / start_deploy.ps1
```

---

## 8. Comandos útiles

```bash
# ── Entrenar ───────────────────────────────────────────────
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -u backend\model\train_model.py      # ~10-12 min

# ── API (desde backend/) ───────────────────────────────────
..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# ── Con el comando de despliegue (Render/Railway) ──────────
.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir backend --host 0.0.0.0 --port 8000

# ── Tests (desde backend/) ─────────────────────────────────
..\.venv\Scripts\python.exe -m pytest tests\ -v

# ── Docker ─────────────────────────────────────────────────
docker build -t clima-ml .
docker run -d --name clima-ml-test -p 8099:8000 clima-ml

# ── Git ────────────────────────────────────────────────────
git add -A && git commit -m "..." && git push origin master
```

> ⚠️ **Al hacer `curl` en PowerShell** el JSON con comillas se rompe:
> usa `-d '{\"temp\":22,...}'` (comillas escapadas con `\`).

---

## 9. Pendientes

### ✅ Completado en esta sesión

- [x] **Integración de datos reales de estaciones** — parser de dumps,
      dataset horario, re-entrenamiento y despliegue del modelo REAL
- [x] **Comparar modelos con el mismo CV** — XGBoost vs stacking vs
      persistencia (XGB gana: 2.147 < 2.154 < 2.622)
- [x] **Métricas reales en readme y frontend**

### 🔴 Requiere tu intervención

- [ ] **URL permanente** — iniciar sesión en Render / Railway / Fly.io
      (necesito tu contraseña o un `FLY_API_TOKEN` / `RENDER_API_KEY`).
      El repo ya está listo para despliegue con un clic.
- [ ] **Capturar más días de datos** — el margen sobre el baseline de
      persistencia es estrecho (2.147 vs 2.154) por la cobertura de
      solo 17 días. Más días de muestreo continuo lo ampliarían.

### 🟡 Opcional (para la tesis)

- [ ] **SHAP / explicabilidad** — para la sección de discusión
- [ ] **Notebook de EDA** — análisis exploratorio de los datos reales
- [ ] **Validación con holdout** — reservar un periodo para test final
- [ ] **Modelo LSTM** — comparar contra XGBoost con más datos

### 🔵 Técnico

- [ ] Registrar `cloudflared` como servicio de Windows (requiere admin)
- [ ] Añadir el `.joblib` a un almacenamiento externo si el repo crece
      (GitHub tiene límite de 100 MB por archivo)

---

## 10. Historial de commits

```
6bdbf3c  Pipeline de datos reales: cargador flexible para CSV de estaciones
3f23c76  Fix GHCR: la imagen debe ir en minusculas
291a5e9  Script de despliegue local + documentacion de la URL publica
7e211b4  Requirements separadas (prod/dev) + CI/CD en GitHub Actions
5a5545c  Fix HTML mal formado: las pestañas forecast y model estaban anidadas
8a3fd59  Fix frontend para despliegue en la nube + configs de despliegue
05b948e  Fix ML, frontend conectado a la API, tests y limpieza del repo
11f9b6c  El primer commit del sitio web, predicción del clima en micro-zonas
```

---

## 11. Cómo retomar

1. `git pull` en `H:\doctorado\estudiar\trabajo_2025_2026\clima_ml_project`
2. Levantar la API: `..\.venv\Scripts\python.exe -m uvicorn api.main:app` (desde `backend/`)
3. Abrir http://localhost:8000/
4. Si el túnel no está vivo:
   `.\start_deploy.ps1` (desde la raíz del proyecto)

---

*Documento generado el 26/11/2026. Actualizar al cerrar cada bloque de trabajo.*
