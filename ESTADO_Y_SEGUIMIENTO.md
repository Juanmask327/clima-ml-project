# 📋 Registro de trabajo · Predicción Climática ML

> **Última actualización:** 26/11/2026
> **Estado general:** ✅ Funcionando y desplegado · 26/26 tests · CI/CD en verde

---

## 1. Resumen del estado

| Elemento | Estado | Detalle |
|----------|--------|---------|
| App local | ✅ | http://localhost:8000/ |
| App pública | ✅ | https://breakfast-physician-bacterial-sum.trycloudflare.com |
| API REST | ✅ | `/predict` `/forecast` `/model/info` `/health` |
| Frontend | ✅ | 3 pestañas verificadas en navegador real |
| Tests | ✅ | 26/26 (`pytest backend/tests/`) |
| Docker | ✅ | Contenedor `healthy` + healthcheck |
| CI/CD | ✅ | Tests + imagen en GHCR |
| Git | ✅ | 8 commits, árbol limpio, todo pusheado |

**Repositorio:** https://github.com/Juanmask327/clima-ml-project

---

## 2. Métricas actuales del modelo

```
Validación cruzada temporal (TimeSeriesSplit, 5 folds, sin shuffle)

R²   : 0.9309 ± 0.0453
RMSE : 1.1138 ± 0.3587 °C
MAE  : 0.8417 ± 0.1976 °C
```

**Hiperparámetros óptimos** (Optuna, 50 trials):

| Parámetro | Valor |
|-----------|-------|
| n_estimators | 571 |
| max_depth | 3 |
| learning_rate | 0.05124 |
| subsample | 0.6289 |
| colsample_bytree | 0.6851 |
| reg_alpha (L1) | 0.4619 |
| reg_lambda (L2) | 1.0292 |

**Importancia de features** (%): `temp_actual` 81.3 · `heat_index` 15.3 ·
`indice_uv` 1.1 · `nubosidad` 0.5 · `cloud_effect` 0.5 · `humedad` 0.4

> ⚠️ **Estas métricas son de datos SINTÉTICOS.** El target deriva de
> `temp_actual`, de ahí el 81% de importancia y el R² alto. Con datos
> reales el R² **bajará** — es lo correcto y esperable para la tesis.

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

## 4. Componentes del modelo

```
Stacking Ensemble
├── Base learner 1 : XGBoost (GBR)      n_estimators=571, max_depth=3
├── Base learner 2 : Random Forest      n_estimators=300, max_depth=8
└── Meta-learner   : Ridge Regression   alpha=1.0
```

**Features (11):** `temp_actual` `humedad` `presion` `viento` `nubosidad`
`indice_uv` `pressure_delta` `hum_effect` `cloud_effect` `wind_chill`
`heat_index`

**Flujo de entrenamiento:** carga CSV → feature engineering → Optuna
(50 trials, TimeSeriesSplit) → stacking → evaluación CV → serialización
`.joblib` con `pipeline`, `features`, `metrics`, `n_samples`,
`feature_importances`, `xgb_params`.

---

## 5. Datos reales de estaciones (implementado)

El pipeline ya está preparado. Solo hay que dejar el CSV en
`backend/data/weather_data.csv` y re-entrenar.

### Columnas requeridas

| Interna | Significado | Unidad |
|---------|-------------|--------|
| `temp_actual` | Temperatura observada | °C |
| `humedad` | Humedad relativa | % |
| `presion` | Presión atmosférica | hPa |
| `viento` | Velocidad del viento | km/h |
| `nubosidad` | Nubosidad | % |
| `indice_uv` | Índice UV | 0–12 |
| `temp_pred` | *(opcional)* temp de la siguiente hora | °C |

### Adaptaciones automáticas

- **Alias de columnas** — `temperatura`/`temp`/`tamb`, `HR`/`rh`,
  `pressure`/`p_atm`/`bar`, `wind_speed`/`vv`, `cloud`, `uv_index`…
- **Unidades** — con columna `presion_unit`/`temp_unit` (`kPa`, `Pa`,
  `bar`, `F`, `K`…) se convierte a hPa y °C
- **Target** — si falta `temp_pred`, se construye con `shift(-1)`
- **Limpieza** — descarta NaN, no-numéricos y outliers fuera de rango
  físico (humedad 250%, presión 5000 hPa…)
- **Orden temporal** — las filas nunca se reordenan
- **Errores accionables** — indica qué columna falta y qué alias acepta

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
https://breakfast-physician-bacterial-sum.trycloudflare.com
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
│   │   ├── train_model.py       # entrenamiento + load_real_data() + Optuna
│   │   └── clima_model.joblib   # modelo entrenado (10.11 MB, versionado)
│   ├── api/
│   │   └── main.py              # FastAPI: /predict /forecast /model/info /health
│   ├── data/
│   │   └── weather_data.csv     # ← aquí va el CSV de la estación real
│   └── tests/
│       ├── conftest.py
│       ├── test_api.py          # 12 tests de la API
│       └── test_data_loading.py # 14 tests del cargador de datos
├── frontend/
│   ├── index.html
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

### 🔴 Requiere tu intervención

- [ ] **URL permanente** — iniciar sesión en Render / Railway / Fly.io
      (necesito tu contraseña o un `FLY_API_TOKEN` / `RENDER_API_KEY`).
      El repo ya está listo para despliegue con un clic.
- [ ] **CSV de la estación real** — colocarlo en
      `backend/data/weather_data.csv` y re-entrenar. Revisar el impacto
      en las métricas del readme.

### 🟡 Opcional (para la tesis)

- [ ] **SHAP / explicabilidad** — para la sección de discusión
- [ ] **Notebook de EDA** — análisis exploratorio de los datos reales
- [ ] **Validación con holdout** — reservar un periodo para test final
- [ ] **Comparar modelos** — XGBoost vs RF vs Stacking (para justificar
      la elección del ensemble)
- [ ] **Actualizar métricas del readme** tras entrenar con datos reales

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
