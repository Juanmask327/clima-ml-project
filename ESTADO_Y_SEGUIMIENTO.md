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

### Notas de sensibilidad (barridos del modelo entrenado)

- **Hora del día:** respuesta suave y físicamente coherente —
  ~20.1 °C de madrugada, pico ~21.9 °C a las 14:00, descenso
  nocturno. El ciclo diario se aprendió correctamente (los features
  `hour_sin`/`hour_cos` funcionan como se esperaba).
- **Presión:** salto de umbral en ~751 hPa (19.3 → 21.9 °C entre 750
  y 752 hPa). Con 381 horas, los árboles usan la presión como
  indicador de régimen (días/estaciones distintas), no como driver
  físico continuo. Limitación conocida de árboles con pocos datos;
  documentada para la discusión de la tesis.

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

### 🟣 Bugs e incidentes de la integración de datos reales (27/11)

| # | Bug / Incidente | Fix / Decisión |
|---|-----------------|----------------|
| 21 | `prepare_real_dataset.py`: usé `groupby.diff(-1)` comparado contra `+1h` — matemáticamente invertido → **0 filas con target válido** | `hora.shift(-1) - hora` y comparar con `+1h`; los tests lo cubren |
| 22 | Mis tests de `test_data_loading.py` asumían que el CSV traía `temp_pred` → conteos de filas incorrectos (3 fallos falsos) | tests corregidos: el target por shift descarta la última fila |
| 23 | `joblib/loky` crashea de forma intermitente en Windows + Python 3.14 (`resource_tracker: FileNotFoundError`) al evaluar el stacking | try/except alrededor de la evaluación de cada arquitectura; el candidato fallido se descarta y se reporta |
| 24 | El shell del agente mató procesos al expirar su timeout (uvicorn y el primer entrenamiento murieron así) | lanzar con timeout desactivado en procesos largos |
| 25 | **El disco H: se desconectó a mitad del entrenamiento** (USB) → FileNotFoundError en loky + archivos inaccesibles; todo el trabajo no commiteado quedó atrapado en el disco | el usuario reconectó el disco y NO se perdió nada; lección: commitear más a menudo y revisar el cable |
| 26 | Esquema de presión viejo rechazaba los ~750 hPa reales (rango 970–1050 pensado para nivel del mar) | rango ampliado a 600–1100 hPa; slider del frontend 700–800 |
| 27 | El stacking (XGB + RF 300 árboles + Ridge) **sobreajusta** con 381 horas: peor que el baseline naive | selección automática de arquitectura por CV: gana XGBoost solo (RMSE 2.147) |

---

## 3b. Registro de avances por sesión

### Sesión del 26/11/2026
1. ✅ Revisión completa del proyecto + informe de hallazgos
2. ✅ Entorno `.venv` + dependencias pineadas (Py3.14)
3. ✅ Fixes ML: TimeSeriesSplit, RMSE real, importancias, paralelismo
4. ✅ Frontend conectado a la API con fallback + badge de estado
5. ✅ 9 tests pytest
6. ✅ Entrenamiento (sintético): R² 0.9309, RMSE 1.1138
7. ✅ Limpieza de repo + `.gitignore` + primer commit del trabajo

### Sesión del 26-27/11 (noche)
1. ✅ Despliegue: `render.yaml`, `fly.toml`, `Procfile`, `Dockerfile`
2. ✅ Fix crítico `API_BASE_URL` → `window.location.origin`
3. ✅ Fix crítico HTML malformado (pestañas anidadas, canvas 0×0)
4. ✅ Repo GitHub creado + CI/CD (tests + GHCR)
5. ✅ Túnel público verificado end-to-end en navegador
6. ✅ `start_deploy.ps1` + documento de seguimiento

### Sesión del 27/11/2026 — **INTEGRACIÓN DE DATOS REALES** 🎯
1. ✅ Parser de los 4 dumps crudos de estaciones (3 SQL phpMyAdmin +
   1 CSV de una línea): **432,829 registros** de 3 Raspberry Pi
2. ✅ Hallazgo: sensor BME280 mide solo temp/hum/presión → el modelo
   se reestructura a 7 features (3 medidas + ciclo horario + 2 derivadas)
3. ✅ Dataset horario: 381 filas con target contiguo (17 días de muestreo)
4. ✅ Hallazgo: presión ~750 hPa → estación en altura (~2.500 m)
5. ✅ `train_model.py`: esquema dual (real/sintético) + selección
   automática de arquitectura por CV (XGB vs stacking)
6. ✅ **Resultado: XGBoost RMSE 2.1466, R² 0.8262 — supera el baseline
   de persistencia (2.1537); el stacking sobreajusta (2.6215)**
7. ✅ API: input `temp/hum/pres + hora opcional` (hora del navegador),
   `hora_usada` en la respuesta, `mode`/`arch`/`persist_rmse` en `/model/info`
8. ✅ Frontend: 3 sliders reales, métricas/importancias reales
9. ✅ 30/30 tests (16 API + 14 cargador legacy)
10. ✅ Docker reconstruido con modelo real (0.7 MB vs 10 MB)
11. ✅ Análisis de sensibilidad: ciclo horario suave y físico;
    umbral de presión en ~751 hPa (artefacto de split de árbol)
12. ⚠️ Incidente: disco H: se desconectó a mitad del entrenamiento
    (sin pérdida de datos, reconectado por el usuario)

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

Los túneles rápidos de Cloudflare **generan una URL nueva en cada
sesión** (y expiran con reinicios/desconexiones del equipo). La última
fue `modelling-buffalo-adapter-favors.trycloudflare.com`, pero NO hay
que memorizarla:

```powershell
.\start_deploy.ps1   # desde la raíz del proyecto: imprime la URL nueva
```

> ⚠️ **Temporal.** Túnel de Cloudflare sin cuenta: funciona mientras el
> equipo esté encendido, sin garantía de uptime. Para URL fija hay que
> iniciar sesión en Render / Railway / Fly.io (pendiente, requiere tu
> login).

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
# ── Pipeline completo de datos reales (desde la raíz) ───────
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe backend\data\parse_stations.py         # crudos → weather_data_real.csv
.venv\Scripts\python.exe backend\data\prepare_real_dataset.py    # → dataset horario
.venv\Scripts\python.exe -u backend\model\train_model.py         # entrena (~15 min, modo real)

# ── API (desde backend/) ───────────────────────────────────
..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# ── Con el comando de despliegue (Render/Railway) ──────────
.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir backend --host 0.0.0.0 --port 8000

# ── Tests (desde backend/) ─────────────────────────────────
..\.venv\Scripts\python.exe -m pytest tests\ -v

# ── Docker ─────────────────────────────────────────────────
docker build -t clima-ml .
docker rm -f clima-ml-test; docker run -d --name clima-ml-test -p 8099:8000 clima-ml

# ── Despliegue local + túnel público ────────────────────────
.\start_deploy.ps1

# ── Git ────────────────────────────────────────────────────
git add -A && git commit -m "..." && git push origin master
```

> ⚠️ **Al hacer `curl` en PowerShell** el JSON con comillas se rompe:
> usa `-d '{\"temp\":22,...}'` (comillas escapadas con `\`).
>
> ⚠️ **Procesos largos** (entrenamiento, uvicorn en background): no
> ponerles timeout corto o el wrapper los mata a mitad.

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
905bf69  Notas de sensibilidad del modelo real (ciclo horario vs umbral de presion)
93da56a  Modelo entrenado con DATOS REALES de las estaciones BME280
ed68749  Nueva URL del tunel tras reinicio del equipo
9c8a280  Documento de estado y seguimiento para continuar el trabajo
6bdbf3c  Pipeline de datos reales: cargador flexible para CSV de estaciones
3f23c76  Fix GHCR: la imagen debe ir en minusculas
291a5e9  Script de despliegue local + documentacion de la URL publica
7e211b4  Requirements separadas (prod/dev) + CI/CD en GitHub Actions
5a5545c  Fix HTML mal formado: las pestañas forecast y model estaban anidadas
8a3fd59  Fix frontend para despliegue en la nube + configs de despliegue
05b948e  Fix ML, frontend conectado a la API, tests y limpieza del repo
11f9b6c  El primer commit del sitio web, predicción del clima en micro-zonas
```

**Hitos:**
- `11f9b6c` → proyecto original (sintético, backend+frontend sueltos)
- `8a3fd59` → desplegable en la nube (fix `API_BASE_URL`)
- `5a5545c` → las 3 pestañas funcionan (fix HTML)
- `93da56a` → **el modelo aprende de tus estaciones reales**

---

## 11. Cómo retomar mañana

1. **Verificar el disco H:** — se desconectó una vez; si el proyecto
   no aparece, revisa el cable/USB antes que nada
2. `git pull` en `H:\doctorado\estudiar\trabajo_2025_2026\clima_ml_project`
3. Levantar la API: `..\.venv\Scripts\python.exe -m uvicorn api.main:app`
   (desde `backend/`) — sin timeout en el wrapper
4. Abrir http://localhost:8000/ — debe decir "Modelo ML en vivo"
5. Si quieres URL pública: `.\start_deploy.ps1` (desde la raíz)
6. Los pendientes están en la sección 9; el estado del modelo en la 2

**Sesión sugerida (bloques cortos):**
- ¿URL permanente? → login en Render (10 min con el botón del readme)
- ¿Más datos de estaciones? → 3 comandos del pipeline (sección 8)
- ¿Avance de tesis? → SHAP o notebook de EDA sobre los datos reales

---

*Documento actualizado el 27/11/2026. Actualizar al cerrar cada bloque de trabajo.*
