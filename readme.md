# Predicción Climática ML · Micro-zona

Sistema de predicción del clima para micro-zonas usando un **Stacking Ensemble**
(XGBoost + Random Forest + Ridge Regression), expuesto como API REST con FastAPI
y con interfaz web en HTML + JavaScript puro.

| | |
|---|---|
| 🌐 **App pública (temporal)** | https://breakfast-physician-bacterial-sum.trycloudflare.com |
| 📦 **Repositorio** | https://github.com/Juanmask327/clima-ml-project |
| 📊 **Métricas (CV temporal)** | R² 0.931 · RMSE 1.114 °C · MAE 0.842 °C |
| 🧪 **Tests** | 26/26 (`pytest backend/tests/`) |

> 📋 **[ESTADO_Y_SEGUIMIENTO.md](ESTADO_Y_SEGUIMIENTO.md)** — registro de
> trabajo: bugs corregidos, métricas, comandos y pendientes.

---

## Estructura del proyecto

```
clima_ml_project/
│
├── frontend/                   # Interfaz web (cliente)
│   ├── index.html              # HTML principal
│   ├── css/
│   │   └── styles.css          # Estilos (variables CSS, dark mode, responsive)
│   └── js/
│       ├── model.js            # fetchPrediction/fetchForecast/fetchModelInfo + respaldo local
│       └── ui.js               # Controlador de UI, tabs, gráficos (Chart.js)
│
├── backend/                    # Servidor Python
│   ├── model/
│   │   ├── train_model.py      # Entrenamiento, Optuna, evaluación, serialización
│   │   └── clima_model.joblib  # Modelo entrenado (generado al ejecutar train)
│   ├── api/
│   │   └── main.py             # API REST FastAPI: /predict /forecast /model/info
│   ├── tests/
│   │   ├── conftest.py         # Configuración de pytest
│   │   └── test_api.py         # Tests de la API (pytest + TestClient)
│   └── data/
│       └── weather_data.csv    # Dataset (generado automáticamente si no existe)
│
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

---

## 📡 Conectar datos reales de una estación

El pipeline de entrenamiento ya soporta datos reales. Para usarlos,
coloca el CSV de tu estación en `backend/data/weather_data.csv` y
re-entrena:

```bash
python backend/model/train_model.py
```

### Columnas requeridas

| Columna interna | Significado | Unidad |
|-----------------|-------------|--------|
| `temp_actual`   | Temperatura observada | °C |
| `humedad`       | Humedad relativa | % |
| `presion`       | Presión atmosférica | hPa |
| `viento`        | Velocidad del viento | km/h |
| `nubosidad`     | Nubosidad | % |
| `indice_uv`     | Índice UV | 0–12 |
| `temp_pred`     | *(opcional)* Objetivo: temp de la siguiente hora | °C |

### El cargador se adapta solo

`load_real_data()` en `backend/model/train_model.py` reconoce nombres
de columna habituales de una estación, así que **no hace falta
renombrar nada a mano**:

| Interno | Alias aceptados |
|---------|-----------------|
| `temp_actual` | `temp`, `temperatura`, `temperature`, `temp_c`, `tamb` |
| `humedad` | `hum`, `hr`, `humedad_relativa`, `relative_humidity`, `rh` |
| `presion` | `pres`, `presion_atm`, `pressure`, `p_atm`, `bar` |
| `viento` | `wind`, `vel_viento`, `wind_speed`, `vv`, `ws` |
| `nubosidad` | `cloud`, `nub`, `cloudiness` |
| `indice_uv` | `uv`, `uv_index`, `radiacion_uv` |

Además:

- **Unidades**: si el CSV trae una columna `presion_unit` o `temp_unit`
  (`kPa`, `Pa`, `bar`, `F`, `K`…), se convierte automáticamente a hPa y °C.
- **Target**: si no viene `temp_pred`, se construye sola con `shift(-1)`
  (temperatura de la siguiente hora).
- **Limpieza**: descarta NaN, valores no numéricos y outliers fuera de
  rangos físicamente plausibles (p. ej. humedad 250%).
- **Orden temporal**: las filas no se reordenan nunca.

Si falta alguna columna obligatoria, el entrenamiento se detiene con un
mensaje que indica exactamente cuál falta y qué alias se aceptan.

---

## Arquitectura del sistema

```
  [ Sensores / Datos ]
         │
         ▼
  [ train_model.py ]  ──►  clima_model.joblib
         │
         ▼
  [ FastAPI (main.py) ]
    POST /predict
    POST /forecast
    GET  /model/info
         │
         ▼
  [ Frontend (index.html) ]
    js/model.js  ──  fetch('/predict')
    js/ui.js     ──  DOM, Chart.js, tabs
```

---

## Instalación y ejecución

### 1. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo

```bash
cd backend
python model/train_model.py
```

Genera `backend/model/clima_model.joblib` con el pipeline entrenado.

### 3. Iniciar la API + Frontend

```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Documentación interactiva: `http://localhost:8000/docs`
- **Frontend servido por la API**: `http://localhost:8000/` (raíz) o `http://localhost:8000/ui/` (alias)

### 4. Abrir el frontend

Abre `http://localhost:8000/` en el navegador (recomendado) o
`frontend/index.html` directamente.

El frontend **detecta automáticamente** si la API está disponible:

- ✅ API conectada → usa el modelo ML real (XGBoost + RF + Ridge) y
  muestra métricas, importancia de features e hiperparámetros reales.
- ⚠️ API desconectada → usa un modo de respaldo local simulado
  (el badge del header indica en modo estás).

> **Nota:** el frontend llama a la API usando `window.location.origin`
> (el mismo origen que sirvió la página). Por eso funciona idéntico
> en local y en un despliegue en la nube, sin recompilar nada.

### 5. Ejecutar los tests

```bash
cd backend
pytest tests/ -v
```

---

## 🚀 Despliegue en la nube

El proyecto está listo para desplegarse en cualquier plataforma.
El frontend usa rutas relativas al origen, así que **no requiere
ninguna configuración de URL**.

> **Despliegue con un clic** (Render lee `render.yaml` automáticamente):
>
> [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Juanmask327/clima-ml-project)

### Opción A · Render (gratis, la más simple)

1. Sube el repositorio a GitHub.
2. En [render.com](https://render.com) → **New** → **Blueprint**.
3. Apunta al repositorio; Render lee `render.yaml` automáticamente.
4. URL final: `https://clima-ml-api.onrender.com`

> En el plan gratuito la app se duerme tras 15 min de inactividad;
> la primera carga tarda ~1 min.

### Opción B · Railway

1. `railway init` y `railway up` (usa `Procfile`).
2. O conecta el repo en [railway.app](https://railway.app).

### Opción C · Fly.io

```bash
fly launch --no-deploy --copy-config
fly deploy
```

### Opción D · Docker (cualquier servidor)

```bash
docker build -t clima-ml .
docker run -p 8000:8000 clima-ml
```

### Opción E · Script automático (local → Internet)

```powershell
.\start_deploy.ps1
```

Construye la imagen, levanta el contenedor y abre un túnel público con
Cloudflare, mostrando la URL al final. Requiere Docker Desktop y
`cloudflared` (`winget install Cloudflare.cloudflared`).

> ⚠️ El túnel de Cloudflare **sin cuenta es temporal**: funciona mientras
> el equipo esté encendido y no garantiza disponibilidad. Para una URL
> fija y estable (recomendado para la tesis), usa Render (opción A).

### Archivos de despliegue incluidos

| Archivo       | Plataforma           |
|---------------|----------------------|
| `render.yaml` | Render (Blueprint)   |
| `Procfile`    | Railway / Heroku     |
| `fly.toml`    | Fly.io               |
| `Dockerfile`  | Docker / K8s / VPS   |
| `runtime.txt` | Versión de Python    |

---

## Modelo de Machine Learning

| Componente       | Algoritmo                              |
|------------------|----------------------------------------|
| Base learner 1   | XGBoost (GBR)                          |
| Base learner 2   | Random Forest Regressor                |
| Meta-learner     | Ridge Regression                       |
| Optimización     | Bayesian Search (Optuna, 50 trials)    |
| Preprocesamiento | StandardScaler                         |
| Validación       | TimeSeriesSplit CV (5 folds, sin shuffle — respeta el orden temporal) |

### Métricas obtenidas (TimeSeriesSplit CV, 5 folds)

| Métrica | Valor |
|---------|------------|
| R²      | 0.931 ± 0.045 |
| RMSE    | 1.114 ± 0.359 °C |
| MAE     | 0.842 ± 0.198 °C |

> **Nota metodológica:** los datos actuales son sintéticos (generados con
> `generate_synthetic_data()`), por lo que `temp_pred` está derivado
> principalmente de `temp_actual` — de ahí la alta importancia de esa
> feature. Con datos reales de sensores las métricas deberán reevaluarse.

### Features del modelo (importancia real, promedio XGBoost + RF)

| Feature          | Importancia |
|------------------|-------------|
| Temperatura actual | 81.3%      |
| Índice de calor (heat_index) | 15.3% |
| Índice UV          | 1.1%       |
| Nubosidad          | 0.5%       |
| Humedad relativa   | 0.4%       |
| Velocidad viento   | 0.2%       |

*Los valores se sirven dinámicamente desde `GET /model/info`.*

---

## API Endpoints

| Método | Ruta          | Descripción                        |
|--------|---------------|------------------------------------|
| GET    | `/`           | Info general de la API             |
| GET    | `/health`     | Estado del servicio                |
| POST   | `/predict`    | Predicción de temperatura y clima  |
| POST   | `/forecast`   | Pronóstico 7 días                  |
| GET    | `/model/info` | Métricas y configuración del modelo|

### Ejemplo de llamada `/predict`

```json
POST http://localhost:8000/predict
Content-Type: application/json

{
  "temp":  18.5,
  "hum":   65.0,
  "pres":  1013.0,
  "wind":  12.0,
  "cloud": 40.0,
  "uv":    5.0
}
```

### Respuesta

```json
{
  "pred_temp": 21.3,
  "rain_prob": 34.0,
  "sensacion": 21.1,
  "wind_pred": 9.9,
  "confidence": 91.0,
  "rmse": 1.11,
  "condition": "Despejado y soleado",
  "icon": "☀️",
  "model_version": "1.0.0",
  "timestamp": "2026-11-26T12:19:18"
}
```

> `pred_temp` sale del modelo ML; `rmse` es el RMSE real de la
> validación cruzada. `rain_prob`, `sensacion`, `wind_pred` y
> `confidence` son **heurísticas auxiliares** (no las produce el modelo).

---

## Roadmap (tesis doctoral)

- [ ] Integración con sensores IoT reales (MQTT / InfluxDB)
- [ ] Modelo LSTM para series temporales largas
- [ ] Mapa de micro-zonas con predicción geoespacial
- [ ] Dashboard en tiempo real (WebSocket)
- [ ] Comparación de modelos: GBR vs LSTM vs Transformer
- [ ] Dockerización del backend
- [ ] Despliegue en cloud (AWS / GCP)

---

**Autor:** Doctorado en Ingeniería de Software · Fábrica de Software Universitaria  
**Versión:** 1.0.0