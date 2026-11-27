# Predicción Climática ML · Micro-zona

Sistema de predicción del clima para micro-zonas usando un **Stacking Ensemble**
(XGBoost + Random Forest + Ridge Regression), expuesto como API REST con FastAPI
y con interfaz web en HTML + JavaScript puro.

| | |
|---|---|
| 🌐 **App pública (temporal)** | https://counters-capitol-ago-institution.trycloudflare.com |
| 📦 **Repositorio** | https://github.com/Juanmask327/clima-ml-project |
| 📊 **Métricas (CV temporal, datos reales)** | R² 0.826 · RMSE 2.147 °C · MAE 1.395 °C |
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

## 📡 Datos reales de estaciones (pipeline en uso)

El modelo se entrena con **datos reales capturados por estaciones
Raspberry Pi con sensor BME280** (temperatura, humedad y presión —
no miden viento, nubosidad ni UV).

### Pipeline de datos

```
estacion_1.sql / estacion_2.sql / estacion4.sql / estacion3...csv
        │  (dumps crudos: 432,829 registros de 3 estaciones, cada 5 s)
        ▼
backend/data/parse_stations.py
        │  unifica y limpia → weather_data_real.csv
        ▼
backend/data/prepare_real_dataset.py
        │  agrega por hora + target "temperatura de la siguiente hora"
        ▼
backend/data/weather_data.csv   (381 horas válidas, 17 días de muestreo)
        │
        ▼
backend/model/train_model.py    (detecta el esquema real y entrena)
```

Reproducir todo el pipeline:

```bash
python backend/data/parse_stations.py
python backend/data/prepare_real_dataset.py
python backend/model/train_model.py
```

### Esquema del dataset de entrenamiento

| Columna | Significado | Unidad |
|---------|-------------|--------|
| `estacion`  | Identificador de estación | — |
| `time`     | Hora de medición | hora local |
| `temp_actual` | Temperatura media horaria | °C |
| `humedad`  | Humedad relativa media | % |
| `presion`  | Presión atmosférica media (~750 hPa, estación en altura) | hPa |
| `temp_next`| **Target**: temperatura de la siguiente hora | °C |

### Features del modelo (7)

| Feature | Tipo |
|---------|------|
| `temp_actual`, `humedad`, `presion` | medidas del sensor |
| `hour_sin`, `hour_cos` | ciclo diario codificado (sin discontinuidad) |
| `hum_effect`, `heat_index` | derivadas de humedad |

> La hora del día es clave: el ciclo diario es el principal motor de
> la temperatura horaria. La API la toma del navegador (hora local,
> como registran las estaciones).

### Nota sobre la cobertura de muestreo

Las estaciones no operaron de forma continua: los 432,829 registros
se concentran en **17 días** entre mayo 2023 y diciembre 2024
(rachas de muestreo intenso). El dataset horario resultante tiene
381 filas con target contiguo. Si las estaciones capturan más
días, basta re-ejecutar el pipeline para re-entrenar con más datos.

### Esquema sintético (legacy)

`weather_data_synth.csv` conserva el dataset sintético original
(6 variables + `temp_pred`). El pipeline lo sigue soportando para
pruebas: `load_real_data()` acepta alias de columnas y unidades
(`temp_unit`, `presion_unit`) automáticamente.

---

## Arquitectura del sistema

```
  [ Estaciones Raspberry Pi + BME280 ]   (temp, humedad, presión cada 5 s)
          │
          ▼
  [ data/parse_stations.py ]  →  weather_data_real.csv
          │
          ▼
  [ data/prepare_real_dataset.py ]  →  weather_data.csv (horario)
          │
          ▼
  [ model/train_model.py ]  ──►  clima_model.joblib
          │
          ▼
  [ FastAPI (api/main.py) ]
    POST /predict    (temp, hum, pres, hora → temp de la siguiente hora)
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
| Modelo           | XGBoost (elegido por CV entre XGB solo, stacking XGB+RF+Ridge y baseline de persistencia) |
| Optimización     | Bayesian Search (Optuna, 50 trials)    |
| Preprocesamiento | StandardScaler                         |
| Validación       | TimeSeriesSplit CV (5 folds, sin shuffle — respeta el orden temporal) |
| Datos            | 381 horas reales de 3 estaciones BME280 |

### Métricas del modelo REAL (estaciones BME280, TimeSeriesSplit 5 folds)

| Métrica | Valor |
|---------|------------|
| R²      | 0.826 ± 0.105 |
| RMSE    | 2.147 ± 0.987 °C |
| MAE     | 1.395 ± 0.673 °C |

**Comparativa de arquitecturas (misma validación temporal):**

| Arquitectura | RMSE | R² |
|--------------|------|-----|
| **XGBoost (elegido)** | **2.147** | **0.826** |
| Stacking (XGB + RF + Ridge) | 2.622 | 0.715 |
| Baseline persistencia (t+1h = t) | 2.154 | — |

> El stacking sobreajusta con 381 horas de datos reales (el Random
> Forest de 300 árboles es demasiado complejo), por lo que la
> selección automática por CV elige XGBoost solo. Superar al
> baseline de persistencia es el requisito mínimo de cualquier
> modelo de temperatura horaria: aquí se cumple con margen estrecho
> — capturar más días de datos debería ampliarlo.

### Features del modelo (importancia real)

| Feature          | Importancia |
|------------------|-------------|
| Índice de calor (heat_index) | 40.1% |
| Temperatura actual | 33.6%   |
| Humedad relativa  | 11.0%   |
| Efecto humedad     | 4.9%    |
| Presión atmosférica | 4.3%   |
| Ciclo diario (cos) | 4.3%    |
| Ciclo diario (sin) | 1.9%    |

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
  "temp":  19.0,
  "hum":   68.0,
  "pres":  752.0,
  "hora":  14
}
```

> `temp`, `hum` y `pres` son las tres variables que mide el sensor
> BME280. `hora` (0-23) es opcional: si no se envía se usa la hora
> local. El esquema sintético anterior (`wind`, `cloud`, `uv`)
> sigue aceptado como opcional para compatibilidad.

### Respuesta

```json
{
  "pred_temp": 19.3,
  "rain_prob": 45.0,
  "sensacion": 19.3,
  "confidence": 92.0,
  "rmse": 2.15,
  "condition": "Despejado y soleado",
  "icon": "☀️",
  "hora_usada": 14,
  "model_version": "1.0.0",
  "timestamp": "2026-11-27T03:33:41"
}
```

> `pred_temp` sale del modelo ML (temperatura de la siguiente hora);
> `rmse` es el RMSE real de la validación cruzada. `rain_prob`,
> `sensacion` y `confidence` son **heurísticas auxiliares** (no las
> produce el modelo); `hora_usada` es la hora con la que se evaluó
> el ciclo diario.

---

## Roadmap (tesis doctoral)

- [x] Integración con datos reales de estaciones (BME280, dumps MySQL)
- [x] Dockerización del backend
- [x] Despliegue en cloud (túnel público + configs Render/Fly/Railway)
- [ ] Captura continua de datos (más días → mejor RMSE vs persistencia)
- [ ] Modelo LSTM para series temporales largas
- [ ] Ingesta en vivo de sensores (MQTT / InfluxDB)
- [ ] Mapa de micro-zonas con predicción geoespacial
- [ ] Dashboard en tiempo real (WebSocket)
- [ ] Comparación de modelos: GBR vs LSTM vs Transformer

---

**Autor:** Doctorado en Ingeniería de Software · Fábrica de Software Universitaria  
**Versión:** 1.0.0