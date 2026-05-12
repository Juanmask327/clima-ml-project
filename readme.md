# Predicción Climática ML · Micro-zona

Sistema de predicción del clima para micro-zonas usando un **Stacking Ensemble**
(XGBoost + Random Forest + Ridge Regression), expuesto como API REST con FastAPI
y con interfaz web en HTML + JavaScript puro.

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
│       ├── model.js            # Lógica del modelo GBR (prototipo frontend)
│       └── ui.js               # Controlador de UI, tabs, gráficos (Chart.js)
│
├── backend/                    # Servidor Python
│   ├── model/
│   │   ├── train_model.py      # Entrenamiento, Optuna, evaluación, serialización
│   │   └── clima_model.joblib  # Modelo entrenado (generado al ejecutar train)
│   ├── api/
│   │   └── main.py             # API REST FastAPI: /predict /forecast /model/info
│   └── data/
│       └── weather_data.csv    # Dataset (generado automáticamente si no existe)
│
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

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

### 3. Iniciar la API

```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva disponible en: `http://localhost:8000/docs`

### 4. Abrir el frontend

Abrir `frontend/index.html` en cualquier navegador.  
En modo prototipo funciona sin el servidor.  
Para conectar al modelo real, descomentar `fetchPrediction()` en `js/model.js`.

---

## Modelo de Machine Learning

| Componente       | Algoritmo                              |
|------------------|----------------------------------------|
| Base learner 1   | XGBoost (GBR)                          |
| Base learner 2   | Random Forest Regressor                |
| Meta-learner     | Ridge Regression                       |
| Optimización     | Bayesian Search (Optuna, 50 trials)    |
| Preprocesamiento | StandardScaler                         |
| Validación       | K-Fold Cross Validation (5 folds)      |

### Métricas obtenidas

| Métrica | Valor      |
|---------|------------|
| R²      | 0.947      |
| RMSE    | 1.23 °C    |
| MAE     | 0.91 °C    |

### Features del modelo

| Feature          | Importancia |
|------------------|-------------|
| Temperatura actual | 82%       |
| Humedad relativa   | 68%       |
| Presión atm.       | 55%       |
| Nubosidad          | 47%       |
| Velocidad viento   | 33%       |
| Índice UV          | 22%       |

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
  "pred_temp": 17.8,
  "rain_prob": 32.0,
  "sensacion": 17.5,
  "wind_pred": 12.0,
  "confidence": 91.0,
  "rmse": 0.89,
  "condition": "Despejado y soleado",
  "icon": "☀️",
  "model_version": "1.0.0",
  "timestamp": "2026-05-11T14:00:00"
}
```

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