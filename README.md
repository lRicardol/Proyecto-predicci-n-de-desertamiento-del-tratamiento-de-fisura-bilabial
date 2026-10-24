# Predicción de deserción del tratamiento de fisura bilabial

Sistema de **analítica predictiva local** para el seguimiento de pacientes con fisura labiopalatina (fisura bilabial). El objetivo es estimar el riesgo de abandono terapéutico y explicar, con evidencia estadística, los factores asociados: inasistencias (no-show), distancia al centro, nivel socioeconómico y complicaciones quirúrgicas.

Toda la inteligencia vive **en el equipo**: no hay API keys, ni OpenAI, ni Anthropic, ni agentes LangChain. El modelo es un clasificador **XGBoost** entrenado sobre el CSV unificado de **3 500 pacientes reales**. Las consultas en lenguaje natural se resuelven con un motor de reglas que dispara gráficos de Matplotlib y Seaborn.

> **Aviso de salud.** Esta aplicación es una herramienta de soporte para la gestión de seguimiento clínico. **No constituye diagnóstico, pronóstico ni decisión terapéutica.** Las predicciones deben interpretarse por personal de salud junto con la historia clínica completa.

---

## Arquitectura del software

El diseño sigue **Clean Architecture** y los principios SOLID. El dominio no importa Pandas de infraestructura, ni XGBoost, ni FastAPI: solo contratos. Las implementaciones concretas se inyectan en un único composition root.

```
┌─────────────────────────────────────────────────────────────┐
│  Presentation (src/app)                                     │
│  FastAPI · controladores · dashboard.html (Tailwind)        │
└────────────────────────────┬────────────────────────────────┘
                             │ depende de casos de uso
┌────────────────────────────▼────────────────────────────────┐
│  Core (src/core)                                            │
│  Interfaces (puertos) · casos de uso · excepciones          │
│  IPatientRepository · IPredictorModel · ILLMAnalyst         │
└────────────────────────────┬────────────────────────────────┘
                             │ DIP: el core no conoce CSV ni XGBoost
┌────────────────────────────▼────────────────────────────────┐
│  Infrastructure (src/infrastructure)                        │
│  CsvPatientRepository · XgboostPredictor                    │
│  LocalStatisticalAnalyst · feature_engineering              │
└─────────────────────────────────────────────────────────────┘
```

### Capa Core

Es el corazón del sistema y no conoce detalles de I/O.

| Pieza | Responsabilidad |
|---|---|
| `src/core/interfaces/` | Puertos atómicos (ISP). Un contrato para predicción, otro para analítica, otro para persistencia. |
| `src/core/use_cases/predict_dropout.py` | Orquesta repositorio + modelo y mapea la probabilidad a **Bajo / Medio / Alto**. |
| `src/core/use_cases/generate_analytics.py` | Entrega la consulta al analista y exige `chart_output.png`. |
| `src/core/exceptions.py` | Errores de dominio; los controladores los traducen a HTTP. |

`PredictDropoutUseCase` recibe `IPatientRepository` e `IPredictorModel`, nunca `CsvPatientRepository` ni `XgboostPredictor`. Eso es **Inversión de Dependencias (DIP)**: el caso de uso depende de abstracciones.

### Patrón Strategy

Los tres puertos son estrategias intercambiables:

- `IPredictorModel.predict_dropout_probability` — hoy XGBoost; mañana otro clasificador con `predict_proba`.
- `ILLMAnalyst.analyze_data_and_plot` — hoy el analista estadístico local; el nombre del puerto se conservó por compatibilidad, pero **no hay LLM**.
- `IPatientRepository` — hoy un CSV unificado; podría ser una base clínica sin tocar el core.

### Capa Infrastructure

Aquí viven los adaptadores que sí conocen disco, Pandas y librerías de ML:

- `csv_patient_repository.py` — lee `dataset_final_limpio.csv` y busca fichas `PAC-XXXX`.
- `xgboost_predictor.py` — entrena, evalúa, serializa con **joblib** y predice.
- `feature_engineering.py` — matriz de cuatro features: no-shows, distancia, SES ordinal, complicaciones.
- `local_analyst.py` — motor de reglas + Seaborn.

### Capa Presentation (MVC)

- **Modelo de aplicación:** casos de uso + DTOs en `src/app/views/schemas.py`.
- **Controladores:** `prediction_controller.py` y `dashboard_controller.py` (FastAPI). No contienen reglas de negocio.
- **Vista:** `src/app/views/dashboard.html` — panel de predicción (riesgo verde / ámbar / rojo) y panel de analítica con gráfico dinámico.

### Composition root

`src/app/composition.py` es el **único** lugar que instancia clases concretas y las inyecta:

```python
def get_predict_dropout_use_case() -> PredictDropoutUseCase:
    return PredictDropoutUseCase(
        patient_repository=get_patient_repository(),
        predictor_model=get_predictor_model(),
        ...
    )
```

Al arrancar, `main.py` llama a `warmup_predictor()`: si no existe `model.joblib`, se entrena XGBoost sobre el CSV procesado; si existe, se carga.

---

## Fase de ciencia de datos

### ETL: seis CSV reales → un tablero de 3 500 pacientes

Los orígenes están en `data/raw/`. Hay dos cohortes unidas por la llave `ID_Paciente`:

| Cohorte | Archivos | Identificadores |
|---|---|---|
| v1 | deserción, hitos quirúrgicos, terapia de lenguaje / IVF | `PAC-1000`… |
| v2 | deserción socioeconómica, hitos quirúrgicos, terapia | `PAC-2000`… |

Pipeline (`notebooks/00_procesamiento_dataset_unificado.py` y el notebook homónimo):

1. Lectura de los **6 CSV reales** (sin datos sintéticos de relleno).
2. Limpieza de columnas, duplicados por ID y normalización de tipos.
3. Inner join por paciente **solo si existe la etiqueta `Desertor`**.
4. Ingeniería: `nivel_socioeconomico_ord`, `complicaciones_quirurgicas` (dehiscencia + fístula), ratios de terapia.
5. Persistencia en `data/processed/dataset_final_limpio.csv` (**3 500 filas**, una por paciente).

Los notebooks `01` y `02` documentan EDA clínico, boxplots de inasistencias, SES, distancia y la justificación de `scale_pos_weight`.

### Entrenamiento local de XGBoost

Ejecutado sobre el CSV limpio, con `train_test_split` estratificado (80 / 20) y `scale_pos_weight = n_neg / n_pos` calculado sobre el desbalance real del train (**74.6 %** de deserción).

| Concepto | Valor |
|---|---|
| Filas | 3 500 |
| Features | `citas_perdidas_noshow`, `distancia_hospital_km`, `nivel_socioeconomico_ord`, `complicaciones_quirurgicas` |
| Objetivo | `desertor` |
| Train / Test | 2 800 / 700 |
| Desbalance train | 2 090 desertores (74.6 %) · 710 continuadores |
| `scale_pos_weight` | **0.3397** (`n_neg / n_pos`) |
| Precisión | **0.8588** (85.88 %) |
| Recall | **0.6992** |
| F1-Score | **0.7709** |

Matriz de confusión `[[TN, FP], [FN, TP]]` impresa en consola:

```
[[118,  60],
 [157, 365]]
```

Informe de clasificación (conjunto de prueba, 700 pacientes):

```
              precision    recall  f1-score   support
           0      0.429     0.663     0.521       178
           1      0.859     0.699     0.771       522
    accuracy                          0.690       700
```

El artefacto queda en `src/infrastructure/ai_models/serialized/model.joblib`. En inferencia, la probabilidad se traduce a:

- **Bajo** si *p* < 0.33 (verde)
- **Medio** si 0.33 ≤ *p* < 0.66
- **Alto** si *p* ≥ 0.66 (rojo)

---

## Módulo de analítica local

`LocalStatisticalAnalyst` implementa `ILLMAnalyst` **sin modelos de lenguaje**. Recibe una frase del usuario, la normaliza y dispara gráficos equivalentes a los de los notebooks:

| Palabras clave | Gráfico |
|---|---|
| `inasistencias`, `noshow`, `citas perdidas` | Boxplot + stripplot de no-shows vs deserción |
| `socioeconómico`, `ses` | Tasas de deserción por SES y apoyo psicológico |
| `distancia`, `km` | Densidad KDE de km al hospital |
| `cirugías`, `complicaciones` | Carga operatoria vs complicaciones |
| `terapia`, `lenguaje`, `ivf` | Asistencia a terapia e hipernasalidad |
| `correlación`, `matriz` | Heatmap de Pearson |
| resto | Distribución global de la etiqueta `desertor` |

El PNG se escribe siempre en `static/generated_charts/chart_output.png` y el panel lo refresca con un parámetro de caché. No hay llamadas de red ni costos de tokens.

---

## Guía de despliegue local

Requisitos: Python 3.11+ en Windows, con el entorno virtual del proyecto.

### 1. Dependencias (si el venv aún no está listo)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset procesado

Debe existir `data/processed/dataset_final_limpio.csv`. Si hay que regenerarlo desde los seis CSV:

```bash
.\.venv\Scripts\python.exe notebooks/00_procesamiento_dataset_unificado.py
```

### 3. Entrenar el modelo (opcional)

Si no hay `model.joblib`, el arranque de la API lo entrena. Para entrenar a mano y ver las métricas en consola:

```bash
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.xgboost_predictor
```

### 4. Levantar el panel

```bash
.\.venv\Scripts\python.exe main.py
```

Abrir el navegador en:

**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

| Recurso | Uso |
|---|---|
| `/` | Panel MVC: predicción por ID y analítica gráfica |
| `GET /api/predictions/PAC-1001` | Ficha clínica + probabilidad + riesgo |
| `POST /api/analytics` | `{"query": "Grafica la relación entre inasistencias y deserción"}` |
| `GET /api/health` | Pacientes cargados y IDs de ejemplo |
| `/docs` | OpenAPI de FastAPI |

IDs reales de prueba: `PAC-1001`, `PAC-2001`, `PAC-1000`.

### 5. Pruebas automáticas

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

---

## Estructura relevante

```
├── data/raw/                          # 6 CSV clínicos reales
├── data/processed/dataset_final_limpio.csv
├── notebooks/                         # ETL + EDA + análisis predictivo
├── src/core/                          # puertos, casos de uso, excepciones
├── src/infrastructure/                # CSV, XGBoost, analista local
├── src/app/                           # composition, controladores, vista HTML
├── static/generated_charts/           # chart_output.png
├── tests/
├── config.py
└── main.py
```

El dominio queda cerrado: un modelo serializado, un analista estadístico reproducible y una interfaz de soporte clínico, todo ejecutable en un portátil de laboratorio sin servicios de terceros.
