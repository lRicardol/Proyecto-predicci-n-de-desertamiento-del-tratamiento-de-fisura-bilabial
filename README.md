# Predicción de deserción del tratamiento de fisura bilabial

Sistema de **analítica predictiva local** para el seguimiento de pacientes con fisura labiopalatina (fisura bilabial) en Fisulab. Estima la probabilidad de que un paciente abandone el tratamiento y la traduce en una **prioridad de seguimiento** (alta / media / habitual), explicando qué variables pesaron en cada estimación.

El modelo es una **regresión logística** entrenada con la cohorte v2 (**3 000 pacientes sintéticos**) usando solo información disponible **al ingreso** del paciente (escenario T0). Todo corre en el equipo: no hay API keys ni servicios externos. Las consultas en lenguaje natural del panel se resuelven con un motor de reglas que genera gráficos de Matplotlib y Seaborn.

> **Datos sintéticos.** Los 3 500 pacientes del repositorio no son reales. Los resultados son un ejercicio metodológico y **no se pueden extrapolar a pacientes reales**.

> **Aviso de salud.** Esta aplicación es una herramienta de apoyo para priorizar el seguimiento. **No constituye diagnóstico, pronóstico ni decisión terapéutica.** Las estimaciones deben interpretarse por personal de salud junto con la historia clínica completa.

---

## Inicio rápido

Requisitos: Python 3.11 en Windows.

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
```

Abrir **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** y buscar un paciente de la cohorte v2 (por ejemplo `PAC-2001`).

### Demo en Jupyter

[`notebooks/04_demo_modelo.ipynb`](notebooks/04_demo_modelo.ipynb) recorre todo el flujo con explicaciones celda por celda: datos de entrenamiento, variables excluidas por fuga, comparación de modelos, modelo final, explicación de predicciones, paciente nuevo y llamada a la API. Tarda alrededor de un minuto con *Run All* y no modifica archivos del repositorio. El archivo se guarda ya ejecutado, así que también se puede leer sin correrlo (por ejemplo, en GitHub).

Se abre con VS Code (extensión Jupyter) o JupyterLab, usando el kernel del `.venv`. El kernel necesita `ipykernel`, incluido en `requirements-dev.txt`. Para regenerar el notebook desde su script: `python notebooks/_build_04_demo_modelo.py`.

> **Windows y rutas largas.** `ipykernel` instala archivos con rutas muy largas. Si `pip` falla con *"Windows Long Path support"*, habilita las rutas largas de Windows (requiere administrador; ver [la guía de pip](https://pip.pypa.io/warnings/enable-long-paths)) o clona el repositorio en una carpeta corta, como `C:\proyectos\fisulab`.

---

## Arquitectura del software

El diseño sigue **Clean Architecture** y los principios SOLID. El dominio no importa pandas de infraestructura, ni scikit-learn, ni FastAPI: solo contratos. Las implementaciones concretas se inyectan en un único composition root.

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
                             │ DIP: el core no conoce CSV ni scikit-learn
┌────────────────────────────▼────────────────────────────────┐
│  Infrastructure (src/infrastructure)                        │
│  CsvPatientRepository · LogisticT0Predictor · etl           │
│  experiments · interpretability · LocalStatisticalAnalyst   │
└─────────────────────────────────────────────────────────────┘
```

### Capa Core

| Pieza | Responsabilidad |
|---|---|
| `src/core/interfaces/` | Puertos atómicos (ISP). `IPredictorModel` devuelve probabilidades y, opcionalmente, la explicación de cada predicción (`FactorContribution`). |
| `src/core/use_cases/predict_dropout.py` | Orquesta repositorio + modelo y asigna la **prioridad de seguimiento** con `PriorityPolicy`. Añade el mensaje para el equipo, las variables que más pesaron, la fecha y avisos (por ejemplo, paciente fuera de la cohorte de entrenamiento). |
| `src/core/use_cases/generate_analytics.py` | Entrega la consulta al analista y exige `chart_output.png`. |
| `src/core/exceptions.py` | Errores de dominio; los controladores los traducen a HTTP. |

### Capa Infrastructure

| Módulo | Qué hace |
|---|---|
| `data/etl.py` | Construye el marco de modelado de una cohorte sin columnas con fuga y sin imputar. |
| `ai_models/modeling_features.py` | Contrato de variables: `FEATURES_T0`, `FEATURES_T1`, `EXCLUDED_LEAKAGE`, `select_features`. |
| `ai_models/experiments.py` | Comparación de modelos con validación cruzada anidada y holdout. |
| `ai_models/interpretability.py` | Coeficientes, importancia por permutación y explicación por paciente. |
| `ai_models/final_model.py` | Entrena el modelo final con toda la cohorte v2 y guarda `serialized/modelo_t0.joblib` + `modelo_t0.json`. |
| `ai_models/logistic_predictor.py` | Carga el artefacto, predice y explica. **Nunca entrena.** |
| `repositories/csv_patient_repository.py` | Lee `dataset_final_limpio.csv` y busca fichas `PAC-XXXX`. |
| `llm_agents/local_analyst.py` | Motor de reglas + Seaborn (sin LLM). |
| `ai_models/xgboost_predictor.py`, `feature_engineering.py` | **Legado**: modelo anterior, con fuga. La API ya no lo usa; se conserva como referencia. |

### Capa Presentation

- **Controladores:** `prediction_controller.py` y `dashboard_controller.py` (FastAPI), sin reglas de negocio.
- **DTOs:** `src/app/views/schemas.py`.
- **Vista:** `src/app/views/dashboard.html` — prioridad de seguimiento, datos usados por el modelo, variables que más pesaron y panel de analítica.

### Composition root

`src/app/composition.py` es el único lugar que instancia clases concretas. Los cortes de prioridad viajan dentro del artefacto, así que el modelo y sus cortes siempre son de la misma versión. Al arrancar, `main.py` llama a `warmup_predictor()`, que **solo carga** `modelo_t0.joblib`; si falta, la API no arranca y el error indica cómo generarlo.

---

## El modelo

| Concepto | Valor |
|---|---|
| Algoritmo | Regresión logística regularizada (`C = 0.01`), con imputación, one-hot y estandarización dentro del `Pipeline` |
| Población | Cohorte v2: 3 000 pacientes, 71.4 % desertores (v1 tiene solo 30 no desertores y queda fuera) |
| Variables (T0, al ingreso) | `tipo_fisura`, `edad_inicio_meses`, `distancia_hospital_km`, `nivel_socioeconomico_ord`, `ingresos_mensuales_usd`, `apoyo_psicologico`, `dificultad_alimentacion_preop` |
| Excluidas por fuga | Asistencia y citas perdidas (acumuladas durante el tratamiento), peso a los 2 años y columnas que identifican la cohorte. Ver [`reports/diccionario_datos.md`](reports/diccionario_datos.md) |
| Desempeño | ROC-AUC **0.737** en el holdout (IC 95 % 0.693–0.783) frente a 0.500 de la línea base; PR-AUC 0.868 frente a 0.713. Fuera de muestra con los 3 000 pacientes: ROC-AUC 0.722 |
| Artefacto | `src/infrastructure/ai_models/serialized/modelo_t0.joblib` (ficha legible en `modelo_t0.json`) |

### Salida: prioridad de seguimiento

| Prioridad | Regla (puntaje del modelo) | Deserción observada en el grupo* |
|---|---|---|
| **Alta** | 20 % de pacientes con mayor puntaje (≥ 0.832) | 90 % |
| **Media** | El resto | 77 % |
| **Habitual** | 30 % con menor puntaje (< 0.619) | 49 % |

\* Validación cruzada sobre los 3 000 pacientes de v2. No se habla de "riesgo bajo": incluso el grupo habitual deserta cerca de la mitad de las veces. Los porcentajes 20 / 30 son un punto de partida y **deben ajustarse con Fisulab según cuántos pacientes puede acompañar el equipo por semana**. La prioridad sirve para dirigir apoyo, nunca para limitar la atención.

Ejemplo de respuesta de `GET /api/predictions/PAC-2001` (resumida):

```json
{
  "patient_id": "PAC-2001",
  "dropout_probability": 0.7519,
  "priority_level": "media",
  "priority_message": "Probabilidad estimada intermedia. Seguimiento según la rutina del servicio.",
  "group_observed_dropout_rate": 0.7727,
  "top_factors": [
    {"variable": "apoyo_psicologico", "value": 1, "contribution": -0.7188, "direction": "reduce la estimación"},
    {"variable": "distancia_hospital_km", "value": 155.5, "contribution": 0.6646, "direction": "aumenta la estimación"}
  ],
  "model_inputs": {"tipo_fisura": "Asimetrica", "distancia_hospital_km": 155.5},
  "prediction_date": "2026-11-09",
  "warnings": [],
  "disclaimer": "Herramienta de apoyo a la gestión del seguimiento, entrenada con datos sintéticos. ..."
}
```

`contribution` está en log-odds frente al paciente promedio: positivo sube la estimación, negativo la baja. Es asociación, no causa. Para pacientes de la cohorte v1 la respuesta incluye un aviso en `warnings`.

### Regenerar el modelo

```bash
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.final_model
```

Tarda unos segundos, sobrescribe `modelo_t0.joblib` y `modelo_t0.json` e imprime hiperparámetros, métricas fuera de muestra y cortes de prioridad.

---

## Guía de uso local

| Recurso | Uso |
|---|---|
| `/` | Panel: prioridad de seguimiento por ID y analítica gráfica |
| `GET /api/predictions/PAC-2001` | Probabilidad, prioridad, variables que más pesaron, datos usados y avisos |
| `POST /api/predictions` | Lo mismo con `{"patient_id": "PAC-2001"}` |
| `POST /api/analytics` | `{"query": "Muestra el impacto del nivel socioeconómico"}` |
| `GET /api/health` | Pacientes cargados e IDs de ejemplo |
| `/docs` | OpenAPI de FastAPI |

IDs de ejemplo (sintéticos): `PAC-2001`, `PAC-2002` (cohorte del modelo) y `PAC-1001` (cohorte v1, con aviso).

El panel lee `data/processed/dataset_final_limpio.csv`. Si hay que regenerarlo:

```bash
.\.venv\Scripts\python.exe notebooks/00_procesamiento_dataset_unificado.py
```

### Módulo de analítica local

`LocalStatisticalAnalyst` implementa `ILLMAnalyst` **sin modelos de lenguaje**: recibe una frase, la normaliza y genera un gráfico en `static/generated_charts/chart_output.png`.

| Palabras clave | Gráfico |
|---|---|
| `inasistencias`, `noshow`, `citas perdidas` | Boxplot + stripplot de no-shows vs deserción (variable con fuga: solo descriptivo) |
| `socioeconómico`, `ses` | Tasas de deserción por SES y apoyo psicológico |
| `distancia`, `km` | Densidad KDE de km al hospital |
| `cirugías`, `complicaciones` | Carga operatoria vs complicaciones |
| `terapia`, `lenguaje`, `ivf` | Asistencia a terapia e hipernasalidad (variable con fuga: solo descriptivo) |
| `correlación`, `matriz` | Heatmap de Pearson |
| resto | Distribución global de la etiqueta `desertor` |

---

## Estructura relevante

```
├── data/raw/                          # 6 CSV sintéticos (= contenido del zip)
├── data/processed/dataset_final_limpio.csv
├── notebooks/
│   ├── 00_procesamiento_dataset_unificado.py   # ETL del panel
│   ├── 01_*, 02_*                     # notebooks iniciales (modelo anterior, con fuga)
│   ├── 03_eda.py                      # EDA (frente 3)
│   └── 04_demo_modelo.ipynb           # demo de punta a punta del modelo final
├── reports/                           # auditoría, diccionario, EDA, modelos, interpretabilidad
├── src/core/                          # puertos, casos de uso, excepciones
├── src/infrastructure/                # ETL, modelo, repositorio CSV, analista local
├── src/app/                           # composition, controladores, vista HTML
├── tests/
├── config.py
└── main.py
```

---

## Evaluación, interpretabilidad y limitaciones

### Datos sintéticos

Los 3 500 pacientes del repositorio son **sintéticos**, no reales (confirmado por el equipo; ver [`reports/auditoria.md`](reports/auditoria.md)). Los resultados son un ejercicio metodológico y **no se pueden extrapolar a pacientes reales**. Antes de cualquier uso real faltan datos reales con fechas, validación con Fisulab y revisión ética.

### El modelo anterior (XGBoost) y por qué se reemplazó

| Problema del modelo anterior | Detalle |
|---|---|
| Datos descritos como "reales" | Los datos son **sintéticos**. |
| ETL "sin datos sintéticos de relleno" | El ETL sí imputa columnas faltantes de v2 con valores de v1 (`citas_asistidas` queda constante en 11; `cirugias_programadas` se estima con otra fórmula). Esas columnas identifican la cohorte y se excluyen del modelado. |
| Métricas reportadas (precisión 0.8588, F1 0.7709) | **No son evidencia de desempeño.** El modelo usa `citas_perdidas_noshow`, una variable con fuga, mezcla dos cohortes distintas y la accuracy (0.690) queda por debajo del 0.746 de un clasificador que siempre predice "desertor". Se evaluó con una sola partición, sin línea base ni validación cruzada. |
| Cortes Bajo / Medio / Alto de 0.33 y 0.66 | No tienen respaldo (ver "Salida recomendada"). |
| Gráficos de inasistencias y de asistencia a terapia en el analista local | Muestran variables con fuga: son descriptivos y no deben interpretarse como factores predictivos. |

### Resultados de la evaluación

La evaluación se hizo solo sobre la **cohorte v2** (3 000 pacientes; la cohorte v1 tiene 30 no desertores y se usa solo de forma descriptiva), con validación cruzada anidada 5 × 3 sobre 2 400 pacientes de desarrollo y un holdout de 600 evaluado una sola vez. Detalle en [`reports/comparacion_modelos.md`](reports/comparacion_modelos.md).

| Escenario | Cuándo se predice | Variables | ROC-AUC en holdout (IC 95 %) | Línea base |
|---|---|---|---|---|
| **T0** | Al ingreso | Tipo de fisura, edad de inicio, distancia, SES, ingresos, apoyo psicológico, dificultad de alimentación | **0.737** [0.693–0.783] | 0.500 (`DummyClassifier`) |
| T1 sin habla | Tras cirugías primarias | T0 + hitos quirúrgicos, complicaciones, hipoacusia | 0.731 [0.687–0.777] | 0.500 |
| T1 (cota superior) | Tras cirugías primarias | T1 sin habla + hipernasalidad e IVF | 0.877 [0.846–0.908] | 0.500 |

- **Modelo de referencia: regresión logística regularizada en T0.** Random Forest y XGBoost no la mejoran (diferencias ≤ 0.003 de ROC-AUC en T0) y la regresión logística está bien calibrada (ECE 0.026).
- **T1 es una cota superior, no un resultado.** Todo el salto de 0.73 a 0.88 viene de `grado_hipernasalidad` e IVF, que pueden ser consecuencia del abandono (no hay fechas para saber cuándo se miden).
- En T0, PR-AUC 0.868 frente a 0.713 (prevalencia). Exigir recall ≥ 0.90 obliga a alertar a ≈ 83 % de los pacientes (especificidad 0.34): la alerta filtra poco.
- **Variables excluidas por fuga:** `citas_perdidas_noshow`, `citas_terapia_asistidas`, `ratio_asistencia_terapia`, `citas_asistidas`, `cirugias_programadas`, `peso_z_score_2anos` y otras. Motivos en [`reports/diccionario_datos.md`](reports/diccionario_datos.md).

### Qué pesa en el modelo (T0)

Caída de ROC-AUC al permutar la variable en el holdout: `apoyo_psicologico` 0.150, `distancia_hospital_km` 0.043, `ingresos_mensuales_usd` 0.026; el resto, < 0.003. Regresión logística, Random Forest y XGBoost coinciden en el ranking. Son **asociaciones, no causas**: que el apoyo psicológico se asocie con menos deserción no demuestra que ofrecerlo la reduzca. En T1 los coeficientes de fístula e IVF salen "protectores" por su relación con la hipernasalidad y no deben leerse como efectos clínicos. Detalle en [`reports/interpretabilidad.md`](reports/interpretabilidad.md).

### Salida recomendada: prioridad de seguimiento

Con los cortes 0.33 y 0.66, el 1.2 % de los pacientes del holdout quedaría en "Bajo" y el 62.7 % en "Alto". Ni siquiera el 30 % de pacientes con menor puntaje es de bajo riesgo (deserción observada: 49 %). Se recomienda presentar una **prioridad de seguimiento** (alta / media / habitual) definida por la capacidad semanal del equipo, con textos que no hablen de diagnóstico, y usarla para **dirigir apoyo**, nunca para limitar la atención de quien vive lejos o tiene menos recursos. Detalle en `reports/interpretabilidad.md` §6 y en [`reports/limitaciones_y_plan_mejora.md`](reports/limitaciones_y_plan_mejora.md).

### Estado de integración

| Componente | Estado |
|---|---|
| ETL, EDA, contrato de features, comparación de modelos, interpretabilidad | Documentados en `reports/` |
| Modelo final y API | **Integrados.** La API sirve la regresión logística T0 (`modelo_t0.joblib`) con prioridad de seguimiento. |
| Entrenamiento al arrancar la API | Eliminado: la API solo carga el artefacto versionado. |
| Cortes de prioridad (20 % / 30 %) | Punto de partida; pendiente de validar con la capacidad real de Fisulab. |
| Datos reales y fechas | No disponibles. Ver el plan de mejora. |

### Nuevos módulos de infraestructura

- `modeling_features.py` — contrato de variables por escenario: `FEATURES_T0`, `FEATURES_T1`, `EXCLUDED_LEAKAGE` y `select_features`.
- `experiments.py` — protocolo de evaluación (validación cruzada repetida, líneas base, calibración, subgrupos).
- `interpretability.py` — coeficientes con IC bootstrap, importancia por permutación, efectos marginales y explicación por paciente.
- `final_model.py` — entrena y guarda el modelo final (`modelo_t0.joblib`) con sus cortes de prioridad.
- `logistic_predictor.py` — adaptador de inferencia que usa la API.

### Reproducir la evaluación y la interpretabilidad

```bash
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.experiments
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.interpretability --scenario T0 --output-dir reports\modelos\interp_T0
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.interpretability --scenario T1 --output-dir reports\modelos\interp_T1
```

La evaluación completa tarda alrededor de 12 minutos (semilla 42). Interpretabilidad escribe cuatro CSV y una figura por escenario; usa `--output-dir` para que un escenario no sobrescriba al otro.

### Documentación

| Documento | Contenido |
|---|---|
| [`reports/auditoria.md`](reports/auditoria.md) | Auditoría del repositorio y del dataset |
| [`reports/diccionario_datos.md`](reports/diccionario_datos.md) | Diccionario de datos y decisión por variable |
| [`reports/definicion_problema.md`](reports/definicion_problema.md) | Qué es abandonar, población, escenarios T0 / T1, preguntas para Fisulab |
| [`reports/eda.md`](reports/eda.md) · [`reports/eda_tables.md`](reports/eda_tables.md) | Análisis exploratorio y tablas |
| [`reports/comparacion_modelos.md`](reports/comparacion_modelos.md) | Comparación de modelos, calibración, subgrupos |
| [`reports/interpretabilidad.md`](reports/interpretabilidad.md) | Coeficientes, importancia, explicación por paciente, salida del sistema |
| [`reports/limitaciones_y_plan_mejora.md`](reports/limitaciones_y_plan_mejora.md) | Limitaciones, ética y plan de mejora |
| [`notebooks/04_demo_modelo.ipynb`](notebooks/04_demo_modelo.ipynb) | Demo ejecutable de todo el flujo |

### Privacidad

**Nunca subir datos de pacientes reales a GitHub ni a ningún repositorio público.** Este repositorio contiene datos y un modelo serializado solo porque los datos son sintéticos. Con datos reales: repositorio privado o datos fuera del repositorio, `.gitignore` para `data/` y modelos, y seudonimización acordada con la institución. Los CSV de explicación por paciente (`interpretabilidad_pacientes.csv`) contienen identificadores y tampoco deben subirse en ese caso.
