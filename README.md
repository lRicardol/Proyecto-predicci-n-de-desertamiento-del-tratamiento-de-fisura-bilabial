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

---

## Evaluación, interpretabilidad y limitaciones

> **Esta sección corrige y complementa el texto anterior.** Donde haya diferencias, prevalece lo que dice aquí (ver "Correcciones al texto anterior").

### Datos sintéticos

Los 3 500 pacientes del repositorio son **sintéticos**, no reales (confirmado por el equipo; ver [`reports/auditoria.md`](reports/auditoria.md)). Los resultados son un ejercicio metodológico y **no se pueden extrapolar a pacientes reales**. Antes de cualquier uso real faltan datos reales con fechas, validación con Fisulab y revisión ética.

### Correcciones al texto anterior

| Texto anterior | Corrección |
|---|---|
| "3 500 pacientes reales", "6 CSV reales", "IDs reales de prueba", "CSV clínicos reales" | Los datos son **sintéticos**. |
| "sin datos sintéticos de relleno" | El ETL sí imputa columnas faltantes de v2 con valores de v1 (`citas_asistidas` queda constante en 11; `cirugias_programadas` se estima con otra fórmula). Esas columnas identifican la cohorte y se excluyen del modelado. |
| Métricas del XGBoost (precisión 0.8588, F1 0.7709) | **No son evidencia de desempeño.** El modelo usa `citas_perdidas_noshow`, una variable con fuga, mezcla dos cohortes distintas y la accuracy (0.690) queda por debajo del 0.746 de un clasificador que siempre predice "desertor". Se evaluó con una sola partición, sin línea base ni validación cruzada. |
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
| `model.joblib` y la API (`src/app`) | **Aún sirven el modelo anterior** (XGBoost con `citas_perdidas_noshow`). Reemplazar por la regresión logística en T0 antes de usar el panel para algo más que una demostración. |
| Entrenamiento al arrancar la API (`warmup_predictor`) | Pendiente de eliminar: el modelo debe ser un artefacto versionado. |
| Datos reales y fechas | No disponibles. Ver el plan de mejora. |

### Nuevos módulos de infraestructura

- `modeling_features.py` — contrato de variables por escenario: `FEATURES_T0`, `FEATURES_T1`, `EXCLUDED_LEAKAGE` y `select_features`.
- `experiments.py` — protocolo de evaluación (validación cruzada repetida, líneas base, calibración, subgrupos).
- `interpretability.py` — coeficientes con IC bootstrap, importancia por permutación, efectos marginales y explicación por paciente.

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

### Privacidad

**Nunca subir datos de pacientes reales a GitHub ni a ningún repositorio público.** Este repositorio contiene datos y un modelo serializado solo porque los datos son sintéticos. Con datos reales: repositorio privado o datos fuera del repositorio, `.gitignore` para `data/` y modelos, y seudonimización acordada con la institución. Los CSV de explicación por paciente (`interpretabilidad_pacientes.csv`) contienen identificadores y tampoco deben subirse en ese caso.
