# Informe de auditoría — repositorio y dataset

Fecha: 2026-10-26 · Rama: `feat/auditoria-contrato-features` · Base auditada: commit `36040f4` ("cuarto avance del proyecto").

Convención: **[H]** hecho comprobado · **[R]** resultado calculado sobre los datos · **[P]** hipótesis pendiente de validar · **[Rec]** recomendación.

## 1. Recursos inspeccionados

| Recurso | Estado |
|---|---|
| Repositorio local (clon de `lRicardol/Proyecto-predicci-n-de-desertamiento-del-tratamiento-de-fisura-bilabial`) | Inspeccionado. Local = remoto (5 commits, 24-oct-2026). |
| `data sets para el proyecto.zip` | Inspeccionado. 6 CSV, idénticos a `data/raw/` (md5 igual tras normalizar finales de línea CRLF). [H] |
| Documentación de las columnas | **No existe.** Solo nombres de columnas y README. [H] |
| Origen de los datos | **Sintéticos**, confirmado por el equipo. El README dice "3 500 pacientes reales": hay que corregirlo. [H] |

## 2. Repositorio

**Stack [H]:** Python 3.11, FastAPI + Uvicorn, pandas, scikit-learn, XGBoost, joblib, Matplotlib/Seaborn, pydantic-settings, pytest (`requirements.txt` con versiones mínimas, sin fijar).

**Estructura [H]:** Clean Architecture.

| Capa | Contenido | Reutilizable |
|---|---|---|
| `src/core` | Puertos (`IPatientRepository`, `IPredictorModel`, `ILLMAnalyst`), casos de uso, excepciones de dominio | Sí, tal cual |
| `src/infrastructure/repositories` | `CsvPatientRepository` | Sí |
| `src/infrastructure/ai_models` | `feature_engineering.py` (4 features, `SES_ORDINAL`), `xgboost_predictor.py`, `model.joblib` | `SES_ORDINAL` sí; features y modelo requieren rehacerse (ver §4) |
| `src/infrastructure/llm_agents` | `LocalStatisticalAnalyst` (reglas + gráficos); `langchain_analyst.py` es solo un alias | Sí, para el panel |
| `src/app` | Composition root, controladores, `dashboard.html` | Sí |
| `notebooks/` | `00_*.py` ETL; `01`, `02` notebooks generados por `_build_notebooks.py` | `load_csv`, `merge_cohort`, `harmonize_tipo_fisura`, `encode_socioeconomic` |
| `tests/` | 2 archivos: repositorio, entrenamiento XGBoost, casos de uso con dobles | Sí |

**Problemas de ingeniería [H]:**

1. No había `.gitignore` → se agregó en esta rama.
2. Repo **público** con datos y modelo serializado. Al ser datos sintéticos no hay fuga de datos personales, pero la regla debe quedar escrita para cuando lleguen datos reales: **nunca** subir datos de pacientes a GitHub. [Rec]
3. `requirements.txt` sin versiones fijadas → resultados no exactamente reproducibles. [Rec: fijar con `pip freeze` al cerrar el entrenamiento]
4. El ETL vive en `notebooks/`, no en un paquete importable → no se puede probar ni reutilizar desde `src/`. [Rec: frente 2]
5. `xgboost_predictor.py` entrena automáticamente si falta `model.joblib` al arrancar la API: el modelo de producción depende de un efecto secundario del arranque. [Rec]
6. Tests de `test_predictors.py` asumen `len == 3500` y entrenan con datos reales (lentos, acoplados al CSV).

## 3. Dataset

### 3.1 Archivos [H]

| Archivo | Filas | Columnas | IDs | Cohorte |
|---|---|---|---|---|
| `dataset_desercion_fisura_bilabial.csv` | 500 | 11 | PAC-1000…1499 | v1 |
| `dataset_hitos_quirurgicos_complicaciones.csv` | 1000 | 7 | PAC-1000…1999 | v1 |
| `dataset_terapia_lenguaje_ivf.csv` | 1000 | 6 | PAC-1000…1999 | v1 |
| `dataset_v2_desercion_socioeconomica.csv` | 3000 | 9 | PAC-2000…4999 | v2 |
| `dataset_v2_hitos_quirurgicos.csv` | 3000 | 7 | PAC-2000…4999 | v2 |
| `dataset_v2_terapia_lenguaje.csv` | 3000 | 6 | PAC-2000…4999 | v2 |

- Llave `ID_Paciente`: única en cada archivo, sin nulos, sin filas duplicadas. [H]
- v1: hitos y terapia traen 1000 IDs, pero solo 500 tienen etiqueta → 500 pacientes sin `Desertor` (el ETL los descarta). [H]
- v1 y v2 no comparten IDs. [H]
- v2 **no tiene** `Cirugias_Programadas` ni `Citas_Asistidas`. [H]
- Ningún archivo tiene **fechas** (de ingreso, citas, cirugías ni cierre). [H]
- Una fila por paciente: no hay datos longitudinales. [H]

### 3.2 Variable objetivo [R]

| Cohorte | Pacientes | Desertores | No desertores | Tasa |
|---|---|---|---|---|
| v1 | 500 | 470 | **30** | 94.0 % |
| v2 | 3000 | 2142 | 858 | 71.4 % |
| Total | 3500 | 2612 | 888 | 74.6 % |

v1 tiene solo 30 no desertores: insuficiente para entrenar o evaluar sobre esa cohorte.

### 3.3 Efecto cohorte [R]

Las dos cohortes no vienen de la misma distribución (medias sobre todas las filas de cada archivo):

| Variable | v1 (media) | v2 (media) |
|---|---|---|
| Citas perdidas (no-show) | 8.6 | 4.0 |
| Citas de terapia asistidas | 26.9 | 14.7 |
| Peso Z a los 2 años | −0.14 | −0.80 |
| Dificultad de alimentación preop. | 23.7 % | 64.1 % |

Mezclarlas sin control hace que el modelo aprenda "de qué cohorte viene el paciente" en lugar del riesgo.

### 3.4 Fuga de información [R]

AUC univariado (0.5 = sin señal; cerca de 0 o 1 = separa la etiqueta):

| Variable | AUC v1 | AUC v2 | Lectura |
|---|---|---|---|
| `ratio_asistencia_terapia` (procesada) | 0.538 | **0.000** | En v2 separa perfecto: desertor ≤ 0.49, no desertor ≥ 0.577 |
| `Citas_Terapia_Asistidas` | 0.509 | **0.030** | Casi perfecta en v2 |
| `Citas_Perdidas_NoShow` | **0.991** | 0.669 | En v1, no-show ≤ 2 → 4 % deserción; ≥ 6 → 100 % |

Estas variables son **conteos acumulados de todo el tratamiento**: un paciente que abandona deja de asistir, así que su asistencia baja *porque* abandonó. Usarlas para "predecir" el abandono es circular. [R + P: se confirmaría con fechas]

`Grado_Hipernasalidad` tiene AUC 0.770 en v2 y 0.544 en v1. Es un resultado del habla que puede depender de haber recibido terapia → posible consecuencia del abandono. **[P] Pendiente de validar cuándo se mide.**

### 3.5 Problemas del ETL actual [H]

- `citas_asistidas` de v2 (inexistente) se rellena con la mediana de v1 → constante **11** para los 3000 pacientes de v2.
- `cirugias_programadas` de v2 se estima con otra fórmula (2 + complicaciones) → valores 2–4 contra 2–5 en v1.
- Ambas columnas terminan identificando la cohorte.
- La imputación se hace sobre todo el dataset antes de dividir train/test.

### 3.6 Modelo actual [H/R]

- Features: `citas_perdidas_noshow` (fuga), distancia, SES ordinal, complicaciones.
- Un solo split 80/20; sin validación cruzada, sin línea base, sin ROC-AUC, PR-AUC ni calibración.
- Accuracy reportada 0.690 < **0.746** de un clasificador que siempre dice "desertor".
- Umbrales Bajo/Medio/Alto (0.33/0.66) no validados.

Conclusión: las métricas del README no sirven como evidencia de desempeño.

## 4. Viabilidad

- **Modelo supervisado: viable solo como ejercicio metodológico** sobre datos sintéticos, con población v2 y sin las variables de §3.4. Los resultados no se pueden extrapolar a pacientes reales.
- No hay predicción temporal posible (sin fechas): se definen escenarios por *etapa* del tratamiento (ver `definicion_problema.md`).
- Variables de experiencia del paciente y psicológicas (salvo `Apoyo_Psicologico`, binaria): **no existen**. Logística: solo `Distancia_Hospital_KM`.

## 5. Reparto del trabajo restante

| Frente | Responsable | Entregables |
|---|---|---|
| 1. Auditoría, diccionario, definición, contrato de features | Autor de esta rama | Este informe, `diccionario_datos.md`, `definicion_problema.md`, `modeling_features.py` + tests |
| 2. ETL sin fuga | Ricardo (sugerido) | Mover ETL a `src/infrastructure/data/etl.py`; `build_modeling_frame()` sin imputación cruzada entre cohortes; tests |
| 3. EDA | Compañero A | `notebooks/03_eda.py`, `reports/eda.md`, figuras |
| 4. Modelos y evaluación | Compañero B | `experiments.py`: CV repetida, Dummy/LR/árbol/RF/XGBoost, ROC-AUC, PR-AUC, recall, especificidad, Brier, calibración; `comparacion_modelos.md` |
| 5. Interpretabilidad, salida, ética, plan de mejora | Compañero C | `interpretabilidad.md`, `limitaciones_y_plan_mejora.md`, README final |

Los frentes 2–5 consumen `src/infrastructure/ai_models/modeling_features.py` (`FEATURES_T0`, `FEATURES_T1`, `EXCLUDED_LEAKAGE`, `select_features`). La API y `model.joblib` no se tocan hasta integrar.
