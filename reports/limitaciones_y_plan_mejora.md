# Limitaciones, ética y plan de mejora — frente 5

Datos **sintéticos** · Fuentes: `auditoria.md`, `diccionario_datos.md`, `definicion_problema.md`, `eda.md`, `comparacion_modelos.md`, `interpretabilidad.md`.

Convención: **[H]** hecho comprobado · **[R]** resultado calculado · **[P]** hipótesis pendiente · **[Rec]** recomendación.

---

## Parte A. Limitaciones

### A.1 Los datos

| # | Limitación | Evidencia | Consecuencia |
|---|---|---|---|
| 1 | **Datos sintéticos.** El README original decía "3 500 pacientes reales"; era incorrecto. | Confirmado por el equipo (`auditoria.md` §1). [H] | Ningún resultado se puede extrapolar a pacientes reales. El proyecto vale como ejercicio metodológico. |
| 2 | **La etiqueta no tiene regla.** `Desertor` es 0/1, sin fecha, sin motivo de cierre y sin definición operativa. | `diccionario_datos.md`; `definicion_problema.md` §1. [H] | No se distingue abandono de alta, remisión, traslado o suspensión médica. |
| 3 | **La etiqueta de v2 parece derivada de la asistencia a terapia.** Los desertores tienen ratio ≤ 0.49 y los no desertores ≥ 0.58, sin solapamiento. | `eda.md` §5.2. [R + P] | Si es así, el problema original es reconstruir una regla y no predecir un abandono. Sin esas variables, el desempeño es modesto (ROC-AUC ≈ 0.72). |
| 4 | **Sin fechas ni datos longitudinales.** Una fila por paciente. | `eda_tables.md` §1. [H] | No se puede definir "alerta con X semanas de anticipación"; solo se definen etapas T0 y T1. |
| 5 | **Dos cohortes incompatibles.** SMD de 1.55 en no-shows, 1.35 en asistencia a terapia y 0.85 en dificultad de alimentación. | `eda_tables.md` §4. [R] | Solo se modela v2 (3000 pacientes). v1 tiene 30 no desertores y no sirve para entrenar ni evaluar. |
| 6 | **Escalas no equivalentes y artefactos del ETL.** El SES tiene 3 niveles en v1 y 4 en v2; en v2 se codifica 0, 1, 2 y 4 (el nivel 3 no existe, así que la distancia Medio–Alto vale 2 y no 1). `citas_asistidas` es constante (11) y `cirugias_programadas` se estima distinto en v2. | `eda_tables.md` §1, §6, §7. [H] | Las columnas imputadas identifican la cohorte y se excluyen. La codificación ordinal del SES supone espaciado uniforme que no se cumple. |
| 7 | **Inconsistencias temporales.** En 2141 pacientes la edad de inicio es posterior a la edad de la queiloplastia. | `eda_tables.md` §1. [H] | La definición de `edad_inicio_meses` es ambigua; no se sabe qué fecha representa. |
| 8 | **Ingresos en USD y SES casi redundantes** (Spearman 0.926). | `eda_tables.md` §11. [R] | Los coeficientes de ambas variables no se pueden interpretar por separado (`interpretabilidad.md` §2). |

### A.2 Variables que faltan

No existen satisfacción, percepción de progreso, relación con el personal, intención de continuar, barreras reportadas, motivación, estrés, ansiedad, autoeficacia, tiempo de desplazamiento, transporte, horario ni motivo de cierre (`diccionario_datos.md`). [H] Solo hay una variable psicológica (`apoyo_psicologico`, binaria) y una logística (`distancia_hospital_km`). El modelo no puede captar las razones por las que una familia abandona; solo correlatos demográficos y clínicos.

### A.3 El modelo

- **Desempeño modesto sin las variables con fuga [R].** En T0, ROC-AUC 0.721 ± 0.021 en validación cruzada y 0.737 [0.693–0.783] en el holdout, frente a 0.5 sin información; PR-AUC 0.858 frente a 0.713 de la prevalencia.
- **Los modelos complejos no mejoran [R].** Random Forest y XGBoost difieren de la regresión logística en ≤ 0.003 de ROC-AUC en T0 y T1 sin habla (`comparacion_modelos.md` §5).
- **T1 con habla es una cota superior, no un resultado [P].** Todo el salto de 0.72 a 0.89 viene de hipernasalidad e IVF, que pueden medirse después de iniciar la terapia (`comparacion_modelos.md` §4).
- **La alerta casi no filtra [R].** Para detectar ≈ 90 % de los desertores en T0 hay que alertar a ≈ 83 % de los pacientes, con especificidad de 0.34. Precisión 0.771 frente a 0.713 de alertar a todos.
- **El recall objetivo no está garantizado [R].** El umbral se fija en entrenamiento para llegar a 0.90; en validación el recall medio de la regresión logística en T0 es 0.876 ± 0.019.
- **Hiperparámetros en el borde de la rejilla [R].** La regresión logística elige `C = 0.01` (el menor probado) y XGBoost la menor tasa de aprendizaje, profundidad y número de árboles. La rejilla puede estar recortando la comparación; no se amplió.
- **Un solo holdout de 600 pacientes [H].** Los intervalos bootstrap no incluyen la variabilidad del particionado.
- **No se hicieron pruebas de significancia formales [H].** Las diferencias de ≈ 0.001–0.003 de ROC-AUC son ruido frente a una desviación de ≈ 0.01 entre pliegues.
- **Los umbrales Bajo / Medio / Alto (0.33 / 0.66) no tienen respaldo [R].** Con ellos, 1.2 % de los pacientes quedaría en "Bajo" (`interpretabilidad.md` §6).
- **El modelo servido hoy no es el que se evaluó [H].** `model.joblib` usa `citas_perdidas_noshow` (fuga) y debe reemplazarse (`auditoria.md` §3.6).

---

## Parte B. Ética y uso responsable

### B.1 Principio de uso

La herramienta es **apoyo a la gestión del seguimiento clínico**. No es diagnóstico, pronóstico ni decisión terapéutica. Toda alerta debe revisarla personal de salud con la historia completa. Este aviso debe aparecer en el panel y en el README. [Rec]

### B.2 Sesgo y equidad

- **Riesgo [R].** El modelo usa distancia, ingresos y nivel socioeconómico. La deserción sube con la distancia (66 % en el cuartil más cercano, 82 % en el más lejano) y baja con el SES (74.6 % en SES 0, 47.7 % en SES 4; `eda_tables.md` §9). Con un umbral único, los pacientes lejanos y de bajo nivel reciben más alertas (recall 0.98 en el cuartil más lejano frente a 0.75 en el más cercano; especificidad 0.08 frente a 0.60; `comparacion_modelos.md` §8).
- **Regla de uso [Rec].** La alerta sirve para **dirigir apoyo** (transporte, acompañamiento, teleconsulta, recordatorios), nunca para limitar, retrasar o condicionar la atención de quien vive lejos o tiene menos recursos. Esto debe quedar escrito en el protocolo del servicio.
- **Monitoreo [Rec].** Antes de cualquier uso real, reportar recall, especificidad y precisión por grupo de SES y distancia, y repetirlo periódicamente.
- **Grupos pequeños.** SES 4 tiene 120 pacientes en desarrollo; sus métricas son inestables. [R]

### B.3 Etiquetado y estigma

Una etiqueta de "posible desertor" puede cambiar el trato hacia una familia. Se recomienda: llamar a la salida *prioridad de seguimiento* y no "riesgo de deserción"; no mostrar la probabilidad cruda al paciente ni a su familia; y no usarla en evaluaciones del desempeño del personal. [Rec]

### B.4 Asociación no es causa

Que el apoyo psicológico se asocie con menos deserción no prueba que ofrecerlo la reduzca; puede asignarse a pacientes que el equipo ya considera en riesgo (`interpretabilidad.md` §5). Las explicaciones deben redactarse como asociación. [Rec]

### B.5 Privacidad y datos

- **Regla [Rec]: nunca subir datos de pacientes reales a GitHub, ni a ningún repositorio público.** El repositorio actual es público y contiene datos y modelo serializado; es aceptable solo porque los datos son sintéticos (`auditoria.md` §2). Cuando lleguen datos reales: repositorio privado o datos fuera del repositorio, `.gitignore` para `data/` y `model.joblib`, y acuerdo de tratamiento de datos con Fisulab.
- Datos de salud de menores, con identificador seudónimo (`PAC-NNNN`): son sensibles incluso sin nombre, porque pueden reidentificarse al cruzarse con otras fuentes.
- Almacenar la distancia solo de forma agregada o por rangos cuando haya datos reales. [Rec]
- Cumplir la normativa colombiana de protección de datos personales (Ley 1581 de 2012) y de datos de salud antes de cualquier uso real. [Rec: validar con el área jurídica de la institución]
- Consentimiento y finalidad: informar a las familias que se usa un sistema de apoyo al seguimiento. [Rec]

### B.6 Transparencia

Documentar con cada versión del modelo: datos de entrenamiento, escenario (T0 / T1), variables, métricas con línea base, umbral y fecha. Reportar siempre la línea base (`DummyClassifier`) junto a cualquier métrica; el README original reportaba accuracy 0.690, por debajo del 0.746 de un clasificador que siempre dice "desertor". [H]

---

## Parte C. Plan de mejora

Ordenado por prioridad. La columna "Resuelve" enlaza cada acción con el problema documentado.

### C.1 Datos y definición (prioridad alta, dependen de Fisulab)

| # | Acción | Resuelve | Responsable sugerido |
|---|---|---|---|
| 1 | **Definir con Fisulab la regla de `Desertor`** (≥ N semanas sin asistir, sin alta, remisión, traslado ni suspensión) y documentarla. | Limitaciones A.1 #2 y #3. | Fisulab + equipo |
| 2 | **Obtener fechas** de ingreso, citas (programadas, asistidas, canceladas), cirugías y cierre del tratamiento con su motivo. | A.1 #4. Permite definir horizontes de predicción y usar `citas_perdidas_noshow` hasta el momento de la predicción sin fuga. | Fisulab |
| 3 | **Confirmar cuándo se asigna el apoyo psicológico y cuándo se miden hipernasalidad e IVF.** | A.3 (T1 como cota superior); B.4. Si se miden antes de la terapia, T1 pasa a ser el escenario principal. | Fisulab |
| 4 | **Aclarar el significado de `edad_inicio_meses`** y las 2141 inconsistencias con la queiloplastia. | A.1 #7. | Fisulab |
| 5 | **Documentar la distancia** (cómo se calcula) y la unidad de ingresos. | A.1 #8; preguntas 5 de `definicion_problema.md`. | Fisulab |
| 6 | **Capturar variables psicosociales y logísticas:** barreras reportadas, transporte, tiempo de desplazamiento, horario, satisfacción, intención de continuar. | A.2. | Fisulab + equipo |
| 7 | **Obtener datos reales** (con convenio y seudonimización), con volumen suficiente para validación temporal. | A.1 #1. | Fisulab |

### C.2 Modelo y evaluación

| # | Acción | Resuelve |
|---|---|---|
| 8 | Unificar la escala de SES (equivalencia entre v1 y v2) y codificarlo sin saltos, o tratarlo como categórico. | A.1 #6. |
| 9 | Ampliar la rejilla hacia más regularización (`C` < 0.01; XGBoost con menos árboles y más `min_child_weight`) y repetir la comparación. | A.3 (hiperparámetros en el borde). |
| 10 | Con fechas: validación **temporal** (entrenar con el pasado, evaluar con pacientes posteriores) y validación externa en otra sede. | A.3 (un solo holdout; sin horizonte). |
| 11 | Fijar el umbral con **margen** (por ejemplo, exigir recall 0.92 en entrenamiento) o con un conjunto de calibración adicional. | A.3 (recall de validación 0.876). |
| 12 | Evaluar por subgrupos con umbrales por grupo o criterios de equidad explícitos (igualdad de recall) y decidir con Fisulab cuál es aceptable. | B.2. |
| 13 | Definir niveles de prioridad por **capacidad semanal** del equipo y validarlos con Fisulab. | Umbrales 0.33 / 0.66; `interpretabilidad.md` §6. |
| 14 | Evaluar impacto real: piloto con grupo de comparación para medir si la intervención sobre pacientes priorizados reduce el abandono. | B.4 (asociación ≠ causa). |

### C.3 Ingeniería (`auditoria.md` §2)

| # | Acción | Resuelve |
|---|---|---|
| 15 | **Mover el ETL a un paquete** (`src/infrastructure/data/etl.py`) con `build_modeling_frame()` sin imputación cruzada entre cohortes, con pruebas. | ETL en `notebooks/`; imputación sobre todo el dataset antes de dividir. (Frente 2) |
| 16 | **Reemplazar `model.joblib`** por el modelo de referencia (regresión logística, T0) y no tocar la API hasta decidir el escenario. | Modelo servido con fuga (`citas_perdidas_noshow`). |
| 17 | **No entrenar al arrancar la API:** el modelo debe ser un artefacto versionado, y la API debe fallar con un mensaje claro si no existe. | Problema de ingeniería #5. |
| 18 | **Fijar versiones** en `requirements.txt` (`pip freeze` al cerrar el entrenamiento). | Problema #3; reproducibilidad. |
| 19 | **Pruebas rápidas e independientes del CSV real:** quitar `len == 3500`, usar datos de prueba pequeños. | Problema #6. |
| 20 | Agregar `.gitignore` para `data/` y modelos cuando haya datos reales. | B.5. |
| 21 | Cambiar los nombres que inducen a error (`ILLMAnalyst`, `llm_agents`, `langchain_analyst.py`): no hay modelo de lenguaje. | Claridad del código. |

### C.4 Orden recomendado

1. Preguntas a Fisulab (#1–#5): desbloquean casi todo lo demás.
2. Integración mínima honesta (#15–#18): reemplazar el modelo servido por el de referencia y corregir el README.
3. Mejoras de evaluación que no requieren datos nuevos (#8, #9, #11, #13).
4. Con datos reales y fechas: #2, #7, #10, #12, #14.

---

## Parte D. Preguntas abiertas para Fisulab

Las seis de `definicion_problema.md` §6 siguen vigentes:

1. ¿Qué regla generó `Desertor`? ¿Con qué N semanas sin asistir?
2. ¿Existen fechas de citas y de cierre del tratamiento con su motivo?
3. ¿Con cuánta anticipación necesita el equipo la alerta?
4. ¿Cuándo se asigna el apoyo psicológico y cuándo se miden hipernasalidad e IVF?
5. ¿Cómo se calculó la distancia y por qué el ingreso está en USD?
6. ¿Cuántos pacientes puede acompañar el equipo por semana?

Se agregan dos que surgieron en este frente:

7. ¿Qué significa `edad_inicio_meses` y por qué es posterior a la queiloplastia en 2141 pacientes?
8. ¿Qué acción concreta se tomaría con un paciente de prioridad alta, y quién la ejecutaría? (define qué métrica importa: recall, precisión o capacidad).
