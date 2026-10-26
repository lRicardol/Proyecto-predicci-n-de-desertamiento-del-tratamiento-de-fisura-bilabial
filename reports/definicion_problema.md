# Definición del problema predictivo

Estado: **propuesta para validar con Fisulab.** Datos sintéticos; las decisiones de aquí valen para el ejercicio académico y deben revisarse cuando haya datos reales.

## 1. Qué es abandonar

**Definición conceptual (equipo):** desertor es el paciente que abandona la terapia y el tratamiento de Fisulab. *Posible desertor* es el que tiene una probabilidad alta de abandonarlos.

**En los datos [H]:** `Desertor` es binaria (0/1), una por paciente, sin fecha de abandono, sin motivo de cierre y sin regla documentada. No permite distinguir abandono de alta satisfactoria, suspensión médica, remisión, traslado o pérdida temporal de seguimiento.

**Definición operativa propuesta (pendiente de validar):**

> Paciente con tratamiento activo que acumula **≥ N semanas** sin asistir a ninguna cita programada (cirugía, control o terapia), sin registro de alta, remisión, traslado, suspensión médica ni fallecimiento.

N, y qué cuenta como "cita programada", los define Fisulab.

## 2. Población

| Cohorte | Uso | Motivo |
|---|---|---|
| **v2** (3000 pacientes, 858 no desertores) | Entrenamiento y evaluación | Tamaño y balance suficientes |
| v1 (500 etiquetados, 30 no desertores) | Solo descriptiva / comparación | 30 negativos no bastan para entrenar ni evaluar; distribución distinta de v2 |
| v1 sin etiqueta (500) | No se usa | Sin `Desertor` |

No se mezclan cohortes en el entrenamiento: las escalas de SES difieren (3 vs 4 niveles) y el ETL rellena columnas faltantes de v2 con valores de v1.

## 3. Momento de predicción

Sin fechas no se puede definir "predecir con X semanas de anticipación". Se definen dos **etapas** del tratamiento:

| Escenario | Cuándo se predice | Features (`modeling_features.py`) |
|---|---|---|
| **T0 — ingreso** | Al abrir la historia del paciente | `tipo_fisura`, `edad_inicio_meses`, `distancia_hospital_km`, `nivel_socioeconomico_ord`, `ingresos_mensuales_usd`, `apoyo_psicologico`, `dificultad_alimentacion_preop` |
| **T1 — tras cirugías primarias** | Después de la palatoplastia (~9–21 meses de edad) | T0 + `edad_queiloplastia_meses`, `edad_palatoplastia_meses`, `dehiscencia_herida_postop`, `fistula_palatina_residual`, `hipoacusia_conductiva`, `grado_hipernasalidad`, `insuficiencia_velofaringea_ivf` |

**Excluidas** (`EXCLUDED_LEAKAGE`): `citas_terapia_asistidas`, `ratio_asistencia_terapia`, `citas_perdidas_noshow`, `citas_asistidas`, `citas_asistidas_imputada`, `cirugias_programadas`, `citas_terapia_programadas`, `peso_z_score_2anos`, `cohorte`, `id_paciente`, `desertor`. Motivo de cada una en `diccionario_datos.md`.

Supuestos que hay que validar con Fisulab:
- `apoyo_psicologico` se conoce al ingreso (si se asigna después, pasa a T1 o se excluye).
- `dificultad_alimentacion_preop` se conoce al ingreso.
- `grado_hipernasalidad` e `insuficiencia_velofaringea_ivf` se miden antes de la terapia de lenguaje. Si se miden después, son fuga. El frente 4 debe reportar T1 **con y sin** estas dos.

## 4. Volumen

v2: 3000 pacientes, 858 de la clase minoritaria (no desertor). Suficiente para validación cruzada estratificada repetida y un holdout del 20 %. [R]

## 5. Reglas para los frentes siguientes

1. Toda matriz de entrenamiento se obtiene con `select_features(frame, "T0" | "T1")`.
2. Imputación, codificación y escalado van **dentro** de un `sklearn.Pipeline`, ajustados solo con train.
3. Línea base obligatoria: `DummyClassifier` (clase mayoritaria y estratificado).
4. Umbral de decisión elegido en validación cruzada, nunca en el test.
5. Un experimento de control con las columnas excluidas puede mostrarse **solo** como evidencia de fuga (se espera AUC ≈ 1 en v2), nunca como resultado del modelo.
6. Bajo/Medio/Alto: los umbrales se proponen a partir de la curva de capacidad operativa (cuántos pacientes puede acompañar el equipo por semana) y se validan con Fisulab; 0.33/0.66 no tienen respaldo.

## 6. Preguntas abiertas para Fisulab

1. ¿Qué regla generó `Desertor`? ¿Con qué N semanas sin asistir?
2. ¿Existen fechas de citas (programadas, asistidas, canceladas) y de cierre del tratamiento con su motivo?
3. ¿Con cuánta anticipación necesita el equipo la alerta para actuar?
4. ¿Cuándo se asigna el apoyo psicológico y cuándo se miden hipernasalidad e IVF?
5. ¿Cómo se calculó la distancia y por qué el ingreso está en USD?
6. ¿Cuántos pacientes puede acompañar el equipo por semana? (define el umbral de "Alto")
