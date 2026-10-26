# Diccionario de datos

Fuente: los 6 CSV de `data/raw/` (= contenido del zip). Datos **sintéticos**. No hay documentación oficial de las columnas: los significados vienen del nombre de la columna y de los valores observados; lo no confirmado se marca **pendiente de validar**.

- Faltantes: **0 (0 %) en todas las columnas de los 6 archivos**, así que no se repite en la tabla.
- AUC: AUC univariado frente a `Desertor` en pacientes etiquetados (0.5 = sin señal). Es asociación, no causalidad.
- Nombre procesado: columna en `data/processed/dataset_final_limpio.csv`.
- Decisión: aplica a los escenarios T0/T1 definidos en `definicion_problema.md` y codificados en `src/infrastructure/ai_models/modeling_features.py`.

## Archivos de deserción (v1: `dataset_desercion_fisura_bilabial.csv`; v2: `dataset_v2_desercion_socioeconomica.csv`)

| Variable (procesada) | Tipo | Significado | Valores observados | Utilidad predictiva potencial | Riesgos | Decisión |
|---|---|---|---|---|---|---|
| `ID_Paciente` (`id_paciente`) | texto | Identificador seudónimo `PAC-NNNN` | v1 1000–1499, v2 2000–4999; únicos | Ninguna | Re-identificación si se cruza con datos reales | Excluir del modelo; conservar como llave |
| `Tipo_Fisura_Bilateral` (`tipo_fisura`) | categórica (3) | Tipo de fisura bilateral | v1: Completa 241, Incompleta 153, Asimétrica 106. v2: Completa 1499, Incompleta 868, Asimétrica 633. Etiquetas distintas entre cohortes ("Bilateral Incompleta" vs "Incompleta"); el ETL las unifica | Baja-media | Dato clínico sensible | **Conservar** (T0), one-hot |
| `Edad_Inicio_Meses` (`edad_inicio_meses`) | entero | Edad al iniciar el tratamiento (meses). **Pendiente de validar**: en v2 la mediana (6) es posterior a la mediana de queiloplastia (4.5) | 1–11 | AUC v1 0.593, v2 0.489 | Fecha de inicio ambigua | **Conservar** (T0); **investigar** definición |
| `Distancia_Hospital_KM` (`distancia_hospital_km`) | real | Distancia residencia–hospital (km). Método de cálculo **pendiente de validar** | v1 5.4–375.9 (mediana 46.7); v2 5.0–300.0 (mediana 40.2). Cola larga a la derecha | AUC v1 0.725, v2 0.586 | Proxy de ubicación; en datos reales almacenar solo agregada | **Conservar** (T0); evaluar log-transformación |
| `Nivel_Socioeconomico` (`nivel_socioeconomico`, `nivel_socioeconomico_ord`) | ordinal | Nivel socioeconómico | v1: Bajo 249, Medio 200, Alto 51 (3 niveles). v2: Bajo 1393, Medio-Bajo 1016, Medio 440, Alto 151 (4 niveles). Escalas distintas entre cohortes | v1: Bajo → 100 % deserción | Sesgo socioeconómico; las escalas de v1 y v2 no son equivalentes | **Conservar** (T0) como ordinal; **investigar** equivalencia entre escalas |
| `Ingresos_Mensuales_USD` (`ingresos_mensuales_usd`) | entero | Ingreso mensual del hogar (USD). Moneda poco realista para Colombia: **pendiente de validar** | v1 300–4982; v2 250–5977 | AUC v1 0.128, v2 0.436 | Dato socioeconómico sensible; muy correlacionado con SES | **Conservar** (T0); revisar colinealidad con SES |
| `Apoyo_Psicologico` (`apoyo_psicologico`) | binaria | Recibe apoyo psicológico. **Pendiente de validar**: si se asigna al ingreso o durante el tratamiento | v1 0:310 / 1:190; v2 0:1818 / 1:1182 | AUC v1 0.277, v2 0.338 (asociado a menos deserción) | Si se asigna por riesgo percibido, es intervención, no rasgo | **Conservar** (T0); **investigar** momento |
| `Cirugias_Programadas` (`cirugias_programadas`) | entero | Número de cirugías programadas | Solo v1: 2–5. v2 no la tiene; el ETL la estima | AUC v1 0.668 | En v2 es estimada → proxy de cohorte | **Excluir** |
| `Citas_Asistidas` (`citas_asistidas`) | entero | Citas asistidas en todo el tratamiento | Solo v1: 2–26. v2 no la tiene; el ETL pone 11 a todos | AUC v1 0.160 | Acumulado del tratamiento (fuga); constante en v2 | **Excluir** |
| — (`citas_asistidas_imputada`) | binaria | Creada por el ETL: 1 si `citas_asistidas` fue imputada | = cohorte v2 | Ninguna legítima | Identifica cohorte | **Excluir** |
| `Citas_Perdidas_NoShow` (`citas_perdidas_noshow`) | entero | Citas perdidas sin aviso en todo el tratamiento | v1 0–18 (mediana 9); v2 0–13 (mediana 4) | AUC v1 **0.991**, v2 0.669 | **Fuga**: acumulado que incluye el periodo posterior al abandono | **Excluir**. Útil en el futuro solo si se cuenta con fechas hasta el momento de predicción |
| `Desertor` (`desertor`) | binaria | 1 = abandonó terapia y tratamiento (definición del equipo; sin regla operativa) | v1 470/30; v2 2142/858 | Objetivo | Etiqueta sin fecha ni motivo de cierre | **Objetivo** |

## Hitos quirúrgicos (v1: `dataset_hitos_quirurgicos_complicaciones.csv`, 1000 filas, 500 sin etiqueta; v2: `dataset_v2_hitos_quirurgicos.csv`)

| Variable (procesada) | Tipo | Significado | Valores observados | Utilidad predictiva potencial | Riesgos | Decisión |
|---|---|---|---|---|---|---|
| `Edad_Queiloplastia_Meses` (`edad_queiloplastia_meses`) | real | Edad en la cirugía de labio (meses) | v1 3–7 (entero); v2 3.0–9.8 | AUC ≈ 0.49–0.54 | Ninguno especial | **Conservar** (T1) |
| `Edad_Palatoplastia_Meses` (`edad_palatoplastia_meses`) | real | Edad en la cirugía de paladar (meses) | v1 9–18; v2 9.0–20.8 | AUC ≈ 0.49 | Ninguno especial | **Conservar** (T1) |
| — (`intervalo_cirugias_meses`) | real | Creada por el ETL: palatoplastia − queiloplastia | — | Redundante | — | **Investigar** (frente 4 decide) |
| `Dificultad_Alimentacion_Preop` (`dificultad_alimentacion_preop`) | binaria | Dificultad de alimentación antes de cirugía | v1 23.7 % (de 1000); v2 64.1 % | AUC ≈ 0.47–0.51 | Muy distinta entre cohortes | **Conservar** (T0) |
| `Dehiscencia_Herida_Postop` (`dehiscencia_herida_postop`) | binaria | Apertura de la herida tras cirugía | v1 6.8 %; v2 12.0 % | AUC ≈ 0.50–0.53 | — | **Conservar** (T1) |
| `Fistula_Palatina_Residual` (`fistula_palatina_residual`) | binaria | Fístula en el paladar tras cirugía | v1 12.8 %; v2 14.0 % | AUC ≈ 0.49–0.55 | — | **Conservar** (T1) |
| — (`complicaciones_quirurgicas`) | entero | Creada por el ETL: dehiscencia + fístula | 0–2 | Redundante con sus componentes | — | Excluir de T1 (se usan los componentes) |
| `Peso_Z_Score_2Anos` (`peso_z_score_2anos`) | real | Puntaje Z de peso a los 2 años | v1 −3.4–2.64; v2 −4.0–2.0 | AUC ≈ 0.50–0.55 | Medido a los 2 años: posterior a la predicción | **Excluir** |

## Terapia de lenguaje (v1: `dataset_terapia_lenguaje_ivf.csv`, 1000 filas; v2: `dataset_v2_terapia_lenguaje.csv`)

| Variable (procesada) | Tipo | Significado | Valores observados | Utilidad predictiva potencial | Riesgos | Decisión |
|---|---|---|---|---|---|---|
| `Citas_Terapia_Programadas` (`citas_terapia_programadas`) | entero | Sesiones de terapia programadas | 20–49 en ambas | AUC ≈ 0.51 | Puede depender de la duración real del tratamiento | **Excluir** hasta validar si se conoce al inicio |
| `Citas_Terapia_Asistidas` (`citas_terapia_asistidas`) | entero | Sesiones de terapia asistidas | v1 9–48; v2 2–46 | AUC v2 **0.030** | **Fuga**: casi define la etiqueta en v2 | **Excluir** |
| — (`ratio_asistencia_terapia`) | real | Creada por el ETL: asistidas / programadas | 0–1 | AUC v2 **0.000** (separación perfecta en 0.49–0.58) | **Fuga** | **Excluir** |
| `Hipoacusia_Conductiva` (`hipoacusia_conductiva`) | binaria | Pérdida auditiva conductiva | v1 35.9 %; v2 37.3 % | AUC ≈ 0.50 | Dato clínico sensible | **Conservar** (T1) |
| `Grado_Hipernasalidad` (`grado_hipernasalidad`) | ordinal 0–3 | Grado de hipernasalidad del habla | v1 mayoría 0; v2 mayoría 1 | AUC v2 **0.770** | **Posible fuga**: resultado del habla que puede depender de la terapia recibida | **Conservar** (T1) **con análisis de sensibilidad** (con/sin) |
| `Insuficiencia_Velofaringea_IVF` (`insuficiencia_velofaringea_ivf`) | binaria | Diagnóstico de insuficiencia velofaríngea | v1 27.0 %; v2 33.4 % | AUC v2 0.571 | Igual que hipernasalidad si se diagnostica tarde | **Conservar** (T1); **investigar** momento |

## Columnas añadidas por el ETL (no vienen en el zip)

`cohorte` (excluir: identifica el archivo de origen) y `fisura_Completa` / `fisura_Asimetrica` / `fisura_Incompleta` (one-hot de `tipo_fisura`; el pipeline del frente 4 debe codificar dentro del `Pipeline`, no reutilizarlas).

## Variables deseadas que NO existen

Satisfacción, percepción de progreso, relación con el personal, intención de continuar, barreras reportadas, motivación, estrés, ansiedad, estado de ánimo, autoeficacia, tiempo de desplazamiento, transporte, horario, fechas de citas, motivo de cierre del tratamiento. Ver frente 5 (`limitaciones_y_plan_mejora.md`).
