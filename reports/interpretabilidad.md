# Interpretabilidad y salida del sistema

Datos **sintéticos** · cohorte v2 · modelo de referencia del frente 4: regresión logística regularizada · Módulo de reproducción: `src/infrastructure/ai_models/interpretability.py` (salidas en `reports/modelos/interpretabilidad_*.csv`).

Convención: **[H]** hecho comprobado · **[R]** resultado calculado · **[P]** hipótesis pendiente · **[Rec]** recomendación.

Este documento explica qué señales usa el modelo, cómo se lee una predicción y cómo debería presentarse la alerta. **No** afirma que lo aprendido describa pacientes reales: los datos son sintéticos, no hay fechas y la etiqueta `Desertor` no tiene regla documentada (ver `auditoria.md`, `definicion_problema.md` y `comparacion_modelos.md` §10).

---

## Resumen ejecutivo

1. **Sin habla, el modelo se apoya en tres variables:** `apoyo_psicologico`, `distancia_hospital_km` e `ingresos_mensuales_usd`. Permutar el apoyo psicológico baja el ROC-AUC del holdout en 0.150; la distancia, en 0.043; los ingresos, en 0.026. Todas las demás variables de T0 aportan menos de 0.003. Regresión logística, Random Forest y XGBoost coinciden en este ranking. [R]
2. **En T1, una sola variable domina:** `grado_hipernasalidad` (caída de 0.306 de ROC-AUC al permutarla; razón de momios ≈ 7.4 por desviación estándar, IC 6.3–8.8). Es la misma variable que el frente 4 mostró como responsable de todo el salto de 0.72 a 0.89, y que puede ser consecuencia del abandono. T1 con habla es una **cota superior**, no un resultado. [R + P]
3. **Los coeficientes de T1 no se pueden leer variable por variable.** Fístula residual e IVF salen con signo *protector* en el modelo aunque su asociación simple con la deserción es nula o positiva; es un efecto de su relación con la hipernasalidad (§4), y llega a producir explicaciones absurdas por paciente (§7). [R]
4. **La alerta no es un nivel de riesgo clínico.** Aun el 30 % de pacientes con menor puntaje en T0 deserta en 49 %. Se propone presentar una *prioridad de seguimiento* definida por la capacidad del equipo, no los cortes 0.33 / 0.66 (§6). [R + Rec]
5. Todo lo anterior es **asociación**, no causa: que el apoyo psicológico se asocie con menos deserción no demuestra que darlo la reduzca (§5). [H]

---

## 1. Método

- **Modelo [H]:** regresión logística del frente 4, sin pesos de clase, variables numéricas estandarizadas y `tipo_fisura` en one-hot dentro del `Pipeline`. `C = 0.01` en T0 y T1 sin habla; en T1 el frente 4 usó `C = 1.0`, pero `interpretability.py` repite la selección y eligió `C = 0.1` (sus coeficientes coinciden exactamente con un ajuste independiente con `C = 0.1`). El ROC-AUC del holdout es prácticamente el mismo (0.8768 con `C = 0.1` frente a 0.8772 con `C = 1.0`); lo que cambia es que los coeficientes de T1 quedan algo más encogidos.
- **Reproducción [H]:** `interpretability.py` reutiliza `fit_and_select` y la partición de desarrollo / holdout del frente 4 (2400 y 600 pacientes), así que el modelo explicado es el mismo que se evaluó. Los coeficientes del módulo coinciden con los de un cálculo independiente hecho con scikit-learn sobre la misma partición (apoyo −0.551, distancia +0.310, ingresos −0.232).
- **Coeficientes [R]:** escala de log-momios por **1 desviación estándar** de cada variable (por eso son comparables entre sí). Intervalos: percentiles 2,5–97,5 de remuestreos bootstrap del conjunto de desarrollo; cada remuestreo reajusta los coeficientes con el `C` ya elegido (no repite la búsqueda de hiperparámetros), por lo que subestiman algo la incertidumbre. Un intervalo que contiene el 0 (o una razón de momios que contiene el 1) indica que no hay evidencia de efecto. El módulo también reporta la **estabilidad de signo**: fracción de remuestreos con el mismo signo que el coeficiente.
- **Importancia por permutación [R]:** caída media del ROC-AUC en el holdout al barajar una variable, repetida varias veces y calculada para regresión logística, Random Forest y XGBoost. Mide cuánto *usa* el modelo la variable, no cuánto la causa.
- **Por qué no SHAP [H]:** el frente 4 mostró que Random Forest y XGBoost no rinden más que la regresión logística (diferencias ≤ 0.003 de ROC-AUC en T0 y T1 sin habla). Con un modelo lineal los coeficientes ya son la explicación exacta; una explicación posterior sobre un modelo más complejo añadiría costo sin añadir información (`comparacion_modelos.md` §11, punto 4).

---

## 2. Escenario T0 (al ingreso) — modelo principal

| Variable | Coeficiente (log-momios por 1 DE) | Razón de momios por 1 DE | IC 95 % de la razón de momios | Estabilidad de signo | Caída de ROC-AUC al permutar |
|---|---|---|---|---|---|
| `apoyo_psicologico` | −0.551 | 0.58 | [0.54, 0.62] | 1.00 | **0.150** |
| `distancia_hospital_km` | +0.310 | 1.36 | [1.27, 1.48] | 1.00 | **0.043** |
| `ingresos_mensuales_usd` | −0.232 | 0.79 | [0.73, 0.85] | 1.00 | **0.026** |
| `nivel_socioeconomico_ord` | −0.055 | 0.95 | [0.88, 1.03] | 0.92 | 0.002 |
| `tipo_fisura` Completa / Incompleta / Asimétrica | −0.054 / +0.046 / +0.007 | 0.95 / 1.05 / 1.01 | [0.88, 1.01] / [0.97, 1.13] / [0.94, 1.09] | 0.93 / 0.89 / 0.61 | 0.002 |
| `edad_inicio_meses` | −0.024 | 0.98 | [0.91, 1.05] | 0.71 | 0.002 |
| `dificultad_alimentacion_preop` | +0.016 | 1.02 | [0.94, 1.10] | 0.63 | 0.000 |

Fuente: `interpretability.py`, escenario T0, 200 remuestreos bootstrap y 20 repeticiones de permutación. La estabilidad de signo es la fracción de remuestreos con el mismo signo que el coeficiente; un intervalo que incluye 1 y una estabilidad baja indican que no hay evidencia de efecto. Los otros dos modelos coinciden en el ranking: apoyo 0.150 / 0.153, distancia 0.042 / 0.047 e ingresos 0.024 / 0.039 (Random Forest / XGBoost).

**Lectura [R]:**

- **Apoyo psicológico.** Pasar de no recibirlo a recibirlo multiplica los momios de deserción por ≈ 0.32, manteniendo las demás variables iguales. Coincide con las tasas crudas del EDA: 82.3 % de deserción sin apoyo y 54.7 % con apoyo (`eda_tables.md` §9). Es la señal principal del modelo.
- **Distancia.** Cada 50 km adicionales multiplican los momios por ≈ 1.37. En crudo, la deserción sube de 66 % en el cuartil más cercano a 82 % en el más lejano; el aumento es sobre todo en el último cuartil (> 75 km), así que el efecto no es lineal y el modelo lo aproxima como recta. En el efecto marginal del modelo (probabilidad media si todos los pacientes tuvieran esa distancia), pasa de 0.66 a 9.8 km a 0.70 a 40 km y 0.79 a 118.5 km. [R]
- **Ingresos y nivel socioeconómico.** En el efecto marginal, llevar a todos los pacientes de SES 0 a SES 4 reduce la probabilidad media de 0.72 a 0.68, una diferencia pequeña. En T1 el mismo ejercicio da el signo contrario (de 0.70 a 0.76): el coeficiente del SES cambia de signo entre escenarios porque comparte señal con los ingresos, otra razón para no interpretarlo. Ingresos tiene coeficiente negativo claro; el SES ordinal, casi nulo. No significa que el SES no importe: ambas variables tienen Spearman de 0.926 (`eda_tables.md` §11), de modo que el modelo reparte entre ellas una misma señal y la regularización fuerte (`C = 0.01`) la concentra en los ingresos. Quitar los ingresos deja el ROC-AUC en 0.734 (frente a 0.737) en un cálculo independiente, así que la información queda cubierta por el SES. **No se debe interpretar la diferencia entre ambos coeficientes.** [R]
- **Tipo de fisura, edad de inicio y dificultad de alimentación** no aportan: sus razones de momios incluyen el 1 y su importancia por permutación es ≈ 0. [R]
- **Costo de la regularización [H]:** la regresión logística eligió el menor `C` de la rejilla (0.01) en T0 (`comparacion_modelos.md` §9). Los coeficientes están encogidos hacia 0; las razones de momios reales podrían ser algo mayores. No se amplió la rejilla.

### T1 sin habla es T0

Agregar edades quirúrgicas, dehiscencia, fístula e hipoacusia no cambia nada: los coeficientes de T0 casi no se mueven (apoyo −0.551, distancia +0.308, ingresos −0.231) y los de las nuevas variables tienen intervalos que incluyen el 0 y caídas de ROC-AUC ≤ 0.002 al permutarlas. [R] Es coherente con la diferencia de −0.0006 de ROC-AUC entre T1 sin habla y T0 del frente 4.

---

## 3. Escenario T1 (con habla) — cota superior, no resultado

| Variable | Coeficiente | Razón de momios por 1 DE | IC 95 % de la razón de momios | Caída de ROC-AUC (RL / RF / XGB) |
|---|---|---|---|---|
| `grado_hipernasalidad` | **+1.996** | 7.36 | [6.34, 8.76] | **0.306** / 0.230 / 0.286 |
| `fistula_palatina_residual` | −1.142 | 0.32 | [0.27, 0.36] | 0.070 / 0.025 / 0.053 |
| `apoyo_psicologico` | −0.635 | 0.53 | [0.48, 0.58] | 0.044 / 0.067 / 0.047 |
| `ingresos_mensuales_usd` | −0.406 | 0.67 | [0.54, 0.80] | 0.027 / 0.013 / 0.010 |
| `distancia_hospital_km` | +0.445 | 1.56 | [1.38, 1.78] | 0.014 / 0.015 / 0.010 |
| `insuficiencia_velofaringea_ivf` | −0.135 | 0.87 | [0.75, 1.00] | 0.004 / 0.005 / 0.000 |
| `nivel_socioeconomico_ord` | +0.140 | 1.15 | [0.94, 1.44] | 0.000 / 0.001 / 0.000 |
| Resto (fisura, edades, dehiscencia, hipoacusia, alimentación) | cerca de 0 | 0.93 a 1.07 | incluyen 1 | ≤ 0.004 |

Fuente: `interpretability.py`, escenario T1 (`C = 0.1`), 200 remuestreos y 20 repeticiones. RL = regresión logística, RF = Random Forest, XGB = XGBoost. Los tres modelos ponen a la hipernasalidad en primer lugar con mucha ventaja.

**Hipernasalidad.** La tasa de deserción en v2 pasa de 20.6 % en grado 0 a 94.5 % en grado 3 (`eda_tables.md` §10). El modelo convierte eso en una razón de momios de ≈ 7 por cada desviación estándar (IC 6.3–8.8), y ninguna otra variable se le acerca. [R]

**Por qué importa que sea una variable sospechosa [P].** Si el grado de hipernasalidad se mide después de iniciar la terapia, refleja en parte cuánta terapia recibió el paciente, y esa terapia depende de que no haya abandonado. En ese caso el modelo no estaría detectando un riesgo previo sino leyendo un efecto del abandono. Los datos no permiten decidirlo porque no hay fechas (pregunta 4 para Fisulab, `definicion_problema.md` §6). Mientras tanto, **ningún resultado de T1 con habla debe presentarse como evidencia de desempeño**, solo como el máximo que se obtendría si la variable estuviera disponible antes del abandono.

---

## 4. Advertencia: los coeficientes de T1 engañan

En T1, fístula residual e IVF aparecen como **protectoras** (razones de momios por desviación estándar de 0.32 y 0.87; el intervalo del IVF, [0.75, 1.00], roza el 1). Pero:

| Variable | Deserción sin la condición | Deserción con la condición |
|---|---|---|
| Fístula residual | 71.9 % | 68.5 % |
| IVF | 67.1 % | 80.1 % |

Fuente: `eda_tables.md` §10. La fístula casi no se asocia con la deserción y el IVF se asocia con **más** deserción; ninguna es protectora. [R]

La explicación más plausible es colinealidad: fístula e IVF están correlacionadas con hipernasalidad (Spearman 0.512 y 0.546, `eda_tables.md` §11). Una vez que el modelo conoce el grado de hipernasalidad, un paciente *con* fístula ya tiene "explicado" parte de ese grado por la fístula, y a igual hipernasalidad tiene menos deserción esperada que uno *sin* fístula. Es un efecto de supresión estadística, no un efecto clínico. [R + P]

No es solo un artefacto del modelo lineal: XGBoost también usa la fístula (caída de ROC-AUC de 0.053), así que los datos contienen información condicional real sobre ella una vez conocida la hipernasalidad. Eso es coherente con que la hipernasalidad sea un indicador indirecto de la asistencia a terapia (que parece originar la etiqueta, `eda.md` §5.2) y que la fístula "explique" parte de ella [P], pero **no autoriza a leerla como protección clínica**.

Consecuencia práctica: en un paciente concreto, la presencia o ausencia de fístula mueve la explicación en un sentido contrario al clínico (§7). **Los coeficientes de variables correlacionadas no deben leerse como efectos individuales**, y esto es otra razón para no basar mensajes clínicos en T1.

---

## 5. Qué significan y qué no significan estos resultados

- **Asociación, no causa [H].** El modelo solo ve correlaciones en un conjunto de datos. Que `apoyo_psicologico` pese mucho no prueba que ofrecer apoyo reduzca la deserción.
- **El apoyo psicológico puede ser una intervención, no un rasgo [P].** Si se asigna a pacientes que el equipo ya considera en riesgo, el modelo estaría aprendiendo la decisión del equipo. Falta saber cuándo se asigna (pregunta 4). Si se asigna después del ingreso, no debería estar en T0.
- **Datos sintéticos [H].** Las relaciones aprendidas son las que el generador de datos puso, incluida una distancia con efecto concentrado en los pacientes más lejanos y una etiqueta que, según el EDA (§5.2), parece construida a partir de la asistencia a terapia. No describen a pacientes reales.
- **Capacidad modesta [R].** En T0, ROC-AUC 0.737 [0.693–0.783] en el holdout, frente a 0.5 de un clasificador sin información. Sirve para ordenar pacientes, no para decidir sobre uno solo.

---

## 6. La salida del sistema

### 6.1 Los umbrales 0.33 / 0.66 no sirven

En el holdout de T0, las probabilidades predichas van de 0.21 a 0.96 con mediana 0.758. Con los cortes actuales quedarían **1.2 % de pacientes en "Bajo", 36.2 % en "Medio" y 62.7 % en "Alto"**: la categoría "Bajo" casi no existe y "Alto" incluye a la mayoría. [R] El frente 4 ya había observado que el decil de menor puntaje promedia ≈ 0.44 sin habla (`comparacion_modelos.md` §6).

Esto es esperable: con una prevalencia de 71 %, casi todos los pacientes tienen probabilidad alta de desertar. La probabilidad está bien calibrada (ECE 0.026 en T0), así que no es un error del modelo, sino un rasgo del problema.

### 6.2 Qué observamos por grupos de puntaje (T0, holdout, 600 pacientes)

| Grupo por puntaje | Pacientes | Deserción observada | Probabilidad media predicha |
|---|---|---|---|
| 20 % más alto | 120 | 90.8 % | 0.87 |
| 30 % más alto | 180 | 88.3 % | 0.85 |
| Medio (30–70 %) | 240 | 75.4 % | 0.73 |
| 30 % más bajo | 180 | 48.9 % | 0.53 |
| 20 % más bajo | 120 | 42.5 % | 0.51 |

El orden funciona (de 43 % a 91 %), pero **ni siquiera el grupo "bajo" es de bajo riesgo**: casi la mitad deserta. [R]

### 6.3 Propuesta [Rec]

1. **No llamar "riesgo Bajo/Medio/Alto" a la salida.** Usar **prioridad de seguimiento** (alta / media / habitual), porque la categoría más baja no significa "poco riesgo".
2. **Definir los cortes por capacidad, no por probabilidad.** Si el equipo puede acompañar a *k* pacientes por semana, la prioridad alta son los *k* con mayor puntaje. Un punto de partida de trabajo, a validar con Fisulab (pregunta 6): alta = 20 % superior, media = 50 % central, habitual = 30 % inferior.
3. **Mostrar la probabilidad con su incertidumbre y su base.** Por ejemplo: "puntaje 0.87; en este grupo desertó el 91 % de los pacientes del conjunto de prueba (datos sintéticos)", y no un diagnóstico.
4. **Si el objetivo es no perder desertores (recall ≥ 0.90)**, el costo es alertar a ≈ 83 % de los pacientes con una especificidad de 0.34 (`comparacion_modelos.md` §6). La alerta casi no filtra; conviene decirlo explícitamente antes de usarla.

### 6.4 Texto que debe acompañar a cada nivel [Rec]

| Nivel | Texto sugerido |
|---|---|
| Prioridad alta | "Este paciente está entre los de mayor probabilidad estimada de abandono. Se sugiere priorizar un contacto de seguimiento. No es un diagnóstico ni un pronóstico." |
| Prioridad media | "Probabilidad estimada intermedia. Seguimiento según la rutina del servicio." |
| Prioridad habitual | "Probabilidad estimada menor que la del resto de pacientes, aunque no es baja en términos absolutos. Mantener el seguimiento habitual." |
| En todos | "Herramienta de apoyo a la gestión del seguimiento. La decisión clínica y de contacto la toma siempre el personal de salud con la historia completa." |

No usar lenguaje como "va a desertar", "paciente problemático" ni mostrar la probabilidad sin contexto.

### 6.5 Equidad en la alerta

Con un umbral único, la sensibilidad cambia mucho por grupo (`comparacion_modelos.md` §8): en T0, el recall es 0.94 en SES 0 y 0.57 en SES 4, y 0.75 en el cuartil de menor distancia frente a 0.98 en el de mayor. Los pacientes de bajo nivel y lejanos reciben más alertas porque su prevalencia es mayor (SES 0: 74.6 %; SES 4: 50 %; cuartil más lejano: 82.2 %), no necesariamente porque el modelo los ordene peor (ROC-AUC por grupo 0.67–0.73). [R] **La alerta debe usarse para dirigir apoyo (transporte, acompañamiento, teleconsulta), nunca para restringir atención.** SES 4 tiene solo 120 pacientes: sus cifras son inestables.

---

## 7. Cómo se lee la explicación de un paciente

La explicación es **exacta**: el modelo es lineal en log-momios, así que log-momios del paciente = log-momios del paciente promedio + suma de contribuciones. Cada contribución es (valor estandarizado × coeficiente) respecto al promedio del holdout. Positivo = sube la probabilidad de deserción. Los pacientes de abajo son del **holdout** (salida de `interpretabilidad_pacientes.csv`, T0), de modo que el modelo no los vio al entrenar.

| Paciente | Probabilidad | Factor 1 | Factor 2 | Factor 3 |
|---|---|---|---|---|
| PAC-3209 (puntaje más alto) | 0.960 | Distancia: **+1.52** (+11.9 pts) | Sin apoyo psicológico: +0.47 (+2.2 pts) | Ingresos: +0.12 (+0.5 pts) |
| PAC-3054 | 0.952 | Distancia: **+1.53** (+14.1 pts) | Sin apoyo psicológico: +0.47 | Ingresos: +0.06 |
| PAC-3791 (puntaje más bajo de los 5) | 0.287 | Ingresos: **−1.10** (−26.0 pts) | Con apoyo psicológico: −0.66 (−15.0 pts) | SES: −0.17 (−3.7 pts) |
| PAC-3966 (puntaje más bajo del holdout) | 0.214 | Ingresos: **−1.12** (−24.2 pts) | Con apoyo psicológico: −0.66 (−13.1 pts) | Distancia: −0.26 (−4.8 pts) |

Los puntos son el cambio en probabilidad al quitar ese factor. Lectura: los pacientes de mayor puntaje combinan una distancia muy grande con no recibir apoyo; los de menor puntaje, ingresos altos y apoyo psicológico. Las tres variables que el ranking global ya señalaba son las únicas que aparecen entre los factores principales. [R]

**Límites de esta explicación:**

- Son **asociaciones del modelo**, no causas: "la distancia explica +1.52" significa que, comparado con el paciente promedio, la distancia empuja la predicción hacia arriba, no que acercar al paciente al hospital lo evite.
- El centrado usa el promedio del holdout; con otro conjunto de referencia las contribuciones se desplazan una constante. Para uso real conviene fijar el paciente promedio con los datos de desarrollo.
- **En T1 la explicación por paciente engaña** (§4). Con el modelo de T1, en el holdout: PAC-4685 tiene probabilidad **0.028** y su factor principal es tener `fistula_palatina_residual` (−2.83 log-momios, −29.7 puntos de probabilidad), más que su hipernasalidad (−0.76). A la inversa, los pacientes de mayor puntaje (PAC-4721, PAC-4273, PAC-4442; probabilidad ≈ 0.998) reciben +0.47 por *no* tener fístula. La explicación diría "tiene fístula, por eso hay menos riesgo", lo cual es clínicamente absurdo (en crudo, la deserción con fístula es 68.5 % frente a 71.9 % sin ella). En los pacientes de mayor probabilidad el factor dominante es una hipernasalidad alta (+3.91 con grado 3: PAC-4721, PAC-4273, PAC-3301, PAC-4442). Por eso las explicaciones por paciente se limitan a T0.

**Recomendación [Rec]:** en la interfaz, mostrar solo los 2–3 factores con mayor contribución absoluta en T0, redactados como asociación ("vive a mayor distancia que la mayoría de pacientes: se asocia con más abandono") y no como causa. Con datos reales, estos CSV contienen identificadores de pacientes y no deben subirse al repositorio.

---

## 8. Variables excluidas por fuga y por qué

Estas variables **no están en el modelo** y no deben reaparecer en explicaciones (`diccionario_datos.md`, `eda_tables.md` §5):

| Variable | AUC univariado v2 | Motivo |
|---|---|---|
| `ratio_asistencia_terapia` | 0.000 | Separa la etiqueta perfectamente (desertor ≤ 0.49; no desertor ≥ 0.58): acumulado de todo el tratamiento. |
| `citas_terapia_asistidas` | 0.030 | Casi define la etiqueta. Un paciente que abandona deja de asistir *porque* abandonó. |
| `citas_perdidas_noshow` | 0.669 (v1: 0.991) | Acumulado que incluye el periodo posterior al abandono. Es la variable que usa el `model.joblib` actual. |
| `cirugias_programadas`, `citas_asistidas` (+ `citas_asistidas_imputada`) | — | En v2 se imputan con valores de v1: identifican la cohorte. |
| `peso_z_score_2anos` | 0.495 | Medido a los 2 años, posterior al momento de predicción. |

Si hubiera fechas, `citas_perdidas_noshow` hasta el momento de la predicción sí podría ser un predictor legítimo (ver `limitaciones_y_plan_mejora.md`, plan de mejora).

---

## 9. Reproducción

```powershell
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.interpretability
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.interpretability --scenario T1
```

Opciones útiles: `--scenario {T0,T1,T1_sin_habla}`, `--bootstrap N` (por defecto 200), `--repeats-permutation N` (por defecto 20). Genera en `reports/modelos/`: `interpretabilidad_coeficientes.csv`, `interpretabilidad_permutacion.csv`, `interpretabilidad_marginal.csv`, `interpretabilidad_pacientes.csv` y `figuras/interpretabilidad_or.png`.

**Procedencia de las cifras [H].** Todas las tablas de §2, §3 y §7 (coeficientes, razones de momios, intervalos, estabilidad de signo, permutación, efectos marginales y pacientes) vienen de la corrida completa de `interpretability.py` en T0 y T1 (200 remuestreos bootstrap, 20 repeticiones de permutación). Los números de §6 (grupos de puntaje, umbrales 0.33 / 0.66, distribución de probabilidades) y el párrafo "T1 sin habla es T0" de §2 vienen de un cálculo exploratorio independiente sobre la misma partición, y los coeficientes del módulo en T0 coinciden con ese cálculo. La caída de ROC-AUC de la distancia en una corrida rápida previa (0.029, con 3 repeticiones) no se repitió: la corrida completa da 0.043, igual que el cálculo independiente.

**Archivos:** `reports/modelos/interp_T0/` y `reports/modelos/interp_T1/` (CSV y `figuras/interpretabilidad_or.png`). La figura de T1 lleva el título "T0" porque `plot_odds_ratios` lo tiene fijo; corregir pasando el escenario como parámetro antes de usarla en el informe.
