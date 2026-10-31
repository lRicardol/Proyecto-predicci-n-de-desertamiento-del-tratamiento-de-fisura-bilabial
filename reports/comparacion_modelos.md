# Comparación de modelos — frente 4

**Frente 4** de `reports/auditoria.md` · Datos **sintéticos** · Corrida de `experiments.py` (713.7 s, Python 3.11.9).

Convención: **[H]** hecho comprobado · **[R]** resultado calculado · **[P]** hipótesis pendiente · **[Rec]** recomendación.

Este documento compara modelos bajo un único protocolo. **No** afirma que el modelo sirva para pacientes reales: los datos son sintéticos, no hay fechas y la etiqueta no tiene regla documentada (ver `auditoria.md` y `eda.md`).

---

## Resumen ejecutivo

1. **Todo el desempeño extra de T1 viene de hipernasalidad e IVF.** La regresión logística pasa de ROC-AUC 0.720 (T1 sin habla) a 0.886 (T1), una mejora de 0.166 que aparece en 15 de 15 pliegues; con XGBoost es de 0.168 (15/15). T1 sin habla es indistinguible de T0 (diferencia -0.0006 ± 0.0045): las edades quirúrgicas, las complicaciones y la hipoacusia no aportan. [R]
2. **Esas dos variables podrían ser consecuencia del abandono, no causa.** Si se miden después de la terapia, usarlas es fuga (`definicion_problema.md` §3). Mientras Fisulab no confirme cuándo se miden, **el resultado principal no puede ser T1 con habla**; los números de T1 se reportan como cota superior. [P]
3. **Sin habla, el desempeño es modesto.** En T0 la regresión logística obtiene ROC-AUC 0.721 ± 0.021 en validación cruzada y 0.737 [0.693–0.783] en el holdout; el PR-AUC es 0.858 frente a 0.713 de un clasificador sin información (la prevalencia). [R]
4. **Los cuatro modelos rinden casi igual.** En T0 y T1 sin habla, Random Forest y XGBoost difieren de la regresión logística en ≤ 0.003 de AUC, muy por debajo de la variación entre pliegues (≈ 0.01); el árbol es peor (-0.018 en T0, supera a la regresión logística en 1 de 15 pliegues). Ningún modelo complejo justifica su costo; la regresión logística queda como **modelo de referencia**. [R + Rec]
5. **Exigir recall ≥ 0.90 cuesta casi toda la especificidad.** En T0, la regresión logística en el holdout detecta 89.5 % de los desertores pero marca como alerta al 82.8 % de los pacientes y solo reconoce 33.7 % de los que continúan; su precisión es 0.771 frente a 0.713 de marcar a todos. [R]
6. **El recall objetivo no se cumple de forma garantizada fuera de la muestra.** El umbral se fija para llegar a 0.90 en el entrenamiento; en validación el recall medio de la regresión logística en T0 es 0.876 ± 0.019. [R]

---

## 1. Protocolo

- **Población [H]:** cohorte v2, 3000 pacientes, tasa de deserción 71.4 % (desarrollo) y 71.3 % (holdout). v1 no se usa.
- **Holdout [H]:** 600 pacientes (20 %) apartados al inicio, estratificados. Se evalúan una sola vez, con el modelo y el umbral elegidos en desarrollo.
- **Validación cruzada [H]:** sobre los 2400 pacientes de desarrollo, 5 pliegues estratificados × 3 repeticiones (15 evaluaciones por modelo y escenario). En cada pliegue, la búsqueda de hiperparámetros usa una validación cruzada interna de 3 pliegues y optimiza ROC-AUC (CV anidada).
- **Escenarios [H]:** T0 (al ingreso), T1 (T0 + hitos quirúrgicos y hallazgos posteriores) y T1 sin habla (T1 sin `grado_hipernasalidad` ni `insuficiencia_velofaringea_ivf`). Las matrices salen de `select_features`; ninguna columna de `EXCLUDED_LEAKAGE` entra.
- **Preprocesamiento [H]:** imputación (mediana / moda), codificación one-hot de `tipo_fisura` y, solo para la regresión logística, estandarización; todo dentro del `Pipeline`, ajustado únicamente con los datos de entrenamiento de cada pliegue.
- **Modelos [H]:** tres Dummy (mayoritaria, prior, estratificado), regresión logística, árbol, Random Forest y XGBoost, **sin pesos de clase**.
- **Umbral [H]:** se marca *alerta* si la probabilidad de deserción ≥ umbral. El umbral maximiza la especificidad sujeto a recall ≥ 0.90, calculado con predicciones fuera de muestra del entrenamiento (nunca con el pliegue evaluado ni el holdout).
- **Clase positiva [H]:** `desertor = 1`. Recall = desertores detectados / desertores reales; especificidad = continuadores no alertados / continuadores reales.
- **Variación [H]:** `±` es la desviación estándar entre los pliegues (no un intervalo de confianza: los pliegues comparten datos). Los intervalos `[ , ]` del holdout son percentiles 2,5–97,5 de 1000 remuestreos bootstrap del holdout; reflejan el azar del muestreo de esos 600 pacientes, no la variabilidad de la búsqueda de hiperparámetros.

---

## 2. Discriminación y error de probabilidad (validación cruzada)

**ROC-AUC** (0.5 = sin información):

| Modelo | T0 | T1 | T1 sin habla |
|---|---|---|---|
| Dummy mayoritaria | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Dummy prior | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Dummy estratificado | 0.505 ± 0.026 | 0.505 ± 0.026 | 0.505 ± 0.026 |
| Regresión logística | 0.721 ± 0.021 | 0.886 ± 0.017 | 0.720 ± 0.023 |
| Árbol | 0.703 ± 0.029 | 0.868 ± 0.016 | 0.703 ± 0.027 |
| Random Forest | 0.720 ± 0.023 | 0.865 ± 0.019 | 0.719 ± 0.021 |
| XGBoost | 0.719 ± 0.025 | 0.886 ± 0.016 | 0.717 ± 0.023 |

**PR-AUC** (la línea base es la prevalencia, ≈ 0.71):

| Modelo | T0 | T1 | T1 sin habla |
|---|---|---|---|
| Dummy mayoritaria | 0.714 ± 0.001 | 0.714 ± 0.001 | 0.714 ± 0.001 |
| Dummy prior | 0.714 ± 0.001 | 0.714 ± 0.001 | 0.714 ± 0.001 |
| Dummy estratificado | 0.716 ± 0.011 | 0.716 ± 0.011 | 0.716 ± 0.011 |
| Regresión logística | 0.858 ± 0.019 | 0.948 ± 0.008 | 0.857 ± 0.019 |
| Árbol | 0.832 ± 0.019 | 0.930 ± 0.010 | 0.831 ± 0.018 |
| Random Forest | 0.856 ± 0.019 | 0.935 ± 0.014 | 0.852 ± 0.020 |
| XGBoost | 0.855 ± 0.020 | 0.946 ± 0.009 | 0.854 ± 0.019 |

**Brier** (menor es mejor; Dummy prior = 0.204 es la referencia de quien solo conoce la prevalencia):

| Modelo | T0 | T1 | T1 sin habla |
|---|---|---|---|
| Dummy mayoritaria | 0.286 ± 0.001 | 0.286 ± 0.001 | 0.286 ± 0.001 |
| Dummy prior | 0.204 ± 0.000 | 0.204 ± 0.000 | 0.204 ± 0.000 |
| Dummy estratificado | 0.401 ± 0.018 | 0.401 ± 0.018 | 0.401 ± 0.018 |
| Regresión logística | 0.179 ± 0.004 | 0.116 ± 0.009 | 0.179 ± 0.004 |
| Árbol | 0.183 ± 0.007 | 0.127 ± 0.009 | 0.182 ± 0.006 |
| Random Forest | 0.180 ± 0.004 | 0.129 ± 0.006 | 0.182 ± 0.003 |
| XGBoost | 0.179 ± 0.005 | 0.119 ± 0.007 | 0.180 ± 0.005 |

- Sin habla, los modelos reducen el Brier de 0.204 a ≈ 0.179; con habla (T1) llegan a ≈ 0.116. [R]
- Los Dummy confirman que las métricas están bien calculadas: ROC-AUC ≈ 0.5, PR-AUC ≈ prevalencia, y el Dummy mayoritaria tiene el peor Brier entre los que no son aleatorios. [R]

---

## 3. Holdout (evaluado una sola vez)

**ROC-AUC** [IC 95 % bootstrap]:

| Modelo | T0 | T1 | T1 sin habla |
|---|---|---|---|
| Dummy prior | 0.500 [0.500–0.500] | 0.500 [0.500–0.500] | 0.500 [0.500–0.500] |
| Regresión logística | 0.737 [0.693–0.783] | 0.877 [0.846–0.908] | 0.731 [0.687–0.777] |
| Árbol | 0.721 [0.678–0.763] | 0.854 [0.819–0.891] | 0.721 [0.679–0.763] |
| Random Forest | 0.732 [0.689–0.777] | 0.864 [0.831–0.897] | 0.731 [0.689–0.777] |
| XGBoost | 0.734 [0.690–0.778] | 0.868 [0.835–0.900] | 0.732 [0.689–0.776] |

**PR-AUC** [IC 95 % bootstrap]:

| Modelo | T0 | T1 | T1 sin habla |
|---|---|---|---|
| Dummy prior | 0.713 [0.678–0.750] | 0.713 [0.678–0.750] | 0.713 [0.678–0.750] |
| Regresión logística | 0.868 [0.833–0.902] | 0.938 [0.916–0.960] | 0.867 [0.834–0.900] |
| Árbol | 0.844 [0.809–0.878] | 0.914 [0.886–0.942] | 0.844 [0.809–0.878] |
| Random Forest | 0.867 [0.831–0.901] | 0.932 [0.909–0.955] | 0.869 [0.835–0.901] |
| XGBoost | 0.868 [0.833–0.901] | 0.933 [0.911–0.954] | 0.869 [0.837–0.901] |

- El holdout coincide con la validación cruzada (diferencias dentro de la variación entre pliegues), y no hay señal de que la selección de hiperparámetros y de umbral haya sobreajustado los datos de desarrollo. [R]

---

## 4. Efecto de las variables de habla

| Modelo | T0 | T1 sin habla | T1 | T1 − T1 sin habla (pliegues a favor) | T1 sin habla − T0 |
|---|---|---|---|---|---|
| Regresión logística | 0.721 | 0.720 | 0.886 | +0.166 (15/15) | -0.0006 |
| Árbol | 0.703 | 0.703 | 0.868 | +0.165 (15/15) | +0.0001 |
| Random Forest | 0.720 | 0.719 | 0.865 | +0.146 (15/15) | -0.0005 |
| XGBoost | 0.719 | 0.717 | 0.886 | +0.168 (15/15) | -0.0015 |

ROC-AUC en validación cruzada; las diferencias son pareadas sobre los mismos pliegues.

- Quitar `grado_hipernasalidad` e `insuficiencia_velofaringea_ivf` devuelve el desempeño al de T0 (diferencias ≤ 0.002 de ROC-AUC, dentro de la variación entre pliegues). El EDA ya mostraba la señal sospechosa: AUC univariado 0.770 para hipernasalidad y tasa de deserción de 20.6 % en grado 0 a 94.5 % en grado 3 (`eda.md` §8). [R]
- Que una variable clínica explique todo el rendimiento es compatible con que se mida después de iniciar la terapia (consecuencia del abandono o de la terapia recibida), pero los datos no permiten distinguirlo: no hay fechas. **Pregunta 4 para Fisulab** (`definicion_problema.md` §6). [P]

---

## 5. Comparación entre modelos

| Escenario | Modelo | Δ ROC-AUC vs regresión logística (media) | sd entre pliegues | Pliegues en que supera |
|---|---|---|---|---|
| T0 | Árbol | -0.0176 | 0.0148 | 1/15 |
| T0 | Random Forest | -0.0008 | 0.0100 | 6/15 |
| T0 | XGBoost | -0.0018 | 0.0116 | 7/15 |
| T1 | Árbol | -0.0182 | 0.0095 | 0/15 |
| T1 | Random Forest | -0.0208 | 0.0073 | 0/15 |
| T1 | XGBoost | -0.0002 | 0.0042 | 7/15 |
| T1 sin habla | Árbol | -0.0170 | 0.0126 | 1/15 |
| T1 sin habla | Random Forest | -0.0007 | 0.0081 | 7/15 |
| T1 sin habla | XGBoost | -0.0027 | 0.0102 | 7/15 |

- Random Forest y XGBoost empatan con la regresión logística en T0 y T1 sin habla; en T1, el Random Forest queda por debajo en todos los pliegues y XGBoost empata. [R]
- El árbol es consistentemente peor en los tres escenarios. Además, con profundidad 3 produce pocas probabilidades distintas, y el umbral resultante marca a más pacientes (ver §6). [R]
- No se hicieron pruebas de significancia formales; las diferencias en ROC-AUC de ≈ 0.001–0.003 son ruido frente a una desviación entre pliegues de ≈ 0.01. [R]

---

## 6. Punto de operación (holdout, objetivo recall ≥ 0.90)

Los tres Dummy marcan a todos como alerta (recall 1.00, especificidad 0.00, precisión = prevalencia). Para los demás modelos:

**T0**

| Modelo | Recall | Especificidad | Precisión | Pacientes en alerta | Umbral |
|---|---|---|---|---|---|
| Regresión logística | 0.895 [0.864–0.925] | 0.337 [0.266–0.413] | 0.771 [0.733–0.811] | 82.8 % | 0.558 |
| Árbol | 0.953 [0.933–0.973] | 0.122 [0.076–0.171] | 0.730 [0.695–0.768] | 93.2 % | 0.520 |
| Random Forest | 0.888 [0.856–0.919] | 0.314 [0.246–0.387] | 0.763 [0.724–0.804] | 83.0 % | 0.553 |
| XGBoost | 0.895 [0.866–0.923] | 0.314 [0.244–0.388] | 0.764 [0.727–0.804] | 83.5 % | 0.524 |

**T1**

| Modelo | Recall | Especificidad | Precisión | Pacientes en alerta | Umbral |
|---|---|---|---|---|---|
| Regresión logística | 0.909 [0.882–0.935] | 0.622 [0.543–0.695] | 0.857 [0.821–0.892] | 75.7 % | 0.570 |
| Árbol | 0.928 [0.903–0.951] | 0.558 [0.480–0.631] | 0.839 [0.807–0.874] | 78.8 % | 0.525 |
| Random Forest | 0.904 [0.876–0.931] | 0.599 [0.519–0.673] | 0.849 [0.815–0.882] | 76.0 % | 0.609 |
| XGBoost | 0.921 [0.895–0.946] | 0.570 [0.487–0.644] | 0.842 [0.807–0.877] | 78.0 % | 0.601 |

**T1 sin habla**

| Modelo | Recall | Especificidad | Precisión | Pacientes en alerta | Umbral |
|---|---|---|---|---|---|
| Regresión logística | 0.883 [0.851–0.914] | 0.337 [0.268–0.413] | 0.768 [0.730–0.809] | 82.0 % | 0.560 |
| Árbol | 0.953 [0.933–0.973] | 0.122 [0.076–0.171] | 0.730 [0.695–0.768] | 93.2 % | 0.520 |
| Random Forest | 0.900 [0.872–0.928] | 0.302 [0.231–0.373] | 0.762 [0.725–0.802] | 84.2 % | 0.604 |
| XGBoost | 0.897 [0.869–0.927] | 0.308 [0.241–0.383] | 0.763 [0.725–0.803] | 83.8 % | 0.532 |

- Sin habla, para detectar ≈ 90 % de los desertores hay que alertar a ≈ 83 % de los pacientes: la alerta casi no filtra. Con habla (T1) baja a ≈ 76 %. [R]
- **Umbrales Bajo/Medio/Alto:** los 0.33 / 0.66 actuales no tienen respaldo (`definicion_problema.md` §5). Sin habla, el 10 % de pacientes con menor probabilidad tiene en promedio ≈ 0.44, así que con 0.33 casi nadie quedaría en riesgo Bajo (en T1 sí: ese decil ronda 0.10). El corte debe definirse según la capacidad semanal del equipo (**pregunta 6**). [R + Rec]

**Revisión del grupo de mayor puntaje** (validación cruzada, media): fracción de los desertores totales que quedan en el 10 / 20 / 30 % de pacientes con mayor puntaje (captura) y fracción de desertores dentro de ese grupo (precisión). Sin información, la captura es 0.10 / 0.20 / 0.30 y la precisión ≈ prevalencia.

| Escenario | Modelo | Captura 10 % | Captura 20 % | Captura 30 % | Precisión 10 % | Precisión 30 % |
|---|---|---|---|---|---|---|
| T0 | Regresión logística | 0.132 | 0.251 | 0.365 | 0.943 | 0.870 |
| T0 | Árbol | 0.129 | 0.251 | 0.365 | 0.919 | 0.868 |
| T0 | Random Forest | 0.129 | 0.254 | 0.372 | 0.922 | 0.885 |
| T0 | XGBoost | 0.130 | 0.255 | 0.369 | 0.926 | 0.878 |
| T1 | Regresión logística | 0.140 | 0.278 | 0.410 | 1.000 | 0.976 |
| T1 | Árbol | 0.138 | 0.276 | 0.406 | 0.985 | 0.968 |
| T1 | Random Forest | 0.137 | 0.274 | 0.404 | 0.979 | 0.961 |
| T1 | XGBoost | 0.139 | 0.276 | 0.408 | 0.996 | 0.972 |

- Los pacientes de mayor puntaje sí son casi todos desertores (precisión ≈ 0.92–0.94 en el 10 % superior sin habla), pero como la prevalencia ya es 71 %, ese grupo solo concentra ≈ 13 % de los desertores. [R]

---

## 7. Calibración

Error de calibración esperado (ECE, 10 cajas de igual tamaño, predicciones fuera de muestra de las tres repeticiones; menor es mejor):

| Modelo | T0 | T1 | T1 sin habla |
|---|---|---|---|
| Dummy mayoritaria | 0.286 | 0.286 | 0.286 |
| Dummy prior | 0.010 | 0.010 | 0.010 |
| Dummy estratificado | 0.387 | 0.387 | 0.387 |
| Regresión logística | 0.026 | 0.012 | 0.030 |
| Árbol | 0.024 | 0.019 | 0.022 |
| Random Forest | 0.033 | 0.045 | 0.045 |
| XGBoost | 0.017 | 0.037 | 0.023 |

- Los cuatro modelos quedan bien calibrados (ECE ≈ 0.02–0.05): sin pesos de clase, las probabilidades se pueden leer como probabilidades. Ver `reports/modelos/figuras/calibracion_*.png`. [R]
- Random Forest y XGBoost se calibran algo peor en T1 (ECE 0.045 y 0.037): sobreestiman la deserción en el rango 0.3–0.6 de probabilidad predicha. La regresión logística no (ECE 0.012). [R]
- Dummy mayoritaria y Dummy estratificado tienen ECE alto por construcción (probabilidades 0 / 1). [H]

---

## 8. Subgrupos (regresión logística)

Puntaje medio por paciente entre repeticiones; recall y especificidad con el umbral de cada pliegue. Los grupos pequeños (SES 4: 120 pacientes) son inestables.

**T0**

| Variable | Grupo | n | Prevalencia | ROC-AUC | Recall | Especificidad |
|---|---|---|---|---|---|---|
| nivel socioeconómico | SES 0 | 1110 | 0.746 | 0.713 | 0.936 | 0.194 |
| nivel socioeconómico | SES 1 | 821 | 0.731 | 0.711 | 0.853 | 0.365 |
| nivel socioeconómico | SES 2 | 349 | 0.648 | 0.671 | 0.796 | 0.453 |
| nivel socioeconómico | SES 4 | 120 | 0.500 | 0.698 | 0.572 | 0.706 |
| distancia al hospital | Q1 distancia | 601 | 0.652 | 0.728 | 0.753 | 0.598 |
| distancia al hospital | Q2 distancia | 599 | 0.694 | 0.673 | 0.827 | 0.350 |
| distancia al hospital | Q3 distancia | 600 | 0.688 | 0.685 | 0.915 | 0.193 |
| distancia al hospital | Q4 distancia | 600 | 0.822 | 0.732 | 0.982 | 0.078 |

**T1**

| Variable | Grupo | n | Prevalencia | ROC-AUC | Recall | Especificidad |
|---|---|---|---|---|---|---|
| nivel socioeconómico | SES 0 | 1110 | 0.746 | 0.880 | 0.913 | 0.628 |
| nivel socioeconómico | SES 1 | 821 | 0.731 | 0.888 | 0.902 | 0.658 |
| nivel socioeconómico | SES 2 | 349 | 0.648 | 0.884 | 0.892 | 0.691 |
| nivel socioeconómico | SES 4 | 120 | 0.500 | 0.871 | 0.706 | 0.850 |
| distancia al hospital | Q1 distancia | 601 | 0.652 | 0.887 | 0.838 | 0.794 |
| distancia al hospital | Q2 distancia | 599 | 0.694 | 0.875 | 0.893 | 0.616 |
| distancia al hospital | Q3 distancia | 600 | 0.688 | 0.872 | 0.914 | 0.640 |
| distancia al hospital | Q4 distancia | 600 | 0.822 | 0.892 | 0.941 | 0.561 |

- En T0, la sensibilidad cae en los grupos de menor prevalencia: SES 4 tiene recall 0.57 frente a 0.94 en SES 0, y el cuartil de menor distancia 0.75 frente a 0.98 en el de mayor distancia. Con un solo umbral global, el modelo alerta más a los pacientes de bajo nivel socioeconómico y lejanos, que son justamente los que ya tienen más deserción; el ROC-AUC por grupo es parecido (≈ 0.67–0.73). [R]
- Esto es un efecto del umbral único y de la prevalencia distinta por grupo, no necesariamente de un sesgo del ordenamiento. Debe revisarse en el frente 5 antes de usar la alerta para asignar recursos. [Rec]

---

## 9. Hiperparámetros elegidos

Valor más frecuente en los pliegues externos (veces / total) y valor elegido para el modelo del holdout:

| Escenario | Modelo | Más frecuente en CV | Holdout |
|---|---|---|---|
| T0 | Regresión logística | C=0.01 (10/15) | C=0.01 |
| T0 | Árbol | max_depth=3, min_samples_leaf=60 (12/15) | max_depth=3, min_samples_leaf=60 |
| T0 | Random Forest | max_depth=4, min_samples_leaf=5 (9/15) | max_depth=4, min_samples_leaf=20 |
| T0 | XGBoost | learning_rate=0.05, max_depth=2, n_estimators=100 (15/15) | learning_rate=0.05, max_depth=2, n_estimators=100 |
| T1 | Regresión logística | C=1.0 (6/15) | C=1.0 |
| T1 | Árbol | max_depth=5, min_samples_leaf=30 (6/15) | max_depth=5, min_samples_leaf=30 |
| T1 | Random Forest | max_depth=8, min_samples_leaf=5 (15/15) | max_depth=8, min_samples_leaf=5 |
| T1 | XGBoost | learning_rate=0.05, max_depth=2, n_estimators=100 (14/15) | learning_rate=0.05, max_depth=2, n_estimators=100 |
| T1 sin habla | Regresión logística | C=0.01 (14/15) | C=0.01 |
| T1 sin habla | Árbol | max_depth=3, min_samples_leaf=60 (13/15) | max_depth=3, min_samples_leaf=60 |
| T1 sin habla | Random Forest | max_depth=4, min_samples_leaf=20 (8/15) | max_depth=4, min_samples_leaf=20 |
| T1 sin habla | XGBoost | learning_rate=0.05, max_depth=2, n_estimators=100 (15/15) | learning_rate=0.05, max_depth=2, n_estimators=100 |

- **Los valores elegidos están en el borde de la rejilla** en casi todos los casos: la regresión logística elige el menor `C` probado (0.01) en T0 y T1 sin habla, y XGBoost elige la menor tasa de aprendizaje, la menor profundidad y el menor número de árboles en 14 o 15 de los 15 pliegues de cada escenario. La señal es débil y los modelos prefieren la mayor regularización disponible. [R]
- [Rec] Ampliar las rejillas hacia más regularización (`C` < 0.01; XGBoost con menos árboles / más `min_child_weight`) antes de cerrar la comparación. No sabemos si cambiaría la comparación: el límite actual de la rejilla puede estar recortando a la regresión logística y a XGBoost.

---

## 10. Límites

- **Datos sintéticos [H]:** nada de esto describe pacientes reales.
- **Etiqueta [P]:** `eda.md` §5.2 encontró que la etiqueta de v2 parece generada a partir de la asistencia a terapia. Si es así, el problema es reconstruir una regla y no predecir un abandono; sin esa variable (excluida por fuga) el rendimiento queda en ≈ 0.72 de ROC-AUC.
- **Sin fechas [H]:** los escenarios son etapas, no horizontes de predicción; no se puede decir con cuánta anticipación alertaría el modelo.
- **Una sola partición de holdout [H]:** 600 pacientes; los IC bootstrap no incluyen la variabilidad del particionado.
- **Umbral [R]:** se fija en entrenamiento y el recall de validación queda ligeramente por debajo del objetivo; para una garantía hay que añadir un margen (por ejemplo, exigir 0.92 en entrenamiento) o calibrar el umbral con un conjunto adicional.
- **Comparación [H]:** las rejillas son pequeñas y se definieron a priori (§9); no se probó selección de variables ni interacciones.
- **Equidad [R]:** la alerta no es uniforme entre niveles socioeconómicos ni distancias (§8).

---

## 11. Recomendaciones

1. **Modelo de referencia [Rec]:** regresión logística regularizada, escenario T0 (o T1 sin habla, equivalente). Es el más simple, rinde igual que los complejos y sus probabilidades están calibradas.
2. **T1 con habla [Rec]:** reportarlo solo como cota superior hasta que Fisulab responda cuándo se miden hipernasalidad e IVF. Si se miden antes de la terapia, T1 pasa a ser el escenario principal.
3. **Alertas [Rec]:** no usar los umbrales 0.33 / 0.66; fijar el corte con la capacidad semanal del equipo (pregunta 6) y revisar el comportamiento por subgrupos.
4. **Frente 5 [Rec]:** la regresión logística con variables estandarizadas permite interpretar los coeficientes directamente; no hace falta una explicación posterior (p. ej. valores SHAP) sobre un modelo más complejo que no rinde más. Dejar escrito que la alerta no sustituye la valoración clínica.
5. **Integración [Rec]:** no tocar `model.joblib` ni la API hasta decidir el escenario; el modelo servido hoy usa `citas_perdidas_noshow` (fuga) y debe reemplazarse.

---

## 12. Reproducción

```powershell
.\.venv\Scripts\python.exe -m src.infrastructure.ai_models.experiments
```

Semilla 42; Python 3.11.9, scikit-learn 1.9.1, XGBoost 3.2.0, pandas 3.1.0, numpy 2.4.6. Tiempo total 713.7 s. No usar `--quick` para reportar.

Salidas en `reports/modelos/`: `cv_resultados_por_pliegue.csv`, `cv_resumen.csv`, `cv_hiperparametros.csv`, `holdout_resultados.csv`, `calibracion_cajas.csv`, `calibracion_resumen.csv`, `subgrupos.csv`, `metadatos.json` y `figuras/calibracion_*.png`. Este documento se generó a partir de esos archivos.
