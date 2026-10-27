# Tablas del EDA (generadas automáticamente)

Generado por `notebooks/03_eda.py`. No editar a mano.

## 1. Calidad y estructura

| Chequeo | Resultado | Detalle |
|---|---|---|
| Pacientes en el CSV procesado | 3500 | v1=500, v2=3000; una fila por paciente. |
| IDs únicos / duplicados | 3500 / 0 | La llave no se repite. |
| Solapamiento de IDs v1 vs v2 | 0 | Las cohortes no comparten pacientes. |
| Pacientes v1 con cirugía/terapia sin etiqueta | 500 | Se excluyen del modelado por no tener Desertor. |
| Pacientes v2 con cirugía/terapia sin etiqueta | 0 | No se usan. |
| Nulos en archivos raw | 0 | Los seis CSV vienen completos. |
| Nulos en el CSV procesado | 0 | 0 por construcción: el ETL imputó. |
| Columnas de fecha | 0 | No hay fechas de ingreso, citas, cirugías ni cierre. |
| Estructura longitudinal | Una fila por paciente | No hay historial de citas ni de etapas. |
| Etiqueta binaria | 0, 1 | Sin fecha ni motivo de cierre. |
| No desertores en v1 | 30 | Insuficiente para entrenar o evaluar v1. |
| Citas_Asistidas en v2 | Constante 11 | La columna no existe en v2; el ETL la imputa. |
| Cirugías_Programadas en v2 | 2 a 4 (estimadas) | En v1 es 2 a 5; el ETL usa otra fórmula. |
| Niveles de SES | v1=['Alto', 'Bajo', 'Medio'] · v2=['Alto', 'Bajo', 'Medio', 'Medio-Bajo'] | Escalas de 3 y 4 niveles; no son equivalentes. |
| Etiquetas de tipo de fisura | v1=['Bilateral Asimétrica', 'Bilateral Completa', 'Bilateral Incompleta'] · v2=['Asimétrica', 'Bilateral Completa', 'Incompleta'] | El ETL armoniza 'Bilateral ...' con el resto. |
| Edad de inicio > edad de queiloplastia | 2141 | Inconsistencia temporal a revisar. |
| Intervalo de cirugías negativo | 0 | Palatoplastia antes de queiloplastia. |

## 2. Balance de clases

| Cohorte | Pacientes | Desertores | No desertores | Tasa | IC95 inferior | IC95 superior |
|---|---|---|---|---|---|---|
| v1 | 500 | 470 | 30 | 0.940 | 0.916 | 0.958 |
| v2 | 3000 | 2142 | 858 | 0.714 | 0.698 | 0.730 |
| Total | 3500 | 2612 | 888 | 0.746 | 0.732 | 0.760 |

## 3. Disponibilidad de columnas por cohorte

| Variable | v1 | v2 |
|---|---|---|
| Tipo_Fisura_Bilateral | 1 | 1 |
| Edad_Inicio_Meses | 1 | 1 |
| Distancia_Hospital_KM | 1 | 1 |
| Nivel_Socioeconomico | 1 | 1 |
| Ingresos_Mensuales_USD | 1 | 1 |
| Apoyo_Psicologico | 1 | 1 |
| Cirugias_Programadas | 1 | 0 |
| Citas_Asistidas | 1 | 0 |
| Citas_Perdidas_NoShow | 1 | 1 |
| Edad_Queiloplastia_Meses | 1 | 1 |
| Edad_Palatoplastia_Meses | 1 | 1 |
| Dificultad_Alimentacion_Preop | 1 | 1 |
| Dehiscencia_Herida_Postop | 1 | 1 |
| Fistula_Palatina_Residual | 1 | 1 |
| Peso_Z_Score_2Anos | 1 | 1 |
| Citas_Terapia_Programadas | 1 | 1 |
| Citas_Terapia_Asistidas | 1 | 1 |
| Hipoacusia_Conductiva | 1 | 1 |
| Grado_Hipernasalidad | 1 | 1 |
| Insuficiencia_Velofaringea_IVF | 1 | 1 |
| Desertor | 1 | 1 |

## 4. Efecto cohorte

SMD = diferencia estandarizada de medias (v1 − v2). |SMD| > 0,2 indica diferencia relevante.

| Variable | v1 media | v2 media | v1 mediana | v2 mediana | Diferencia | SMD (v1-v2) | |SMD| |
|---|---|---|---|---|---|---|---|
| citas_perdidas_noshow | 8.598 | 4.043 | 9.000 | 4.000 | 4.555 | 1.552 | 1.552 |
| citas_terapia_asistidas | 26.640 | 14.678 | 26.000 | 12.000 | 11.962 | 1.348 | 1.348 |
| peso_z_score_2anos | -0.117 | -0.803 | -0.130 | -0.810 | 0.687 | 0.671 | 0.671 |
| dificultad_alimentacion_preop | 0.252 | 0.641 | 0.000 | 1.000 | -0.389 | -0.850 | 0.850 |
| distancia_hospital_km | 65.118 | 55.099 | 46.700 | 40.200 | 10.019 | 0.182 | 0.182 |
| ingresos_mensuales_usd | 1195.902 | 990.939 | 806.500 | 660.500 | 204.963 | 0.213 | 0.213 |
| edad_inicio_meses | 5.910 | 6.088 | 6.000 | 6.000 | -0.178 | -0.056 | 0.056 |
| grado_hipernasalidad | 0.910 | 1.310 | 0.000 | 1.000 | -0.400 | -0.387 | 0.387 |
| nivel_socioeconomico_ord | 1.208 | 0.833 | 2.000 | 1.000 | 0.375 | 0.316 | 0.316 |

## 5. AUC univariado

0,5 = sin señal; cerca de 0 o 1 = separación. Es asociación, no causalidad.

| Variable | AUC v1 | AUC v2 | Señal v2 | Decisión |
|---|---|---|---|---|
| ratio_asistencia_terapia | 0.538 | 0.000 | 0.500 | Excluir: derivada de citas_terapia_asistidas; AUC 1.0 en v2 |
| citas_terapia_asistidas | 0.509 | 0.030 | 0.470 | Excluir: acumulado del tratamiento; separa la etiqueta casi perfecto en v2 |
| grado_hipernasalidad | 0.544 | 0.770 | 0.270 | T1 |
| citas_perdidas_noshow | 0.991 | 0.669 | 0.169 | Excluir: acumulado del tratamiento; AUC 0.991 en v1 |
| apoyo_psicologico | 0.277 | 0.338 | 0.162 | T0 |
| distancia_hospital_km | 0.725 | 0.586 | 0.086 | T0 |
| insuficiencia_velofaringea_ivf | 0.513 | 0.571 | 0.071 | T1 |
| ingresos_mensuales_usd | 0.128 | 0.436 | 0.064 | T0 |
| nivel_socioeconomico_ord | 0.129 | 0.441 | 0.059 | T0 |
| edad_queiloplastia_meses | 0.537 | 0.487 | 0.013 | T1 |
| edad_inicio_meses | 0.593 | 0.489 | 0.011 | T0 |
| cirugias_programadas | 0.668 | 0.490 | 0.010 | Excluir: en v2 se estima con otra fórmula; proxy de cohorte |
| fistula_palatina_residual | 0.551 | 0.490 | 0.010 | T1 |
| edad_palatoplastia_meses | 0.485 | 0.492 | 0.008 | T1 |
| dificultad_alimentacion_preop | 0.474 | 0.505 | 0.005 | T0 |
| peso_z_score_2anos | 0.546 | 0.495 | 0.005 | Excluir: medido a los 2 años; posterior al momento de predicción |
| hipoacusia_conductiva | 0.500 | 0.503 | 0.003 | T1 |
| dehiscencia_herida_postop | 0.526 | 0.499 | 0.001 | T1 |
| citas_asistidas | 0.160 |  |  | Excluir: acumulado; en v2 no existe y el ETL la imputa con la mediana de v1 |

## 6. Artefactos del ETL

| Cohorte | citas_asistidas únicos | citas_asistidas rango | cirugias_programadas rango | citas_asistidas_imputada=1 |
|---|---|---|---|---|
| v1 | 25 | 2–26 | 2–5 | 0 |
| v2 | 1 | 11–11 | 2–4 | 3000 |

## 7. Resumen de features T0 (v2)

| Feature | Tipo | n | Faltantes | Resumen v2 | AUC v2 |
|---|---|---|---|---|---|
| tipo_fisura | categórica | 3000 | 0 | Asimetrica: 633, Completa: 1499, Incompleta: 868 |  |
| edad_inicio_meses | continua | 3000 | 0 | mediana 6.0 [3.0, 9.0] | 0.489 |
| distancia_hospital_km | continua | 3000 | 0 | mediana 40.2 [19.1, 75.2] | 0.586 |
| nivel_socioeconomico_ord | ordinal/binaria | 3000 | 0 | 0: 1393, 1: 1016, 2: 440, 4: 151 | 0.441 |
| ingresos_mensuales_usd | continua | 3000 | 0 | mediana 660.5 [445.0, 1097.0] | 0.436 |
| apoyo_psicologico | ordinal/binaria | 3000 | 0 | 0: 1818, 1: 1182 | 0.338 |
| dificultad_alimentacion_preop | ordinal/binaria | 3000 | 0 | 0: 1077, 1: 1923 | 0.505 |

## 8. Resumen de features T1 (v2)

| Feature | Tipo | n | Faltantes | Resumen v2 | AUC v2 |
|---|---|---|---|---|---|
| tipo_fisura | categórica | 3000 | 0 | Asimetrica: 633, Completa: 1499, Incompleta: 868 |  |
| edad_inicio_meses | continua | 3000 | 0 | mediana 6.0 [3.0, 9.0] | 0.489 |
| distancia_hospital_km | continua | 3000 | 0 | mediana 40.2 [19.1, 75.2] | 0.586 |
| nivel_socioeconomico_ord | ordinal/binaria | 3000 | 0 | 0: 1393, 1: 1016, 2: 440, 4: 151 | 0.441 |
| ingresos_mensuales_usd | continua | 3000 | 0 | mediana 660.5 [445.0, 1097.0] | 0.436 |
| apoyo_psicologico | ordinal/binaria | 3000 | 0 | 0: 1818, 1: 1182 | 0.338 |
| dificultad_alimentacion_preop | ordinal/binaria | 3000 | 0 | 0: 1077, 1: 1923 | 0.505 |
| edad_queiloplastia_meses | continua | 3000 | 0 | mediana 4.5 [3.5, 5.5] | 0.487 |
| edad_palatoplastia_meses | continua | 3000 | 0 | mediana 12.6 [11.0, 14.1] | 0.492 |
| dehiscencia_herida_postop | ordinal/binaria | 3000 | 0 | 0: 2641, 1: 359 | 0.499 |
| fistula_palatina_residual | ordinal/binaria | 3000 | 0 | 0: 2581, 1: 419 | 0.490 |
| hipoacusia_conductiva | ordinal/binaria | 3000 | 0 | 0: 1881, 1: 1119 | 0.503 |
| grado_hipernasalidad | ordinal/binaria | 3000 | 0 | 0: 510, 1: 1321, 2: 898, 3: 271 | 0.770 |
| insuficiencia_velofaringea_ivf | ordinal/binaria | 3000 | 0 | 0: 1997, 1: 1003 | 0.571 |

## 9. Tasas de deserción por grupo (T0, v2)

| Grupo | n | Desertores | Tasa | IC95 inferior | IC95 superior | Variable |
|---|---|---|---|---|---|---|
| 0 | 1393 | 1039 | 0.746 | 0.722 | 0.768 | SES ordinal |
| 1 | 1016 | 743 | 0.731 | 0.703 | 0.758 | SES ordinal |
| 2 | 440 | 288 | 0.655 | 0.609 | 0.697 | SES ordinal |
| 4 | 151 | 72 | 0.477 | 0.399 | 0.556 | SES ordinal |
| Completa | 1499 | 1044 | 0.696 | 0.673 | 0.719 | Tipo de fisura |
| Incompleta | 868 | 639 | 0.736 | 0.706 | 0.764 | Tipo de fisura |
| Asimetrica | 633 | 459 | 0.725 | 0.689 | 0.758 | Tipo de fisura |
| 0 | 1818 | 1496 | 0.823 | 0.805 | 0.840 | Apoyo psicológico |
| 1 | 1182 | 646 | 0.547 | 0.518 | 0.575 | Apoyo psicológico |
| 0 | 1077 | 763 | 0.708 | 0.681 | 0.735 | Dificultad alimentación |
| 1 | 1923 | 1379 | 0.717 | 0.697 | 0.737 | Dificultad alimentación |

## 10. Tasas de deserción por grupo (T1, v2)

| Grupo | n | Desertores | Tasa | IC95 inferior | IC95 superior | Variable |
|---|---|---|---|---|---|---|
| 0 | 510 | 105 | 0.206 | 0.173 | 0.243 | Grado de hipernasalidad |
| 1 | 1321 | 1002 | 0.759 | 0.735 | 0.781 | Grado de hipernasalidad |
| 2 | 898 | 779 | 0.867 | 0.844 | 0.888 | Grado de hipernasalidad |
| 3 | 271 | 256 | 0.945 | 0.911 | 0.966 | Grado de hipernasalidad |
| 0 | 1997 | 1339 | 0.671 | 0.650 | 0.691 | IVF |
| 1 | 1003 | 803 | 0.801 | 0.775 | 0.824 | IVF |
| 0 | 1881 | 1339 | 0.712 | 0.691 | 0.732 | Hipoacusia conductiva |
| 1 | 1119 | 803 | 0.718 | 0.691 | 0.743 | Hipoacusia conductiva |
| 0 | 2641 | 1887 | 0.715 | 0.697 | 0.731 | Dehiscencia |
| 1 | 359 | 255 | 0.710 | 0.661 | 0.755 | Dehiscencia |
| 0 | 2581 | 1855 | 0.719 | 0.701 | 0.736 | Fístula residual |
| 1 | 419 | 287 | 0.685 | 0.639 | 0.728 | Fístula residual |

## 11. Correlaciones altas (|Spearman| ≥ 0,30, v2)

| Variable 1 | Variable 2 | Spearman | |r| |
|---|---|---|---|
| nivel_socioeconomico_ord | ingresos_mensuales_usd | 0.926 | 0.926 |
| grado_hipernasalidad | insuficiencia_velofaringea_ivf | 0.546 | 0.546 |
| edad_queiloplastia_meses | edad_palatoplastia_meses | 0.541 | 0.541 |
| fistula_palatina_residual | grado_hipernasalidad | 0.512 | 0.512 |
| grado_hipernasalidad | desertor | 0.450 | 0.450 |
| fistula_palatina_residual | insuficiencia_velofaringea_ivf | 0.377 | 0.377 |
| dehiscencia_herida_postop | fistula_palatina_residual | 0.331 | 0.331 |

## 12. Figuras generadas

- `reports/figures/01_balance_clases.png`
- `reports/figures/02_disponibilidad_columnas.png`
- `reports/figures/03_efecto_cohorte.png`
- `reports/figures/04_fuga_noshow.png`
- `reports/figures/05_fuga_terapia.png`
- `reports/figures/06_artefactos_etl.png`
- `reports/figures/07_t0_distribuciones.png`
- `reports/figures/08_t0_tasas.png`
- `reports/figures/09_t1_habla.png`
- `reports/figures/10_correlacion.png`
- `reports/figures/11_edad_inicio_vs_cirugia.png`
- `reports/figures/12_ses_cohorte.png`
- `reports/figures/10_correlaciones_altas.csv` (pares con |Spearman| ≥ 0,30)
